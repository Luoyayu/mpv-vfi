/*
 * vf_migraphx: RIFE frame interpolation (2x) through libmigraphx_vfi_backend.so.
 *
 * This file is part of mpv (fork: Luoyayu/mpv-vfi, branch vfi/v0.41).
 *
 * mpv is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * mpv is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with mpv.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 * Design (docs/designs/phase2_export_and_playback.md, sections 2 and 5 of the MIGraphX-VFI project):
 *
 *  - The filter has no link-time dependency on ROCm and includes no GPU runtime header. It dlopen()s the backend library
 *    (RTLD_LOCAL | RTLD_NOW) on the first frame and talks to it through migraphx_backend.h (plain C, ABI 1).
 *  - process() never waits for the GPU. Frames are copied into pinned host slots (mgb_acquire_input / mgb_submit),
 *    results are collected with the non-blocking mgb_fetch(). The backend's worker thread calls mp_filter_wakeup()
 *    when results or the state change.
 *  - Output order is kept in a queue of entries: ORIG(n), MID(n,n+1), ORIG(n+1), ... Each MID is a placeholder until
 *    its result arrives; an ORIG waits for its successor (its duration is half the source interval). When the
 *    backend converts or shrinks the source (10-bit, > 1920 wide) the ORIG frames also come from the backend.
 *  - A bypass (the backend is initializing, has no plan, the format is unsupported, the source frame rate is above
 *    max-fps, the GPU does not keep up) passes frames through unchanged. A failed session fails the filter, which
 *    makes mpv disable it and play on.
 *  - reset (seek): the queue is dropped and mgb_reset() starts a new epoch; the first frame after a seek has no
 *    midpoint before it.
 */

#include <dlfcn.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common/common.h"
#include "common/msg.h"
#include "filters/f_autoconvert.h"
#include "filters/f_utils.h"
#include "filters/filter.h"
#include "filters/filter_internal.h"
#include "filters/user_filters.h"
#include "options/m_option.h"
#include "osdep/timer.h"
#include "video/img_format.h"
#include "video/mp_image.h"

#include "migraphx_backend.h"

#define STARVE_BUSY_NS      250000000LL  // backend queue full while the output starves this long: GPU too slow
#define OVERLOAD_PASS_MIN   120          // frames passed through before trying to interpolate again
#define OVERLOAD_PASS_MAX   960
#define PREROLL_PACED_FRAMES 2          // consecutive frame-paced reads that end the post-seek pass-through
#define PREROLL_PACED_RATIO  0.6        // a read this long after the previous one (in source frame durations) is paced
#define MAX_TIMING_SAMPLES  (1 << 18)

struct migraphx_opts {
    char *model;
    char *models_dir;
    int queue;
    double max_fps;
    char *backend;
    float scene_threshold;
    bool bypass_log;
    bool compile;
    bool wait_ready;
    bool preroll_bypass;
};

struct backend_api {
    void *handle;
    char path[PATH_MAX];
    int (*abi_version)(void);
    void (*open_params_init)(mgb_open_params_t *);
    mgb_session *(*open)(const mgb_open_params_t *);
    int (*state)(mgb_session *);
    int (*bypass_reason)(mgb_session *);
    const char *(*last_error)(mgb_session *);
    uint32_t (*epoch)(mgb_session *);
    int (*acquire_input)(mgb_session *, mgb_slot_t *);
    int (*submit)(mgb_session *, const mgb_slot_t *, int64_t, uint32_t);
    void (*cancel_input)(mgb_session *, const mgb_slot_t *);
    int (*flush)(mgb_session *, uint32_t);
    int (*fetch)(mgb_session *, mgb_result_t *);
    void (*release_output)(mgb_session *, const mgb_result_t *);
    uint32_t (*reset)(mgb_session *);
    uint32_t (*reconfigure)(mgb_session *, int, int, int);
    int (*get_stats)(mgb_session *, mgb_stats_t *);
    void (*close)(mgb_session *);
};

enum entry_kind { E_ORIG, E_MID, E_DUP };

struct entry {
    struct entry *next;
    enum entry_kind kind;
    struct mp_image *img;     // the output image once available
    struct mp_image *donor;   // attribute donor while img is not available (ORIG converted, MID)
    double pts, dur;
    bool filled;              // img is available (DUP: resolved at emit time)
    bool dur_known;
};

struct priv {
    struct mp_log *log;
    struct mp_filter *f;
    struct migraphx_opts *opts;
    struct mp_pin *in_pin;

    struct backend_api be;
    bool be_loaded;
    mgb_session *sess;
    uint32_t epoch;
    bool dirty;               // the pair chain was cut without mgb_reset(); reset before the next chain

    struct mp_image_params cfg;
    bool cfg_valid;
    int mgb_fmt;
    char *pass_reason;        // non-NULL: this format is passed through for good (until reconfigured)
    bool scaled;              // the backend returns ORIGINAL frames
    int out_w, out_h;

    int last_st, last_reason;
    char *last_bypass_msg;

    bool chain_open;
    struct entry *open_orig;  // the ORIG entry of the newest submitted frame
    double prev_pts, last_half;
    double nominal_fps;

    struct entry *head, *tail;
    int nentries;
    struct mp_image *last_out;
    struct mp_image *held;    // a frame we could not process yet
    bool eof_pending;

    // After a reset the player may decode from the keyframe before the seek target and throw the frames before the
    // target away after the filter chain (exact seek). Those frames are pulled as fast as the decoder delivers them,
    // while played frames are pulled at the frame rate (the VO throttles). Frames read faster than that right after
    // a reset are passed through instead of being interpolated, until reads are paced.
    bool preroll;
    int64_t last_read_ns;
    int paced_reads;
    uint64_t n_preroll;

    int64_t starve_since;
    bool overloaded;
    int overload_pass, overload_pass_target;

    // process() timing (debug log at destroy)
    int64_t *tm;
    size_t tm_n, tm_next;
    int64_t tm_max, tm_max_steady;
    bool slow_call;           // this process() call loaded the backend or opened the session
    uint64_t n_calls, n_in, n_pass, n_orig, n_mid, n_dup, n_resets, n_busy;
};

// ---------------------------------------------------------------------------------------------------------------
// backend loading

#define LOAD_SYM(name)                                                      \
    do {                                                                    \
        *(void **)&be->name = dlsym(be->handle, "mgb_" #name);              \
        if (!be->name) {                                                    \
            MP_ERR(p, "backend %s lacks mgb_%s\n", be->path, #name);        \
            return false;                                                   \
        }                                                                   \
    } while (0)

static bool resolve_backend(struct priv *p, void *handle, const char *path)
{
    struct backend_api *be = &p->be;
    be->handle = handle;
    snprintf(be->path, sizeof(be->path), "%s", path);
    Dl_info info;
    // Report the real file for the usual case of a bare soname.
    if (dlsym(handle, "mgb_abi_version") && dladdr(dlsym(handle, "mgb_abi_version"), &info) && info.dli_fname)
        snprintf(be->path, sizeof(be->path), "%s", info.dli_fname);
    *(void **)&be->abi_version = dlsym(handle, "mgb_abi_version");
    if (!be->abi_version) {
        MP_ERR(p, "%s is not a migraphx-vfi backend (no mgb_abi_version)\n", be->path);
        return false;
    }
    if (be->abi_version() != MGB_ABI_VERSION) {
        MP_ERR(p, "backend %s has ABI version %d, the filter needs %d\n", be->path, be->abi_version(),
               MGB_ABI_VERSION);
        return false;
    }
    be->open_params_init = NULL;
    *(void **)&be->open_params_init = dlsym(handle, "mgb_open_params_init");
    if (!be->open_params_init) {
        MP_ERR(p, "backend %s lacks mgb_open_params_init\n", be->path);
        return false;
    }
    LOAD_SYM(open);
    LOAD_SYM(state);
    LOAD_SYM(bypass_reason);
    LOAD_SYM(last_error);
    LOAD_SYM(epoch);
    LOAD_SYM(acquire_input);
    LOAD_SYM(submit);
    LOAD_SYM(cancel_input);
    LOAD_SYM(flush);
    LOAD_SYM(fetch);
    LOAD_SYM(release_output);
    LOAD_SYM(reset);
    LOAD_SYM(reconfigure);
    LOAD_SYM(get_stats);
    LOAD_SYM(close);
    return true;
}

static bool load_backend(struct priv *p)
{
    if (p->be_loaded)
        return true;
    const char *cands[2];
    int count = 0;
    const char *env = getenv("MIGRAPHX_VFI_BACKEND");
    // An explicit path (option, then environment) is the only candidate: do not silently load something else.
    if (p->opts->backend && p->opts->backend[0]) {
        cands[count++] = p->opts->backend;
    } else if (env && env[0]) {
        cands[count++] = env;
    } else {
        cands[count++] = "libmigraphx_vfi_backend.so.1";   // LD_LIBRARY_PATH, RUNPATH, ld.so.cache
        cands[count++] = "/opt/migraphx/lib/libmigraphx_vfi_backend.so.1";
    }
    for (int i = 0; i < count; i++) {
        void *h = dlopen(cands[i], RTLD_LOCAL | RTLD_NOW);
        if (!h) {
            const char *err = dlerror();
            if (i == count - 1)
                MP_ERR(p, "cannot load the backend library %s: %s\n", cands[i], err ? err : "?");
            else
                MP_VERBOSE(p, "dlopen(%s) failed: %s\n", cands[i], err ? err : "?");
            continue;
        }
        if (!resolve_backend(p, h, cands[i]))
            return false;
        MP_INFO(p, "backend: %s (ABI %d)\n", p->be.path, MGB_ABI_VERSION);
        p->be_loaded = true;
        return true;
    }
    return false;
}

static void backend_log(int level, const char *message, void *user)
{
    struct mp_log *log = user;
    static const int lv[] = {MSGL_ERR, MSGL_WARN, MSGL_INFO, MSGL_DEBUG};
    mp_msg(log, lv[level < 0 ? 0 : level > 3 ? 3 : level], "backend: %s\n", message);
}

static void backend_wakeup(void *ctx)
{
    mp_filter_wakeup(ctx);   // thread-safe
}

// ---------------------------------------------------------------------------------------------------------------
// bypass logging

static void log_bypass(struct priv *p, const char *fmt, ...)
{
    if (!p->opts->bypass_log)
        return;
    char *msg = NULL;
    va_list ap;
    va_start(ap, fmt);
    msg = talloc_vasprintf(NULL, fmt, ap);
    va_end(ap);
    if (!p->last_bypass_msg || strcmp(p->last_bypass_msg, msg) != 0) {
        MP_INFO(p, "passing frames through: %s\n", msg);
        talloc_free(p->last_bypass_msg);
        p->last_bypass_msg = talloc_steal(p, msg);
    } else {
        talloc_free(msg);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// output queue

static void entry_free(struct entry *e)
{
    mp_image_unrefp(&e->img);
    mp_image_unrefp(&e->donor);
    talloc_free(e);
}

static struct entry *queue_push(struct priv *p, enum entry_kind kind, double pts, double dur)
{
    struct entry *e = talloc_zero(p, struct entry);
    e->kind = kind;
    e->pts = pts;
    e->dur = dur;
    e->dur_known = kind != E_ORIG;
    if (p->tail)
        p->tail->next = e;
    else
        p->head = e;
    p->tail = e;
    p->nentries++;
    return e;
}

static void queue_clear(struct priv *p)
{
    while (p->head) {
        struct entry *e = p->head;
        p->head = e->next;
        entry_free(e);
    }
    p->tail = NULL;
    p->open_orig = NULL;
    p->nentries = 0;
}

static struct mp_image *entry_src(struct entry *e)
{
    return e->donor ? e->donor : e->img;
}

// Remove entries whose image will never arrive (and a DUP that followed a removed entry).
static void queue_abandon_unfilled(struct priv *p)
{
    struct entry *prev = NULL, *e = p->head;
    bool prev_removed = false;
    while (e) {
        struct entry *next = e->next;
        bool remove = !e->filled || (e->kind == E_DUP && prev_removed);
        if (remove) {
            if (prev)
                prev->next = next;
            else
                p->head = next;
            if (p->tail == e)
                p->tail = prev;
            if (p->open_orig == e)
                p->open_orig = NULL;
            p->nentries--;
            entry_free(e);
            prev_removed = true;
        } else {
            prev = e;
            prev_removed = false;
        }
        e = next;
    }
    // ORIG entries that wait for a successor that will not come
    for (e = p->head; e; e = e->next) {
        if (e->kind == E_ORIG && !e->dur_known) {
            e->dur = p->last_half > 0 ? p->last_half : 1.0 / 48;
            e->dur_known = true;
        }
    }
    p->open_orig = NULL;
}

static double frame_half(struct priv *p)
{
    if (p->last_half > 0)
        return p->last_half;
    if (p->nominal_fps > 0)
        return 0.5 / p->nominal_fps;
    return 1.0 / 48;
}

// End the pair chain. With flush: ask the backend for the last frame's converted copy and append the duplicate that
// keeps the frame count at 2N. Without: drop whatever is not finished.
static void close_chain(struct priv *p, bool flush)
{
    if (!p->chain_open)
        return;
    struct entry *last = p->open_orig;
    if (last && !last->dur_known) {
        last->dur = frame_half(p);
        last->dur_known = true;
    }
    p->chain_open = false;
    p->dirty = true;
    if (flush) {
        if (p->sess)
            p->be.flush(p->sess, p->epoch);
        if (last) {
            struct entry *d = queue_push(p, E_DUP, last->pts + last->dur, last->dur);
            d->filled = true;
        }
        p->open_orig = NULL;
    } else {
        queue_abandon_unfilled(p);
    }
}

static void tm_record(struct priv *p, int64_t ns)
{
    if (!p->tm)
        p->tm = talloc_array(p, int64_t, MAX_TIMING_SAMPLES);
    p->tm[p->tm_next] = ns;
    p->tm_next = (p->tm_next + 1) % MAX_TIMING_SAMPLES;
    if (p->tm_n < MAX_TIMING_SAMPLES)
        p->tm_n++;
    if (ns > p->tm_max)
        p->tm_max = ns;
    if (!p->slow_call && ns > p->tm_max_steady)
        p->tm_max_steady = ns;
    p->slow_call = false;
}

static int cmp_i64(const void *a, const void *b)
{
    int64_t x = *(const int64_t *)a, y = *(const int64_t *)b;
    return x < y ? -1 : x > y;
}

static void log_timing(struct priv *p)
{
    if (!p->tm_n || !mp_msg_test(p->log, MSGL_DEBUG))
        return;
    size_t n = p->tm_n;
    int64_t *s = talloc_memdup(NULL, p->tm, n * sizeof(*s));
    qsort(s, n, sizeof(*s), cmp_i64);
    double sum = 0;
    for (size_t i = 0; i < n; i++)
        sum += s[i];
    MP_DBG(p, "process() timing: %" PRIu64 " calls, max %.3f ms (without session setup %.3f ms), p99 %.3f ms, p50 %.3f ms, mean %.3f ms "
              "(frames in %" PRIu64 ", passed %" PRIu64 ", orig %" PRIu64 ", mid %" PRIu64 ", dup %" PRIu64
              ", resets %" PRIu64 ", busy %" PRIu64 ")\n",
           p->n_calls, p->tm_max / 1e6, p->tm_max_steady / 1e6, s[(size_t)((n - 1) * 0.99)] / 1e6, s[n / 2] / 1e6, sum / n / 1e6,
           p->n_in, p->n_pass, p->n_orig, p->n_mid, p->n_dup, p->n_resets, p->n_busy);
    talloc_free(s);
}

// ---------------------------------------------------------------------------------------------------------------
// session

static void close_session(struct priv *p)
{
    if (p->sess) {
        mgb_stats_t st = {.struct_size = sizeof(st)};
        if (mp_msg_test(p->log, MSGL_DEBUG) && p->be.get_stats(p->sess, &st) == MGB_OK) {
            MP_DBG(p, "backend stats: submitted %" PRIu64 ", pairs %" PRIu64 ", dropped frames %" PRIu64
                      ", dropped results %" PRIu64 ", busy %" PRIu64 ", scene cuts %" PRIu64
                      ", pair ms last/avg/max %.2f/%.2f/%.2f, plan %s\n",
                   st.submitted, st.pairs, st.dropped_frames, st.dropped_results, st.busy, st.scene_cuts,
                   st.last_pair_ms, st.avg_pair_ms, st.max_pair_ms, st.plan);
        }
        p->be.close(p->sess);   // bounded: waits for the pair in flight
        p->sess = NULL;
    }
}

// Why this frame format must be passed through (NULL: fine).
static char *unsupported_reason(struct priv *p, struct mp_image *mpi)
{
    if (mpi->imgfmt != IMGFMT_NV12 && mpi->imgfmt != IMGFMT_P010)
        return talloc_asprintf(p, "pixel format %s is not NV12/P010", mp_imgfmt_to_name(mpi->imgfmt));
    if (mpi->nominal_fps > p->opts->max_fps + 0.001)
        return talloc_asprintf(p, "source frame rate %.3f is above max-fps %.3f", mpi->nominal_fps,
                               p->opts->max_fps);
    const struct mp_image_params *pr = &mpi->params;
    if (pr->repr.sys != PL_COLOR_SYSTEM_UNKNOWN && pr->repr.sys != PL_COLOR_SYSTEM_BT_709)
        return talloc_asprintf(p, "color matrix %d is not BT.709", (int)pr->repr.sys);
    if (pr->repr.levels == PL_COLOR_LEVELS_FULL)
        return talloc_asprintf(p, "full range video");
    if (pr->color.transfer == PL_COLOR_TRC_PQ || pr->color.transfer == PL_COLOR_TRC_HLG)
        return talloc_asprintf(p, "HDR transfer function");
    return NULL;
}

static bool params_changed(struct priv *p, struct mp_image *mpi)
{
    return !p->cfg_valid || mpi->imgfmt != p->cfg.imgfmt || mpi->w != p->cfg.w || mpi->h != p->cfg.h ||
           mpi->params.p_w != p->cfg.p_w || mpi->params.p_h != p->cfg.p_h;
}

static const char *models_dir(struct priv *p, char *buf, size_t size)
{
    if (p->opts->models_dir && p->opts->models_dir[0])
        return p->opts->models_dir;
    const char *env = getenv("MIGRAPHX_VFI_MODELS");
    if (env && env[0])
        return env;
    // Repository layout: <repo>/build/libmigraphx_vfi_backend.so, <repo>/models
    char *slash = strrchr(p->be.path, '/');
    if (slash) {
        snprintf(buf, size, "%.*s/../models", (int)(slash - p->be.path), p->be.path);
        return buf;
    }
    return "models";
}

static void note_state(struct priv *p, int st, int reason)
{
    if (st == p->last_st && reason == p->last_reason)
        return;
    p->last_st = st;
    p->last_reason = reason;
    const char *err = p->be.last_error(p->sess);
    switch (st) {
    case MGB_STATE_INITIALIZING:
        log_bypass(p, "the backend is creating the engine");
        break;
    case MGB_STATE_READY: {
        mgb_stats_t s = {.struct_size = sizeof(s)};
        if (p->be.get_stats(p->sess, &s) == MGB_OK) {
            p->scaled = s.scaled;
            p->out_w = s.out_w;
            p->out_h = s.out_h;
            MP_INFO(p, "interpolating %dx%d with plan %s (output %dx%d%s)\n", p->cfg.w, p->cfg.h, s.plan,
                    s.out_w, s.out_h, s.scaled ? ", converted" : "");
        }
        p->last_bypass_msg = NULL;
        break;
    }
    case MGB_STATE_BYPASS:
        switch (reason) {
        case MGB_BYPASS_NO_PLAN:
            log_bypass(p, "no plan for %dx%d (%s)", p->cfg.w, p->cfg.h, err ? err : "compiling is off");
            break;
        case MGB_BYPASS_COMPILING:
            log_bypass(p, "no plan for %dx%d yet, compiling it in the background (takes minutes; "
                          "it takes effect after the next seek)", p->cfg.w, p->cfg.h);
            break;
        case MGB_BYPASS_PLAN_COMPILED:
            log_bypass(p, "the plan for %dx%d is compiled; it takes effect after the next seek or filter toggle",
                       p->cfg.w, p->cfg.h);
            break;
        default:
            log_bypass(p, "unsupported format or size %dx%d (%s)", p->cfg.w, p->cfg.h, err ? err : "");
            break;
        }
        break;
    default:
        break;
    }
}

static bool open_or_reconfigure(struct priv *p, struct mp_image *mpi)
{
    p->mgb_fmt = mpi->imgfmt == IMGFMT_P010 ? MGB_PIX_P010 : MGB_PIX_NV12;
    p->scaled = false;
    p->last_st = p->last_reason = -1;
    if (p->sess) {
        p->epoch = p->be.reconfigure(p->sess, mpi->w, mpi->h, p->mgb_fmt);
        MP_VERBOSE(p, "reconfigure %dx%d, epoch %u\n", mpi->w, mpi->h, p->epoch);
        return true;
    }
    p->slow_call = true;
    if (!load_backend(p))
        return false;
    mgb_open_params_t op;
    p->be.open_params_init(&op);
    char dirbuf[PATH_MAX + 16];
    op.model_dir = models_dir(p, dirbuf, sizeof(dirbuf));
    op.model = p->opts->model;
    op.width = mpi->w;
    op.height = mpi->h;
    op.fmt = p->mgb_fmt;
    op.scene_threshold = p->opts->scene_threshold;
    op.queue_depth = p->opts->queue;
    op.flags = p->opts->compile ? 0 : MGB_FLAG_NO_COMPILE;
    op.log = backend_log;
    op.log_user = p->log;
    int lvl = mp_msg_level(p->log);
    op.log_level = lvl >= MSGL_DEBUG ? 3 : lvl >= MSGL_V ? 2 : 1;
    op.wakeup = backend_wakeup;
    op.wakeup_ctx = p->f;
    MP_VERBOSE(p, "opening a session: models %s, model %s, %dx%d %s\n", op.model_dir, op.model ? op.model : "4.26",
               mpi->w, mpi->h, p->mgb_fmt == MGB_PIX_P010 ? "P010" : "NV12");
    p->sess = p->be.open(&op);
    if (!p->sess) {
        MP_ERR(p, "mgb_open failed (invalid parameters or ABI mismatch)\n");
        return false;
    }
    p->epoch = p->be.epoch(p->sess);
    if (p->opts->wait_ready) {
        // Offline use (--o=...): there is no real-time clock, so without this the first frames would pass while the
        // engine is created. Waits once, for the engine creation only, never for GPU work.
        int64_t end = mp_time_ns() + 30 * 1000000000LL;
        while (p->be.state(p->sess) == MGB_STATE_INITIALIZING && mp_time_ns() < end)
            mp_sleep_ns(2000000);
    }
    return true;
}

// Called with an empty queue and no open chain: adopt the format of mpi.
static bool configure(struct priv *p, struct mp_image *mpi)
{
    talloc_free(p->pass_reason);
    p->pass_reason = unsupported_reason(p, mpi);
    p->cfg = mpi->params;
    p->cfg.w = mpi->w;
    p->cfg.h = mpi->h;
    p->cfg.imgfmt = mpi->imgfmt;
    p->cfg_valid = true;
    p->nominal_fps = mpi->nominal_fps;
    p->last_half = 0;
    p->dirty = false;
    p->starve_since = 0;
    if (p->pass_reason) {
        log_bypass(p, "%s", p->pass_reason);
        return true;
    }
    return open_or_reconfigure(p, mpi);
}

// ---------------------------------------------------------------------------------------------------------------
// results

static void fill_entry(struct priv *p, struct entry *e, const mgb_result_t *r)
{
    struct mp_image *out = mp_image_alloc(IMGFMT_NV12, r->width, r->height);
    if (!out) {
        MP_ERR(p, "out of memory for a %dx%d frame\n", r->width, r->height);
        return;
    }
    memcpy_pic(out->planes[0], r->y, r->width, r->height, out->stride[0], r->y_stride);
    memcpy_pic(out->planes[1], r->uv, r->width, r->height / 2, out->stride[1], r->uv_stride);
    struct mp_image *src = entry_src(e);
    mp_image_copy_attributes(out, src);
    if (out->w != src->w || out->h != src->h) {
        // The backend downscaled the frame (wider than 1920): the source crop is in source pixels.
        const struct mp_rect c = src->params.crop;
        out->params.crop = (struct mp_rect){
            (int)((int64_t)c.x0 * out->w / src->w), (int)((int64_t)c.y0 * out->h / src->h),
            (int)((int64_t)c.x1 * out->w / src->w), (int)((int64_t)c.y1 * out->h / src->h)};
        if (!mp_image_crop_valid(&out->params))
            out->params.crop = (struct mp_rect){0, 0, out->w, out->h};
    }
    e->img = out;
    mp_image_unrefp(&e->donor);
    e->filled = true;
}

// Collect every result the backend has finished, without blocking. Returns the last fetch code.
static int fetch_results(struct priv *p)
{
    int rc;
    mgb_result_t r;
    while ((rc = p->be.fetch(p->sess, &r)) == MGB_OK) {
        struct entry *e = p->head;
        while (e && e->filled)
            e = e->next;
        enum entry_kind want = r.kind == MGB_KIND_MID ? E_MID : E_ORIG;
        if (e && e->kind == want) {
            fill_entry(p, e, &r);
        } else {
            MP_DBG(p, "dropping an unexpected %s result (pts %" PRId64 ", epoch %u)\n",
                   r.kind == MGB_KIND_MID ? "MID" : "ORIGINAL", r.pts, r.epoch);
        }
        p->be.release_output(p->sess, &r);
    }
    return rc;
}

// Backend bookkeeping at the start of process(). Returns false when the filter failed.
static bool pump_backend(struct priv *p)
{
    if (!p->sess)
        return true;
    int st = p->be.state(p->sess);
    if (st == MGB_STATE_FAILED) {
        MP_ERR(p, "the interpolation backend failed: %s\n", p->be.last_error(p->sess));
        mp_filter_internal_mark_failed(p->f);
        return false;
    }
    note_state(p, st, p->be.bypass_reason(p->sess));
    int rc = fetch_results(p);
    if (rc == MGB_E_FAILED) {
        MP_ERR(p, "the interpolation backend failed: %s\n", p->be.last_error(p->sess));
        mp_filter_internal_mark_failed(p->f);
        return false;
    }
    if (st != MGB_STATE_READY && (rc == MGB_BYPASS || rc == MGB_EMPTY)) {
        if (p->chain_open)
            close_chain(p, false);
        else
            queue_abandon_unfilled(p);
    }
    return true;
}

// ---------------------------------------------------------------------------------------------------------------
// output

static bool emit_head(struct priv *p)
{
    struct entry *e = p->head;
    if (!e || !e->filled || !e->dur_known)
        return false;
    struct mp_image *out = NULL;
    if (e->kind == E_DUP) {
        if (p->last_out)
            out = mp_image_new_ref(p->last_out);
        p->n_dup++;
    } else {
        out = e->img;
        e->img = NULL;
        if (e->kind == E_MID)
            p->n_mid++;
        else
            p->n_orig++;
    }
    if (out) {
        if (e->next && e->next->kind == E_DUP) {
            mp_image_unrefp(&p->last_out);
            p->last_out = mp_image_new_ref(out);
        }
        out->pts = e->pts;
        out->pkt_duration = e->dur;
        if (out->nominal_fps > 0)
            out->nominal_fps *= 2;
        else if (p->nominal_fps > 0)
            out->nominal_fps = p->nominal_fps * 2;
    }
    enum entry_kind e_kind = e->kind;
    p->head = e->next;
    if (!p->head)
        p->tail = NULL;
    if (p->open_orig == e)
        p->open_orig = NULL;
    p->nentries--;
    entry_free(e);
    if (!out)
        return true;   // nothing to write (allocation failure), keep going
    MP_TRACE(p, "out %s pts %.6f dur %.6f\n", e_kind == E_ORIG ? "orig" : e_kind == E_MID ? "mid" : "dup",
             out->pts, out->pkt_duration);
    mp_pin_in_write(p->f->ppins[1], MAKE_FRAME(MP_FRAME_VIDEO, out));
    return true;
}

// ---------------------------------------------------------------------------------------------------------------
// input

enum submit_result { SUBMIT_OK, SUBMIT_BUSY, SUBMIT_NOT_TAKEN };

static enum submit_result try_submit(struct priv *p, struct mp_image *mpi)
{
    mgb_slot_t slot;
    int rc = p->be.acquire_input(p->sess, &slot);
    if (rc == MGB_BUSY)
        return SUBMIT_BUSY;
    if (rc != MGB_OK)
        return SUBMIT_NOT_TAKEN;
    int bpp = p->mgb_fmt == MGB_PIX_P010 ? 2 : 1;
    memcpy_pic(slot.y, mpi->planes[0], mpi->w * bpp, mpi->h, slot.y_stride, mpi->stride[0]);
    memcpy_pic(slot.uv, mpi->planes[1], mpi->w * bpp, mpi->h / 2, slot.uv_stride, mpi->stride[1]);
    int64_t pts_us = llrint(mpi->pts * 1e6);
    rc = p->be.submit(p->sess, &slot, pts_us, p->epoch);
    if (rc == MGB_E_ARG)
        p->be.cancel_input(p->sess, &slot);
    if (rc == MGB_STALE)
        p->epoch = p->be.epoch(p->sess);
    return rc == MGB_OK ? SUBMIT_OK : SUBMIT_NOT_TAKEN;
}

static void enqueue_frame(struct priv *p, struct mp_image *mpi)
{
    double pts = mpi->pts;
    if (p->chain_open) {
        double iv = pts - p->prev_pts;
        if (!(iv > 0)) {
            MP_WARN(p, "non-increasing pts %.6f after %.6f\n", pts, p->prev_pts);
            iv = 2 * frame_half(p);
        }
        double half = iv / 2;
        struct entry *prev = p->open_orig;
        prev->dur = half;
        prev->dur_known = true;
        struct entry *mid = queue_push(p, E_MID, p->prev_pts + half, half);
        mid->donor = mp_image_new_ref(entry_src(prev));
        p->last_half = half;
    }
    struct entry *orig = queue_push(p, E_ORIG, pts, 0);
    if (p->scaled) {
        orig->donor = mpi;      // attributes only; the frame comes back converted
    } else {
        orig->img = mpi;
        orig->filled = true;
    }
    p->open_orig = orig;
    p->prev_pts = pts;
    p->chain_open = true;
    p->starve_since = 0;
}

static bool write_through(struct priv *p, struct mp_image *mpi)
{
    if (!mp_pin_in_needs_data(p->f->ppins[1]))
        return false;
    p->n_pass++;
    MP_TRACE(p, "pass pts %.6f\n", mpi->pts);
    if (p->overloaded && ++p->overload_pass >= p->overload_pass_target) {
        p->overloaded = false;
        p->dirty = true;
        MP_INFO(p, "trying to interpolate again\n");
    }
    mp_pin_in_write(p->f->ppins[1], MAKE_FRAME(MP_FRAME_VIDEO, mpi));
    return true;
}

static void enter_overload(struct priv *p)
{
    p->overloaded = true;
    p->overload_pass = 0;
    p->overload_pass_target = p->overload_pass_target ? MPMIN(p->overload_pass_target * 2, OVERLOAD_PASS_MAX)
                                                      : OVERLOAD_PASS_MIN;
    MP_WARN(p, "the GPU does not keep up; passing frames through for %d frames\n", p->overload_pass_target);
    close_chain(p, false);
    log_bypass(p, "the GPU does not keep up");
}

// Returns true when mpi was consumed (queued, written or dropped), false when it must wait (held).
static void preroll_note_read(struct priv *p, struct mp_image *mpi)
{
    if (!p->preroll)
        return;
    int64_t now = mp_time_ns();
    double dur = mpi->pkt_duration > 0 ? mpi->pkt_duration : mpi->nominal_fps > 0 ? 1.0 / mpi->nominal_fps : 0.04;
    if (p->last_read_ns) {
        if ((now - p->last_read_ns) * 1e-9 >= dur * PREROLL_PACED_RATIO)
            p->paced_reads++;
        else
            p->paced_reads = 0;
        if (p->paced_reads >= PREROLL_PACED_FRAMES) {
            p->preroll = false;
            MP_DBG(p, "reads are paced after %" PRIu64 " fast frames since the reset, interpolating from pts %.6f\n",
                   p->n_preroll, mpi->pts);
        }
    }
    p->last_read_ns = now;
}

static bool handle_video(struct priv *p, struct mp_image *mpi)
{
    if (mpi->pts == MP_NOPTS_VALUE)
        mpi->pts = p->chain_open ? p->prev_pts + 2 * frame_half(p) : 0;

    if (params_changed(p, mpi)) {
        // Drain what is queued under the old format, then reconfigure (as vf_vapoursynth does).
        if (p->chain_open)
            close_chain(p, true);
        if (p->head)
            return false;
        MP_VERBOSE(p, "format %dx%d %s\n", mpi->w, mpi->h, mp_imgfmt_to_name(mpi->imgfmt));
        if (!configure(p, mpi)) {
            mp_image_unrefp(&mpi);
            mp_filter_internal_mark_failed(p->f);
            return true;
        }
    }

    bool active = false;
    if (p->preroll && !p->overloaded && p->sess && !p->pass_reason) {
        p->n_preroll++;
        MP_TRACE(p, "preroll pass pts %.6f\n", mpi->pts);
    } else if (p->sess && !p->pass_reason && !p->overloaded)
        active = p->be.state(p->sess) == MGB_STATE_READY;

    if (!active) {
        if (p->chain_open)
            close_chain(p, false);
        if (p->head)
            return false;       // let the finished frames go first
        return write_through(p, mpi);
    }

    if (!p->chain_open && p->dirty) {
        p->epoch = p->be.reset(p->sess);
        p->dirty = false;
        MP_DBG(p, "chain restarted, epoch %u\n", p->epoch);
    }

    switch (try_submit(p, mpi)) {
    case SUBMIT_OK:
        p->n_in++;
        enqueue_frame(p, mpi);
        return true;
    case SUBMIT_BUSY: {
        p->n_busy++;
        bool starving = mp_pin_in_needs_data(p->f->ppins[1]) && !(p->head && p->head->filled);
        if (!starving) {
            p->starve_since = 0;
        } else if (!p->starve_since) {
            p->starve_since = mp_time_ns();
        } else if (mp_time_ns() - p->starve_since > STARVE_BUSY_NS) {
            p->starve_since = 0;
            enter_overload(p);
            mp_filter_internal_mark_progress(p->f);
        }
        return false;
    }
    default:
        // The backend did not take the frame (state changed, stale epoch): pass it through.
        if (p->chain_open)
            close_chain(p, false);
        if (p->head)
            return false;
        return write_through(p, mpi);
    }
}

static int input_limit(struct priv *p)
{
    return 2 * (MPMAX(p->opts->queue, 1) + 3);
}

static bool read_input(struct priv *p)
{
    bool progress = false;
    for (int guard = 0; guard < 8; guard++) {
        if (p->eof_pending)
            break;
        struct mp_frame frame = MP_NO_FRAME;
        bool was_held = p->held != NULL;
        if (p->held) {
            frame = MAKE_FRAME(MP_FRAME_VIDEO, p->held);
            p->held = NULL;
        } else {
            if (p->nentries >= input_limit(p))
                break;
            // With nothing queued a frame can only go straight through, so only read it when it can be written.
            if (!p->chain_open && !p->head && !mp_pin_in_needs_data(p->f->ppins[1]))
                break;
            frame = mp_pin_out_read(p->in_pin);
        }
        if (frame.type == MP_FRAME_NONE)
            break;
        if (frame.type == MP_FRAME_EOF) {
            close_chain(p, true);
            p->eof_pending = true;
            progress = true;
            break;
        }
        if (frame.type != MP_FRAME_VIDEO) {
            MP_ERR(p, "discarding an unknown frame type\n");
            mp_frame_unref(&frame);
            continue;
        }
        struct mp_image *mpi = frame.data;
        if (!was_held)   // (a held frame was already counted when it was first read)
            preroll_note_read(p, mpi);
        if (!handle_video(p, mpi)) {
            p->held = mpi;
            break;
        }
        progress = true;
    }
    return progress;
}

static void vf_migraphx_process(struct mp_filter *f)
{
    struct priv *p = f->priv;
    int64_t t0 = mp_time_ns();

    if (!pump_backend(p))
        return;

    bool progress = false;

    // Pass-through of a held frame or new input, then queue output.
    progress |= read_input(p);

    if (mp_pin_in_needs_data(f->ppins[1])) {
        if (p->eof_pending && !p->head && !p->chain_open && !p->held) {
            p->eof_pending = false;
            mp_pin_in_write(f->ppins[1], MP_EOF_FRAME);
            progress = false;
        } else if (emit_head(p)) {
            progress = true;
        }
    }

    // More to do right away?
    if (progress && ((p->head && p->head->filled && p->head->dur_known) || p->held || p->eof_pending))
        mp_filter_internal_mark_progress(f);

    p->n_calls++;
    tm_record(p, mp_time_ns() - t0);
}

static void vf_migraphx_reset(struct mp_filter *f)
{
    struct priv *p = f->priv;
    int dropped = p->nentries;
    queue_clear(p);
    mp_image_unrefp(&p->held);
    mp_image_unrefp(&p->last_out);
    p->chain_open = false;
    p->eof_pending = false;
    p->dirty = false;
    p->last_half = 0;
    p->starve_since = 0;
    if (p->overloaded) {
        p->overloaded = false;
        p->overload_pass_target = 0;
    }
    p->n_resets++;
    p->preroll = p->opts->preroll_bypass && !p->opts->wait_ready;
    p->last_read_ns = 0;
    p->paced_reads = 0;
    p->n_preroll = 0;
    if (p->sess) {
        uint32_t old = p->epoch;
        p->epoch = p->be.reset(p->sess);
        // the first frame after the seek starts a new chain: it gets no midpoint before it
        MP_DBG(p, "reset: epoch %u -> %u, dropped %d queued entries\n", old, p->epoch, dropped);
        p->last_st = p->last_reason = -1;
    } else {
        MP_DBG(p, "reset (no session), dropped %d queued entries\n", dropped);
    }
}

static void vf_migraphx_destroy(struct mp_filter *f)
{
    struct priv *p = f->priv;
    queue_clear(p);
    mp_image_unrefp(&p->held);
    mp_image_unrefp(&p->last_out);
    close_session(p);
    log_timing(p);
    // The library stays loaded: unloading the GPU runtime is not safe.
}

static const struct mp_filter_info vf_migraphx_filter = {
    .name = "migraphx",
    .priv_size = sizeof(struct priv),
    .process = vf_migraphx_process,
    .reset = vf_migraphx_reset,
    .destroy = vf_migraphx_destroy,
};

static struct mp_filter *vf_migraphx_create(struct mp_filter *parent, void *options)
{
    struct mp_filter *f = mp_filter_create(parent, &vf_migraphx_filter);
    if (!f) {
        talloc_free(options);
        return NULL;
    }

    mp_filter_add_pin(f, MP_PIN_IN, "in");
    mp_filter_add_pin(f, MP_PIN_OUT, "out");

    struct priv *p = f->priv;
    p->opts = talloc_steal(p, options);
    p->log = f->log;
    p->f = f;
    p->last_st = p->last_reason = -1;

    // NV12 and P010 only: other formats are converted by autoconvert (swscale or hwdownload), never inside the filter.
    struct mp_autoconvert *conv = mp_autoconvert_create(f);
    if (!conv)
        goto error;
    mp_autoconvert_add_imgfmt(conv, IMGFMT_NV12, 0);
    mp_autoconvert_add_imgfmt(conv, IMGFMT_P010, 0);

    struct mp_filter *dur = mp_compute_frame_duration_create(f);
    if (!dur)
        goto error;

    mp_pin_connect(conv->f->pins[0], f->ppins[0]);
    mp_pin_connect(dur->pins[0], conv->f->pins[1]);
    p->in_pin = dur->pins[1];
    return f;

error:
    talloc_free(f);
    return NULL;
}

#define OPT_BASE_STRUCT struct migraphx_opts
static const m_option_t vf_opts_fields[] = {
    {"model", OPT_STRING(model), OPTDEF_STR("4.26")},
    {"models-dir", OPT_STRING(models_dir), .flags = M_OPT_FILE},
    {"queue", OPT_INT(queue), M_RANGE(1, 16), OPTDEF_INT(4)},
    {"max-fps", OPT_DOUBLE(max_fps), M_RANGE(1, 1000), OPTDEF_DOUBLE(32.0)},
    {"backend", OPT_STRING(backend), .flags = M_OPT_FILE},
    {"scene-threshold", OPT_FLOAT(scene_threshold), M_RANGE(-1, 1), OPTDEF_FLOAT(-1)},
    {"bypass-log", OPT_BOOL(bypass_log), OPTDEF_INT(1)},
    {"compile", OPT_BOOL(compile), OPTDEF_INT(1)},
    {"wait-ready", OPT_BOOL(wait_ready)},
    {"preroll-bypass", OPT_BOOL(preroll_bypass), OPTDEF_INT(1)},
    {0}
};

const struct mp_user_filter_entry vf_migraphx = {
    .desc = {
        .description = "RIFE frame interpolation (MIGraphX backend)",
        .name = "migraphx",
        .priv_size = sizeof(OPT_BASE_STRUCT),
        .options = vf_opts_fields,
    },
    .create = vf_migraphx_create,
};
