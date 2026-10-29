#ifndef MIGRAPHX_BACKEND_H
#define MIGRAPHX_BACKEND_H

/*
 * libmigraphx_vfi_backend: the playback side of the RIFE engine, for the mpv filter vf_migraphx (Phase 2B).
 *
 * Plain C99, no HIP types. The filter loads the library with dlopen(RTLD_LOCAL | RTLD_NOW), checks mgb_abi_version()
 * and resolves the mgb_* symbols; this header is the only thing it shares with the project. ABI version 1.
 *
 * What it does. A session owns one worker thread. The worker creates the engine (plan loading, ~0.6-1 s) so mgb_open()
 * returns at once, copies submitted host frames to the GPU, interpolates every consecutive pair and copies the
 * results back, all pipelined on one HIP stream. The caller never waits for the GPU: every call below returns
 * immediately, except mgb_close() (waits for the pair in flight) and the "worker is busy creating an engine" part of
 * mgb_close().
 *
 * Frames and timestamps. The caller submits source frames in order. For every two consecutive frames of an epoch the
 * backend returns the interpolated midpoint (kind MGB_KIND_MID) with pts = pts0 + (pts1 - pts0) / 2 (integer
 * arithmetic, the unit is the caller's: microseconds, mpv's pts * 1e6, ...). N frames give N-1 midpoints. The first
 * frame of an epoch gets none. Results come back in submit order.
 *
 * When the engine converts or shrinks the source (10-bit input, or a frame wider than 1920 under playback
 * downscaling) the unmodified frame can not be shown, so each pair also returns the converted/shrunk first frame of
 * the pair (kind MGB_KIND_ORIGINAL, pts = pts0, NV12 at the output size), before its midpoint. mgb_flush() asks for
 * the last frame's converted copy at end of stream. Without conversion no ORIGINAL results exist; the caller keeps
 * its own frames.
 *
 * Epochs. Every submit carries the epoch the caller believes is current. mgb_reset() (seek, discontinuity) starts a new
 * epoch and returns it: frames of older epochs that are still queued are dropped, results of older epochs are never
 * returned, GPU work already in flight finishes in the background and is discarded, and the first frame of the new
 * epoch starts a new pair chain. It does not wait for the GPU.
 *
 * Ownership. Input: mgb_acquire_input() hands out a pinned host buffer of the session's size and format; fill it (tight
 * or strided rows as given), then mgb_submit() it. That is the one copy the caller makes. Output: mgb_fetch() returns
 * pointers into backend-owned pinned host memory (NV12, 8 bit); the caller copies the planes into its own frame and then
 * calls mgb_release_output(). A result holds a backend buffer until released: release promptly. Results may be held
 * across mgb_reset()/mgb_reconfigure(); their memory stays valid until released (or mgb_close()).
 *
 * Threading. A session is used by ONE caller thread (mpv's filter thread) plus the backend's internal worker. All
 * functions except mgb_abi_version() must be called from that one thread (or externally serialised).
 *
 * Error handling. The session state (mgb_state) is one of INITIALIZING, READY, BYPASS (with a reason) or FAILED
 * (with a text). Only READY interpolates; in the other states mgb_submit() returns MGB_BYPASS (the frame was NOT taken,
 * the caller passes it through) and mgb_fetch() returns MGB_BYPASS once nothing is left to return. FAILED is final.
 *
 * Unknown frame size: when no plan covers the size the session goes to BYPASS(NO_PLAN) and, unless
 * MGB_FLAG_NO_COMPILE is set, compiles the plan in a background process (BYPASS(COMPILING), minutes). When that
 * finishes the state becomes BYPASS(PLAN_COMPILED); the plan takes effect at the next mgb_reset() (the session then goes
 * INITIALIZING -> READY).
 *
 * Size or format change: mgb_reconfigure() (also triggered by mgb_submit() with a different size) destroys the engine and
 * creates a new one for the new size (one engine per plan size, never two at once); it bumps the epoch.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__GNUC__)
#define MGB_API __attribute__((visibility("default")))
#else
#define MGB_API
#endif

#define MGB_ABI_VERSION 1

/** Return codes. Negative values are errors. */
#define MGB_OK        0   /**< done; for mgb_fetch: a result was returned */
#define MGB_BUSY      1   /**< mgb_acquire_input / mgb_submit: queue full, try again later (nothing was taken) */
#define MGB_PENDING   2   /**< mgb_fetch: nothing ready yet, work is in flight */
#define MGB_EMPTY     3   /**< mgb_fetch: nothing ready and nothing in flight */
#define MGB_BYPASS    4   /**< not interpolating (state INITIALIZING or BYPASS); the frame was not taken */
#define MGB_STALE     5   /**< mgb_submit: the epoch is not current; the frame was dropped (read mgb_epoch()) */
#define MGB_E_ARG    (-1) /**< invalid argument */
#define MGB_E_FAILED (-2) /**< the session is FAILED (mgb_last_error() says why) */

/** Session state (mgb_state). */
#define MGB_STATE_INITIALIZING 0 /**< the worker is creating the engine */
#define MGB_STATE_READY        1
#define MGB_STATE_BYPASS       2 /**< see mgb_bypass_reason() */
#define MGB_STATE_FAILED       3

/** Bypass reasons (mgb_bypass_reason). */
#define MGB_BYPASS_NONE          0
#define MGB_BYPASS_NO_PLAN       1 /**< no plan for this size; compiling is off or failed */
#define MGB_BYPASS_COMPILING     2 /**< no plan yet; being compiled in the background */
#define MGB_BYPASS_PLAN_COMPILED 3 /**< the plan now exists; takes effect at the next mgb_reset() */
#define MGB_BYPASS_UNSUPPORTED   4 /**< format or size the engine can not do (not 4:2:0 NV12/P010, odd or tiny size) */

/** Pixel formats (same values as the engine's migraphx_pix_fmt_t). */
#define MGB_PIX_NV12 0 /**< 8 bit; Y plane then interleaved CbCr (height / 2 rows) */
#define MGB_PIX_P010 1 /**< 10 bit in 16-bit words, same layout */

/** mgb_open_params_t.flags */
#define MGB_FLAG_NO_COMPILE  1 /**< never compile a missing plan (bypass instead) */
#define MGB_FLAG_NO_DOWNSCALE 2 /**< do not shrink frames wider than 1920 (default: shrink; that is the playback policy) */
#define MGB_FLAG_AFMF        4 /**< method AFMF instead of RIFE (no plans, any even size from 128x128) */

/** Result kinds. */
#define MGB_KIND_MID      0 /**< interpolated midpoint */
#define MGB_KIND_ORIGINAL 1 /**< the pair's first frame converted/shrunk to NV12 at the output size */

/** Log callback: level 0 error, 1 warn, 2 info, 3 debug. Called from the worker or the caller thread. The engine's own
 *  log is process-global; the callback of the most recently opened session receives it. */
typedef void (*mgb_log_fn)(int level, const char* message, void* user);

typedef struct mgb_session mgb_session;

typedef struct mgb_open_params {
    uint32_t struct_size;          /**< sizeof(mgb_open_params_t); use mgb_open_params_init() */
    uint32_t abi_version;          /**< MGB_ABI_VERSION of the caller's header */
    const char* model_dir;         /**< models directory (holds <model>/manifest.json); required */
    const char* model;             /**< "4.26" (default when NULL) or "4.6" */
    int width, height, fmt;        /**< expected source frame size and MGB_PIX_*; mgb_reconfigure() changes them */
    float scene_threshold;         /**< < 0: engine default (0.1); 0: off */
    int queue_depth;               /**< max source frames submitted but not yet on the GPU; 0 = 4 */
    int flags;                     /**< MGB_FLAG_* */
    mgb_log_fn log;                /**< NULL: stderr */
    void* log_user;
    int log_level;                 /**< 0..3, default 1 (warn) when < 0 */
    void (*wakeup)(void* ctx);     /**< worker thread calls this when results are ready or the state changed; NULL ok. Must be thread-safe and not call mgb_*. */
    void* wakeup_ctx;
    const char* compile_script;    /**< tools/compile_mxr.sh; NULL: <dir of this library>/../tools/compile_mxr.sh */
    const char* exporter;          /**< migraphx-vfi-export-stages; NULL: <dir of this library>/migraphx-vfi-export-stages */
} mgb_open_params_t;

/** An input buffer from mgb_acquire_input(). Valid until submitted (or the session is closed). */
typedef struct mgb_slot {
    int id;
    int width, height, fmt;
    uint8_t* y;                    /**< luma plane, height rows of y_stride bytes */
    uint8_t* uv;                   /**< interleaved chroma plane, height / 2 rows of uv_stride bytes */
    int y_stride, uv_stride;
} mgb_slot_t;

/** One result of mgb_fetch(). The planes are NV12, 8 bit, backend-owned until mgb_release_output(). */
typedef struct mgb_result {
    int kind;                      /**< MGB_KIND_* */
    int64_t pts;                   /**< midpoint pts, or the first frame's pts for ORIGINAL */
    uint32_t epoch;
    int scene_cut;                 /**< MID: the pair was judged a scene change (the midpoint is then a copy of the first frame) */
    float scene_score;
    int width, height;             /**< output size */
    const uint8_t* y;
    const uint8_t* uv;
    int y_stride, uv_stride;
    uint64_t handle;               /**< opaque, for mgb_release_output */
} mgb_result_t;

typedef struct mgb_stats {
    uint32_t struct_size;          /**< set to sizeof(mgb_stats_t) before the call; fields beyond it are not written */
    int state, bypass_reason;
    uint32_t epoch;
    int queue_depth;               /**< source frames submitted but not yet finished on the GPU */
    int results_ready;             /**< results waiting for mgb_fetch */
    uint64_t submitted;            /**< frames taken */
    uint64_t pairs;                /**< pairs completed (current epochs and dropped ones) */
    uint64_t dropped_frames;       /**< submitted frames dropped by a reset or a stale epoch */
    uint64_t dropped_results;      /**< finished results discarded because of a reset */
    uint64_t busy;                 /**< BUSY returns (backpressure events) */
    uint64_t scene_cuts;
    double last_pair_ms;           /**< GPU time of the last pair: upload + interpolation + download */
    double avg_pair_ms;            /**< mean over all pairs */
    double max_pair_ms;
    int plan_w, plan_h;            /**< plan size of the engine, 0 before READY */
    int out_w, out_h;
    int scaled;                    /**< 1: the engine shrinks/converts and ORIGINAL results exist */
    char plan[48];
} mgb_stats_t;

MGB_API int mgb_abi_version(void);

/** Fill defaults: struct_size, abi_version, model "4.26", NV12, scene_threshold -1, queue_depth 4, log_level -1. */
MGB_API void mgb_open_params_init(mgb_open_params_t* params);

/** Start a session; returns at once (the engine is created on the worker thread). NULL on invalid parameters or ABI mismatch. */
MGB_API mgb_session* mgb_open(const mgb_open_params_t* params);

MGB_API int mgb_state(mgb_session*);
MGB_API int mgb_bypass_reason(mgb_session*);
/** Text of the failure when the state is FAILED, otherwise the last note (e.g. why a bypass); valid until the next call. */
MGB_API const char* mgb_last_error(mgb_session*);
/** The current epoch. */
MGB_API uint32_t mgb_epoch(mgb_session*);

/** Get a pinned input buffer for a frame of the session's size. MGB_OK, MGB_BUSY (all buffers in use), MGB_BYPASS (not READY),
 *  MGB_E_FAILED. */
MGB_API int mgb_acquire_input(mgb_session*, mgb_slot_t* out);
/** Hand a filled buffer to the GPU. pts: caller's unit. Returns MGB_OK, MGB_STALE (epoch is old; frame dropped), MGB_E_ARG. */
MGB_API int mgb_submit(mgb_session*, const mgb_slot_t* slot, int64_t pts, uint32_t epoch);
/** Give an acquired buffer back without submitting it. */
MGB_API void mgb_cancel_input(mgb_session*, const mgb_slot_t* slot);
/** Convenience: acquire, copy the planes (strides in bytes), submit. width/height/fmt that differ from the session's trigger
 *  mgb_reconfigure() and return MGB_BYPASS. MGB_OK, MGB_BUSY, MGB_BYPASS, MGB_STALE, MGB_E_*. */
MGB_API int mgb_submit_frame(mgb_session*, int width, int height, int fmt, const uint8_t* y, int y_stride,
                             const uint8_t* uv, int uv_stride, int64_t pts, uint32_t epoch);

/** End of stream: queue the converted copy of the last frame (only when the session converts; otherwise a no-op). */
MGB_API int mgb_flush(mgb_session*, uint32_t epoch);

/** Next finished result, in order, without blocking. MGB_OK (filled), MGB_PENDING, MGB_EMPTY, MGB_BYPASS, MGB_E_FAILED. */
MGB_API int mgb_fetch(mgb_session*, mgb_result_t* out);
MGB_API void mgb_release_output(mgb_session*, const mgb_result_t* result);

/** New epoch (returned). Drops queued frames and finished-but-unfetched results of older epochs; does not wait for the GPU.
 *  Also makes a compiled plan take effect (BYPASS(PLAN_COMPILED) -> INITIALIZING). */
MGB_API uint32_t mgb_reset(mgb_session*);
/** Change the source size/format: destroy the engine, create one for the new size, new epoch (returned). Release held
 *  results and input buffers first. Does nothing (returns the epoch) if the size is unchanged. */
MGB_API uint32_t mgb_reconfigure(mgb_session*, int width, int height, int fmt);

MGB_API int mgb_get_stats(mgb_session*, mgb_stats_t* out);

/** Stop the worker (waits for the pair in flight and for engine creation, if running), free everything. NULL ok. */
MGB_API void mgb_close(mgb_session*);

#ifdef __cplusplus
}
#endif

#endif /* MIGRAPHX_BACKEND_H */
