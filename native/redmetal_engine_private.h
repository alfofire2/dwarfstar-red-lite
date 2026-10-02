#pragma once
/* Private layout of the persistent Metal engine, shared by redmetal_engine.m and redmetal_engine_prefill.m. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "redlite_native_engine_internal.h"
#include "redlite_native_metal.h"

#define RL_ENGINE_KV_PAD 32u   /* KV cache positions allocated past the context (tiled prefill attention) */

typedef struct {
    __unsafe_unretained id<MTLBuffer> buf;
    NSUInteger off;
    uint32_t type;
    uint32_t rows;
    uint32_t cols;
} mweight;

typedef struct {
    mweight attn_norm, post_norm;
    mweight qkv, z, ba, conv, dt, a, ssm_norm, ssm_out;
    mweight q, k, v, q_norm, k_norm, o;
    mweight router, sh_gate_inp, sh_gate, sh_up, sh_down;
} mlayer;

struct rl_metal_engine {
    int profile;
    double prof_ms[16];
    id<MTLDevice> dev;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> lib;
    id<MTLComputePipelineState> p_rms, p_resid_rms, p_scale_add;
    id<MTLComputePipelineState> p_moe_tail;                    /* dev39: fused end of a GPU-routed layer */
    /* dev45: MTP block (kcache/vcache[n_attention] hold its KV rows) */
    int mtp;
    mlayer mtp_w;
    mweight mtp_eh_proj, mtp_enorm, mtp_hnorm, mtp_head_norm, mtp_head;
    id<MTLBuffer> mtp_emb, mtp_ea, mtp_eb, mtp_cat, mtp_slots, mtp_weights, mtp_ids, mtp_miss;
    rl_native_metal_runtime *mtp_experts;
    /* dev45: 2-row verify: two-vector GEMV kernels, row-1 copies of the per-token buffers (swapped into the fields
     * above while row 1 is encoded), DeltaNet state snapshots after row 0 */
    id<MTLComputePipelineState> p_r2_f32, p_r2_q8, p_r2_q4k, p_r2_q5k, p_r2_q6k, p_r2_iq2xxs;
    id<MTLBuffer> alt[40];
    __unsafe_unretained id<MTLBuffer> *snap_conv, *snap_rec;
    int verify_ready;                  /* buffers allocated */
    int verify_pending;                /* a verify ran: rl_metal_engine_verify_commit must follow */
    id<MTLComputePipelineState> p_rows_f32, p_rows_q8, p_rows_q4k, p_rows_q5k, p_rows_q6k, p_rows_iq2xxs, p_rows_iq3;
    id<MTLComputePipelineState> p_dn_ba, p_dn_conv, p_dn_l2, p_dn_shift, p_dn_state, p_dn_tail;
    id<MTLComputePipelineState> p_attn_prep, p_attn_gqa;
    id<MTLComputePipelineState> p_attn_split, p_attn_merge;   /* dev26 split-K decode attention */
    id<MTLComputePipelineState> p_attn_split_g;               /* dev35: one threadgroup per KV head and block */
    int attn_group;                                           /* RL_ENGINE_ATTN_GROUP != 0 (default) */
    int fuse_tail;                                            /* dev39: rl_moe_tail (RL_ENGINE_FUSE_TAIL != 0, default) */
    int concurrent;                                           /* dev38: concurrent decode encoders (default; RL_ENGINE_CONCURRENT=0 or profile: serial) */
    uint32_t attn_blk;                                        /* positions per grouped split-K block (RL_ENGINE_ATTN_BLK, default 128) */
    id<MTLBuffer> attn_ml, attn_acc;                          /* per (head, 256-position block) partials */
    int attn_split;
    id<MTLComputePipelineState> p_sh_scalar, p_sh_silu;
    id<MTLBuffer> grid;
    mlayer *layers;
    mweight output_norm, output;
    __unsafe_unretained id<MTLBuffer> *conv_state, *rec_state, *kcache, *vcache;
    NSMutableArray *keep;
    uint32_t n_recurrent, n_attention;
    id<MTLBuffer> x, normed, branch, resid, ffn_in;
    id<MTLBuffer> qkv, z, ba, beta, gate, conv_silu, q, k, delta, core, ng, next_conv, rec_scratch;
    id<MTLBuffer> qgate_raw, k_raw, value, query, agate, key, query_rope, gated;
    id<MTLBuffer> router_logits, sh_gate, sh_up, sh_act, sh_out, scalar, routed, final_norm, logits;
    rl_native_metal_runtime *experts;
    float *routed_host;
    uint64_t resident_bytes;
    uint64_t last_bytes_read, last_calls, last_hits, last_misses, last_loads;
    double last_read_ms;
    struct rl_metal_prefill *pf;   /* batched prefill state (dev20), created on first use */
    id engine_rs;                  /* dev30: MTLResidencySet (macOS 15+) of every engine buffer, attached to the queue;
                                    * RL_ENGINE_RESIDENCY=0 leaves residency to each command buffer */
    /* dev21: GPU-routed decode */
    int spec_enabled;              /* RL_ENGINE_SPECULATIVE != 0 and residency table available */
    uint64_t misses_seen;          /* pool cache misses observed so far (to know whether the last token missed) */
    int last_token_missed;         /* the previous token needed a load: decode the next one synchronously */
    id<MTLComputePipelineState> p_route, p_copy;
    id<MTLComputePipelineState> p_rows2_q4k, p_rows2_q6k, p_rows2_iq2xxs, p_rows2_f32;   /* dev22 sub-block decode GEMV */
    int rows2;
    id<MTLBuffer> abort, abort_zero;
    int prefetch;                      /* dev23: RL_ENGINE_PREFETCH != 0 (default): pre-gated expert prefetch in the synchronous decode */
    id<MTLBuffer> pred_logits;         /* next layer's router logits from this layer's FFN input */
    double prefetch_ms;                /* CPU time spent in prefetch loads (overlapped with the GPU) */
    uint32_t sync_left;                /* dev23: tokens to decode synchronously before probing the GPU-routed path again */   /* dev23 early-out flag (index 30 of every decode kernel) and its never-set twin */                     /* RL_ENGINE_ROWS2 != 0 (default): decode uses the sub-block kernels */
    id<MTLBuffer> plan_slots, plan_weights, plan_ids, plan_miss, layer_out_gpu;
    int preloaded;                 /* every routed expert was loaded at open (cache holds them all) */
    double preload_ms;
};


/* helpers defined in redmetal_engine.m */
id<MTLBuffer> new_buf(rl_metal_engine *m, size_t bytes);
id<MTLComputePipelineState> rows_pipe(rl_metal_engine *m, uint32_t type);
id<MTLComputePipelineState> make_pipe(id<MTLDevice> dev, id<MTLLibrary> lib, NSString *name, char *error, size_t cap);
void enc_1d(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p, NSUInteger n, NSUInteger tg_max);
uint32_t lanes_for(uint32_t type, uint32_t ncols);
void enc_rows(rl_metal_engine *m, id<MTLCommandBuffer> cb, const mweight *w, id<MTLBuffer> x, id<MTLBuffer> out);
id<MTLLibrary> rl_metal_engine_library(id<MTLDevice> dev, NSError **err);
void emit_rows(rl_metal_engine *m, id<MTLComputeCommandEncoder> enc, const mweight *w, id<MTLBuffer> x, id<MTLBuffer> out);
void emit_rms(rl_metal_engine *m, id<MTLComputeCommandEncoder> enc, id<MTLBuffer> x, const mweight *w, id<MTLBuffer> y, uint32_t n, float eps);
void enc_rms(rl_metal_engine *m, id<MTLCommandBuffer> cb, id<MTLBuffer> x, const mweight *w, id<MTLBuffer> y, uint32_t n, float eps);
int commit_wait(id<MTLCommandBuffer> cb, const char *what, double *gpu_ms, char *error, size_t cap);

/* defined in redmetal_engine_prefill.m */
void rl_metal_prefill_destroy(struct rl_metal_prefill *pf);
