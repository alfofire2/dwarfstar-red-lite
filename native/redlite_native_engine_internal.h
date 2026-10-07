#pragma once

/* Private engine structures shared by the C core, the CPU oracle and the Metal backend. */

#include "redlite_native_engine.h"
#include "redlite_native_gguf.h"
#include "redlite_native_gguf_dir.h"
#include "redlite_native_iq2_xxs.h"
#include "redlite_native_layer_map.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RL_ENGINE_MAX_TOPK 64u

typedef struct {
    rl_layer_map_kind kind;
    uint32_t recurrent_index;   /* index among recurrent layers, or UINT32_MAX */
    uint32_t attention_index;   /* index among attention layers, or UINT32_MAX */
    const rl_gguf_tensor *attn_norm;
    const rl_gguf_tensor *post_norm;
    /* recurrent (Gated DeltaNet) */
    const rl_gguf_tensor *qkv, *z, *ba, *conv, *dt, *a, *ssm_norm, *ssm_out;
    /* full attention */
    const rl_gguf_tensor *q, *k, *v, *q_norm, *k_norm, *o;
    /* MoE */
    const rl_gguf_tensor *router, *sh_gate_inp, *sh_gate, *sh_up, *sh_down;
} rl_layer_tensors;

typedef struct {
    uint32_t position;
    int host_state;     /* 1 when conv/rec/kcache/vcache are allocated (CPU oracle); the Metal backend keeps them on the GPU */
    float *conv;        /* [n_recurrent][(d_conv-1)*channels] */
    float *rec;         /* [n_recurrent][dt_rank*head_v*head_v] */
    float *kcache;      /* [n_attention][context][n_head_kv*head_dim] */
    float *vcache;      /* same layout as kcache */
    float *embed;       /* [hidden] */
    float *layer_out;   /* [n_layer][hidden] */
    float *final_norm;  /* [hidden] */
    uint32_t *router_ids; /* [n_layer][top_k] */
} rl_backend_state;

typedef struct rl_metal_engine rl_metal_engine;

struct rl_engine {
    rl_gguf_model gguf;
    rl_engine_config cfg;
    rl_engine_info info;
    rl_layer_tensors *layers;
    const rl_gguf_tensor *tok_embd;
    const rl_gguf_tensor *output_norm;
    const rl_gguf_tensor *output;
    rl_expert_map expert_map;
    rl_native_layer_map layer_map;
    uint8_t iq2_grid[RL_IQ2_XXS_GRID_COUNT];
    int cpu_enabled;
    int gpu_enabled;
    rl_backend_state cpu;
    rl_backend_state gpu;
    rl_backend_state gpu_parked;          /* dev56: the other slot's host state (rl_engine_select_slot swaps) */
    int slot, slots;                      /* selected slot; 2 once rl_engine_slots_enable ran */
    rl_metal_engine *metal;
    int cpu_threads;
    /* dev45: MTP block */
    int mtp_enabled;
    rl_gguf_model mtp_gguf;
    rl_layer_tensors mtp;                 /* attention + MoE of the MTP block (attention_index = info.n_attention) */
    const rl_gguf_tensor *mtp_eh_proj, *mtp_enorm, *mtp_hnorm, *mtp_head_norm, *mtp_head, *mtp_embd;
    rl_expert_map mtp_expert_map;
    uint32_t mtp_layer;                   /* block index in the MTP file (== trunk n_layer) */
    float *steer_vec;                     /* dev52: hidden floats, NULL = no steering */
    uint32_t steer_first, steer_last;
    float steer_strength;
};
/* dev52: 1 when layer l of a decoded token is steered */
static inline int rl_engine_steers(const rl_engine *e, uint32_t l) {
    return e->steer_vec && e->steer_strength != 0.0f && l >= e->steer_first && l <= e->steer_last;
}

/* CPU oracle (redlite_native_engine_cpu.c) */
int rl_engine_cpu_step(rl_engine *e, uint32_t token, float *logits, rl_engine_step_stats *stats, char *error, size_t cap);

/* Metal backend (redmetal_engine.m) */
rl_metal_engine *rl_metal_engine_create(rl_engine *e, char *error, size_t cap);
void rl_metal_engine_destroy(rl_metal_engine *m);
int rl_metal_engine_reset(rl_metal_engine *m, char *error, size_t cap);
int rl_metal_engine_set_steering(rl_metal_engine *m, const float *vector, uint32_t hidden, char *error, size_t cap);   /* dev52 */
int rl_metal_engine_state_io(rl_metal_engine *m, FILE *f, size_t kv_bytes, int save);   /* dev43 */
int rl_metal_engine_verify2(rl_engine *e, rl_metal_engine *m, uint32_t t0, uint32_t t1, float *logits0, float *logits1, char *error, size_t cap);
int rl_metal_engine_verify_commit(rl_engine *e, rl_metal_engine *m, int accepted, char *error, size_t cap);   /* dev45 */
int rl_metal_engine_verify_available(const rl_metal_engine *m);   /* dev72 */
int rl_metal_prefill_kernel_bench(rl_engine *e, rl_metal_engine *m, uint32_t B, uint32_t position0, uint32_t layer, char *error, size_t cap);   /* dev72 */
int rl_metal_prefill_attn_bench_standalone(uint32_t B, uint32_t position0, char *error, size_t cap);   /* dev72 */
int rl_metal_expert_bench_standalone(rl_engine *e, const char *model_path, uint32_t layer, uint32_t B, char *error, size_t cap);   /* dev72 */
int rl_metal_engine_slots_enable(rl_engine *e, rl_metal_engine *m, char *error, size_t cap);   /* dev56 */
void rl_metal_engine_swap_slot(rl_metal_engine *m);
int rl_metal_engine_step_pair(rl_engine *e, rl_metal_engine *m, uint32_t t0, uint32_t t1, float *logits0, float *logits1,
                              char *error, size_t cap);
int rl_metal_engine_mtp_draft(rl_engine *e, rl_metal_engine *m, const float *embedding, uint32_t mtp_position, uint32_t *draft,
                              float *logits, char *error, size_t cap);   /* dev45 */
int rl_metal_engine_prefill(rl_engine *e, rl_metal_engine *m, const uint32_t *tokens, uint32_t count, float *logits,
                            rl_engine_step_stats *stats, char *error, size_t cap);
int rl_metal_engine_preloaded(const rl_metal_engine *m, double *preload_ms);
int rl_metal_engine_step_sync(rl_engine *e, rl_metal_engine *m, uint32_t token, float *logits,
                              rl_engine_step_stats *stats, char *error, size_t cap);
int rl_metal_engine_step(rl_engine *e, rl_metal_engine *m, uint32_t token, float *logits,
                         rl_engine_step_stats *stats, char *error, size_t cap);
uint64_t rl_metal_engine_resident_bytes(const rl_metal_engine *m);

/* shared helpers (redlite_native_engine.c) */
int rl_backend_state_alloc(rl_engine *e, rl_backend_state *s, int host_state, char *error, size_t cap);
void rl_backend_state_free(rl_backend_state *s);
void rl_backend_state_reset(rl_engine *e, rl_backend_state *s);
size_t rl_engine_conv_count(const rl_engine *e);
size_t rl_engine_rec_count(const rl_engine *e);
size_t rl_engine_kv_row_count(const rl_engine *e);
double rl_engine_now_ms(void);

#ifdef __cplusplus
}
#endif
