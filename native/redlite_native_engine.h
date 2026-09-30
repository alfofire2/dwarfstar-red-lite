#pragma once

/*
 * Persistent native Qwen3-Next inference engine.
 *
 * One rl_engine owns the mmap'd GGUF, the audited per-layer tensor table, the
 * routed-expert map and up to two independent stateful backends:
 *   - CPU: the scalar double-precision oracle (correctness reference)
 *   - GPU: the persistent Metal runtime (production path)
 * Both backends keep their own DeltaNet conv/recurrent states and full-attention
 * KV caches so a token sequence can be replayed through each and compared.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct rl_engine rl_engine;

typedef enum {
    RL_BACKEND_CPU = 0,
    RL_BACKEND_GPU = 1,
} rl_engine_backend;

typedef struct {
    uint32_t context;      /* KV-cache capacity in positions (default 4096) */
    uint64_t cache_mib;    /* routed-expert cache budget for the Metal path (default 4096) */
    uint32_t top_k;        /* 0 -> model expert_used_count */
    int enable_cpu;        /* keep a CPU oracle backend */
    int enable_gpu;        /* create the Metal backend */
    int cpu_threads;       /* worker threads for the CPU oracle (0 -> hardware count) */
    uint32_t prefill_batch; /* tokens per batched Metal prefill chunk (0 -> 512, 1 -> token-by-token); larger chunks amortize the per-layer expert union */
} rl_engine_config;

typedef struct {
    uint32_t n_layer;
    uint32_t n_recurrent;
    uint32_t n_attention;
    uint32_t hidden;
    uint32_t vocab;
    uint32_t n_expert;
    uint32_t top_k;
    uint32_t context;
    /* DeltaNet */
    uint32_t d_conv, d_inner, d_state, n_group, dt_rank, channels, head_v;
    /* attention */
    uint32_t n_head, n_head_kv, head_dim, rope_dims;
    float rope_freq_base;
    float rms_eps;
    /* memory */
    uint64_t dense_bytes;       /* resident dense weight bytes */
    uint64_t state_bytes;       /* per-backend recurrent + conv + KV bytes */
} rl_engine_info;

typedef struct {
    double embed_ms;
    double layers_ms;
    double recurrent_ms;
    double attention_ms;
    double router_ms;
    double routed_ms;
    double shared_ms;
    double output_ms;
    double total_ms;
    double routed_load_ms;   /* expert miss loading (SSD/page cache) */
    double routed_gpu_ms;    /* expert Metal execution */
    double gpu_ms;           /* GPU-side time of the dense command buffers */
    uint64_t expert_loads;
    uint64_t cache_hits;
    uint64_t cache_misses;
    uint64_t ssd_bytes;
    uint64_t ssd_reads;
    uint32_t resident_slots;
    uint32_t slot_capacity;
    uint32_t expert_plans;   /* batched prefill: expert plans (groups) executed */
    double prep_lru_ms, prep_load_ms, prep_commit_ms, expert_wait_ms;   /* prefill expert phase wall split */
    double prefetch_ms;                                                  /* dev24 prefill: predicted-expert loads overlapped with the GPU */
    uint32_t speculative;          /* dev21: token decoded by the GPU-routed single-command-buffer path */
    uint32_t speculative_fallback; /* dev23: per-layer early-outs of a GPU-routed token (layers whose experts were loaded by the CPU) */
} rl_engine_step_stats;

void rl_engine_config_default(rl_engine_config *cfg);

rl_engine *rl_engine_open(const char *model_path, const rl_engine_config *cfg, char *error, size_t error_cap);
void rl_engine_close(rl_engine *engine);

const rl_engine_info *rl_engine_info_get(const rl_engine *engine);

/* Reset states and position of one backend. */
int rl_engine_reset(rl_engine *engine, rl_engine_backend backend, char *error, size_t error_cap);

uint32_t rl_engine_position(const rl_engine *engine, rl_engine_backend backend);

/*
 * Run one token through a backend at its current position, write vocab logits
 * (may be NULL to skip the LM head) and advance the position.
 */
int rl_engine_step(rl_engine *engine, rl_engine_backend backend, uint32_t token,
                   float *logits, rl_engine_step_stats *stats, char *error, size_t error_cap);

/*
 * Run a token sequence through a backend starting at its current position and
 * advance the position by count. Logits (may be NULL) are those of the last
 * token. The CPU oracle runs the tokens one by one; the Metal backend batches
 * them in chunks of prefill_batch (dev20). Stats accumulate over the call.
 */
int rl_engine_prefill(rl_engine *engine, rl_engine_backend backend, const uint32_t *tokens, uint32_t count,
                      float *logits, rl_engine_step_stats *stats, char *error, size_t error_cap);

/* Host views of the last step's intermediate vectors (hidden floats each). */
const float *rl_engine_last_embedding(const rl_engine *engine, rl_engine_backend backend);
const float *rl_engine_last_layer_output(const rl_engine *engine, rl_engine_backend backend, uint32_t layer);
const float *rl_engine_last_final_norm(const rl_engine *engine, rl_engine_backend backend);
/* Router selection of the last step for a layer (top_k ids); NULL if unavailable. */
const uint32_t *rl_engine_last_router_ids(const rl_engine *engine, rl_engine_backend backend, uint32_t layer);

/* dev21: 1 when the Metal backend preloaded every routed expert at open (full residency); preload_ms receives the time. */
int rl_engine_experts_preloaded(const rl_engine *engine, double *preload_ms);

/* Monotonic milliseconds (same clock as the step statistics). */
double rl_engine_now_ms_public(void);

/* Dequantized token embedding row (hidden floats) into out. */
int rl_engine_embed_token(const rl_engine *engine, uint32_t token, float *out, char *error, size_t error_cap);

#ifdef __cplusplus
}
#endif
