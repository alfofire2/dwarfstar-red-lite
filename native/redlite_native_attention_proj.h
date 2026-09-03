#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RL_ATTN_PROJ_TENSOR_COUNT 7u
#define RL_ATTN_PROJ_MAX_DIMS 8u
#define RL_ATTN_MAX_CONTEXT 64u

typedef enum {
    RL_ATTN_PROJ_INPUT_NORM = 0,
    RL_ATTN_PROJ_Q_GATE = 1,
    RL_ATTN_PROJ_K = 2,
    RL_ATTN_PROJ_V = 3,
    RL_ATTN_PROJ_Q_NORM = 4,
    RL_ATTN_PROJ_K_NORM = 5,
    RL_ATTN_PROJ_OUTPUT = 6,
} rl_attn_proj_kind;

typedef struct {
    uint32_t layer;
    rl_attn_proj_kind kind;
    uint32_t ggml_type;
    uint32_t n_dims;
    uint64_t shape[RL_ATTN_PROJ_MAX_DIMS];
    uint64_t tensor_offset;
    uint64_t tensor_span_bytes;
} rl_attn_proj_tensor_info;

typedef struct {
    uint64_t bytes_read;
    uint64_t read_calls;
    double read_ms;
    double compute_ms;
} rl_attn_proj_telemetry;

#ifdef __APPLE__
int rl_attention_gpu_execute(
    const char *model_path,
    const rl_attn_proj_tensor_info tensors[RL_ATTN_PROJ_TENSOR_COUNT],
    const float *input,
    uint32_t input_count,
    float rms_eps,
    uint32_t position,
    uint32_t rope_dims,
    float rope_freq_base,
    float *key_cache,
    float *value_cache,
    uint32_t cache_float_count,
    float *input_norm_out,
    uint32_t input_norm_out_count,
    float *query_out,
    uint32_t query_out_count,
    float *gate_out,
    uint32_t gate_out_count,
    float *key_out,
    uint32_t key_out_count,
    float *value_out,
    uint32_t value_out_count,
    float *attention_out,
    uint32_t attention_out_count,
    float *gated_out,
    uint32_t gated_out_count,
    float *output,
    uint32_t output_count,
    rl_attn_proj_telemetry *telemetry,
    char *error,
    size_t error_cap);
#endif

#ifdef __cplusplus
}
#endif
