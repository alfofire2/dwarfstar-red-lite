#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RL_DN_PROJ_TENSOR_COUNT 4u
#define RL_DN_MAX_DIMS 8u

typedef enum {
    RL_DN_ATTN_NORM = 0,
    RL_DN_QKV = 1,
    RL_DN_Z_GATE = 2,
    RL_DN_BETA_ALPHA = 3,
} rl_dn_proj_kind;

typedef struct {
    uint32_t layer;
    rl_dn_proj_kind kind;
    uint32_t ggml_type;
    uint32_t n_dims;
    uint64_t shape[RL_DN_MAX_DIMS];
    uint64_t tensor_offset;
    uint64_t tensor_span_bytes;
} rl_dn_proj_tensor_info;

typedef struct {
    uint64_t bytes_read;
    uint64_t read_calls;
    double read_ms;
    double compute_ms;
} rl_dn_proj_telemetry;

#ifdef __APPLE__
int rl_deltanet_proj_gpu_execute(
    const char *model_path,
    const rl_dn_proj_tensor_info tensors[RL_DN_PROJ_TENSOR_COUNT],
    const float *input,
    uint32_t input_count,
    float rms_eps,
    uint32_t rows,
    float *norm_out,
    uint32_t norm_out_count,
    float *qkv_out,
    float *z_out,
    float *ba_out,
    rl_dn_proj_telemetry *telemetry,
    char *error,
    size_t error_cap);
#endif

#ifdef __cplusplus
}
#endif
