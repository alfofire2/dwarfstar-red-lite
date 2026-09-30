#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RL_DN_PRESTATE_TENSOR_COUNT 3u
#define RL_DN_PRESTATE_MAX_DIMS 8u

typedef enum {
    RL_DN_CONV1D = 0,
    RL_DN_DT = 1,
    RL_DN_A = 2,
} rl_dn_prestate_kind;

typedef struct {
    uint32_t layer;
    rl_dn_prestate_kind kind;
    uint32_t ggml_type;
    uint32_t n_dims;
    uint64_t shape[RL_DN_PRESTATE_MAX_DIMS];
    uint64_t tensor_offset;
    uint64_t tensor_span_bytes;
} rl_dn_prestate_tensor_info;

typedef struct {
    uint64_t bytes_read;
    uint64_t read_calls;
    uint64_t ssd_during_compute_bytes;
    uint64_t ssd_during_compute_calls;
    double read_ms;
    double compute_ms;
} rl_dn_prestate_telemetry;

#ifdef __APPLE__
int rl_deltanet_prestate_gpu_execute(
    const char *model_path,
    const rl_dn_prestate_tensor_info tensors[RL_DN_PRESTATE_TENSOR_COUNT],
    const float *qkv_mixed,
    uint32_t qkv_count,
    const float *ba,
    uint32_t ba_count,
    const float *conv_state,
    uint32_t conv_state_count,
    uint32_t d_conv,
    uint32_t d_inner,
    uint32_t d_state,
    uint32_t dt_rank,
    uint32_t n_group,
    float eps,
    float *beta,
    float *gate,
    float *conv_silu,
    float *q_norm,
    float *k_norm,
    float *v,
    float *next_conv_state,
    rl_dn_prestate_telemetry *telemetry,
    char *error,
    size_t error_cap);
#endif

#ifdef __cplusplus
}
#endif
