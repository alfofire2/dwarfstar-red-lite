#pragma once

#include "redlite_native_router.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    double read_ms;
    double compute_ms;
} rl_native_router_telemetry;

int rl_native_router_cpu_f32(
    const char *model_path,
    const rl_router_tensor_info *router,
    const float *input,
    uint32_t input_count,
    float *logits,
    uint32_t logits_count,
    rl_native_router_telemetry *telemetry,
    char *error,
    size_t error_cap);

int rl_native_router_select_softmax_topk(
    const float *logits,
    uint32_t expert_count,
    uint32_t top_k,
    uint32_t *expert_ids,
    float *router_weights,
    float *probabilities,
    char *error,
    size_t error_cap);

#ifdef __APPLE__
int rl_native_router_gpu_f32(
    const char *model_path,
    const rl_router_tensor_info *router,
    const float *input,
    uint32_t input_count,
    float *logits,
    uint32_t logits_count,
    rl_native_router_telemetry *telemetry,
    char *error,
    size_t error_cap);
#endif

#ifdef __cplusplus
}
#endif
