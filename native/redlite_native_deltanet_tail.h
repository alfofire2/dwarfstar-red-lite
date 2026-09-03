#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    double compute_ms;
} rl_dn_tail_telemetry;

#ifdef __APPLE__
int rl_deltanet_tail_gpu_execute(
    const float *core,
    const float *z,
    const float *norm_w,
    const uint8_t *q4_out,
    size_t q4_out_bytes,
    float rms_eps,
    uint32_t head_dim,
    uint32_t value_heads,
    uint32_t hidden,
    float *normalized_gated,
    float *output,
    rl_dn_tail_telemetry *telemetry,
    char *error,
    size_t error_cap);
#endif

#ifdef __cplusplus
}
#endif
