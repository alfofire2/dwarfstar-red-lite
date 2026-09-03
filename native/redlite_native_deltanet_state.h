#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    double compute_ms;
} rl_dn_state_telemetry;

#ifdef __APPLE__
int rl_deltanet_state_gpu_execute(
    const float *q,
    const float *k,
    const float *v,
    const float *gate,
    const float *beta,
    const float *prev_state,
    uint32_t state_size,
    uint32_t key_heads,
    uint32_t value_heads,
    float *delta,
    float *next_state,
    float *output,
    rl_dn_state_telemetry *telemetry,
    char *error,
    size_t error_cap);
#endif

#ifdef __cplusplus
}
#endif
