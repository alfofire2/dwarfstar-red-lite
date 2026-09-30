#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    double compute_ms;
} rl_block_ops_telemetry;

#ifdef __APPLE__
int rl_block_residual_rmsnorm_gpu(
    const float *residual,
    const float *branch,
    const float *weight,
    uint32_t n,
    float eps,
    float *sum_out,
    float *norm_out,
    rl_block_ops_telemetry *telemetry,
    char *error,
    size_t error_cap);

int rl_block_residual_gpu(
    const float *residual,
    const float *branch,
    uint32_t n,
    float *out,
    rl_block_ops_telemetry *telemetry,
    char *error,
    size_t error_cap);
#endif

#ifdef __cplusplus
}
#endif
