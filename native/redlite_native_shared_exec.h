#pragma once

#include "redlite_native_shared.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint64_t bytes_read;
    uint64_t read_calls;
    double read_ms;
    double compute_ms;
    double scalar_gate;
} rl_native_shared_telemetry;

size_t rl_native_shared_row_bytes(uint32_t ggml_type, uint32_t ncols);

int rl_native_shared_quant_row_dot(
    const uint8_t *row,
    size_t row_bytes,
    uint32_t ggml_type,
    const float *input,
    uint32_t ncols,
    const uint8_t *iq2_grid,
    size_t iq2_grid_count,
    double *out,
    char *error,
    size_t error_cap);

int rl_native_shared_cpu_execute(
    const char *model_path,
    const rl_shared_tensor_info tensors[4],
    const float *input,
    uint32_t input_count,
    uint32_t row_start,
    uint32_t row_count,
    double *output,
    uint32_t output_count,
    rl_native_shared_telemetry *telemetry,
    char *error,
    size_t error_cap);

#ifdef __APPLE__
int rl_native_shared_gpu_execute(
    const char *model_path,
    const rl_shared_tensor_info tensors[4],
    const float *input,
    uint32_t input_count,
    uint32_t row_start,
    uint32_t row_count,
    float *output,
    uint32_t output_count,
    rl_native_shared_telemetry *telemetry,
    char *error,
    size_t error_cap);
#endif

#ifdef __cplusplus
}
#endif
