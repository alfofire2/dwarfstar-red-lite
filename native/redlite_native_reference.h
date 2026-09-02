#pragma once

#include "redlite_native_gguf.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

size_t rl_native_row_bytes(uint32_t ggml_type, uint32_t ncols);

int rl_native_quant_row_dot(
    const uint8_t *row,
    size_t row_bytes,
    uint32_t ggml_type,
    const float *input,
    uint32_t ncols,
    const int8_t *grid,
    size_t grid_count,
    float *out,
    char *error,
    size_t error_cap);

int rl_native_reference_topk(
    const char *model_path,
    const rl_expert_map *map,
    uint32_t layer,
    const uint32_t *expert_ids,
    const float *router_weights,
    uint32_t top_k,
    uint32_t row_start,
    uint32_t row_count,
    const float *input,
    uint32_t input_count,
    float *output,
    uint32_t output_count,
    double *elapsed_ms,
    char *error,
    size_t error_cap);

#ifdef __cplusplus
}
#endif
