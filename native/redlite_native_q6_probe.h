#pragma once

#include "redlite_native_shared.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __APPLE__
int rl_native_q6_gpu_rows(
    const char *model_path,
    const rl_shared_tensor_info *tensor,
    const float *input,
    uint32_t input_count,
    uint32_t row_start,
    uint32_t row_count,
    float *output,
    uint32_t output_count,
    char *error,
    size_t error_cap);
#endif

#ifdef __cplusplus
}
#endif
