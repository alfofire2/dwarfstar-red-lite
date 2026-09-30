#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

uint32_t redmetal_quant_abi_version(void);
const char * redmetal_quant_last_error(void);

int redmetal_quant_rows(
    const char *model_path,
    uint64_t matrix_file_offset,
    uint64_t matrix_bytes,
    uint32_t ggml_type,
    uint32_t ncols,
    uint32_t nrows,
    uint32_t row_start,
    uint32_t row_count,
    const float *input,
    uint32_t input_count,
    const int8_t *grid,
    uint32_t grid_count,
    float *output,
    uint32_t output_count,
    double *io_ms,
    double *gpu_ms);

#ifdef __cplusplus
}
#endif
