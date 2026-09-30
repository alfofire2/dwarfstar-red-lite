#pragma once

/*
 * Scalar CPU reference arithmetic for the dense GGML quant formats used by the
 * Qwen3-Next engine outside the routed experts:
 *   F32 (0), F16 (1), Q8_0 (8), Q2_K (10), Q4_K (12), Q5_K (13), Q6_K (14), IQ2_XXS (16).
 * Block layouts follow the pinned llama.cpp ggml-common.h; dot products accumulate
 * in double so they can serve as an independent oracle for the Metal kernels.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

float rl_quant_f16_to_f32(uint16_t h);

/* Dequantize one row (ncols values, ncols % 256 == 0 for K-quants) into out. */
int rl_quant_dequant_row(uint32_t ggml_type, const uint8_t *row, uint32_t ncols,
                         const uint8_t *iq2_xxs_grid, float *out);

/*
 * Row dot product: sum_i row[i] * x[i] in double precision.
 * iq2_xxs_grid may be NULL unless ggml_type == 16.
 */
int rl_quant_row_dot(uint32_t ggml_type, const uint8_t *row, const float *x, uint32_t ncols,
                     const uint8_t *iq2_xxs_grid, double *out);

/* Synthetic self-test of every supported format against hand-computed values. */
int rl_quant_selftest(char *error, size_t error_cap);

#ifdef __cplusplus
}
#endif
