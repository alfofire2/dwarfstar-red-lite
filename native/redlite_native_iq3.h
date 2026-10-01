#pragma once
/*
 * dev31: CPU reference for the quant formats of the IQ3_XXS Qwen3-Next GGUF that the IQ2_XXS
 * file does not use:
 *   IQ3_XXS (18)  routed experts, attn_qkv / attn_gate          98 bytes / 256 values
 *   IQ3_S   (21)  routed experts (down, some layers), token_embd 110 bytes / 256 values
 *   IQ2_S   (22)  attn_q                                         82 bytes / 256 values
 *   IQ4_XS  (23)  shared expert (some layers)                   136 bytes / 256 values
 * Dequantization follows the pinned llama.cpp ggml-quants.c (dequantize_row_*) value for value, with
 * the codebooks copied from its ggml-common.h (redlite_native_iq3_tables.h). Row dots accumulate the
 * dequantized f32 values in double, like the other CPU oracles.
 */
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RL_GGML_IQ3_XXS 18u
#define RL_GGML_IQ3_S   21u
#define RL_GGML_IQ2_S   22u
#define RL_GGML_IQ4_XS  23u

/* 1 for the four types above */
int rl_iq3_supported(uint32_t ggml_type);

/* dev36: rl_iq3_group8 / the Metal decoders also cover Q4_K (12) for routed experts (the IQ3_M GGUF has Q4_K down
 * projections); rl_iq3_supported() stays IQ-only so dense Q4_K keeps its own kernels */
int rl_iq3_group8_supported(uint32_t ggml_type);

/* bytes of one 256-value block (0 for other types) */
uint32_t rl_iq3_block_bytes(uint32_t ggml_type);

/* bytes of a row of ncols values (0 when unsupported or ncols % 256 != 0) */
size_t rl_iq3_row_bytes(uint32_t ggml_type, uint32_t ncols);

/* values 8*g .. 8*g+7 (g in 0..31) of one 256-value block, in f32 exactly as llama.cpp dequantizes them */
void rl_iq3_group8(uint32_t ggml_type, const uint8_t *block, uint32_t g, float out[8]);

/* whole row; returns 0 for unsupported types or ncols % 256 != 0 */
int rl_iq3_dequant_row(uint32_t ggml_type, const uint8_t *row, uint32_t ncols, float *out);

/* sum_i row[i] * x[i] in double over the dequantized f32 values */
int rl_iq3_row_dot(uint32_t ggml_type, const uint8_t *row, const float *x, uint32_t ncols, double *out);

/* same with a double input vector (the routed-expert down projection of the CPU oracle) */
int rl_iq3_row_dot_d(uint32_t ggml_type, const uint8_t *row, const double *x, uint32_t ncols, double *out);

/* Metal source text declaring the codebooks as `constant` arrays (rl_iq3xxs_grid, rl_iq3s_grid,
 * rl_iq2s_grid, rl_kvalues_iq4nl) and the rl_iq3_group8 decoder for device memory; malloc'd. */
char *rl_iq3_metal_source(void);

/* synthetic checks: block sizes, table spot values, a block decoded by hand */
int rl_iq3_selftest(char *error, size_t error_cap);

#ifdef __cplusplus
}
#endif
