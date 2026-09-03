#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float branch_max_abs;
    float branch_max_rel;
    float cache_max_abs;
    float cache_max_rel;
    float residual_max_abs;
    float residual_max_rel;
    float norm_max_abs;
    float norm_max_rel;
    float ffn_max_abs;
    float ffn_max_rel;
    float output_max_abs;
    float output_max_rel;
    int branch_ok;
    int cache_ok;
    int residual_ok;
    int norm_ok;
    int ffn_ok;
    int output_ok;
    int all_ok;
    double branch_cpu_ms;
    double branch_gpu_ms;
    double residual_norm_gpu_ms;
    double final_residual_gpu_ms;
} rl_decoder_block_stats;

int rl_recurrent_block_parity_execute(
    const char *model,
    uint32_t layer,
    uint32_t top_k,
    uint64_t cache_mib,
    const float *cpu_input,
    const float *gpu_input,
    uint32_t input_count,
    float *cpu_output,
    float *gpu_output,
    uint32_t output_count,
    rl_decoder_block_stats *stats,
    char *error,
    size_t error_cap);

int rl_full_attention_block_parity_execute(
    const char *model,
    uint32_t layer,
    uint32_t position,
    uint32_t top_k,
    uint64_t cache_mib,
    const float *cpu_input,
    const float *gpu_input,
    uint32_t input_count,
    float *cpu_output,
    float *gpu_output,
    uint32_t output_count,
    rl_decoder_block_stats *stats,
    char *error,
    size_t error_cap);

#ifdef __cplusplus
}
#endif
