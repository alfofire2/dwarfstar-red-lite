#pragma once

#include "redlite_native_decoder_block.h"
#include "redlite_native_layer_map.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RL_DECODER_STACK_HIDDEN 2048u
#define RL_DECODER_STACK_MAX_TOPK 64u

typedef struct {
    rl_layer_map_kind kind;
    rl_decoder_block_stats block;
    float output_max_abs;
    float output_max_rel;
} rl_decoder_stack_layer_stats;

typedef struct {
    uint32_t model_layers;
    uint32_t requested_layers;
    uint32_t executed_layers;
    uint32_t model_recurrent;
    uint32_t model_full_attention;
    uint32_t recurrent_executed;
    uint32_t full_attention_executed;
    float max_abs;
    float max_rel;
    double elapsed_ms;
    int all_ok;
    int complete;
    rl_decoder_stack_layer_stats layer[RL_LAYER_MAP_MAX_LAYERS];
} rl_decoder_stack_stats;

int rl_decoder_stack_parity_execute(
    const char *model,
    uint32_t position,
    uint32_t top_k,
    uint64_t cache_mib,
    uint32_t requested_layers,
    const float *cpu_input,
    const float *gpu_input,
    uint32_t input_count,
    float *cpu_output,
    float *gpu_output,
    uint32_t output_count,
    rl_decoder_stack_stats *stats,
    char *error,
    size_t error_cap);

#ifdef __cplusplus
}
#endif
