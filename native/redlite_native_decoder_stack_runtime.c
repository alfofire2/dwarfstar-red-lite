#define _POSIX_C_SOURCE 200809L

#include "redlite_native_decoder_stack.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double stack_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

static void stack_compare(const float *gpu, const float *cpu, uint32_t count,
        float *max_abs, float *max_rel) {
    *max_abs = 0.0f;
    *max_rel = 0.0f;
    for (uint32_t i = 0; i < count; ++i) {
        const float absolute = fabsf(gpu[i] - cpu[i]);
        const float relative = absolute / fmaxf(fabsf(cpu[i]), 1.0e-12f);
        if (absolute > *max_abs) *max_abs = absolute;
        if (relative > *max_rel) *max_rel = relative;
    }
}

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
        size_t error_cap) {
    if (!model || !cpu_input || !gpu_input || !cpu_output || !gpu_output || !stats ||
        input_count != RL_DECODER_STACK_HIDDEN || output_count < RL_DECODER_STACK_HIDDEN ||
        !top_k || top_k > RL_DECODER_STACK_MAX_TOPK || !cache_mib) {
        if (error && error_cap) snprintf(error, error_cap, "invalid decoder-stack arguments");
        return 0;
    }
    memset(stats, 0, sizeof(*stats));
    rl_native_layer_map map;
    if (!rl_native_layer_map_audit(model, &map, error, error_cap)) return 0;
    const uint32_t layers = requested_layers ? requested_layers : map.layer_count;
    if (layers > map.layer_count) {
        if (error && error_cap) snprintf(error, error_cap,
            "requested %u layers but GGUF contains %u", layers, map.layer_count);
        return 0;
    }

    float *cpu_current = (float *)malloc((size_t)RL_DECODER_STACK_HIDDEN * sizeof(float));
    float *gpu_current = (float *)malloc((size_t)RL_DECODER_STACK_HIDDEN * sizeof(float));
    float *cpu_next = (float *)calloc(RL_DECODER_STACK_HIDDEN, sizeof(float));
    float *gpu_next = (float *)calloc(RL_DECODER_STACK_HIDDEN, sizeof(float));
    if (!cpu_current || !gpu_current || !cpu_next || !gpu_next) {
        free(cpu_current); free(gpu_current); free(cpu_next); free(gpu_next);
        if (error && error_cap) snprintf(error, error_cap, "decoder-stack allocation failed");
        return 0;
    }
    memcpy(cpu_current, cpu_input, (size_t)RL_DECODER_STACK_HIDDEN * sizeof(float));
    memcpy(gpu_current, gpu_input, (size_t)RL_DECODER_STACK_HIDDEN * sizeof(float));

    stats->model_layers = map.layer_count;
    stats->requested_layers = layers;
    stats->model_recurrent = map.recurrent_count;
    stats->model_full_attention = map.full_attention_count;
    stats->all_ok = 1;
    const double start = stack_now_ms();
    for (uint32_t layer = 0; layer < layers; ++layer) {
        rl_decoder_stack_layer_stats *ls = &stats->layer[layer];
        ls->kind = map.kind[layer];
        int executed = 0;
        if (map.kind[layer] == RL_LAYER_MAP_RECURRENT) {
            executed = rl_recurrent_block_parity_execute(model, layer, top_k, cache_mib,
                cpu_current, gpu_current, RL_DECODER_STACK_HIDDEN,
                cpu_next, gpu_next, RL_DECODER_STACK_HIDDEN,
                &ls->block, error, error_cap);
            stats->recurrent_executed++;
        } else if (map.kind[layer] == RL_LAYER_MAP_FULL_ATTENTION) {
            executed = rl_full_attention_block_parity_execute(model, layer, position, top_k, cache_mib,
                cpu_current, gpu_current, RL_DECODER_STACK_HIDDEN,
                cpu_next, gpu_next, RL_DECODER_STACK_HIDDEN,
                &ls->block, error, error_cap);
            stats->full_attention_executed++;
        }
        if (!executed) {
            char detail[512] = {0};
            if (error && error_cap) snprintf(detail, sizeof(detail), "%s", error);
            if (error && error_cap) snprintf(error, error_cap, "layer %u (%s) execution failed: %s",
                layer, rl_native_layer_map_kind_name(map.kind[layer]), detail[0] ? detail : "unknown error");
            stats->all_ok = 0;
            stats->elapsed_ms = stack_now_ms() - start;
            free(cpu_current); free(gpu_current); free(cpu_next); free(gpu_next);
            return 0;
        }
        stack_compare(gpu_next, cpu_next, RL_DECODER_STACK_HIDDEN,
            &ls->output_max_abs, &ls->output_max_rel);
        if (ls->output_max_abs > stats->max_abs) stats->max_abs = ls->output_max_abs;
        if (ls->output_max_rel > stats->max_rel) stats->max_rel = ls->output_max_rel;
        stats->executed_layers++;
        if (!ls->block.all_ok) {
            stats->all_ok = 0;
            break;
        }
        float *swap = cpu_current; cpu_current = cpu_next; cpu_next = swap;
        swap = gpu_current; gpu_current = gpu_next; gpu_next = swap;
    }
    stats->elapsed_ms = stack_now_ms() - start;
    stats->complete = stats->all_ok && stats->executed_layers == map.layer_count &&
        stats->recurrent_executed == map.recurrent_count &&
        stats->full_attention_executed == map.full_attention_count;
    memcpy(cpu_output, cpu_current, (size_t)RL_DECODER_STACK_HIDDEN * sizeof(float));
    memcpy(gpu_output, gpu_current, (size_t)RL_DECODER_STACK_HIDDEN * sizeof(float));
    free(cpu_current); free(gpu_current); free(cpu_next); free(gpu_next);
    if (error && error_cap) error[0] = '\0';
    return 1;
}
