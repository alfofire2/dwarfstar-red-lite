#define _POSIX_C_SOURCE 200809L

#include "redlite_native_decoder_stack.h"

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int ds_parse_u32(const char *text, uint32_t *out) {
    if (!text || !*text) return 0;
    errno = 0;
    char *end = NULL;
    const unsigned long value = strtoul(text, &end, 10);
    if (errno || !end || *end || value > UINT32_MAX) return 0;
    *out = (uint32_t)value;
    return 1;
}

static int ds_parse_u64(const char *text, uint64_t *out) {
    if (!text || !*text) return 0;
    errno = 0;
    char *end = NULL;
    const unsigned long long value = strtoull(text, &end, 10);
    if (errno || !end || *end) return 0;
    *out = (uint64_t)value;
    return 1;
}

static void ds_make_input(float *values, uint32_t count) {
    for (uint32_t i = 0; i < count; ++i)
        values[i] = (float)(((int)((i * 17u + 3u) % 61u) - 30) / 32.0);
}

static void ds_usage(FILE *out) {
    fprintf(out,
        "redlite-decoder-stack 0.3.0.dev17\n\n"
        "Usage:\n"
        "  redlite-decoder-stack parity MODEL [--position N] [--top-k N] [--cache-mib N] [--layers N]\n\n"
        "Runs a single hidden vector through the audited Qwen3-Next recurrent/full-attention layer sequence.\n");
}

int main(int argc, char **argv) {
    if (argc == 2 && (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0)) {
        ds_usage(stdout);
        return 0;
    }
    if (argc < 3 || strcmp(argv[1], "parity") != 0) {
        ds_usage(argc > 1 ? stderr : stdout);
        return 2;
    }
    const char *model = argv[2];
    uint32_t position = 7u, topk = 10u, requested_layers = 0u;
    uint64_t cache_mib = 256u;
    for (int i = 3; i < argc; ++i) {
        if (i + 1 >= argc) return 2;
        if (strcmp(argv[i], "--position") == 0) {
            if (!ds_parse_u32(argv[++i], &position)) return 2;
        } else if (strcmp(argv[i], "--top-k") == 0) {
            if (!ds_parse_u32(argv[++i], &topk)) return 2;
        } else if (strcmp(argv[i], "--cache-mib") == 0) {
            if (!ds_parse_u64(argv[++i], &cache_mib)) return 2;
        } else if (strcmp(argv[i], "--layers") == 0) {
            if (!ds_parse_u32(argv[++i], &requested_layers) || !requested_layers) return 2;
        } else {
            fprintf(stderr, "unknown option: %s\n", argv[i]);
            return 2;
        }
    }
    if (position >= 64u || !topk || topk > RL_DECODER_STACK_MAX_TOPK || !cache_mib) {
        fprintf(stderr, "decoder-stack arguments out of range\n");
        return 2;
    }

    float *input = (float *)calloc(RL_DECODER_STACK_HIDDEN, sizeof(float));
    float *cpu_output = (float *)calloc(RL_DECODER_STACK_HIDDEN, sizeof(float));
    float *gpu_output = (float *)calloc(RL_DECODER_STACK_HIDDEN, sizeof(float));
    rl_decoder_stack_stats *stats = (rl_decoder_stack_stats *)calloc(1, sizeof(*stats));
    if (!input || !cpu_output || !gpu_output || !stats) {
        fprintf(stderr, "decoder-stack allocation failed\n");
        free(input); free(cpu_output); free(gpu_output); free(stats);
        return 1;
    }
    ds_make_input(input, RL_DECODER_STACK_HIDDEN);

    char error[512] = {0};
    printf("runtime              : native Qwen3-Next audited decoder-stack parity\n");
    fflush(stdout);
    const int executed = rl_decoder_stack_parity_execute(model, position, topk, cache_mib, requested_layers,
        input, input, RL_DECODER_STACK_HIDDEN, cpu_output, gpu_output, RL_DECODER_STACK_HIDDEN,
        stats, error, sizeof(error));
    printf("GGUF layer map       : %u layers / recurrent=%u full-attention=%u\n",
        stats->model_layers, stats->model_recurrent, stats->model_full_attention);
    printf("execution            : layers=%u position=%u top-k=%u cache=%llu MiB\n",
        stats->requested_layers, position, topk, (unsigned long long)cache_mib);
    for (uint32_t layer = 0; layer < stats->executed_layers; ++layer) {
        const rl_decoder_stack_layer_stats *ls = &stats->layer[layer];
        printf("layer %02u %-14s out_abs=%-10.6g cache_abs=%-10.6g parity=%s\n",
            layer, rl_native_layer_map_kind_name(ls->kind), ls->output_max_abs,
            ls->block.cache_max_abs, ls->block.all_ok ? "YES" : "NO");
    }
    if (!executed) fprintf(stderr, "decoder stack failed: %s\n", error);
    printf("dispatch totals       : recurrent=%u full-attention=%u\n", stats->recurrent_executed, stats->full_attention_executed);
    printf("stack max abs/rel     : %.6g / %.6g\n", stats->max_abs, stats->max_rel);
    printf("elapsed               : %.3f ms\n", stats->elapsed_ms);
    const int full = stats->requested_layers == stats->model_layers;
    if (full)
        printf("COMPLETE 48-LAYER DECODER STACK: %s\n", stats->complete && stats->model_layers == 48u ? "YES" : "NO");
    else
        printf("PARTIAL DECODER STACK : %s (%u/%u layers)\n", executed && stats->all_ok ? "YES" : "NO",
            stats->executed_layers, stats->model_layers);
    for (uint32_t i = 0; i < 4u && executed && stats->all_ok; ++i)
        printf("row %-3u              : gpu=%+.7f cpu=%+.7f delta=%+.3e\n", i,
            gpu_output[i], cpu_output[i], (double)gpu_output[i] - cpu_output[i]);
    const int ok = executed && stats->all_ok && (!full || (stats->complete && stats->model_layers == 48u));
    free(input); free(cpu_output); free(gpu_output); free(stats);
    return ok ? 0 : 3;
}
