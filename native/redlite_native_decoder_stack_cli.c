#define _POSIX_C_SOURCE 200809L

#include "redlite_native_decoder_block.h"
#include "redlite_native_layer_map.h"

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define DS_HIDDEN 2048u
#define DS_MAX_TOPK 64u

static double ds_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

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

static void ds_compare(const float *gpu, const float *cpu, uint32_t count,
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
    uint32_t position = 7u, topk = 10u, requested_layers = UINT32_MAX;
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
            if (!ds_parse_u32(argv[++i], &requested_layers)) return 2;
        } else {
            fprintf(stderr, "unknown option: %s\n", argv[i]);
            return 2;
        }
    }
    if (position >= 64u || !topk || topk > DS_MAX_TOPK || !cache_mib || requested_layers == 0u) {
        fprintf(stderr, "decoder-stack arguments out of range\n");
        return 2;
    }
    char error[512] = {0};
    rl_native_layer_map map;
    if (!rl_native_layer_map_audit(model, &map, error, sizeof(error))) {
        fprintf(stderr, "decoder layer-map audit failed: %s\n", error);
        return 1;
    }
    const uint32_t layers = requested_layers == UINT32_MAX ? map.layer_count : requested_layers;
    if (layers > map.layer_count) {
        fprintf(stderr, "requested %u layers but GGUF contains %u\n", layers, map.layer_count);
        return 2;
    }

    float *cpu_current = (float *)calloc(DS_HIDDEN, sizeof(float));
    float *gpu_current = (float *)calloc(DS_HIDDEN, sizeof(float));
    float *cpu_next = (float *)calloc(DS_HIDDEN, sizeof(float));
    float *gpu_next = (float *)calloc(DS_HIDDEN, sizeof(float));
    if (!cpu_current || !gpu_current || !cpu_next || !gpu_next) {
        fprintf(stderr, "decoder-stack allocation failed\n");
        free(cpu_current); free(gpu_current); free(cpu_next); free(gpu_next);
        return 1;
    }
    ds_make_input(cpu_current, DS_HIDDEN);
    memcpy(gpu_current, cpu_current, (size_t)DS_HIDDEN * sizeof(float));
    uint32_t recurrent_executed = 0, full_executed = 0;
    float stack_max_abs = 0.0f, stack_max_rel = 0.0f;
    int stack_ok = 1;
    const double start = ds_now_ms();
    printf("runtime              : native Qwen3-Next audited decoder-stack parity\n");
    printf("GGUF layer map       : %u layers / recurrent=%u full-attention=%u\n",
        map.layer_count, map.recurrent_count, map.full_attention_count);
    printf("execution            : layers=%u position=%u top-k=%u cache=%llu MiB\n",
        layers, position, topk, (unsigned long long)cache_mib);
    fflush(stdout);

    for (uint32_t layer = 0; layer < layers; ++layer) {
        rl_decoder_block_stats stats;
        int executed = 0;
        if (map.kind[layer] == RL_LAYER_MAP_RECURRENT) {
            executed = rl_recurrent_block_parity_execute(model, layer, topk, cache_mib,
                cpu_current, gpu_current, DS_HIDDEN, cpu_next, gpu_next, DS_HIDDEN,
                &stats, error, sizeof(error));
            recurrent_executed++;
        } else if (map.kind[layer] == RL_LAYER_MAP_FULL_ATTENTION) {
            executed = rl_full_attention_block_parity_execute(model, layer, position, topk, cache_mib,
                cpu_current, gpu_current, DS_HIDDEN, cpu_next, gpu_next, DS_HIDDEN,
                &stats, error, sizeof(error));
            full_executed++;
        }
        if (!executed) {
            fprintf(stderr, "layer %u (%s) execution failed: %s\n", layer,
                rl_native_layer_map_kind_name(map.kind[layer]), error);
            stack_ok = 0;
            break;
        }
        float layer_abs = 0.0f, layer_rel = 0.0f;
        ds_compare(gpu_next, cpu_next, DS_HIDDEN, &layer_abs, &layer_rel);
        if (layer_abs > stack_max_abs) stack_max_abs = layer_abs;
        if (layer_rel > stack_max_rel) stack_max_rel = layer_rel;
        printf("layer %02u %-14s out_abs=%-10.6g cache_abs=%-10.6g parity=%s\n",
            layer, rl_native_layer_map_kind_name(map.kind[layer]), layer_abs,
            stats.cache_max_abs, stats.all_ok ? "YES" : "NO");
        fflush(stdout);
        if (!stats.all_ok) { stack_ok = 0; break; }
        float *swap = cpu_current; cpu_current = cpu_next; cpu_next = swap;
        swap = gpu_current; gpu_current = gpu_next; gpu_next = swap;
    }
    const double elapsed = ds_now_ms() - start;
    const int complete = stack_ok && layers == map.layer_count && map.layer_count == 48u &&
        recurrent_executed == map.recurrent_count && full_executed == map.full_attention_count;
    printf("dispatch totals       : recurrent=%u full-attention=%u\n", recurrent_executed, full_executed);
    printf("stack max abs/rel     : %.6g / %.6g\n", stack_max_abs, stack_max_rel);
    printf("elapsed               : %.3f ms\n", elapsed);
    if (layers == map.layer_count)
        printf("COMPLETE 48-LAYER DECODER STACK: %s\n", complete ? "YES" : "NO");
    else
        printf("PARTIAL DECODER STACK : %s (%u/%u layers)\n", stack_ok ? "YES" : "NO", layers, map.layer_count);
    for (uint32_t i = 0; i < 4u && stack_ok; ++i)
        printf("row %-3u              : gpu=%+.7f cpu=%+.7f delta=%+.3e\n", i,
            gpu_current[i], cpu_current[i], (double)gpu_current[i] - cpu_current[i]);
    free(cpu_current); free(gpu_current); free(cpu_next); free(gpu_next);
    return (layers == map.layer_count ? complete : stack_ok) ? 0 : 3;
}
