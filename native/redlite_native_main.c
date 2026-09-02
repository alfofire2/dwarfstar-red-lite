#include "redlite_native_cache.h"
#include "redlite_native_gguf.h"
#include "redlite_native_model.h"
#include "redlite_native_tables.h"

#ifdef __APPLE__
#include "redlite_native_metal.h"
#endif

#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MIB (1024ull * 1024ull)
#define GIB (1024ull * 1024ull * 1024ull)
#define NATIVE_TOPK_MAX 64u

static uint64_t round_up_u64(uint64_t value, uint64_t alignment) {
    if (!alignment) return value;
    const uint64_t rem = value % alignment;
    if (!rem) return value;
    return value + alignment - rem;
}

static void usage(FILE *out) {
    fprintf(out,
        "redlite-native 0.3.0.dev10\n"
        "Standalone native runtime for DwarfStar Red Lite.\n\n"
        "Usage:\n"
        "  redlite-native inspect MODEL [--cache-mib N] [--expert-count N] [--layers]\n"
        "  redlite-native topk-probe MODEL [--layer N] [--top-k N] [--rows N] [--cache-mib N]\n"
        "  redlite-native selftest\n\n"
        "inspect/selftest require no Python. On macOS, topk-probe additionally executes\n"
        "the resident IQ2_XS/IQ1_M top-k Metal path directly from the native C runtime.\n");
}

static int parse_u64(const char *s, uint64_t *out) {
    if (!s || !*s) return 0;
    errno = 0;
    char *end = NULL;
    unsigned long long value = strtoull(s, &end, 10);
    if (errno || !end || *end) return 0;
    *out = (uint64_t)value;
    return 1;
}

static int cache_selftest(void) {
    rl_native_lru cache;
    if (!rl_native_lru_init(&cache, 3)) {
        fprintf(stderr, "native LRU init failed\n");
        return 1;
    }
    char error[256];
    rl_cache_key first[] = {{0, 7}, {0, 54}, {0, 101}};
    uint32_t slots[3];
    if (!rl_native_lru_acquire_many(&cache, first, 3, slots, error, sizeof(error))) {
        fprintf(stderr, "initial native acquire_many failed: %s\n", error);
        rl_native_lru_free(&cache);
        return 1;
    }
    if (cache.misses != 3 || cache.hits != 0 || cache.resident != 3) {
        fprintf(stderr, "unexpected initial native LRU counters\n");
        rl_native_lru_free(&cache);
        return 1;
    }
    uint32_t protected_slot = slots[0];
    if (!rl_native_lru_set_inflight(&cache, protected_slot, 1)) {
        fprintf(stderr, "failed to mark native slot in-flight\n");
        rl_native_lru_free(&cache);
        return 1;
    }
    rl_cache_key second[] = {{0, 7}, {0, 148}};
    uint32_t second_slots[2];
    if (!rl_native_lru_acquire_many(&cache, second, 2, second_slots, error, sizeof(error))) {
        fprintf(stderr, "protected native acquire_many failed: %s\n", error);
        rl_native_lru_free(&cache);
        return 1;
    }
    if (second_slots[0] != protected_slot || cache.evictions != 1 || cache.hits != 1 || cache.misses != 4) {
        fprintf(stderr, "native top-k protection/LRU invariant failed\n");
        rl_native_lru_free(&cache);
        return 1;
    }
    uint32_t lookup = UINT32_MAX;
    if (!rl_native_lru_lookup(&cache, second[0], &lookup) || lookup != protected_slot) {
        fprintf(stderr, "selected in-flight native expert was incorrectly evicted\n");
        rl_native_lru_free(&cache);
        return 1;
    }
    rl_native_lru_set_inflight(&cache, protected_slot, 0);

    rl_native_lru too_small;
    if (!rl_native_lru_init(&too_small, 1)) {
        rl_native_lru_free(&cache);
        return 1;
    }
    if (rl_native_lru_acquire_many(&too_small, second, 2, second_slots, error, sizeof(error))) {
        fprintf(stderr, "native cache incorrectly accepted top-k larger than capacity\n");
        rl_native_lru_free(&too_small);
        rl_native_lru_free(&cache);
        return 1;
    }
    rl_native_lru_free(&too_small);

    rl_native_quant_tables tables;
    if (!rl_native_quant_tables_init(&tables, error, sizeof(error))) {
        fprintf(stderr, "native quant table expansion failed: %s\n", error);
        rl_native_lru_free(&cache);
        return 1;
    }
    const uint64_t iq2_hash = rl_native_quant_table_fnv1a(tables.iq2_xs, RL_IQ2_XS_GRID_COUNT);
    const uint64_t iq1_hash = rl_native_quant_table_fnv1a(tables.iq1_m, RL_IQ1_M_GRID_COUNT);
    rl_native_quant_tables_free(&tables);

    printf("native selftest     : OK\n");
    printf("LRU top-k protect  : OK\n");
    printf("in-flight protect  : OK\n");
    printf("hard capacity      : OK\n");
    printf("quant tables       : OK\n");
    printf("IQ2_XS grid FNV    : 0x%016" PRIx64 "\n", iq2_hash);
    printf("IQ1_M grid FNV     : 0x%016" PRIx64 "\n", iq1_hash);
    printf("python dependency  : NONE\n");
    rl_native_lru_free(&cache);
    return 0;
}

static int inspect_model(int argc, char **argv) {
    if (argc < 3) {
        usage(stderr);
        return 2;
    }
    const char *model = argv[2];
    uint64_t cache_mib = 256;
    uint64_t expert_count64 = 512;
    int show_layers = 0;
    for (int i = 3; i < argc; ++i) {
        if (strcmp(argv[i], "--layers") == 0) {
            show_layers = 1;
        } else if (strcmp(argv[i], "--cache-mib") == 0 && i + 1 < argc) {
            if (!parse_u64(argv[++i], &cache_mib) || cache_mib == 0) {
                fprintf(stderr, "invalid --cache-mib\n");
                return 2;
            }
        } else if (strcmp(argv[i], "--expert-count") == 0 && i + 1 < argc) {
            if (!parse_u64(argv[++i], &expert_count64) || expert_count64 == 0 || expert_count64 > UINT32_MAX) {
                fprintf(stderr, "invalid --expert-count\n");
                return 2;
            }
        } else {
            fprintf(stderr, "unknown inspect option: %s\n", argv[i]);
            return 2;
        }
    }
    if (cache_mib > UINT64_MAX / MIB) {
        fprintf(stderr, "cache size overflow\n");
        return 2;
    }

    rl_expert_map map;
    char error[512];
    if (!rl_native_build_expert_map(model, (uint32_t)expert_count64, &map, error, sizeof(error))) {
        fprintf(stderr, "native GGUF parse failed: %s\n", error);
        return 1;
    }

    uint32_t iq2 = 0, iq1 = 0, other = 0;
    uint32_t layer_type[RL_NATIVE_MAX_LAYERS][3];
    int layer_present[RL_NATIVE_MAX_LAYERS] = {0};
    for (uint32_t l = 0; l < RL_NATIVE_MAX_LAYERS; ++l)
        for (uint32_t k = 0; k < 3; ++k) layer_type[l][k] = UINT32_MAX;

    for (uint32_t i = 0; i < map.routed_tensor_count; ++i) {
        const rl_expert_tensor *t = &map.routed[i];
        if (t->ggml_type == 17u) iq2++;
        else if (t->ggml_type == 29u) iq1++;
        else other++;
        layer_present[t->layer] = 1;
        layer_type[t->layer][(uint32_t)t->kind] = t->ggml_type;
    }

    uint32_t iq2_layers = 0, iq1_layers = 0, mixed_layers = 0;
    for (uint32_t l = 0; l < RL_NATIVE_MAX_LAYERS; ++l) {
        if (!layer_present[l]) continue;
        const uint32_t g = layer_type[l][RL_EXPERT_GATE];
        const uint32_t u = layer_type[l][RL_EXPERT_UP];
        const uint32_t d = layer_type[l][RL_EXPERT_DOWN];
        if (g == 17u && u == 17u && d == 17u) iq2_layers++;
        else if (g == 29u && u == 29u && d == 29u) iq1_layers++;
        else mixed_layers++;
    }

    const uint64_t aligned_slot = round_up_u64(map.max_expert_triplet_bytes, 4096u);
    const uint64_t cache_bytes = cache_mib * MIB;
    const uint64_t capacity = aligned_slot ? cache_bytes / aligned_slot : 0;

    printf("runtime            : native C (no Python)\n");
    printf("GGUF version       : %u\n", map.version);
    printf("alignment          : %u\n", map.alignment);
    printf("tensor count       : %" PRIu64 "\n", map.tensor_count);
    printf("metadata entries   : %" PRIu64 "\n", map.kv_count);
    printf("routed tensors     : %u\n", map.routed_tensor_count);
    printf("routed layers      : %u\n", map.layer_count);
    printf("expert count       : %u\n", map.expert_count);
    printf("slice safe         : %s\n", map.all_slice_safe ? "YES" : "NO");
    printf("routed payload     : %.3f GiB\n", (double)map.total_routed_payload_bytes / (double)GIB);
    printf("quant tensors      : IQ2_XS=%u IQ1_M=%u OTHER=%u\n", iq2, iq1, other);
    printf("layer patterns     : IQ2_XS=%u IQ1_M=%u mixed=%u\n", iq2_layers, iq1_layers, mixed_layers);
    printf("max expert triplet : %.3f MiB\n", (double)map.max_expert_triplet_bytes / (double)MIB);
    printf("aligned slot       : %.3f MiB\n", (double)aligned_slot / (double)MIB);
    printf("cache budget       : %" PRIu64 " MiB\n", cache_mib);
    printf("slot capacity      : %" PRIu64 "\n", capacity);
    printf("top-10 fits        : %s\n", capacity >= 10 ? "YES" : "NO");

    if (show_layers) {
        printf("per layer          :\n");
        for (uint32_t l = 0; l < RL_NATIVE_MAX_LAYERS; ++l) {
            if (!layer_present[l]) continue;
            printf("  %2u: gate=%s up=%s down=%s\n", l,
                rl_native_quant_name(layer_type[l][RL_EXPERT_GATE]),
                rl_native_quant_name(layer_type[l][RL_EXPERT_UP]),
                rl_native_quant_name(layer_type[l][RL_EXPERT_DOWN]));
        }
    }

    if (map.all_slice_safe && map.layer_count) {
        uint32_t first_layer = 0;
        while (first_layer < RL_NATIVE_MAX_LAYERS && !layer_present[first_layer]) first_layer++;
        rl_expert_layout layout;
        rl_native_layer_info info;
        if (!rl_native_expert_layout(&map, first_layer, 0, &layout, error, sizeof(error)) ||
            !rl_native_get_layer_info(&map, first_layer, &info, error, sizeof(error))) {
            fprintf(stderr, "native expert layout verification failed: %s\n", error);
            rl_native_free_expert_map(&map);
            return 1;
        }
        printf("layout probe       : layer=%u expert=0 type=%s triplet=%.3f MiB hidden=%u ffn=%u\n",
            first_layer, rl_native_quant_name(layout.ggml_type), (double)layout.total_bytes / (double)MIB,
            info.hidden_size, info.ffn_size);
    }

    rl_native_free_expert_map(&map);
    return 0;
}

#ifdef __APPLE__
static void default_experts(uint32_t *out, uint32_t top_k, uint32_t expert_count) {
    uint32_t candidate = 7u % expert_count;
    const uint32_t step = 47u;
    uint32_t count = 0;
    while (count < top_k) {
        int duplicate = 0;
        for (uint32_t i = 0; i < count; ++i) duplicate |= out[i] == candidate;
        if (!duplicate) out[count++] = candidate;
        candidate = (candidate + step) % expert_count;
    }
}

static void default_weights(float *out, uint32_t top_k) {
    float total = 0.0f;
    for (uint32_t i = 0; i < top_k; ++i) total += (float)(top_k - i);
    for (uint32_t i = 0; i < top_k; ++i) out[i] = (float)(top_k - i) / total;
}

static int topk_probe(int argc, char **argv) {
    if (argc < 3) {
        usage(stderr);
        return 2;
    }
    const char *model = argv[2];
    uint64_t layer64 = 0, top_k64 = 10, rows64 = 8, cache_mib = 256;
    for (int i = 3; i < argc; ++i) {
        uint64_t *target = NULL;
        if (strcmp(argv[i], "--layer") == 0) target = &layer64;
        else if (strcmp(argv[i], "--top-k") == 0) target = &top_k64;
        else if (strcmp(argv[i], "--rows") == 0) target = &rows64;
        else if (strcmp(argv[i], "--cache-mib") == 0) target = &cache_mib;
        else {
            fprintf(stderr, "unknown topk-probe option: %s\n", argv[i]);
            return 2;
        }
        if (i + 1 >= argc || !parse_u64(argv[++i], target)) {
            fprintf(stderr, "invalid value for topk-probe option\n");
            return 2;
        }
    }
    if (layer64 >= RL_NATIVE_MAX_LAYERS || !top_k64 || top_k64 > NATIVE_TOPK_MAX ||
        !rows64 || rows64 > UINT32_MAX || !cache_mib || cache_mib > UINT64_MAX / MIB) {
        fprintf(stderr, "topk-probe arguments out of range\n");
        return 2;
    }

    char error[512];
    rl_expert_map map;
    if (!rl_native_build_expert_map(model, 512u, &map, error, sizeof(error))) {
        fprintf(stderr, "native GGUF parse failed: %s\n", error);
        return 1;
    }
    rl_native_layer_info info;
    if (!rl_native_get_layer_info(&map, (uint32_t)layer64, &info, error, sizeof(error))) {
        fprintf(stderr, "native layer info failed: %s\n", error);
        rl_native_free_expert_map(&map);
        return 1;
    }
    if (top_k64 > info.expert_count || rows64 > info.hidden_size) {
        fprintf(stderr, "topk-probe top-k/rows exceed model dimensions\n");
        rl_native_free_expert_map(&map);
        return 2;
    }

    uint32_t experts[NATIVE_TOPK_MAX];
    float weights[NATIVE_TOPK_MAX];
    default_experts(experts, (uint32_t)top_k64, info.expert_count);
    default_weights(weights, (uint32_t)top_k64);

    float *input = (float *)malloc((size_t)info.hidden_size * sizeof(float));
    float *output = (float *)calloc((size_t)rows64, sizeof(float));
    if (!input || !output) {
        fprintf(stderr, "out of memory for native top-k activations\n");
        free(input); free(output); rl_native_free_expert_map(&map);
        return 1;
    }
    for (uint32_t i = 0; i < info.hidden_size; ++i)
        input[i] = (float)(((int)((i * 17u + 3u) % 61u) - 30) / 32.0);

    rl_native_metal_runtime *runtime = rl_native_metal_create(
        model, &map, cache_mib * MIB, 64u, error, sizeof(error));
    if (!runtime) {
        fprintf(stderr, "native Metal runtime creation failed: %s\n", error);
        free(input); free(output); rl_native_free_expert_map(&map);
        return 1;
    }

    rl_native_metal_telemetry telemetry;
    const int ok = rl_native_metal_execute_topk(
        runtime, &map, (uint32_t)layer64, experts, weights, (uint32_t)top_k64,
        0u, (uint32_t)rows64, input, info.hidden_size, output, (uint32_t)rows64,
        &telemetry, error, sizeof(error));
    if (!ok) {
        fprintf(stderr, "native top-k execution failed: %s\n", error);
        rl_native_metal_destroy(runtime);
        free(input); free(output); rl_native_free_expert_map(&map);
        return 1;
    }

    float weight_sum = 0.0f;
    for (uint32_t i = 0; i < (uint32_t)top_k64; ++i) weight_sum += weights[i];
    printf("runtime            : native C + Objective-C/Metal (no Python/ctypes)\n");
    printf("quant type         : %s (%u)\n", rl_native_quant_name(info.ggml_type), info.ggml_type);
    printf("layer / top-k      : %u / %u\n", info.layer, (uint32_t)top_k64);
    printf("hidden / ffn       : %u / %u\n", info.hidden_size, info.ffn_size);
    printf("experts            : ");
    for (uint32_t i = 0; i < (uint32_t)top_k64; ++i) printf("%s%u", i ? "," : "", experts[i]);
    printf("\nrouter weights     : ");
    for (uint32_t i = 0; i < (uint32_t)top_k64; ++i) printf("%s%.6f", i ? "," : "", weights[i]);
    printf("\nweight sum         : %.7f\n", weight_sum);
    printf("resident slots     : %u/%u\n", telemetry.resident_slots, telemetry.slot_capacity);
    printf("Metal slabs        : %u\n", telemetry.slab_count);
    printf("Metal allocated    : %.3f GiB\n", (double)telemetry.allocated_bytes / (double)GIB);
    printf("expert loads       : %" PRIu64 "\n", telemetry.expert_loads);
    printf("cache hits/misses  : %" PRIu64 "/%" PRIu64 "\n", telemetry.cache_hits, telemetry.cache_misses);
    printf("evictions          : %" PRIu64 "\n", telemetry.cache_evictions);
    printf("SSD read           : %.3f MiB\n", (double)telemetry.bytes_read_total / (double)MIB);
    printf("pread calls        : %" PRIu64 "\n", telemetry.read_calls_total);
    printf("SSD during top-k   : %" PRIu64 " bytes / %" PRIu64 " calls\n",
        telemetry.bytes_read_during_execute, telemetry.read_calls_during_execute);
    printf("GPU top-k layer    : %.3f ms\n", telemetry.gpu_ms);
    for (uint32_t row = 0; row < (uint32_t)rows64; ++row)
        printf("row %4u           : %+0.7f\n", row, output[row]);
    printf("note               : This path parses GGUF, plans LRU residency, loads expert misses, and submits Metal top-k execution without Python or ctypes.\n");

    rl_native_metal_destroy(runtime);
    free(input);
    free(output);
    rl_native_free_expert_map(&map);
    return 0;
}
#endif

int main(int argc, char **argv) {
    if (argc < 2 || strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0) {
        usage(stdout);
        return argc < 2 ? 2 : 0;
    }
    if (strcmp(argv[1], "selftest") == 0) return cache_selftest();
    if (strcmp(argv[1], "inspect") == 0) return inspect_model(argc, argv);
    if (strcmp(argv[1], "topk-probe") == 0) {
#ifdef __APPLE__
        return topk_probe(argc, argv);
#else
        fprintf(stderr, "topk-probe requires macOS Metal\n");
        return 2;
#endif
    }
    fprintf(stderr, "unknown command: %s\n", argv[1]);
    usage(stderr);
    return 2;
}
