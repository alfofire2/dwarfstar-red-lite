#define _FILE_OFFSET_BITS 64
#define _POSIX_C_SOURCE 200809L

#include "redlite_native_gguf.h"
#include "redlite_native_metal.h"
#include "redlite_native_reference.h"
#include "redlite_native_router.h"
#include "redlite_native_router_exec.h"

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MIB (1024ull * 1024ull)
#define MAX_EXPERTS 512u
#define MAX_TOPK 64u

static void usage(FILE *out) {
    fprintf(out,
        "redlite-router 0.3.0.dev13\n\n"
        "Usage:\n"
        "  redlite-router router-parity MODEL [--layer N] [--top-k N]\n"
        "  redlite-router routed-parity MODEL [--layer N] [--top-k N] [--rows N] [--cache-mib N]\n\n"
        "router-parity compares native CPU and Metal F32 router logits/top-k/weights.\n"
        "routed-parity additionally feeds the real router selection into the resident Metal routed-expert executor.\n"
        "The routed parity intentionally excludes Qwen3-Next's separate shared-expert branch.\n");
}

static int parse_u64(const char *s, uint64_t *out) {
    if (!s || !*s) return 0;
    errno = 0;
    char *end = NULL;
    const unsigned long long value = strtoull(s, &end, 10);
    if (errno || !end || *end) return 0;
    *out = (uint64_t)value;
    return 1;
}

static const rl_router_tensor_info *find_router(
        const rl_router_tensor_info *routers, uint32_t count, uint32_t layer) {
    for (uint32_t i = 0; i < count; ++i) if (routers[i].layer == layer) return &routers[i];
    return NULL;
}

static void make_input(float *input, uint32_t hidden) {
    for (uint32_t i = 0; i < hidden; ++i)
        input[i] = (float)(((int)((i * 17u + 3u) % 61u) - 30) / 32.0);
}

static void print_selection(const char *label, const uint32_t *ids, const float *weights, uint32_t top_k) {
    printf("%s ids       : ", label);
    for (uint32_t i = 0; i < top_k; ++i) printf("%s%u", i ? "," : "", ids[i]);
    printf("\n%s weights   : ", label);
    for (uint32_t i = 0; i < top_k; ++i) printf("%s%.7f", i ? "," : "", weights[i]);
    printf("\n");
}

int main(int argc, char **argv) {
    if (argc < 2 || strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0) {
        usage(stdout);
        return argc < 2 ? 2 : 0;
    }
    const int do_routed = strcmp(argv[1], "routed-parity") == 0;
    if (!do_routed && strcmp(argv[1], "router-parity") != 0) {
        usage(stderr);
        return 2;
    }
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
        else if (strcmp(argv[i], "--rows") == 0 && do_routed) target = &rows64;
        else if (strcmp(argv[i], "--cache-mib") == 0 && do_routed) target = &cache_mib;
        else {
            fprintf(stderr, "unknown option: %s\n", argv[i]);
            return 2;
        }
        if (i + 1 >= argc || !parse_u64(argv[++i], target)) {
            fprintf(stderr, "invalid option value\n");
            return 2;
        }
    }
    if (layer64 >= RL_ROUTER_MAX_LAYERS || !top_k64 || top_k64 > MAX_TOPK ||
        !rows64 || rows64 > UINT32_MAX || !cache_mib) {
        fprintf(stderr, "router parity arguments out of range\n");
        return 2;
    }

    char error[512] = {0};
    rl_router_tensor_info routers[RL_ROUTER_MAX_LAYERS];
    uint32_t router_count = 0;
    if (!rl_native_router_audit(model, routers, RL_ROUTER_MAX_LAYERS, &router_count, error, sizeof(error))) {
        fprintf(stderr, "router audit failed: %s\n", error);
        return 1;
    }
    const rl_router_tensor_info *router = find_router(routers, router_count, (uint32_t)layer64);
    if (!router || router->ggml_type != 0u || router->n_dims != 2u ||
        router->shape[0] > UINT32_MAX || router->shape[1] > UINT32_MAX) {
        fprintf(stderr, "requested layer does not contain the expected F32 rank-2 router\n");
        return 1;
    }
    const uint32_t hidden = (uint32_t)router->shape[0];
    const uint32_t experts = (uint32_t)router->shape[1];
    if (experts > MAX_EXPERTS || top_k64 > experts) {
        fprintf(stderr, "router expert count/top-k exceeds native parity limits\n");
        return 2;
    }

    float *input = (float *)malloc((size_t)hidden * sizeof(float));
    float *cpu_logits = (float *)malloc((size_t)experts * sizeof(float));
    float *gpu_logits = (float *)malloc((size_t)experts * sizeof(float));
    float *cpu_probs = (float *)malloc((size_t)experts * sizeof(float));
    float *gpu_probs = (float *)malloc((size_t)experts * sizeof(float));
    if (!input || !cpu_logits || !gpu_logits || !cpu_probs || !gpu_probs) {
        fprintf(stderr, "out of memory for router parity buffers\n");
        free(input); free(cpu_logits); free(gpu_logits); free(cpu_probs); free(gpu_probs);
        return 1;
    }
    make_input(input, hidden);

    rl_native_router_telemetry cpu_tel = {0}, gpu_tel = {0};
    if (!rl_native_router_cpu_f32(model, router, input, hidden, cpu_logits, experts, &cpu_tel, error, sizeof(error))) {
        fprintf(stderr, "CPU router failed: %s\n", error);
        free(input); free(cpu_logits); free(gpu_logits); free(cpu_probs); free(gpu_probs);
        return 1;
    }
#ifdef __APPLE__
    if (!rl_native_router_gpu_f32(model, router, input, hidden, gpu_logits, experts, &gpu_tel, error, sizeof(error))) {
        fprintf(stderr, "Metal router failed: %s\n", error);
        free(input); free(cpu_logits); free(gpu_logits); free(cpu_probs); free(gpu_probs);
        return 1;
    }
#else
    fprintf(stderr, "router Metal parity requires macOS\n");
    free(input); free(cpu_logits); free(gpu_logits); free(cpu_probs); free(gpu_probs);
    return 2;
#endif

    uint32_t cpu_ids[MAX_TOPK], gpu_ids[MAX_TOPK];
    float cpu_weights[MAX_TOPK], gpu_weights[MAX_TOPK];
    if (!rl_native_router_select_softmax_topk(cpu_logits, experts, (uint32_t)top_k64,
            cpu_ids, cpu_weights, cpu_probs, error, sizeof(error)) ||
        !rl_native_router_select_softmax_topk(gpu_logits, experts, (uint32_t)top_k64,
            gpu_ids, gpu_weights, gpu_probs, error, sizeof(error))) {
        fprintf(stderr, "router selection failed: %s\n", error);
        free(input); free(cpu_logits); free(gpu_logits); free(cpu_probs); free(gpu_probs);
        return 1;
    }

    double max_logit_abs = 0.0, max_logit_rel = 0.0, max_weight_abs = 0.0;
    uint32_t max_logit_index = 0;
    int ids_match = 1;
    for (uint32_t i = 0; i < experts; ++i) {
        const double abs_err = fabs((double)gpu_logits[i] - cpu_logits[i]);
        const double rel_err = abs_err / fmax(fabs((double)cpu_logits[i]), 1e-12);
        if (abs_err > max_logit_abs) { max_logit_abs = abs_err; max_logit_index = i; }
        if (rel_err > max_logit_rel) max_logit_rel = rel_err;
    }
    for (uint32_t k = 0; k < (uint32_t)top_k64; ++k) {
        if (cpu_ids[k] != gpu_ids[k]) ids_match = 0;
        const double err = fabs((double)gpu_weights[k] - cpu_weights[k]);
        if (err > max_weight_abs) max_weight_abs = err;
    }
    const int router_match = ids_match && max_logit_abs <= 1e-3 && max_weight_abs <= 1e-5;

    printf("runtime            : native C + Metal router (no Python/ctypes)\n");
    printf("router tensor      : blk.%u.ffn_gate_inp.weight\n", (uint32_t)layer64);
    printf("router type        : F32 (0)\n");
    printf("shape              : (%u,%u)\n", hidden, experts);
    printf("top-k              : %u\n", (uint32_t)top_k64);
    printf("CPU router read    : %.3f ms\n", cpu_tel.read_ms);
    printf("CPU router matvec  : %.3f ms\n", cpu_tel.compute_ms);
    printf("GPU router read    : %.3f ms\n", gpu_tel.read_ms);
    printf("GPU router matvec  : %.3f ms\n", gpu_tel.compute_ms);
    printf("max logit abs err  : %.6g @ expert %u\n", max_logit_abs, max_logit_index);
    printf("max logit rel err  : %.6g\n", max_logit_rel);
    printf("top-k ids match    : %s\n", ids_match ? "YES" : "NO");
    printf("max weight abs err : %.6g\n", max_weight_abs);
    printf("router parity      : %s\n", router_match ? "YES" : "NO");
    print_selection("CPU", cpu_ids, cpu_weights, (uint32_t)top_k64);
    print_selection("GPU", gpu_ids, gpu_weights, (uint32_t)top_k64);

    int routed_match = 1;
    if (do_routed) {
        if (!router_match) {
            fprintf(stderr, "refusing routed parity because router parity failed\n");
            routed_match = 0;
        } else {
            rl_expert_map map;
            if (!rl_native_build_expert_map(model, experts, &map, error, sizeof(error))) {
                fprintf(stderr, "expert map failed: %s\n", error);
                routed_match = 0;
            } else {
                if (rows64 > hidden) {
                    fprintf(stderr, "--rows exceeds hidden size\n");
                    routed_match = 0;
                } else {
                    float *gpu_out = (float *)calloc((size_t)rows64, sizeof(float));
                    double *cpu_out = (double *)calloc((size_t)rows64, sizeof(double));
                    rl_native_metal_runtime *runtime = NULL;
                    if (!gpu_out || !cpu_out) {
                        fprintf(stderr, "out of memory for routed parity outputs\n");
                        routed_match = 0;
                    } else {
                        runtime = rl_native_metal_create(model, &map, cache_mib * MIB, 64u, error, sizeof(error));
                        rl_native_metal_telemetry tel = {0};
                        double cpu_ms = 0.0;
                        if (!runtime || !rl_native_metal_execute_topk(runtime, &map, (uint32_t)layer64,
                                gpu_ids, gpu_weights, (uint32_t)top_k64, 0u, (uint32_t)rows64,
                                input, hidden, gpu_out, (uint32_t)rows64, &tel, error, sizeof(error))) {
                            fprintf(stderr, "real-router Metal top-k failed: %s\n", error);
                            routed_match = 0;
                        } else if (!rl_native_reference_topk(model, &map, (uint32_t)layer64,
                                cpu_ids, cpu_weights, (uint32_t)top_k64, 0u, (uint32_t)rows64,
                                input, hidden, cpu_out, (uint32_t)rows64, &cpu_ms, error, sizeof(error))) {
                            fprintf(stderr, "real-router CPU routed reference failed: %s\n", error);
                            routed_match = 0;
                        } else {
                            double max_abs = 0.0, max_rel = 0.0;
                            for (uint32_t r = 0; r < (uint32_t)rows64; ++r) {
                                const double err = fabs((double)gpu_out[r] - cpu_out[r]);
                                const double rel = err / fmax(fabs(cpu_out[r]), 1e-12);
                                if (err > max_abs) max_abs = err;
                                if (rel > max_rel) max_rel = rel;
                                if (err > 1e-3 + 1e-4 * fabs(cpu_out[r])) routed_match = 0;
                            }
                            printf("expert loads       : %" PRIu64 "\n", tel.expert_loads);
                            printf("cache hits/misses  : %" PRIu64 "/%" PRIu64 "\n", tel.cache_hits, tel.cache_misses);
                            printf("pread calls        : %" PRIu64 "\n", tel.read_calls_total);
                            printf("SSD during top-k   : %" PRIu64 " bytes / %" PRIu64 " calls\n",
                                tel.bytes_read_during_execute, tel.read_calls_during_execute);
                            printf("GPU routed top-k   : %.3f ms\n", tel.gpu_ms);
                            printf("CPU routed ref     : %.3f ms\n", cpu_ms);
                            printf("routed max abs err : %.6g\n", max_abs);
                            printf("routed max rel err : %.6g\n", max_rel);
                            printf("routed parity      : %s\n", routed_match ? "YES" : "NO");
                            for (uint32_t r = 0; r < (uint32_t)rows64; ++r)
                                printf("row %4u           : gpu=%+0.7f cpu=%+0.7f delta=%+.3e\n",
                                    r, gpu_out[r], cpu_out[r], (double)gpu_out[r] - cpu_out[r]);
                        }
                        if (runtime) rl_native_metal_destroy(runtime);
                    }
                    free(gpu_out); free(cpu_out);
                }
                rl_native_free_expert_map(&map);
            }
        }
        printf("note               : routed parity covers the real router-selected routed-expert branch only; shared expert is not integrated yet.\n");
    }

    free(input); free(cpu_logits); free(gpu_logits); free(cpu_probs); free(gpu_probs);
    if (!router_match) return 2;
    return do_routed && !routed_match ? 2 : 0;
}
