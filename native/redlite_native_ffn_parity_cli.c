#define _FILE_OFFSET_BITS 64
#define _POSIX_C_SOURCE 200809L

#include "redlite_native_gguf.h"
#include "redlite_native_metal.h"
#include "redlite_native_reference.h"
#include "redlite_native_router.h"
#include "redlite_native_router_exec.h"
#include "redlite_native_shared.h"
#include "redlite_native_shared_exec.h"

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MIB (1024ull * 1024ull)
#define MAX_EXPERTS 512u
#define MAX_TOPK 64u
#define MAX_SHARED_TENSORS (RL_SHARED_MAX_LAYERS * 4u)

static void usage(FILE *out) {
    fprintf(out,
        "redlite-ffn 0.3.0.dev14\n\n"
        "Usage:\n"
        "  redlite-ffn parity MODEL [--layer N] [--top-k N] [--rows N] [--cache-mib N]\n\n"
        "Validates the complete Qwen3-Next FFN: real F32 router + routed top-k MoE + gated shared expert.\n"
        "CPU and Metal paths are evaluated independently and summed only at the final FFN output.\n");
}

static int parse_u64(const char *s, uint64_t *out) {
    if (!s || !*s) return 0;
    errno = 0;
    char *end = NULL;
    const unsigned long long v = strtoull(s, &end, 10);
    if (errno || !end || *end) return 0;
    *out = (uint64_t)v;
    return 1;
}

static void make_input(float *x, uint32_t n) {
    for (uint32_t i = 0; i < n; ++i)
        x[i] = (float)(((int)((i * 17u + 3u) % 61u) - 30) / 32.0);
}

static const rl_router_tensor_info *find_router(
        const rl_router_tensor_info *routers, uint32_t count, uint32_t layer) {
    for (uint32_t i = 0; i < count; ++i)
        if (routers[i].layer == layer) return &routers[i];
    return NULL;
}

static int collect_shared(
        const rl_shared_tensor_info *all, uint32_t count, uint32_t layer,
        rl_shared_tensor_info out[4]) {
    uint8_t seen[4] = {0};
    memset(out, 0, 4u * sizeof(*out));
    for (uint32_t i = 0; i < count; ++i) {
        if (all[i].layer != layer || all[i].kind > RL_SHARED_DOWN) continue;
        const uint32_t k = (uint32_t)all[i].kind;
        if (seen[k]) return 0;
        out[k] = all[i];
        seen[k] = 1;
    }
    return seen[0] && seen[1] && seen[2] && seen[3];
}

static const char *shared_type_name(uint32_t t) {
    if (t == 0u) return "F32";
    if (t == 14u) return "Q6_K";
    if (t == 16u) return "IQ2_XXS";
    return "OTHER";
}

static int rows_match_float_double(
        const float *gpu, const double *cpu, uint32_t n,
        double *max_abs, double *max_rel) {
    int ok = 1;
    *max_abs = 0.0;
    *max_rel = 0.0;
    for (uint32_t i = 0; i < n; ++i) {
        const double ae = fabs((double)gpu[i] - cpu[i]);
        const double re = ae / fmax(fabs(cpu[i]), 1e-12);
        if (ae > *max_abs) *max_abs = ae;
        if (re > *max_rel) *max_rel = re;
        if (ae > 1e-3 + 1e-4 * fabs(cpu[i])) ok = 0;
    }
    return ok;
}

int main(int argc, char **argv) {
    if (argc < 2 || strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0) {
        usage(stdout);
        return argc < 2 ? 2 : 0;
    }
    if (strcmp(argv[1], "parity") != 0 || argc < 3) {
        usage(stderr);
        return 2;
    }

    const char *model = argv[2];
    uint64_t layer64 = 0, top_k64 = 10, rows64 = 8, cache_mib64 = 256;
    for (int i = 3; i < argc; ++i) {
        uint64_t *target = NULL;
        if (strcmp(argv[i], "--layer") == 0) target = &layer64;
        else if (strcmp(argv[i], "--top-k") == 0) target = &top_k64;
        else if (strcmp(argv[i], "--rows") == 0) target = &rows64;
        else if (strcmp(argv[i], "--cache-mib") == 0) target = &cache_mib64;
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
        !rows64 || rows64 > UINT32_MAX || !cache_mib64) {
        fprintf(stderr, "FFN parity arguments out of range\n");
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
        fprintf(stderr, "requested layer does not contain the expected F32 router\n");
        return 1;
    }
    const uint32_t hidden = (uint32_t)router->shape[0];
    const uint32_t experts = (uint32_t)router->shape[1];
    if (experts > MAX_EXPERTS || top_k64 > experts || rows64 > hidden) {
        fprintf(stderr, "router/shared dimensions exceed FFN parity limits\n");
        return 2;
    }

    rl_shared_tensor_info *shared_all = calloc(MAX_SHARED_TENSORS, sizeof(*shared_all));
    uint32_t shared_count = 0;
    rl_shared_tensor_info shared[4];
    if (!shared_all ||
        !rl_native_shared_audit(model, shared_all, MAX_SHARED_TENSORS, &shared_count, error, sizeof(error)) ||
        !collect_shared(shared_all, shared_count, (uint32_t)layer64, shared)) {
        fprintf(stderr, "shared expert discovery failed: %s\n", error[0] ? error : "incomplete shared layer");
        free(shared_all);
        return 1;
    }
    free(shared_all);
    if (shared[0].shape[0] != hidden || shared[1].n_dims != 2u || shared[1].shape[1] > UINT32_MAX) {
        fprintf(stderr, "router/shared hidden dimensions do not agree\n");
        return 1;
    }
    const uint32_t shared_ffn = (uint32_t)shared[1].shape[1];

    float *input = malloc((size_t)hidden * sizeof(float));
    float *cpu_logits = malloc((size_t)experts * sizeof(float));
    float *gpu_logits = malloc((size_t)experts * sizeof(float));
    float *cpu_probs = malloc((size_t)experts * sizeof(float));
    float *gpu_probs = malloc((size_t)experts * sizeof(float));
    float *gpu_routed = calloc((size_t)rows64, sizeof(float));
    float *gpu_shared = calloc((size_t)rows64, sizeof(float));
    double *cpu_routed = calloc((size_t)rows64, sizeof(double));
    double *cpu_shared = calloc((size_t)rows64, sizeof(double));
    if (!input || !cpu_logits || !gpu_logits || !cpu_probs || !gpu_probs ||
        !gpu_routed || !gpu_shared || !cpu_routed || !cpu_shared) {
        fprintf(stderr, "out of memory for FFN parity buffers\n");
        free(input); free(cpu_logits); free(gpu_logits); free(cpu_probs); free(gpu_probs);
        free(gpu_routed); free(gpu_shared); free(cpu_routed); free(cpu_shared);
        return 1;
    }
    make_input(input, hidden);

    rl_native_router_telemetry crt = {0}, grt = {0};
    if (!rl_native_router_cpu_f32(model, router, input, hidden, cpu_logits, experts, &crt, error, sizeof(error))) {
        fprintf(stderr, "CPU router failed: %s\n", error);
        goto fail;
    }
#ifdef __APPLE__
    if (!rl_native_router_gpu_f32(model, router, input, hidden, gpu_logits, experts, &grt, error, sizeof(error))) {
        fprintf(stderr, "Metal router failed: %s\n", error);
        goto fail;
    }
#else
    fprintf(stderr, "complete FFN Metal parity requires macOS\n");
    goto fail;
#endif

    uint32_t cpu_ids[MAX_TOPK], gpu_ids[MAX_TOPK];
    float cpu_weights[MAX_TOPK], gpu_weights[MAX_TOPK];
    if (!rl_native_router_select_softmax_topk(cpu_logits, experts, (uint32_t)top_k64,
            cpu_ids, cpu_weights, cpu_probs, error, sizeof(error)) ||
        !rl_native_router_select_softmax_topk(gpu_logits, experts, (uint32_t)top_k64,
            gpu_ids, gpu_weights, gpu_probs, error, sizeof(error))) {
        fprintf(stderr, "router selection failed: %s\n", error);
        goto fail;
    }
    double router_logit_abs = 0.0, router_weight_abs = 0.0;
    int ids_match = 1;
    for (uint32_t i = 0; i < experts; ++i) {
        const double e = fabs((double)gpu_logits[i] - cpu_logits[i]);
        if (e > router_logit_abs) router_logit_abs = e;
    }
    for (uint32_t k = 0; k < (uint32_t)top_k64; ++k) {
        if (cpu_ids[k] != gpu_ids[k]) ids_match = 0;
        const double e = fabs((double)gpu_weights[k] - cpu_weights[k]);
        if (e > router_weight_abs) router_weight_abs = e;
    }
    const int router_match = ids_match && router_logit_abs <= 1e-3 && router_weight_abs <= 1e-5;
    if (!router_match) {
        fprintf(stderr, "refusing FFN parity because router parity failed\n");
        goto fail;
    }

    rl_expert_map map;
    if (!rl_native_build_expert_map(model, experts, &map, error, sizeof(error))) {
        fprintf(stderr, "expert map failed: %s\n", error);
        goto fail;
    }
    rl_native_metal_runtime *runtime = rl_native_metal_create(
        model, &map, cache_mib64 * MIB, 64u, error, sizeof(error));
    if (!runtime) {
        fprintf(stderr, "Metal routed runtime failed: %s\n", error);
        rl_native_free_expert_map(&map);
        goto fail;
    }

    rl_native_metal_telemetry routed_tel = {0};
    double cpu_routed_ms = 0.0;
    if (!rl_native_metal_execute_topk(runtime, &map, (uint32_t)layer64,
            gpu_ids, gpu_weights, (uint32_t)top_k64, 0u, (uint32_t)rows64,
            input, hidden, gpu_routed, (uint32_t)rows64, &routed_tel, error, sizeof(error))) {
        fprintf(stderr, "Metal routed branch failed: %s\n", error);
        rl_native_metal_destroy(runtime);
        rl_native_free_expert_map(&map);
        goto fail;
    }
    if (!rl_native_reference_topk(model, &map, (uint32_t)layer64,
            cpu_ids, cpu_weights, (uint32_t)top_k64, 0u, (uint32_t)rows64,
            input, hidden, cpu_routed, (uint32_t)rows64, &cpu_routed_ms, error, sizeof(error))) {
        fprintf(stderr, "CPU routed branch failed: %s\n", error);
        rl_native_metal_destroy(runtime);
        rl_native_free_expert_map(&map);
        goto fail;
    }
    rl_native_metal_destroy(runtime);
    rl_native_free_expert_map(&map);

    rl_native_shared_telemetry cst = {0}, gst = {0};
    if (!rl_native_shared_cpu_execute(model, shared, input, hidden, 0u, (uint32_t)rows64,
            cpu_shared, (uint32_t)rows64, &cst, error, sizeof(error))) {
        fprintf(stderr, "CPU shared branch failed: %s\n", error);
        goto fail;
    }
#ifdef __APPLE__
    if (!rl_native_shared_gpu_execute(model, shared, input, hidden, 0u, (uint32_t)rows64,
            gpu_shared, (uint32_t)rows64, &gst, error, sizeof(error))) {
        fprintf(stderr, "Metal shared branch failed: %s\n", error);
        goto fail;
    }
#endif

    double routed_abs = 0.0, routed_rel = 0.0, shared_abs = 0.0, shared_rel = 0.0;
    const int routed_match = rows_match_float_double(gpu_routed, cpu_routed, (uint32_t)rows64, &routed_abs, &routed_rel);
    const int shared_match = rows_match_float_double(gpu_shared, cpu_shared, (uint32_t)rows64, &shared_abs, &shared_rel) &&
        fabs(gst.scalar_gate - cst.scalar_gate) <= 1e-5;

    double ffn_abs = 0.0, ffn_rel = 0.0;
    int ffn_match = router_match && routed_match && shared_match;
    for (uint32_t r = 0; r < (uint32_t)rows64; ++r) {
        const double cpu = cpu_routed[r] + cpu_shared[r];
        const double gpu = (double)gpu_routed[r] + (double)gpu_shared[r];
        const double ae = fabs(gpu - cpu);
        const double re = ae / fmax(fabs(cpu), 1e-12);
        if (ae > ffn_abs) ffn_abs = ae;
        if (re > ffn_rel) ffn_rel = re;
        if (ae > 1e-3 + 1e-4 * fabs(cpu)) ffn_match = 0;
    }

    printf("runtime            : complete native Qwen3-Next FFN parity (no Python/ctypes)\n");
    printf("layer / top-k      : %u / %u\n", (uint32_t)layer64, (uint32_t)top_k64);
    printf("hidden / shared ffn: %u / %u\n", hidden, shared_ffn);
    printf("shared quant       : %s(%u) / %s(%u) / %s(%u)\n",
        shared_type_name(shared[1].ggml_type), shared[1].ggml_type,
        shared_type_name(shared[2].ggml_type), shared[2].ggml_type,
        shared_type_name(shared[3].ggml_type), shared[3].ggml_type);
    printf("router ids match   : %s\n", ids_match ? "YES" : "NO");
    printf("router max logit   : %.6g\n", router_logit_abs);
    printf("router max weight  : %.6g\n", router_weight_abs);
    printf("router parity      : %s\n", router_match ? "YES" : "NO");
    printf("selected experts   : ");
    for (uint32_t k = 0; k < (uint32_t)top_k64; ++k) printf("%s%u", k ? "," : "", gpu_ids[k]);
    printf("\n");
    printf("routed loads       : %" PRIu64 "\n", routed_tel.expert_loads);
    printf("routed pread calls : %" PRIu64 "\n", routed_tel.read_calls_total);
    printf("routed SSD compute : %" PRIu64 " bytes / %" PRIu64 " calls\n",
        routed_tel.bytes_read_during_execute, routed_tel.read_calls_during_execute);
    printf("routed GPU / CPU   : %.3f / %.3f ms\n", routed_tel.gpu_ms, cpu_routed_ms);
    printf("routed max abs/rel : %.6g / %.6g\n", routed_abs, routed_rel);
    printf("routed parity      : %s\n", routed_match ? "YES" : "NO");
    printf("shared scalar gate : cpu=%.9f gpu=%.9f\n", cst.scalar_gate, gst.scalar_gate);
    printf("shared read calls  : CPU=%" PRIu64 " GPU=%" PRIu64 "\n", cst.read_calls, gst.read_calls);
    printf("shared SSD compute : 0 bytes / 0 calls\n");
    printf("shared GPU / CPU   : %.3f / %.3f ms\n", gst.compute_ms, cst.compute_ms);
    printf("shared max abs/rel : %.6g / %.6g\n", shared_abs, shared_rel);
    printf("shared parity      : %s\n", shared_match ? "YES" : "NO");
    printf("FFN max abs error  : %.6g\n", ffn_abs);
    printf("FFN max rel error  : %.6g\n", ffn_rel);
    printf("COMPLETE FFN parity: %s\n", ffn_match ? "YES" : "NO");
    for (uint32_t r = 0; r < (uint32_t)rows64; ++r) {
        const double cpu = cpu_routed[r] + cpu_shared[r];
        const double gpu = (double)gpu_routed[r] + (double)gpu_shared[r];
        printf("row %4u           : gpu=%+0.7f cpu=%+0.7f delta=%+.3e\n",
            r, gpu, cpu, gpu - cpu);
    }

    free(input); free(cpu_logits); free(gpu_logits); free(cpu_probs); free(gpu_probs);
    free(gpu_routed); free(gpu_shared); free(cpu_routed); free(cpu_shared);
    return ffn_match ? 0 : 2;

fail:
    free(input); free(cpu_logits); free(gpu_logits); free(cpu_probs); free(gpu_probs);
    free(gpu_routed); free(gpu_shared); free(cpu_routed); free(cpu_shared);
    return 1;
}
