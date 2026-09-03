#define _FILE_OFFSET_BITS 64
#define _POSIX_C_SOURCE 200809L

#include "redlite_native_shared.h"
#include "redlite_native_shared_exec.h"

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_SHARED_TENSORS (RL_SHARED_MAX_LAYERS * 4u)

static void usage(FILE *out) {
    fprintf(out,
        "redlite-shared 0.3.0.dev14\n\n"
        "Usage:\n"
        "  redlite-shared parity MODEL [--layer N] [--rows N]\n\n"
        "Runs the real Qwen3-Next shared-expert branch through an independent native CPU oracle and Metal.\n"
        "Supported shared FFN formats: Q6_K and IQ2_XXS; scalar gate: F32.\n");
}

static int parse_u64(const char *s, uint64_t *out) {
    if (!s || !*s) return 0;
    errno = 0; char *end = NULL;
    const unsigned long long v = strtoull(s, &end, 10);
    if (errno || !end || *end) return 0;
    *out = (uint64_t)v; return 1;
}

static void make_input(float *input, uint32_t hidden) {
    for (uint32_t i = 0; i < hidden; ++i)
        input[i] = (float)(((int)((i * 17u + 3u) % 61u) - 30) / 32.0);
}

static const char *type_name(uint32_t t) {
    switch (t) {
        case 0u: return "F32";
        case 14u: return "Q6_K";
        case 16u: return "IQ2_XXS";
        default: return "OTHER";
    }
}

static int collect_layer(const rl_shared_tensor_info *all, uint32_t count, uint32_t layer, rl_shared_tensor_info out[4]) {
    uint8_t seen[4] = {0};
    memset(out, 0, 4u * sizeof(*out));
    for (uint32_t i = 0; i < count; ++i) {
        if (all[i].layer != layer || all[i].kind > RL_SHARED_DOWN) continue;
        const uint32_t k = (uint32_t)all[i].kind;
        if (seen[k]) return 0;
        out[k] = all[i]; seen[k] = 1;
    }
    return seen[0] && seen[1] && seen[2] && seen[3];
}

int main(int argc, char **argv) {
    if (argc < 2 || strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0) {
        usage(stdout); return argc < 2 ? 2 : 0;
    }
    if (strcmp(argv[1], "parity") != 0 || argc < 3) { usage(stderr); return 2; }
    const char *model = argv[2];
    uint64_t layer64 = 0, rows64 = 8;
    for (int i = 3; i < argc; ++i) {
        uint64_t *target = NULL;
        if (strcmp(argv[i], "--layer") == 0) target = &layer64;
        else if (strcmp(argv[i], "--rows") == 0) target = &rows64;
        else { fprintf(stderr, "unknown option: %s\n", argv[i]); return 2; }
        if (i + 1 >= argc || !parse_u64(argv[++i], target)) { fprintf(stderr, "invalid option value\n"); return 2; }
    }
    if (layer64 >= RL_SHARED_MAX_LAYERS || !rows64 || rows64 > UINT32_MAX) {
        fprintf(stderr, "shared parity arguments out of range\n"); return 2;
    }

    char error[512] = {0};
    rl_shared_tensor_info *all = calloc(MAX_SHARED_TENSORS, sizeof(*all));
    if (!all) { fprintf(stderr, "out of memory for shared audit\n"); return 1; }
    uint32_t count = 0;
    if (!rl_native_shared_audit(model, all, MAX_SHARED_TENSORS, &count, error, sizeof(error))) {
        fprintf(stderr, "shared audit failed: %s\n", error); free(all); return 1;
    }
    rl_shared_tensor_info t[4];
    if (!collect_layer(all, count, (uint32_t)layer64, t)) {
        fprintf(stderr, "requested layer does not contain a complete shared expert\n"); free(all); return 1;
    }
    free(all);
    if (t[0].n_dims != 1u || t[1].n_dims != 2u || t[0].shape[0] > UINT32_MAX || rows64 > t[0].shape[0]) {
        fprintf(stderr, "shared layer shape is not supported\n"); return 1;
    }
    const uint32_t hidden = (uint32_t)t[0].shape[0];
    const uint32_t ffn = (uint32_t)t[1].shape[1];
    float *input = malloc((size_t)hidden * sizeof(float));
    float *gpu = calloc((size_t)rows64, sizeof(float));
    double *cpu = calloc((size_t)rows64, sizeof(double));
    if (!input || !gpu || !cpu) { fprintf(stderr, "out of memory for shared parity\n"); free(input); free(gpu); free(cpu); return 1; }
    make_input(input, hidden);

    rl_native_shared_telemetry ct = {0}, gt = {0};
    if (!rl_native_shared_cpu_execute(model, t, input, hidden, 0u, (uint32_t)rows64,
            cpu, (uint32_t)rows64, &ct, error, sizeof(error))) {
        fprintf(stderr, "shared CPU reference failed: %s\n", error); free(input); free(gpu); free(cpu); return 1;
    }
#ifdef __APPLE__
    if (!rl_native_shared_gpu_execute(model, t, input, hidden, 0u, (uint32_t)rows64,
            gpu, (uint32_t)rows64, &gt, error, sizeof(error))) {
        fprintf(stderr, "shared Metal failed: %s\n", error); free(input); free(gpu); free(cpu); return 1;
    }
#else
    fprintf(stderr, "shared Metal parity requires macOS\n"); free(input); free(gpu); free(cpu); return 2;
#endif

    double max_abs = 0.0, max_rel = 0.0;
    int match = 1;
    for (uint32_t r = 0; r < (uint32_t)rows64; ++r) {
        const double e = fabs((double)gpu[r] - cpu[r]);
        const double rel = e / fmax(fabs(cpu[r]), 1e-12);
        if (e > max_abs) max_abs = e;
        if (rel > max_rel) max_rel = rel;
        if (e > 1e-3 + 1e-4 * fabs(cpu[r])) match = 0;
    }
    const double gate_abs = fabs(gt.scalar_gate - ct.scalar_gate);
    if (gate_abs > 1e-5) match = 0;

    printf("runtime            : native C + Metal shared expert (no Python/ctypes)\n");
    printf("layer              : %u\n", (uint32_t)layer64);
    printf("hidden / shared ffn: %u / %u\n", hidden, ffn);
    printf("gate input type    : %s (%u)\n", type_name(t[0].ggml_type), t[0].ggml_type);
    printf("gate / up / down   : %s(%u) / %s(%u) / %s(%u)\n",
        type_name(t[1].ggml_type), t[1].ggml_type, type_name(t[2].ggml_type), t[2].ggml_type,
        type_name(t[3].ggml_type), t[3].ggml_type);
    printf("rows tested        : 0..%u\n", (uint32_t)rows64 - 1u);
    printf("CPU read           : %.3f ms / %" PRIu64 " calls / %.3f MiB\n",
        ct.read_ms, ct.read_calls, (double)ct.bytes_read / (1024.0 * 1024.0));
    printf("GPU read           : %.3f ms / %" PRIu64 " calls / %.3f MiB\n",
        gt.read_ms, gt.read_calls, (double)gt.bytes_read / (1024.0 * 1024.0));
    printf("SSD during shared  : 0 bytes / 0 calls\n");
    printf("CPU shared compute : %.3f ms\n", ct.compute_ms);
    printf("GPU shared compute : %.3f ms\n", gt.compute_ms);
    printf("CPU scalar gate    : %.9f\n", ct.scalar_gate);
    printf("GPU scalar gate    : %.9f\n", gt.scalar_gate);
    printf("scalar gate abs err: %.6g\n", gate_abs);
    printf("shared max abs err : %.6g\n", max_abs);
    printf("shared max rel err : %.6g\n", max_rel);
    printf("shared parity      : %s\n", match ? "YES" : "NO");
    for (uint32_t r = 0; r < (uint32_t)rows64; ++r)
        printf("row %4u           : gpu=%+0.7f cpu=%+0.7f delta=%+.3e\n",
            r, gpu[r], cpu[r], (double)gpu[r] - cpu[r]);

    free(input); free(gpu); free(cpu);
    return match ? 0 : 2;
}
