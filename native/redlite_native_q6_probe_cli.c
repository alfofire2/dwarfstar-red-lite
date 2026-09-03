#define _FILE_OFFSET_BITS 64
#define _POSIX_C_SOURCE 200809L

#include "redlite_native_q6_probe.h"
#include "redlite_native_shared.h"
#include "redlite_native_shared_exec.h"
#include "redlite_native_iq2_xxs.h"

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define MAX_SHARED_TENSORS (RL_SHARED_MAX_LAYERS * 4u)

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

static void make_input(float *x, uint32_t n) {
    for (uint32_t i = 0; i < n; ++i) x[i] = (float)(((int)((i * 19u + 5u) % 67u) - 33) / 32.0);
}

static int pread_full(int fd, void *dst, size_t bytes, uint64_t offset) {
    uint8_t *p = (uint8_t *)dst; size_t done = 0;
    while (done < bytes) {
        const ssize_t n = pread(fd, p + done, bytes - done, (off_t)(offset + done));
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return 0;
        done += (size_t)n;
    }
    return 1;
}

static const char *kind_name(uint32_t k) {
    return k == RL_SHARED_GATE ? "gate" : k == RL_SHARED_UP ? "up" : k == RL_SHARED_DOWN ? "down" : "?";
}

int main(int argc, char **argv) {
    if (argc < 2 || strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0) {
        fprintf(stdout, "usage: redlite-q6-probe MODEL [--layer N] [--rows N]\n");
        return argc < 2 ? 2 : 0;
    }
    const char *model = argv[1];
    uint32_t layer = 0, rows = 8;
    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--layer") == 0 && i + 1 < argc) layer = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (strcmp(argv[i], "--rows") == 0 && i + 1 < argc) rows = (uint32_t)strtoul(argv[++i], NULL, 10);
        else { fprintf(stderr, "unknown option: %s\n", argv[i]); return 2; }
    }
    if (!rows) { fprintf(stderr, "--rows must be > 0\n"); return 2; }

    char error[512] = {0};
    rl_shared_tensor_info *all = calloc(MAX_SHARED_TENSORS, sizeof(*all));
    uint32_t count = 0;
    if (!all || !rl_native_shared_audit(model, all, MAX_SHARED_TENSORS, &count, error, sizeof(error))) {
        fprintf(stderr, "shared audit failed: %s\n", error); free(all); return 1;
    }
    rl_shared_tensor_info t[4];
    if (!collect_layer(all, count, layer, t)) { fprintf(stderr, "layer %u shared tensors incomplete\n", layer); free(all); return 1; }
    free(all);
    if (t[1].ggml_type != 14u || t[2].ggml_type != 14u || t[3].ggml_type != 14u) {
        fprintf(stderr, "layer %u is not an all-Q6_K shared FFN\n", layer); return 2;
    }

    uint8_t grid[RL_IQ2_XXS_GRID_COUNT];
    if (!rl_native_iq2_xxs_build_grid(grid, error, sizeof(error))) { fprintf(stderr, "grid init failed: %s\n", error); return 1; }
    int fd = open(model, O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }

    printf("runtime            : native C vs isolated Metal Q6_K matrix probe\n");
    printf("layer              : %u\n", layer);
    int overall = 1;
    for (uint32_t kind = RL_SHARED_GATE; kind <= RL_SHARED_DOWN; ++kind) {
        const rl_shared_tensor_info *m = &t[kind];
        const uint32_t ncols = (uint32_t)m->shape[0];
        const uint32_t nrows = (uint32_t)m->shape[1];
        const uint32_t use_rows = rows < nrows ? rows : nrows;
        const size_t rb = rl_native_shared_row_bytes(14u, ncols);
        float *input = malloc((size_t)ncols * sizeof(float));
        float *gpu = calloc(use_rows, sizeof(float));
        double *cpu = calloc(use_rows, sizeof(double));
        uint8_t *rowbuf = malloc(rb);
        if (!input || !gpu || !cpu || !rowbuf) { fprintf(stderr, "oom\n"); overall = 0; free(input); free(gpu); free(cpu); free(rowbuf); break; }
        make_input(input, ncols);
        for (uint32_t r = 0; r < use_rows; ++r) {
            if (!pread_full(fd, rowbuf, rb, m->tensor_offset + (uint64_t)r * rb) ||
                !rl_native_shared_quant_row_dot(rowbuf, rb, 14u, input, ncols, grid, sizeof(grid), &cpu[r], error, sizeof(error))) {
                fprintf(stderr, "%s CPU row %u failed: %s\n", kind_name(kind), r, error); overall = 0; break;
            }
        }
#ifdef __APPLE__
        if (overall && !rl_native_q6_gpu_rows(model, m, input, ncols, 0u, use_rows, gpu, use_rows, error, sizeof(error))) {
            fprintf(stderr, "%s GPU rows failed: %s\n", kind_name(kind), error); overall = 0;
        }
#else
        fprintf(stderr, "Metal Q6 probe requires macOS\n"); overall = 0;
#endif
        double max_abs = 0.0, max_rel = 0.0;
        int match = overall;
        for (uint32_t r = 0; r < use_rows; ++r) {
            const double ae = fabs((double)gpu[r] - cpu[r]);
            const double re = ae / fmax(fabs(cpu[r]), 1e-12);
            if (ae > max_abs) max_abs = ae;
            if (re > max_rel) max_rel = re;
            if (ae > 1e-4 + 1e-5 * fabs(cpu[r])) match = 0;
        }
        printf("%-19s: ncols=%u rows=%u max_abs=%.6g max_rel=%.6g parity=%s\n",
            kind_name(kind), ncols, use_rows, max_abs, max_rel, match ? "YES" : "NO");
        for (uint32_t r = 0; r < use_rows && r < 4u; ++r)
            printf("  row %u: gpu=%+0.7f cpu=%+0.7f delta=%+.3e\n", r, gpu[r], cpu[r], (double)gpu[r] - cpu[r]);
        if (!match) overall = 0;
        free(input); free(gpu); free(cpu); free(rowbuf);
    }
    close(fd);
    printf("Q6 matrix parity   : %s\n", overall ? "YES" : "NO");
    return overall ? 0 : 2;
}
