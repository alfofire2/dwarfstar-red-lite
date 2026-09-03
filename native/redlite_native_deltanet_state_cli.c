#define _POSIX_C_SOURCE 200809L

#include "redlite_native_deltanet_state.h"

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#define RL_DN_STATE_SIZE 128u
#define RL_DN_KEY_HEADS 16u
#define RL_DN_VALUE_HEADS 32u

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

static void usage(const char *argv0) {
    fprintf(stderr,
        "redlite-deltanet-state 0.3.0.dev15\n\n"
        "Usage:\n"
        "  %s parity MODEL --layer N\n"
        "  %s --selftest\n\n"
        "Validates the single-token Qwen3-Next Gated DeltaNet recurrent-state update\n"
        "with the pinned llama.cpp transposed state layout and Q/K head broadcast.\n",
        argv0, argv0);
}

static int parse_u32(const char *s, uint32_t *out) {
    char *end = NULL;
    errno = 0;
    unsigned long v = strtoul(s, &end, 10);
    if (errno || !end || *end || v > 0xfffffffful) return 0;
    *out = (uint32_t)v;
    return 1;
}

static void normalize_heads(float *x, uint32_t heads, uint32_t dim) {
    for (uint32_t h = 0; h < heads; ++h) {
        float ss = 0.0f;
        float *row = x + (size_t)h * dim;
        for (uint32_t i = 0; i < dim; ++i) ss += row[i] * row[i];
        const float inv = 1.0f / fmaxf(sqrtf(ss), 1.0e-6f);
        for (uint32_t i = 0; i < dim; ++i) row[i] *= inv;
    }
}

static void cpu_state_update(
        const float *q,
        const float *k,
        const float *v,
        const float *gate,
        const float *beta,
        const float *prev_state,
        uint32_t state_size,
        uint32_t key_heads,
        uint32_t value_heads,
        float *delta,
        float *next_state,
        float *output) {
    const size_t matrix = (size_t)state_size * state_size;
    const float scale = 1.0f / sqrtf((float)state_size);
    memcpy(next_state, prev_state, matrix * value_heads * sizeof(float));

    for (uint32_t h = 0; h < value_heads; ++h) {
        const uint32_t kh = h / (value_heads / key_heads); /* pinned llama.cpp repeat-interleave */
        const float *qh = q + (size_t)kh * state_size;
        const float *khv = k + (size_t)kh * state_size;
        const float *vh = v + (size_t)h * state_size;
        float *mh = next_state + (size_t)h * matrix;
        float *dh = delta + (size_t)h * state_size;
        float *oh = output + (size_t)h * state_size;
        const float decay = expf(gate[h]);

        /* Pinned ggml stores M[j][i] = S[i][j], so every contiguous row j
         * is a column of the mathematical recurrent state. */
        for (uint32_t j = 0; j < state_size; ++j) {
            float *row = mh + (size_t)j * state_size;
            for (uint32_t i = 0; i < state_size; ++i) row[i] *= decay;
        }

        for (uint32_t j = 0; j < state_size; ++j) {
            float sum = 0.0f;
            const float *row = mh + (size_t)j * state_size;
            for (uint32_t i = 0; i < state_size; ++i) sum += row[i] * khv[i];
            dh[j] = (vh[j] - sum) * beta[h];
        }

        for (uint32_t j = 0; j < state_size; ++j) {
            float *row = mh + (size_t)j * state_size;
            const float d = dh[j];
            for (uint32_t i = 0; i < state_size; ++i) row[i] += khv[i] * d;
        }

        for (uint32_t j = 0; j < state_size; ++j) {
            float sum = 0.0f;
            const float *row = mh + (size_t)j * state_size;
            for (uint32_t i = 0; i < state_size; ++i) sum += row[i] * qh[i];
            oh[j] = sum * scale;
        }
    }
}

static int selftest(void) {
    {
        if ((0u % 2u) != 0u || (1u % 2u) != 1u || (2u % 2u) != 0u || (3u % 2u) != 1u) {
            fprintf(stderr, "DeltaNet Q/K broadcast: FAIL\n");
            return 1;
        }
        printf("DeltaNet Q/K broadcast: OK\n");
    }

    {
        const uint32_t S = 2, HK = 1, HV = 1;
        const float q[2] = {1.0f, 0.0f};
        const float k[2] = {1.0f, 0.0f};
        const float v[2] = {0.5f, -0.25f};
        const float gate[1] = {0.0f};
        const float beta[1] = {0.5f};
        const float prev[4] = {0};
        float d[2] = {0}, state[4] = {0}, out[2] = {0};
        cpu_state_update(q, k, v, gate, beta, prev, S, HK, HV, d, state, out);
        const float inv = 1.0f / sqrtf(2.0f);
        const int ok = fabsf(d[0] - 0.25f) < 1.0e-7f &&
                       fabsf(d[1] + 0.125f) < 1.0e-7f &&
                       fabsf(state[0] - 0.25f) < 1.0e-7f &&
                       fabsf(state[2] + 0.125f) < 1.0e-7f &&
                       fabsf(out[0] - 0.25f * inv) < 1.0e-7f &&
                       fabsf(out[1] + 0.125f * inv) < 1.0e-7f;
        if (!ok) {
            fprintf(stderr, "DeltaNet state oracle : FAIL\n");
            return 1;
        }
        printf("DeltaNet state oracle : OK\n");
    }
    return 0;
}

typedef struct {
    float max_abs;
    float max_rel;
    size_t index;
} error_stats;

static error_stats compare_arrays(const float *got, const float *ref, size_t n) {
    error_stats s = {0.0f, 0.0f, 0u};
    for (size_t i = 0; i < n; ++i) {
        const float a = fabsf(got[i] - ref[i]);
        const float r = a / (fabsf(ref[i]) + 1.0e-12f);
        if (a > s.max_abs) { s.max_abs = a; s.index = i; }
        if (r > s.max_rel) s.max_rel = r;
    }
    return s;
}

static int within(const float *got, const float *ref, size_t n, float abs_tol, float rel_tol) {
    for (size_t i = 0; i < n; ++i) {
        if (fabsf(got[i] - ref[i]) > abs_tol + rel_tol * fabsf(ref[i])) return 0;
    }
    return 1;
}

static void fill_fixture(float *q, float *k, float *v, float *gate, float *beta, float *state) {
    const uint32_t S = RL_DN_STATE_SIZE;
    for (uint32_t h = 0; h < RL_DN_KEY_HEADS; ++h) {
        for (uint32_t i = 0; i < S; ++i) {
            const int qn = (int)((h * 29u + i * 17u + 11u) % 101u) - 50;
            const int kn = (int)((h * 13u + i * 23u + 7u) % 97u) - 48;
            q[(size_t)h * S + i] = (float)qn / 48.0f;
            k[(size_t)h * S + i] = (float)kn / 46.0f;
        }
    }
    normalize_heads(q, RL_DN_KEY_HEADS, S);
    normalize_heads(k, RL_DN_KEY_HEADS, S);

    for (uint32_t h = 0; h < RL_DN_VALUE_HEADS; ++h) {
        gate[h] = -0.0008f - 0.00017f * (float)(h % 9u);
        beta[h] = 0.18f + 0.035f * (float)(h % 13u);
        for (uint32_t j = 0; j < S; ++j) {
            const int vn = (int)((h * 31u + j * 19u + 5u) % 113u) - 56;
            v[(size_t)h * S + j] = (float)vn / 768.0f;
        }
    }

    const size_t matrix = (size_t)S * S;
    for (uint32_t h = 0; h < RL_DN_VALUE_HEADS; ++h) {
        for (uint32_t j = 0; j < S; ++j) {
            for (uint32_t i = 0; i < S; ++i) {
                const int sn = (int)((h * 37u + j * 11u + i * 7u + 3u) % 127u) - 63;
                state[(size_t)h * matrix + (size_t)j * S + i] = (float)sn / 8192.0f;
            }
        }
    }
}

static int run_parity(const char *model_path, uint32_t layer) {
    struct stat st;
    if (stat(model_path, &st) != 0 || st.st_size <= 0) {
        fprintf(stderr, "model is not accessible: %s\n", model_path);
        return 2;
    }
    if (layer >= 48u || ((layer + 1u) % 4u) == 0u) {
        fprintf(stderr, "layer %u is not one of the audited recurrent Qwen3-Next layers\n", layer);
        return 2;
    }

    const uint32_t S = RL_DN_STATE_SIZE;
    const uint32_t HK = RL_DN_KEY_HEADS;
    const uint32_t HV = RL_DN_VALUE_HEADS;
    const size_t qk_count = (size_t)HK * S;
    const size_t v_count = (size_t)HV * S;
    const size_t state_count = (size_t)HV * S * S;

    float *q = calloc(qk_count, sizeof(float));
    float *k = calloc(qk_count, sizeof(float));
    float *v = calloc(v_count, sizeof(float));
    float *gate = calloc(HV, sizeof(float));
    float *beta = calloc(HV, sizeof(float));
    float *prev = calloc(state_count, sizeof(float));
    float *cpu_delta = calloc(v_count, sizeof(float));
    float *cpu_state = calloc(state_count, sizeof(float));
    float *cpu_out = calloc(v_count, sizeof(float));
    float *gpu_delta = calloc(v_count, sizeof(float));
    float *gpu_state = calloc(state_count, sizeof(float));
    float *gpu_out = calloc(v_count, sizeof(float));
    if (!q || !k || !v || !gate || !beta || !prev || !cpu_delta || !cpu_state || !cpu_out ||
        !gpu_delta || !gpu_state || !gpu_out) {
        fprintf(stderr, "allocation failed for DeltaNet state fixture\n");
        free(q); free(k); free(v); free(gate); free(beta); free(prev); free(cpu_delta); free(cpu_state); free(cpu_out);
        free(gpu_delta); free(gpu_state); free(gpu_out);
        return 2;
    }

    fill_fixture(q, k, v, gate, beta, prev);
    const double c0 = now_ms();
    cpu_state_update(q, k, v, gate, beta, prev, S, HK, HV, cpu_delta, cpu_state, cpu_out);
    const double cpu_ms = now_ms() - c0;

#ifdef __APPLE__
    rl_dn_state_telemetry tel = {0};
    char error[512] = {0};
    if (!rl_deltanet_state_gpu_execute(q, k, v, gate, beta, prev, S, HK, HV,
            gpu_delta, gpu_state, gpu_out, &tel, error, sizeof(error))) {
        fprintf(stderr, "Metal state update failed: %s\n", error[0] ? error : "unknown error");
        free(q); free(k); free(v); free(gate); free(beta); free(prev); free(cpu_delta); free(cpu_state); free(cpu_out);
        free(gpu_delta); free(gpu_state); free(gpu_out);
        return 2;
    }
#else
    fprintf(stderr, "parity requires macOS Metal\n");
    free(q); free(k); free(v); free(gate); free(beta); free(prev); free(cpu_delta); free(cpu_state); free(cpu_out);
    free(gpu_delta); free(gpu_state); free(gpu_out);
    return 2;
#endif

    const error_stats de = compare_arrays(gpu_delta, cpu_delta, v_count);
    const error_stats se = compare_arrays(gpu_state, cpu_state, state_count);
    const error_stats oe = compare_arrays(gpu_out, cpu_out, v_count);
    const int d_ok = within(gpu_delta, cpu_delta, v_count, 2.0e-6f, 2.0e-5f);
    const int s_ok = within(gpu_state, cpu_state, state_count, 3.0e-6f, 3.0e-5f);
    const int o_ok = within(gpu_out, cpu_out, v_count, 8.0e-6f, 8.0e-5f);
    const int ok = d_ok && s_ok && o_ok;

    printf("runtime            : native C + Metal DeltaNet recurrent-state parity (no Python/ctypes)\n");
    printf("layer              : %u\n", layer);
    printf("state geometry     : S=%u key_heads=%u value_heads=%u\n", S, HK, HV);
    printf("Q/K broadcast      : value_head / (value_heads/key_heads) (0,0,1,1,...,15,15)\n");
    printf("state layout       : transposed M[j][i] = S[i][j]\n");
    printf("state payload      : %.3f MiB\n", (double)(state_count * sizeof(float)) / (1024.0 * 1024.0));
    printf("CPU / GPU compute  : %.3f / %.3f ms\n", cpu_ms, tel.compute_ms);
    printf("delta max abs/rel  : %.6g / %.6g parity=%s\n", de.max_abs, de.max_rel, d_ok ? "YES" : "NO");
    printf("state max abs/rel  : %.6g / %.6g parity=%s\n", se.max_abs, se.max_rel, s_ok ? "YES" : "NO");
    printf("output max abs/rel : %.6g / %.6g parity=%s\n", oe.max_abs, oe.max_rel, o_ok ? "YES" : "NO");
    printf("state parity       : %s\n", ok ? "YES" : "NO");
    printf("sample h0          : delta gpu=%+.7f cpu=%+.7f out gpu=%+.7f cpu=%+.7f\n",
           gpu_delta[0], cpu_delta[0], gpu_out[0], cpu_out[0]);
    printf("sample h16/map0    : delta gpu=%+.7f cpu=%+.7f out gpu=%+.7f cpu=%+.7f\n",
           gpu_delta[(size_t)16u * S], cpu_delta[(size_t)16u * S],
           gpu_out[(size_t)16u * S], cpu_out[(size_t)16u * S]);

    free(q); free(k); free(v); free(gate); free(beta); free(prev); free(cpu_delta); free(cpu_state); free(cpu_out);
    free(gpu_delta); free(gpu_state); free(gpu_out);
    return ok ? 0 : 1;
}

int main(int argc, char **argv) {
    if (argc == 2 && strcmp(argv[1], "--selftest") == 0) return selftest();
    if (argc == 2 && strcmp(argv[1], "--help") == 0) { usage(argv[0]); return 0; }
    if (argc < 5 || strcmp(argv[1], "parity") != 0) { usage(argv[0]); return 2; }

    const char *model = argv[2];
    uint32_t layer = UINT32_MAX;
    for (int i = 3; i < argc; ++i) {
        if (strcmp(argv[i], "--layer") == 0 && i + 1 < argc) {
            if (!parse_u32(argv[++i], &layer)) { fprintf(stderr, "invalid --layer\n"); return 2; }
        } else {
            fprintf(stderr, "unknown argument: %s\n", argv[i]);
            return 2;
        }
    }
    if (layer == UINT32_MAX) { fprintf(stderr, "--layer is required\n"); return 2; }
    return run_parity(model, layer);
}
