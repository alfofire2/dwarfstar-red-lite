#define _POSIX_C_SOURCE 200809L

/*
 * redlite-engine: diagnostics for the persistent native Qwen3-Next engine.
 *
 *   info   MODEL                      geometry, layer map, memory
 *   parity MODEL --tokens a,b,c ...   replay a token sequence through the CPU
 *                                     oracle and the Metal backend, compare every
 *                                     layer output, router selection and logits
 *   logits MODEL --tokens ... --backend cpu|gpu [--out FILE]
 *                                     dump per-token logits (binary f32) for
 *                                     external comparison
 */

#include "redlite_native_engine.h"

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int parse_u32(const char *s, uint32_t *out) {
    if (!s || !*s) return 0;
    errno = 0;
    char *end = NULL;
    const unsigned long v = strtoul(s, &end, 10);
    if (errno || !end || *end || v > UINT32_MAX) return 0;
    *out = (uint32_t)v;
    return 1;
}

static uint32_t parse_tokens(const char *list, uint32_t *out, uint32_t cap) {
    uint32_t n = 0;
    const char *p = list;
    while (p && *p && n < cap) {
        char *end = NULL;
        errno = 0;
        const unsigned long v = strtoul(p, &end, 10);
        if (errno || end == p) return 0;
        out[n++] = (uint32_t)v;
        p = end;
        if (*p == ',') ++p;
        else if (*p) return 0;
    }
    return n;
}

typedef struct { float max_abs; float max_rel; size_t index; float ref_at; float got_at; } cmp;

static cmp compare(const float *got, const float *ref, size_t n) {
    cmp c = {0.0f, 0.0f, 0u, 0.0f, 0.0f};
    for (size_t i = 0; i < n; ++i) {
        const float a = fabsf(got[i] - ref[i]);
        const float r = a / fmaxf(fabsf(ref[i]), 1.0e-12f);
        if (a > c.max_abs) { c.max_abs = a; c.index = i; c.ref_at = ref[i]; c.got_at = got[i]; }
        if (r > c.max_rel) c.max_rel = r;
    }
    return c;
}

static uint32_t argmax(const float *v, uint32_t n) {
    uint32_t best = 0;
    for (uint32_t i = 1; i < n; ++i) if (v[i] > v[best]) best = i;
    return best;
}

static void usage(FILE *out) {
    fprintf(out,
        "redlite-engine 0.3.0.dev18\n\n"
        "Usage:\n"
        "  redlite-engine info MODEL [--context N] [--cache-mib N]\n"
        "  redlite-engine parity MODEL --tokens a,b,c [--context N] [--cache-mib N] [--threads N] [--layers]\n"
        "  redlite-engine logits MODEL --tokens a,b,c --backend cpu|gpu [--out DUMP] [--context N] [--cache-mib N]\n"
        "      DUMP per token (f32): [hidden] embed, [layers][hidden] outputs, [hidden] final norm, [vocab] logits\n");
}

static void print_info(const rl_engine_info *in) {
    printf("layers               : %u (recurrent=%u full-attention=%u)\n", in->n_layer, in->n_recurrent, in->n_attention);
    printf("hidden / vocab       : %u / %u\n", in->hidden, in->vocab);
    printf("experts / top-k      : %u / %u\n", in->n_expert, in->top_k);
    printf("DeltaNet             : conv=%u inner=%u state=%u groups=%u heads=%u channels=%u\n",
        in->d_conv, in->d_inner, in->d_state, in->n_group, in->dt_rank, in->channels);
    printf("attention            : heads=%u kv=%u head_dim=%u rope=%u base=%.9g\n",
        in->n_head, in->n_head_kv, in->head_dim, in->rope_dims, in->rope_freq_base);
    printf("context              : %u positions\n", in->context);
    printf("dense weights        : %.3f MiB (resident, mmap-backed)\n", (double)in->dense_bytes / (1024.0 * 1024.0));
    printf("state per backend    : %.3f MiB\n", (double)in->state_bytes / (1024.0 * 1024.0));
}

int main(int argc, char **argv) {
    if (argc < 3) { usage(argc > 1 ? stderr : stdout); return argc > 1 ? 2 : 0; }
    const char *cmd = argv[1];
    const char *model = argv[2];
    rl_engine_config cfg;
    rl_engine_config_default(&cfg);
    uint32_t tokens[4096];
    uint32_t token_count = 0;
    const char *backend_name = "gpu";
    const char *out_path = NULL;
    int report_layers = 0;
    for (int i = 3; i < argc; ++i) {
        if (strcmp(argv[i], "--layers") == 0) { report_layers = 1; continue; }
        if (i + 1 >= argc) { usage(stderr); return 2; }
        if (strcmp(argv[i], "--tokens") == 0) {
            token_count = parse_tokens(argv[++i], tokens, 4096u);
            if (!token_count) { fprintf(stderr, "invalid --tokens list\n"); return 2; }
        } else if (strcmp(argv[i], "--context") == 0) { if (!parse_u32(argv[++i], &cfg.context)) return 2; }
        else if (strcmp(argv[i], "--cache-mib") == 0) { uint32_t v; if (!parse_u32(argv[++i], &v)) return 2; cfg.cache_mib = v; }
        else if (strcmp(argv[i], "--threads") == 0) { uint32_t v; if (!parse_u32(argv[++i], &v)) return 2; cfg.cpu_threads = (int)v; }
        else if (strcmp(argv[i], "--backend") == 0) backend_name = argv[++i];
        else if (strcmp(argv[i], "--out") == 0) out_path = argv[++i];
        else { fprintf(stderr, "unknown option %s\n", argv[i]); return 2; }
    }
    char error[512] = {0};

    if (strcmp(cmd, "info") == 0) {
        cfg.enable_cpu = 0; cfg.enable_gpu = 0;
        rl_engine *e = rl_engine_open(model, &cfg, error, sizeof(error));
        if (!e) { fprintf(stderr, "engine open failed: %s\n", error); return 1; }
        printf("runtime              : native Qwen3-Next persistent engine\n");
        print_info(rl_engine_info_get(e));
        rl_engine_close(e);
        return 0;
    }

    if (!token_count) { fprintf(stderr, "--tokens is required\n"); return 2; }

    if (strcmp(cmd, "logits") == 0) {
        const int gpu = strcmp(backend_name, "gpu") == 0;
        cfg.enable_cpu = !gpu; cfg.enable_gpu = gpu;
        rl_engine *e = rl_engine_open(model, &cfg, error, sizeof(error));
        if (!e) { fprintf(stderr, "engine open failed: %s\n", error); return 1; }
        const rl_engine_info *in = rl_engine_info_get(e);
        float *logits = (float *)malloc((size_t)in->vocab * sizeof(float));
        FILE *out = out_path ? fopen(out_path, "wb") : NULL;
        if (out_path && !out) { fprintf(stderr, "cannot open %s\n", out_path); rl_engine_close(e); return 1; }
        for (uint32_t i = 0; i < token_count; ++i) {
            rl_engine_step_stats st;
            if (!rl_engine_step(e, gpu ? RL_BACKEND_GPU : RL_BACKEND_CPU, tokens[i], logits, &st, error, sizeof(error))) {
                fprintf(stderr, "step %u failed: %s\n", i, error); rl_engine_close(e); return 1;
            }
            const uint32_t best = argmax(logits, in->vocab);
            printf("token[%u]=%u -> argmax=%u logit=%.6f total=%.1f ms (layers %.1f, output %.1f)\n",
                i, tokens[i], best, logits[best], st.total_ms, st.layers_ms, st.output_ms);
            if (out) {
                fwrite(rl_engine_last_embedding(e, gpu ? RL_BACKEND_GPU : RL_BACKEND_CPU), sizeof(float), in->hidden, out);
                for (uint32_t l = 0; l < in->n_layer; ++l)
                    fwrite(rl_engine_last_layer_output(e, gpu ? RL_BACKEND_GPU : RL_BACKEND_CPU, l), sizeof(float), in->hidden, out);
                fwrite(rl_engine_last_final_norm(e, gpu ? RL_BACKEND_GPU : RL_BACKEND_CPU), sizeof(float), in->hidden, out);
                fwrite(logits, sizeof(float), in->vocab, out);
            }
        }
        if (out) fclose(out);
        free(logits);
        rl_engine_close(e);
        return 0;
    }

    if (strcmp(cmd, "parity") != 0) { usage(stderr); return 2; }
    cfg.enable_cpu = 1; cfg.enable_gpu = 1;
    rl_engine *e = rl_engine_open(model, &cfg, error, sizeof(error));
    if (!e) { fprintf(stderr, "engine open failed: %s\n", error); return 1; }
    const rl_engine_info *in = rl_engine_info_get(e);
    printf("runtime              : native Qwen3-Next engine CPU-vs-Metal multi-token parity\n");
    print_info(in);
    printf("tokens               : %u\n", token_count);
    float *cpu_logits = (float *)malloc((size_t)in->vocab * sizeof(float));
    float *gpu_logits = (float *)malloc((size_t)in->vocab * sizeof(float));
    int all_ok = 1;
    float worst_layer = 0.0f, worst_logit = 0.0f;
    for (uint32_t i = 0; i < token_count; ++i) {
        rl_engine_step_stats cs, gs;
        if (!rl_engine_step(e, RL_BACKEND_GPU, tokens[i], gpu_logits, &gs, error, sizeof(error))) {
            fprintf(stderr, "GPU step %u failed: %s\n", i, error); all_ok = 0; break;
        }
        if (!rl_engine_step(e, RL_BACKEND_CPU, tokens[i], cpu_logits, &cs, error, sizeof(error))) {
            fprintf(stderr, "CPU step %u failed: %s\n", i, error); all_ok = 0; break;
        }
        const cmp emb = compare(rl_engine_last_embedding(e, RL_BACKEND_GPU), rl_engine_last_embedding(e, RL_BACKEND_CPU), in->hidden);
        float layer_worst = 0.0f;
        uint32_t router_mismatch = 0;
        for (uint32_t l = 0; l < in->n_layer; ++l) {
            const cmp c = compare(rl_engine_last_layer_output(e, RL_BACKEND_GPU, l), rl_engine_last_layer_output(e, RL_BACKEND_CPU, l), in->hidden);
            if (c.max_abs > layer_worst) layer_worst = c.max_abs;
            const uint32_t *gi = rl_engine_last_router_ids(e, RL_BACKEND_GPU, l);
            const uint32_t *ci = rl_engine_last_router_ids(e, RL_BACKEND_CPU, l);
            uint32_t mism = 0;
            for (uint32_t k = 0; k < in->top_k; ++k) if (gi[k] != ci[k]) ++mism;
            router_mismatch += mism;
            if (report_layers)
                printf("  token %u layer %02u max_abs=%.6g max_rel=%.4g at[%zu] gpu=%.6f cpu=%.6f router_mismatch=%u\n",
                    i, l, c.max_abs, c.max_rel, c.index, c.got_at, c.ref_at, mism);
        }
        const cmp fn = compare(rl_engine_last_final_norm(e, RL_BACKEND_GPU), rl_engine_last_final_norm(e, RL_BACKEND_CPU), in->hidden);
        const cmp lg = compare(gpu_logits, cpu_logits, in->vocab);
        const uint32_t ga = argmax(gpu_logits, in->vocab), ca = argmax(cpu_logits, in->vocab);
        float lmax = -INFINITY, lmin = INFINITY;
        for (uint32_t v = 0; v < in->vocab; ++v) { if (cpu_logits[v] > lmax) lmax = cpu_logits[v]; if (cpu_logits[v] < lmin) lmin = cpu_logits[v]; }
        const int token_ok = router_mismatch == 0 && ga == ca && lg.max_abs <= 0.05f * fmaxf(1.0f, lmax - lmin) / 10.0f + 0.05f;
        if (layer_worst > worst_layer) worst_layer = layer_worst;
        if (lg.max_abs > worst_logit) worst_logit = lg.max_abs;
        printf("token %u id=%u pos=%u : embed_abs=%.3g layer_max_abs=%.6g final_norm_abs=%.6g logits_abs=%.6g (rel %.3g) argmax gpu=%u cpu=%u router_mismatch=%u -> %s\n",
            i, tokens[i], rl_engine_position(e, RL_BACKEND_GPU) - 1u, emb.max_abs, layer_worst, fn.max_abs, lg.max_abs, lg.max_rel, ga, ca, router_mismatch,
            token_ok ? "OK" : "MISMATCH");
        printf("  timing gpu=%.1f ms (rec %.1f attn %.1f router %.1f routed %.1f shared %.1f out %.1f) cpu=%.1f ms | experts loads=%" PRIu64 " hits=%" PRIu64 " misses=%" PRIu64 " resident=%u/%u ssd=%.1f MiB\n",
            gs.total_ms, gs.recurrent_ms, gs.attention_ms, gs.router_ms, gs.routed_ms, gs.shared_ms, gs.output_ms, cs.total_ms,
            gs.expert_loads, gs.cache_hits, gs.cache_misses, gs.resident_slots, gs.slot_capacity, (double)gs.ssd_bytes / (1024.0 * 1024.0));
        fflush(stdout);
        if (!token_ok) all_ok = 0;
    }
    printf("worst layer abs       : %.6g\n", worst_layer);
    printf("worst logits abs      : %.6g\n", worst_logit);
    printf("MULTI-TOKEN ENGINE PARITY: %s\n", all_ok ? "YES" : "NO");
    free(cpu_logits); free(gpu_logits);
    rl_engine_close(e);
    return all_ok ? 0 : 3;
}
