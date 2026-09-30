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
#include "redlite_native_tokenizer.h"

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
    if (p && *p) return 0;   /* more ids than the capacity: reject instead of truncating */
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

/* Decode the corpus escape sequences (backslash n, t, r and double backslash) in place; returns the new length. */
static size_t unescape_line(char *s, size_t n) {
    size_t w = 0;
    for (size_t r = 0; r < n; ++r) {
        if (s[r] == '\\' && r + 1 < n) {
            const char c = s[r + 1];
            if (c == 'n') { s[w++] = '\n'; ++r; continue; }
            if (c == 't') { s[w++] = '\t'; ++r; continue; }
            if (c == 'r') { s[w++] = '\r'; ++r; continue; }
            if (c == '\\') { s[w++] = '\\'; ++r; continue; }
        }
        s[w++] = s[r];
    }
    return w;
}

static uint32_t argmax(const float *v, uint32_t n) {
    uint32_t best = 0;
    for (uint32_t i = 1; i < n; ++i) if (v[i] > v[best]) best = i;
    return best;
}

static void usage(FILE *out) {
    fprintf(out,
        "redlite-engine 0.3.0.dev21\n\n"
        "Usage:\n"
        "  redlite-engine info MODEL [--context N] [--cache-mib N]\n"
        "  redlite-engine tokenize MODEL --text \"...\" [--no-special] [--chat]\n"
        "  redlite-engine tokenize MODEL --file CORPUS [--no-special]   (one input per line, \\n \\t \\\\ escapes; one id list per line)\n"
        "  redlite-engine parity MODEL --tokens a,b,c [--context N] [--cache-mib N] [--threads N] [--layers] [--repeat N]\n"
        "      --repeat N: replay the sequence N times with a reset in between (a warm cache exercises the dev21 GPU-routed decode)\n"
        "  redlite-engine logits MODEL --tokens a,b,c --backend cpu|gpu [--out DUMP] [--dump-last | --dump-from N] [--batch N] [--context N] [--cache-mib N]\n"
        "      --batch N: ingest the tokens before --dump-from with the batched Metal prefill (chunks of N) instead of token by token\n"
        "  redlite-engine prefill MODEL --tokens a,b,c [--batch N] [--cpu] [--context N] [--cache-mib N]\n"
        "      batched Metal prefill vs token-by-token Metal steps (and the CPU oracle with --cpu): last-token layer outputs, router ids, logits\n"
        "      DUMP per token (f32): [hidden] embed, [layers][hidden] outputs, [hidden] final norm, [vocab] logits\n"
        "  redlite-engine kernel-selftest\n"
        "      model-free: decode GEMV kernels (block and sub-block), early-out guard, rl_copy_f32, rl_route and decode attention vs the CPU reference\n");
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
    if (argc >= 2 && (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0)) { usage(stdout); return 0; }
#ifdef __APPLE__
    if (argc == 2 && strcmp(argv[1], "kernel-selftest") == 0) {
        char report[1024] = {0}, err[512] = {0};
        if (!rl_metal_kernel_selftest(report, sizeof(report), err, sizeof(err))) { fprintf(stderr, "kernel self-test FAILED: %s\n", err); return 1; }
        printf("%s\nENGINE KERNEL SELFTEST: OK\n", report);
        return 0;
    }
#endif
    if (argc < 3) { usage(stderr); return 2; }
    const char *cmd = argv[1];
    const char *model = argv[2];
    rl_engine_config cfg;
    rl_engine_config_default(&cfg);
    /* long-position parity (dev26) needs prompts well beyond 4096 ids */
    enum { RL_CLI_MAX_TOKENS = 65536 };
    static uint32_t tokens[RL_CLI_MAX_TOKENS];
    uint32_t token_count = 0;
    const char *backend_name = "gpu";
    const char *out_path = NULL;
    const char *text = NULL, *corpus = NULL;
    int report_layers = 0, no_special = 0, chat = 0, dump_last = 0, with_cpu = 0;
    uint32_t dump_from = 0, batch = 0, repeat = 1;
    int router_layer = -1;
    for (int i = 3; i < argc; ++i) {
        if (strcmp(argv[i], "--layers") == 0) { report_layers = 1; continue; }
        if (strcmp(argv[i], "--no-special") == 0) { no_special = 1; continue; }
        if (strcmp(argv[i], "--chat") == 0) { chat = 1; continue; }
        if (strcmp(argv[i], "--dump-last") == 0) { dump_last = 1; continue; }
        if (strcmp(argv[i], "--cpu") == 0) { with_cpu = 1; continue; }
        if (i + 1 >= argc) { usage(stderr); return 2; }
        if (strcmp(argv[i], "--tokens") == 0) {
            token_count = parse_tokens(argv[++i], tokens, RL_CLI_MAX_TOKENS);
            if (!token_count) { fprintf(stderr, "invalid --tokens list\n"); return 2; }
        } else if (strcmp(argv[i], "--context") == 0) { if (!parse_u32(argv[++i], &cfg.context)) return 2; }
        else if (strcmp(argv[i], "--cache-mib") == 0) { uint32_t v; if (!parse_u32(argv[++i], &v)) return 2; cfg.cache_mib = v; }
        else if (strcmp(argv[i], "--threads") == 0) { uint32_t v; if (!parse_u32(argv[++i], &v)) return 2; cfg.cpu_threads = (int)v; }
        else if (strcmp(argv[i], "--backend") == 0) backend_name = argv[++i];
        else if (strcmp(argv[i], "--out") == 0) out_path = argv[++i];
        else if (strcmp(argv[i], "--text") == 0) text = argv[++i];
        else if (strcmp(argv[i], "--file") == 0) corpus = argv[++i];
        else if (strcmp(argv[i], "--dump-from") == 0) { if (!parse_u32(argv[++i], &dump_from)) return 2; }
        else if (strcmp(argv[i], "--batch") == 0) { if (!parse_u32(argv[++i], &batch)) return 2; }
        else if (strcmp(argv[i], "--repeat") == 0) { if (!parse_u32(argv[++i], &repeat) || !repeat) return 2; }
        else if (strcmp(argv[i], "--router-layer") == 0) { uint32_t v; if (!parse_u32(argv[++i], &v)) return 2; router_layer = (int)v; }
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

    if (strcmp(cmd, "tokenize") == 0) {
        if (!text && !corpus) { fprintf(stderr, "--text or --file is required\n"); return 2; }
        rl_gguf_model g;
        if (!rl_gguf_model_open(model, &g, error, sizeof(error))) { fprintf(stderr, "open failed: %s\n", error); return 1; }
        rl_tokenizer *tk = rl_tokenizer_create(&g, error, sizeof(error));
        if (!tk) { fprintf(stderr, "tokenizer failed: %s\n", error); rl_gguf_model_close(&g); return 1; }
        if (corpus) {
            FILE *cf = fopen(corpus, "rb");
            if (!cf) { fprintf(stderr, "cannot open %s\n", corpus); return 1; }
            static char line[65536];
            static uint32_t line_ids[16384];
            while (fgets(line, sizeof(line), cf)) {
                size_t len = strlen(line);
                while (len && (line[len - 1u] == '\n' || line[len - 1u] == '\r')) line[--len] = '\0';
                len = unescape_line(line, len);
                const int32_t n = rl_tokenizer_encode(tk, line, len, !no_special, line_ids, 16384u, error, sizeof(error));
                if (n < 0) { fprintf(stderr, "tokenize failed: %s\n", error); fclose(cf); return 1; }
                for (int32_t i = 0; i < n && i < 16384; ++i) printf("%s%u", i ? "," : "", line_ids[i]);
                printf("\n");
            }
            fclose(cf);
            rl_tokenizer_destroy(tk);
            rl_gguf_model_close(&g);
            return 0;
        }
        char prompt[65536];
        const char *input = text;
        if (chat) { if (!rl_tokenizer_chat_prompt(NULL, text, prompt, sizeof(prompt))) { fprintf(stderr, "prompt too long\n"); return 1; } input = prompt; }
        uint32_t ids[16384];
        const int32_t n = rl_tokenizer_encode(tk, input, strlen(input), !no_special, ids, 16384u, error, sizeof(error));
        if (n < 0) { fprintf(stderr, "tokenize failed: %s\n", error); return 1; }
        for (int32_t i = 0; i < n && i < 16384; ++i) printf("%s%u", i ? "," : "", ids[i]);
        printf("\n");
        for (int32_t i = 0; i < n && i < 16384; ++i) {
            char piece[512];
            const int32_t len = rl_tokenizer_decode(tk, ids[i], piece, sizeof(piece));
            printf("%u\t%.*s\n", ids[i], len > 0 ? len : 0, piece);
        }
        rl_tokenizer_destroy(tk);
        rl_gguf_model_close(&g);
        return 0;
    }

    if (!token_count) { fprintf(stderr, "--tokens is required\n"); return 2; }

    if (strcmp(cmd, "logits") == 0) {
        const int gpu = strcmp(backend_name, "gpu") == 0;
        cfg.enable_cpu = !gpu; cfg.enable_gpu = gpu;
        cfg.prefill_batch = batch;
        rl_engine *e = rl_engine_open(model, &cfg, error, sizeof(error));
        if (!e) { fprintf(stderr, "engine open failed: %s\n", error); return 1; }
        const rl_engine_info *in = rl_engine_info_get(e);
        float *logits = (float *)malloc((size_t)in->vocab * sizeof(float));
        FILE *out = out_path ? fopen(out_path, "wb") : NULL;
        if (out_path && !out) { fprintf(stderr, "cannot open %s\n", out_path); rl_engine_close(e); return 1; }
        uint32_t first = 0;
        if (gpu && batch > 1u && dump_from > 0u) {
            const uint32_t n = dump_from < token_count ? dump_from : token_count - 1u;
            rl_engine_step_stats st;
            const double t0 = rl_engine_now_ms_public();
            if (!rl_engine_prefill(e, RL_BACKEND_GPU, tokens, n, NULL, &st, error, sizeof(error))) {
                fprintf(stderr, "prefill failed: %s\n", error); rl_engine_close(e); return 1;
            }
            printf("prefill %u tokens in chunks of %u: %.1f ms (%.1f tok/s)\n", n, batch, rl_engine_now_ms_public() - t0, n * 1000.0 / (rl_engine_now_ms_public() - t0));
            printf("prefill split: dense wall %.1f (rec %.1f attn %.1f) dense GPU %.1f | router select %.1f | experts wall %.1f "
                   "[lru %.1f load %.1f commit %.1f gpu %.1f wait %.1f prefetch %.1f] | plans %u\n",
                st.recurrent_ms + st.attention_ms, st.recurrent_ms, st.attention_ms, st.gpu_ms, st.router_ms, st.routed_ms,
                st.prep_lru_ms, st.prep_load_ms, st.prep_commit_ms, st.routed_gpu_ms, st.expert_wait_ms, st.prefetch_ms, st.expert_plans);
            first = n;
        }
        for (uint32_t i = first; i < token_count; ++i) {
            rl_engine_step_stats st;
            if (!rl_engine_step(e, gpu ? RL_BACKEND_GPU : RL_BACKEND_CPU, tokens[i], logits, &st, error, sizeof(error))) {
                fprintf(stderr, "step %u failed: %s\n", i, error); rl_engine_close(e); return 1;
            }
            const uint32_t best = argmax(logits, in->vocab);
            printf("token[%u]=%u -> argmax=%u logit=%.6f total=%.1f ms (layers %.1f, output %.1f)\n",
                i, tokens[i], best, logits[best], st.total_ms, st.layers_ms, st.output_ms);
            if (router_layer >= 0 && i >= dump_from) {
                const uint32_t *rid = rl_engine_last_router_ids(e, gpu ? RL_BACKEND_GPU : RL_BACKEND_CPU, (uint32_t)router_layer);
                printf("  router[%d] topk:", router_layer);
                for (uint32_t k = 0; rid && k < in->top_k; ++k) printf(" %u", rid[k]);
                printf("\n");
            }
            if (out && (!dump_last || i + 1u == token_count) && i >= dump_from) {
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

    if (strcmp(cmd, "prefill") == 0) {
        cfg.enable_cpu = with_cpu; cfg.enable_gpu = 1;
        cfg.prefill_batch = batch ? batch : 32u;
        rl_engine *e = rl_engine_open(model, &cfg, error, sizeof(error));
        if (!e) { fprintf(stderr, "engine open failed: %s\n", error); return 1; }
        const rl_engine_info *in = rl_engine_info_get(e);
        const size_t hb = (size_t)in->hidden;
        float *seq_logits = (float *)malloc((size_t)in->vocab * sizeof(float));
        float *bat_logits = (float *)malloc((size_t)in->vocab * sizeof(float));
        float *cpu_logits = with_cpu ? (float *)malloc((size_t)in->vocab * sizeof(float)) : NULL;
        float *seq_layers = (float *)malloc((size_t)in->n_layer * hb * sizeof(float));
        float *seq_final = (float *)malloc(hb * sizeof(float));
        uint32_t *seq_ids = (uint32_t *)malloc((size_t)in->n_layer * in->top_k * sizeof(uint32_t));
        if (!seq_logits || !bat_logits || !seq_layers || !seq_final || !seq_ids || (with_cpu && !cpu_logits)) { fprintf(stderr, "allocation failed\n"); return 1; }
        printf("runtime              : native Qwen3-Next batched prefill parity (chunks of %u)\n", cfg.prefill_batch);
        print_info(in);
        printf("tokens               : %u\n", token_count);
        /* 1. token-by-token Metal */
        rl_engine_step_stats st;
        double t0 = rl_engine_now_ms_public();
        for (uint32_t i = 0; i < token_count; ++i) {
            if (!rl_engine_step(e, RL_BACKEND_GPU, tokens[i], i + 1u == token_count ? seq_logits : NULL, &st, error, sizeof(error))) {
                fprintf(stderr, "sequential step %u failed: %s\n", i, error); return 1;
            }
        }
        const double seq_ms = rl_engine_now_ms_public() - t0;
        for (uint32_t l = 0; l < in->n_layer; ++l) {
            memcpy(seq_layers + (size_t)l * hb, rl_engine_last_layer_output(e, RL_BACKEND_GPU, l), hb * sizeof(float));
            memcpy(seq_ids + (size_t)l * in->top_k, rl_engine_last_router_ids(e, RL_BACKEND_GPU, l), (size_t)in->top_k * sizeof(uint32_t));
        }
        memcpy(seq_final, rl_engine_last_final_norm(e, RL_BACKEND_GPU), hb * sizeof(float));
        /* 2. batched Metal prefill after a reset */
        if (!rl_engine_reset(e, RL_BACKEND_GPU, error, sizeof(error))) { fprintf(stderr, "reset failed: %s\n", error); return 1; }
        memset(&st, 0, sizeof(st));
        t0 = rl_engine_now_ms_public();
        if (!rl_engine_prefill(e, RL_BACKEND_GPU, tokens, token_count, bat_logits, &st, error, sizeof(error))) {
            fprintf(stderr, "batched prefill failed: %s\n", error); return 1;
        }
        const double bat_ms = rl_engine_now_ms_public() - t0;
        int ok = rl_engine_position(e, RL_BACKEND_GPU) == token_count;
        double worst_layer = 0.0;
        uint32_t id_mismatch = 0;
        for (uint32_t l = 0; l < in->n_layer; ++l) {
            const cmp c = compare(rl_engine_last_layer_output(e, RL_BACKEND_GPU, l), seq_layers + (size_t)l * hb, hb);
            if (c.max_abs > worst_layer) worst_layer = c.max_abs;
            const uint32_t *bid = rl_engine_last_router_ids(e, RL_BACKEND_GPU, l);
            uint32_t mism = 0;
            for (uint32_t k = 0; k < in->top_k; ++k) if (bid[k] != seq_ids[(size_t)l * in->top_k + k]) mism++;
            id_mismatch += mism;
            if (report_layers || mism) printf("layer %2u : max_abs=%.3e (rel %.3e) router_mismatch=%u\n", l, c.max_abs, c.max_rel, mism);
        }
        const cmp cf = compare(rl_engine_last_final_norm(e, RL_BACKEND_GPU), seq_final, hb);
        const cmp cl = compare(bat_logits, seq_logits, in->vocab);
        const uint32_t a_seq = argmax(seq_logits, in->vocab), a_bat = argmax(bat_logits, in->vocab);
        printf("sequential Metal     : %.1f ms (%.1f tok/s)\n", seq_ms, token_count * 1000.0 / seq_ms);
        printf("batched Metal        : %.1f ms (%.1f tok/s), dense GPU %.1f ms, experts GPU %.1f ms, expert loads %.1f ms (summed over reads), expert plans %u\n",
            bat_ms, token_count * 1000.0 / bat_ms, st.gpu_ms, st.routed_gpu_ms, st.routed_load_ms, st.expert_plans);
        printf("batched wall split   : embed %.1f, dense recurrent %.1f + attention %.1f, router select %.1f, experts (prepare+GPU+wait) %.1f, output %.1f ms\n",
            st.embed_ms, st.recurrent_ms, st.attention_ms, st.router_ms, st.routed_ms, st.output_ms);
        printf("expert phase split   : LRU reserve %.1f, miss loads (wall) %.1f, commit %.1f, GPU commit+wait %.1f ms\n",
            st.prep_lru_ms, st.prep_load_ms, st.prep_commit_ms, st.expert_wait_ms);
        printf("worst layer abs      : %.6g\n", worst_layer);
        printf("final norm abs       : %.6g\n", cf.max_abs);
        printf("logits abs           : %.6g (argmax batched=%u sequential=%u)\n", cl.max_abs, a_bat, a_seq);
        printf("router id mismatches : %u\n", id_mismatch);
        ok = ok && id_mismatch == 0 && a_seq == a_bat && cl.max_abs <= 1e-2 && worst_layer <= 1e-2;
        if (with_cpu) {
            memset(&st, 0, sizeof(st));
            t0 = rl_engine_now_ms_public();
            if (!rl_engine_prefill(e, RL_BACKEND_CPU, tokens, token_count, cpu_logits, &st, error, sizeof(error))) {
                fprintf(stderr, "CPU oracle failed: %s\n", error); return 1;
            }
            const cmp cc = compare(bat_logits, cpu_logits, in->vocab);
            const uint32_t a_cpu = argmax(cpu_logits, in->vocab);
            uint32_t cpu_mism = 0;
            for (uint32_t l = 0; l < in->n_layer; ++l) {
                const uint32_t *cid = rl_engine_last_router_ids(e, RL_BACKEND_CPU, l);
                const uint32_t *bid = rl_engine_last_router_ids(e, RL_BACKEND_GPU, l);
                for (uint32_t k = 0; k < in->top_k; ++k) if (cid[k] != bid[k]) cpu_mism++;
            }
            printf("CPU oracle           : %.1f ms; logits abs vs batched %.6g (argmax cpu=%u), router id mismatches %u\n",
                rl_engine_now_ms_public() - t0, cc.max_abs, a_cpu, cpu_mism);
            ok = ok && cpu_mism == 0 && a_cpu == a_bat && cc.max_abs <= 1e-2;
        }
        printf("BATCHED PREFILL PARITY: %s\n", ok ? "YES" : "NO");
        free(seq_logits); free(bat_logits); free(cpu_logits); free(seq_layers); free(seq_final); free(seq_ids);
        rl_engine_close(e);
        return ok ? 0 : 3;
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
    uint32_t spec_tokens = 0, spec_fallbacks = 0;
    for (uint32_t pass = 0; pass < repeat; ++pass) {
    if (pass) {
        if (!rl_engine_reset(e, RL_BACKEND_GPU, error, sizeof(error)) || !rl_engine_reset(e, RL_BACKEND_CPU, error, sizeof(error))) {
            fprintf(stderr, "reset failed: %s\n", error); all_ok = 0; break;
        }
        printf("-- pass %u (after reset; expert cache warm) --\n", pass + 1u);
    }
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
        if (gs.speculative || gs.speculative_fallback) printf("  path: %s\n", gs.speculative ? "GPU-routed (speculative)" : "speculative attempt fell back to synchronous");
        fflush(stdout);
        spec_tokens += gs.speculative; spec_fallbacks += gs.speculative_fallback;
        if (!token_ok) all_ok = 0;
    }
    }
    printf("worst layer abs       : %.6g\n", worst_layer);
    printf("worst logits abs      : %.6g\n", worst_logit);
    printf("GPU-routed tokens     : %u speculative, %u per-layer early-outs, %u synchronous\n", spec_tokens, spec_fallbacks, token_count * repeat - spec_tokens);
    printf("MULTI-TOKEN ENGINE PARITY: %s\n", all_ok ? "YES" : "NO");
    free(cpu_logits); free(gpu_logits);
    rl_engine_close(e);
    return all_ok ? 0 : 3;
}
