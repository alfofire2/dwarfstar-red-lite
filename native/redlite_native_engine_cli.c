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
#include "redlite_native_quant_cpu.h"
#include "redlite_native_iq2_xxs.h"
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

/* dev65: --tokens @FILE reads the comma-separated ids from a file (long contexts exceed the argument size limit) */
static char *read_text_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    char *buf = NULL;
    size_t len = 0, cap = 0;
    for (;;) {
        if (len + 65536u + 1u > cap) {
            cap = (len + 65536u + 1u) * 2u;
            char *nb = (char *)realloc(buf, cap);
            if (!nb) { free(buf); fclose(f); return NULL; }
            buf = nb;
        }
        const size_t got = fread(buf + len, 1, 65536u, f);
        len += got;
        if (got < 65536u) break;
    }
    fclose(f);
    while (len && (buf[len - 1] == '\n' || buf[len - 1] == '\r' || buf[len - 1] == ' ')) --len;
    buf[len] = '\0';
    return buf;
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
        "redlite-engine 0.4.0\n\n"
        "Usage:\n"
        "  redlite-engine info MODEL [--context N] [--cache-mib N]\n"
        "  redlite-engine decode-bench MODEL --tokens a,b,c --start-position N --context C [--cache-mib full] [--fill-state FILE]\n"
        "      dev65: decode ms/token at position N without a prefill; --fill-state reads rl_engine_state_bytes(N) bytes into the state\n"
        "  redlite-engine tokenize MODEL --text \"...\" [--no-special] [--chat]\n"
        "  redlite-engine perplexity MODEL --tokens IDS [--context 512] [--cache-mib N]   (dev46: engine PPL, second half of each chunk)\n"
        "  redlite-engine tokenize MODEL --file CORPUS [--no-special]   (one input per line, \\n \\t \\\\ escapes; one id list per line)\n"
        "  redlite-engine parity MODEL --tokens a,b,c [--context N] [--cache-mib N] [--threads N] [--layers] [--repeat N]\n"
        "      --repeat N: replay the sequence N times with a reset in between (a warm cache exercises the dev21 GPU-routed decode)\n"
        "  redlite-engine logits MODEL --tokens a,b,c --backend cpu|gpu [--out DUMP] [--dump-last | --dump-from N] [--batch N] [--context N] [--cache-mib N]\n"
        "      --batch N: ingest the tokens before --dump-from with the batched Metal prefill (chunks of N) instead of token by token\n"
        "  redlite-engine prefill MODEL --tokens a,b,c [--batch N] [--cpu] [--context N] [--cache-mib N]\n"
        "      batched Metal prefill vs token-by-token Metal steps (and the CPU oracle with --cpu): last-token layer outputs, router ids, logits\n"
        "      DUMP per token (f32): [hidden] embed, [layers][hidden] outputs, [hidden] final norm, [vocab] logits\n"
        "  redlite-engine dequant MODEL --tensor NAME [--row-first N] [--rows N] --out FILE\n"
        "      development: rows of a tensor (flattened to [rows][ne0]) dequantized to f32 by the CPU reference, for a bitwise\n"
        "      comparison with llama.cpp (redlite-ref-llama MODEL dequant); no engine is opened\n"
        "  redlite-engine pair MODEL --tokens a,b,c,... [--context N] (--cache-mib full)\n"
        "      dev56: two sequences decoded together (rl_engine_step_pair) vs each one alone: sequence A = the tokens,\n"
        "      B = the tokens reversed, B two positions ahead; logits and argmax of every pair step and of the steps after\n"
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
    if (argc == 2 && strcmp(argv[1], "kernel-bench") == 0) {
        char report[4096] = {0}, err[512] = {0};
        if (!rl_metal_kernel_bench(report, sizeof(report), err, sizeof(err))) { fprintf(stderr, "kernel bench failed: %s\n", err); return 1; }
        printf("%s", report);
        return 0;
    }
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
    enum { RL_CLI_MAX_TOKENS = 524288 };   /* dev65: up to 512K ids (2 MiB static), for 262K-position checks */
    static uint32_t tokens[RL_CLI_MAX_TOKENS];
    uint32_t token_count = 0;
    const char *backend_name = "gpu";
    const char *out_path = NULL;
    const char *text = NULL, *corpus = NULL, *tensor_name = NULL;
    uint32_t row_first = 0, row_count = 16;
    int report_layers = 0, no_special = 0, chat = 0, dump_last = 0, with_cpu = 0;
    uint32_t dump_from = 0, batch = 0, repeat = 1, start_position = 0;
    const char *fill_state = NULL;
    int router_layer = -1;
    for (int i = 3; i < argc; ++i) {
        if (strcmp(argv[i], "--layers") == 0) { report_layers = 1; continue; }
        if (strcmp(argv[i], "--no-special") == 0) { no_special = 1; continue; }
        if (strcmp(argv[i], "--chat") == 0) { chat = 1; continue; }
        if (strcmp(argv[i], "--dump-last") == 0) { dump_last = 1; continue; }
        if (strcmp(argv[i], "--cpu") == 0) { with_cpu = 1; continue; }
        if (i + 1 >= argc) { usage(stderr); return 2; }
        if (strcmp(argv[i], "--tokens") == 0) {
            const char *arg = argv[++i];
            char *file_text = arg[0] == '@' ? read_text_file(arg + 1) : NULL;
            if (arg[0] == '@' && !file_text) { fprintf(stderr, "cannot read %s\n", arg + 1); return 2; }
            token_count = parse_tokens(file_text ? file_text : arg, tokens, RL_CLI_MAX_TOKENS);
            free(file_text);
            if (!token_count) { fprintf(stderr, "invalid --tokens list\n"); return 2; }
        } else if (strcmp(argv[i], "--context") == 0) { if (!parse_u32(argv[++i], &cfg.context)) return 2; }
        else if (strcmp(argv[i], "--cache-mib") == 0) { if (!rl_engine_parse_cache_mib(argv[++i], &cfg.cache_mib)) return 2; }
        else if (strcmp(argv[i], "--threads") == 0) { uint32_t v; if (!parse_u32(argv[++i], &v)) return 2; cfg.cpu_threads = (int)v; }
        else if (strcmp(argv[i], "--backend") == 0) backend_name = argv[++i];
        else if (strcmp(argv[i], "--out") == 0) out_path = argv[++i];
        else if (strcmp(argv[i], "--text") == 0) text = argv[++i];
        else if (strcmp(argv[i], "--file") == 0) corpus = argv[++i];
        else if (strcmp(argv[i], "--tensor") == 0) tensor_name = argv[++i];
        else if (strcmp(argv[i], "--row-first") == 0) { if (!parse_u32(argv[++i], &row_first)) return 2; }
        else if (strcmp(argv[i], "--rows") == 0) { if (!parse_u32(argv[++i], &row_count) || !row_count) return 2; }
        else if (strcmp(argv[i], "--dump-from") == 0) { if (!parse_u32(argv[++i], &dump_from)) return 2; }
        else if (strcmp(argv[i], "--start-position") == 0) { if (!parse_u32(argv[++i], &start_position)) return 2; }
        else if (strcmp(argv[i], "--fill-state") == 0) fill_state = argv[++i];
        else if (strcmp(argv[i], "--batch") == 0) { if (!parse_u32(argv[++i], &batch)) return 2; }
        else if (strcmp(argv[i], "--repeat") == 0) { if (!parse_u32(argv[++i], &repeat) || !repeat) return 2; }
        else if (strcmp(argv[i], "--router-layer") == 0) { uint32_t v; if (!parse_u32(argv[++i], &v)) return 2; router_layer = (int)v; }
        else { fprintf(stderr, "unknown option %s\n", argv[i]); return 2; }
    }
    char error[512] = {0};

    if (strcmp(cmd, "dequant") == 0) {
        if (!tensor_name || !out_path) { usage(stderr); return 2; }
        rl_gguf_model g;
        if (!rl_gguf_model_open(model, &g, error, sizeof(error))) { fprintf(stderr, "GGUF open failed: %s\n", error); return 1; }
        if (!rl_gguf_model_map(&g, error, sizeof(error))) { fprintf(stderr, "GGUF map failed: %s\n", error); rl_gguf_model_close(&g); return 1; }
        const rl_gguf_tensor *t = rl_gguf_find(&g, tensor_name);
        int rc = 1;
        if (!t) fprintf(stderr, "tensor %s not found\n", tensor_name);
        else {
            uint64_t rows = 1;
            for (uint32_t d = 1; d < t->n_dims; ++d) rows *= t->shape[d];
            const uint32_t ncols = (uint32_t)t->shape[0];
            const size_t rb = rl_gguf_row_bytes(t->ggml_type, ncols);
            const uint8_t *data = rl_gguf_tensor_data(&g, t);
            float *buf = (float *)malloc((size_t)ncols * sizeof(float));
            uint8_t grid[RL_IQ2_XXS_GRID_COUNT];
            char gerr[128];
            const uint8_t *gp = t->ggml_type == 16u && rl_native_iq2_xxs_build_grid(grid, gerr, sizeof(gerr)) ? grid : NULL;
            FILE *out = fopen(out_path, "wb");
            if (!rb || !data || !buf || !out || (uint64_t)row_first + row_count > rows) fprintf(stderr, "cannot dequantize %s (type %s)\n", tensor_name, rl_gguf_type_name(t->ggml_type));
            else {
                rc = 0;
                for (uint32_t r = row_first; r < row_first + row_count && !rc; ++r) {
                    if (!rl_quant_dequant_row(t->ggml_type, data + (size_t)r * rb, ncols, gp, buf)) { fprintf(stderr, "type %s has no CPU dequantizer\n", rl_gguf_type_name(t->ggml_type)); rc = 1; }
                    else fwrite(buf, sizeof(float), ncols, out);
                }
                if (!rc) printf("dequantized %s (%s) rows %u..%u of %llu, %u columns\n", tensor_name, rl_gguf_type_name(t->ggml_type),
                    row_first, row_first + row_count - 1u, (unsigned long long)rows, ncols);
            }
            if (out) fclose(out);
            free(buf);
        }
        rl_gguf_model_close(&g);
        return rc;
    }

    if (strcmp(cmd, "info") == 0) {
        cfg.enable_cpu = 0; cfg.enable_gpu = 0;
        rl_engine *e = rl_engine_open(model, &cfg, error, sizeof(error));
        if (!e) { fprintf(stderr, "engine open failed: %s\n", error); return 1; }
        printf("runtime              : native Qwen3-Next persistent engine\n");
        print_info(rl_engine_info_get(e));
        printf("full residency cache : %llu MiB (--cache-mib full)\n", (unsigned long long)rl_engine_full_residency_mib(e));
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
        if (n > 16384) fprintf(stderr, "warning: the text has %d ids; only the first 16384 are printed\n", n);   /* dev65 */
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

    if (strcmp(cmd, "verify-check") == 0) {
        /* dev72: rl_engine_verify2 of the last two --tokens against plain steps, on the GPU backend (full residency or a
         * bounded cache): both rows' logits within 1e-3 of the steps', same argmax */
        if (token_count < 3u) { fprintf(stderr, "verify-check needs at least 3 tokens\n"); return 2; }
        cfg.enable_cpu = 0; cfg.enable_gpu = 1;
        rl_engine *e = rl_engine_open(model, &cfg, error, sizeof(error));
        if (!e) { fprintf(stderr, "engine open failed: %s\n", error); return 1; }
        const rl_engine_info *in = rl_engine_info_get(e);
        float *ref0 = (float *)malloc((size_t)in->vocab * sizeof(float)), *ref1 = (float *)malloc((size_t)in->vocab * sizeof(float));
        float *got0 = (float *)malloc((size_t)in->vocab * sizeof(float)), *got1 = (float *)malloc((size_t)in->vocab * sizeof(float));
        rl_engine_step_stats st;
        const uint32_t n = token_count;
        for (uint32_t i = 0; i < n; ++i)
            if (!rl_engine_step(e, RL_BACKEND_GPU, tokens[i], i == n - 2u ? ref0 : i == n - 1u ? ref1 : NULL, &st, error, sizeof(error))) { fprintf(stderr, "step: %s\n", error); return 1; }
        if (!rl_engine_reset(e, RL_BACKEND_GPU, error, sizeof(error))) { fprintf(stderr, "%s\n", error); return 1; }
        for (uint32_t i = 0; i + 2u < n; ++i)
            if (!rl_engine_step(e, RL_BACKEND_GPU, tokens[i], NULL, &st, error, sizeof(error))) { fprintf(stderr, "step: %s\n", error); return 1; }
        if (!rl_engine_verify2(e, tokens[n - 2u], tokens[n - 1u], got0, got1, error, sizeof(error)) ||
            !rl_engine_verify_commit(e, 1, error, sizeof(error))) { fprintf(stderr, "verify: %s\n", error); return 1; }
        double d0 = 0.0, d1 = 0.0;
        for (uint32_t v = 0; v < in->vocab; ++v) { d0 = fmax(d0, fabs((double)got0[v] - ref0[v])); d1 = fmax(d1, fabs((double)got1[v] - ref1[v])); }
        uint32_t a0 = 0, a1 = 0, r0 = 0, r1 = 0;
        for (uint32_t v = 1; v < in->vocab; ++v) { if (got0[v] > got0[a0]) a0 = v; if (got1[v] > got1[a1]) a1 = v; if (ref0[v] > ref0[r0]) r0 = v; if (ref1[v] > ref1[r1]) r1 = v; }
        /* the GPU-routed and synchronous paths sum in different orders: ~1e-5, not bit-identical */
        const int ok = d0 <= 1e-3 && d1 <= 1e-3 && a0 == r0 && a1 == r1;
        printf("verify row 0 max |d| %.3e, row 1 max |d| %.3e\nVERIFY CHECK: %s\n", d0, d1, ok ? "YES" : "NO");
        free(ref0); free(ref1); free(got0); free(got1);
        rl_engine_close(e);
        return ok ? 0 : 1;
    }

    if (strcmp(cmd, "decode-bench") == 0) {
        /* dev65: decode speed at a long context without the prefill: the GPU backend jumps to --start-position and
         * decodes the --tokens one at a time; prints the median ms per token after two warm-up tokens */
        cfg.enable_cpu = 0; cfg.enable_gpu = 1;
        rl_engine *e = rl_engine_open(model, &cfg, error, sizeof(error));
        if (!e) { fprintf(stderr, "engine open failed: %s\n", error); return 1; }
        if (fill_state) {   /* real data in every state buffer: rl_engine_state_bytes(position) bytes, e.g. from a FIFO */
            FILE *sf = fopen(fill_state, "rb");
            if (!sf || !rl_engine_state_read(e, RL_BACKEND_GPU, sf, start_position, error, sizeof(error))) {
                fprintf(stderr, "state fill failed: %s\n", sf ? error : "cannot open"); if (sf) fclose(sf); rl_engine_close(e); return 1;
            }
            fclose(sf);
        } else if (!rl_engine_bench_set_position(e, RL_BACKEND_GPU, start_position, error, sizeof(error))) { fprintf(stderr, "%s\n", error); rl_engine_close(e); return 1; }
        double *ms = (double *)calloc(token_count, sizeof(double));
        uint32_t routed = 0;
        for (uint32_t i = 0; i < token_count; ++i) {
            rl_engine_step_stats st;
            const double t0 = rl_engine_now_ms_public();
            if (!rl_engine_step(e, RL_BACKEND_GPU, tokens[i], NULL, &st, error, sizeof(error))) { fprintf(stderr, "step failed: %s\n", error); return 1; }
            ms[i] = rl_engine_now_ms_public() - t0;
            routed += st.speculative ? 1u : 0u;
        }
        const uint32_t n = token_count > 2u ? token_count - 2u : token_count;
        double *v = ms + (token_count - n);
        for (uint32_t i = 0; i < n; ++i) for (uint32_t j = i + 1u; j < n; ++j) if (v[j] < v[i]) { const double t = v[i]; v[i] = v[j]; v[j] = t; }
        printf("decode-bench: position %u, context %u, %u tokens, median %.2f ms/token (%.1f tok/s), %u GPU-routed\n",
               start_position, cfg.context, n, v[n / 2u], 1000.0 / v[n / 2u], routed);
        free(ms);
        rl_engine_close(e);
        return 0;
    }

    if (strcmp(cmd, "perplexity") == 0) {
        /* dev46: perplexity of the engine itself (token by token on the Metal step path, so runtime options such as
         * RL_ROUTE_CACHE_BIAS apply): chunks of --context ids, each from a reset state, scoring the second half */
        cfg.enable_cpu = 0; cfg.enable_gpu = 1;
        const uint32_t ctx = cfg.context ? cfg.context : 512u;
        cfg.context = ctx;
        rl_engine *e = rl_engine_open(model, &cfg, error, sizeof(error));
        if (!e) { fprintf(stderr, "engine open failed: %s\n", error); return 1; }
        const rl_engine_info *in = rl_engine_info_get(e);
        float *logits = (float *)malloc((size_t)in->vocab * sizeof(float));
        double nll = 0.0; uint64_t scored = 0, hits = 0, misses = 0;
        const uint32_t chunks = token_count / ctx;
        if (!logits || !chunks) { fprintf(stderr, "need at least --context ids\n"); rl_engine_close(e); return 2; }
        for (uint32_t c = 0; c < chunks; ++c) {
            const uint32_t *t = tokens + (size_t)c * ctx;
            if (!rl_engine_reset(e, RL_BACKEND_GPU, error, sizeof(error))) { fprintf(stderr, "%s\n", error); return 1; }
            for (uint32_t i = 0; i + 1u < ctx; ++i) {
                rl_engine_step_stats st;
                if (!rl_engine_step(e, RL_BACKEND_GPU, t[i], logits, &st, error, sizeof(error))) { fprintf(stderr, "step failed: %s\n", error); return 1; }
                hits = st.cache_hits; misses = st.cache_misses;
                if (i + 1u < ctx / 2u) continue;
                double mx = logits[0];
                for (uint32_t v = 1; v < in->vocab; ++v) if (logits[v] > mx) mx = logits[v];
                double se = 0.0;
                for (uint32_t v = 0; v < in->vocab; ++v) se += exp((double)logits[v] - mx);
                nll += mx + log(se) - (double)logits[t[i + 1u]];
                scored++;
            }
        }
        printf("perplexity: %u chunks x %u ids, %llu scored, cache hits %llu misses %llu\n", chunks, ctx,
               (unsigned long long)scored, (unsigned long long)hits, (unsigned long long)misses);
        printf("PPL = %.4f\n", exp(nll / (double)scored));
        free(logits);
        rl_engine_close(e);
        return 0;
    }

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
        const uint32_t chunk = rl_engine_prefill_batch(e);   /* --batch, or the engine default (dev30: 2048 with full residency) */
        if (gpu && chunk > 1u && dump_from > 0u) {
            const uint32_t n = dump_from < token_count ? dump_from : token_count - 1u;
            rl_engine_step_stats st;
            const double t0 = rl_engine_now_ms_public();
            if (!rl_engine_prefill(e, RL_BACKEND_GPU, tokens, n, NULL, &st, error, sizeof(error))) {
                fprintf(stderr, "prefill failed: %s\n", error); rl_engine_close(e); return 1;
            }
            printf("prefill %u tokens in chunks of %u: %.1f ms (%.1f tok/s)\n", n, chunk, rl_engine_now_ms_public() - t0, n * 1000.0 / (rl_engine_now_ms_public() - t0));
            printf("prefill split: dense wall %.1f (rec %.1f attn %.1f) dense GPU %.1f | router select %.1f | experts wall %.1f "
                   "[lru %.1f load %.1f commit %.1f gpu %.1f wait %.1f prefetch %.1f] | plans %u | expert loads %llu (%.0f MiB read)\n",
                st.recurrent_ms + st.attention_ms, st.recurrent_ms, st.attention_ms, st.gpu_ms, st.router_ms, st.routed_ms,
                st.prep_lru_ms, st.prep_load_ms, st.prep_commit_ms, st.routed_gpu_ms, st.expert_wait_ms, st.prefetch_ms, st.expert_plans,
                (unsigned long long)st.expert_loads, (double)st.ssd_bytes / (1024.0 * 1024.0));
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

    if (strcmp(cmd, "pair") == 0) {
        if (token_count < 4u) { fprintf(stderr, "pair needs --tokens with at least 4 ids\n"); return 2; }
        cfg.enable_cpu = 0; cfg.enable_gpu = 1;
        rl_engine *e = rl_engine_open(model, &cfg, error, sizeof(error));
        if (!e) { fprintf(stderr, "engine open failed: %s\n", error); return 1; }
        const rl_engine_info *in = rl_engine_info_get(e);
        const uint32_t n = token_count, h = 2u, V = in->vocab;
        uint32_t *rev = (uint32_t *)malloc(n * sizeof(uint32_t));
        float *ref_a = (float *)malloc((size_t)n * V * sizeof(float)), *ref_b = (float *)malloc((size_t)n * V * sizeof(float));
        float *la = (float *)malloc((size_t)V * sizeof(float)), *lb = (float *)malloc((size_t)V * sizeof(float));
        rl_engine_step_stats st;
        int ok = rev && ref_a && ref_b && la && lb;
        for (uint32_t i = 0; ok && i < n; ++i) rev[i] = tokens[n - 1u - i];
        /* references: each sequence alone, ordinary steps in slot 0 */
        ok = ok && rl_engine_reset(e, RL_BACKEND_GPU, error, sizeof(error));
        for (uint32_t i = 0; ok && i < n; ++i) ok = rl_engine_step(e, RL_BACKEND_GPU, tokens[i], ref_a + (size_t)i * V, &st, error, sizeof(error));
        ok = ok && rl_engine_reset(e, RL_BACKEND_GPU, error, sizeof(error));
        const double s0 = rl_engine_now_ms_public();
        for (uint32_t i = 0; ok && i < n; ++i) ok = rl_engine_step(e, RL_BACKEND_GPU, rev[i], ref_b + (size_t)i * V, &st, error, sizeof(error));
        const double single_ms = (rl_engine_now_ms_public() - s0) / n;
        ok = ok && rl_engine_slots_enable(e, error, sizeof(error));
        /* B alone in slot 1 for h tokens, then pairs (A in slot 0 as row 0), then the rest of A alone in slot 0 */
        ok = ok && rl_engine_select_slot(e, 0) && rl_engine_reset(e, RL_BACKEND_GPU, error, sizeof(error));
        ok = ok && rl_engine_select_slot(e, 1) && rl_engine_reset(e, RL_BACKEND_GPU, error, sizeof(error));
        double worst = 0.0;
        uint32_t mism = 0, checked = 0;
#define RL_PAIR_CHECK(got, ref) do { const cmp c_ = compare((got), (ref), V); if (c_.max_abs > worst) worst = c_.max_abs; \
            mism += argmax((got), V) != argmax((ref), V); checked++; } while (0)
        for (uint32_t i = 0; ok && i < h; ++i) {
            ok = rl_engine_step(e, RL_BACKEND_GPU, rev[i], lb, &st, error, sizeof(error));
            if (ok) RL_PAIR_CHECK(lb, ref_b + (size_t)i * V);
        }
        ok = ok && rl_engine_select_slot(e, 0);
        const double p0 = rl_engine_now_ms_public();
        for (uint32_t i = 0; ok && i + h < n; ++i) {
            ok = rl_engine_step_pair(e, tokens[i], rev[i + h], la, lb, error, sizeof(error));
            if (ok) { RL_PAIR_CHECK(la, ref_a + (size_t)i * V); RL_PAIR_CHECK(lb, ref_b + (size_t)(i + h) * V); }
        }
        const double pair_ms = (rl_engine_now_ms_public() - p0) / (n - h);
        for (uint32_t i = n - h; ok && i < n; ++i) {
            ok = rl_engine_step(e, RL_BACKEND_GPU, tokens[i], la, &st, error, sizeof(error));
            if (ok) RL_PAIR_CHECK(la, ref_a + (size_t)i * V);
        }
#undef RL_PAIR_CHECK
        int rc = 1;
        if (!ok) fprintf(stderr, "pair failed: %s\n", error);
        else {
            printf("pair: %u logit rows checked, max abs diff %.3g, argmax mismatches %u\n", checked, worst, mism);
            printf("time (includes logits copies; not a benchmark): single step %.1f ms, pair step %.1f ms (%.2f x one step for 2 tokens)\n",
                   single_ms, pair_ms, pair_ms / single_ms);
            rc = mism ? 1 : 0;
            printf("%s\n", rc ? "FAIL" : "PASS");
        }
        free(rev); free(ref_a); free(ref_b); free(la); free(lb);
        rl_engine_close(e);
        return rc;
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
    if (getenv("RL_STEER_FILE")) {   /* dev52: parity with steering on (RL_STEER_LAYERS=A-B, RL_STEER_STRENGTH=S) */
        unsigned a = 16u, b = 31u; float s = 1.0f;
        if (getenv("RL_STEER_LAYERS")) sscanf(getenv("RL_STEER_LAYERS"), "%u-%u", &a, &b);
        if (getenv("RL_STEER_STRENGTH")) s = strtof(getenv("RL_STEER_STRENGTH"), NULL);
        if (!rl_engine_load_steering(e, getenv("RL_STEER_FILE"), a, b, s, error, sizeof(error))) { fprintf(stderr, "steering: %s\n", error); return 1; }
        printf("steering             : layers %u-%u, strength %g\n", a, b, (double)s);
    }
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
