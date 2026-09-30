#define _POSIX_C_SOURCE 200809L
#define _DARWIN_C_SOURCE 1

/*
 * redlite-server: minimal OpenAI-compatible HTTP server on the persistent native engine.
 *
 *   redlite-server MODEL [--host H] [--port P] [--context N] [--cache-mib N] [--batch N]
 *                  [--max-tokens N] [--temperature T] [--top-k K] [--top-p P] [--queue N]
 *                  [--no-reuse] [--cpu]
 *
 * POST /v1/chat/completions (stream true/false), GET /v1/models, GET /health.
 * Requests run one at a time in arrival order (FIFO queue in the server core). The engine
 * remembers the token ids its state holds (prompt + every generated token that was fed
 * back). When a new prompt extends exactly that sequence (the next turn of the same
 * conversation), only the new suffix is prefilled (dev29); otherwise the engine is reset
 * and the whole conversation (ChatML) is ingested with the batched prefill. The DeltaNet
 * state cannot be rolled back, so a partial match is never reused. Then it decodes like
 * redlite-generate.
 * Metal backend on macOS; --cpu (the only choice elsewhere) uses the CPU oracle.
 * Ctrl-C stops the current generation, answers the client and closes the engine.
 */

#include "redlite_native_engine.h"
#include "redlite_native_gguf_dir.h"
#include "redlite_native_sampler.h"
#include "redlite_native_server.h"
#include "redlite_native_tokenizer.h"

#include <errno.h>
#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static volatile sig_atomic_t g_stop = 0;
static void on_signal(int sig) { (void)sig; g_stop = 1; }

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

typedef struct {
    rl_engine *engine;
    const rl_engine_info *info;
    rl_tokenizer *tokenizer;
    rl_engine_backend backend;
    float *logits;
    int reuse;             /* dev29 prefix reuse enabled (--no-reuse clears it) */
    uint32_t *history;     /* token ids the engine state holds, in order (context entries) */
    uint32_t history_len;
} engine_ctx;

typedef struct {               /* wraps the server sink to time the first token */
    rl_server_emit_fn emit;
    void *sink;
    double first_ms;
} ttft_sink;

static int ttft_emit(void *user, const char *bytes, size_t len) {
    ttft_sink *t = (ttft_sink *)user;
    if (t->first_ms < 0.0) t->first_ms = now_ms();
    return t->emit(t->sink, bytes, len);
}

static int engine_generate(void *user, const rl_chat_request *req, rl_server_emit_fn emit, void *sink,
                           rl_server_result *result, int *status, char *error, size_t cap) {
    engine_ctx *c = (engine_ctx *)user;
    const double t_request = now_ms();
    char *prompt = rl_server_chatml(req);
    if (!prompt) { snprintf(error, cap, "out of memory"); return 0; }
    const int32_t needed = rl_tokenizer_encode(c->tokenizer, prompt, strlen(prompt), 1, NULL, 0, error, cap);
    if (needed <= 0) { free(prompt); if (needed == 0) { *status = 400; snprintf(error, cap, "empty prompt"); } return 0; }
    result->prompt_tokens = (uint32_t)needed;
    if ((uint64_t)needed + req->max_tokens > c->info->context) {
        free(prompt);
        *status = 400;
        snprintf(error, cap, "prompt (%d tokens) + max_tokens (%u) exceed the server context (%u); start the server with a larger --context",
            needed, req->max_tokens, c->info->context);
        return 0;
    }
    uint32_t *ids = (uint32_t *)malloc((size_t)needed * sizeof(uint32_t));
    if (!ids) { free(prompt); snprintf(error, cap, "out of memory"); return 0; }
    const int32_t encoded = rl_tokenizer_encode(c->tokenizer, prompt, strlen(prompt), 1, ids, (size_t)needed, error, cap);
    free(prompt);
    if (encoded != needed) { free(ids); if (!error[0]) snprintf(error, cap, "tokenization changed between passes"); return 0; }

    rl_sampler_params sp;
    rl_sampler_params_default(&sp);
    sp.temperature = req->temperature;
    sp.top_p = req->top_p;
    sp.top_k = (uint32_t)req->top_k;
    sp.min_p = req->min_p;
    if (req->has_seed) sp.seed = req->seed;
    rl_sampler sampler;
    if (!rl_sampler_init(&sampler, &sp, c->info->vocab)) { free(ids); snprintf(error, cap, "sampler allocation failed"); return 0; }

    /* dev29: keep the state when the prompt extends exactly the ids it holds */
    uint32_t reused = 0;
    if (c->reuse && rl_engine_position(c->engine, c->backend) == c->history_len)
        reused = rl_prefix_reuse(c->history, c->history_len, ids, (uint32_t)needed);
    c->history_len = 0;   /* invalid until this request's ids are in the state */
    int ok = reused ? 1 : rl_engine_reset(c->engine, c->backend, error, cap);
    rl_engine_step_stats st;
    memset(&st, 0, sizeof(st));
    const double t_prefill = now_ms();
    ok = ok && rl_engine_prefill(c->engine, c->backend, ids + reused, (uint32_t)needed - reused, c->logits, &st, error, cap);
    const double prefill_ms = now_ms() - t_prefill;
    if (ok) { memcpy(c->history, ids, (size_t)needed * sizeof(uint32_t)); c->history_len = (uint32_t)needed; }
    free(ids);
    result->cached_tokens = reused;

    ttft_sink ts = {emit, sink, -1.0};
    uint32_t generated = 0, gpu_routed = 0;
    const double t_gen = now_ms();
    while (ok) {
        const uint32_t next = rl_sampler_sample(&sampler, c->logits);
        if (rl_tokenizer_is_eog(c->tokenizer, next)) break;
        char piece[512];
        const int32_t n = rl_tokenizer_decode(c->tokenizer, next, piece, sizeof(piece));
        if (n < 0) { snprintf(error, cap, "token decode failed"); ok = 0; break; }
        generated++;
        if (!ttft_emit(&ts, piece, (size_t)n)) break;           /* client gone, Ctrl-C or stop sequence */
        if (generated >= req->max_tokens) { result->finish_length = 1; break; }
        if (!rl_engine_step(c->engine, c->backend, next, c->logits, &st, error, cap)) { ok = 0; break; }
        c->history[c->history_len++] = next;                     /* fed back: part of the state */
        gpu_routed += st.speculative;
    }
    const double gen_ms = now_ms() - t_gen;
    rl_sampler_free(&sampler);
    result->completion_tokens = generated;
    if (!ok) c->history_len = 0;   /* unknown state: the next request resets */
    if (ok) {
        fprintf(stderr, "[redlite-server] prefill %u tok (%u cached) %.0f ms (%.1f tok/s), ttft %.0f ms, decode %u tok %.0f ms (%.1f tok/s, %u GPU-routed)\n",
            (uint32_t)needed - reused, reused, prefill_ms, prefill_ms > 0 ? (needed - reused) * 1000.0 / prefill_ms : 0.0,
            ts.first_ms >= 0.0 ? ts.first_ms - t_request : 0.0,
            generated, gen_ms, generated > 1u ? (generated - 1u) * 1000.0 / gen_ms : 0.0, gpu_routed);
    }
    return ok;
}

static void usage(FILE *out) {
    fprintf(out,
        "redlite-server 0.4.0 - OpenAI-compatible HTTP server on the native Red Lite engine\n\n"
        "Usage: redlite-server MODEL [options]\n\n"
        "  --host H            bind address (default 127.0.0.1)\n"
        "  --port P            TCP port (default 8080; 0 = ephemeral)\n"
        "  --context N         KV cache positions per request (default 4096)\n"
        "  --cache-mib N|full  routed-expert cache in MiB (default 4096); full = every expert of the file (preloaded)\n"
        "  --batch N           prompt tokens per batched prefill chunk (default 2048 with every expert preloaded, else 512)\n"
        "  --max-tokens N      default max_tokens when a request omits it (default 256)\n"
        "  --temperature T     default temperature (default 0.7)\n"
        "  --top-k K           default top-k (default 40)\n"
        "  --top-p P           default top-p (default 0.95)\n"
        "  --min-p M           default min-p (default 0 = off)\n"
        "  --queue N           chat requests that may wait behind the running one (default 16; more get 503)\n"
        "  --no-reuse          reset the engine for every request (default: a prompt that extends the previous\n"
        "                      conversation exactly only prefills the new tokens)\n"
        "  --cpu               use the CPU oracle backend (slow; the only backend off macOS)\n\n"
        "Endpoints: POST /v1/chat/completions (stream true/false), GET /v1/models, GET /health\n");
}

static int parse_u32(const char *s, uint32_t *out) {
    if (!s || !*s) return 0;
    errno = 0;
    char *end = NULL;
    const unsigned long v = strtoul(s, &end, 10);
    if (errno || !end || *end || v > UINT32_MAX) return 0;
    *out = (uint32_t)v;
    return 1;
}

static int parse_f32(const char *s, float *out) {
    if (!s || !*s) return 0;
    char *end = NULL;
    const double v = strtod(s, &end);
    if (!end || *end || !isfinite(v)) return 0;
    *out = (float)v;
    return 1;
}

int main(int argc, char **argv) {
    if (argc < 2 || strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0) { usage(argc < 2 ? stderr : stdout); return argc < 2 ? 2 : 0; }
    const char *model = argv[1];
    rl_server_config scfg;
    rl_server_config_default(&scfg);
    rl_engine_config cfg;
    rl_engine_config_default(&cfg);
    int reuse = 1;
#ifdef __APPLE__
    int use_cpu = 0;
#else
    int use_cpu = 1;
#endif
    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--cpu") == 0) { use_cpu = 1; continue; }
        if (strcmp(argv[i], "--no-reuse") == 0) { reuse = 0; continue; }
        if (i + 1 >= argc) { usage(stderr); return 2; }
        const char *v = argv[++i];
        uint32_t u = 0;
        if (strcmp(argv[i - 1], "--host") == 0) scfg.host = v;
        else if (strcmp(argv[i - 1], "--port") == 0) { if (!parse_u32(v, &u) || u > 65535u) return 2; scfg.port = (uint16_t)u; }
        else if (strcmp(argv[i - 1], "--context") == 0) { if (!parse_u32(v, &cfg.context)) return 2; }
        else if (strcmp(argv[i - 1], "--cache-mib") == 0) { if (!rl_engine_parse_cache_mib(v, &cfg.cache_mib)) return 2; }
        else if (strcmp(argv[i - 1], "--batch") == 0) { if (!parse_u32(v, &cfg.prefill_batch)) return 2; }
        else if (strcmp(argv[i - 1], "--max-tokens") == 0) { if (!parse_u32(v, &scfg.default_max_tokens) || !scfg.default_max_tokens) return 2; }
        else if (strcmp(argv[i - 1], "--temperature") == 0) { if (!parse_f32(v, &scfg.default_temperature) || scfg.default_temperature < 0.0f) return 2; }
        else if (strcmp(argv[i - 1], "--top-k") == 0) { if (!parse_u32(v, &scfg.default_top_k)) return 2; }
        else if (strcmp(argv[i - 1], "--top-p") == 0) { if (!parse_f32(v, &scfg.default_top_p) || scfg.default_top_p <= 0.0f) return 2; }
        else if (strcmp(argv[i - 1], "--queue") == 0) { if (!parse_u32(v, &scfg.queue_max)) return 2; }
        else if (strcmp(argv[i - 1], "--min-p") == 0) { if (!parse_f32(v, &scfg.default_min_p) || scfg.default_min_p < 0.0f || scfg.default_min_p > 1.0f) return 2; }
        else { fprintf(stderr, "unknown option %s\n", argv[i - 1]); usage(stderr); return 2; }
    }
#ifndef __APPLE__
    if (!use_cpu) { fprintf(stderr, "the Metal backend requires macOS; use --cpu\n"); return 2; }
#endif

    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = on_signal;
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);

    char error[512] = {0};
    const double t_open = now_ms();
    cfg.enable_cpu = use_cpu;
    cfg.enable_gpu = !use_cpu;
    rl_engine *e = rl_engine_open(model, &cfg, error, sizeof(error));
    if (!e) { fprintf(stderr, "engine open failed: %s\n", error); return 1; }
    rl_gguf_model g;
    if (!rl_gguf_model_open(model, &g, error, sizeof(error))) { fprintf(stderr, "tokenizer model open failed: %s\n", error); rl_engine_close(e); return 1; }
    rl_tokenizer *tk = rl_tokenizer_create(&g, error, sizeof(error));
    if (!tk) { fprintf(stderr, "tokenizer failed: %s\n", error); rl_gguf_model_close(&g); rl_engine_close(e); return 1; }
    if (rl_tokenizer_lookup(tk, "<|im_start|>") < 0 || rl_tokenizer_lookup(tk, "<|im_end|>") < 0) {
        fprintf(stderr, "the model vocabulary has no ChatML tokens; redlite-server only speaks ChatML\n");
        rl_tokenizer_destroy(tk); rl_gguf_model_close(&g); rl_engine_close(e);
        return 1;
    }
    /* The GGUF template is not interpreted (no Jinja); it only confirms the ChatML format. */
    fprintf(stderr, "[redlite-server] chat template: %s\n", rl_tokenizer_chat_template_source(g.chat_template, NULL));
    double preload_ms = 0.0;
    const int preloaded = rl_engine_experts_preloaded(e, &preload_ms);
    fprintf(stderr, "[redlite-server] engine ready in %.1f s (%s backend, cache %llu MiB%s, context %u)\n",
        (now_ms() - t_open) / 1000.0, use_cpu ? "CPU oracle" : "Metal",
        (unsigned long long)(cfg.cache_mib == RL_ENGINE_CACHE_FULL ? rl_engine_full_residency_mib(e) : cfg.cache_mib),
        preloaded ? ", every expert preloaded" : "", rl_engine_info_get(e)->context);

    engine_ctx ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.engine = e;
    ctx.info = rl_engine_info_get(e);
    ctx.tokenizer = tk;
    ctx.backend = use_cpu ? RL_BACKEND_CPU : RL_BACKEND_GPU;
    ctx.logits = (float *)malloc((size_t)ctx.info->vocab * sizeof(float));
    ctx.history = (uint32_t *)malloc(((size_t)ctx.info->context + 1u) * sizeof(uint32_t));
    ctx.reuse = reuse;
    int rc = 1;
    if (!ctx.logits || !ctx.history) {
        fprintf(stderr, "logits allocation failed\n");
    } else {
        rl_server_backend backend = {&ctx, "qwen3-next-80b-a3b-redlite", engine_generate};
        if (rl_server_run(&scfg, &backend, &g_stop, error, sizeof(error))) rc = 0;
        else fprintf(stderr, "server failed: %s\n", error);
    }
    free(ctx.logits);
    free(ctx.history);
    rl_tokenizer_destroy(tk);
    rl_gguf_model_close(&g);
    rl_engine_close(e);
    if (!rc) fprintf(stderr, "[redlite-server] shut down cleanly\n");
    return rc;
}
