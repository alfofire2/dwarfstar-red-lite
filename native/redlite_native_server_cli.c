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
 * Requests run one at a time in arrival order (FIFO queue in the server core; dev56 --parallel 2: two at a time,
 * each with its own engine slot, their decode steps paired into one pass). The engine
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
#include "redlite_native_statecache.h"
#include "redlite_native_gguf_dir.h"
#include "redlite_native_sampler.h"
#include "redlite_native_server.h"
#include "redlite_native_tokenizer.h"

#include <errno.h>
#include <pthread.h>
#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef REDLITE_VERSION
#define REDLITE_VERSION "dev"   /* the build scripts pass the VERSION file */
#endif

static volatile sig_atomic_t g_stop = 0;
static void on_signal(int sig) { (void)sig; g_stop = 1; }

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

typedef struct {               /* dev56: one sequence state of the engine */
    float *logits, *logits1;
    uint32_t *history;         /* token ids the slot's state holds, in order (context entries) */
    uint32_t history_len;
    int busy;
} srv_slot;

typedef struct {
    rl_engine *engine;
    const rl_engine_info *info;
    rl_tokenizer *tokenizer;
    rl_engine_backend backend;
    int reuse;             /* dev29 prefix reuse enabled (--no-reuse clears it) */
    rl_statecache states;  /* dev43 --state-dir: disk checkpoints when the in-memory state does not apply */
    int speculate;         /* dev45 --mtp with every expert resident: MTP drafts + 2-row verify */
    uint32_t mtp_max_context;   /* dev55: no speculation for answers starting past this position (0 = no limit) */
    int steering;               /* dev55: --steer loaded */
    float steer_base;
    uint32_t steer_tokens;      /* steer only the first N generated tokens (0 = all) */
    int tool_format;            /* dev59: RL_TOOLS_JSON (Qwen3-Next Instruct) or RL_TOOLS_XML (Qwen3-Coder) */
    /* dev56 --parallel 2: two slots, one per worker. The engine runs one call at a time (engine_busy); a decode
     * step posts its token (want) and, when the other slot is decoding too, waits up to SRV_PAIR_WAIT_MS for it so
     * that both run as one rl_engine_step_pair. */
    srv_slot slot[2];
    uint32_t nslots;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    int engine_busy;
    int decoding[2], want[2], done[2], res[2];
    uint32_t tok[2];
    float *out[2];
    char *err[2];
    size_t errcap[2];
} engine_ctx;

#define SRV_PAIR_WAIT_MS 20   /* ponytail: fixed; a client that reads slowly costs its partner up to this per token */

/* dev55: steering strength for the next decode step (the batched prefill is never steered) */
static void srv_steer(engine_ctx *c, uint32_t generated) {
    if (c->steering) rl_engine_set_steering_strength(c->engine, c->steer_tokens && generated >= c->steer_tokens ? 0.0f : c->steer_base);
}

/* exclusive use of the engine with slot k selected (prefill, reset, verify cycles) */
static void eng_lock(engine_ctx *c, int k) {
    pthread_mutex_lock(&c->mu);
    while (c->engine_busy) pthread_cond_wait(&c->cv, &c->mu);
    c->engine_busy = 1;
    pthread_mutex_unlock(&c->mu);
    rl_engine_select_slot(c->engine, k);
}

static void eng_unlock(engine_ctx *c) {
    pthread_mutex_lock(&c->mu);
    c->engine_busy = 0;
    pthread_cond_broadcast(&c->cv);
    pthread_mutex_unlock(&c->mu);
}

static void set_decoding(engine_ctx *c, int k, int on) {
    pthread_mutex_lock(&c->mu);
    c->decoding[k] = on;
    pthread_cond_broadcast(&c->cv);
    pthread_mutex_unlock(&c->mu);
}

static int other_busy(engine_ctx *c, int k) {   /* the other slot serves a request */
    if (c->nslots < 2) return 0;
    pthread_mutex_lock(&c->mu);
    const int d = c->slot[1 - k].busy;
    pthread_mutex_unlock(&c->mu);
    return d;
}

/* One decode step of slot k: 0 = failed, 1 = alone, 2 = paired with the other slot's step.
 * ponytail: a pair shares one steering strength (that of the slot that runs it). */
static int eng_step(engine_ctx *c, int k, uint32_t token, float *logits, uint32_t generated, rl_engine_step_stats *st,
                    char *error, size_t cap) {
    const int o = 1 - k;
    struct timespec deadline = {0, 0};
    pthread_mutex_lock(&c->mu);
    c->want[k] = 1; c->tok[k] = token; c->out[k] = logits; c->err[k] = error; c->errcap[k] = cap; c->done[k] = 0;
    pthread_cond_broadcast(&c->cv);
    int timed_out = 0, waiting = 0, r = 0;
    for (;;) {
        if (c->done[k]) { r = c->res[k]; break; }
        if (c->engine_busy) { pthread_cond_wait(&c->cv, &c->mu); continue; }
        const int pair = c->nslots == 2 && c->want[o];
        if (!pair && c->nslots == 2 && c->decoding[o] && !timed_out) {
            if (!waiting) {   /* the wait starts when the engine is free (the partner may have just run a step) */
                clock_gettime(CLOCK_REALTIME, &deadline);
                deadline.tv_nsec += SRV_PAIR_WAIT_MS * 1000000L;
                if (deadline.tv_nsec >= 1000000000L) { deadline.tv_sec++; deadline.tv_nsec -= 1000000000L; }
                waiting = 1;
            }
            timed_out = pthread_cond_timedwait(&c->cv, &c->mu, &deadline) == ETIMEDOUT;
            continue;
        }
        c->engine_busy = 1;
        pthread_mutex_unlock(&c->mu);
        rl_engine_select_slot(c->engine, k);
        srv_steer(c, generated);
        const int ok = pair ? rl_engine_step_pair(c->engine, token, c->tok[o], logits, c->out[o], error, cap)
                            : rl_engine_step(c->engine, c->backend, token, logits, st, error, cap);
        pthread_mutex_lock(&c->mu);
        c->engine_busy = 0;
        c->want[k] = 0; c->done[k] = 1; c->res[k] = ok ? 1 + pair : 0;
        if (pair) {
            c->want[o] = 0; c->done[o] = 1; c->res[o] = c->res[k];
            if (!ok) snprintf(c->err[o], c->errcap[o], "%s", error);
        }
        pthread_cond_broadcast(&c->cv);
    }
    pthread_mutex_unlock(&c->mu);
    return r;
}

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
    char *prompt = rl_server_prompt(req, c->tool_format);
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
    sp.presence_penalty = req->presence_penalty;
    sp.frequency_penalty = req->frequency_penalty;
    if (req->has_seed) sp.seed = req->seed;
    rl_sampler sampler;
    if (!rl_sampler_init(&sampler, &sp, c->info->vocab)) { free(ids); snprintf(error, cap, "sampler allocation failed"); return 0; }

    /* dev56: a free slot, preferably the one whose state this prompt extends (one worker per slot: one is free) */
    pthread_mutex_lock(&c->mu);
    int k = -1;
    uint32_t best = 0;
    for (int j = 0; j < (int)c->nslots; ++j) {
        if (c->slot[j].busy) continue;
        const uint32_t m = c->reuse ? rl_prefix_reuse(c->slot[j].history, c->slot[j].history_len, ids, (uint32_t)needed) : 0u;
        if (k < 0 || m > best) { k = j; best = m; }
    }
    if (k >= 0) c->slot[k].busy = 1;
    pthread_mutex_unlock(&c->mu);
    if (k < 0) { free(ids); rl_sampler_free(&sampler); snprintf(error, cap, "no free engine slot"); return 0; }
    srv_slot *S = &c->slot[k];

    /* dev29: keep the state when the prompt extends exactly the ids it holds */
    eng_lock(c, k);
    uint32_t reused = 0;
    if (c->reuse && rl_engine_position(c->engine, c->backend) == S->history_len)
        reused = rl_prefix_reuse(S->history, S->history_len, ids, (uint32_t)needed);
    if (!reused && S->history_len && getenv("RL_SERVER_DEBUG_REUSE")) {   /* dev59b: where the new prompt diverges */
        uint32_t at = 0;
        while (at < S->history_len && at < (uint32_t)needed && S->history[at] == ids[at]) at++;
        char held[160] = {0}, got[160] = {0};
        for (uint32_t i = at; i < at + 12u && i < S->history_len; ++i) {
            char piece[64]; const int32_t n = rl_tokenizer_decode(c->tokenizer, S->history[i], piece, sizeof(piece) - 1u);
            if (n > 0 && strlen(held) + (size_t)n < sizeof(held)) strncat(held, piece, (size_t)n);
        }
        for (uint32_t i = at; i < at + 12u && i < (uint32_t)needed; ++i) {
            char piece[64]; const int32_t n = rl_tokenizer_decode(c->tokenizer, ids[i], piece, sizeof(piece) - 1u);
            if (n > 0 && strlen(got) + (size_t)n < sizeof(got)) strncat(got, piece, (size_t)n);
        }
        fprintf(stderr, "[redlite-server] reuse: prompt diverges at token %u of %u held: held \"%s\" | new \"%s\"\n",
                at, S->history_len, held, got);
    }
    S->history_len = 0;   /* invalid until this request's ids are in the state */
    rl_engine_step_stats st;
    memset(&st, 0, sizeof(st));
    const double t_prefill = now_ms();
    int ok;
    if (reused) {
        ok = rl_engine_prefill(c->engine, c->backend, ids + reused, (uint32_t)needed - reused, S->logits, &st, error, cap);
    } else {   /* dev43: restore the longest stored prefix (rl_statecache_prefill resets when there is none) */
        uint32_t loaded = 0, saved = 0;
        ok = rl_statecache_prefill(c->engine, c->backend, &c->states, ids, (uint32_t)needed, S->logits, &st, &loaded, &saved, error, cap);
        reused = loaded;
    }
    const uint32_t start_position = rl_engine_position(c->engine, c->backend);
    eng_unlock(c);
    const double prefill_ms = now_ms() - t_prefill;
    if (ok) { memcpy(S->history, ids, (size_t)needed * sizeof(uint32_t)); S->history_len = (uint32_t)needed; }
    free(ids);
    result->cached_tokens = reused;

    ttft_sink ts = {emit, sink, -1.0};
    uint32_t generated = 0, gpu_routed = 0, paired = 0;
    int fin = 0, r = 0;
    const double t_gen = now_ms();
    /* dev45: speculative decoding. The pending token u is emitted but not in the state yet (as in the plain loop);
     * a cycle drafts d, verifies u and d, and puts u (and d when accepted) into the state and the history.
     * dev56: it is used only while the other slot is idle; when a request arrives there, u is stepped and the
     * answer goes on in the plain loop, whose steps pair with the other slot's (two MTP slots alternating were
     * slower than serving them one after the other: 66 vs 71 tok/s). */
    uint32_t spec_cycles = 0, spec_accepted = 0;
#define SRV_EMIT(tok, fin) do { \
        if (rl_tokenizer_is_eog(c->tokenizer, (tok))) { fin = 1; break; } \
        char piece_[512]; const int32_t n_ = rl_tokenizer_decode(c->tokenizer, (tok), piece_, sizeof(piece_)); \
        if (n_ < 0) { snprintf(error, cap, "token decode failed"); ok = 0; fin = 1; break; } \
        generated++; \
        rl_sampler_accept(&sampler, (tok)); \
        if (!ttft_emit(&ts, piece_, (size_t)n_)) { fin = 1; break; } \
        if (generated >= req->max_tokens) { result->finish_length = 1; fin = 1; break; } \
    } while (0)
    const int spec = c->speculate && c->backend == RL_BACKEND_GPU &&
                     (!c->mtp_max_context || start_position <= c->mtp_max_context) && !other_busy(c, k);
    int switched = 0;
    uint32_t u = 0;
    if (ok && spec) {
        uint32_t mtp_pos = 0;
        u = rl_sampler_sample(&sampler, S->logits);
        SRV_EMIT(u, fin);
        if (!fin) {   /* one plain step so the hidden state the MTP block reads belongs to u's position */
            if (!(r = eng_step(c, k, u, S->logits, generated, &st, error, cap))) ok = 0;
            else { paired += r == 2; S->history[S->history_len++] = u; u = rl_sampler_sample(&sampler, S->logits); SRV_EMIT(u, fin); }
        }
        while (ok && !fin) {
            if (other_busy(c, k)) { switched = 1; break; }
            eng_lock(c, k);
            const int room = rl_engine_position(c->engine, c->backend) + 2u <= c->info->context;
            uint32_t d = 0, v = 0;
            int accepted = 0;
            if (room) {
                srv_steer(c, generated);
                ok = rl_engine_mtp_draft(c->engine, u, mtp_pos++, &d, NULL, error, cap) &&
                     rl_engine_verify2(c->engine, u, d, S->logits, S->logits1, error, cap);
                if (ok) {
                    v = rl_sampler_sample(&sampler, S->logits);
                    accepted = v == d;
                    ok = rl_engine_verify_commit(c->engine, accepted, error, cap);
                }
            }
            eng_unlock(c);
            if (!ok) break;
            if (!room) {   /* no room for two rows: plain step */
                if (!(r = eng_step(c, k, u, S->logits, generated, &st, error, cap))) { ok = 0; break; }
                S->history[S->history_len++] = u; u = rl_sampler_sample(&sampler, S->logits); SRV_EMIT(u, fin);
                continue;
            }
            spec_cycles++;
            S->history[S->history_len++] = u;
            if (accepted) { S->history[S->history_len++] = d; spec_accepted++; }
            SRV_EMIT(v, fin);
            if (fin) break;
            if (accepted) { u = rl_sampler_sample(&sampler, S->logits1); SRV_EMIT(u, fin); }
            else u = v;
        }
    }
    if (ok && !fin && (!spec || switched)) {
        set_decoding(c, k, 1);
        if (switched) {   /* the pending token u goes into the state; the plain loop samples what follows it */
            if (!(r = eng_step(c, k, u, S->logits, generated, &st, error, cap))) ok = 0;
            else { paired += r == 2; S->history[S->history_len++] = u; }
        }
        while (ok) {
            const uint32_t next = rl_sampler_sample(&sampler, S->logits);
            if (rl_tokenizer_is_eog(c->tokenizer, next)) break;
            char piece[512];
            const int32_t n = rl_tokenizer_decode(c->tokenizer, next, piece, sizeof(piece));
            if (n < 0) { snprintf(error, cap, "token decode failed"); ok = 0; break; }
            generated++;
            rl_sampler_accept(&sampler, next);
            if (!ttft_emit(&ts, piece, (size_t)n)) break;           /* client gone, Ctrl-C or stop sequence */
            if (generated >= req->max_tokens) { result->finish_length = 1; break; }
            if (!(r = eng_step(c, k, next, S->logits, generated, &st, error, cap))) { ok = 0; break; }
            S->history[S->history_len++] = next;                     /* fed back: part of the state */
            if (r == 2) paired++; else gpu_routed += st.speculative;
        }
        set_decoding(c, k, 0);
    }
    const double gen_ms = now_ms() - t_gen;
    rl_sampler_free(&sampler);
    result->completion_tokens = generated;
    if (!ok) S->history_len = 0;   /* unknown state: the next request resets */
    pthread_mutex_lock(&c->mu);
    S->busy = 0;
    pthread_mutex_unlock(&c->mu);
    if (ok) {
        fprintf(stderr, "[redlite-server] slot %d: prefill %u tok (%u cached) %.0f ms (%.1f tok/s), ttft %.0f ms, decode %u tok %.0f ms (%.1f tok/s, %u GPU-routed, %u paired)\n",
            k, (uint32_t)needed - reused, reused, prefill_ms, prefill_ms > 0 ? (needed - reused) * 1000.0 / prefill_ms : 0.0,
            ts.first_ms >= 0.0 ? ts.first_ms - t_request : 0.0,
            generated, gen_ms, generated > 1u ? (generated - 1u) * 1000.0 / gen_ms : 0.0, gpu_routed, paired);
        if (spec_cycles) fprintf(stderr, "[redlite-server] MTP speculation: %u cycles, %u drafts accepted (%.3f)\n",
                                 spec_cycles, spec_accepted, (double)spec_accepted / spec_cycles);
    }
    return ok;
}

static void usage(FILE *out) {
    fprintf(out,
        "redlite-server " REDLITE_VERSION " - OpenAI-compatible HTTP server on the native Red Lite engine\n\n"
        "Usage: redlite-server MODEL [options]\n\n"
        "  --host H            bind address (default 127.0.0.1)\n"
        "  --port P            TCP port (default 8080; 0 = ephemeral)\n"
        "  --context N         KV cache positions per request (default 4096)\n"
        "  --cache-mib N|full  routed-expert cache in MiB (default 4096); full = every expert of the file (preloaded)\n"
        "  --batch N           prompt tokens per batched prefill chunk (default 2048)\n"
        "  --max-tokens N      default max_tokens when a request omits it (default 256)\n"
        "  --temperature T     default temperature (default 0.7)\n"
        "  --top-k K           default top-k (default 40)\n"
        "  --top-p P           default top-p (default 0.95)\n"
        "  --min-p M           default min-p (default 0 = off)\n"
        "  --presence-penalty P   dev63: default presence penalty, -2..2 (default 0); Qwen advises 0-2 against endless\n"
        "                      repetitions\n"
        "  --frequency-penalty F  dev63: default frequency penalty, -2..2 (default 0)\n"
        "  --queue N           chat requests that may wait behind the running one (default 16; more get 503)\n"
        "  --no-reuse          reset the engine for every request (default: a prompt that extends the previous\n"
        "                      conversation exactly only prefills the new tokens)\n"
        "  --state-dir DIR     dev43: store chunk-aligned prompt-prefix checkpoints in DIR and restore the longest\n"
        "                      match when the in-memory state does not apply (first request, restart)\n"
        "  --state-max-mib N   size limit of --state-dir in MiB, least recently used first (default 8192)\n"
        "  --mtp FILE          dev45: speculative decoding with a Qwen3-Next MTP block GGUF (needs --cache-mib full;\n"
        "                      output is exactly that of plain decoding)\n"
        "  --mtp-max-context N dev55: no speculation for answers that start past position N (0 = no limit, the default)\n"
        "  --steer FILE        dev55: activation steering vector for the generated tokens (see redlite-generate)\n"
        "  --steer-layers A-B  steered layers (default 16-31)\n"
        "  --steer-strength S  steering strength (default 1)\n"
        "  --steer-tokens N    steer only the first N tokens of each answer (default 0 = all)\n"
        "  --parallel N        dev56: 1 or 2 requests at the same time (2 needs --cache-mib full; their decode steps run\n"
        "                      as one pass over the weights, MTP only while one request is decoding; default 1)\n"
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
    const char *state_dir = NULL;
    uint32_t state_max_mib = 8192u;
    uint32_t mtp_max_context = 0u, steer_tokens = 0u, steer_first = 16u, steer_last = 31u, parallel = 1u;
    const char *steer_path = NULL;
    float steer_base = 1.0f;
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
        else if (strcmp(argv[i - 1], "--state-dir") == 0) state_dir = v;
        else if (strcmp(argv[i - 1], "--mtp") == 0) cfg.mtp_path = v;
        else if (strcmp(argv[i - 1], "--mtp-max-context") == 0) { if (!parse_u32(v, &mtp_max_context)) return 2; }
        else if (strcmp(argv[i - 1], "--parallel") == 0) { if (!parse_u32(v, &parallel) || parallel < 1u || parallel > 2u) return 2; }
        else if (strcmp(argv[i - 1], "--steer") == 0) steer_path = v;
        else if (strcmp(argv[i - 1], "--steer-strength") == 0) { if (!parse_f32(v, &steer_base)) return 2; }
        else if (strcmp(argv[i - 1], "--steer-tokens") == 0) { if (!parse_u32(v, &steer_tokens)) return 2; }
        else if (strcmp(argv[i - 1], "--steer-layers") == 0) { if (sscanf(v, "%u-%u", &steer_first, &steer_last) != 2) return 2; }
        else if (strcmp(argv[i - 1], "--state-max-mib") == 0) { if (!parse_u32(v, &state_max_mib)) return 2; }
        else if (strcmp(argv[i - 1], "--presence-penalty") == 0) { if (!parse_f32(v, &scfg.default_presence_penalty) || fabsf(scfg.default_presence_penalty) > 2.0f) return 2; }
        else if (strcmp(argv[i - 1], "--frequency-penalty") == 0) { if (!parse_f32(v, &scfg.default_frequency_penalty) || fabsf(scfg.default_frequency_penalty) > 2.0f) return 2; }
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
    ctx.reuse = reuse;
    ctx.tool_format = rl_server_tool_format(g.chat_template);
    fprintf(stderr, "[redlite-server] tool calls: %s format\n", ctx.tool_format == RL_TOOLS_XML ? "Qwen3-Coder XML" : "Qwen3 JSON");
    pthread_mutex_init(&ctx.mu, NULL);
    pthread_cond_init(&ctx.cv, NULL);
    ctx.states.dir = state_dir;
    {
        double pre_ms = 0.0;
        ctx.speculate = !use_cpu && rl_engine_mtp_enabled(e) && rl_engine_experts_preloaded(e, &pre_ms);
        if (cfg.mtp_path && !ctx.speculate) fprintf(stderr, "[redlite-server] --mtp ignored: it needs --cache-mib full (every expert resident)\n");
        ctx.nslots = 1;
        if (parallel > 1) {
            if (use_cpu || !rl_engine_experts_preloaded(e, &pre_ms) || !rl_engine_slots_enable(e, error, sizeof(error)))
                fprintf(stderr, "[redlite-server] --parallel ignored: it needs the Metal backend and --cache-mib full%s%s\n",
                        error[0] ? ": " : "", error);
            else ctx.nslots = 2;
            error[0] = 0;
        }
        scfg.workers = ctx.nslots;
        for (uint32_t j = 0; j < ctx.nslots; ++j) {
            ctx.slot[j].logits = (float *)malloc((size_t)ctx.info->vocab * sizeof(float));
            ctx.slot[j].logits1 = (float *)malloc((size_t)ctx.info->vocab * sizeof(float));
            ctx.slot[j].history = (uint32_t *)malloc(((size_t)ctx.info->context + 1u) * sizeof(uint32_t));
            if (!ctx.slot[j].logits || !ctx.slot[j].logits1 || !ctx.slot[j].history) { fprintf(stderr, "slot allocation failed\n"); return 1; }
        }
        if (ctx.nslots == 2) fprintf(stderr, "[redlite-server] 2 parallel requests (their decode steps run as pairs)\n");
    }
    ctx.states.max_bytes = (uint64_t)state_max_mib * 1024u * 1024u;
    ctx.mtp_max_context = mtp_max_context;
    if (steer_path) {
        if (!rl_engine_load_steering(e, steer_path, steer_first, steer_last, 0.0f, error, sizeof(error))) {
            fprintf(stderr, "steering: %s\n", error); return 1;
        }
        ctx.steering = 1; ctx.steer_base = steer_base; ctx.steer_tokens = steer_tokens;
        fprintf(stderr, "[redlite-server] steering %s: layers %u-%u, strength %g%s\n", steer_path, steer_first, steer_last,
                (double)steer_base, steer_tokens ? " (first tokens of each answer)" : "");
    }
    int rc = 1;
    {
        rl_server_backend backend = {&ctx, "qwen3-next-80b-a3b-redlite", engine_generate, ctx.tool_format};
        if (rl_server_run(&scfg, &backend, &g_stop, error, sizeof(error))) rc = 0;
        else fprintf(stderr, "server failed: %s\n", error);
    }
    for (uint32_t j = 0; j < ctx.nslots; ++j) { free(ctx.slot[j].logits); free(ctx.slot[j].logits1); free(ctx.slot[j].history); }
    pthread_cond_destroy(&ctx.cv);
    pthread_mutex_destroy(&ctx.mu);
    rl_tokenizer_destroy(tk);
    rl_gguf_model_close(&g);
    rl_engine_close(e);
    if (!rc) fprintf(stderr, "[redlite-server] shut down cleanly\n");
    return rc;
}
