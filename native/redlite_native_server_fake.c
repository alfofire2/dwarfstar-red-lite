#define _POSIX_C_SOURCE 200809L
#define _DARWIN_C_SOURCE 1

/*
 * redlite-server-fake: the native HTTP server core (redlite_native_server.c) bound to a
 * deterministic echo backend instead of rl_engine. Test tool for the OpenAI protocol
 * (tests/test_native_server.py); no model, no Metal.
 *
 * Reply: "Echo: <last user message>" emitted in 3-byte "tokens" (so multi-byte UTF-8 is
 * split across tokens), one token per emit call, capped by max_tokens.
 * prompt_tokens = byte length of the ChatML prompt. Like the engine backend (dev29), the fake keeps
 * a "state": the prompt bytes plus every reply byte it emitted and would have fed back; when the
 * next prompt starts with exactly those bytes, they are reported as usage cached_tokens.
 * Special last-user contents:
 *   "__fail__"         backend error before any output (500)
 *   "__fail_late__"    backend error after two tokens (SSE error event when streaming)
 *   "__too_long__"     request-level error (400), like a prompt exceeding the context
 *   "__params__"       replies with the effective sampling parameters
 */

#include "redlite_native_server.h"

#include <signal.h>
#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static volatile sig_atomic_t g_stop = 0;
static void on_signal(int sig) { (void)sig; g_stop = 1; }

typedef struct {
    unsigned token_delay_ms;
    char *state;        /* bytes "ingested" by the previous request (NULL: reset) */
    size_t state_len;
    pthread_mutex_t mu; /* dev56: --workers N runs generations at the same time */
} fake_ctx;

static void state_append(fake_ctx *f, const char *p, size_t n) {
    char *grown = (char *)realloc(f->state, f->state_len + n + 1u);
    if (!grown) { free(f->state); f->state = NULL; f->state_len = 0; return; }
    memcpy(grown + f->state_len, p, n);
    f->state = grown;
    f->state_len += n;
}

static void sleep_ms(unsigned ms) {
    if (!ms) return;
    struct timespec ts = {(time_t)(ms / 1000u), (long)(ms % 1000u) * 1000000L};
    while (nanosleep(&ts, &ts) != 0 && !g_stop) {}
}

static int fake_generate(void *ctx, const rl_chat_request *req, rl_server_emit_fn emit, void *sink,
                         rl_server_result *result, int *status, char *error, size_t cap) {
    fake_ctx *f = (fake_ctx *)ctx;
    const char *last = "";
    for (uint32_t i = 0; i < req->message_count; ++i)
        if (strcmp(req->messages[i].role, "user") == 0) last = req->messages[i].content;
    char *prompt = rl_server_chatml(req);
    if (!prompt) { snprintf(error, cap, "out of memory"); return 0; }
    const size_t prompt_len = strlen(prompt);
    result->prompt_tokens = (uint32_t)prompt_len;
    pthread_mutex_lock(&f->mu);
    const int reuse = f->state && f->state_len < prompt_len && memcmp(f->state, prompt, f->state_len) == 0;
    result->cached_tokens = reuse ? (uint32_t)f->state_len : 0u;
    free(f->state);
    f->state = NULL;
    f->state_len = 0;
    state_append(f, prompt, prompt_len);
    free(prompt);

    if (strcmp(last, "__fail__") == 0 || strcmp(last, "__too_long__") == 0) { free(f->state); f->state = NULL; f->state_len = 0; }
    pthread_mutex_unlock(&f->mu);
    if (strcmp(last, "__fail__") == 0) { snprintf(error, cap, "fake backend failure"); return 0; }
    if (strcmp(last, "__too_long__") == 0) {
        *status = 400;
        snprintf(error, cap, "prompt (%u tokens) + max_tokens (%u) exceed the context", result->prompt_tokens, req->max_tokens);
        return 0;
    }
    char reply[4096];
    if (strcmp(last, "__params__") == 0) {
        snprintf(reply, sizeof(reply), "temperature=%.2f top_p=%.2f top_k=%d min_p=%.2f max_tokens=%u seed=%s%llu",
            (double)req->temperature, (double)req->top_p, (int)req->top_k, (double)req->min_p, req->max_tokens,
            req->has_seed ? "" : "none/", (unsigned long long)req->seed);
    } else {
        snprintf(reply, sizeof(reply), "Echo: %s", last);
    }
    const int fail_late = strcmp(last, "__fail_late__") == 0;
    const size_t n = strlen(reply);
    result->completion_tokens = 0;
    result->finish_length = 0;
    for (size_t off = 0; off < n; off += 3u) {
        if (result->completion_tokens == req->max_tokens) { result->finish_length = 1; break; }
        if (fail_late && result->completion_tokens == 2u) {
            pthread_mutex_lock(&f->mu);
            free(f->state); f->state = NULL; f->state_len = 0;   /* a failed request leaves no reusable state */
            pthread_mutex_unlock(&f->mu);
            snprintf(error, cap, "fake backend failure mid-stream");
            return 0;
        }
        const size_t len = n - off < 3u ? n - off : 3u;
        result->completion_tokens++;
        if (!emit(sink, reply + off, len)) return 1; /* client gone, shutdown or stop sequence: stop cleanly */
        if (result->completion_tokens < req->max_tokens) {   /* the engine feeds it back */
            pthread_mutex_lock(&f->mu); state_append(f, reply + off, len); pthread_mutex_unlock(&f->mu);
        }
        sleep_ms(f->token_delay_ms);
    }
    return 1;
}

int main(int argc, char **argv) {
    rl_server_config cfg;
    rl_server_config_default(&cfg);
    fake_ctx ctx = {0};
    pthread_mutex_init(&ctx.mu, NULL);
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--selftest") == 0) {
            char error[512] = {0};
            if (!rl_server_selftest(error, sizeof(error))) { fprintf(stderr, "server selftest: FAIL (%s)\n", error); return 1; }
            printf("server selftest     : OK (JSON parser, ChatML, UTF-8 hold-back, stop scan, prefix reuse, escaping)\n");
            return 0;
        }
        if (strcmp(argv[i], "--help") == 0) {
            printf("Usage: redlite-server-fake [--host H] [--port P] [--token-delay-ms N] [--queue N] [--workers N] [--selftest]\n");
            return 0;
        }
        if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", argv[i]); return 2; }
        if (strcmp(argv[i], "--host") == 0) cfg.host = argv[++i];
        else if (strcmp(argv[i], "--port") == 0) cfg.port = (uint16_t)strtoul(argv[++i], NULL, 10);
        else if (strcmp(argv[i], "--token-delay-ms") == 0) ctx.token_delay_ms = (unsigned)strtoul(argv[++i], NULL, 10);
        else if (strcmp(argv[i], "--queue") == 0) cfg.queue_max = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (strcmp(argv[i], "--workers") == 0) cfg.workers = (uint32_t)strtoul(argv[++i], NULL, 10);
        else { fprintf(stderr, "unknown option %s\n", argv[i]); return 2; }
    }
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = on_signal;
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
    rl_server_backend backend = {&ctx, "redlite-fake-echo", fake_generate};
    char error[512] = {0};
    const int ok = rl_server_run(&cfg, &backend, &g_stop, error, sizeof(error));
    free(ctx.state);
    if (!ok) { fprintf(stderr, "server failed: %s\n", error); return 1; }
    fprintf(stderr, "[redlite-server] shut down cleanly\n");
    return 0;
}
