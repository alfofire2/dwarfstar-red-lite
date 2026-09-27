#pragma once
/*
 * Minimal OpenAI-compatible HTTP server core for the native Red Lite runtime.
 *
 * Portable C11 + POSIX sockets, no Metal: the transport, the JSON request parser,
 * the ChatML prompt builder and the SSE writer live here, the model lives behind
 * rl_server_backend. redlite-server binds the backend to rl_engine (macOS/Metal);
 * redlite-server-fake binds a deterministic echo backend so the protocol can be
 * tested anywhere without a model.
 *
 * Endpoints: GET /health, GET /v1/models, POST /v1/chat/completions (stream true/false).
 * One request at a time (the engine is single-sequence); every response closes the
 * connection. Each request is independent: the backend resets its state per request.
 */
#include <signal.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RL_SERVER_MAX_MESSAGES 256u

typedef struct {
    char *role;     /* "system" | "user" | "assistant" */
    char *content;  /* UTF-8 */
} rl_chat_message;

typedef struct {
    rl_chat_message messages[RL_SERVER_MAX_MESSAGES];
    uint32_t message_count;
    uint32_t max_tokens;   /* 0 -> server default */
    float temperature;     /* < 0 -> server default */
    float top_p;           /* < 0 -> server default */
    int32_t top_k;         /* < 0 -> server default (extension, not in the OpenAI schema) */
    uint64_t seed;
    int has_seed;
    int stream;
} rl_chat_request;

typedef struct {
    uint32_t prompt_tokens;
    uint32_t completion_tokens;
    int finish_length;     /* 1: stopped by max_tokens, 0: end-of-generation token */
} rl_server_result;

/* Receives generated UTF-8 bytes (any split). Returns 0 when generation must stop
 * (client disconnected or shutdown requested). */
typedef int (*rl_server_emit_fn)(void *sink, const char *bytes, size_t len);

typedef struct {
    void *ctx;
    const char *model_id;
    /* Returns 1 on success (also when stopped by emit returning 0), 0 on failure with error set.
     * status_out may be set to 400 for request-level errors (e.g. prompt exceeds context); default 500. */
    int (*generate)(void *ctx, const rl_chat_request *request, rl_server_emit_fn emit, void *sink,
                    rl_server_result *result, int *status_out, char *error, size_t error_cap);
} rl_server_backend;

typedef struct {
    const char *host;           /* default 127.0.0.1 */
    uint16_t port;              /* default 8080; 0 -> ephemeral, printed on stdout */
    uint32_t default_max_tokens;
    float default_temperature;
    float default_top_p;
    uint32_t default_top_k;
    int read_timeout_s;         /* per-connection receive timeout (default 30) */
} rl_server_config;

void rl_server_config_default(rl_server_config *cfg);

/* Serve until *stop becomes non-zero (e.g. set by a SIGINT handler). Returns 1 on clean shutdown. */
int rl_server_run(const rl_server_config *cfg, const rl_server_backend *backend,
                  volatile sig_atomic_t *stop, char *error, size_t error_cap);

/* ---- helpers exposed for backends and the model-free selftest ---- */

/* Parse an OpenAI chat-completions JSON body. Returns 1 on success; on failure error holds a
 * client-facing message. rl_chat_request_free releases the strings. */
int rl_chat_request_parse(const char *body, size_t len, rl_chat_request *out, char *error, size_t error_cap);
void rl_chat_request_free(rl_chat_request *req);

/* ChatML prompt (Qwen3-Next): every message as <|im_start|>role\ncontent<|im_end|>\n, then
 * <|im_start|>assistant\n. Returns a malloc'd string or NULL on allocation failure. */
char *rl_server_chatml(const rl_chat_request *req);

/* Length of the longest prefix of buf[0..len) that does not end inside an incomplete UTF-8
 * sequence (bytes after it must wait for the next token). */
size_t rl_utf8_complete_prefix(const char *buf, size_t len);

/* Model-free checks of the parser, prompt builder, UTF-8 hold-back and JSON escaping. */
int rl_server_selftest(char *error, size_t error_cap);

#ifdef __cplusplus
}
#endif
