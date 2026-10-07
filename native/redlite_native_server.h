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
 * dev59: OpenAI tool calling. `tools`, assistant `tool_calls` and `tool` messages are rendered exactly as the
 * model's own chat template does (RL_TOOLS_JSON: Qwen3-Next Instruct, RL_TOOLS_XML: Qwen3-Coder), and the calls the
 * model writes come back as `tool_calls` with finish_reason "tool_calls".
 * The engine is single-sequence: chat requests are parsed by the accepting thread and run
 * one at a time, in arrival order, by one worker thread (dev29 FIFO queue; /health and
 * /v1/models are answered while a generation runs). Every response closes the connection.
 * A backend may keep its state between requests when the new prompt extends the previous
 * one exactly (dev29, see rl_prefix_reuse); otherwise it resets.
 */
#include <signal.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RL_SERVER_MAX_MESSAGES 16384u   /* dev62: agent sessions add two messages per tool call (256 was hit) */
#define RL_SERVER_MAX_STOP 4u          /* OpenAI: up to 4 stop sequences */
#define RL_SERVER_MAX_STOP_BYTES 256u  /* per stop sequence */
#define RL_SERVER_MAX_TOOL_CALLS 16u   /* per assistant message */

enum { RL_TOOLS_JSON = 0, RL_TOOLS_XML = 1 };   /* dev59: tool-call format of the chat template */

struct rl_json;   /* parsed JSON value (opaque) */

typedef struct {
    char *id;
    char *name;
    struct rl_json *arguments;   /* a JSON string (OpenAI: the arguments as text) or an object */
} rl_tool_call;

typedef struct {
    char *role;     /* "system" | "user" | "assistant" | "tool" */
    char *content;  /* UTF-8 ("" for an assistant message that only calls tools) */
    rl_tool_call tool_calls[RL_SERVER_MAX_TOOL_CALLS];
    uint32_t tool_call_count;
} rl_chat_message;

typedef struct {
    rl_chat_message *messages;   /* dev62: heap array, grown while parsing (the request is copied onto a worker stack) */
    uint32_t message_count;
    uint32_t message_cap;
    uint32_t max_tokens;   /* 0 -> server default */
    float temperature;     /* < 0 -> server default */
    float top_p;           /* < 0 -> server default */
    int32_t top_k;         /* < 0 -> server default (extension, not in the OpenAI schema) */
    float min_p;           /* < 0 -> server default (extension, as in llama-server) */
    float presence_penalty;   /* dev63: OpenAI, -2..2; NAN -> server default */
    float frequency_penalty;  /* dev63: OpenAI, -2..2; NAN -> server default */
    uint64_t seed;
    int has_seed;
    int stream;
    char *stop[RL_SERVER_MAX_STOP];   /* stop sequences (non-empty UTF-8), not included in the output */
    uint32_t stop_count;
    struct rl_json *tools;   /* dev59: the "tools" array, NULL when absent or tool_choice is "none" */
    int tool_choice_none;
} rl_chat_request;

typedef struct {
    uint32_t prompt_tokens;
    uint32_t completion_tokens;
    int finish_length;     /* 1: stopped by max_tokens, 0: end-of-generation token or stop sequence */
    uint32_t cached_tokens; /* prompt tokens reused from the previous request's state (usage.prompt_tokens_details) */
} rl_server_result;

/* Receives generated UTF-8 bytes (any split). Returns 0 when generation must stop
 * (client disconnected, shutdown requested or a stop sequence matched). */
typedef int (*rl_server_emit_fn)(void *sink, const char *bytes, size_t len);

typedef struct {
    void *ctx;
    const char *model_id;
    /* Returns 1 on success (also when stopped by emit returning 0), 0 on failure with error set.
     * status_out may be set to 400 for request-level errors (e.g. prompt exceeds context); default 500. */
    int (*generate)(void *ctx, const rl_chat_request *request, rl_server_emit_fn emit, void *sink,
                    rl_server_result *result, int *status_out, char *error, size_t error_cap);
    int tool_format;   /* dev59: RL_TOOLS_JSON or RL_TOOLS_XML (rl_server_tool_format of the GGUF chat template) */
} rl_server_backend;

typedef struct {
    const char *host;           /* default 127.0.0.1 */
    uint16_t port;              /* default 8080; 0 -> ephemeral, printed on stdout */
    uint32_t default_max_tokens;
    float default_temperature;
    float default_top_p;
    uint32_t default_top_k;
    float default_min_p;
    float default_presence_penalty;    /* dev63 (default 0) */
    float default_frequency_penalty;
    int read_timeout_s;         /* per-connection receive timeout (default 30) */
    uint32_t queue_max;         /* chat requests waiting behind the running ones before 503 (default 16) */
    uint32_t workers;           /* dev56: generations run at the same time (default 1; the backend must be re-entrant) */
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

/* The prompt, rendered as the model's chat template does (generation prompt included). Without tools and tool
 * messages both formats are plain ChatML: every message as <|im_start|>role\ncontent<|im_end|>\n, then
 * <|im_start|>assistant\n. Returns a malloc'd string or NULL on allocation failure. */
char *rl_server_prompt(const rl_chat_request *req, int tool_format);
char *rl_server_chatml(const rl_chat_request *req);   /* rl_server_prompt(req, RL_TOOLS_JSON) */

/* dev59: RL_TOOLS_XML when the GGUF chat template uses Qwen3-Coder's <function=...> calls, else RL_TOOLS_JSON. */
int rl_server_tool_format(const char *chat_template);

/* Length of the longest prefix of buf[0..len) that does not end inside an incomplete UTF-8
 * sequence (bytes after it must wait for the next token). */
size_t rl_utf8_complete_prefix(const char *buf, size_t len);

/* dev29 state reuse: history_len when history[0..history_len) is a proper prefix of
 * ids[0..count) (the new prompt extends exactly what the backend state holds), else 0.
 * Recurrent state cannot be rolled back, so any other case needs a reset and a full prefill. */
uint32_t rl_prefix_reuse(const uint32_t *history, uint32_t history_len, const uint32_t *ids, uint32_t count);

/* Stop-sequence scan of not-yet-sent text. Returns 1 when a stop sequence occurs in buf:
 * *emit_len is the offset of the earliest occurrence (the text before it is sent, the rest
 * dropped). Returns 0 otherwise: *emit_len = len minus the longest suffix of buf that is a
 * proper prefix of some stop sequence (that suffix is held back until more text arrives). */
int rl_stop_scan(const char *buf, size_t len, char *const *stops, uint32_t stop_count, size_t *emit_len);

/* Model-free checks of the parser, prompt builder, UTF-8 hold-back, stop scan, prefix reuse and JSON escaping. */
int rl_server_selftest(char *error, size_t error_cap);

#ifdef __cplusplus
}
#endif
