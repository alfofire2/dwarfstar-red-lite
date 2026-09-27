#define _POSIX_C_SOURCE 200809L
#define _DARWIN_C_SOURCE 1

#include "redlite_native_server.h"

#include <errno.h>
#include <math.h>
#include <netdb.h>
#include <netinet/in.h>
#include <poll.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0 /* macOS: SIGPIPE is ignored by rl_server_run instead */
#endif

#define RL_SERVER_MAX_HEADER (64u * 1024u)
#define RL_SERVER_MAX_BODY (8u * 1024u * 1024u)
#define RL_JSON_MAX_DEPTH 64

static void set_error(char *error, size_t cap, const char *fmt, ...) {
    if (!error || !cap) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(error, cap, fmt, ap);
    va_end(ap);
}

/* ------------------------------------------------------------------ string builder */

typedef struct {
    char *data;
    size_t len;
    size_t cap;
    int oom;
} sb;

static int sb_reserve(sb *b, size_t extra) {
    if (b->oom) return 0;
    if (extra > SIZE_MAX - b->len - 1u) { b->oom = 1; return 0; }
    if (b->len + extra + 1u <= b->cap) return 1;
    size_t cap = b->cap ? b->cap : 256u;
    while (cap < b->len + extra + 1u) cap = cap > SIZE_MAX / 2u ? b->len + extra + 1u : cap * 2u;
    char *p = (char *)realloc(b->data, cap);
    if (!p) { b->oom = 1; return 0; }
    b->data = p;
    b->cap = cap;
    return 1;
}

static void sb_put(sb *b, const char *s, size_t n) {
    if (!sb_reserve(b, n)) return;
    memcpy(b->data + b->len, s, n);
    b->len += n;
    b->data[b->len] = '\0';
}

static void sb_puts(sb *b, const char *s) { sb_put(b, s, strlen(s)); }

static void sb_printf(sb *b, const char *fmt, ...) {
    char tmp[256];
    va_list ap;
    va_start(ap, fmt);
    const int n = vsnprintf(tmp, sizeof(tmp), fmt, ap);
    va_end(ap);
    if (n < 0) { b->oom = 1; return; }
    if ((size_t)n < sizeof(tmp)) { sb_put(b, tmp, (size_t)n); return; }
    if (!sb_reserve(b, (size_t)n)) return;
    va_start(ap, fmt);
    vsnprintf(b->data + b->len, (size_t)n + 1u, fmt, ap);
    va_end(ap);
    b->len += (size_t)n;
}

static void sb_free(sb *b) { free(b->data); memset(b, 0, sizeof(*b)); }

/* ------------------------------------------------------------------ UTF-8 */

/* Length of a well-formed UTF-8 sequence starting at s (max bytes available n):
 * >0 = sequence length, 0 = truncated but could still become valid, -1 = invalid. */
static int utf8_seq(const unsigned char *s, size_t n) {
    const unsigned char c = s[0];
    int len;
    unsigned char lo = 0x80, hi = 0xBF;
    if (c < 0x80) return 1;
    else if (c >= 0xC2 && c <= 0xDF) len = 2;
    else if (c >= 0xE0 && c <= 0xEF) { len = 3; if (c == 0xE0) lo = 0xA0; if (c == 0xED) hi = 0x9F; }
    else if (c >= 0xF0 && c <= 0xF4) { len = 4; if (c == 0xF0) lo = 0x90; if (c == 0xF4) hi = 0x8F; }
    else return -1;
    for (int i = 1; i < len; ++i) {
        if ((size_t)i >= n) return 0;
        const unsigned char lower = i == 1 ? lo : 0x80, upper = i == 1 ? hi : 0xBF;
        if (s[i] < lower || s[i] > upper) return -1;
    }
    return len;
}

size_t rl_utf8_complete_prefix(const char *buf, size_t len) {
    const unsigned char *s = (const unsigned char *)buf;
    size_t i = 0;
    while (i < len) {
        const int k = utf8_seq(s + i, len - i);
        if (k == 0) return i;         /* incomplete tail: hold it back */
        i += k > 0 ? (size_t)k : 1u;  /* invalid bytes are passed on (escaped to U+FFFD) */
    }
    return len;
}

/* JSON string body (no quotes): escapes control characters, quote and backslash; replaces
 * invalid UTF-8 with U+FFFD so the output is always valid JSON text. */
static void json_escape(sb *b, const char *str, size_t n) {
    const unsigned char *s = (const unsigned char *)str;
    size_t i = 0;
    while (i < n) {
        const unsigned char c = s[i];
        if (c < 0x80) {
            switch (c) {
                case '"': sb_puts(b, "\\\""); break;
                case '\\': sb_puts(b, "\\\\"); break;
                case '\n': sb_puts(b, "\\n"); break;
                case '\r': sb_puts(b, "\\r"); break;
                case '\t': sb_puts(b, "\\t"); break;
                case '\b': sb_puts(b, "\\b"); break;
                case '\f': sb_puts(b, "\\f"); break;
                default:
                    if (c < 0x20 || c == 0x7F) sb_printf(b, "\\u%04x", (unsigned)c);
                    else sb_put(b, (const char *)&s[i], 1u);
            }
            ++i;
            continue;
        }
        const int k = utf8_seq(s + i, n - i);
        if (k > 0) { sb_put(b, (const char *)&s[i], (size_t)k); i += (size_t)k; }
        else { sb_puts(b, "\xEF\xBF\xBD"); ++i; }
    }
}

static void json_str(sb *b, const char *s, size_t n) {
    sb_put(b, "\"", 1u);
    json_escape(b, s, n);
    sb_put(b, "\"", 1u);
}

/* ------------------------------------------------------------------ JSON request parser */

typedef struct {
    const char *p;
    const char *end;
    int depth;
    char *error;
    size_t cap;
} jp;

static void jws(jp *j) {
    while (j->p < j->end && (*j->p == ' ' || *j->p == '\t' || *j->p == '\n' || *j->p == '\r')) j->p++;
}

static int jfail(jp *j, const char *msg) {
    if (j->error && j->cap && !j->error[0]) set_error(j->error, j->cap, "%s", msg);
    return 0;
}

static int hexval(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static int read_hex4(jp *j, uint32_t *out) {
    if (j->end - j->p < 4) return 0;
    uint32_t v = 0;
    for (int i = 0; i < 4; ++i) {
        const int h = hexval(j->p[i]);
        if (h < 0) return 0;
        v = (v << 4) | (uint32_t)h;
    }
    j->p += 4;
    *out = v;
    return 1;
}

static void put_utf8(sb *b, uint32_t cp) {
    char o[4];
    if (cp < 0x80) { o[0] = (char)cp; sb_put(b, o, 1u); }
    else if (cp < 0x800) { o[0] = (char)(0xC0 | (cp >> 6)); o[1] = (char)(0x80 | (cp & 0x3F)); sb_put(b, o, 2u); }
    else if (cp < 0x10000) {
        o[0] = (char)(0xE0 | (cp >> 12)); o[1] = (char)(0x80 | ((cp >> 6) & 0x3F)); o[2] = (char)(0x80 | (cp & 0x3F));
        sb_put(b, o, 3u);
    } else {
        o[0] = (char)(0xF0 | (cp >> 18)); o[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
        o[2] = (char)(0x80 | ((cp >> 6) & 0x3F)); o[3] = (char)(0x80 | (cp & 0x3F));
        sb_put(b, o, 4u);
    }
}

/* Parses a string at j->p (which must be '"'). out (may be NULL to skip) receives a malloc'd copy. */
static int jstring(jp *j, char **out) {
    if (j->p >= j->end || *j->p != '"') return jfail(j, "expected a JSON string");
    j->p++;
    sb b = {0};
    for (;;) {
        if (j->p >= j->end) { sb_free(&b); return jfail(j, "unterminated JSON string"); }
        const unsigned char c = (unsigned char)*j->p;
        if (c == '"') { j->p++; break; }
        if (c < 0x20) { sb_free(&b); return jfail(j, "control character in JSON string"); }
        if (c != '\\') {
            const char *start = j->p;
            while (j->p < j->end && *j->p != '"' && *j->p != '\\' && (unsigned char)*j->p >= 0x20) j->p++;
            if (out) sb_put(&b, start, (size_t)(j->p - start));
            continue;
        }
        j->p++;
        if (j->p >= j->end) { sb_free(&b); return jfail(j, "unterminated JSON escape"); }
        const char e = *j->p++;
        switch (e) {
            case '"': sb_put(&b, "\"", 1u); break;
            case '\\': sb_put(&b, "\\", 1u); break;
            case '/': sb_put(&b, "/", 1u); break;
            case 'b': sb_put(&b, "\b", 1u); break;
            case 'f': sb_put(&b, "\f", 1u); break;
            case 'n': sb_put(&b, "\n", 1u); break;
            case 'r': sb_put(&b, "\r", 1u); break;
            case 't': sb_put(&b, "\t", 1u); break;
            case 'u': {
                uint32_t cp = 0;
                if (!read_hex4(j, &cp)) { sb_free(&b); return jfail(j, "invalid \\u escape"); }
                if (cp >= 0xD800 && cp <= 0xDBFF) {
                    uint32_t lo = 0;
                    if (j->end - j->p >= 6 && j->p[0] == '\\' && j->p[1] == 'u') {
                        const char *save = j->p;
                        j->p += 2;
                        if (read_hex4(j, &lo) && lo >= 0xDC00 && lo <= 0xDFFF) cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                        else { j->p = save; cp = 0xFFFD; }
                    } else {
                        cp = 0xFFFD;
                    }
                } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                    cp = 0xFFFD;
                }
                if (cp == 0) { sb_free(&b); return jfail(j, "NUL characters are not supported in strings"); }
                put_utf8(&b, cp);
                break;
            }
            default: sb_free(&b); return jfail(j, "invalid JSON escape");
        }
    }
    if (b.oom) { sb_free(&b); return jfail(j, "out of memory"); }
    if (out) {
        if (!b.data) { b.data = (char *)calloc(1u, 1u); if (!b.data) return jfail(j, "out of memory"); }
        *out = b.data;
    } else {
        sb_free(&b);
    }
    return 1;
}

static int jnumber(jp *j, double *out) {
    const char *s = j->p;
    if (s < j->end && *s == '-') s++;
    if (s >= j->end || *s < '0' || *s > '9') return jfail(j, "invalid JSON number");
    if (*s == '0') s++;
    else while (s < j->end && *s >= '0' && *s <= '9') s++;
    if (s < j->end && *s == '.') {
        s++;
        if (s >= j->end || *s < '0' || *s > '9') return jfail(j, "invalid JSON number");
        while (s < j->end && *s >= '0' && *s <= '9') s++;
    }
    if (s < j->end && (*s == 'e' || *s == 'E')) {
        s++;
        if (s < j->end && (*s == '+' || *s == '-')) s++;
        if (s >= j->end || *s < '0' || *s > '9') return jfail(j, "invalid JSON number");
        while (s < j->end && *s >= '0' && *s <= '9') s++;
    }
    char tmp[64];
    const size_t n = (size_t)(s - j->p);
    if (n >= sizeof(tmp)) return jfail(j, "JSON number too long");
    memcpy(tmp, j->p, n);
    tmp[n] = '\0';
    j->p = s;
    const double v = strtod(tmp, NULL);
    if (!isfinite(v)) return jfail(j, "JSON number out of range");
    if (out) *out = v;
    return 1;
}

static int jliteral(jp *j, const char *lit) {
    const size_t n = strlen(lit);
    if ((size_t)(j->end - j->p) < n || memcmp(j->p, lit, n) != 0) return jfail(j, "invalid JSON literal");
    j->p += n;
    return 1;
}

static int jskip(jp *j);

static int jskip_container(jp *j, char open, char close) {
    if (++j->depth > RL_JSON_MAX_DEPTH) return jfail(j, "JSON nesting too deep");
    j->p++;
    jws(j);
    if (j->p < j->end && *j->p == close) { j->p++; j->depth--; return 1; }
    for (;;) {
        jws(j);
        if (open == '{') {
            if (!jstring(j, NULL)) return 0;
            jws(j);
            if (j->p >= j->end || *j->p != ':') return jfail(j, "expected ':' in JSON object");
            j->p++;
        }
        if (!jskip(j)) return 0;
        jws(j);
        if (j->p < j->end && *j->p == ',') { j->p++; continue; }
        if (j->p < j->end && *j->p == close) { j->p++; j->depth--; return 1; }
        return jfail(j, "expected ',' or closing bracket in JSON");
    }
}

static int jskip(jp *j) {
    jws(j);
    if (j->p >= j->end) return jfail(j, "unexpected end of JSON");
    switch (*j->p) {
        case '{': return jskip_container(j, '{', '}');
        case '[': return jskip_container(j, '[', ']');
        case '"': return jstring(j, NULL);
        case 't': return jliteral(j, "true");
        case 'f': return jliteral(j, "false");
        case 'n': return jliteral(j, "null");
        default: return jnumber(j, NULL);
    }
}

/* Iterate the members of an object: calls fn(j, key, user) with j->p at the value; fn must consume it. */
typedef int (*jmember_fn)(jp *j, const char *key, void *user);

static int jobject(jp *j, jmember_fn fn, void *user) {
    jws(j);
    if (j->p >= j->end || *j->p != '{') return jfail(j, "expected a JSON object");
    if (++j->depth > RL_JSON_MAX_DEPTH) return jfail(j, "JSON nesting too deep");
    j->p++;
    jws(j);
    if (j->p < j->end && *j->p == '}') { j->p++; j->depth--; return 1; }
    for (;;) {
        jws(j);
        char *key = NULL;
        if (!jstring(j, &key)) return 0;
        jws(j);
        if (j->p >= j->end || *j->p != ':') { free(key); return jfail(j, "expected ':' in JSON object"); }
        j->p++;
        jws(j);
        const int ok = fn(j, key, user);
        free(key);
        if (!ok) return 0;
        jws(j);
        if (j->p < j->end && *j->p == ',') { j->p++; continue; }
        if (j->p < j->end && *j->p == '}') { j->p++; j->depth--; return 1; }
        return jfail(j, "expected ',' or '}' in JSON object");
    }
}

static int jis_null(jp *j) { return j->p < j->end && *j->p == 'n'; }

typedef struct {
    char *role;
    sb content;
    int have_content;
} msg_state;

static int content_part_member(jp *j, const char *key, void *user) {
    char **slot = (char **)user;
    if (strcmp(key, "type") == 0) {
        char *type = NULL;
        if (!jstring(j, &type)) return 0;
        const int text = strcmp(type, "text") == 0;
        free(type);
        if (!text) return jfail(j, "only text content parts are supported");
        return 1;
    }
    if (strcmp(key, "text") == 0) {
        free(slot[0]);
        slot[0] = NULL;
        return jstring(j, &slot[0]);
    }
    return jskip(j);
}

static int message_member(jp *j, const char *key, void *user) {
    msg_state *m = (msg_state *)user;
    if (strcmp(key, "role") == 0) { free(m->role); m->role = NULL; return jstring(j, &m->role); }
    if (strcmp(key, "content") == 0) {
        if (jis_null(j)) return jfail(j, "message content must not be null (tool calls are not supported)");
        m->have_content = 1;
        m->content.len = 0;
        if (j->p < j->end && *j->p == '"') {
            char *s = NULL;
            if (!jstring(j, &s)) return 0;
            sb_puts(&m->content, s);
            free(s);
            return !m->content.oom || jfail(j, "out of memory");
        }
        if (j->p < j->end && *j->p == '[') {
            if (++j->depth > RL_JSON_MAX_DEPTH) return jfail(j, "JSON nesting too deep");
            j->p++;
            jws(j);
            if (j->p < j->end && *j->p == ']') { j->p++; j->depth--; return 1; }
            for (;;) {
                char *text[1] = {NULL};
                if (!jobject(j, content_part_member, text)) { free(text[0]); return 0; }
                if (text[0]) { sb_puts(&m->content, text[0]); free(text[0]); }
                jws(j);
                if (j->p < j->end && *j->p == ',') { j->p++; jws(j); continue; }
                if (j->p < j->end && *j->p == ']') { j->p++; j->depth--; break; }
                return jfail(j, "expected ',' or ']' in content array");
            }
            return !m->content.oom || jfail(j, "out of memory");
        }
        return jfail(j, "message content must be a string or an array of text parts");
    }
    if (strcmp(key, "tool_calls") == 0 || strcmp(key, "function_call") == 0) {
        if (jis_null(j)) return jskip(j);
        return jfail(j, "tool calls are not supported");
    }
    return jskip(j);
}

static int parse_messages(jp *j, rl_chat_request *req) {
    jws(j);
    if (j->p >= j->end || *j->p != '[') return jfail(j, "\"messages\" must be an array");
    if (++j->depth > RL_JSON_MAX_DEPTH) return jfail(j, "JSON nesting too deep");
    j->p++;
    jws(j);
    if (j->p < j->end && *j->p == ']') { j->p++; j->depth--; return 1; }
    for (;;) {
        if (req->message_count >= RL_SERVER_MAX_MESSAGES) return jfail(j, "too many messages");
        msg_state m = {0};
        if (!jobject(j, message_member, &m)) { free(m.role); sb_free(&m.content); return 0; }
        if (!m.role) { sb_free(&m.content); return jfail(j, "every message needs a \"role\""); }
        if (strcmp(m.role, "developer") == 0) { free(m.role); m.role = strdup("system"); }
        if (!m.role || (strcmp(m.role, "system") != 0 && strcmp(m.role, "user") != 0 && strcmp(m.role, "assistant") != 0)) {
            free(m.role); sb_free(&m.content);
            return jfail(j, "message role must be system, user or assistant");
        }
        if (!m.have_content) { free(m.role); sb_free(&m.content); return jfail(j, "every message needs \"content\""); }
        if (!m.content.data) { m.content.data = (char *)calloc(1u, 1u); if (!m.content.data) { free(m.role); return jfail(j, "out of memory"); } }
        req->messages[req->message_count].role = m.role;
        req->messages[req->message_count].content = m.content.data;
        req->message_count++;
        jws(j);
        if (j->p < j->end && *j->p == ',') { j->p++; jws(j); continue; }
        if (j->p < j->end && *j->p == ']') { j->p++; j->depth--; return 1; }
        return jfail(j, "expected ',' or ']' in messages");
    }
}

static int number_member(jp *j, double *out, const char *name) {
    if (jis_null(j)) return jliteral(j, "null");
    if (j->p < j->end && (*j->p == '"' || *j->p == '{' || *j->p == '[' || *j->p == 't' || *j->p == 'f')) {
        char msg[96];
        snprintf(msg, sizeof(msg), "\"%s\" must be a number", name);
        return jfail(j, msg);
    }
    return jnumber(j, out);
}

static int integer_in(double v, double lo, double hi) { return v == floor(v) && v >= lo && v <= hi; }

static int request_member(jp *j, const char *key, void *user) {
    rl_chat_request *req = (rl_chat_request *)user;
    double v = NAN;
    if (strcmp(key, "messages") == 0) return parse_messages(j, req);
    if (strcmp(key, "stream") == 0) {
        if (jis_null(j)) return jliteral(j, "null");
        if (j->p < j->end && *j->p == 't') { req->stream = 1; return jliteral(j, "true"); }
        if (j->p < j->end && *j->p == 'f') { req->stream = 0; return jliteral(j, "false"); }
        return jfail(j, "\"stream\" must be a boolean");
    }
    if (strcmp(key, "max_tokens") == 0 || strcmp(key, "max_completion_tokens") == 0) {
        if (!number_member(j, &v, key)) return 0;
        if (isnan(v)) return 1;
        if (!integer_in(v, 1.0, 1048576.0)) return jfail(j, "max_tokens must be an integer >= 1");
        req->max_tokens = (uint32_t)v;
        return 1;
    }
    if (strcmp(key, "temperature") == 0) {
        if (!number_member(j, &v, key)) return 0;
        if (isnan(v)) return 1;
        if (v < 0.0 || v > 2.0) return jfail(j, "temperature must be between 0 and 2");
        req->temperature = (float)v;
        return 1;
    }
    if (strcmp(key, "top_p") == 0) {
        if (!number_member(j, &v, key)) return 0;
        if (isnan(v)) return 1;
        if (v <= 0.0 || v > 1.0) return jfail(j, "top_p must be in (0, 1]");
        req->top_p = (float)v;
        return 1;
    }
    if (strcmp(key, "top_k") == 0) {
        if (!number_member(j, &v, key)) return 0;
        if (isnan(v)) return 1;
        if (!integer_in(v, 0.0, 1048576.0)) return jfail(j, "top_k must be an integer >= 0");
        req->top_k = (int32_t)v;
        return 1;
    }
    if (strcmp(key, "min_p") == 0) {
        if (!number_member(j, &v, key)) return 0;
        if (isnan(v)) return 1;
        if (v < 0.0 || v > 1.0) return jfail(j, "min_p must be between 0 and 1");
        req->min_p = (float)v;
        return 1;
    }
    if (strcmp(key, "seed") == 0) {
        if (!number_member(j, &v, key)) return 0;
        if (isnan(v)) return 1;
        if (!integer_in(v, 0.0, 9007199254740991.0)) return jfail(j, "seed must be a non-negative integer");
        req->seed = (uint64_t)v;
        req->has_seed = 1;
        return 1;
    }
    if (strcmp(key, "n") == 0) {
        if (!number_member(j, &v, key)) return 0;
        if (!isnan(v) && v != 1.0) return jfail(j, "only n = 1 is supported");
        return 1;
    }
    if (strcmp(key, "stop") == 0) {
        if (jis_null(j)) return jliteral(j, "null");
        return jfail(j, "stop sequences are not supported; generation stops at the model's end-of-turn token");
    }
    if (strcmp(key, "tools") == 0 || strcmp(key, "functions") == 0) {
        if (jis_null(j)) return jliteral(j, "null");
        return jfail(j, "tools are not supported");
    }
    return jskip(j); /* model, user, stream_options, presence_penalty, ... are accepted and ignored */
}

void rl_chat_request_free(rl_chat_request *req) {
    if (!req) return;
    for (uint32_t i = 0; i < req->message_count; ++i) {
        free(req->messages[i].role);
        free(req->messages[i].content);
    }
    req->message_count = 0;
}

int rl_chat_request_parse(const char *body, size_t len, rl_chat_request *out, char *error, size_t error_cap) {
    memset(out, 0, sizeof(*out));
    out->temperature = -1.0f;
    out->top_p = -1.0f;
    out->top_k = -1;
    out->min_p = -1.0f;
    if (error && error_cap) error[0] = '\0';
    jp j = {body, body + len, 0, error, error_cap};
    int ok = jobject(&j, request_member, out);
    if (ok) {
        jws(&j);
        if (j.p != j.end) ok = jfail(&j, "trailing data after the JSON body");
    }
    if (ok && out->message_count == 0) ok = jfail(&j, "\"messages\" must contain at least one message");
    if (ok && strcmp(out->messages[out->message_count - 1u].role, "assistant") == 0)
        ok = jfail(&j, "the last message must not be from the assistant (continuation is not supported)");
    if (!ok) {
        if (error && error_cap && !error[0]) set_error(error, error_cap, "invalid request body");
        rl_chat_request_free(out);
    }
    return ok;
}

char *rl_server_chatml(const rl_chat_request *req) {
    sb b = {0};
    for (uint32_t i = 0; i < req->message_count; ++i) {
        sb_puts(&b, "<|im_start|>");
        sb_puts(&b, req->messages[i].role);
        sb_puts(&b, "\n");
        sb_puts(&b, req->messages[i].content);
        sb_puts(&b, "<|im_end|>\n");
    }
    sb_puts(&b, "<|im_start|>assistant\n");
    if (b.oom) { sb_free(&b); return NULL; }
    return b.data;
}

/* ------------------------------------------------------------------ HTTP transport */

static int send_all(int fd, const char *p, size_t n) {
    while (n) {
        const ssize_t w = send(fd, p, n, MSG_NOSIGNAL);
        if (w < 0) {
            if (errno == EINTR) continue;
            return 0;
        }
        p += w;
        n -= (size_t)w;
    }
    return 1;
}

static const char *status_text(int status) {
    switch (status) {
        case 200: return "OK";
        case 400: return "Bad Request";
        case 404: return "Not Found";
        case 405: return "Method Not Allowed";
        case 408: return "Request Timeout";
        case 411: return "Length Required";
        case 413: return "Payload Too Large";
        case 500: return "Internal Server Error";
        case 501: return "Not Implemented";
        case 503: return "Service Unavailable";
        default: return "Error";
    }
}

static int send_response(int fd, int status, const char *content_type, const char *body, size_t len) {
    char head[512];
    const int n = snprintf(head, sizeof(head),
        "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\nConnection: close\r\n\r\n",
        status, status_text(status), content_type, len);
    return n > 0 && (size_t)n < sizeof(head) && send_all(fd, head, (size_t)n) && send_all(fd, body, len);
}

static void send_json_error(int fd, int status, const char *message) {
    sb b = {0};
    const char *type = status >= 500 ? "server_error" : "invalid_request_error";
    sb_puts(&b, "{\"error\":{\"message\":");
    json_str(&b, message, strlen(message));
    sb_printf(&b, ",\"type\":\"%s\",\"param\":null,\"code\":null}}", type);
    if (!b.oom) send_response(fd, status, "application/json", b.data, b.len);
    sb_free(&b);
}

typedef struct {
    char method[16];
    char path[256];
    char *body;
    size_t body_len;
} http_request;

/* Returns 0 on success, otherwise the HTTP status to answer with (-1: connection unusable). */
static int read_request(int fd, http_request *req, char *msg, size_t msg_cap) {
    memset(req, 0, sizeof(*req));
    char *buf = (char *)malloc(RL_SERVER_MAX_HEADER + 1u);
    if (!buf) { set_error(msg, msg_cap, "out of memory"); return 500; }
    size_t len = 0;
    char *header_end = NULL;
    while (!header_end) {
        if (len >= RL_SERVER_MAX_HEADER) { free(buf); set_error(msg, msg_cap, "request headers too large"); return 413; }
        const ssize_t r = recv(fd, buf + len, RL_SERVER_MAX_HEADER - len, 0);
        if (r < 0 && errno == EINTR) continue;
        if (r < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) { free(buf); set_error(msg, msg_cap, "request timed out"); return 408; }
        if (r <= 0) { free(buf); return -1; }
        len += (size_t)r;
        buf[len] = '\0';
        header_end = strstr(buf, "\r\n\r\n");
    }
    const size_t header_len = (size_t)(header_end - buf) + 4u;

    /* request line */
    char *line_end = strstr(buf, "\r\n");
    char *sp1 = memchr(buf, ' ', (size_t)(line_end - buf));
    char *sp2 = sp1 ? memchr(sp1 + 1, ' ', (size_t)(line_end - sp1 - 1)) : NULL;
    if (!sp1 || !sp2 || (size_t)(sp1 - buf) >= sizeof(req->method) || (size_t)(sp2 - sp1 - 1) >= sizeof(req->path) ||
        strncmp(sp2 + 1, "HTTP/1.", 7) != 0) {
        free(buf); set_error(msg, msg_cap, "malformed request line"); return 400;
    }
    memcpy(req->method, buf, (size_t)(sp1 - buf));
    memcpy(req->path, sp1 + 1, (size_t)(sp2 - sp1 - 1));
    char *q = strchr(req->path, '?');
    if (q) *q = '\0';

    /* headers */
    long long content_length = -1;
    int chunked = 0;
    for (char *h = line_end + 2; h < header_end;) {
        char *e = strstr(h, "\r\n");
        if (!e || e > header_end) break;
        char *colon = memchr(h, ':', (size_t)(e - h));
        if (colon) {
            const size_t name_len = (size_t)(colon - h);
            char *v = colon + 1;
            while (v < e && (*v == ' ' || *v == '\t')) v++;
            if (name_len == 14 && strncasecmp(h, "Content-Length", 14) == 0) {
                char *endp = NULL;
                errno = 0;
                const long long cl = strtoll(v, &endp, 10);
                while (endp && endp < e && (*endp == ' ' || *endp == '\t')) endp++;
                if (errno || cl < 0 || endp != e || (content_length >= 0 && content_length != cl)) {
                    free(buf); set_error(msg, msg_cap, "invalid Content-Length"); return 400;
                }
                content_length = cl;
            } else if (name_len == 17 && strncasecmp(h, "Transfer-Encoding", 17) == 0) {
                chunked = 1;
            }
        }
        h = e + 2;
    }
    if (chunked) { free(buf); set_error(msg, msg_cap, "chunked request bodies are not supported"); return 501; }
    if (content_length < 0) content_length = 0;
    if (content_length > (long long)RL_SERVER_MAX_BODY) { free(buf); set_error(msg, msg_cap, "request body too large"); return 413; }

    const size_t body_len = (size_t)content_length;
    char *body = (char *)malloc(body_len + 1u);
    if (!body) { free(buf); set_error(msg, msg_cap, "out of memory"); return 500; }
    size_t have = len - header_len;
    if (have > body_len) have = body_len;
    memcpy(body, buf + header_len, have);
    free(buf);
    while (have < body_len) {
        const ssize_t r = recv(fd, body + have, body_len - have, 0);
        if (r < 0 && errno == EINTR) continue;
        if (r < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) { free(body); set_error(msg, msg_cap, "request timed out"); return 408; }
        if (r <= 0) { free(body); return -1; }
        have += (size_t)r;
    }
    body[body_len] = '\0';
    req->body = body;
    req->body_len = body_len;
    return 0;
}

/* ------------------------------------------------------------------ chat completions */

typedef struct {
    int fd;
    int stream;
    int headers_sent;
    int client_gone;
    int stopped;
    volatile sig_atomic_t *stop;
    const char *id;
    const char *model;
    long created;
    sb pending;   /* bytes held back until they form complete UTF-8 */
    sb text;      /* non-streaming: the whole completion */
} sink_state;

static int sse_send(sink_state *s, sb *event) {
    if (event->oom) return 0;
    if (!send_all(s->fd, event->data, event->len)) { s->client_gone = 1; return 0; }
    return 1;
}

static void chunk_prefix(sink_state *s, sb *b) {
    sb_puts(b, "data: {\"id\":");
    json_str(b, s->id, strlen(s->id));
    sb_printf(b, ",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":", s->created);
    json_str(b, s->model, strlen(s->model));
    sb_puts(b, ",\"system_fingerprint\":null,\"choices\":[{\"index\":0,\"delta\":");
}

static int sse_headers(sink_state *s) {
    if (s->headers_sent) return 1;
    static const char head[] =
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n";
    if (!send_all(s->fd, head, sizeof(head) - 1u)) { s->client_gone = 1; return 0; }
    s->headers_sent = 1;
    sb b = {0};
    chunk_prefix(s, &b);
    sb_puts(&b, "{\"role\":\"assistant\",\"content\":\"\"},\"logprobs\":null,\"finish_reason\":null}]}\n\n");
    const int ok = sse_send(s, &b);
    sb_free(&b);
    return ok;
}

static int sse_content(sink_state *s, const char *bytes, size_t n) {
    if (!n) return 1;
    if (!sse_headers(s)) return 0;
    sb b = {0};
    chunk_prefix(s, &b);
    sb_puts(&b, "{\"content\":");
    json_str(&b, bytes, n);
    sb_puts(&b, "},\"logprobs\":null,\"finish_reason\":null}]}\n\n");
    const int ok = sse_send(s, &b);
    sb_free(&b);
    return ok;
}

static int sink_emit(void *user, const char *bytes, size_t len) {
    sink_state *s = (sink_state *)user;
    if (s->client_gone || (s->stop && *s->stop)) { s->stopped = 1; return 0; }
    sb_put(&s->pending, bytes, len);
    if (s->pending.oom) return 0;
    const size_t k = rl_utf8_complete_prefix(s->pending.data, s->pending.len);
    if (!k) return 1;
    int ok = 1;
    if (s->stream) ok = sse_content(s, s->pending.data, k);
    else { sb_put(&s->text, s->pending.data, k); ok = !s->text.oom; }
    memmove(s->pending.data, s->pending.data + k, s->pending.len - k);
    s->pending.len -= k;
    return ok;
}

static void handle_chat(int fd, const rl_server_config *cfg, const rl_server_backend *backend,
                        const http_request *hreq, volatile sig_atomic_t *stop, unsigned long request_no) {
    char error[512] = {0};
    rl_chat_request req;
    if (!rl_chat_request_parse(hreq->body, hreq->body_len, &req, error, sizeof(error))) {
        send_json_error(fd, 400, error);
        return;
    }
    if (!req.max_tokens) req.max_tokens = cfg->default_max_tokens;
    if (req.temperature < 0.0f) req.temperature = cfg->default_temperature;
    if (req.top_p < 0.0f) req.top_p = cfg->default_top_p;
    if (req.top_k < 0) req.top_k = (int32_t)cfg->default_top_k;
    if (req.min_p < 0.0f) req.min_p = cfg->default_min_p;

    char id[64];
    const long created = (long)time(NULL);
    snprintf(id, sizeof(id), "chatcmpl-redlite-%ld-%lu", created, request_no);
    sink_state s = {0};
    s.fd = fd;
    s.stream = req.stream;
    s.stop = stop;
    s.id = id;
    s.model = backend->model_id;
    s.created = created;

    rl_server_result result = {0};
    int status = 500;
    const int ok = backend->generate(backend->ctx, &req, sink_emit, &s, &result, &status, error, sizeof(error));
    rl_chat_request_free(&req);

    /* flush a held-back incomplete UTF-8 tail (escaped as U+FFFD) */
    if (ok && s.pending.len && !s.client_gone) {
        if (s.stream) sse_content(&s, s.pending.data, s.pending.len);
        else sb_put(&s.text, s.pending.data, s.pending.len);
        s.pending.len = 0;
    }
    const char *finish = result.finish_length ? "length" : "stop";

    if (!ok) {
        if (!error[0]) snprintf(error, sizeof(error), "generation failed");
        fprintf(stderr, "[redlite-server] request %lu failed: %s\n", request_no, error);
        if (s.stream && s.headers_sent) {
            sb b = {0};
            sb_puts(&b, "data: {\"error\":{\"message\":");
            json_str(&b, error, strlen(error));
            sb_puts(&b, ",\"type\":\"server_error\",\"param\":null,\"code\":null}}\n\n");
            sse_send(&s, &b);
            sb_free(&b);
        } else if (!s.client_gone) {
            send_json_error(fd, status == 400 ? 400 : 500, error);
        }
    } else if (s.client_gone) {
        fprintf(stderr, "[redlite-server] request %lu: client disconnected after %u tokens\n", request_no, result.completion_tokens);
    } else if (s.stream) {
        if (sse_headers(&s)) {
            sb b = {0};
            chunk_prefix(&s, &b);
            sb_printf(&b, "{},\"logprobs\":null,\"finish_reason\":\"%s\"}],\"usage\":{\"prompt_tokens\":%u,\"completion_tokens\":%u,\"total_tokens\":%u}}\n\ndata: [DONE]\n\n",
                finish, result.prompt_tokens, result.completion_tokens, result.prompt_tokens + result.completion_tokens);
            sse_send(&s, &b);
            sb_free(&b);
        }
    } else {
        sb b = {0};
        sb_puts(&b, "{\"id\":");
        json_str(&b, id, strlen(id));
        sb_printf(&b, ",\"object\":\"chat.completion\",\"created\":%ld,\"model\":", created);
        json_str(&b, backend->model_id, strlen(backend->model_id));
        sb_puts(&b, ",\"system_fingerprint\":null,\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":");
        json_str(&b, s.text.data ? s.text.data : "", s.text.len);
        sb_printf(&b, "},\"logprobs\":null,\"finish_reason\":\"%s\"}],\"usage\":{\"prompt_tokens\":%u,\"completion_tokens\":%u,\"total_tokens\":%u}}",
            finish, result.prompt_tokens, result.completion_tokens, result.prompt_tokens + result.completion_tokens);
        if (b.oom) send_json_error(fd, 500, "out of memory");
        else send_response(fd, 200, "application/json", b.data, b.len);
        sb_free(&b);
    }
    if (ok && !s.client_gone) {
        fprintf(stderr, "[redlite-server] request %lu: prompt=%u completion=%u finish=%s%s\n", request_no,
            result.prompt_tokens, result.completion_tokens, finish, s.stopped ? " (shutdown)" : "");
    }
    sb_free(&s.pending);
    sb_free(&s.text);
}

static void handle_connection(int fd, const rl_server_config *cfg, const rl_server_backend *backend,
                              volatile sig_atomic_t *stop, unsigned long request_no) {
    char msg[256] = {0};
    http_request req;
    const int status = read_request(fd, &req, msg, sizeof(msg));
    if (status < 0) return;
    if (status > 0) { send_json_error(fd, status, msg); return; }

    if (strcmp(req.path, "/health") == 0 || strcmp(req.path, "/v1/health") == 0) {
        if (strcmp(req.method, "GET") != 0) send_json_error(fd, 405, "use GET");
        else send_response(fd, 200, "application/json", "{\"status\":\"ok\"}", 15u);
    } else if (strcmp(req.path, "/v1/models") == 0) {
        if (strcmp(req.method, "GET") != 0) send_json_error(fd, 405, "use GET");
        else {
            sb b = {0};
            sb_puts(&b, "{\"object\":\"list\",\"data\":[{\"id\":");
            json_str(&b, backend->model_id, strlen(backend->model_id));
            sb_puts(&b, ",\"object\":\"model\",\"created\":0,\"owned_by\":\"redlite\"}]}");
            if (!b.oom) send_response(fd, 200, "application/json", b.data, b.len);
            sb_free(&b);
        }
    } else if (strcmp(req.path, "/v1/chat/completions") == 0) {
        if (strcmp(req.method, "POST") != 0) send_json_error(fd, 405, "use POST");
        else handle_chat(fd, cfg, backend, &req, stop, request_no);
    } else {
        send_json_error(fd, 404, "unknown endpoint");
    }
    free(req.body);
}

void rl_server_config_default(rl_server_config *cfg) {
    memset(cfg, 0, sizeof(*cfg));
    cfg->host = "127.0.0.1";
    cfg->port = 8080;
    cfg->default_max_tokens = 256;
    cfg->default_temperature = 0.7f;
    cfg->default_top_p = 0.95f;
    cfg->default_top_k = 40;
    cfg->default_min_p = 0.0f;
    cfg->read_timeout_s = 30;
}

int rl_server_run(const rl_server_config *cfg, const rl_server_backend *backend,
                  volatile sig_atomic_t *stop, char *error, size_t error_cap) {
    signal(SIGPIPE, SIG_IGN);
    char port_text[16];
    snprintf(port_text, sizeof(port_text), "%u", (unsigned)cfg->port);
    struct addrinfo hints, *res = NULL;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;
    const int gai = getaddrinfo(cfg->host, port_text, &hints, &res);
    if (gai != 0) { set_error(error, error_cap, "cannot resolve %s: %s", cfg->host, gai_strerror(gai)); return 0; }
    int lfd = -1;
    for (struct addrinfo *ai = res; ai; ai = ai->ai_next) {
        lfd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (lfd < 0) continue;
        const int one = 1;
        setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
        if (bind(lfd, ai->ai_addr, ai->ai_addrlen) == 0 && listen(lfd, 16) == 0) break;
        close(lfd);
        lfd = -1;
    }
    freeaddrinfo(res);
    if (lfd < 0) { set_error(error, error_cap, "cannot listen on %s:%u: %s", cfg->host, (unsigned)cfg->port, strerror(errno)); return 0; }

    struct sockaddr_storage bound;
    socklen_t bound_len = sizeof(bound);
    unsigned port = cfg->port;
    if (getsockname(lfd, (struct sockaddr *)&bound, &bound_len) == 0) {
        if (bound.ss_family == AF_INET) port = ntohs(((struct sockaddr_in *)&bound)->sin_port);
        else if (bound.ss_family == AF_INET6) port = ntohs(((struct sockaddr_in6 *)&bound)->sin6_port);
    }
    printf("redlite-server listening on http://%s:%u (model %s)\n", cfg->host, port, backend->model_id);
    fflush(stdout);

    unsigned long request_no = 0;
    while (!(stop && *stop)) {
        struct pollfd p = {lfd, POLLIN, 0};
        const int pr = poll(&p, 1, 250);
        if (pr < 0 && errno != EINTR) { set_error(error, error_cap, "poll failed: %s", strerror(errno)); close(lfd); return 0; }
        if (pr <= 0) continue;
        const int cfd = accept(lfd, NULL, NULL);
        if (cfd < 0) continue;
#ifdef SO_NOSIGPIPE
        { const int one = 1; setsockopt(cfd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one)); }
#endif
        struct timeval tv = {cfg->read_timeout_s > 0 ? cfg->read_timeout_s : 30, 0};
        setsockopt(cfd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        handle_connection(cfd, cfg, backend, stop, ++request_no);
        shutdown(cfd, SHUT_WR);
        close(cfd);
    }
    close(lfd);
    return 1;
}

/* ------------------------------------------------------------------ selftest */

static int expect_parse_error(const char *body, const char *needle, char *error, size_t cap) {
    rl_chat_request r;
    char e[256] = {0};
    if (rl_chat_request_parse(body, strlen(body), &r, e, sizeof(e))) {
        rl_chat_request_free(&r);
        set_error(error, cap, "accepted invalid body: %s", body);
        return 0;
    }
    if (needle && !strstr(e, needle)) { set_error(error, cap, "error '%s' lacks '%s' for %s", e, needle, body); return 0; }
    return 1;
}

int rl_server_selftest(char *error, size_t cap) {
    const char *body =
        "{\"model\":\"x\",\"stream\":true,\"max_tokens\":12,\"temperature\":0,\"top_p\":0.5,\"top_k\":3,\"min_p\":0.05,\"seed\":7,"
        "\"stream_options\":{\"include_usage\":true},\"stop\":null,"
        "\"messages\":[{\"role\":\"developer\",\"content\":\"Be brief.\"},"
        "{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"Ciao \"},{\"type\":\"text\",\"text\":\"\\u00e8 \\ud83d\\ude00\\n\"}]},"
        "{\"role\":\"assistant\",\"content\":\"Hi\",\"tool_calls\":null},"
        "{\"role\":\"user\",\"content\":\"q\\\"\\\\/\"}]}";
    rl_chat_request r;
    char e[256] = {0};
    if (!rl_chat_request_parse(body, strlen(body), &r, e, sizeof(e))) { set_error(error, cap, "valid body rejected: %s", e); return 0; }
    int ok = r.stream == 1 && r.max_tokens == 12u && r.temperature == 0.0f && r.top_p == 0.5f && r.top_k == 3 && r.min_p == 0.05f &&
             r.has_seed && r.seed == 7u && r.message_count == 4u &&
             strcmp(r.messages[0].role, "system") == 0 &&
             strcmp(r.messages[1].content, "Ciao \xC3\xA8 \xF0\x9F\x98\x80\n") == 0 &&
             strcmp(r.messages[3].content, "q\"\\/") == 0;
    char *prompt = ok ? rl_server_chatml(&r) : NULL;
    ok = ok && prompt && strcmp(prompt,
        "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nCiao \xC3\xA8 \xF0\x9F\x98\x80\n<|im_end|>\n"
        "<|im_start|>assistant\nHi<|im_end|>\n<|im_start|>user\nq\"\\/<|im_end|>\n<|im_start|>assistant\n") == 0;
    free(prompt);
    rl_chat_request_free(&r);
    if (!ok) { set_error(error, cap, "parsed request or ChatML prompt differs from the expected values"); return 0; }

    /* the single-user prompt matches the CLI template (rl_tokenizer_chat_prompt) byte for byte */
    const char *single = "{\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}";
    if (!rl_chat_request_parse(single, strlen(single), &r, e, sizeof(e))) { set_error(error, cap, "minimal body rejected: %s", e); return 0; }
    prompt = rl_server_chatml(&r);
    ok = prompt && strcmp(prompt, "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n") == 0 &&
         r.stream == 0 && r.max_tokens == 0u && r.temperature < 0.0f && r.top_k < 0 && r.min_p < 0.0f && !r.has_seed;
    free(prompt);
    rl_chat_request_free(&r);
    if (!ok) { set_error(error, cap, "minimal request defaults or prompt are wrong"); return 0; }

    if (!expect_parse_error("", NULL, error, cap) ||
        !expect_parse_error("{\"messages\":[]}", "at least one", error, cap) ||
        !expect_parse_error("{\"messages\":[{\"role\":\"tool\",\"content\":\"x\"}]}", "role", error, cap) ||
        !expect_parse_error("{\"messages\":[{\"role\":\"user\"}]}", "content", error, cap) ||
        !expect_parse_error("{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"stop\":[\"a\"]}", "stop", error, cap) ||
        !expect_parse_error("{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"n\":2}", "n = 1", error, cap) ||
        !expect_parse_error("{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"temperature\":3}", "temperature", error, cap) ||
        !expect_parse_error("{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"max_tokens\":0}", "max_tokens", error, cap) ||
        !expect_parse_error("{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"max_tokens\":\"5\"}", "number", error, cap) ||
        !expect_parse_error("{\"messages\":[{\"role\":\"user\",\"content\":\"x\"},{\"role\":\"assistant\",\"content\":\"y\"}]}", "assistant", error, cap) ||
        !expect_parse_error("{\"messages\":[{\"role\":\"user\",\"content\":\"a\\u0000\"}]}", "NUL", error, cap) ||
        !expect_parse_error("{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]} x", "trailing", error, cap) ||
        !expect_parse_error("{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"tools\":[{}]}", "tools", error, cap) ||
        !expect_parse_error("{\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"image_url\"}]}]}", "text", error, cap)) {
        return 0;
    }
    /* deep nesting is bounded, not recursive without limit */
    {
        sb deep = {0};
        sb_puts(&deep, "{\"x\":");
        for (int i = 0; i < 1000; ++i) sb_puts(&deep, "[");
        for (int i = 0; i < 1000; ++i) sb_puts(&deep, "]");
        sb_puts(&deep, ",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}");
        ok = !deep.oom && expect_parse_error(deep.data, "deep", error, cap);
        sb_free(&deep);
        if (!ok) return 0;
    }

    /* UTF-8 hold-back: "è" = C3 A8, "😀" = F0 9F 98 80 */
    if (rl_utf8_complete_prefix("ab\xC3", 3) != 2u || rl_utf8_complete_prefix("ab\xC3\xA8", 4) != 4u ||
        rl_utf8_complete_prefix("\xF0\x9F\x98", 3) != 0u || rl_utf8_complete_prefix("x\xFFy", 3) != 3u ||
        rl_utf8_complete_prefix("", 0) != 0u) {
        set_error(error, cap, "UTF-8 complete-prefix is wrong");
        return 0;
    }
    /* escaping yields valid JSON text: controls escaped, invalid UTF-8 replaced */
    sb b = {0};
    json_escape(&b, "a\"\\\n\x01\xC3\xA8\xFF", 8);
    ok = !b.oom && strcmp(b.data, "a\\\"\\\\\\n\\u0001\xC3\xA8\xEF\xBF\xBD") == 0;
    sb_free(&b);
    if (!ok) { set_error(error, cap, "JSON escaping is wrong"); return 0; }
    return 1;
}
