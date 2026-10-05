#define _POSIX_C_SOURCE 200809L
#define _DARWIN_C_SOURCE 1

#include "redlite_native_server.h"

#include <errno.h>
#include <math.h>
#include <netdb.h>
#include <netinet/in.h>
#include <poll.h>
#include <pthread.h>
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
                    if (c < 0x20) sb_printf(b, "\\u%04x", (unsigned)c);
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

/* ------------------------------------------------------------------ dev59: JSON values (tools, tool-call arguments) */

struct rl_json {
    char type;              /* 'o' object, 'a' array, 's' string, 'n' number, 't' true, 'f' false, 'z' null */
    char *str;              /* the string (unescaped UTF-8) or the number as written */
    uint32_t count;
    char **keys;            /* object member names, in order */
    struct rl_json *items;
};

static void rl_json_clear(struct rl_json *v) {
    if (!v) return;
    for (uint32_t i = 0; i < v->count; ++i) { if (v->keys) free(v->keys[i]); rl_json_clear(&v->items[i]); }
    free(v->keys); free(v->items); free(v->str);
    memset(v, 0, sizeof(*v));
}

static void rl_json_free(struct rl_json *v) { rl_json_clear(v); free(v); }

static int jvalue(jp *j, struct rl_json *out) {
    memset(out, 0, sizeof(*out));
    jws(j);
    if (j->p >= j->end) return jfail(j, "unexpected end of JSON");
    const char c = *j->p;
    if (c == '"') { out->type = 's'; return jstring(j, &out->str); }
    if (c == 't') { out->type = 't'; return jliteral(j, "true"); }
    if (c == 'f') { out->type = 'f'; return jliteral(j, "false"); }
    if (c == 'n') { out->type = 'z'; return jliteral(j, "null"); }
    if (c == '{' || c == '[') {
        const char close = c == '{' ? '}' : ']';
        out->type = c == '{' ? 'o' : 'a';
        if (++j->depth > RL_JSON_MAX_DEPTH) return jfail(j, "JSON nesting too deep");
        j->p++;
        jws(j);
        if (j->p < j->end && *j->p == close) { j->p++; j->depth--; return 1; }
        for (;;) {
            jws(j);
            char *key = NULL;
            if (out->type == 'o') {
                if (!jstring(j, &key)) return 0;
                jws(j);
                if (j->p >= j->end || *j->p != ':') { free(key); return jfail(j, "expected ':' in JSON object"); }
                j->p++;
            }
            struct rl_json *items = (struct rl_json *)realloc(out->items, (out->count + 1u) * sizeof(*items));
            if (!items) { free(key); return jfail(j, "out of memory"); }
            out->items = items;
            if (out->type == 'o') {
                char **keys = (char **)realloc(out->keys, (out->count + 1u) * sizeof(*keys));
                if (!keys) { free(key); return jfail(j, "out of memory"); }
                out->keys = keys;
                keys[out->count] = key;
            }
            memset(&out->items[out->count], 0, sizeof(*items));
            out->count++;
            if (!jvalue(j, &out->items[out->count - 1u])) return 0;
            jws(j);
            if (j->p < j->end && *j->p == ',') { j->p++; continue; }
            if (j->p < j->end && *j->p == close) { j->p++; j->depth--; return 1; }
            return jfail(j, "expected ',' or closing bracket in JSON");
        }
    }
    const char *s = j->p;
    if (!jnumber(j, NULL)) return 0;
    out->type = 'n';
    out->str = strndup(s, (size_t)(j->p - s));
    return out->str ? 1 : jfail(j, "out of memory");
}

/* A JSON value in a fresh allocation (NULL on failure). */
static struct rl_json *jvalue_new(jp *j) {
    struct rl_json *v = (struct rl_json *)calloc(1u, sizeof(*v));
    if (!v) { jfail(j, "out of memory"); return NULL; }
    if (!jvalue(j, v)) { rl_json_free(v); return NULL; }
    return v;
}

/* Parses text as one JSON value (NULL when it is not exactly one). */
static struct rl_json *jparse_text(const char *text, size_t len) {
    char err[8] = {0};
    jp j = {text, text + len, 0, err, sizeof(err)};
    struct rl_json *v = jvalue_new(&j);
    if (v) { jws(&j); if (j.p != j.end) { rl_json_free(v); v = NULL; } }
    return v;
}

static const struct rl_json *jget(const struct rl_json *o, const char *key) {
    if (!o || o->type != 'o') return NULL;
    for (uint32_t i = 0; i < o->count; ++i) if (strcmp(o->keys[i], key) == 0) return &o->items[i];
    return NULL;
}

/* Python's json.dumps(value, ensure_ascii=False) layout, as chat templates' tojson writes it. Numbers keep their
 * written form (ponytail: json.dumps would print 1E5 as 100000.0; clients send plain numbers). */
static void jdump(sb *b, const struct rl_json *v) {
    switch (v->type) {
        case 's': json_str(b, v->str, strlen(v->str)); break;
        case 'n': sb_puts(b, v->str); break;
        case 't': sb_puts(b, "true"); break;
        case 'f': sb_puts(b, "false"); break;
        case 'z': sb_puts(b, "null"); break;
        default: {
            const int obj = v->type == 'o';
            sb_puts(b, obj ? "{" : "[");
            for (uint32_t i = 0; i < v->count; ++i) {
                if (i) sb_puts(b, ", ");
                if (obj) { json_str(b, v->keys[i], strlen(v->keys[i])); sb_puts(b, ": "); }
                jdump(b, &v->items[i]);
            }
            sb_puts(b, obj ? "}" : "]");
        }
    }
}

typedef struct {
    char *role;
    sb content;
    int have_content;
    int content_null;
    rl_tool_call calls[RL_SERVER_MAX_TOOL_CALLS];
    uint32_t call_count;
} msg_state;

static void tool_call_clear(rl_tool_call *c) {
    free(c->id); free(c->name); rl_json_free(c->arguments);
    memset(c, 0, sizeof(*c));
}

static int tool_call_function_member(jp *j, const char *key, void *user) {
    rl_tool_call *c = (rl_tool_call *)user;
    if (strcmp(key, "name") == 0) { free(c->name); c->name = NULL; return jstring(j, &c->name); }
    if (strcmp(key, "arguments") == 0) {
        rl_json_free(c->arguments);
        c->arguments = jvalue_new(j);
        if (!c->arguments) return 0;
        if (c->arguments->type != 's' && c->arguments->type != 'o') return jfail(j, "tool call arguments must be a string or an object");
        return 1;
    }
    return jskip(j);
}

static int tool_call_member(jp *j, const char *key, void *user) {
    rl_tool_call *c = (rl_tool_call *)user;
    if (strcmp(key, "id") == 0) { free(c->id); c->id = NULL; return jis_null(j) ? jliteral(j, "null") : jstring(j, &c->id); }
    if (strcmp(key, "function") == 0) return jobject(j, tool_call_function_member, c);
    return jskip(j);   /* type ("function"), index */
}

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
        m->have_content = 1;
        if (jis_null(j)) { m->content_null = 1; m->content.len = 0; return jliteral(j, "null"); }   /* checked below */
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
    if (strcmp(key, "tool_calls") == 0) {
        if (jis_null(j)) return jliteral(j, "null");
        if (j->p >= j->end || *j->p != '[') return jfail(j, "\"tool_calls\" must be an array");
        if (++j->depth > RL_JSON_MAX_DEPTH) return jfail(j, "JSON nesting too deep");
        j->p++;
        jws(j);
        if (j->p < j->end && *j->p == ']') { j->p++; j->depth--; return 1; }
        for (;;) {
            if (m->call_count >= RL_SERVER_MAX_TOOL_CALLS) return jfail(j, "too many tool calls in one message");
            rl_tool_call *c = &m->calls[m->call_count++];
            if (!jobject(j, tool_call_member, c)) return 0;
            if (!c->name || !c->name[0]) return jfail(j, "every tool call needs a function name");
            jws(j);
            if (j->p < j->end && *j->p == ',') { j->p++; jws(j); continue; }
            if (j->p < j->end && *j->p == ']') { j->p++; j->depth--; return 1; }
            return jfail(j, "expected ',' or ']' in tool_calls");
        }
    }
    if (strcmp(key, "function_call") == 0) {
        if (jis_null(j)) return jliteral(j, "null");
        return jfail(j, "\"function_call\" is not supported; use \"tool_calls\"");
    }
    return jskip(j);   /* tool_call_id, name, ... */
}

static void msg_state_free(msg_state *m) {
    free(m->role); sb_free(&m->content);
    for (uint32_t i = 0; i < m->call_count; ++i) tool_call_clear(&m->calls[i]);
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
        if (req->message_count == req->message_cap) {
            const uint32_t cap = req->message_cap ? 2u * req->message_cap : 16u;
            rl_chat_message *grown = (rl_chat_message *)realloc(req->messages, (size_t)cap * sizeof(*grown));
            if (!grown) return jfail(j, "out of memory");
            req->messages = grown;
            req->message_cap = cap;
        }
        msg_state m = {0};
        if (!jobject(j, message_member, &m)) { msg_state_free(&m); return 0; }
        if (!m.role) { msg_state_free(&m); return jfail(j, "every message needs a \"role\""); }
        if (strcmp(m.role, "developer") == 0) { free(m.role); m.role = strdup("system"); }
        if (!m.role || (strcmp(m.role, "system") != 0 && strcmp(m.role, "user") != 0 && strcmp(m.role, "assistant") != 0 &&
                        strcmp(m.role, "tool") != 0)) {
            msg_state_free(&m);
            return jfail(j, "message role must be system, user, assistant or tool");
        }
        const int assistant = strcmp(m.role, "assistant") == 0;
        if (m.call_count && !assistant) { msg_state_free(&m); return jfail(j, "only assistant messages can have tool_calls"); }
        if ((!m.have_content && !m.call_count) || (m.content_null && !assistant)) {
            msg_state_free(&m);
            return jfail(j, "every message needs \"content\" (null only for an assistant message)");
        }
        if (!m.content.data) { m.content.data = (char *)calloc(1u, 1u); if (!m.content.data) { msg_state_free(&m); return jfail(j, "out of memory"); } }
        rl_chat_message *msg = &req->messages[req->message_count];
        msg->role = m.role;
        msg->content = m.content.data;
        memcpy(msg->tool_calls, m.calls, sizeof(m.calls));
        msg->tool_call_count = m.call_count;
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

static int stop_string(jp *j, rl_chat_request *req) {
    if (req->stop_count >= RL_SERVER_MAX_STOP) return jfail(j, "at most 4 stop sequences are supported");
    char *s = NULL;
    if (!jstring(j, &s)) return 0;
    const size_t n = strlen(s);
    if (!n || n > RL_SERVER_MAX_STOP_BYTES) { free(s); return jfail(j, "stop sequences must be 1 to 256 bytes long"); }
    req->stop[req->stop_count++] = s;
    return 1;
}

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
        for (uint32_t i = 0; i < req->stop_count; ++i) { free(req->stop[i]); req->stop[i] = NULL; }
        req->stop_count = 0;
        if (j->p < j->end && *j->p == '"') return stop_string(j, req);
        if (j->p >= j->end || *j->p != '[') return jfail(j, "\"stop\" must be a string or an array of strings");
        if (++j->depth > RL_JSON_MAX_DEPTH) return jfail(j, "JSON nesting too deep");
        j->p++;
        jws(j);
        if (j->p < j->end && *j->p == ']') { j->p++; j->depth--; return 1; }
        for (;;) {
            if (!stop_string(j, req)) return 0;
            jws(j);
            if (j->p < j->end && *j->p == ',') { j->p++; jws(j); continue; }
            if (j->p < j->end && *j->p == ']') { j->p++; j->depth--; return 1; }
            return jfail(j, "expected ',' or ']' in stop");
        }
    }
    if (strcmp(key, "tools") == 0) {
        rl_json_free(req->tools);
        req->tools = NULL;
        if (jis_null(j)) return jliteral(j, "null");
        req->tools = jvalue_new(j);
        if (!req->tools) return 0;
        if (req->tools->type != 'a') return jfail(j, "\"tools\" must be an array");
        for (uint32_t i = 0; i < req->tools->count; ++i) {
            const struct rl_json *t = &req->tools->items[i], *f = jget(t, "function");
            const struct rl_json *name = jget(f ? f : t, "name");
            if (t->type != 'o' || !name || name->type != 's' || !name->str[0]) return jfail(j, "every tool needs a function with a name");
        }
        return 1;
    }
    if (strcmp(key, "tool_choice") == 0) {   /* "none" drops the tools; "auto", "required" and objects act as auto */
        if (j->p < j->end && *j->p == '"') {
            char *choice = NULL;
            if (!jstring(j, &choice)) return 0;
            if (strcmp(choice, "none") == 0) req->tool_choice_none = 1;
            free(choice);
            return 1;
        }
        return jskip(j);
    }
    if (strcmp(key, "functions") == 0) {
        if (jis_null(j)) return jliteral(j, "null");
        return jfail(j, "\"functions\" is not supported; use \"tools\"");
    }
    return jskip(j); /* model, user, stream_options, presence_penalty, ... are accepted and ignored */
}

void rl_chat_request_free(rl_chat_request *req) {
    if (!req) return;
    for (uint32_t i = 0; i < req->message_count; ++i) {
        free(req->messages[i].role);
        free(req->messages[i].content);
        for (uint32_t k = 0; k < req->messages[i].tool_call_count; ++k) tool_call_clear(&req->messages[i].tool_calls[k]);
    }
    free(req->messages);
    req->messages = NULL;
    req->message_count = req->message_cap = 0;
    rl_json_free(req->tools);
    req->tools = NULL;
    for (uint32_t i = 0; i < req->stop_count; ++i) { free(req->stop[i]); req->stop[i] = NULL; }
    req->stop_count = 0;
}

uint32_t rl_prefix_reuse(const uint32_t *history, uint32_t history_len, const uint32_t *ids, uint32_t count) {
    if (!history || !ids || !history_len || history_len >= count) return 0;
    return memcmp(history, ids, (size_t)history_len * sizeof(uint32_t)) == 0 ? history_len : 0u;
}

int rl_stop_scan(const char *buf, size_t len, char *const *stops, uint32_t stop_count, size_t *emit_len) {
    size_t first = SIZE_MAX, hold = 0;
    for (uint32_t i = 0; i < stop_count; ++i) {
        const size_t n = strlen(stops[i]);
        if (!n) continue;
        for (size_t at = 0; at + n <= len && at < first; ++at)
            if (memcmp(buf + at, stops[i], n) == 0) { first = at; break; }
        /* longest proper prefix of stops[i] that ends buf */
        for (size_t k = n - 1u < len ? n - 1u : len; k > hold; --k)
            if (memcmp(buf + len - k, stops[i], k) == 0) { hold = k; break; }
    }
    if (first != SIZE_MAX) { *emit_len = first; return 1; }
    *emit_len = len - hold;
    return 0;
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
    if (ok && out->tools && (out->tool_choice_none || out->tools->count == 0)) { rl_json_free(out->tools); out->tools = NULL; }
    if (!ok) {
        if (error && error_cap && !error[0]) set_error(error, error_cap, "invalid request body");
        rl_chat_request_free(out);
    }
    return ok;
}

/* ------------------------------------------------------------------ dev59: prompt rendering
 * Each branch follows the model's own Jinja chat template line by line (tests/fixtures/chat_template_*.jinja; the
 * protocol tests render both with Jinja and compare byte for byte). */

static int is_role(const rl_chat_message *m, const char *role) { return strcmp(m->role, role) == 0; }

static void put_trimmed(sb *b, const char *s) {   /* Python str.strip() */
    size_t n = strlen(s);
    while (n && strchr(" \t\n\r\f\v", s[n - 1u])) n--;
    while (n && strchr(" \t\n\r\f\v", *s)) { s++; n--; }
    sb_put(b, s, n);
}

static void put_msg(sb *b, const rl_chat_message *m) {
    sb_puts(b, "<|im_start|>"); sb_puts(b, m->role); sb_puts(b, "\n"); sb_puts(b, m->content); sb_puts(b, "<|im_end|>\n");
}

/* Qwen3-Next Instruct: tools as JSON lines in the system message, calls as {"name": ..., "arguments": ...}. */
static void render_json(sb *b, const rl_chat_request *req) {
    const uint32_t n = req->message_count;
    const rl_chat_message *ms = req->messages;
    if (req->tools) {
        sb_puts(b, "<|im_start|>system\n");
        if (is_role(&ms[0], "system")) { sb_puts(b, ms[0].content); sb_puts(b, "\n\n"); }
        sb_puts(b, "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
                   "You are provided with function signatures within <tools></tools> XML tags:\n<tools>");
        for (uint32_t i = 0; i < req->tools->count; ++i) { sb_puts(b, "\n"); jdump(b, &req->tools->items[i]); }
        sb_puts(b, "\n</tools>\n\nFor each function call, return a json object with function name and arguments within "
                   "<tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n"
                   "</tool_call><|im_end|>\n");
    } else if (is_role(&ms[0], "system")) {
        put_msg(b, &ms[0]);
    }
    for (uint32_t i = 0; i < n; ++i) {
        const rl_chat_message *m = &ms[i];
        if (is_role(m, "user") || (is_role(m, "system") && i > 0)) {
            put_msg(b, m);
        } else if (is_role(m, "assistant")) {
            sb_puts(b, "<|im_start|>assistant\n"); sb_puts(b, m->content);
            for (uint32_t k = 0; k < m->tool_call_count; ++k) {
                const rl_tool_call *c = &m->tool_calls[k];
                if (k || m->content[0]) sb_puts(b, "\n");
                sb_puts(b, "<tool_call>\n{\"name\": \""); sb_puts(b, c->name); sb_puts(b, "\", \"arguments\": ");
                /* arguments in json.dumps form whatever the client did to them: agents re-serialize the JSON they
                 * got back (pi sends {"a":1} for the model's {"a": 1}), and a turn that differs from what the model
                 * wrote loses the engine state (dev59b: F2 reused 0 % of the prompt in an agent loop) */
                struct rl_json *parsed = c->arguments && c->arguments->type == 's' ?
                    jparse_text(c->arguments->str, strlen(c->arguments->str)) : NULL;
                if (!c->arguments) sb_puts(b, "{}");
                else if (parsed) jdump(b, parsed);
                else if (c->arguments->type == 's') sb_puts(b, c->arguments->str);
                else jdump(b, c->arguments);
                rl_json_free(parsed);
                sb_puts(b, "}\n</tool_call>");
            }
            sb_puts(b, "<|im_end|>\n");
        } else if (is_role(m, "tool")) {
            if (i == 0 || !is_role(&ms[i - 1u], "tool")) sb_puts(b, "<|im_start|>user");
            sb_puts(b, "\n<tool_response>\n"); sb_puts(b, m->content); sb_puts(b, "\n</tool_response>");
            if (i + 1u == n || !is_role(&ms[i + 1u], "tool")) sb_puts(b, "<|im_end|>\n");
        }
    }
}

/* render_extra_keys of the Qwen3-Coder template: the members not in skip, as <key>value</key> lines. */
static void xml_extra_keys(sb *b, const struct rl_json *o, const char *const *skip) {
    if (!o || o->type != 'o') return;
    for (uint32_t i = 0; i < o->count; ++i) {
        int skipped = 0;
        for (const char *const *s = skip; *s; ++s) if (strcmp(o->keys[i], *s) == 0) skipped = 1;
        if (skipped) continue;
        sb_puts(b, "\n<"); sb_puts(b, o->keys[i]); sb_puts(b, ">");
        if (o->items[i].type == 's') sb_puts(b, o->items[i].str); else jdump(b, &o->items[i]);
        sb_puts(b, "</"); sb_puts(b, o->keys[i]); sb_puts(b, ">");
    }
}

/* Jinja's `string` filter on a scalar (ponytail: a list renders as JSON here, as a Python repr there). */
static void put_jinja_string(sb *b, const struct rl_json *v) {
    if (v->type == 's') sb_puts(b, v->str);
    else if (v->type == 't') sb_puts(b, "True");
    else if (v->type == 'f') sb_puts(b, "False");
    else if (v->type == 'z') sb_puts(b, "None");
    else jdump(b, v);
}

/* Qwen3-Coder: tools as XML in the system message, calls as <function=name><parameter=key>value</parameter>. */
static void render_xml(sb *b, const rl_chat_request *req) {
    const uint32_t n = req->message_count;
    const rl_chat_message *ms = req->messages;
    const int has_system = is_role(&ms[0], "system");
    const uint32_t first = has_system ? 1u : 0u;
    if (has_system) { sb_puts(b, "<|im_start|>system\n"); sb_puts(b, ms[0].content); }
    else if (req->tools) sb_puts(b, "<|im_start|>system\nYou are Qwen, a helpful AI assistant that can interact with a computer to solve tasks.");
    if (req->tools) {
        static const char *const fn_skip[] = {"type", "name", "description", "parameters", NULL};
        static const char *const params_skip[] = {"type", "properties", NULL};
        static const char *const param_skip[] = {"name", "type", "description", NULL};
        sb_puts(b, "\n\n# Tools\n\nYou have access to the following functions:\n\n<tools>");
        for (uint32_t i = 0; i < req->tools->count; ++i) {
            const struct rl_json *t = &req->tools->items[i];
            if (jget(t, "function")) t = jget(t, "function");
            const struct rl_json *name = jget(t, "name"), *desc = jget(t, "description"), *params = jget(t, "parameters");
            sb_puts(b, "\n<function>\n<name>"); if (name) put_jinja_string(b, name); sb_puts(b, "</name>");
            if (desc) { sb_puts(b, "\n<description>"); if (desc->type == 's') put_trimmed(b, desc->str); else put_jinja_string(b, desc); sb_puts(b, "</description>"); }
            sb_puts(b, "\n<parameters>");
            const struct rl_json *props = jget(params, "properties");
            if (props && props->type == 'o') {
                for (uint32_t k = 0; k < props->count; ++k) {
                    const struct rl_json *f = &props->items[k], *ty = jget(f, "type"), *pd = jget(f, "description");
                    sb_puts(b, "\n<parameter>\n<name>"); sb_puts(b, props->keys[k]); sb_puts(b, "</name>");
                    if (ty) { sb_puts(b, "\n<type>"); put_jinja_string(b, ty); sb_puts(b, "</type>"); }
                    if (pd) { sb_puts(b, "\n<description>"); if (pd->type == 's') put_trimmed(b, pd->str); else put_jinja_string(b, pd); sb_puts(b, "</description>"); }
                    xml_extra_keys(b, f, param_skip);
                    sb_puts(b, "\n</parameter>");
                }
            }
            xml_extra_keys(b, params, params_skip);
            sb_puts(b, "\n</parameters>");
            xml_extra_keys(b, t, fn_skip);
            sb_puts(b, "\n</function>");
        }
        sb_puts(b, "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
                   "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
                   "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n"
                   "</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified "
                   "format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
                   "- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in "
                   "natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer "
                   "the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>");
    }
    if (has_system || req->tools) sb_puts(b, "<|im_end|>\n");
    for (uint32_t i = first; i < n; ++i) {
        const rl_chat_message *m = &ms[i];
        if (is_role(m, "assistant") && m->tool_call_count) {
            sb_puts(b, "<|im_start|>assistant");
            size_t len = strlen(m->content);
            const char *c = m->content;
            while (len && strchr(" \t\n\r\f\v", *c)) { c++; len--; }
            if (len) { sb_puts(b, "\n"); put_trimmed(b, m->content); sb_puts(b, "\n"); }
            for (uint32_t k = 0; k < m->tool_call_count; ++k) {
                const rl_tool_call *tc = &m->tool_calls[k];
                sb_puts(b, "\n<tool_call>\n<function="); sb_puts(b, tc->name); sb_puts(b, ">\n");
                struct rl_json *parsed = NULL;
                const struct rl_json *args = tc->arguments;
                if (args && args->type == 's') { parsed = jparse_text(args->str, strlen(args->str)); args = parsed; }
                if (args && args->type == 'o') {
                    for (uint32_t a = 0; a < args->count; ++a) {
                        sb_puts(b, "<parameter="); sb_puts(b, args->keys[a]); sb_puts(b, ">\n");
                        if (args->items[a].type == 's') sb_puts(b, args->items[a].str); else jdump(b, &args->items[a]);
                        sb_puts(b, "\n</parameter>\n");
                    }
                }
                rl_json_free(parsed);
                sb_puts(b, "</function>\n</tool_call>");
            }
            sb_puts(b, "<|im_end|>\n");
        } else if (is_role(m, "tool")) {
            if (i > first && !is_role(&ms[i - 1u], "tool")) sb_puts(b, "<|im_start|>user");   /* loop.previtem */
            sb_puts(b, "\n<tool_response>\n"); sb_puts(b, m->content); sb_puts(b, "\n</tool_response>");
            if (i + 1u == n || !is_role(&ms[i + 1u], "tool")) sb_puts(b, "<|im_end|>\n");
        } else {
            put_msg(b, m);
        }
    }
}

char *rl_server_prompt(const rl_chat_request *req, int tool_format) {
    sb b = {0};
    if (req->message_count) {
        if (tool_format == RL_TOOLS_XML) render_xml(&b, req);
        else render_json(&b, req);
    }
    sb_puts(&b, "<|im_start|>assistant\n");
    if (b.oom) { sb_free(&b); return NULL; }
    return b.data;
}

char *rl_server_chatml(const rl_chat_request *req) { return rl_server_prompt(req, RL_TOOLS_JSON); }

int rl_server_tool_format(const char *chat_template) {
    return chat_template && strstr(chat_template, "<function=") ? RL_TOOLS_XML : RL_TOOLS_JSON;
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
    sb pending;   /* bytes held back until they form complete UTF-8 and cannot start a stop sequence */
    sb text;      /* non-streaming: the whole completion */
    char *const *stops;
    uint32_t stop_count;
    int stop_matched;  /* a stop sequence ended the completion */
    int tools_on;      /* dev59: the request has tools: <tool_call> switches the rest of the output to capture */
    int capturing;
    sb calls;          /* the output from the first <tool_call> on */
} sink_state;

static const char RL_TOOL_OPEN[] = "<tool_call>";
static const char RL_TOOL_CLOSE[] = "</tool_call>";

static int is_space(char c) { return c == ' ' || c == '\n' || c == '\t' || c == '\r'; }

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

static int sink_out(sink_state *s, const char *bytes, size_t n) {
    if (!n) return 1;
    if (s->stream) return sse_content(s, bytes, n);
    sb_put(&s->text, bytes, n);
    return !s->text.oom;
}

static int sink_emit(void *user, const char *bytes, size_t len) {
    sink_state *s = (sink_state *)user;
    if (s->stop_matched) return 0;
    if (s->client_gone || (s->stop && *s->stop)) { s->stopped = 1; return 0; }
    if (s->capturing) { sb_put(&s->calls, bytes, len); return !s->calls.oom; }
    sb_put(&s->pending, bytes, len);
    if (s->pending.oom) return 0;
    size_t k = s->pending.len;
    if (s->tools_on) {
        char *marker[1] = {(char *)RL_TOOL_OPEN};
        size_t tcut = 0, scut = SIZE_MAX;
        const int found = rl_stop_scan(s->pending.data, s->pending.len, marker, 1u, &tcut);
        const int stop_first = s->stop_count && rl_stop_scan(s->pending.data, s->pending.len, s->stops, s->stop_count, &scut) &&
                               scut < tcut;
        if (found && !stop_first) {
            /* the content before the call, without the one newline the template puts back between them: a model that
             * leaves a blank line (".\n\n<tool_call>", Qwen3-Next does) keeps ".\n", so the turn renders back
             * exactly as written and the next request reuses the state (dev59b) */
            size_t c = tcut;
            if (c && s->pending.data[c - 1u] == '\n') c--;
            const int ok = sink_out(s, s->pending.data, c);
            s->capturing = 1;
            sb_put(&s->calls, s->pending.data + tcut, s->pending.len - tcut);
            s->pending.len = 0;
            return ok && !s->calls.oom;
        }
        if (tcut < k) k = tcut;   /* a partial <tool_call> is held back */
    }
    if (s->stop_count) {
        size_t cut = 0;
        if (rl_stop_scan(s->pending.data, s->pending.len, s->stops, s->stop_count, &cut)) {
            /* the text before the stop sequence is complete UTF-8 (the stop sequence starts a character) */
            s->stop_matched = 1;
            const int ok = sink_out(s, s->pending.data, cut);
            s->pending.len = 0;
            (void)ok;
            return 0;
        }
        if (cut < k) k = cut;
    }
    if (s->tools_on) while (k && is_space(s->pending.data[k - 1u])) k--;   /* whitespace may precede a call */
    const size_t u = rl_utf8_complete_prefix(s->pending.data, k);
    if (!u) return 1;
    const int ok = sink_out(s, s->pending.data, u);
    memmove(s->pending.data, s->pending.data + u, s->pending.len - u);
    s->pending.len -= u;
    return ok;
}

/* ------------------------------------------------------------------ dev59: tool calls in the output */

typedef struct {
    char *name;
    char *arguments;   /* JSON text, as the OpenAI API carries it */
} out_call;

typedef struct { char **name; char **arguments; } call_slots;

static int out_call_member(jp *j, const char *key, void *user) {
    call_slots *c = (call_slots *)user;
    if (strcmp(key, "name") == 0) { free(*c->name); *c->name = NULL; return jstring(j, c->name); }
    if (strcmp(key, "arguments") == 0) {
        free(*c->arguments); *c->arguments = NULL;
        if (j->p < j->end && *j->p == '"') return jstring(j, c->arguments);   /* arguments written as a JSON string */
        const char *start = j->p;
        if (!jskip(j)) return 0;
        *c->arguments = strndup(start, (size_t)(j->p - start));   /* the model's own text: rendered back unchanged */
        return *c->arguments ? 1 : jfail(j, "out of memory");
    }
    return jskip(j);
}

/* The schema type of parameter key of tool name ("string", "integer", ...), or NULL. */
static const char *param_type(const struct rl_json *tools, const char *name, const char *key) {
    for (uint32_t i = 0; tools && i < tools->count; ++i) {
        const struct rl_json *t = &tools->items[i];
        if (jget(t, "function")) t = jget(t, "function");
        const struct rl_json *n = jget(t, "name");
        if (!n || n->type != 's' || strcmp(n->str, name) != 0) continue;
        const struct rl_json *ty = jget(jget(jget(jget(t, "parameters"), "properties"), key), "type");
        return ty && ty->type == 's' ? ty->str : NULL;
    }
    return NULL;
}

static const char *find(const char *p, const char *end, const char *needle) {
    const size_t n = strlen(needle);
    for (; p + n <= end; ++p) if (memcmp(p, needle, n) == 0) return p;
    return NULL;
}

/* <function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n...</function> -> name and a JSON object of the values.
 * A value is a JSON string unless the tool's schema gives it another type and it parses as JSON. */
static int parse_xml_call(const char *p, const char *end, const struct rl_json *tools, out_call *c) {
    while (p < end && is_space(*p)) p++;
    if (!(end - p > 10 && memcmp(p, "<function=", 10) == 0)) return 0;
    p += 10;
    const char *gt = memchr(p, '>', (size_t)(end - p));
    if (!gt || gt == p) return 0;
    c->name = strndup(p, (size_t)(gt - p));
    if (!c->name) return 0;
    p = gt + 1;
    sb a = {0};
    sb_puts(&a, "{");
    uint32_t count = 0;
    for (;;) {
        while (p < end && is_space(*p)) p++;
        if (end - p >= 11 && memcmp(p, "</function>", 11) == 0) { p += 11; break; }
        /* <parameter=KEY>; dev62: also <KEY>, which the 2-bit Qwen3-Coder sometimes writes ("<command>" ...
         * "</parameter>"), closed by </parameter> or </KEY> */
        const int short_form = !(end - p > 11 && memcmp(p, "<parameter=", 11) == 0);
        if (short_form && !(end - p > 2 && p[0] == '<' && p[1] != '/')) { sb_free(&a); return 0; }
        p += short_form ? 1 : 11;
        const char *kgt = memchr(p, '>', (size_t)(end - p));
        if (!kgt || kgt == p) { sb_free(&a); return 0; }
        for (const char *k = p; k < kgt; ++k)
            if (!((*k >= 'a' && *k <= 'z') || (*k >= 'A' && *k <= 'Z') || (*k >= '0' && *k <= '9') || *k == '_' || *k == '-')) {
                sb_free(&a); return 0;
            }
        char *key = strndup(p, (size_t)(kgt - p));
        const char *v = kgt + 1, *vend = find(v, end, "</parameter>"), *close_end = vend ? vend + 12 : NULL;
        if (key && short_form) {
            char tag[80];
            snprintf(tag, sizeof(tag), "</%.70s>", key);
            const char *alt = find(v, end, tag);
            if (alt && (!vend || alt < vend)) { vend = alt; close_end = alt + strlen(tag); }
        }
        if (!key || !vend) { free(key); sb_free(&a); return 0; }
        p = close_end;
        if (v < vend && *v == '\n') v++;
        if (vend > v && vend[-1] == '\n') vend--;
        if (count++) sb_puts(&a, ", ");
        json_str(&a, key, strlen(key));
        sb_puts(&a, ": ");
        const char *ty = param_type(tools, c->name, key);
        struct rl_json *parsed = ty && strcmp(ty, "string") != 0 ? jparse_text(v, (size_t)(vend - v)) : NULL;
        if (parsed) jdump(&a, parsed); else json_str(&a, v, (size_t)(vend - v));
        rl_json_free(parsed);
        free(key);
    }
    while (p < end && is_space(*p)) p++;
    sb_puts(&a, "}");
    if (p != end || a.oom) { sb_free(&a); return 0; }
    c->arguments = a.data;
    return 1;
}

/* dev60: right content, wrong closing brackets. At 2 bits Qwen3-Next closes nested arguments as '...}}]}' or
 * '...}}}}' instead of '...}]}' (pi's edit tool: 3 of 3 runs). When everything from the first bracket that does not
 * match (or from the end of the value) is closers and whitespace, that tail is replaced by the closers the open
 * brackets need. Returns a malloc'd repaired text, or NULL when the error is anything else. */
static char *repair_closers(const char *p, const char *end) {
    char stack[RL_JSON_MAX_DEPTH];
    int depth = 0, in_str = 0, started = 0;
    const char *q = p;
    for (; q < end; ++q) {
        const char c = *q;
        if (in_str) {
            if (c == '\\') { if (++q >= end) return NULL; continue; }
            if (c == '"') in_str = 0;
            continue;
        }
        if (c == '"') { in_str = 1; continue; }
        if (c == '{' || c == '[') { if (depth == RL_JSON_MAX_DEPTH) return NULL; stack[depth++] = c; started = 1; continue; }
        if (c == '}' || c == ']') {
            if (!depth || stack[depth - 1] != (c == '}' ? '{' : '[')) break;
            if (--depth == 0) { ++q; break; }
        }
    }
    if (in_str || !started) return NULL;
    for (const char *r = q; r < end; ++r) if (!is_space(*r) && *r != '}' && *r != ']') return NULL;
    sb b = {0};
    sb_put(&b, p, (size_t)(q - p));
    while (depth > 0) sb_put(&b, stack[--depth] == '{' ? "}" : "]", 1u);
    if (b.oom) { sb_free(&b); return NULL; }
    return b.data;
}

static int parse_json_call(const char *p, const char *end, out_call *c) {
    char err[8] = {0};
    jp j = {p, end, 0, err, sizeof(err)};
    call_slots slots = {&c->name, &c->arguments};
    if (!jobject(&j, out_call_member, &slots)) return 0;
    jws(&j);
    if (j.p != end || !c->name || !c->name[0]) return 0;
    if (!c->arguments && !(c->arguments = strdup("{}"))) return 0;
    return 1;
}

/* The captured output: one or more <tool_call>...</tool_call> blocks and nothing else. 0 when it is not that. */
static int parse_tool_calls(const char *text, size_t len, int format, const struct rl_json *tools, out_call *calls, uint32_t *count) {
    const char *p = text, *end = text + len;
    *count = 0;
    for (;;) {
        while (p < end && is_space(*p)) p++;
        if (p == end) return *count > 0;
        const size_t ol = sizeof(RL_TOOL_OPEN) - 1u;
        if ((size_t)(end - p) < ol || memcmp(p, RL_TOOL_OPEN, ol) != 0 || *count >= RL_SERVER_MAX_TOOL_CALLS) return 0;
        p += ol;
        const char *close = find(p, end, RL_TOOL_CLOSE);
        if (!close) return 0;
        out_call *c = &calls[(*count)++];
        if (format == RL_TOOLS_XML) {
            if (!parse_xml_call(p, close, tools, c)) return 0;
        } else if (!parse_json_call(p, close, c)) {
            free(c->name); free(c->arguments); c->name = c->arguments = NULL;
            const char *s = p;
            while (s < close && is_space(*s)) s++;
            char *fixed = repair_closers(s, close);
            const int ok = fixed && parse_json_call(fixed, fixed + strlen(fixed), c);
            free(fixed);
            if (!ok) return 0;
        }
        p = close + sizeof(RL_TOOL_CLOSE) - 1u;
    }
}

static void put_tool_calls(sb *b, const out_call *calls, uint32_t count, const char *id, int with_index) {
    sb_puts(b, "[");
    for (uint32_t i = 0; i < count; ++i) {
        if (i) sb_puts(b, ",");
        sb_puts(b, "{");
        if (with_index) sb_printf(b, "\"index\":%u,", i);
        sb_printf(b, "\"id\":\"call_%s_%u\",\"type\":\"function\",\"function\":{\"name\":", id, i);
        json_str(b, calls[i].name, strlen(calls[i].name));
        sb_puts(b, ",\"arguments\":");
        json_str(b, calls[i].arguments, strlen(calls[i].arguments));
        sb_puts(b, "}}");
    }
    sb_puts(b, "]");
}

/* Runs one parsed chat request (worker thread); frees it. */
static void run_chat(int fd, const rl_server_backend *backend, rl_chat_request *reqp,
                     volatile sig_atomic_t *stop, unsigned long request_no) {
    char error[512] = {0};
    rl_chat_request req = *reqp;
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
    s.stops = req.stop;
    s.stop_count = req.stop_count;
    s.tools_on = req.tools != NULL;

    rl_server_result result = {0};
    int status = 500;
    const int ok = backend->generate(backend->ctx, &req, sink_emit, &s, &result, &status, error, sizeof(error));

    /* flush a held-back tail: a partial stop-sequence prefix, or incomplete UTF-8 (escaped as U+FFFD) */
    if (ok && s.pending.len && !s.client_gone && !s.stop_matched) {
        sink_out(&s, s.pending.data, s.pending.len);
        s.pending.len = 0;
    }
    /* dev59: the captured calls; output that is not a well-formed call goes out as text, nothing is lost */
    out_call calls[RL_SERVER_MAX_TOOL_CALLS];
    memset(calls, 0, sizeof(calls));
    uint32_t call_count = 0;
    if (ok && s.capturing && !s.client_gone) {
        if (!parse_tool_calls(s.calls.data, s.calls.len, backend->tool_format, req.tools, calls, &call_count)) {
            /* dev62: the model wrote <tool_call> but not a well-formed call; what it wrote goes to the client as text */
            fprintf(stderr, "[redlite-server] request %lu: unparsed tool call (%zu bytes, sent as text): %.*s\n", request_no,
                    s.calls.len, (int)(s.calls.len < 400u ? s.calls.len : 400u), s.calls.data);
            for (uint32_t i = 0; i < RL_SERVER_MAX_TOOL_CALLS; ++i) { free(calls[i].name); free(calls[i].arguments); }
            memset(calls, 0, sizeof(calls));
            call_count = 0;
            sink_out(&s, s.calls.data, s.calls.len);
        }
    }
    rl_chat_request_free(&req);
    const char *finish = call_count ? "tool_calls" : result.finish_length && !s.stop_matched ? "length" : "stop";

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
        if (call_count && sse_headers(&s)) {
            sb b = {0};
            chunk_prefix(&s, &b);
            sb_puts(&b, "{\"tool_calls\":");
            put_tool_calls(&b, calls, call_count, id, 1);
            sb_puts(&b, "},\"logprobs\":null,\"finish_reason\":null}]}\n\n");
            sse_send(&s, &b);
            sb_free(&b);
        }
        if (sse_headers(&s)) {
            sb b = {0};
            chunk_prefix(&s, &b);
            sb_printf(&b, "{},\"logprobs\":null,\"finish_reason\":\"%s\"}],\"usage\":{\"prompt_tokens\":%u,\"completion_tokens\":%u,\"total_tokens\":%u,"
                "\"prompt_tokens_details\":{\"cached_tokens\":%u}}}\n\ndata: [DONE]\n\n",
                finish, result.prompt_tokens, result.completion_tokens, result.prompt_tokens + result.completion_tokens, result.cached_tokens);
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
        if (call_count && !s.text.len) sb_puts(&b, "null");
        else json_str(&b, s.text.data ? s.text.data : "", s.text.len);
        if (call_count) { sb_puts(&b, ",\"tool_calls\":"); put_tool_calls(&b, calls, call_count, id, 0); }
        sb_printf(&b, "},\"logprobs\":null,\"finish_reason\":\"%s\"}],\"usage\":{\"prompt_tokens\":%u,\"completion_tokens\":%u,\"total_tokens\":%u,"
            "\"prompt_tokens_details\":{\"cached_tokens\":%u}}}",
            finish, result.prompt_tokens, result.completion_tokens, result.prompt_tokens + result.completion_tokens, result.cached_tokens);
        if (b.oom) send_json_error(fd, 500, "out of memory");
        else send_response(fd, 200, "application/json", b.data, b.len);
        sb_free(&b);
    }
    if (ok && !s.client_gone) {
        fprintf(stderr, "[redlite-server] request %lu: prompt=%u cached=%u completion=%u finish=%s%s%s\n", request_no,
            result.prompt_tokens, result.cached_tokens, result.completion_tokens, finish,
            s.stop_matched ? " (stop sequence)" : "", s.stopped ? " (shutdown)" : "");
    }
    for (uint32_t i = 0; i < RL_SERVER_MAX_TOOL_CALLS; ++i) { free(calls[i].name); free(calls[i].arguments); }
    sb_free(&s.pending);
    sb_free(&s.text);
    sb_free(&s.calls);
}

/* ------------------------------------------------------------------ FIFO request queue (dev29) */

typedef struct chat_job {
    int fd;
    unsigned long request_no;
    rl_chat_request req;
    struct chat_job *next;
} chat_job;

typedef struct {
    pthread_mutex_t mu;
    pthread_cond_t cv;
    chat_job *head, *tail;
    uint32_t waiting;     /* queued, not yet started */
    uint32_t running;     /* workers inside a generation */
    int closing;
    const rl_server_backend *backend;
    volatile sig_atomic_t *stop;
    uint32_t workers;
} chat_queue;

static void close_client(int fd) {
    shutdown(fd, SHUT_WR);
    close(fd);
}

/* A queued client that gave up (closed its socket) is skipped instead of generating for nobody. */
static int client_closed(int fd) {
    struct pollfd p = {fd, POLLIN, 0};
    if (poll(&p, 1, 0) <= 0) return 0;
    if (p.revents & (POLLHUP | POLLERR)) return 1;
    char c;
    const ssize_t r = recv(fd, &c, 1u, MSG_PEEK | MSG_DONTWAIT);
    return r == 0;
}

static void *chat_worker(void *user) {
    chat_queue *q = (chat_queue *)user;
    for (;;) {
        pthread_mutex_lock(&q->mu);
        while (!q->head && !q->closing) pthread_cond_wait(&q->cv, &q->mu);
        chat_job *job = q->head;
        if (!job) { pthread_mutex_unlock(&q->mu); break; }
        q->head = job->next;
        if (!q->head) q->tail = NULL;
        q->waiting--;
        q->running++;
        pthread_mutex_unlock(&q->mu);

        if (q->stop && *q->stop) {
            send_json_error(job->fd, 503, "server is shutting down");
            rl_chat_request_free(&job->req);
        } else if (client_closed(job->fd)) {
            fprintf(stderr, "[redlite-server] request %lu: client closed the connection while queued\n", job->request_no);
            rl_chat_request_free(&job->req);
        } else {
            run_chat(job->fd, q->backend, &job->req, q->stop, job->request_no);
        }
        close_client(job->fd);
        free(job);
        pthread_mutex_lock(&q->mu);
        q->running--;
        pthread_mutex_unlock(&q->mu);
    }
    return NULL;
}

static void handle_connection(int fd, const rl_server_config *cfg, const rl_server_backend *backend,
                              chat_queue *q, unsigned long request_no) {
    char msg[256] = {0};
    http_request req;
    const int status = read_request(fd, &req, msg, sizeof(msg));
    if (status < 0) { close_client(fd); return; }
    if (status > 0) { send_json_error(fd, status, msg); close_client(fd); return; }

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
        if (strcmp(req.method, "POST") != 0) { send_json_error(fd, 405, "use POST"); free(req.body); close_client(fd); return; }
        chat_job *job = (chat_job *)calloc(1u, sizeof(chat_job));
        char error[512] = {0};
        if (!job) { send_json_error(fd, 500, "out of memory"); free(req.body); close_client(fd); return; }
        if (!rl_chat_request_parse(req.body, req.body_len, &job->req, error, sizeof(error))) {
            send_json_error(fd, 400, error);
            free(job); free(req.body); close_client(fd);
            return;
        }
        free(req.body);
        rl_chat_request *r = &job->req;
        if (!r->max_tokens) r->max_tokens = cfg->default_max_tokens;
        if (r->temperature < 0.0f) r->temperature = cfg->default_temperature;
        if (r->top_p < 0.0f) r->top_p = cfg->default_top_p;
        if (r->top_k < 0) r->top_k = (int32_t)cfg->default_top_k;
        if (r->min_p < 0.0f) r->min_p = cfg->default_min_p;
        job->fd = fd;
        job->request_no = request_no;
        pthread_mutex_lock(&q->mu);
        const uint32_t limit = cfg->queue_max;
        if (q->waiting > limit || (q->waiting == limit && q->running >= q->workers)) {   /* queue_max may wait behind the running ones */
            const uint32_t waiting = q->waiting;
            pthread_mutex_unlock(&q->mu);
            char busy[128];
            snprintf(busy, sizeof(busy), "server busy: %u requests already waiting (--queue %u)", waiting, limit);
            send_json_error(fd, 503, busy);
            rl_chat_request_free(&job->req);
            free(job);
            close_client(fd);
            return;
        }
        const uint32_t ahead = q->running >= q->workers ? q->waiting + q->running : q->waiting;
        if (q->tail) q->tail->next = job; else q->head = job;
        q->tail = job;
        q->waiting++;
        pthread_cond_signal(&q->cv);
        pthread_mutex_unlock(&q->mu);
        if (ahead) fprintf(stderr, "[redlite-server] request %lu queued behind %u\n", request_no, ahead);
        return; /* the worker answers and closes fd */
    } else {
        send_json_error(fd, 404, "unknown endpoint");
    }
    free(req.body);
    close_client(fd);
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
    cfg->queue_max = 16;
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
        if (bind(lfd, ai->ai_addr, ai->ai_addrlen) == 0 && listen(lfd, 64) == 0) break;
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

    chat_queue q;
    memset(&q, 0, sizeof(q));
    q.backend = backend;
    q.stop = stop;
    q.workers = cfg->workers ? (cfg->workers < 8u ? cfg->workers : 8u) : 1u;
    pthread_t worker[8];
    uint32_t started = 0;
    int init_ok = pthread_mutex_init(&q.mu, NULL) == 0 && pthread_cond_init(&q.cv, NULL) == 0;
    while (init_ok && started < q.workers && pthread_create(&worker[started], NULL, chat_worker, &q) == 0) started++;
    if (started < q.workers) {
        set_error(error, error_cap, "cannot start the request worker threads");
        pthread_mutex_lock(&q.mu); q.closing = 1; pthread_cond_broadcast(&q.cv); pthread_mutex_unlock(&q.mu);
        for (uint32_t w = 0; w < started; ++w) pthread_join(worker[w], NULL);
        close(lfd);
        return 0;
    }
    printf("redlite-server listening on http://%s:%u (model %s)\n", cfg->host, port, backend->model_id);
    fflush(stdout);

    unsigned long request_no = 0;
    int rc = 1;
    while (!(stop && *stop)) {
        struct pollfd p = {lfd, POLLIN, 0};
        const int pr = poll(&p, 1, 250);
        if (pr < 0 && errno != EINTR) { set_error(error, error_cap, "poll failed: %s", strerror(errno)); rc = 0; break; }
        if (pr <= 0) continue;
        const int cfd = accept(lfd, NULL, NULL);
        if (cfd < 0) continue;
#ifdef SO_NOSIGPIPE
        { const int one = 1; setsockopt(cfd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one)); }
#endif
        struct timeval tv = {cfg->read_timeout_s > 0 ? cfg->read_timeout_s : 30, 0};
        setsockopt(cfd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        handle_connection(cfd, cfg, backend, &q, ++request_no);
    }
    close(lfd);
    /* the running generation sees *stop through its sink; queued requests are answered 503 */
    pthread_mutex_lock(&q.mu);
    q.closing = 1;
    pthread_cond_broadcast(&q.cv);
    pthread_mutex_unlock(&q.mu);
    for (uint32_t w = 0; w < started; ++w) pthread_join(worker[w], NULL);
    pthread_cond_destroy(&q.cv);
    pthread_mutex_destroy(&q.mu);
    return rc;
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
        !expect_parse_error("{\"messages\":[{\"role\":\"function\",\"content\":\"x\"}]}", "role", error, cap) ||
        !expect_parse_error("{\"messages\":[{\"role\":\"user\",\"content\":null}]}", "null", error, cap) ||
        !expect_parse_error("{\"messages\":[{\"role\":\"user\",\"content\":\"x\",\"tool_calls\":[{\"function\":{\"name\":\"f\"}}]}]}", "assistant", error, cap) ||
        !expect_parse_error("{\"messages\":[{\"role\":\"user\"}]}", "content", error, cap) ||
        !expect_parse_error("{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"stop\":[\"a\",\"b\",\"c\",\"d\",\"e\"]}", "at most 4", error, cap) ||
        !expect_parse_error("{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"stop\":\"\"}", "1 to 256", error, cap) ||
        !expect_parse_error("{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"stop\":[1]}", "string", error, cap) ||
        !expect_parse_error("{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"stop\":5}", "stop", error, cap) ||
        !expect_parse_error("{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"n\":2}", "n = 1", error, cap) ||
        !expect_parse_error("{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"temperature\":3}", "temperature", error, cap) ||
        !expect_parse_error("{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"max_tokens\":0}", "max_tokens", error, cap) ||
        !expect_parse_error("{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"max_tokens\":\"5\"}", "number", error, cap) ||
        !expect_parse_error("{\"messages\":[{\"role\":\"user\",\"content\":\"x\"},{\"role\":\"assistant\",\"content\":\"y\"}]}", "assistant", error, cap) ||
        !expect_parse_error("{\"messages\":[{\"role\":\"user\",\"content\":\"a\\u0000\"}]}", "NUL", error, cap) ||
        !expect_parse_error("{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]} x", "trailing", error, cap) ||
        !expect_parse_error("{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"tools\":[{}]}", "tool", error, cap) ||
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

    /* stop sequences: a string or an array, stored in order */
    {
        const char *sb1 = "{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"stop\":\"END\"}";
        const char *sb2 = "{\"stop\":[\"\\n\\n\",\"</s>\"],\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}";
        if (!rl_chat_request_parse(sb1, strlen(sb1), &r, e, sizeof(e))) { set_error(error, cap, "stop string rejected: %s", e); return 0; }
        ok = r.stop_count == 1u && strcmp(r.stop[0], "END") == 0;
        rl_chat_request_free(&r);
        if (ok && !rl_chat_request_parse(sb2, strlen(sb2), &r, e, sizeof(e))) { set_error(error, cap, "stop array rejected: %s", e); return 0; }
        ok = ok && r.stop_count == 2u && strcmp(r.stop[0], "\n\n") == 0 && strcmp(r.stop[1], "</s>") == 0;
        if (ok) rl_chat_request_free(&r);
        if (!ok) { set_error(error, cap, "stop sequences parsed wrongly"); return 0; }
    }
    /* stop scan: earliest match wins, partial prefixes at the end are held back */
    {
        char *stops[2] = {(char *)"END", (char *)"xyz"};
        size_t cut = 0;
        const int m1 = rl_stop_scan("hello E", 7u, stops, 2u, &cut);
        const size_t c1 = cut;
        const int m2 = rl_stop_scan("hello EN", 8u, stops, 2u, &cut);
        const size_t c2 = cut;
        const int m3 = rl_stop_scan("a xyz b END", 11u, stops, 2u, &cut);
        const size_t c3 = cut;
        const int m4 = rl_stop_scan("ENDING", 6u, stops, 2u, &cut);
        const size_t c4 = cut;
        const int m5 = rl_stop_scan("plain", 5u, stops, 2u, &cut);
        const size_t c5 = cut;
        const int m6 = rl_stop_scan("x", 1u, stops, 2u, &cut);
        const size_t c6 = cut;
        if (m1 || c1 != 6u || m2 || c2 != 6u || !m3 || c3 != 2u || !m4 || c4 != 0u || m5 || c5 != 5u || m6 || c6 != 0u) {
            set_error(error, cap, "stop scan is wrong (%d/%zu %d/%zu %d/%zu %d/%zu %d/%zu %d/%zu)", m1, c1, m2, c2, m3, c3, m4, c4, m5, c5, m6, c6);
            return 0;
        }
    }
    /* prefix reuse: only a proper prefix of the new prompt is reused */
    {
        const uint32_t h[3] = {1u, 2u, 3u}, a4[4] = {1u, 2u, 3u, 4u}, b4[4] = {1u, 2u, 9u, 4u};
        if (rl_prefix_reuse(h, 3u, a4, 4u) != 3u || rl_prefix_reuse(h, 3u, a4, 3u) != 0u || rl_prefix_reuse(h, 3u, b4, 4u) != 0u ||
            rl_prefix_reuse(h, 0u, a4, 4u) != 0u || rl_prefix_reuse(a4, 4u, h, 3u) != 0u) {
            set_error(error, cap, "prefix reuse is wrong");
            return 0;
        }
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
