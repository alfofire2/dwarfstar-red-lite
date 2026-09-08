#define _POSIX_C_SOURCE 200809L

#include "redlite_native_tokenizer.h"
#include "redlite_native_unicode_data.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* llama.cpp token attribute/type values */
enum {
    RL_TOKTYPE_NORMAL = 1,
    RL_TOKTYPE_UNKNOWN = 2,
    RL_TOKTYPE_CONTROL = 3,
    RL_TOKTYPE_USER_DEFINED = 4,
    RL_TOKTYPE_UNUSED = 5,
    RL_TOKTYPE_BYTE = 6,
};

enum {
    FLAG_UNDEFINED = 0x0001, FLAG_NUMBER = 0x0002, FLAG_LETTER = 0x0004, FLAG_SEPARATOR = 0x0008,
    FLAG_ACCENT = 0x0010, FLAG_PUNCT = 0x0020, FLAG_SYMBOL = 0x0040, FLAG_CONTROL = 0x0080,
    FLAG_WHITESPACE = 0x0100,
};

/* ---- string hash map (open addressing) ---- */

typedef struct { const char *key; size_t len; int32_t value; } map_entry;
typedef struct { map_entry *slots; size_t cap; } strmap;

static uint64_t fnv1a(const char *s, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) { h ^= (uint8_t)s[i]; h *= 1099511628211ull; }
    return h;
}

static int strmap_init(strmap *m, size_t count) {
    size_t cap = 16;
    while (cap < count * 2u) cap <<= 1;
    m->slots = (map_entry *)calloc(cap, sizeof(map_entry));
    m->cap = cap;
    return m->slots != NULL;
}

static void strmap_put(strmap *m, const char *key, size_t len, int32_t value) {
    size_t i = (size_t)fnv1a(key, len) & (m->cap - 1u);
    while (m->slots[i].key) {
        if (m->slots[i].len == len && memcmp(m->slots[i].key, key, len) == 0) { m->slots[i].value = value; return; }
        i = (i + 1u) & (m->cap - 1u);
    }
    m->slots[i].key = key; m->slots[i].len = len; m->slots[i].value = value;
}

static int32_t strmap_get(const strmap *m, const char *key, size_t len) {
    size_t i = (size_t)fnv1a(key, len) & (m->cap - 1u);
    while (m->slots[i].key) {
        if (m->slots[i].len == len && memcmp(m->slots[i].key, key, len) == 0) return m->slots[i].value;
        i = (i + 1u) & (m->cap - 1u);
    }
    return -1;
}

/* ---- tokenizer ---- */

struct rl_tokenizer {
    const rl_gguf_model *model;
    uint32_t vocab;
    strmap token_ids;      /* token text -> id */
    strmap merge_ranks;    /* "left right" -> rank */
    char *merge_storage;   /* owned copies of merge strings */
    uint32_t *special;     /* special token ids sorted by text length desc */
    uint32_t special_count;
    char byte_to_utf8[256][4];
    uint8_t byte_to_utf8_len[256];
    int32_t byte_token[256];   /* id of the token whose text is byte_to_utf8[b] */
    int32_t eos, bos, endoftext, im_end;
};

static uint32_t utf8_encode(uint32_t cp, char *out) {
    if (cp < 0x80u) { out[0] = (char)cp; return 1; }
    if (cp < 0x800u) { out[0] = (char)(0xC0u | (cp >> 6)); out[1] = (char)(0x80u | (cp & 0x3Fu)); return 2; }
    if (cp < 0x10000u) { out[0] = (char)(0xE0u | (cp >> 12)); out[1] = (char)(0x80u | ((cp >> 6) & 0x3Fu)); out[2] = (char)(0x80u | (cp & 0x3Fu)); return 3; }
    out[0] = (char)(0xF0u | (cp >> 18)); out[1] = (char)(0x80u | ((cp >> 12) & 0x3Fu)); out[2] = (char)(0x80u | ((cp >> 6) & 0x3Fu)); out[3] = (char)(0x80u | (cp & 0x3Fu)); return 4;
}

/* decode one code point; invalid sequences yield the byte value (like llama.cpp's fallback) */
static uint32_t utf8_decode(const char *s, size_t len, size_t *consumed) {
    const uint8_t *p = (const uint8_t *)s;
    if (!(p[0] & 0x80u)) { *consumed = 1; return p[0]; }
    if ((p[0] & 0xE0u) == 0xC0u && len >= 2 && (p[1] & 0xC0u) == 0x80u) { *consumed = 2; return ((uint32_t)(p[0] & 0x1Fu) << 6) | (p[1] & 0x3Fu); }
    if ((p[0] & 0xF0u) == 0xE0u && len >= 3 && (p[1] & 0xC0u) == 0x80u && (p[2] & 0xC0u) == 0x80u) { *consumed = 3; return ((uint32_t)(p[0] & 0x0Fu) << 12) | ((uint32_t)(p[1] & 0x3Fu) << 6) | (p[2] & 0x3Fu); }
    if ((p[0] & 0xF8u) == 0xF0u && len >= 4 && (p[1] & 0xC0u) == 0x80u && (p[2] & 0xC0u) == 0x80u && (p[3] & 0xC0u) == 0x80u) { *consumed = 4; return ((uint32_t)(p[0] & 0x07u) << 18) | ((uint32_t)(p[1] & 0x3Fu) << 12) | ((uint32_t)(p[2] & 0x3Fu) << 6) | (p[3] & 0x3Fu); }
    *consumed = 1;
    return p[0];
}

static uint16_t cpt_flags(uint32_t cp) {
    /* binary search the range table: flags apply from start until the next start */
    size_t lo = 0, hi = RL_UNICODE_RANGE_COUNT;
    if (cp < rl_unicode_ranges[0].start) return FLAG_UNDEFINED;
    while (hi - lo > 1) {
        const size_t mid = (lo + hi) / 2u;
        if (rl_unicode_ranges[mid].start <= cp) lo = mid; else hi = mid;
    }
    uint16_t flags = rl_unicode_ranges[lo].flags;
    for (size_t i = 0; i < RL_UNICODE_WHITESPACE_COUNT; ++i) if (rl_unicode_whitespace[i] == cp) { flags |= FLAG_WHITESPACE; break; }
    return flags;
}

static uint32_t cpt_tolower(uint32_t cp) {
    size_t lo = 0, hi = RL_UNICODE_LOWERCASE_COUNT;
    while (lo < hi) {
        const size_t mid = (lo + hi) / 2u;
        if (rl_unicode_lowercase[mid].from < cp) lo = mid + 1u; else hi = mid;
    }
    return (lo < RL_UNICODE_LOWERCASE_COUNT && rl_unicode_lowercase[lo].from == cp) ? rl_unicode_lowercase[lo].to : cp;
}

static int is_special_type(int32_t type) {
    return type == RL_TOKTYPE_CONTROL || type == RL_TOKTYPE_USER_DEFINED || type == RL_TOKTYPE_UNKNOWN;
}

/* insertion sort: special tokens by text length descending, then id ascending */
static void sort_special(const rl_tokenizer *t) {
    for (uint32_t i = 1; i < t->special_count; ++i) {
        const uint32_t v = t->special[i];
        const size_t lv = strlen(t->model->tokens[v]);
        uint32_t j = i;
        while (j > 0) {
            const uint32_t u = t->special[j - 1u];
            const size_t lu = strlen(t->model->tokens[u]);
            if (lu > lv || (lu == lv && u < v)) break;
            t->special[j] = u;
            --j;
        }
        t->special[j] = v;
    }
}

rl_tokenizer *rl_tokenizer_create(const rl_gguf_model *model, char *error, size_t cap) {
    if (!model || !model->tokens || !model->vocab_count) { if (error && cap) snprintf(error, cap, "GGUF has no tokenizer vocabulary"); return NULL; }
    if (!model->tokenizer_model || strcmp(model->tokenizer_model, "gpt2") != 0 ||
        !model->tokenizer_pre || strcmp(model->tokenizer_pre, "qwen2") != 0) {
        if (error && cap) snprintf(error, cap, "unsupported tokenizer %s/%s (expected gpt2/qwen2)",
            model->tokenizer_model ? model->tokenizer_model : "?", model->tokenizer_pre ? model->tokenizer_pre : "?");
        return NULL;
    }
    rl_tokenizer *t = (rl_tokenizer *)calloc(1, sizeof(*t));
    if (!t) return NULL;
    t->model = model;
    t->vocab = model->vocab_count;
    t->eos = model->eos_id; t->bos = model->bos_id; t->endoftext = -1; t->im_end = -1;
    if (!strmap_init(&t->token_ids, t->vocab) || !strmap_init(&t->merge_ranks, model->merge_count ? model->merge_count : 1u)) {
        rl_tokenizer_destroy(t); if (error && cap) snprintf(error, cap, "tokenizer map allocation failed"); return NULL;
    }
    for (uint32_t i = 0; i < t->vocab; ++i) strmap_put(&t->token_ids, model->tokens[i], strlen(model->tokens[i]), (int32_t)i);
    for (uint32_t i = 0; i < model->merge_count; ++i) strmap_put(&t->merge_ranks, model->merges[i], strlen(model->merges[i]), (int32_t)i);

    /* GPT-2 byte -> unicode mapping (unicode_byte_to_utf8_map) */
    uint32_t n = 0;
    for (uint32_t b = 0; b < 256u; ++b) {
        const int direct = (b >= 0x21u && b <= 0x7Eu) || (b >= 0xA1u && b <= 0xACu) || (b >= 0xAEu && b <= 0xFFu);
        const uint32_t cp = direct ? b : 256u + n++;
        t->byte_to_utf8_len[b] = (uint8_t)utf8_encode(cp, t->byte_to_utf8[b]);
        t->byte_token[b] = strmap_get(&t->token_ids, t->byte_to_utf8[b], t->byte_to_utf8_len[b]);
    }
    /* special tokens sorted longest first (llama.cpp tokenizer_st_partition order) */
    t->special = (uint32_t *)calloc(t->vocab, sizeof(uint32_t));
    if (!t->special) { rl_tokenizer_destroy(t); return NULL; }
    for (uint32_t i = 0; i < t->vocab; ++i) {
        const int32_t type = model->token_types ? model->token_types[i] : RL_TOKTYPE_NORMAL;
        if (is_special_type(type) && model->tokens[i][0]) t->special[t->special_count++] = i;
    }
    sort_special(t);
    t->endoftext = strmap_get(&t->token_ids, "<|endoftext|>", 13u);
    t->im_end = strmap_get(&t->token_ids, "<|im_end|>", 10u);
    if (error && cap) error[0] = '\0';
    return t;
}

void rl_tokenizer_destroy(rl_tokenizer *t) {
    if (!t) return;
    free(t->token_ids.slots); free(t->merge_ranks.slots); free(t->merge_storage); free(t->special);
    free(t);
}

uint32_t rl_tokenizer_vocab_size(const rl_tokenizer *t) { return t ? t->vocab : 0u; }
int32_t rl_tokenizer_eos(const rl_tokenizer *t) { return t ? t->eos : -1; }
int32_t rl_tokenizer_bos(const rl_tokenizer *t) { return t ? t->bos : -1; }

int rl_tokenizer_is_control(const rl_tokenizer *t, uint32_t id) {
    if (!t || id >= t->vocab || !t->model->token_types) return 0;
    return t->model->token_types[id] == RL_TOKTYPE_CONTROL;
}

int rl_tokenizer_is_eog(const rl_tokenizer *t, uint32_t id) {
    if (!t) return 0;
    return (t->eos >= 0 && id == (uint32_t)t->eos) || (t->endoftext >= 0 && id == (uint32_t)t->endoftext) ||
           (t->im_end >= 0 && id == (uint32_t)t->im_end);
}

int32_t rl_tokenizer_lookup(const rl_tokenizer *t, const char *piece) {
    return t && piece ? strmap_get(&t->token_ids, piece, strlen(piece)) : -1;
}

/* ---- pre-tokenizer: llama.cpp unicode_regex_split_custom_qwen2 ---- */

typedef struct { uint32_t *cpts; uint16_t *flags; size_t count; } cpt_text;

#define OUT_OF_RANGE 0xFFFFFFFFu

static uint32_t get_cpt(const cpt_text *tx, size_t pos) { return pos < tx->count ? tx->cpts[pos] : OUT_OF_RANGE; }
static uint16_t get_flags(const cpt_text *tx, size_t pos) { return pos < tx->count ? tx->flags[pos] : 0u; }
#define IS_LETTER(f) (((f) & FLAG_LETTER) != 0)
#define IS_NUMBER(f) (((f) & FLAG_NUMBER) != 0)
#define IS_WS(f) (((f) & FLAG_WHITESPACE) != 0)

/* Produces piece lengths (in code points) into out; returns piece count */
static size_t qwen2_split(const cpt_text *tx, size_t *out) {
    size_t n = 0, prev_end = 0;
    const size_t end = tx->count;
#define ADD_TOKEN(e) do { const size_t _e = (e); if (_e > prev_end) out[n++] = _e - prev_end; prev_end = _e; } while (0)
    for (size_t pos = 0; pos < end;) {
        const uint32_t cpt = get_cpt(tx, pos);
        const uint16_t flags = get_flags(tx, pos);
        if (cpt == '\'' && pos + 1 < end) {
            const uint32_t n1 = cpt_tolower(get_cpt(tx, pos + 1));
            if (n1 == 's' || n1 == 't' || n1 == 'm' || n1 == 'd') { pos += 2; ADD_TOKEN(pos); continue; }
            if (pos + 2 < end) {
                const uint32_t n2 = cpt_tolower(get_cpt(tx, pos + 2));
                if ((n1 == 'r' && n2 == 'e') || (n1 == 'v' && n2 == 'e') || (n1 == 'l' && n2 == 'l')) { pos += 3; ADD_TOKEN(pos); continue; }
            }
        }
        if (!(cpt == '\r' || cpt == '\n' || IS_NUMBER(flags))) {
            if (IS_LETTER(flags) || IS_LETTER(get_flags(tx, pos + 1))) {
                pos++;
                while (IS_LETTER(get_flags(tx, pos))) pos++;
                ADD_TOKEN(pos);
                continue;
            }
        }
        if (IS_NUMBER(flags)) { pos++; ADD_TOKEN(pos); continue; }
        {
            uint16_t flags2 = (cpt == ' ') ? get_flags(tx, pos + 1) : flags;
            if (!(IS_WS(flags2) || IS_LETTER(flags2) || IS_NUMBER(flags2)) && flags != 0u) {
                pos += (cpt == ' ');
                while (!(IS_WS(flags2) || IS_LETTER(flags2) || IS_NUMBER(flags2)) && flags2 != 0u) flags2 = get_flags(tx, ++pos);
                uint32_t cpt2 = get_cpt(tx, pos);
                while (cpt2 == '\r' || cpt2 == '\n') cpt2 = get_cpt(tx, ++pos);
                ADD_TOKEN(pos);
                continue;
            }
        }
        size_t num_ws = 0, last_end_r_or_n = 0;
        while (IS_WS(get_flags(tx, pos + num_ws))) {
            const uint32_t c2 = get_cpt(tx, pos + num_ws);
            if (c2 == '\r' || c2 == '\n') last_end_r_or_n = pos + num_ws + 1;
            num_ws++;
        }
        if (last_end_r_or_n > 0) { pos = last_end_r_or_n; ADD_TOKEN(pos); continue; }
        if (num_ws > 1 && get_cpt(tx, pos + num_ws) != OUT_OF_RANGE) { pos += num_ws - 1; ADD_TOKEN(pos); continue; }
        if (num_ws > 0) { pos += num_ws; ADD_TOKEN(pos); continue; }
        ADD_TOKEN(++pos);
    }
#undef ADD_TOKEN
    return n;
}

/* ---- BPE over one pre-tokenized word (byte-encoded UTF-8 text) ---- */

typedef struct { const char *text; size_t n; int prev, next; } symbol;

static int32_t merge_rank(const rl_tokenizer *t, const symbol *l, const symbol *r, char *scratch, size_t scratch_cap) {
    if (l->n + r->n + 1u > scratch_cap) return -1;
    memcpy(scratch, l->text, l->n); scratch[l->n] = ' '; memcpy(scratch + l->n + 1u, r->text, r->n);
    return strmap_get(&t->merge_ranks, scratch, l->n + r->n + 1u);
}

static int32_t bpe_word(const rl_tokenizer *t, const char *word, size_t len, uint32_t *out, size_t out_cap, size_t *out_count) {
    if (!len) return 1;
    symbol *syms = (symbol *)malloc(len * sizeof(symbol));
    if (!syms) return 0;
    int count = 0;
    for (size_t off = 0; off < len;) {
        size_t cl = 1;
        const uint8_t b = (uint8_t)word[off];
        if ((b & 0xE0u) == 0xC0u) cl = 2; else if ((b & 0xF0u) == 0xE0u) cl = 3; else if ((b & 0xF8u) == 0xF0u) cl = 4;
        if (cl > len - off) cl = len - off;
        syms[count].text = word + off; syms[count].n = cl; syms[count].prev = count - 1; syms[count].next = off + cl == len ? -1 : count + 1;
        off += cl; count++;
    }
    char scratch[1024];
    for (;;) {
        int best_left = -1; int32_t best_rank = 0x7fffffff;
        for (int i = 0; i != -1; i = syms[i].next) {
            const int j = syms[i].next;
            if (j == -1) break;
            const int32_t rank = merge_rank(t, &syms[i], &syms[j], scratch, sizeof(scratch));
            if (rank >= 0 && (rank < best_rank || (rank == best_rank && i < best_left))) { best_rank = rank; best_left = i; }
        }
        if (best_left < 0) break;
        symbol *l = &syms[best_left]; symbol *r = &syms[l->next];
        l->n += r->n; r->n = 0;
        l->next = r->next;
        if (r->next >= 0) syms[r->next].prev = best_left;
    }
    for (int i = 0; i != -1; i = syms[i].next) {
        if (!syms[i].n) continue;
        const int32_t id = strmap_get(&t->token_ids, syms[i].text, syms[i].n);
        if (id >= 0) {
            if (out && *out_count < out_cap) out[*out_count] = (uint32_t)id;
            (*out_count)++;
        } else {
            /* byte fallback: each byte-encoded character maps back to its byte token */
            for (size_t k = 0; k < syms[i].n;) {
                size_t cl = 1;
                const uint8_t b = (uint8_t)syms[i].text[k];
                if ((b & 0xE0u) == 0xC0u) cl = 2; else if ((b & 0xF0u) == 0xE0u) cl = 3; else if ((b & 0xF8u) == 0xF0u) cl = 4;
                const int32_t bid = strmap_get(&t->token_ids, syms[i].text + k, cl);
                if (bid >= 0) { if (out && *out_count < out_cap) out[*out_count] = (uint32_t)bid; (*out_count)++; }
                k += cl;
            }
        }
    }
    free(syms);
    return 1;
}

/* tokenize a raw (non-special) fragment */
static int tokenize_fragment(const rl_tokenizer *t, const char *text, size_t len, uint32_t *out, size_t out_cap, size_t *out_count) {
    if (!len) return 1;
    cpt_text tx;
    tx.cpts = (uint32_t *)malloc(len * sizeof(uint32_t));
    tx.flags = (uint16_t *)malloc(len * sizeof(uint16_t));
    size_t *pieces = (size_t *)malloc((len + 1u) * sizeof(size_t));
    char *encoded = (char *)malloc(len * 4u + 4u);
    if (!tx.cpts || !tx.flags || !pieces || !encoded) { free(tx.cpts); free(tx.flags); free(pieces); free(encoded); return 0; }
    tx.count = 0;
    for (size_t off = 0; off < len;) {
        size_t consumed = 0;
        const uint32_t cp = utf8_decode(text + off, len - off, &consumed);
        tx.cpts[tx.count] = cp; tx.flags[tx.count] = cpt_flags(cp); tx.count++;
        off += consumed;
    }
    const size_t piece_count = qwen2_split(&tx, pieces);
    size_t cp_index = 0;
    int ok = 1;
    for (size_t p = 0; p < piece_count && ok; ++p) {
        /* byte-encode the piece: re-encode code points to UTF-8 bytes, then map each byte */
        size_t elen = 0;
        for (size_t c = 0; c < pieces[p]; ++c) {
            char tmp[4];
            const uint32_t nb = utf8_encode(tx.cpts[cp_index + c], tmp);
            for (uint32_t k = 0; k < nb; ++k) {
                const uint8_t b = (uint8_t)tmp[k];
                memcpy(encoded + elen, t->byte_to_utf8[b], t->byte_to_utf8_len[b]);
                elen += t->byte_to_utf8_len[b];
            }
        }
        cp_index += pieces[p];
        ok = bpe_word(t, encoded, elen, out, out_cap, out_count);
    }
    free(tx.cpts); free(tx.flags); free(pieces); free(encoded);
    return ok;
}

int32_t rl_tokenizer_encode(const rl_tokenizer *t, const char *text, size_t text_len, int parse_special,
                            uint32_t *out, size_t out_cap, char *error, size_t error_cap) {
    if (!t || !text) { if (error && error_cap) snprintf(error, error_cap, "invalid tokenizer arguments"); return -1; }
    size_t count = 0;
    /* fragments: split the text around special tokens (longest special first, left to right) */
    typedef struct { size_t start, len; int32_t special; } frag;
    frag *frags = (frag *)malloc((2u * text_len + 3u) * sizeof(frag)); /* each hit splits one fragment into three */
    if (!frags) return -1;
    size_t nfrag = 1;
    frags[0].start = 0; frags[0].len = text_len; frags[0].special = -1;
    if (parse_special) {
        for (uint32_t si = 0; si < t->special_count; ++si) {
            const uint32_t id = t->special[si];
            const char *sp = t->model->tokens[id];
            const size_t sl = strlen(sp);
            for (size_t f = 0; f < nfrag; ++f) {
                if (frags[f].special >= 0 || frags[f].len < sl) continue;
                const char *base = text + frags[f].start;
                const char *hit = NULL;
                for (size_t k = 0; k + sl <= frags[f].len; ++k) if (memcmp(base + k, sp, sl) == 0) { hit = base + k; break; }
                if (!hit) continue;
                const size_t before = (size_t)(hit - base);
                const size_t after = frags[f].len - before - sl;
                /* replace frag f with [before][special][after] */
                memmove(&frags[f + 3], &frags[f + 1], (nfrag - f - 1u) * sizeof(frag));
                const size_t start = frags[f].start;
                frags[f].start = start; frags[f].len = before; frags[f].special = -1;
                frags[f + 1].start = start + before; frags[f + 1].len = sl; frags[f + 1].special = (int32_t)id;
                frags[f + 2].start = start + before + sl; frags[f + 2].len = after; frags[f + 2].special = -1;
                nfrag += 2;
                f += 1; /* continue scanning the 'after' fragment for the same special */
            }
        }
    }
    int ok = 1;
    for (size_t f = 0; f < nfrag && ok; ++f) {
        if (frags[f].special >= 0) { if (out && count < out_cap) out[count] = (uint32_t)frags[f].special; count++; }
        else if (frags[f].len) ok = tokenize_fragment(t, text + frags[f].start, frags[f].len, out, out_cap, &count);
    }
    free(frags);
    if (!ok) { if (error && error_cap) snprintf(error, error_cap, "tokenizer allocation failed"); return -1; }
    if (error && error_cap) error[0] = '\0';
    return (int32_t)count;
}

int32_t rl_tokenizer_decode(const rl_tokenizer *t, uint32_t id, char *out, size_t out_cap) {
    if (!t || id >= t->vocab || !out) return -1;
    const char *piece = t->model->tokens[id];
    const int32_t type = t->model->token_types ? t->model->token_types[id] : RL_TOKTYPE_NORMAL;
    if (type == RL_TOKTYPE_CONTROL || type == RL_TOKTYPE_USER_DEFINED || type == RL_TOKTYPE_UNKNOWN) {
        const size_t n = strlen(piece);
        if (n > out_cap) return -1;
        memcpy(out, piece, n);
        return (int32_t)n;
    }
    /* byte-level: every code point of the piece maps back to one byte */
    size_t written = 0;
    const size_t len = strlen(piece);
    for (size_t off = 0; off < len;) {
        size_t consumed = 0;
        const uint32_t cp = utf8_decode(piece + off, len - off, &consumed);
        int found = -1;
        for (uint32_t b = 0; b < 256u; ++b) {
            if (t->byte_to_utf8_len[b] == consumed && memcmp(t->byte_to_utf8[b], piece + off, consumed) == 0) { found = (int)b; break; }
        }
        if (found < 0) { (void)cp; return -1; }
        if (written >= out_cap) return -1;
        out[written++] = (char)found;
        off += consumed;
    }
    return (int32_t)written;
}

int rl_tokenizer_chat_prompt(const char *system_prompt, const char *user_prompt, char *out, size_t out_cap) {
    if (!user_prompt || !out) return 0;
    int n;
    if (system_prompt && *system_prompt)
        n = snprintf(out, out_cap, "<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", system_prompt, user_prompt);
    else
        n = snprintf(out, out_cap, "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", user_prompt);
    return n > 0 && (size_t)n < out_cap;
}

int rl_tokenizer_chat_continuation(const char *user_prompt, char *out, size_t out_cap) {
    if (!user_prompt || !out) return 0;
    const int n = snprintf(out, out_cap,
        "\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", user_prompt);
    return n > 0 && (size_t)n < out_cap;
}
