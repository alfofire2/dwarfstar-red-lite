#define _FILE_OFFSET_BITS 64
#define _POSIX_C_SOURCE 200809L

#include "redlite_native_attention_proj.h"
#include "redlite_native_iq2_xxs.h"
#include "redlite_native_shared_exec.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define GGUF_DEFAULT_ALIGNMENT 32u
#define GGUF_MAX_STRING (1ull << 30)
#define GGUF_MAX_ARRAY  (1ull << 30)
#define QK_K 256u
#define IQ2_XXS_BLOCK_BYTES 66u
#define Q4_K_BLOCK_BYTES 144u

enum {
    GGUF_U8 = 0, GGUF_I8, GGUF_U16, GGUF_I16, GGUF_U32, GGUF_I32,
    GGUF_F32, GGUF_BOOL, GGUF_STRING, GGUF_ARRAY, GGUF_U64, GGUF_I64, GGUF_F64
};

typedef struct {
    char *name;
    uint32_t n_dims;
    uint64_t shape[RL_ATTN_PROJ_MAX_DIMS];
    uint32_t ggml_type;
    uint64_t relative_offset;
    uint64_t absolute_offset;
    uint64_t span_bytes;
} raw_tensor;

typedef struct {
    uint32_t alignment;
    uint32_t hidden;
    uint32_t query_heads;
    uint32_t kv_heads;
    uint32_t key_length;
    uint32_t value_length;
    uint32_t rope_dims;
    float rms_eps;
    float rope_freq_base;
    uint32_t seen;
} model_meta;

typedef struct {
    float max_abs;
    float max_rel;
    size_t index;
} error_stats;

enum {
    META_HIDDEN = 1u << 0,
    META_Q_HEADS = 1u << 1,
    META_KV_HEADS = 1u << 2,
    META_KEY_LENGTH = 1u << 3,
    META_VALUE_LENGTH = 1u << 4,
    META_ROPE_DIMS = 1u << 5,
    META_RMS_EPS = 1u << 6,
    META_FREQ_BASE = 1u << 7,
    META_REQUIRED = 0xffu,
};

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

static int read_exact(FILE *f, void *dst, size_t n) {
    return n == 0 || fread(dst, 1, n, f) == n;
}

static int read_u32(FILE *f, uint32_t *v) {
    uint8_t b[4];
    if (!read_exact(f, b, sizeof(b))) return 0;
    *v = (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
    return 1;
}

static int read_u64(FILE *f, uint64_t *v) {
    uint8_t b[8];
    if (!read_exact(f, b, sizeof(b))) return 0;
    *v = (uint64_t)b[0] | ((uint64_t)b[1] << 8) | ((uint64_t)b[2] << 16) |
         ((uint64_t)b[3] << 24) | ((uint64_t)b[4] << 32) | ((uint64_t)b[5] << 40) |
         ((uint64_t)b[6] << 48) | ((uint64_t)b[7] << 56);
    return 1;
}

static int read_f32(FILE *f, float *v) {
    uint32_t bits = 0;
    if (!read_u32(f, &bits)) return 0;
    memcpy(v, &bits, sizeof(*v));
    return 1;
}

static int skip_bytes(FILE *f, uint64_t n) {
    while (n) {
        const uint64_t chunk = n > 0x3fffffffULL ? 0x3fffffffULL : n;
        if (fseeko(f, (off_t)chunk, SEEK_CUR) != 0) return 0;
        n -= chunk;
    }
    return 1;
}

static int read_string(FILE *f, char **out) {
    uint64_t n = 0;
    if (!read_u64(f, &n) || n > GGUF_MAX_STRING || n > SIZE_MAX - 1u) return 0;
    char *s = (char *)malloc((size_t)n + 1u);
    if (!s) return 0;
    if (!read_exact(f, s, (size_t)n)) { free(s); return 0; }
    s[n] = '\0';
    *out = s;
    return 1;
}

static int skip_string(FILE *f) {
    uint64_t n = 0;
    return read_u64(f, &n) && n <= GGUF_MAX_STRING && skip_bytes(f, n);
}

static size_t scalar_size(uint32_t type) {
    switch (type) {
        case GGUF_U8: case GGUF_I8: case GGUF_BOOL: return 1u;
        case GGUF_U16: case GGUF_I16: return 2u;
        case GGUF_U32: case GGUF_I32: case GGUF_F32: return 4u;
        case GGUF_U64: case GGUF_I64: case GGUF_F64: return 8u;
        default: return 0u;
    }
}

static int skip_value(FILE *f, uint32_t type) {
    const size_t fixed = scalar_size(type);
    if (fixed) return skip_bytes(f, fixed);
    if (type == GGUF_STRING) return skip_string(f);
    if (type == GGUF_ARRAY) {
        uint32_t subtype = 0;
        uint64_t count = 0;
        if (!read_u32(f, &subtype) || !read_u64(f, &count) || count > GGUF_MAX_ARRAY) return 0;
        const size_t sub_fixed = scalar_size(subtype);
        if (sub_fixed) {
            if (count && count > UINT64_MAX / sub_fixed) return 0;
            return skip_bytes(f, count * sub_fixed);
        }
        for (uint64_t i = 0; i < count; ++i) if (!skip_value(f, subtype)) return 0;
        return 1;
    }
    return 0;
}

static int read_metadata(FILE *f, uint64_t count, model_meta *m) {
    memset(m, 0, sizeof(*m));
    m->alignment = GGUF_DEFAULT_ALIGNMENT;
    for (uint64_t i = 0; i < count; ++i) {
        char *key = NULL;
        uint32_t type = 0;
        if (!read_string(f, &key) || !read_u32(f, &type)) { free(key); return 0; }
        uint32_t *u32 = NULL;
        uint32_t flag = 0;
        float *f32 = NULL;
        if (strcmp(key, "general.alignment") == 0 && type == GGUF_U32) u32 = &m->alignment;
        else if (strcmp(key, "qwen3next.embedding_length") == 0 && type == GGUF_U32) { u32 = &m->hidden; flag = META_HIDDEN; }
        else if (strcmp(key, "qwen3next.attention.head_count") == 0 && type == GGUF_U32) { u32 = &m->query_heads; flag = META_Q_HEADS; }
        else if (strcmp(key, "qwen3next.attention.head_count_kv") == 0 && type == GGUF_U32) { u32 = &m->kv_heads; flag = META_KV_HEADS; }
        else if (strcmp(key, "qwen3next.attention.key_length") == 0 && type == GGUF_U32) { u32 = &m->key_length; flag = META_KEY_LENGTH; }
        else if (strcmp(key, "qwen3next.attention.value_length") == 0 && type == GGUF_U32) { u32 = &m->value_length; flag = META_VALUE_LENGTH; }
        else if (strcmp(key, "qwen3next.rope.dimension_count") == 0 && type == GGUF_U32) { u32 = &m->rope_dims; flag = META_ROPE_DIMS; }
        else if (strcmp(key, "qwen3next.attention.layer_norm_rms_epsilon") == 0 && type == GGUF_F32) { f32 = &m->rms_eps; flag = META_RMS_EPS; }
        else if (strcmp(key, "qwen3next.rope.freq_base") == 0 && type == GGUF_F32) { f32 = &m->rope_freq_base; flag = META_FREQ_BASE; }
        free(key);
        if (u32) {
            if (!read_u32(f, u32) || !*u32) return 0;
            m->seen |= flag;
        } else if (f32) {
            if (!read_f32(f, f32) || !isfinite(*f32) || *f32 <= 0.0f) return 0;
            m->seen |= flag;
        } else if (!skip_value(f, type)) return 0;
    }
    return (m->seen & META_REQUIRED) == META_REQUIRED;
}

static uint64_t round_up_u64(uint64_t v, uint64_t a) {
    if (!a) return 0;
    const uint64_t r = v % a;
    if (!r) return v;
    return v > UINT64_MAX - (a - r) ? 0 : v + a - r;
}

static int tensor_cmp(const void *a, const void *b) {
    const raw_tensor *ta = (const raw_tensor *)a;
    const raw_tensor *tb = (const raw_tensor *)b;
    if (ta->relative_offset < tb->relative_offset) return -1;
    if (ta->relative_offset > tb->relative_offset) return 1;
    return 0;
}

static void free_raw(raw_tensor *raw, uint64_t count) {
    if (!raw) return;
    for (uint64_t i = 0; i < count; ++i) free(raw[i].name);
    free(raw);
}

static const char *kind_name(rl_attn_proj_kind kind) {
    switch (kind) {
        case RL_ATTN_PROJ_INPUT_NORM: return "attn_norm";
        case RL_ATTN_PROJ_Q_GATE: return "attn_q";
        case RL_ATTN_PROJ_K: return "attn_k";
        case RL_ATTN_PROJ_V: return "attn_v";
        case RL_ATTN_PROJ_Q_NORM: return "attn_q_norm";
        case RL_ATTN_PROJ_K_NORM: return "attn_k_norm";
        case RL_ATTN_PROJ_OUTPUT: return "attn_output";
        default: return "unknown";
    }
}

static int target_kind(const char *name, uint32_t layer, rl_attn_proj_kind *kind) {
    static const char *suffixes[RL_ATTN_PROJ_TENSOR_COUNT] = {
        "attn_norm.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight",
        "attn_q_norm.weight", "attn_k_norm.weight", "attn_output.weight",
    };
    char expected[96];
    for (uint32_t i = 0; i < RL_ATTN_PROJ_TENSOR_COUNT; ++i) {
        snprintf(expected, sizeof(expected), "blk.%u.%s", layer, suffixes[i]);
        if (strcmp(name, expected) == 0) { *kind = (rl_attn_proj_kind)i; return 1; }
    }
    return 0;
}

static int audit_layer(const char *model, uint32_t layer, model_meta *meta,
        rl_attn_proj_tensor_info out[RL_ATTN_PROJ_TENSOR_COUNT], char *error, size_t cap) {
    memset(out, 0, RL_ATTN_PROJ_TENSOR_COUNT * sizeof(*out));
    FILE *f = fopen(model, "rb");
    if (!f) { snprintf(error, cap, "open failed: %s", strerror(errno)); return 0; }
    if (fseeko(f, 0, SEEK_END) != 0) { fclose(f); snprintf(error, cap, "seek failed"); return 0; }
    const off_t end = ftello(f);
    if (end <= 0 || fseeko(f, 0, SEEK_SET) != 0) { fclose(f); snprintf(error, cap, "invalid GGUF size"); return 0; }
    const uint64_t file_size = (uint64_t)end;

    uint8_t magic[4];
    uint32_t version = 0;
    uint64_t tensor_count = 0, kv_count = 0;
    if (!read_exact(f, magic, sizeof(magic)) || memcmp(magic, "GGUF", 4) != 0 ||
        !read_u32(f, &version) || (version != 2u && version != 3u) ||
        !read_u64(f, &tensor_count) || !read_u64(f, &kv_count) || !read_metadata(f, kv_count, meta)) {
        fclose(f); snprintf(error, cap, "invalid GGUF/Qwen3-Next metadata"); return 0;
    }
    if (tensor_count > SIZE_MAX / sizeof(raw_tensor)) { fclose(f); snprintf(error, cap, "tensor directory too large"); return 0; }
    raw_tensor *raw = (raw_tensor *)calloc((size_t)tensor_count, sizeof(*raw));
    if (!raw) { fclose(f); snprintf(error, cap, "out of memory for tensor directory"); return 0; }
    for (uint64_t i = 0; i < tensor_count; ++i) {
        uint32_t dims = 0;
        if (!read_string(f, &raw[i].name) || !read_u32(f, &dims) || dims > RL_ATTN_PROJ_MAX_DIMS) {
            free_raw(raw, tensor_count); fclose(f); snprintf(error, cap, "tensor descriptor parse failed"); return 0;
        }
        raw[i].n_dims = dims;
        for (uint32_t d = 0; d < dims; ++d) if (!read_u64(f, &raw[i].shape[d])) {
            free_raw(raw, tensor_count); fclose(f); snprintf(error, cap, "tensor shape parse failed"); return 0;
        }
        if (!read_u32(f, &raw[i].ggml_type) || !read_u64(f, &raw[i].relative_offset)) {
            free_raw(raw, tensor_count); fclose(f); snprintf(error, cap, "tensor type/offset parse failed"); return 0;
        }
    }
    const off_t directory_end = ftello(f);
    fclose(f);
    const uint64_t data_base = directory_end < 0 ? 0 : round_up_u64((uint64_t)directory_end, meta->alignment);
    if (!data_base || data_base > file_size) { free_raw(raw, tensor_count); snprintf(error, cap, "invalid GGUF data base"); return 0; }

    qsort(raw, (size_t)tensor_count, sizeof(*raw), tensor_cmp);
    uint8_t seen[RL_ATTN_PROJ_TENSOR_COUNT] = {0};
    for (uint64_t i = 0; i < tensor_count; ++i) {
        const uint64_t rel = raw[i].relative_offset;
        const uint64_t next = i + 1u < tensor_count ? raw[i + 1u].relative_offset : file_size - data_base;
        raw[i].absolute_offset = data_base + rel;
        raw[i].span_bytes = next >= rel ? next - rel : 0;
        rl_attn_proj_kind kind;
        if (!target_kind(raw[i].name, layer, &kind)) continue;
        if (seen[(uint32_t)kind]) { free_raw(raw, tensor_count); snprintf(error, cap, "duplicate full-attention tensor"); return 0; }
        seen[(uint32_t)kind] = 1;
        rl_attn_proj_tensor_info *dst = &out[(uint32_t)kind];
        dst->layer = layer;
        dst->kind = kind;
        dst->ggml_type = raw[i].ggml_type;
        dst->n_dims = raw[i].n_dims;
        memcpy(dst->shape, raw[i].shape, sizeof(dst->shape));
        dst->tensor_offset = raw[i].absolute_offset;
        dst->tensor_span_bytes = raw[i].span_bytes;
    }
    free_raw(raw, tensor_count);
    for (uint32_t i = 0; i < RL_ATTN_PROJ_TENSOR_COUNT; ++i) if (!seen[i]) {
        snprintf(error, cap, "layer %u is not complete full attention (missing %s)", layer, kind_name((rl_attn_proj_kind)i));
        return 0;
    }
    if (error && cap) error[0] = '\0';
    return 1;
}

static size_t iq2_row_bytes(uint32_t ncols) {
    return ncols && ncols % QK_K == 0 ? (size_t)(ncols / QK_K) * IQ2_XXS_BLOCK_BYTES : 0;
}

static size_t q4_row_bytes(uint32_t ncols) {
    return ncols && ncols % QK_K == 0 ? (size_t)(ncols / QK_K) * Q4_K_BLOCK_BYTES : 0;
}

static uint16_t read_u16_le(const uint8_t *p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x03ffu;
    uint32_t bits;
    if (exp == 0) {
        if (!mant) bits = sign;
        else {
            int shift = 0;
            while ((mant & 0x0400u) == 0) { mant <<= 1; ++shift; }
            mant &= 0x03ffu;
            bits = sign | ((uint32_t)(127 - 14 - shift) << 23) | (mant << 13);
        }
    } else if (exp == 31u) bits = sign | 0x7f800000u | (mant << 13);
    else bits = sign | ((exp + 112u) << 23) | (mant << 13);
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static void get_scale_min_k4(uint32_t j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4u) { *d = q[j] & 63u; *m = q[j + 4u] & 63u; }
    else {
        *d = (q[j + 4u] & 0x0fu) | ((q[j - 4u] >> 6) << 4);
        *m = (q[j + 4u] >> 4) | ((q[j] >> 6) << 4);
    }
}

static int q4_k_row_dot(const uint8_t *row, const float *input, uint32_t ncols, double *result) {
    if (!row || !input || !result || !ncols || ncols % QK_K) return 0;
    double acc = 0.0;
    const uint32_t blocks = ncols / QK_K;
    for (uint32_t ib = 0; ib < blocks; ++ib) {
        const uint8_t *bp = row + (size_t)ib * Q4_K_BLOCK_BYTES;
        const double d = f16_to_f32(read_u16_le(bp));
        const double dmin = f16_to_f32(read_u16_le(bp + 2u));
        const uint8_t *scales = bp + 4u;
        const uint8_t *qs = bp + 16u;
        for (uint32_t g = 0; g < 8u; ++g) {
            uint8_t sc = 0, m = 0;
            get_scale_min_k4(g, scales, &sc, &m);
            const double ds = d * sc;
            const double dm = dmin * m;
            const uint8_t *q = qs + (g / 2u) * 32u;
            const uint32_t xb = ib * QK_K + g * 32u;
            for (uint32_t l = 0; l < 32u; ++l) {
                const uint8_t quant = (g & 1u) ? (q[l] >> 4) : (q[l] & 0x0fu);
                acc += (double)input[xb + l] * (ds * quant - dm);
            }
        }
    }
    *result = acc;
    return 1;
}

static int pread_full(int fd, void *dst, size_t bytes, uint64_t offset, uint64_t *calls) {
    uint8_t *p = (uint8_t *)dst;
    size_t done = 0;
    while (done < bytes) {
        const ssize_t n = pread(fd, p + done, bytes - done, (off_t)(offset + done));
        if (n < 0 && errno == EINTR) continue;
        if (calls) (*calls)++;
        if (n <= 0) return 0;
        done += (size_t)n;
    }
    return 1;
}

static void make_input(float *x, uint32_t n) {
    for (uint32_t i = 0; i < n; ++i) x[i] = (float)(((int)((i * 17u + 3u) % 61u) - 30) / 32.0);
}

static void rmsnorm_f32(const float *x, const float *w, uint32_t n, float eps, float *out) {
    double sum = 0.0;
    for (uint32_t i = 0; i < n; ++i) sum += (double)x[i] * x[i];
    const double inv = 1.0 / sqrt(sum / (double)n + (double)eps);
    for (uint32_t i = 0; i < n; ++i) out[i] = (float)((double)x[i] * inv * w[i]);
}

static void rope_neox(float *x, uint32_t head_dim, uint32_t rope_dims, uint32_t position, float freq_base) {
    const uint32_t half = rope_dims / 2u;
    for (uint32_t i = 0; i < half; ++i) {
        const float theta = (float)position * powf(freq_base, -2.0f * (float)i / (float)rope_dims);
        const float c = cosf(theta), s = sinf(theta);
        const float x0 = x[i], x1 = x[half + i];
        x[i] = x0 * c - x1 * s;
        x[half + i] = x0 * s + x1 * c;
    }
    (void)head_dim;
}

static void make_prior_cache(float *key_cache, float *value_cache, uint32_t position,
        uint32_t kv_heads, uint32_t head_dim, uint32_t rope_dims, float freq_base) {
    for (uint32_t pos = 0; pos < position; ++pos) {
        for (uint32_t h = 0; h < kv_heads; ++h) {
            float *k = key_cache + ((size_t)pos * kv_heads + h) * head_dim;
            float *v = value_cache + ((size_t)pos * kv_heads + h) * head_dim;
            for (uint32_t i = 0; i < head_dim; ++i) {
                k[i] = (float)((int)((pos * 31u + h * 17u + i * 13u + 5u) % 97u) - 48) / 96.0f;
                v[i] = (float)((int)((pos * 43u + h * 11u + i * 7u + 9u) % 89u) - 44) / 88.0f;
            }
            rope_neox(k, head_dim, rope_dims, pos, freq_base);
        }
    }
}

static void cpu_gqa(const float *query, const float *key_cache, const float *value_cache,
        const float *gate, uint32_t seq_len, uint32_t query_heads, uint32_t kv_heads,
        uint32_t head_dim, float *attention, float *gated) {
    const double scale = 1.0 / sqrt((double)head_dim);
    for (uint32_t h = 0; h < query_heads; ++h) {
        const uint32_t kvh = h / (query_heads / kv_heads);
        const float *q = query + (size_t)h * head_dim;
        double scores[RL_ATTN_MAX_CONTEXT];
        double max_score = -INFINITY;
        for (uint32_t pos = 0; pos < seq_len; ++pos) {
            const float *k = key_cache + ((size_t)pos * kv_heads + kvh) * head_dim;
            double dot = 0.0;
            for (uint32_t i = 0; i < head_dim; ++i) dot += (double)q[i] * k[i];
            scores[pos] = dot * scale;
            if (scores[pos] > max_score) max_score = scores[pos];
        }
        double denom = 0.0;
        for (uint32_t pos = 0; pos < seq_len; ++pos) { scores[pos] = exp(scores[pos] - max_score); denom += scores[pos]; }
        for (uint32_t i = 0; i < head_dim; ++i) {
            double acc = 0.0;
            for (uint32_t pos = 0; pos < seq_len; ++pos) {
                const float *v = value_cache + ((size_t)pos * kv_heads + kvh) * head_dim;
                acc += (scores[pos] / denom) * v[i];
            }
            const size_t index = (size_t)h * head_dim + i;
            attention[index] = (float)acc;
            gated[index] = (float)(acc / (1.0 + exp(-(double)gate[index])));
        }
    }
}

static error_stats compare_arrays(const float *got, const float *ref, size_t n) {
    error_stats s = {0.0f, 0.0f, 0u};
    for (size_t i = 0; i < n; ++i) {
        const float ae = fabsf(got[i] - ref[i]);
        const float re = ae / fmaxf(fabsf(ref[i]), 1.0e-12f);
        if (ae > s.max_abs) { s.max_abs = ae; s.index = i; }
        if (re > s.max_rel) s.max_rel = re;
    }
    return s;
}

static int within(const float *got, const float *ref, size_t n, float abs_tol, float rel_tol) {
    for (size_t i = 0; i < n; ++i)
        if (fabsf(got[i] - ref[i]) > abs_tol + rel_tol * fabsf(ref[i])) return 0;
    return 1;
}

static int validate_layout(const model_meta *m, const rl_attn_proj_tensor_info t[RL_ATTN_PROJ_TENSOR_COUNT],
        size_t bytes[RL_ATTN_PROJ_TENSOR_COUNT], char *error, size_t cap) {
    if (m->hidden != 2048u || m->query_heads != 16u || m->kv_heads != 2u ||
        m->key_length != 256u || m->value_length != 256u || m->rope_dims != 64u) {
        snprintf(error, cap, "unexpected Qwen3-Next attention metadata geometry"); return 0;
    }
    const uint32_t qcount = m->query_heads * m->key_length;
    const uint32_t kvcount = m->kv_heads * m->key_length;
    if (t[0].ggml_type != 0u || t[0].n_dims != 1u || t[0].shape[0] != m->hidden ||
        t[1].ggml_type != 16u || t[1].n_dims != 2u || t[1].shape[0] != m->hidden || t[1].shape[1] != 2u * qcount ||
        t[2].ggml_type != 12u || t[2].n_dims != 2u || t[2].shape[0] != m->hidden || t[2].shape[1] != kvcount ||
        t[3].ggml_type != 12u || t[3].n_dims != 2u || t[3].shape[0] != m->hidden || t[3].shape[1] != kvcount ||
        t[4].ggml_type != 0u || t[4].n_dims != 1u || t[4].shape[0] != m->key_length ||
        t[5].ggml_type != 0u || t[5].n_dims != 1u || t[5].shape[0] != m->key_length ||
        t[6].ggml_type != 12u || t[6].n_dims != 2u || t[6].shape[0] != qcount || t[6].shape[1] != m->hidden) {
        snprintf(error, cap, "unexpected full-attention tensor types/shapes"); return 0;
    }
    const size_t iq2_rb = iq2_row_bytes(m->hidden);
    const size_t q4_in_rb = q4_row_bytes(m->hidden);
    const size_t q4_out_rb = q4_row_bytes(qcount);
    bytes[0] = (size_t)m->hidden * sizeof(float);
    bytes[1] = (size_t)(2u * qcount) * iq2_rb;
    bytes[2] = (size_t)kvcount * q4_in_rb;
    bytes[3] = (size_t)kvcount * q4_in_rb;
    bytes[4] = (size_t)m->key_length * sizeof(float);
    bytes[5] = (size_t)m->key_length * sizeof(float);
    bytes[6] = (size_t)m->hidden * q4_out_rb;
    for (uint32_t i = 0; i < RL_ATTN_PROJ_TENSOR_COUNT; ++i) if (t[i].tensor_span_bytes < bytes[i]) {
        snprintf(error, cap, "full-attention tensor span/layout mismatch"); return 0;
    }
    return 1;
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

static void usage(FILE *out) {
    fprintf(out,
        "redlite-attention 0.3.0.dev16\n\n"
        "Usage:\n"
        "  redlite-attention parity MODEL --layer N [--position N]\n"
        "  redlite-attention --selftest\n\n"
        "Validates a complete single-token Qwen3-Next full-attention branch with NeoX RoPE, GQA, KV cache and Q4_K output.\n");
}

static int selftest(void) {
    uint8_t block[Q4_K_BLOCK_BYTES];
    memset(block, 0, sizeof(block));
    block[0] = 0x00; block[1] = 0x3c;
    for (uint32_t j = 0; j < 4u; ++j) block[4u + j] = 1u;
    for (uint32_t j = 8u; j < 12u; ++j) block[4u + j] = 1u;
    memset(block + 16u, 0x11, 128u);
    float x[QK_K];
    for (uint32_t i = 0; i < QK_K; ++i) x[i] = 1.0f;
    double dot = 0.0;
    if (!q4_k_row_dot(block, x, QK_K, &dot) || fabs(dot - 256.0) > 1.0e-9) return 0;

    float head[256], original[256];
    for (uint32_t i = 0; i < 256u; ++i) head[i] = original[i] = (float)((int)i - 127) / 128.0f;
    rope_neox(head, 256u, 64u, 0u, 10000000.0f);
    for (uint32_t i = 0; i < 256u; ++i) if (head[i] != original[i]) return 0;
    return 1;
}

int main(int argc, char **argv) {
    if (argc == 2 && strcmp(argv[1], "--selftest") == 0) {
        if (!selftest()) { fprintf(stderr, "full-attention selftest failed\n"); return 1; }
        printf("attention Q4_K ref : OK\n");
        printf("attention NeoX RoPE: OK\n");
        return 0;
    }
    if (argc < 3 || strcmp(argv[1], "parity") != 0) { usage(argc > 1 ? stderr : stdout); return 2; }
    const char *model = argv[2];
    uint32_t layer = UINT32_MAX, position = 7u;
    for (int i = 3; i < argc; ++i) {
        uint32_t *target = NULL;
        if (strcmp(argv[i], "--layer") == 0) target = &layer;
        else if (strcmp(argv[i], "--position") == 0) target = &position;
        else { fprintf(stderr, "unknown option: %s\n", argv[i]); return 2; }
        if (i + 1 >= argc || !parse_u32(argv[++i], target)) { fprintf(stderr, "invalid option value\n"); return 2; }
    }
    if (layer == UINT32_MAX || position >= RL_ATTN_MAX_CONTEXT) {
        fprintf(stderr, "--layer is required and --position must be below %u\n", RL_ATTN_MAX_CONTEXT); return 2;
    }

    char error[512] = {0};
    model_meta meta;
    rl_attn_proj_tensor_info tensors[RL_ATTN_PROJ_TENSOR_COUNT];
    size_t weight_bytes[RL_ATTN_PROJ_TENSOR_COUNT] = {0};
    if (!audit_layer(model, layer, &meta, tensors, error, sizeof(error))) {
        fprintf(stderr, "full-attention audit failed: %s\n", error); return 1;
    }
    if (!validate_layout(&meta, tensors, weight_bytes, error, sizeof(error))) {
        fprintf(stderr, "full-attention layout failed: %s\n", error); return 1;
    }

    const uint32_t hidden = meta.hidden;
    const uint32_t head_dim = meta.key_length;
    const uint32_t query_count = meta.query_heads * head_dim;
    const uint32_t kv_count = meta.kv_heads * head_dim;
    const uint32_t qgate_count = 2u * query_count;
    const uint32_t seq_len = position + 1u;
    const size_t cache_count = (size_t)seq_len * kv_count;
    const size_t iq2_rb = iq2_row_bytes(hidden);
    const size_t q4_in_rb = q4_row_bytes(hidden);
    const size_t q4_out_rb = q4_row_bytes(query_count);

    uint8_t *weights[RL_ATTN_PROJ_TENSOR_COUNT] = {0};
    float *input = NULL, *cpu_input_norm = NULL, *gpu_input_norm = NULL;
    float *cpu_query = NULL, *gpu_query = NULL, *cpu_gate = NULL, *gpu_gate = NULL;
    float *cpu_key = NULL, *gpu_key = NULL, *cpu_value = NULL, *gpu_value = NULL;
    float *cpu_attention = NULL, *gpu_attention = NULL, *cpu_gated = NULL, *gpu_gated = NULL;
    float *cpu_output = NULL, *gpu_output = NULL;
    float *cpu_key_cache = NULL, *gpu_key_cache = NULL, *cpu_value_cache = NULL, *gpu_value_cache = NULL;
    float *qgate_raw = NULL, *key_raw = NULL;
    int result = 1;

    for (uint32_t i = 0; i < RL_ATTN_PROJ_TENSOR_COUNT; ++i) {
        weights[i] = (uint8_t *)malloc(weight_bytes[i]);
        if (!weights[i]) { fprintf(stderr, "out of memory for full-attention weights\n"); goto cleanup; }
    }
#define ALLOC_FLOAT(name, count) do { name = (float *)calloc((count), sizeof(float)); if (!(name)) { fprintf(stderr, "out of memory for " #name "\n"); goto cleanup; } } while (0)
    ALLOC_FLOAT(input, hidden);
    ALLOC_FLOAT(cpu_input_norm, hidden); ALLOC_FLOAT(gpu_input_norm, hidden);
    ALLOC_FLOAT(cpu_query, query_count); ALLOC_FLOAT(gpu_query, query_count);
    ALLOC_FLOAT(cpu_gate, query_count); ALLOC_FLOAT(gpu_gate, query_count);
    ALLOC_FLOAT(cpu_key, kv_count); ALLOC_FLOAT(gpu_key, kv_count);
    ALLOC_FLOAT(cpu_value, kv_count); ALLOC_FLOAT(gpu_value, kv_count);
    ALLOC_FLOAT(cpu_attention, query_count); ALLOC_FLOAT(gpu_attention, query_count);
    ALLOC_FLOAT(cpu_gated, query_count); ALLOC_FLOAT(gpu_gated, query_count);
    ALLOC_FLOAT(cpu_output, hidden); ALLOC_FLOAT(gpu_output, hidden);
    ALLOC_FLOAT(cpu_key_cache, cache_count); ALLOC_FLOAT(gpu_key_cache, cache_count);
    ALLOC_FLOAT(cpu_value_cache, cache_count); ALLOC_FLOAT(gpu_value_cache, cache_count);
    ALLOC_FLOAT(qgate_raw, qgate_count); ALLOC_FLOAT(key_raw, kv_count);
#undef ALLOC_FLOAT

    const int fd = open(model, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "open model failed: %s\n", strerror(errno)); goto cleanup; }
    uint64_t cpu_read_calls = 0, total_read_bytes = 0;
    const double read_start = now_ms();
    int read_ok = 1;
    for (uint32_t i = 0; i < RL_ATTN_PROJ_TENSOR_COUNT; ++i) {
        if (!pread_full(fd, weights[i], weight_bytes[i], tensors[i].tensor_offset, &cpu_read_calls)) { read_ok = 0; break; }
        total_read_bytes += weight_bytes[i];
    }
    const double cpu_read_ms = now_ms() - read_start;
    close(fd);
    if (!read_ok) { fprintf(stderr, "failed reading real full-attention weights\n"); goto cleanup; }

    uint8_t grid[RL_IQ2_XXS_GRID_COUNT];
    if (!rl_native_iq2_xxs_build_grid(grid, error, sizeof(error))) {
        fprintf(stderr, "IQ2_XXS grid build failed: %s\n", error); goto cleanup;
    }
    make_input(input, hidden);
    make_prior_cache(cpu_key_cache, cpu_value_cache, position, meta.kv_heads, head_dim, meta.rope_dims, meta.rope_freq_base);
    memcpy(gpu_key_cache, cpu_key_cache, cache_count * sizeof(float));
    memcpy(gpu_value_cache, cpu_value_cache, cache_count * sizeof(float));

    const double cpu_start = now_ms();
    rmsnorm_f32(input, (const float *)weights[0], hidden, meta.rms_eps, cpu_input_norm);
    for (uint32_t row = 0; row < qgate_count; ++row) {
        double dot = 0.0;
        if (!rl_native_shared_quant_row_dot(weights[1] + (size_t)row * iq2_rb, iq2_rb, 16u,
                cpu_input_norm, hidden, grid, sizeof(grid), &dot, error, sizeof(error))) {
            fprintf(stderr, "CPU IQ2_XXS query/gate failed: %s\n", error); goto cleanup;
        }
        qgate_raw[row] = (float)dot;
    }
    for (uint32_t row = 0; row < kv_count; ++row) {
        double kd = 0.0, vd = 0.0;
        if (!q4_k_row_dot(weights[2] + (size_t)row * q4_in_rb, cpu_input_norm, hidden, &kd) ||
            !q4_k_row_dot(weights[3] + (size_t)row * q4_in_rb, cpu_input_norm, hidden, &vd)) {
            fprintf(stderr, "CPU Q4_K key/value failed\n"); goto cleanup;
        }
        key_raw[row] = (float)kd;
        cpu_value[row] = (float)vd;
    }
    for (uint32_t h = 0; h < meta.query_heads; ++h) {
        const float *src = qgate_raw + (size_t)h * head_dim * 2u;
        rmsnorm_f32(src, (const float *)weights[4], head_dim, meta.rms_eps, cpu_query + (size_t)h * head_dim);
        memcpy(cpu_gate + (size_t)h * head_dim, src + head_dim, (size_t)head_dim * sizeof(float));
        rope_neox(cpu_query + (size_t)h * head_dim, head_dim, meta.rope_dims, position, meta.rope_freq_base);
    }
    for (uint32_t h = 0; h < meta.kv_heads; ++h) {
        rmsnorm_f32(key_raw + (size_t)h * head_dim, (const float *)weights[5], head_dim, meta.rms_eps,
            cpu_key + (size_t)h * head_dim);
        rope_neox(cpu_key + (size_t)h * head_dim, head_dim, meta.rope_dims, position, meta.rope_freq_base);
    }
    memcpy(cpu_key_cache + (size_t)position * kv_count, cpu_key, (size_t)kv_count * sizeof(float));
    memcpy(cpu_value_cache + (size_t)position * kv_count, cpu_value, (size_t)kv_count * sizeof(float));
    cpu_gqa(cpu_query, cpu_key_cache, cpu_value_cache, cpu_gate, seq_len, meta.query_heads,
        meta.kv_heads, head_dim, cpu_attention, cpu_gated);
    for (uint32_t row = 0; row < hidden; ++row) {
        double dot = 0.0;
        if (!q4_k_row_dot(weights[6] + (size_t)row * q4_out_rb, cpu_gated, query_count, &dot)) {
            fprintf(stderr, "CPU Q4_K attention output failed\n"); goto cleanup;
        }
        cpu_output[row] = (float)dot;
    }
    const double cpu_compute_ms = now_ms() - cpu_start;

    rl_attn_proj_telemetry gpu_tel = {0};
#ifdef __APPLE__
    if (!rl_attention_gpu_execute(model, tensors, input, hidden, meta.rms_eps,
            position, meta.rope_dims, meta.rope_freq_base,
            gpu_key_cache, gpu_value_cache, (uint32_t)cache_count,
            gpu_input_norm, hidden, gpu_query, query_count, gpu_gate, query_count,
            gpu_key, kv_count, gpu_value, kv_count, gpu_attention, query_count,
            gpu_gated, query_count, gpu_output, hidden, &gpu_tel, error, sizeof(error))) {
        fprintf(stderr, "Metal full attention failed: %s\n", error); goto cleanup;
    }
#else
    fprintf(stderr, "full-attention parity requires macOS Metal\n"); goto cleanup;
#endif

    const error_stats norm_e = compare_arrays(gpu_input_norm, cpu_input_norm, hidden);
    const error_stats query_e = compare_arrays(gpu_query, cpu_query, query_count);
    const error_stats gate_e = compare_arrays(gpu_gate, cpu_gate, query_count);
    const error_stats key_e = compare_arrays(gpu_key, cpu_key, kv_count);
    const error_stats value_e = compare_arrays(gpu_value, cpu_value, kv_count);
    const error_stats cache_e = compare_arrays(gpu_key_cache, cpu_key_cache, cache_count);
    const error_stats attention_e = compare_arrays(gpu_attention, cpu_attention, query_count);
    const error_stats gated_e = compare_arrays(gpu_gated, cpu_gated, query_count);
    const error_stats output_e = compare_arrays(gpu_output, cpu_output, hidden);
    const int norm_ok = within(gpu_input_norm, cpu_input_norm, hidden, 3.0e-6f, 3.0e-5f);
    const int query_ok = within(gpu_query, cpu_query, query_count, 2.0e-4f, 2.0e-4f);
    const int gate_ok = within(gpu_gate, cpu_gate, query_count, 2.0e-4f, 2.0e-4f);
    const int key_ok = within(gpu_key, cpu_key, kv_count, 2.0e-4f, 2.0e-4f);
    const int value_ok = within(gpu_value, cpu_value, kv_count, 2.0e-4f, 2.0e-4f);
    const int cache_ok = within(gpu_key_cache, cpu_key_cache, cache_count, 2.0e-4f, 2.0e-4f);
    const int attention_ok = within(gpu_attention, cpu_attention, query_count, 4.0e-4f, 4.0e-4f);
    const int gated_ok = within(gpu_gated, cpu_gated, query_count, 4.0e-4f, 4.0e-4f);
    const int output_ok = within(gpu_output, cpu_output, hidden, 2.0e-3f, 4.0e-4f);
    const int all_ok = norm_ok && query_ok && gate_ok && key_ok && value_ok && cache_ok && attention_ok && gated_ok && output_ok;

    printf("runtime            : complete native single-token Qwen3-Next full-attention parity\n");
    printf("layer / position   : %u / %u (context=%u)\n", layer, position, seq_len);
    printf("hidden / heads     : %u / Q=%u KV=%u head_dim=%u\n", hidden, meta.query_heads, meta.kv_heads, head_dim);
    printf("RoPE               : NeoX dims=%u base=%.9g\n", meta.rope_dims, meta.rope_freq_base);
    printf("Q+gate             : IQ2_XXS(16) (%u,%u)\n", hidden, qgate_count);
    printf("K / V / output     : Q4_K(12) (%u,%u) / (%u,%u)\n", hidden, kv_count, query_count, hidden);
    printf("weight read CPU    : %.3f ms / %" PRIu64 " calls / %.3f MiB\n", cpu_read_ms, cpu_read_calls,
        (double)total_read_bytes / (1024.0 * 1024.0));
    printf("weight read GPU    : %.3f ms / %" PRIu64 " calls / %.3f MiB\n", gpu_tel.read_ms, gpu_tel.read_calls,
        (double)gpu_tel.bytes_read / (1024.0 * 1024.0));
    printf("SSD during compute : 0 bytes / 0 calls\n");
    printf("CPU / GPU compute  : %.3f / %.3f ms\n", cpu_compute_ms, gpu_tel.compute_ms);
    printf("input norm abs/rel : %.6g / %.6g parity=%s\n", norm_e.max_abs, norm_e.max_rel, norm_ok ? "YES" : "NO");
    printf("Q+RoPE abs/rel     : %.6g / %.6g parity=%s\n", query_e.max_abs, query_e.max_rel, query_ok ? "YES" : "NO");
    printf("gate abs/rel       : %.6g / %.6g parity=%s\n", gate_e.max_abs, gate_e.max_rel, gate_ok ? "YES" : "NO");
    printf("K+RoPE abs/rel     : %.6g / %.6g parity=%s\n", key_e.max_abs, key_e.max_rel, key_ok ? "YES" : "NO");
    printf("V abs/rel          : %.6g / %.6g parity=%s\n", value_e.max_abs, value_e.max_rel, value_ok ? "YES" : "NO");
    printf("KV cache abs/rel   : %.6g / %.6g parity=%s\n", cache_e.max_abs, cache_e.max_rel, cache_ok ? "YES" : "NO");
    printf("GQA softmax abs/rel: %.6g / %.6g parity=%s\n", attention_e.max_abs, attention_e.max_rel, attention_ok ? "YES" : "NO");
    printf("sigmoid gate abs/rel: %.6g / %.6g parity=%s\n", gated_e.max_abs, gated_e.max_rel, gated_ok ? "YES" : "NO");
    printf("Q4_K out abs/rel   : %.6g / %.6g parity=%s\n", output_e.max_abs, output_e.max_rel, output_ok ? "YES" : "NO");
    printf("FULL ATTENTION     : %s\n", all_ok ? "YES" : "NO");
    for (uint32_t i = 0; i < 4u; ++i)
        printf("row %-3u            : gpu=%+.7f cpu=%+.7f delta=%+.3e\n", i, gpu_output[i], cpu_output[i],
            (double)gpu_output[i] - cpu_output[i]);
    result = all_ok ? 0 : 3;

cleanup:
    for (uint32_t i = 0; i < RL_ATTN_PROJ_TENSOR_COUNT; ++i) free(weights[i]);
    free(input); free(cpu_input_norm); free(gpu_input_norm);
    free(cpu_query); free(gpu_query); free(cpu_gate); free(gpu_gate);
    free(cpu_key); free(gpu_key); free(cpu_value); free(gpu_value);
    free(cpu_attention); free(gpu_attention); free(cpu_gated); free(gpu_gated);
    free(cpu_output); free(gpu_output); free(cpu_key_cache); free(gpu_key_cache);
    free(cpu_value_cache); free(gpu_value_cache); free(qgate_raw); free(key_raw);
    return result;
}
