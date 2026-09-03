#define _FILE_OFFSET_BITS 64
#define _POSIX_C_SOURCE 200809L

#include "redlite_native_deltanet_layer.h"
#include "redlite_native_deltanet_prestate.h"
#include "redlite_native_deltanet_state.h"
#include "redlite_native_deltanet_tail.h"
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
#define RL_LAYER_MAX_DIMS 8u
#define IQ2_XXS_BLOCK_BYTES 66u
#define Q8_0_BLOCK_BYTES 34u
#define Q4_K_BLOCK_BYTES 144u
#define QK_K 256u

enum {
    GGUF_U8 = 0, GGUF_I8, GGUF_U16, GGUF_I16, GGUF_U32, GGUF_I32,
    GGUF_F32, GGUF_BOOL, GGUF_STRING, GGUF_ARRAY, GGUF_U64, GGUF_I64, GGUF_F64
};

enum {
    LT_ATTN_NORM = 0,
    LT_QKV,
    LT_Z,
    LT_BA,
    LT_CONV,
    LT_DT,
    LT_A,
    LT_SSM_NORM,
    LT_SSM_OUT,
    LT_COUNT
};

typedef struct {
    char *name;
    uint32_t n_dims;
    uint64_t shape[RL_LAYER_MAX_DIMS];
    uint32_t ggml_type;
    uint64_t relative_offset;
    uint64_t absolute_offset;
    uint64_t span_bytes;
} raw_tensor;

typedef struct {
    uint32_t alignment;
    float rms_eps;
    uint32_t d_conv;
    uint32_t d_inner;
    uint32_t d_state;
    uint32_t dt_rank;
    uint32_t n_group;
    int have_eps, have_conv, have_inner, have_state, have_rank, have_group;
} model_meta;

typedef struct {
    uint32_t ggml_type;
    uint32_t n_dims;
    uint64_t shape[RL_LAYER_MAX_DIMS];
    uint64_t offset;
    uint64_t span_bytes;
    int found;
} tensor_info;

typedef struct {
    float max_abs;
    float max_rel;
    size_t index;
} error_stats;

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

static void usage(const char *argv0) {
    fprintf(stderr,
        "redlite-deltanet-layer 0.3.0.dev15\n\n"
        "Usage:\n"
        "  %s parity MODEL --layer N\n\n"
        "Runs one complete single-token Qwen3-Next recurrent attention branch:\n"
        "RMSNorm -> full QKV/Z/BA -> conv/prestate -> recurrent update -> gated RMSNorm -> Q4_K ssm_out.\n",
        argv0);
}

static int parse_u32(const char *s, uint32_t *out) {
    char *end = NULL;
    errno = 0;
    unsigned long v = strtoul(s, &end, 10);
    if (errno || !end || *end || v > 0xfffffffful) return 0;
    *out = (uint32_t)v;
    return 1;
}

static int read_exact(FILE *f, void *dst, size_t n) {
    return n == 0 || fread(dst, 1, n, f) == n;
}

static int read_u32(FILE *f, uint32_t *v) {
    uint8_t b[4];
    if (!read_exact(f, b, 4)) return 0;
    *v = (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
    return 1;
}

static int read_u64(FILE *f, uint64_t *v) {
    uint8_t b[8];
    if (!read_exact(f, b, 8)) return 0;
    *v = (uint64_t)b[0] | ((uint64_t)b[1] << 8) | ((uint64_t)b[2] << 16) | ((uint64_t)b[3] << 24) |
         ((uint64_t)b[4] << 32) | ((uint64_t)b[5] << 40) | ((uint64_t)b[6] << 48) | ((uint64_t)b[7] << 56);
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
        case GGUF_U8: case GGUF_I8: case GGUF_BOOL: return 1;
        case GGUF_U16: case GGUF_I16: return 2;
        case GGUF_U32: case GGUF_I32: case GGUF_F32: return 4;
        case GGUF_U64: case GGUF_I64: case GGUF_F64: return 8;
        default: return 0;
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
        const size_t sf = scalar_size(subtype);
        if (sf) {
            if (count && count > UINT64_MAX / sf) return 0;
            return skip_bytes(f, count * sf);
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

        uint32_t *u32dst = NULL;
        int *flag = NULL;
        if (strcmp(key, "general.alignment") == 0 && type == GGUF_U32) {
            u32dst = &m->alignment;
        } else if (strcmp(key, "qwen3next.ssm.conv_kernel") == 0 && type == GGUF_U32) {
            u32dst = &m->d_conv; flag = &m->have_conv;
        } else if (strcmp(key, "qwen3next.ssm.inner_size") == 0 && type == GGUF_U32) {
            u32dst = &m->d_inner; flag = &m->have_inner;
        } else if (strcmp(key, "qwen3next.ssm.state_size") == 0 && type == GGUF_U32) {
            u32dst = &m->d_state; flag = &m->have_state;
        } else if (strcmp(key, "qwen3next.ssm.time_step_rank") == 0 && type == GGUF_U32) {
            u32dst = &m->dt_rank; flag = &m->have_rank;
        } else if (strcmp(key, "qwen3next.ssm.group_count") == 0 && type == GGUF_U32) {
            u32dst = &m->n_group; flag = &m->have_group;
        }

        if (u32dst) {
            uint32_t v = 0;
            free(key);
            if (!read_u32(f, &v)) return 0;
            *u32dst = v;
            if (flag) *flag = 1;
            continue;
        }

        if (strcmp(key, "qwen3next.attention.layer_norm_rms_epsilon") == 0 && type == GGUF_F32) {
            free(key);
            if (!read_f32(f, &m->rms_eps) || !isfinite(m->rms_eps) || m->rms_eps < 0.0f) return 0;
            m->have_eps = 1;
            continue;
        }

        free(key);
        if (!skip_value(f, type)) return 0;
    }
    return m->alignment && m->have_eps && m->have_conv && m->have_inner &&
           m->have_state && m->have_rank && m->have_group;
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

static int target_slot(const char *name, uint32_t layer) {
    char expected[96];
    const char *suffix[LT_COUNT] = {
        "attn_norm.weight",
        "attn_qkv.weight",
        "attn_gate.weight",
        "ssm_ba.weight",
        "ssm_conv1d.weight",
        "ssm_dt.bias",
        "ssm_a",
        "ssm_norm.weight",
        "ssm_out.weight",
    };
    for (int i = 0; i < LT_COUNT; ++i) {
        snprintf(expected, sizeof(expected), "blk.%u.%s", layer, suffix[i]);
        if (strcmp(name, expected) == 0) return i;
    }
    return -1;
}

static int audit_layer(const char *model, uint32_t layer, model_meta *meta,
        tensor_info out[LT_COUNT], char *error, size_t cap) {
    memset(out, 0, LT_COUNT * sizeof(*out));
    FILE *f = fopen(model, "rb");
    if (!f) { snprintf(error, cap, "open failed: %s", strerror(errno)); return 0; }
    if (fseeko(f, 0, SEEK_END) != 0) { fclose(f); snprintf(error, cap, "seek failed"); return 0; }
    const off_t end = ftello(f);
    if (end <= 0 || fseeko(f, 0, SEEK_SET) != 0) { fclose(f); snprintf(error, cap, "invalid GGUF size"); return 0; }
    const uint64_t file_size = (uint64_t)end;

    uint8_t magic[4];
    uint32_t version = 0;
    uint64_t tensor_count = 0, kv_count = 0;
    if (!read_exact(f, magic, 4) || memcmp(magic, "GGUF", 4) != 0 ||
        !read_u32(f, &version) || (version != 2u && version != 3u) ||
        !read_u64(f, &tensor_count) || !read_u64(f, &kv_count) ||
        tensor_count > SIZE_MAX / sizeof(raw_tensor) ||
        !read_metadata(f, kv_count, meta)) {
        fclose(f);
        snprintf(error, cap, "invalid GGUF/Qwen3-Next metadata");
        return 0;
    }

    raw_tensor *raw = (raw_tensor *)calloc((size_t)tensor_count, sizeof(*raw));
    if (!raw) { fclose(f); snprintf(error, cap, "out of memory for tensor directory"); return 0; }
    for (uint64_t i = 0; i < tensor_count; ++i) {
        uint32_t dims = 0;
        if (!read_string(f, &raw[i].name) || !read_u32(f, &dims) || dims > RL_LAYER_MAX_DIMS) {
            free_raw(raw, tensor_count); fclose(f); snprintf(error, cap, "tensor descriptor parse failed"); return 0;
        }
        raw[i].n_dims = dims;
        for (uint32_t d = 0; d < dims; ++d) {
            if (!read_u64(f, &raw[i].shape[d])) {
                free_raw(raw, tensor_count); fclose(f); snprintf(error, cap, "tensor shape parse failed"); return 0;
            }
        }
        if (!read_u32(f, &raw[i].ggml_type) || !read_u64(f, &raw[i].relative_offset)) {
            free_raw(raw, tensor_count); fclose(f); snprintf(error, cap, "tensor type/offset parse failed"); return 0;
        }
    }

    const off_t directory_end = ftello(f);
    fclose(f);
    if (directory_end < 0) {
        free_raw(raw, tensor_count);
        snprintf(error, cap, "failed to locate GGUF data section");
        return 0;
    }
    const uint64_t data_base = round_up_u64((uint64_t)directory_end, meta->alignment);
    if (!data_base || data_base > file_size) {
        free_raw(raw, tensor_count);
        snprintf(error, cap, "invalid GGUF data base");
        return 0;
    }

    qsort(raw, (size_t)tensor_count, sizeof(*raw), tensor_cmp);
    for (uint64_t i = 0; i < tensor_count; ++i) {
        const uint64_t rel = raw[i].relative_offset;
        const uint64_t next = i + 1 < tensor_count ? raw[i + 1].relative_offset : file_size - data_base;
        raw[i].absolute_offset = data_base + rel;
        raw[i].span_bytes = next >= rel ? next - rel : 0;
        const int slot = target_slot(raw[i].name, layer);
        if (slot < 0) continue;
        if (out[slot].found) {
            free_raw(raw, tensor_count);
            snprintf(error, cap, "duplicate DeltaNet layer tensor");
            return 0;
        }
        out[slot].found = 1;
        out[slot].ggml_type = raw[i].ggml_type;
        out[slot].n_dims = raw[i].n_dims;
        memcpy(out[slot].shape, raw[i].shape, sizeof(out[slot].shape));
        out[slot].offset = raw[i].absolute_offset;
        out[slot].span_bytes = raw[i].span_bytes;
    }
    free_raw(raw, tensor_count);

    for (int i = 0; i < LT_COUNT; ++i) {
        if (!out[i].found) {
            snprintf(error, cap, "requested layer is not a complete optimized DeltaNet layer (slot %d missing)", i);
            return 0;
        }
    }
    return 1;
}

static size_t iq2_row_bytes(uint32_t ncols) {
    return ncols && ncols % 256u == 0 ? (size_t)(ncols / 256u) * IQ2_XXS_BLOCK_BYTES : 0;
}

static size_t q8_row_bytes(uint32_t ncols) {
    return ncols && ncols % 32u == 0 ? (size_t)(ncols / 32u) * Q8_0_BLOCK_BYTES : 0;
}

static size_t q4_row_bytes(uint32_t ncols) {
    return ncols && ncols % QK_K == 0 ? (size_t)(ncols / QK_K) * Q4_K_BLOCK_BYTES : 0;
}

static int validate_layout(const model_meta *m, const tensor_info t[LT_COUNT],
        uint32_t *hidden, uint32_t *channels, uint32_t *qk_each, uint32_t *head_v,
        char *error, size_t cap) {
    if (!m->d_conv || !m->d_inner || !m->d_state || !m->dt_rank || !m->n_group ||
        m->dt_rank % m->n_group || m->d_inner % m->dt_rank) {
        snprintf(error, cap, "invalid SSM metadata dimensions");
        return 0;
    }
    *head_v = m->d_inner / m->dt_rank;
    *qk_each = m->d_state * m->n_group;
    *channels = 2u * (*qk_each) + m->d_inner;
    if (*head_v != m->d_state) {
        snprintf(error, cap, "DeltaNet head_v_dim != recurrent state size");
        return 0;
    }

    if (t[LT_ATTN_NORM].ggml_type != 0u || t[LT_ATTN_NORM].n_dims != 1u ||
        t[LT_ATTN_NORM].shape[0] == 0 || t[LT_ATTN_NORM].shape[0] > UINT32_MAX) {
        snprintf(error, cap, "attn_norm must be F32[hidden]");
        return 0;
    }
    *hidden = (uint32_t)t[LT_ATTN_NORM].shape[0];

    if (t[LT_QKV].ggml_type != 16u || t[LT_QKV].n_dims != 2u ||
        t[LT_QKV].shape[0] != *hidden || t[LT_QKV].shape[1] != *channels ||
        t[LT_Z].ggml_type != 16u || t[LT_Z].n_dims != 2u ||
        t[LT_Z].shape[0] != *hidden || t[LT_Z].shape[1] != m->d_inner ||
        t[LT_BA].ggml_type != 8u || t[LT_BA].n_dims != 2u ||
        t[LT_BA].shape[0] != *hidden || t[LT_BA].shape[1] != 2u * m->dt_rank) {
        snprintf(error, cap, "projection tensor geometry mismatch");
        return 0;
    }

    if (t[LT_CONV].ggml_type != 0u || t[LT_CONV].n_dims != 2u ||
        t[LT_CONV].shape[0] != m->d_conv || t[LT_CONV].shape[1] != *channels ||
        t[LT_DT].ggml_type != 0u || t[LT_DT].n_dims != 1u || t[LT_DT].shape[0] != m->dt_rank ||
        t[LT_A].ggml_type != 0u || t[LT_A].n_dims != 1u || t[LT_A].shape[0] != m->dt_rank) {
        snprintf(error, cap, "prestate tensor geometry mismatch");
        return 0;
    }

    if (t[LT_SSM_NORM].ggml_type != 0u || t[LT_SSM_NORM].n_dims != 1u ||
        t[LT_SSM_NORM].shape[0] != *head_v ||
        t[LT_SSM_OUT].ggml_type != 12u || t[LT_SSM_OUT].n_dims != 2u ||
        t[LT_SSM_OUT].shape[0] != m->d_inner || t[LT_SSM_OUT].shape[1] != *hidden) {
        snprintf(error, cap, "DeltaNet tail tensor geometry mismatch");
        return 0;
    }

    const size_t iq2_rb = iq2_row_bytes(*hidden);
    const size_t q8_rb = q8_row_bytes(*hidden);
    const size_t q4_rb = q4_row_bytes(m->d_inner);
    if (!iq2_rb || !q8_rb || !q4_rb ||
        t[LT_ATTN_NORM].span_bytes < (uint64_t)*hidden * sizeof(float) ||
        t[LT_QKV].span_bytes < (uint64_t)(*channels) * iq2_rb ||
        t[LT_Z].span_bytes < (uint64_t)m->d_inner * iq2_rb ||
        t[LT_BA].span_bytes < (uint64_t)(2u * m->dt_rank) * q8_rb ||
        t[LT_CONV].span_bytes < (uint64_t)m->d_conv * (*channels) * sizeof(float) ||
        t[LT_DT].span_bytes < (uint64_t)m->dt_rank * sizeof(float) ||
        t[LT_A].span_bytes < (uint64_t)m->dt_rank * sizeof(float) ||
        t[LT_SSM_NORM].span_bytes < (uint64_t)(*head_v) * sizeof(float) ||
        t[LT_SSM_OUT].span_bytes < (uint64_t)(*hidden) * q4_rb) {
        snprintf(error, cap, "physical tensor span mismatch");
        return 0;
    }
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

static int q8_row_dot(const uint8_t *row, const float *x, uint32_t ncols, double *out) {
    const size_t rb = q8_row_bytes(ncols);
    if (!row || !x || !out || !rb) return 0;
    double acc = 0.0;
    const uint32_t blocks = ncols / 32u;
    for (uint32_t ib = 0; ib < blocks; ++ib) {
        const uint8_t *bp = row + (size_t)ib * Q8_0_BLOCK_BYTES;
        const double d = (double)f16_to_f32(read_u16_le(bp));
        const int8_t *q = (const int8_t *)(bp + 2u);
        for (uint32_t j = 0; j < 32u; ++j) acc += (double)x[ib * 32u + j] * d * (double)q[j];
    }
    *out = acc;
    return 1;
}

static void get_scale_min_k4(uint32_t j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4u) {
        *d = q[j] & 63u;
        *m = q[j + 4u] & 63u;
    } else {
        *d = (q[j + 4u] & 0x0fu) | ((q[j - 4u] >> 6) << 4);
        *m = (q[j + 4u] >> 4) | ((q[j] >> 6) << 4);
    }
}

static int q4_row_dot(const uint8_t *row, const float *x, uint32_t ncols, double *out) {
    if (!row || !x || !out || !ncols || ncols % QK_K) return 0;
    double acc = 0.0;
    const uint32_t blocks = ncols / QK_K;
    for (uint32_t ib = 0; ib < blocks; ++ib) {
        const uint8_t *bp = row + (size_t)ib * Q4_K_BLOCK_BYTES;
        const float d = f16_to_f32(read_u16_le(bp));
        const float dmin = f16_to_f32(read_u16_le(bp + 2u));
        const uint8_t *scales = bp + 4u;
        const uint8_t *qs = bp + 16u;
        for (uint32_t g = 0; g < 8u; ++g) {
            uint8_t sc = 0, mn = 0;
            get_scale_min_k4(g, scales, &sc, &mn);
            const double ds = (double)d * sc;
            const double dm = (double)dmin * mn;
            const uint8_t *q = qs + (g / 2u) * 32u;
            const uint32_t xb = ib * 256u + g * 32u;
            for (uint32_t l = 0; l < 32u; ++l) {
                const uint8_t quant = (g & 1u) ? (q[l] >> 4) : (q[l] & 0x0fu);
                acc += (double)x[xb + l] * (ds * quant - dm);
            }
        }
    }
    *out = acc;
    return 1;
}

static void make_input(float *x, uint32_t n) {
    for (uint32_t i = 0; i < n; ++i)
        x[i] = (float)(((int)((i * 17u + 3u) % 61u) - 30) / 32.0);
}

static void make_conv_state(float *x, size_t n) {
    for (size_t i = 0; i < n; ++i)
        x[i] = (float)(((int)(((uint32_t)i * 7u + 19u) % 43u) - 21) / 48.0);
}

static void make_recurrent_state(float *x, uint32_t heads, uint32_t s) {
    const size_t matrix = (size_t)s * s;
    for (uint32_t h = 0; h < heads; ++h) {
        for (uint32_t j = 0; j < s; ++j) {
            for (uint32_t i = 0; i < s; ++i) {
                const int sn = (int)((h * 37u + j * 11u + i * 7u + 3u) % 127u) - 63;
                x[(size_t)h * matrix + (size_t)j * s + i] = (float)sn / 8192.0f;
            }
        }
    }
}

static void rmsnorm_cpu(const float *x, const float *w, uint32_t n, float eps, float *out) {
    double ss = 0.0;
    for (uint32_t i = 0; i < n; ++i) ss += (double)x[i] * x[i];
    const double inv = 1.0 / sqrt(ss / n + (double)eps);
    for (uint32_t i = 0; i < n; ++i) out[i] = (float)((double)x[i] * inv * w[i]);
}

static double sigmoid_stable(double x) {
    if (x >= 0.0) { const double z = exp(-x); return 1.0 / (1.0 + z); }
    const double z = exp(x); return z / (1.0 + z);
}

static double softplus_stable(double x) {
    if (x > 20.0) return x;
    if (x < -20.0) return exp(x);
    return log1p(exp(x));
}

static void cpu_ba_params(const float *ba, const float *dt, const float *avec,
        uint32_t rank, uint32_t groups, float *beta, float *gate) {
    const uint32_t width = rank / groups;
    const uint32_t stride = 2u * width;
    for (uint32_t h = 0; h < rank; ++h) {
        const uint32_t g = h / width;
        const uint32_t l = h - g * width;
        const double b = ba[g * stride + l];
        const double a = (double)ba[g * stride + width + l] + dt[h];
        beta[h] = (float)sigmoid_stable(b);
        gate[h] = (float)(softplus_stable(a) * (double)avec[h]);
    }
}

static void cpu_conv_silu(const float *state, const float *qkv, const float *kernel,
        uint32_t channels, uint32_t dconv, float *out) {
    const uint32_t ns = dconv - 1u;
    for (uint32_t c = 0; c < channels; ++c) {
        double acc = 0.0;
        for (uint32_t j = 0; j < ns; ++j)
            acc += (double)state[(size_t)c * ns + j] * kernel[(size_t)c * dconv + j];
        acc += (double)qkv[c] * kernel[(size_t)c * dconv + ns];
        out[c] = (float)(acc / (1.0 + exp(-acc)));
    }
}

static void cpu_l2_heads(const float *src, uint32_t offset, uint32_t head_dim,
        uint32_t heads, float eps, float *dst) {
    for (uint32_t h = 0; h < heads; ++h) {
        const uint32_t b = offset + h * head_dim;
        double ss = 0.0;
        for (uint32_t j = 0; j < head_dim; ++j) ss += (double)src[b + j] * src[b + j];
        const double inv = 1.0 / fmax(sqrt(ss), (double)eps);
        for (uint32_t j = 0; j < head_dim; ++j)
            dst[(size_t)h * head_dim + j] = (float)((double)src[b + j] * inv);
    }
}

static void cpu_shift_conv_state(const float *state, const float *qkv,
        uint32_t channels, uint32_t dconv, float *next) {
    const uint32_t ns = dconv - 1u;
    for (uint32_t c = 0; c < channels; ++c) {
        const size_t b = (size_t)c * ns;
        for (uint32_t j = 0; j + 1u < ns; ++j) next[b + j] = state[b + j + 1u];
        next[b + ns - 1u] = qkv[c];
    }
}

static void cpu_state_update(const float *q, const float *k, const float *v,
        const float *gate, const float *beta, const float *prev_state,
        uint32_t state_size, uint32_t key_heads, uint32_t value_heads,
        float *delta, float *next_state, float *output) {
    const size_t matrix = (size_t)state_size * state_size;
    const float scale = 1.0f / sqrtf((float)state_size);
    memcpy(next_state, prev_state, matrix * value_heads * sizeof(float));

    for (uint32_t h = 0; h < value_heads; ++h) {
        const uint32_t kh = h / (value_heads / key_heads); /* pinned llama.cpp repeat-interleave */
        const float *qh = q + (size_t)kh * state_size;
        const float *khv = k + (size_t)kh * state_size;
        const float *vh = v + (size_t)h * state_size;
        float *mh = next_state + (size_t)h * matrix;
        float *dh = delta + (size_t)h * state_size;
        float *oh = output + (size_t)h * state_size;
        const float decay = expf(gate[h]);

        for (uint32_t j = 0; j < state_size; ++j) {
            float *row = mh + (size_t)j * state_size;
            for (uint32_t i = 0; i < state_size; ++i) row[i] *= decay;
        }
        for (uint32_t j = 0; j < state_size; ++j) {
            float sum = 0.0f;
            const float *row = mh + (size_t)j * state_size;
            for (uint32_t i = 0; i < state_size; ++i) sum += row[i] * khv[i];
            dh[j] = (vh[j] - sum) * beta[h];
        }
        for (uint32_t j = 0; j < state_size; ++j) {
            float *row = mh + (size_t)j * state_size;
            const float d = dh[j];
            for (uint32_t i = 0; i < state_size; ++i) row[i] += khv[i] * d;
        }
        for (uint32_t j = 0; j < state_size; ++j) {
            float sum = 0.0f;
            const float *row = mh + (size_t)j * state_size;
            for (uint32_t i = 0; i < state_size; ++i) sum += row[i] * qh[i];
            oh[j] = sum * scale;
        }
    }
}

static void cpu_gated_norm(const float *core, const float *z, const float *w, float eps,
        uint32_t head_dim, uint32_t heads, float *out) {
    for (uint32_t h = 0; h < heads; ++h) {
        const float *x = core + (size_t)h * head_dim;
        const float *g = z + (size_t)h * head_dim;
        float *y = out + (size_t)h * head_dim;
        double ss = 0.0;
        for (uint32_t i = 0; i < head_dim; ++i) ss += (double)x[i] * x[i];
        const float inv = 1.0f / sqrtf((float)(ss / head_dim) + eps);
        for (uint32_t i = 0; i < head_dim; ++i) {
            const float silu = g[i] / (1.0f + expf(-g[i]));
            y[i] = x[i] * inv * w[i] * silu;
        }
    }
}

static error_stats compare_arrays(const float *got, const float *ref, size_t n) {
    error_stats s = {0.0f, 0.0f, 0u};
    for (size_t i = 0; i < n; ++i) {
        const float a = fabsf(got[i] - ref[i]);
        const float r = a / (fabsf(ref[i]) + 1.0e-12f);
        if (a > s.max_abs) { s.max_abs = a; s.index = i; }
        if (r > s.max_rel) s.max_rel = r;
    }
    return s;
}

static int within(const float *got, const float *ref, size_t n, float abs_tol, float rel_tol) {
    for (size_t i = 0; i < n; ++i) {
        if (fabsf(got[i] - ref[i]) > abs_tol + rel_tol * fabsf(ref[i])) return 0;
    }
    return 1;
}

static void fill_proj_info(const tensor_info src[LT_COUNT], uint32_t layer,
        rl_dn_proj_tensor_info dst[RL_DN_PROJ_TENSOR_COUNT]) {
    const int map[RL_DN_PROJ_TENSOR_COUNT] = {LT_ATTN_NORM, LT_QKV, LT_Z, LT_BA};
    memset(dst, 0, RL_DN_PROJ_TENSOR_COUNT * sizeof(*dst));
    for (uint32_t i = 0; i < RL_DN_PROJ_TENSOR_COUNT; ++i) {
        const tensor_info *s = &src[map[i]];
        dst[i].layer = layer;
        dst[i].kind = (rl_dn_proj_kind)i;
        dst[i].ggml_type = s->ggml_type;
        dst[i].n_dims = s->n_dims;
        memcpy(dst[i].shape, s->shape, sizeof(dst[i].shape));
        dst[i].tensor_offset = s->offset;
        dst[i].tensor_span_bytes = s->span_bytes;
    }
}

static void fill_prestate_info(const tensor_info src[LT_COUNT], uint32_t layer,
        rl_dn_prestate_tensor_info dst[RL_DN_PRESTATE_TENSOR_COUNT]) {
    const int map[RL_DN_PRESTATE_TENSOR_COUNT] = {LT_CONV, LT_DT, LT_A};
    memset(dst, 0, RL_DN_PRESTATE_TENSOR_COUNT * sizeof(*dst));
    for (uint32_t i = 0; i < RL_DN_PRESTATE_TENSOR_COUNT; ++i) {
        const tensor_info *s = &src[map[i]];
        dst[i].layer = layer;
        dst[i].kind = (rl_dn_prestate_kind)i;
        dst[i].ggml_type = s->ggml_type;
        dst[i].n_dims = s->n_dims;
        memcpy(dst[i].shape, s->shape, sizeof(dst[i].shape));
        dst[i].tensor_offset = s->offset;
        dst[i].tensor_span_bytes = s->span_bytes;
    }
}

int main(int argc, char **argv) {
    if (argc < 5 || strcmp(argv[1], "parity") != 0 || strcmp(argv[3], "--layer") != 0) {
        usage(argv[0]);
        return 2;
    }
    uint32_t layer = 0;
    if (!parse_u32(argv[4], &layer)) {
        fprintf(stderr, "invalid layer\n");
        return 2;
    }
    const char *model = argv[2];

    char error[512] = {0};
    model_meta meta;
    tensor_info t[LT_COUNT];
    if (!audit_layer(model, layer, &meta, t, error, sizeof(error))) {
        fprintf(stderr, "DeltaNet layer audit failed: %s\n", error);
        return 1;
    }

    uint32_t hidden = 0, channels = 0, qk_each = 0, head_v = 0;
    if (!validate_layout(&meta, t, &hidden, &channels, &qk_each, &head_v, error, sizeof(error))) {
        fprintf(stderr, "DeltaNet layer layout failed: %s\n", error);
        return 1;
    }
    const uint32_t z_count = meta.d_inner;
    const uint32_t ba_count = 2u * meta.dt_rank;
    const size_t conv_state_count = (size_t)(meta.d_conv - 1u) * channels;
    const size_t recurrent_state_count = (size_t)meta.dt_rank * head_v * head_v;
    const size_t core_count = (size_t)meta.dt_rank * head_v;

    const size_t iq2_rb = iq2_row_bytes(hidden);
    const size_t q8_rb = q8_row_bytes(hidden);
    const size_t q4_rb = q4_row_bytes(meta.d_inner);

    const size_t norm_bytes = (size_t)hidden * sizeof(float);
    const size_t qkv_weight_bytes = (size_t)channels * iq2_rb;
    const size_t z_weight_bytes = (size_t)z_count * iq2_rb;
    const size_t ba_weight_bytes = (size_t)ba_count * q8_rb;
    const size_t conv_weight_bytes = (size_t)meta.d_conv * channels * sizeof(float);
    const size_t vec_bytes = (size_t)meta.dt_rank * sizeof(float);
    const size_t tail_norm_bytes = (size_t)head_v * sizeof(float);
    const size_t q4_weight_bytes = (size_t)hidden * q4_rb;

    float *attn_norm_w = malloc(norm_bytes);
    uint8_t *qkv_w = malloc(qkv_weight_bytes);
    uint8_t *z_w = malloc(z_weight_bytes);
    uint8_t *ba_w = malloc(ba_weight_bytes);
    float *conv_w = malloc(conv_weight_bytes);
    float *dt_w = malloc(vec_bytes);
    float *a_w = malloc(vec_bytes);
    float *ssm_norm_w = malloc(tail_norm_bytes);
    uint8_t *ssm_out_w = malloc(q4_weight_bytes);

    float *input = calloc(hidden, sizeof(float));
    float *conv_state = calloc(conv_state_count, sizeof(float));
    float *recurrent_state = calloc(recurrent_state_count, sizeof(float));

    float *cpu_norm = calloc(hidden, sizeof(float));
    float *cpu_qkv = calloc(channels, sizeof(float));
    float *cpu_z = calloc(z_count, sizeof(float));
    float *cpu_ba = calloc(ba_count, sizeof(float));
    float *cpu_beta = calloc(meta.dt_rank, sizeof(float));
    float *cpu_gate = calloc(meta.dt_rank, sizeof(float));
    float *cpu_conv = calloc(channels, sizeof(float));
    float *cpu_q = calloc(qk_each, sizeof(float));
    float *cpu_k = calloc(qk_each, sizeof(float));
    float *cpu_v = calloc(meta.d_inner, sizeof(float));
    float *cpu_next_conv = calloc(conv_state_count, sizeof(float));
    float *cpu_delta = calloc(core_count, sizeof(float));
    float *cpu_next_state = calloc(recurrent_state_count, sizeof(float));
    float *cpu_core = calloc(core_count, sizeof(float));
    float *cpu_ng = calloc(meta.d_inner, sizeof(float));
    float *cpu_out = calloc(hidden, sizeof(float));

    float *gpu_norm = calloc(hidden, sizeof(float));
    float *gpu_qkv = calloc(channels, sizeof(float));
    float *gpu_z = calloc(z_count, sizeof(float));
    float *gpu_ba = calloc(ba_count, sizeof(float));
    float *gpu_beta = calloc(meta.dt_rank, sizeof(float));
    float *gpu_gate = calloc(meta.dt_rank, sizeof(float));
    float *gpu_conv = calloc(channels, sizeof(float));
    float *gpu_q = calloc(qk_each, sizeof(float));
    float *gpu_k = calloc(qk_each, sizeof(float));
    float *gpu_v = calloc(meta.d_inner, sizeof(float));
    float *gpu_next_conv = calloc(conv_state_count, sizeof(float));
    float *gpu_delta = calloc(core_count, sizeof(float));
    float *gpu_next_state = calloc(recurrent_state_count, sizeof(float));
    float *gpu_core = calloc(core_count, sizeof(float));
    float *gpu_ng = calloc(meta.d_inner, sizeof(float));
    float *gpu_out = calloc(hidden, sizeof(float));

    if (!attn_norm_w || !qkv_w || !z_w || !ba_w || !conv_w || !dt_w || !a_w || !ssm_norm_w || !ssm_out_w ||
        !input || !conv_state || !recurrent_state || !cpu_norm || !cpu_qkv || !cpu_z || !cpu_ba ||
        !cpu_beta || !cpu_gate || !cpu_conv || !cpu_q || !cpu_k || !cpu_v || !cpu_next_conv ||
        !cpu_delta || !cpu_next_state || !cpu_core || !cpu_ng || !cpu_out ||
        !gpu_norm || !gpu_qkv || !gpu_z || !gpu_ba || !gpu_beta || !gpu_gate || !gpu_conv ||
        !gpu_q || !gpu_k || !gpu_v || !gpu_next_conv || !gpu_delta || !gpu_next_state ||
        !gpu_core || !gpu_ng || !gpu_out) {
        fprintf(stderr, "allocation failed for complete DeltaNet layer parity\n");
        goto fail;
    }

    make_input(input, hidden);
    make_conv_state(conv_state, conv_state_count);
    make_recurrent_state(recurrent_state, meta.dt_rank, head_v);

    int fd = open(model, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "open model failed: %s\n", strerror(errno));
        goto fail;
    }
    uint64_t cpu_read_calls = 0;
    const double r0 = now_ms();
    const int read_ok =
        pread_full(fd, attn_norm_w, norm_bytes, t[LT_ATTN_NORM].offset, &cpu_read_calls) &&
        pread_full(fd, qkv_w, qkv_weight_bytes, t[LT_QKV].offset, &cpu_read_calls) &&
        pread_full(fd, z_w, z_weight_bytes, t[LT_Z].offset, &cpu_read_calls) &&
        pread_full(fd, ba_w, ba_weight_bytes, t[LT_BA].offset, &cpu_read_calls) &&
        pread_full(fd, conv_w, conv_weight_bytes, t[LT_CONV].offset, &cpu_read_calls) &&
        pread_full(fd, dt_w, vec_bytes, t[LT_DT].offset, &cpu_read_calls) &&
        pread_full(fd, a_w, vec_bytes, t[LT_A].offset, &cpu_read_calls) &&
        pread_full(fd, ssm_norm_w, tail_norm_bytes, t[LT_SSM_NORM].offset, &cpu_read_calls) &&
        pread_full(fd, ssm_out_w, q4_weight_bytes, t[LT_SSM_OUT].offset, &cpu_read_calls);
    const double r1 = now_ms();
    close(fd);
    if (!read_ok) {
        fprintf(stderr, "complete DeltaNet CPU weight pread failed\n");
        goto fail;
    }

    uint8_t grid[RL_IQ2_XXS_GRID_COUNT];
    if (!rl_native_iq2_xxs_build_grid(grid, error, sizeof(error))) {
        fprintf(stderr, "IQ2_XXS grid build failed: %s\n", error);
        goto fail;
    }

    const double c0 = now_ms();
    rmsnorm_cpu(input, attn_norm_w, hidden, meta.rms_eps, cpu_norm);

    for (uint32_t r = 0; r < channels; ++r) {
        double dot = 0.0;
        if (!rl_native_shared_quant_row_dot(qkv_w + (size_t)r * iq2_rb, iq2_rb, 16u,
                cpu_norm, hidden, grid, sizeof(grid), &dot, error, sizeof(error))) {
            fprintf(stderr, "CPU full QKV projection failed at row %u: %s\n", r, error);
            goto fail;
        }
        cpu_qkv[r] = (float)dot;
    }
    for (uint32_t r = 0; r < z_count; ++r) {
        double dot = 0.0;
        if (!rl_native_shared_quant_row_dot(z_w + (size_t)r * iq2_rb, iq2_rb, 16u,
                cpu_norm, hidden, grid, sizeof(grid), &dot, error, sizeof(error))) {
            fprintf(stderr, "CPU full Z projection failed at row %u: %s\n", r, error);
            goto fail;
        }
        cpu_z[r] = (float)dot;
    }
    for (uint32_t r = 0; r < ba_count; ++r) {
        double dot = 0.0;
        if (!q8_row_dot(ba_w + (size_t)r * q8_rb, cpu_norm, hidden, &dot)) {
            fprintf(stderr, "CPU full BA projection failed at row %u\n", r);
            goto fail;
        }
        cpu_ba[r] = (float)dot;
    }

    cpu_ba_params(cpu_ba, dt_w, a_w, meta.dt_rank, meta.n_group, cpu_beta, cpu_gate);
    cpu_conv_silu(conv_state, cpu_qkv, conv_w, channels, meta.d_conv, cpu_conv);
    cpu_l2_heads(cpu_conv, 0u, meta.d_state, meta.n_group, meta.rms_eps, cpu_q);
    cpu_l2_heads(cpu_conv, qk_each, meta.d_state, meta.n_group, meta.rms_eps, cpu_k);
    memcpy(cpu_v, cpu_conv + 2u * qk_each, (size_t)meta.d_inner * sizeof(float));
    cpu_shift_conv_state(conv_state, cpu_qkv, channels, meta.d_conv, cpu_next_conv);

    cpu_state_update(cpu_q, cpu_k, cpu_v, cpu_gate, cpu_beta, recurrent_state,
        head_v, meta.n_group, meta.dt_rank, cpu_delta, cpu_next_state, cpu_core);

    cpu_gated_norm(cpu_core, cpu_z, ssm_norm_w, meta.rms_eps, head_v, meta.dt_rank, cpu_ng);
    for (uint32_t r = 0; r < hidden; ++r) {
        double dot = 0.0;
        if (!q4_row_dot(ssm_out_w + (size_t)r * q4_rb, cpu_ng, meta.d_inner, &dot)) {
            fprintf(stderr, "CPU Q4_K ssm_out failed at row %u\n", r);
            goto fail;
        }
        cpu_out[r] = (float)dot;
    }
    const double c1 = now_ms();

#ifdef __APPLE__
    rl_dn_proj_tensor_info proj_info[RL_DN_PROJ_TENSOR_COUNT];
    rl_dn_prestate_tensor_info pre_info[RL_DN_PRESTATE_TENSOR_COUNT];
    fill_proj_info(t, layer, proj_info);
    fill_prestate_info(t, layer, pre_info);

    rl_dn_proj_telemetry p_tel = {0};
    if (!rl_deltanet_proj_gpu_execute_full(model, proj_info, input, hidden, meta.rms_eps,
            gpu_norm, hidden, gpu_qkv, channels, gpu_z, z_count, gpu_ba, ba_count,
            &p_tel, error, sizeof(error))) {
        fprintf(stderr, "Metal full projection failed: %s\n", error);
        goto fail;
    }

    rl_dn_prestate_telemetry pre_tel = {0};
    if (!rl_deltanet_prestate_gpu_execute(model, pre_info,
            gpu_qkv, channels, gpu_ba, ba_count, conv_state, (uint32_t)conv_state_count,
            meta.d_conv, meta.d_inner, meta.d_state, meta.dt_rank, meta.n_group, meta.rms_eps,
            gpu_beta, gpu_gate, gpu_conv, gpu_q, gpu_k, gpu_v, gpu_next_conv,
            &pre_tel, error, sizeof(error))) {
        fprintf(stderr, "Metal composed prestate failed: %s\n", error);
        goto fail;
    }

    rl_dn_state_telemetry state_tel = {0};
    if (!rl_deltanet_state_gpu_execute(gpu_q, gpu_k, gpu_v, gpu_gate, gpu_beta, recurrent_state,
            head_v, meta.n_group, meta.dt_rank, gpu_delta, gpu_next_state, gpu_core,
            &state_tel, error, sizeof(error))) {
        fprintf(stderr, "Metal composed recurrent state failed: %s\n", error);
        goto fail;
    }

    rl_dn_tail_telemetry tail_tel = {0};
    if (!rl_deltanet_tail_gpu_execute(gpu_core, gpu_z, ssm_norm_w, ssm_out_w, q4_weight_bytes,
            meta.rms_eps, head_v, meta.dt_rank, hidden, gpu_ng, gpu_out,
            &tail_tel, error, sizeof(error))) {
        fprintf(stderr, "Metal composed DeltaNet tail failed: %s\n", error);
        goto fail;
    }
#else
    fprintf(stderr, "complete DeltaNet layer parity requires macOS Metal\n");
    goto fail;
#endif

    const error_stats pe_qkv = compare_arrays(gpu_qkv, cpu_qkv, channels);
    const error_stats pe_z = compare_arrays(gpu_z, cpu_z, z_count);
    const error_stats pe_ba = compare_arrays(gpu_ba, cpu_ba, ba_count);
    const error_stats pre_q = compare_arrays(gpu_q, cpu_q, qk_each);
    const error_stats pre_k = compare_arrays(gpu_k, cpu_k, qk_each);
    const error_stats pre_v = compare_arrays(gpu_v, cpu_v, meta.d_inner);
    const error_stats pre_gate = compare_arrays(gpu_gate, cpu_gate, meta.dt_rank);
    const error_stats conv_state_err = compare_arrays(gpu_next_conv, cpu_next_conv, conv_state_count);
    const error_stats state_err = compare_arrays(gpu_next_state, cpu_next_state, recurrent_state_count);
    const error_stats core_err = compare_arrays(gpu_core, cpu_core, core_count);
    const error_stats tail_norm_err = compare_arrays(gpu_ng, cpu_ng, meta.d_inner);
    const error_stats out_err = compare_arrays(gpu_out, cpu_out, hidden);

    const int proj_ok =
        within(gpu_qkv, cpu_qkv, channels, 2.0e-3f, 2.0e-4f) &&
        within(gpu_z, cpu_z, z_count, 2.0e-3f, 2.0e-4f) &&
        within(gpu_ba, cpu_ba, ba_count, 2.0e-3f, 2.0e-4f);
    const int pre_ok =
        within(gpu_q, cpu_q, qk_each, 3.0e-4f, 3.0e-4f) &&
        within(gpu_k, cpu_k, qk_each, 3.0e-4f, 3.0e-4f) &&
        within(gpu_v, cpu_v, meta.d_inner, 3.0e-4f, 3.0e-4f) &&
        within(gpu_gate, cpu_gate, meta.dt_rank, 3.0e-4f, 3.0e-4f);
    const int conv_ok = within(gpu_next_conv, cpu_next_conv, conv_state_count, 2.0e-3f, 2.0e-4f);
    const int state_ok = within(gpu_next_state, cpu_next_state, recurrent_state_count, 5.0e-5f, 5.0e-4f);
    const int core_ok = within(gpu_core, cpu_core, core_count, 1.0e-4f, 8.0e-4f);
    const int norm_ok = within(gpu_ng, cpu_ng, meta.d_inner, 3.0e-4f, 8.0e-4f);
    const int out_ok = within(gpu_out, cpu_out, hidden, 5.0e-4f, 1.0e-3f);
    const int all_ok = proj_ok && pre_ok && conv_ok && state_ok && core_ok && norm_ok && out_ok;

    const uint64_t cpu_weight_bytes =
        (uint64_t)norm_bytes + qkv_weight_bytes + z_weight_bytes + ba_weight_bytes +
        conv_weight_bytes + 2u * vec_bytes + tail_norm_bytes + q4_weight_bytes;

    printf("runtime             : complete native single-token Qwen3-Next DeltaNet layer parity\n");
    printf("layer               : %u\n", layer);
    printf("geometry            : hidden=%u channels=%u qk=%u value=%u heads(k/v)=%u/%u\n",
        hidden, channels, qk_each, meta.d_inner, meta.n_group, meta.dt_rank);
    printf("state               : conv=%.3f KiB recurrent=%.3f MiB\n",
        (double)(conv_state_count * sizeof(float)) / 1024.0,
        (double)(recurrent_state_count * sizeof(float)) / (1024.0 * 1024.0));
    printf("CPU weight read     : %.3f ms / %" PRIu64 " calls / %.3f MiB\n",
        r1 - r0, cpu_read_calls, (double)cpu_weight_bytes / (1024.0 * 1024.0));
    printf("CPU full compute    : %.3f ms\n", c1 - c0);
#ifdef __APPLE__
    printf("GPU projection      : %.3f ms compute / %.3f ms read / %.3f MiB\n",
        p_tel.compute_ms, p_tel.read_ms, (double)p_tel.bytes_read / (1024.0 * 1024.0));
    printf("GPU prestate/state  : %.3f / %.3f ms\n", pre_tel.compute_ms, state_tel.compute_ms);
    printf("GPU tail            : %.3f ms\n", tail_tel.compute_ms);
    printf("SSD during compute  : prestate=%" PRIu64 " bytes/%" PRIu64 " calls; other stages=0\n",
        pre_tel.ssd_during_compute_bytes, pre_tel.ssd_during_compute_calls);
#endif
    printf("full QKV abs/rel    : %.6g / %.6g\n", pe_qkv.max_abs, pe_qkv.max_rel);
    printf("full Z abs/rel      : %.6g / %.6g\n", pe_z.max_abs, pe_z.max_rel);
    printf("full BA abs/rel     : %.6g / %.6g projection=%s\n", pe_ba.max_abs, pe_ba.max_rel, proj_ok ? "YES" : "NO");
    printf("prestate Q/K/V abs  : %.6g / %.6g / %.6g\n", pre_q.max_abs, pre_k.max_abs, pre_v.max_abs);
    printf("prestate gate abs   : %.6g parity=%s\n", pre_gate.max_abs, pre_ok ? "YES" : "NO");
    printf("conv state abs/rel  : %.6g / %.6g parity=%s\n", conv_state_err.max_abs, conv_state_err.max_rel, conv_ok ? "YES" : "NO");
    printf("recurrent abs/rel   : %.6g / %.6g parity=%s\n", state_err.max_abs, state_err.max_rel, state_ok ? "YES" : "NO");
    printf("core output abs/rel : %.6g / %.6g parity=%s\n", core_err.max_abs, core_err.max_rel, core_ok ? "YES" : "NO");
    printf("gated norm abs/rel  : %.6g / %.6g parity=%s\n", tail_norm_err.max_abs, tail_norm_err.max_rel, norm_ok ? "YES" : "NO");
    printf("FINAL output abs/rel: %.6g / %.6g parity=%s\n", out_err.max_abs, out_err.max_rel, out_ok ? "YES" : "NO");
    printf("COMPLETE DELTANET   : %s\n", all_ok ? "YES" : "NO");
    for (uint32_t i = 0; i < 4u && i < hidden; ++i)
        printf("row %-3u            : gpu=%+.7f cpu=%+.7f delta=%+.3e\n",
            i, gpu_out[i], cpu_out[i], (double)gpu_out[i] - cpu_out[i]);

    free(attn_norm_w); free(qkv_w); free(z_w); free(ba_w); free(conv_w); free(dt_w); free(a_w); free(ssm_norm_w); free(ssm_out_w);
    free(input); free(conv_state); free(recurrent_state);
    free(cpu_norm); free(cpu_qkv); free(cpu_z); free(cpu_ba); free(cpu_beta); free(cpu_gate); free(cpu_conv); free(cpu_q); free(cpu_k); free(cpu_v);
    free(cpu_next_conv); free(cpu_delta); free(cpu_next_state); free(cpu_core); free(cpu_ng); free(cpu_out);
    free(gpu_norm); free(gpu_qkv); free(gpu_z); free(gpu_ba); free(gpu_beta); free(gpu_gate); free(gpu_conv); free(gpu_q); free(gpu_k); free(gpu_v);
    free(gpu_next_conv); free(gpu_delta); free(gpu_next_state); free(gpu_core); free(gpu_ng); free(gpu_out);
    return all_ok ? 0 : 3;

fail:
    free(attn_norm_w); free(qkv_w); free(z_w); free(ba_w); free(conv_w); free(dt_w); free(a_w); free(ssm_norm_w); free(ssm_out_w);
    free(input); free(conv_state); free(recurrent_state);
    free(cpu_norm); free(cpu_qkv); free(cpu_z); free(cpu_ba); free(cpu_beta); free(cpu_gate); free(cpu_conv); free(cpu_q); free(cpu_k); free(cpu_v);
    free(cpu_next_conv); free(cpu_delta); free(cpu_next_state); free(cpu_core); free(cpu_ng); free(cpu_out);
    free(gpu_norm); free(gpu_qkv); free(gpu_z); free(gpu_ba); free(gpu_beta); free(gpu_gate); free(gpu_conv); free(gpu_q); free(gpu_k); free(gpu_v);
    free(gpu_next_conv); free(gpu_delta); free(gpu_next_state); free(gpu_core); free(gpu_ng); free(gpu_out);
    return 1;
}
