#define _FILE_OFFSET_BITS 64
#define _POSIX_C_SOURCE 200809L
#define _DARWIN_C_SOURCE 1

#include "redlite_native_gguf_dir.h"

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define GGUF_DEFAULT_ALIGNMENT 32u
#define GGUF_MAX_STRING (1ull << 30)
#define GGUF_MAX_ARRAY (1ull << 30)

enum {
    GGUF_U8 = 0, GGUF_I8, GGUF_U16, GGUF_I16, GGUF_U32, GGUF_I32,
    GGUF_F32, GGUF_BOOL, GGUF_STRING, GGUF_ARRAY, GGUF_U64, GGUF_I64, GGUF_F64
};

static void set_error(char *error, size_t cap, const char *msg) {
    if (error && cap) snprintf(error, cap, "%s", msg ? msg : "unknown GGUF error");
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
    *v = 0;
    for (int i = 7; i >= 0; --i) *v = (*v << 8) | b[i];
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
    if (type == GGUF_STRING) {
        uint64_t n = 0;
        return read_u64(f, &n) && n <= GGUF_MAX_STRING && skip_bytes(f, n);
    }
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

/* Read any scalar numeric value as double (bool -> 0/1). */
static int read_scalar_as_double(FILE *f, uint32_t type, double *out) {
    uint8_t b[8];
    const size_t n = scalar_size(type);
    if (!n || !read_exact(f, b, n)) return 0;
    switch (type) {
        case GGUF_U8: *out = b[0]; return 1;
        case GGUF_I8: *out = (int8_t)b[0]; return 1;
        case GGUF_BOOL: *out = b[0] ? 1.0 : 0.0; return 1;
        case GGUF_U16: *out = (uint16_t)(b[0] | (b[1] << 8)); return 1;
        case GGUF_I16: *out = (int16_t)(b[0] | (b[1] << 8)); return 1;
        case GGUF_U32: { uint32_t v = (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24); *out = v; return 1; }
        case GGUF_I32: { uint32_t v = (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24); *out = (int32_t)v; return 1; }
        case GGUF_F32: { uint32_t v = (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24); float fv; memcpy(&fv, &v, 4); *out = fv; return 1; }
        case GGUF_U64: case GGUF_I64: { uint64_t v = 0; for (int i = 7; i >= 0; --i) v = (v << 8) | b[i]; *out = type == GGUF_U64 ? (double)v : (double)(int64_t)v; return 1; }
        case GGUF_F64: { uint64_t v = 0; for (int i = 7; i >= 0; --i) v = (v << 8) | b[i]; double dv; memcpy(&dv, &v, 8); *out = dv; return 1; }
        default: return 0;
    }
}

typedef struct {
    const char *key;
    uint32_t *u32;
    float *f32;
    int *flag;
    int32_t *i32;
} scalar_binding;

static int bind_scalar(FILE *f, uint32_t type, const scalar_binding *b) {
    double v = 0.0;
    if (!read_scalar_as_double(f, type, &v)) return 0;
    if (b->u32) { if (v < 0 || v > 4294967295.0) return 0; *b->u32 = (uint32_t)v; }
    if (b->i32) { if (v < -2147483648.0 || v > 2147483647.0) return 0; *b->i32 = (int32_t)v; }
    if (b->f32) { if (!isfinite(v)) return 0; *b->f32 = (float)v; }
    if (b->flag) *b->flag = (v != 0.0);
    return 1;
}

static int read_string_array(FILE *f, char ***out, uint32_t *count_out) {
    uint32_t subtype = 0;
    uint64_t count = 0;
    if (!read_u32(f, &subtype) || !read_u64(f, &count) || subtype != GGUF_STRING || count > 0x7fffffffu) return 0;
    char **arr = (char **)calloc((size_t)count + 1u, sizeof(char *));
    if (!arr) return 0;
    for (uint64_t i = 0; i < count; ++i) {
        if (!read_string(f, &arr[i])) {
            for (uint64_t j = 0; j < i; ++j) free(arr[j]);
            free(arr);
            return 0;
        }
    }
    *out = arr;
    *count_out = (uint32_t)count;
    return 1;
}

static int read_i32_array(FILE *f, int32_t **out, uint32_t *count_out) {
    uint32_t subtype = 0;
    uint64_t count = 0;
    if (!read_u32(f, &subtype) || !read_u64(f, &count) || count > 0x7fffffffu) return 0;
    const size_t sf = scalar_size(subtype);
    if (!sf) return 0;
    int32_t *arr = (int32_t *)calloc((size_t)count + 1u, sizeof(int32_t));
    if (!arr) return 0;
    for (uint64_t i = 0; i < count; ++i) {
        double v = 0.0;
        if (!read_scalar_as_double(f, subtype, &v)) { free(arr); return 0; }
        arr[i] = (int32_t)v;
    }
    *out = arr;
    *count_out = (uint32_t)count;
    return 1;
}

static void free_string_array(char **arr, uint32_t count) {
    if (!arr) return;
    for (uint32_t i = 0; i < count; ++i) free(arr[i]);
    free(arr);
}

static int read_metadata(FILE *f, rl_gguf_model *m, char *error, size_t cap) {
    for (uint64_t i = 0; i < m->kv_count; ++i) {
        char *key = NULL;
        uint32_t type = 0;
        if (!read_string(f, &key) || !read_u32(f, &type)) { free(key); set_error(error, cap, "GGUF metadata key parse failed"); return 0; }

        int handled = 1;
        const scalar_binding bindings[] = {
            {"general.alignment", &m->alignment, NULL, NULL, NULL},
            {"qwen3next.block_count", &m->n_layer, NULL, NULL, NULL},
            {"qwen3next.embedding_length", &m->n_embd, NULL, NULL, NULL},
            {"qwen3next.feed_forward_length", &m->n_ff, NULL, NULL, NULL},
            {"qwen3next.attention.head_count", &m->n_head, NULL, NULL, NULL},
            {"qwen3next.attention.head_count_kv", &m->n_head_kv, NULL, NULL, NULL},
            {"qwen3next.attention.key_length", &m->key_length, NULL, NULL, NULL},
            {"qwen3next.attention.value_length", &m->value_length, NULL, NULL, NULL},
            {"qwen3next.rope.dimension_count", &m->rope_dims, NULL, NULL, NULL},
            {"qwen3next.expert_count", &m->n_expert, NULL, NULL, NULL},
            {"qwen3next.expert_used_count", &m->n_expert_used, NULL, NULL, NULL},
            {"qwen3next.expert_feed_forward_length", &m->n_ff_exp, NULL, NULL, NULL},
            {"qwen3next.expert_shared_feed_forward_length", &m->n_ff_shexp, NULL, NULL, NULL},
            {"qwen3next.ssm.conv_kernel", &m->ssm_conv, NULL, NULL, NULL},
            {"qwen3next.ssm.state_size", &m->ssm_state, NULL, NULL, NULL},
            {"qwen3next.ssm.group_count", &m->ssm_group, NULL, NULL, NULL},
            {"qwen3next.ssm.time_step_rank", &m->ssm_dt_rank, NULL, NULL, NULL},
            {"qwen3next.ssm.inner_size", &m->ssm_inner, NULL, NULL, NULL},
            {"qwen3next.context_length", &m->context_length, NULL, NULL, NULL},
            {"qwen3next.full_attention_interval", &m->full_attention_interval, NULL, NULL, NULL},
            {"qwen3next.attention.layer_norm_rms_epsilon", NULL, &m->rms_eps, NULL, NULL},
            {"qwen3next.rope.freq_base", NULL, &m->rope_freq_base, NULL, NULL},
            {"tokenizer.ggml.eos_token_id", NULL, NULL, NULL, &m->eos_id},
            {"tokenizer.ggml.bos_token_id", NULL, NULL, NULL, &m->bos_id},
            {"tokenizer.ggml.padding_token_id", NULL, NULL, NULL, &m->pad_id},
            {"tokenizer.ggml.add_bos_token", NULL, NULL, &m->add_bos, NULL},
            {"tokenizer.ggml.add_eos_token", NULL, NULL, &m->add_eos, NULL},
        };
        int bound = 0;
        for (size_t b = 0; b < sizeof(bindings) / sizeof(bindings[0]); ++b) {
            if (strcmp(key, bindings[b].key) == 0) {
                if (type == GGUF_ARRAY || type == GGUF_STRING || !bind_scalar(f, type, &bindings[b])) {
                    free(key); set_error(error, cap, "GGUF scalar metadata has an unexpected type"); return 0;
                }
                bound = 1;
                break;
            }
        }
        if (bound) { free(key); continue; }

        if (strcmp(key, "general.architecture") == 0 && type == GGUF_STRING) {
            if (!read_string(f, &m->architecture)) handled = 0;
        } else if (strcmp(key, "tokenizer.ggml.model") == 0 && type == GGUF_STRING) {
            if (!read_string(f, &m->tokenizer_model)) handled = 0;
        } else if (strcmp(key, "tokenizer.ggml.pre") == 0 && type == GGUF_STRING) {
            if (!read_string(f, &m->tokenizer_pre)) handled = 0;
        } else if (strcmp(key, "tokenizer.chat_template") == 0 && type == GGUF_STRING) {
            if (!read_string(f, &m->chat_template)) handled = 0;
        } else if (strcmp(key, "tokenizer.ggml.tokens") == 0 && type == GGUF_ARRAY) {
            if (!read_string_array(f, &m->tokens, &m->vocab_count)) handled = 0;
        } else if (strcmp(key, "tokenizer.ggml.merges") == 0 && type == GGUF_ARRAY) {
            if (!read_string_array(f, &m->merges, &m->merge_count)) handled = 0;
        } else if (strcmp(key, "tokenizer.ggml.token_type") == 0 && type == GGUF_ARRAY) {
            uint32_t n = 0;
            if (!read_i32_array(f, &m->token_types, &n)) handled = 0;
        } else if (!skip_value(f, type)) {
            handled = 0;
        }
        free(key);
        if (!handled) { set_error(error, cap, "GGUF metadata value parse failed"); return 0; }
    }
    return 1;
}

static int tensor_cmp(const void *a, const void *b) {
    const rl_gguf_tensor *ta = (const rl_gguf_tensor *)a;
    const rl_gguf_tensor *tb = (const rl_gguf_tensor *)b;
    if (ta->offset < tb->offset) return -1;
    if (ta->offset > tb->offset) return 1;
    return 0;
}

size_t rl_gguf_row_bytes(uint32_t ggml_type, uint64_t ncols) {
    switch (ggml_type) {
        case 0: return (size_t)ncols * 4u;                                  /* F32 */
        case 1: return (size_t)ncols * 2u;                                  /* F16 */
        case 8: return ncols % 32u ? 0 : (size_t)(ncols / 32u) * 34u;       /* Q8_0 */
        case 10: return ncols % 256u ? 0 : (size_t)(ncols / 256u) * 84u;    /* Q2_K */
        case 12: return ncols % 256u ? 0 : (size_t)(ncols / 256u) * 144u;   /* Q4_K */
        case 13: return ncols % 256u ? 0 : (size_t)(ncols / 256u) * 176u;   /* Q5_K */
        case 14: return ncols % 256u ? 0 : (size_t)(ncols / 256u) * 210u;   /* Q6_K */
        case 16: return ncols % 256u ? 0 : (size_t)(ncols / 256u) * 66u;    /* IQ2_XXS */
        case 17: return ncols % 256u ? 0 : (size_t)(ncols / 256u) * 74u;    /* IQ2_XS */
        case 29: return ncols % 256u ? 0 : (size_t)(ncols / 256u) * 56u;    /* IQ1_M */
        case 30: return ncols % 256u ? 0 : (size_t)(ncols / 256u) * 2u;     /* BF16 (unused) */
        default: return 0;
    }
}

const char *rl_gguf_type_name(uint32_t ggml_type) {
    switch (ggml_type) {
        case 0: return "F32"; case 1: return "F16"; case 8: return "Q8_0"; case 10: return "Q2_K";
        case 12: return "Q4_K"; case 13: return "Q5_K"; case 14: return "Q6_K"; case 16: return "IQ2_XXS";
        case 17: return "IQ2_XS"; case 29: return "IQ1_M"; case 30: return "BF16";
        default: return "UNKNOWN";
    }
}

uint64_t rl_gguf_payload_bytes(uint32_t ggml_type, uint32_t n_dims, const uint64_t *shape) {
    if (!n_dims) return 0;
    const size_t rb = rl_gguf_row_bytes(ggml_type, shape[0]);
    if (!rb) return 0;
    uint64_t rows = 1;
    for (uint32_t d = 1; d < n_dims; ++d) {
        if (shape[d] && rows > UINT64_MAX / shape[d]) return 0;
        rows *= shape[d];
    }
    if (rows > UINT64_MAX / rb) return 0;
    return rows * (uint64_t)rb;
}

int rl_gguf_model_open(const char *path, rl_gguf_model *m, char *error, size_t cap) {
    if (!path || !m) { set_error(error, cap, "invalid GGUF open arguments"); return 0; }
    memset(m, 0, sizeof(*m));
    m->fd = -1;
    m->alignment = GGUF_DEFAULT_ALIGNMENT;
    m->eos_id = m->bos_id = m->pad_id = -1;
    m->path = strdup(path);
    FILE *f = fopen(path, "rb");
    if (!f) { snprintf(error, cap, "open %s failed: %s", path, strerror(errno)); rl_gguf_model_close(m); return 0; }
    if (fseeko(f, 0, SEEK_END) != 0) { fclose(f); set_error(error, cap, "seek failed"); rl_gguf_model_close(m); return 0; }
    const off_t end = ftello(f);
    if (end <= 0 || fseeko(f, 0, SEEK_SET) != 0) { fclose(f); set_error(error, cap, "invalid GGUF size"); rl_gguf_model_close(m); return 0; }
    m->file_size = (uint64_t)end;

    uint8_t magic[4];
    if (!read_exact(f, magic, 4) || memcmp(magic, "GGUF", 4) != 0 || !read_u32(f, &m->version) ||
        (m->version != 2u && m->version != 3u) || !read_u64(f, &m->tensor_count) || !read_u64(f, &m->kv_count) ||
        m->tensor_count > (1ull << 24) || m->kv_count > (1ull << 24)) {
        fclose(f); set_error(error, cap, "invalid GGUF header"); rl_gguf_model_close(m); return 0;
    }
    if (!read_metadata(f, m, error, cap)) { fclose(f); rl_gguf_model_close(m); return 0; }

    m->tensors = (rl_gguf_tensor *)calloc((size_t)m->tensor_count, sizeof(*m->tensors));
    if (!m->tensors) { fclose(f); set_error(error, cap, "tensor directory allocation failed"); rl_gguf_model_close(m); return 0; }
    for (uint64_t i = 0; i < m->tensor_count; ++i) {
        rl_gguf_tensor *t = &m->tensors[i];
        uint32_t dims = 0;
        if (!read_string(f, &t->name) || !read_u32(f, &dims) || dims == 0 || dims > RL_GGUF_MAX_DIMS) {
            fclose(f); set_error(error, cap, "tensor descriptor parse failed"); rl_gguf_model_close(m); return 0;
        }
        t->n_dims = dims;
        for (uint32_t d = 0; d < dims; ++d) if (!read_u64(f, &t->shape[d])) {
            fclose(f); set_error(error, cap, "tensor shape parse failed"); rl_gguf_model_close(m); return 0;
        }
        uint64_t rel = 0;
        if (!read_u32(f, &t->ggml_type) || !read_u64(f, &rel)) {
            fclose(f); set_error(error, cap, "tensor type/offset parse failed"); rl_gguf_model_close(m); return 0;
        }
        t->offset = rel; /* relative for now */
        t->payload_bytes = rl_gguf_payload_bytes(t->ggml_type, t->n_dims, t->shape);
    }
    const off_t directory_end = ftello(f);
    fclose(f);
    if (directory_end < 0 || !m->alignment) { set_error(error, cap, "invalid GGUF directory end"); rl_gguf_model_close(m); return 0; }
    uint64_t base = (uint64_t)directory_end;
    const uint64_t r = base % m->alignment;
    if (r) base += m->alignment - r;
    if (base > m->file_size) { set_error(error, cap, "invalid GGUF data base"); rl_gguf_model_close(m); return 0; }
    m->data_base = base;

    for (uint64_t i = 0; i < m->tensor_count; ++i) {
        if (m->tensors[i].offset > m->file_size - base) { set_error(error, cap, "tensor offset beyond file"); rl_gguf_model_close(m); return 0; }
        m->tensors[i].offset += base;
    }
    qsort(m->tensors, (size_t)m->tensor_count, sizeof(*m->tensors), tensor_cmp);
    for (uint64_t i = 0; i < m->tensor_count; ++i) {
        const uint64_t next = i + 1 < m->tensor_count ? m->tensors[i + 1].offset : m->file_size;
        m->tensors[i].span_bytes = next >= m->tensors[i].offset ? next - m->tensors[i].offset : 0;
        if (m->tensors[i].payload_bytes && m->tensors[i].payload_bytes > m->tensors[i].span_bytes) {
            snprintf(error, cap, "tensor %s payload exceeds its physical span", m->tensors[i].name);
            rl_gguf_model_close(m);
            return 0;
        }
    }
    if (error && cap) error[0] = '\0';
    return 1;
}

int rl_gguf_model_map(rl_gguf_model *m, char *error, size_t cap) {
    if (!m || !m->path) { set_error(error, cap, "invalid GGUF map arguments"); return 0; }
    if (m->map) return 1;
    const int fd = open(m->path, O_RDONLY);
    if (fd < 0) { snprintf(error, cap, "open %s for mmap failed: %s", m->path, strerror(errno)); return 0; }
    struct stat st;
    if (fstat(fd, &st) != 0 || (uint64_t)st.st_size != m->file_size) {
        close(fd); set_error(error, cap, "GGUF size changed since the directory was parsed"); return 0;
    }
    void *p = mmap(NULL, (size_t)m->file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (p == MAP_FAILED) { close(fd); snprintf(error, cap, "mmap failed: %s", strerror(errno)); return 0; }
    m->fd = fd;
    m->map = (uint8_t *)p;
    m->map_len = (size_t)m->file_size;
    if (error && cap) error[0] = '\0';
    return 1;
}

void rl_gguf_model_close(rl_gguf_model *m) {
    if (!m) return;
    if (m->map) munmap(m->map, m->map_len);
    if (m->fd >= 0) close(m->fd);
    if (m->tensors) {
        for (uint64_t i = 0; i < m->tensor_count; ++i) free(m->tensors[i].name);
        free(m->tensors);
    }
    free_string_array(m->tokens, m->vocab_count);
    free_string_array(m->merges, m->merge_count);
    free(m->token_types);
    free(m->path);
    free(m->architecture);
    free(m->tokenizer_model);
    free(m->tokenizer_pre);
    free(m->chat_template);
    memset(m, 0, sizeof(*m));
    m->fd = -1;
}

const rl_gguf_tensor *rl_gguf_find(const rl_gguf_model *m, const char *name) {
    if (!m || !name) return NULL;
    for (uint64_t i = 0; i < m->tensor_count; ++i)
        if (strcmp(m->tensors[i].name, name) == 0) return &m->tensors[i];
    return NULL;
}

const rl_gguf_tensor *rl_gguf_find_layer(const rl_gguf_model *m, uint32_t layer, const char *suffix) {
    char name[128];
    snprintf(name, sizeof(name), "blk.%u.%s", layer, suffix);
    return rl_gguf_find(m, name);
}

const uint8_t *rl_gguf_tensor_data(const rl_gguf_model *m, const rl_gguf_tensor *t) {
    if (!m || !m->map || !t || t->offset + t->span_bytes > m->map_len) return NULL;
    return m->map + t->offset;
}
