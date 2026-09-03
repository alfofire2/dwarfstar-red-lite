#define _FILE_OFFSET_BITS 64
#define _POSIX_C_SOURCE 200809L

#include "redlite_native_shared.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GGUF_DEFAULT_ALIGNMENT 32u
#define GGUF_MAX_STRING (1ull << 30)
#define GGUF_MAX_ARRAY  (1ull << 30)

enum {
    GGUF_U8 = 0, GGUF_I8, GGUF_U16, GGUF_I16, GGUF_U32, GGUF_I32,
    GGUF_F32, GGUF_BOOL, GGUF_STRING, GGUF_ARRAY, GGUF_U64, GGUF_I64, GGUF_F64
};

typedef struct {
    char *name;
    uint32_t n_dims;
    uint64_t shape[RL_SHARED_MAX_DIMS];
    uint32_t ggml_type;
    uint64_t relative_offset;
    uint64_t absolute_offset;
    uint64_t span_bytes;
} shared_raw_tensor;

static void set_error(char *dst, size_t cap, const char *msg) {
    if (dst && cap) snprintf(dst, cap, "%s", msg ? msg : "unknown shared-expert audit error");
}

static void set_errno_error(char *dst, size_t cap, const char *prefix) {
    if (dst && cap) snprintf(dst, cap, "%s: %s", prefix, strerror(errno));
}

static int read_exact(FILE *f, void *dst, size_t n) {
    return n == 0 || fread(dst, 1, n, f) == n;
}

static int read_u32(FILE *f, uint32_t *v) {
    unsigned char b[4];
    if (!read_exact(f, b, sizeof(b))) return 0;
    *v = (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
    return 1;
}

static int read_u64(FILE *f, uint64_t *v) {
    unsigned char b[8];
    if (!read_exact(f, b, sizeof(b))) return 0;
    *v = (uint64_t)b[0] | ((uint64_t)b[1] << 8) | ((uint64_t)b[2] << 16) |
         ((uint64_t)b[3] << 24) | ((uint64_t)b[4] << 32) | ((uint64_t)b[5] << 40) |
         ((uint64_t)b[6] << 48) | ((uint64_t)b[7] << 56);
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
    if (!read_u64(f, &n) || n > GGUF_MAX_STRING || n > SIZE_MAX - 1) return 0;
    char *s = (char *)malloc((size_t)n + 1);
    if (!s) return 0;
    if (!read_exact(f, s, (size_t)n)) {
        free(s);
        return 0;
    }
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
        const size_t sub_fixed = scalar_size(subtype);
        if (sub_fixed) {
            if (count && count > UINT64_MAX / sub_fixed) return 0;
            return skip_bytes(f, count * sub_fixed);
        }
        for (uint64_t i = 0; i < count; ++i) {
            if (!skip_value(f, subtype)) return 0;
        }
        return 1;
    }
    return 0;
}

static int read_metadata(FILE *f, uint64_t count, uint32_t *alignment) {
    for (uint64_t i = 0; i < count; ++i) {
        char *key = NULL;
        uint32_t type = 0;
        if (!read_string(f, &key) || !read_u32(f, &type)) {
            free(key);
            return 0;
        }
        if (strcmp(key, "general.alignment") == 0 && type == GGUF_U32) {
            uint32_t value = 0;
            free(key);
            if (!read_u32(f, &value) || value == 0) return 0;
            *alignment = value;
        } else {
            free(key);
            if (!skip_value(f, type)) return 0;
        }
    }
    return 1;
}

static uint64_t round_up_u64(uint64_t value, uint64_t alignment) {
    if (!alignment) return value;
    const uint64_t rem = value % alignment;
    if (!rem) return value;
    if (value > UINT64_MAX - (alignment - rem)) return 0;
    return value + alignment - rem;
}

static int tensor_offset_cmp(const void *a, const void *b) {
    const shared_raw_tensor *ta = (const shared_raw_tensor *)a;
    const shared_raw_tensor *tb = (const shared_raw_tensor *)b;
    if (ta->relative_offset < tb->relative_offset) return -1;
    if (ta->relative_offset > tb->relative_offset) return 1;
    return 0;
}

static void free_raw(shared_raw_tensor *raw, uint64_t count) {
    if (!raw) return;
    for (uint64_t i = 0; i < count; ++i) free(raw[i].name);
    free(raw);
}

static int exact_name(const char *name, const char *fmt, unsigned *layer) {
    unsigned l = 0;
    int consumed = 0;
    if (sscanf(name, fmt, &l, &consumed) != 1) return 0;
    if (!consumed || name[consumed] != '\0' || l >= RL_SHARED_MAX_LAYERS) return 0;
    *layer = l;
    return 1;
}

static int parse_shared_name(const char *name, uint32_t *layer, rl_shared_kind *kind) {
    unsigned l = 0;
    if (exact_name(name, "blk.%u.ffn_gate_inp_shexp.weight%n", &l)) {
        *layer = (uint32_t)l; *kind = RL_SHARED_GATE_INPUT; return 1;
    }
    if (exact_name(name, "blk.%u.ffn_gate_shexp.weight%n", &l)) {
        *layer = (uint32_t)l; *kind = RL_SHARED_GATE; return 1;
    }
    if (exact_name(name, "blk.%u.ffn_up_shexp.weight%n", &l)) {
        *layer = (uint32_t)l; *kind = RL_SHARED_UP; return 1;
    }
    if (exact_name(name, "blk.%u.ffn_down_shexp.weight%n", &l)) {
        *layer = (uint32_t)l; *kind = RL_SHARED_DOWN; return 1;
    }
    return 0;
}

const char *rl_native_shared_kind_name(rl_shared_kind kind) {
    switch (kind) {
        case RL_SHARED_GATE_INPUT: return "gate_input";
        case RL_SHARED_GATE: return "gate";
        case RL_SHARED_UP: return "up";
        case RL_SHARED_DOWN: return "down";
        default: return "unknown";
    }
}

int rl_native_shared_audit(
        const char *model_path,
        rl_shared_tensor_info *out,
        uint32_t out_capacity,
        uint32_t *out_count,
        char *error,
        size_t error_cap) {
    if (!model_path || !out_count || (out_capacity && !out)) {
        set_error(error, error_cap, "invalid shared-expert audit arguments");
        return 0;
    }
    *out_count = 0;

    FILE *f = fopen(model_path, "rb");
    if (!f) {
        set_errno_error(error, error_cap, "open GGUF for shared-expert audit failed");
        return 0;
    }
    if (fseeko(f, 0, SEEK_END) != 0) {
        set_errno_error(error, error_cap, "seek GGUF for shared-expert audit failed");
        fclose(f);
        return 0;
    }
    const off_t end = ftello(f);
    if (end <= 0 || fseeko(f, 0, SEEK_SET) != 0) {
        set_error(error, error_cap, "invalid GGUF size for shared-expert audit");
        fclose(f);
        return 0;
    }
    const uint64_t file_size = (uint64_t)end;

    unsigned char magic[4];
    uint32_t version = 0;
    uint64_t tensor_count = 0, kv_count = 0;
    if (!read_exact(f, magic, 4) || memcmp(magic, "GGUF", 4) != 0 ||
        !read_u32(f, &version) || (version != 2u && version != 3u) ||
        !read_u64(f, &tensor_count) || !read_u64(f, &kv_count)) {
        set_error(error, error_cap, "invalid GGUF header for shared-expert audit");
        fclose(f);
        return 0;
    }
    if (tensor_count > SIZE_MAX / sizeof(shared_raw_tensor)) {
        set_error(error, error_cap, "GGUF tensor directory too large for shared-expert audit");
        fclose(f);
        return 0;
    }

    uint32_t alignment = GGUF_DEFAULT_ALIGNMENT;
    if (!read_metadata(f, kv_count, &alignment)) {
        set_error(error, error_cap, "failed to parse GGUF metadata for shared-expert audit");
        fclose(f);
        return 0;
    }

    shared_raw_tensor *raw = (shared_raw_tensor *)calloc((size_t)tensor_count, sizeof(*raw));
    if (!raw) {
        set_error(error, error_cap, "out of memory for shared-expert tensor directory");
        fclose(f);
        return 0;
    }
    for (uint64_t i = 0; i < tensor_count; ++i) {
        uint32_t n_dims = 0;
        if (!read_string(f, &raw[i].name) || !read_u32(f, &n_dims) || n_dims > RL_SHARED_MAX_DIMS) {
            set_error(error, error_cap, "failed to parse shared-expert tensor descriptor");
            free_raw(raw, tensor_count); fclose(f); return 0;
        }
        raw[i].n_dims = n_dims;
        for (uint32_t d = 0; d < n_dims; ++d) {
            if (!read_u64(f, &raw[i].shape[d])) {
                set_error(error, error_cap, "failed to parse shared-expert tensor shape");
                free_raw(raw, tensor_count); fclose(f); return 0;
            }
        }
        if (!read_u32(f, &raw[i].ggml_type) || !read_u64(f, &raw[i].relative_offset)) {
            set_error(error, error_cap, "failed to parse shared-expert tensor type/offset");
            free_raw(raw, tensor_count); fclose(f); return 0;
        }
    }
    const off_t directory_end = ftello(f);
    fclose(f);
    if (directory_end < 0) {
        set_error(error, error_cap, "failed to locate GGUF data section for shared-expert audit");
        free_raw(raw, tensor_count); return 0;
    }
    const uint64_t data_base = round_up_u64((uint64_t)directory_end, alignment);
    if (!data_base || data_base > file_size) {
        set_error(error, error_cap, "invalid GGUF data base for shared-expert audit");
        free_raw(raw, tensor_count); return 0;
    }

    qsort(raw, (size_t)tensor_count, sizeof(*raw), tensor_offset_cmp);
    uint8_t seen[RL_SHARED_MAX_LAYERS][4] = {{0}};
    uint32_t count = 0;
    for (uint64_t i = 0; i < tensor_count; ++i) {
        const uint64_t rel = raw[i].relative_offset;
        const uint64_t next = i + 1 < tensor_count ? raw[i + 1].relative_offset : file_size - data_base;
        raw[i].absolute_offset = data_base + rel;
        raw[i].span_bytes = next >= rel ? next - rel : 0;

        uint32_t layer = 0;
        rl_shared_kind kind;
        if (!parse_shared_name(raw[i].name, &layer, &kind)) continue;
        if (seen[layer][(unsigned)kind]) {
            set_error(error, error_cap, "duplicate shared-expert tensor for layer/kind");
            free_raw(raw, tensor_count); return 0;
        }
        seen[layer][(unsigned)kind] = 1;
        if (count >= out_capacity) {
            set_error(error, error_cap, "shared-expert audit output capacity is too small");
            free_raw(raw, tensor_count); return 0;
        }
        rl_shared_tensor_info *info = &out[count++];
        memset(info, 0, sizeof(*info));
        info->layer = layer;
        info->kind = kind;
        info->ggml_type = raw[i].ggml_type;
        info->n_dims = raw[i].n_dims;
        memcpy(info->shape, raw[i].shape, sizeof(info->shape));
        info->tensor_offset = raw[i].absolute_offset;
        info->tensor_span_bytes = raw[i].span_bytes;
    }

    free_raw(raw, tensor_count);
    *out_count = count;
    if (error && error_cap) error[0] = '\0';
    return 1;
}
