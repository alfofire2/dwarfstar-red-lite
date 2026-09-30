#define _FILE_OFFSET_BITS 64
#include "redlite_native_gguf.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GGUF_DEFAULT_ALIGNMENT 32u
#define GGUF_MAX_STRING (1ull << 30)
#define GGUF_MAX_ARRAY  (1ull << 30)

/* GGUF metadata value types. */
enum {
    GGUF_U8 = 0, GGUF_I8, GGUF_U16, GGUF_I16, GGUF_U32, GGUF_I32,
    GGUF_F32, GGUF_BOOL, GGUF_STRING, GGUF_ARRAY, GGUF_U64, GGUF_I64, GGUF_F64
};

typedef struct {
    char *name;
    uint32_t n_dims;
    uint64_t shape[RL_NATIVE_MAX_DIMS];
    uint32_t ggml_type;
    uint64_t relative_offset;
    uint64_t absolute_offset;
    uint64_t span_bytes;
} raw_tensor;

static void set_error(char *dst, size_t cap, const char *msg) {
    if (!dst || !cap) return;
    snprintf(dst, cap, "%s", msg ? msg : "unknown error");
}

static void set_errno_error(char *dst, size_t cap, const char *prefix) {
    if (!dst || !cap) return;
    snprintf(dst, cap, "%s: %s", prefix, strerror(errno));
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
    *v = (uint64_t)b[0] | ((uint64_t)b[1] << 8) | ((uint64_t)b[2] << 16) | ((uint64_t)b[3] << 24) |
         ((uint64_t)b[4] << 32) | ((uint64_t)b[5] << 40) | ((uint64_t)b[6] << 48) | ((uint64_t)b[7] << 56);
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
        if (subtype == GGUF_ARRAY) return 0; /* nested arrays are invalid GGUF, as in llama.cpp */
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

static int tensor_offset_cmp(const void *a, const void *b) {
    const raw_tensor *ta = (const raw_tensor *)a;
    const raw_tensor *tb = (const raw_tensor *)b;
    if (ta->relative_offset < tb->relative_offset) return -1;
    if (ta->relative_offset > tb->relative_offset) return 1;
    return 0;
}

static int parse_expert_name(const char *name, uint32_t *layer, rl_expert_kind *kind) {
    unsigned l = 0;
    char suffix[32] = {0};
    /* Stop the suffix at the underscore before `_exps`. The previous `[^.]`
       scanset consumed `gate_exps`/`up_exps`/`down_exps`, so the exact-name
       reconstruction became `*_exps_exps.weight` and rejected every routed tensor. */
    if (sscanf(name, "blk.%u.ffn_%31[^_]_exps.weight", &l, suffix) != 2) return 0;
    char expected[96];
    snprintf(expected, sizeof(expected), "blk.%u.ffn_%s_exps.weight", l, suffix);
    if (strcmp(name, expected) != 0) return 0;
    if (strcmp(suffix, "gate") == 0) *kind = RL_EXPERT_GATE;
    else if (strcmp(suffix, "up") == 0) *kind = RL_EXPERT_UP;
    else if (strcmp(suffix, "down") == 0) *kind = RL_EXPERT_DOWN;
    else return 0;
    *layer = (uint32_t)l;
    return 1;
}

static uint64_t round_up_u64(uint64_t value, uint64_t alignment) {
    if (!alignment) return value;
    const uint64_t rem = value % alignment;
    if (!rem) return value;
    if (value > UINT64_MAX - (alignment - rem)) return 0;
    return value + alignment - rem;
}

static void free_raw(raw_tensor *raw, uint64_t count) {
    if (!raw) return;
    for (uint64_t i = 0; i < count; ++i) free(raw[i].name);
    free(raw);
}

const char *rl_native_quant_name(uint32_t type) {
    if (type == 17u) return "IQ2_XS";
    if (type == 29u) return "IQ1_M";
    return "OTHER";
}

const char *rl_native_kind_name(rl_expert_kind kind) {
    switch (kind) {
        case RL_EXPERT_GATE: return "gate";
        case RL_EXPERT_UP: return "up";
        case RL_EXPERT_DOWN: return "down";
        default: return "unknown";
    }
}

void rl_native_free_expert_map(rl_expert_map *map) {
    if (!map) return;
    free(map->routed);
    memset(map, 0, sizeof(*map));
}

int rl_native_build_expert_map(
        const char *path,
        uint32_t expected_experts,
        rl_expert_map *out,
        char *error,
        size_t error_cap) {
    if (!path || !out || !expected_experts) {
        set_error(error, error_cap, "invalid native GGUF parser arguments");
        return 0;
    }
    memset(out, 0, sizeof(*out));
    FILE *f = fopen(path, "rb");
    if (!f) {
        set_errno_error(error, error_cap, "open GGUF failed");
        return 0;
    }
    if (fseeko(f, 0, SEEK_END) != 0) {
        set_errno_error(error, error_cap, "seek GGUF failed");
        fclose(f);
        return 0;
    }
    const off_t end = ftello(f);
    if (end <= 0 || fseeko(f, 0, SEEK_SET) != 0) {
        set_error(error, error_cap, "invalid GGUF file size");
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
        set_error(error, error_cap, "invalid or unsupported GGUF header");
        fclose(f);
        return 0;
    }
    /* Each descriptor is at least name length (8) + rank (4) + type (4) + offset (8) bytes on disk,
     * so a count the file cannot hold is corrupt; reject it before sizing any allocation from it. */
    if (tensor_count > SIZE_MAX / sizeof(raw_tensor) || tensor_count > file_size / 24u) {
        set_error(error, error_cap, "GGUF tensor directory is too large");
        fclose(f);
        return 0;
    }

    uint32_t alignment = GGUF_DEFAULT_ALIGNMENT;
    if (!read_metadata(f, kv_count, &alignment)) {
        set_error(error, error_cap, "failed to parse GGUF metadata");
        fclose(f);
        return 0;
    }

    raw_tensor *raw = (raw_tensor *)calloc((size_t)tensor_count, sizeof(raw_tensor));
    if (!raw) {
        set_error(error, error_cap, "out of memory for GGUF tensor directory");
        fclose(f);
        return 0;
    }
    for (uint64_t i = 0; i < tensor_count; ++i) {
        uint32_t n_dims = 0;
        if (!read_string(f, &raw[i].name) || !read_u32(f, &n_dims) || n_dims > RL_NATIVE_MAX_DIMS) {
            set_error(error, error_cap, "failed to parse GGUF tensor name/rank");
            free_raw(raw, tensor_count);
            fclose(f);
            return 0;
        }
        raw[i].n_dims = n_dims;
        for (uint32_t d = 0; d < n_dims; ++d) {
            if (!read_u64(f, &raw[i].shape[d])) {
                set_error(error, error_cap, "failed to parse GGUF tensor shape");
                free_raw(raw, tensor_count);
                fclose(f);
                return 0;
            }
        }
        if (!read_u32(f, &raw[i].ggml_type) || !read_u64(f, &raw[i].relative_offset)) {
            set_error(error, error_cap, "failed to parse GGUF tensor type/offset");
            free_raw(raw, tensor_count);
            fclose(f);
            return 0;
        }
    }
    const off_t directory_end = ftello(f);
    if (directory_end < 0) {
        set_error(error, error_cap, "failed to locate GGUF data section");
        free_raw(raw, tensor_count);
        fclose(f);
        return 0;
    }
    const uint64_t data_base = round_up_u64((uint64_t)directory_end, alignment);
    fclose(f);
    if (!data_base || data_base > file_size) {
        set_error(error, error_cap, "invalid GGUF aligned data offset");
        free_raw(raw, tensor_count);
        return 0;
    }

    /* Every payload must start inside the file; otherwise data_base + offset can wrap and the
     * sorted spans stop describing real bytes (same rule as rl_gguf_model_open). */
    for (uint64_t i = 0; i < tensor_count; ++i) {
        if (raw[i].relative_offset > file_size - data_base) {
            set_error(error, error_cap, "GGUF tensor offset beyond file");
            free_raw(raw, tensor_count);
            return 0;
        }
    }

    qsort(raw, (size_t)tensor_count, sizeof(raw_tensor), tensor_offset_cmp);
    uint32_t routed_count = 0;
    for (uint64_t i = 0; i < tensor_count; ++i) {
        const uint64_t rel = raw[i].relative_offset;
        const uint64_t next = i + 1 < tensor_count ? raw[i + 1].relative_offset : file_size - data_base;
        raw[i].absolute_offset = data_base + rel;
        raw[i].span_bytes = next >= rel ? next - rel : 0;
        uint32_t layer = 0;
        rl_expert_kind kind;
        if (parse_expert_name(raw[i].name, &layer, &kind)) routed_count++;
    }
    rl_expert_tensor *routed = (rl_expert_tensor *)calloc(routed_count ? routed_count : 1u, sizeof(*routed));
    if (!routed) {
        set_error(error, error_cap, "out of memory for routed expert map");
        free_raw(raw, tensor_count);
        return 0;
    }

    uint8_t layer_seen[RL_NATIVE_MAX_LAYERS] = {0};
    uint32_t r = 0;
    uint64_t total_payload = 0;
    int all_safe = routed_count != 0;
    for (uint64_t i = 0; i < tensor_count; ++i) {
        uint32_t layer = 0;
        rl_expert_kind kind;
        if (!parse_expert_name(raw[i].name, &layer, &kind)) continue;
        if (layer >= RL_NATIVE_MAX_LAYERS) {
            set_error(error, error_cap, "routed layer index exceeds native limit");
            free(routed);
            free_raw(raw, tensor_count);
            return 0;
        }
        rl_expert_tensor *e = &routed[r++];
        e->layer = layer;
        e->kind = kind;
        e->ggml_type = raw[i].ggml_type;
        e->n_dims = raw[i].n_dims;
        memcpy(e->shape, raw[i].shape, sizeof(e->shape));
        e->tensor_offset = raw[i].absolute_offset;
        e->tensor_span_bytes = raw[i].span_bytes;

        int axis = -1;
        for (uint32_t d = 0; d < raw[i].n_dims; ++d) {
            if (raw[i].shape[d] == expected_experts) axis = (int)d;
        }
        const uint64_t rem = raw[i].span_bytes % expected_experts;
        const uint64_t tail = rem < alignment ? rem : 0;
        const uint64_t payload = raw[i].span_bytes >= tail ? raw[i].span_bytes - tail : 0;
        const int divisible = payload != 0 && payload % expected_experts == 0;
        e->tail_padding_bytes = tail;
        e->payload_bytes = payload;
        e->expert_stride_bytes = divisible ? payload / expected_experts : 0;
        e->slice_safe = axis == (int)raw[i].n_dims - 1 && divisible;
        if (!e->slice_safe) all_safe = 0;
        total_payload += payload;
        layer_seen[layer] = 1;
    }

    uint32_t layer_count = 0;
    for (uint32_t i = 0; i < RL_NATIVE_MAX_LAYERS; ++i) layer_count += layer_seen[i] ? 1u : 0u;

    uint64_t max_triplet = 0;
    if (all_safe) {
        for (uint32_t layer = 0; layer < RL_NATIVE_MAX_LAYERS; ++layer) {
            if (!layer_seen[layer]) continue;
            uint64_t total = 0;
            unsigned found = 0;
            for (uint32_t i = 0; i < routed_count; ++i) {
                if (routed[i].layer != layer) continue;
                total += routed[i].expert_stride_bytes;
                found++;
            }
            if (found == 3u && total > max_triplet) max_triplet = total;
            if (found != 3u) all_safe = 0;
        }
    }

    out->version = version;
    out->alignment = alignment;
    out->tensor_count = tensor_count;
    out->kv_count = kv_count;
    out->expert_count = expected_experts;
    out->layer_count = layer_count;
    out->routed_tensor_count = routed_count;
    out->total_routed_payload_bytes = total_payload;
    out->max_expert_triplet_bytes = max_triplet;
    out->all_slice_safe = all_safe;
    out->routed = routed;
    free_raw(raw, tensor_count);
    if (error && error_cap) error[0] = '\0';
    return 1;
}

int rl_native_expert_layout(
        const rl_expert_map *map,
        uint32_t layer,
        uint32_t expert,
        rl_expert_layout *out,
        char *error,
        size_t error_cap) {
    if (!map || !out || !map->all_slice_safe || expert >= map->expert_count) {
        set_error(error, error_cap, "invalid native expert layout request");
        return 0;
    }
    memset(out, 0, sizeof(*out));
    out->layer = layer;
    out->expert = expert;
    unsigned found = 0;
    uint32_t common_type = UINT32_MAX;
    for (uint32_t i = 0; i < map->routed_tensor_count; ++i) {
        const rl_expert_tensor *t = &map->routed[i];
        if (t->layer != layer) continue;
        if (common_type == UINT32_MAX) common_type = t->ggml_type;
        else if (common_type != t->ggml_type) {
            set_error(error, error_cap, "mixed gate/up/down quant types are not supported natively yet");
            return 0;
        }
        const uint64_t off = t->tensor_offset + (uint64_t)expert * t->expert_stride_bytes;
        if (t->kind == RL_EXPERT_GATE) { out->gate_offset = off; out->gate_bytes = t->expert_stride_bytes; }
        else if (t->kind == RL_EXPERT_UP) { out->up_offset = off; out->up_bytes = t->expert_stride_bytes; }
        else if (t->kind == RL_EXPERT_DOWN) { out->down_offset = off; out->down_bytes = t->expert_stride_bytes; }
        found++;
    }
    if (found != 3u || !out->gate_bytes || !out->up_bytes || !out->down_bytes) {
        set_error(error, error_cap, "requested routed layer does not contain gate/up/down");
        return 0;
    }
    out->ggml_type = common_type;
    out->total_bytes = out->gate_bytes + out->up_bytes + out->down_bytes;
    if (error && error_cap) error[0] = '\0';
    return 1;
}
