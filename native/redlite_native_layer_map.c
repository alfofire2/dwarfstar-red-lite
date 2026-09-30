#define _FILE_OFFSET_BITS 64
#define _POSIX_C_SOURCE 200809L

#include "redlite_native_layer_map.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LM_MAX_STRING (1ull << 30)
#define LM_MAX_ARRAY (1ull << 30)

enum {
    LM_U8 = 0, LM_I8, LM_U16, LM_I16, LM_U32, LM_I32,
    LM_F32, LM_BOOL, LM_STRING, LM_ARRAY, LM_U64, LM_I64, LM_F64
};

typedef struct {
    uint8_t seen;
    uint8_t attn_norm;
    uint8_t post_norm;
    uint8_t recurrent;
    uint8_t full_attention;
} lm_summary;

static void lm_error(char *error, size_t cap, const char *message) {
    if (error && cap) snprintf(error, cap, "%s", message);
}

static int lm_read(FILE *f, void *dst, size_t n) {
    return n == 0 || fread(dst, 1, n, f) == n;
}

static int lm_u32(FILE *f, uint32_t *value) {
    uint8_t b[4];
    if (!lm_read(f, b, sizeof(b))) return 0;
    *value = (uint32_t)b[0] | ((uint32_t)b[1] << 8) |
        ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
    return 1;
}

static int lm_u64(FILE *f, uint64_t *value) {
    uint8_t b[8];
    if (!lm_read(f, b, sizeof(b))) return 0;
    *value = (uint64_t)b[0] | ((uint64_t)b[1] << 8) |
        ((uint64_t)b[2] << 16) | ((uint64_t)b[3] << 24) |
        ((uint64_t)b[4] << 32) | ((uint64_t)b[5] << 40) |
        ((uint64_t)b[6] << 48) | ((uint64_t)b[7] << 56);
    return 1;
}

static int lm_skip(FILE *f, uint64_t n) {
    while (n) {
        const uint64_t chunk = n > 0x3fffffffULL ? 0x3fffffffULL : n;
        if (fseeko(f, (off_t)chunk, SEEK_CUR) != 0) return 0;
        n -= chunk;
    }
    return 1;
}

static int lm_string(FILE *f, char **out) {
    uint64_t n = 0;
    if (!lm_u64(f, &n) || n > LM_MAX_STRING || n > SIZE_MAX - 1u) return 0;
    char *value = (char *)malloc((size_t)n + 1u);
    if (!value || !lm_read(f, value, (size_t)n)) { free(value); return 0; }
    value[n] = '\0';
    *out = value;
    return 1;
}

static size_t lm_scalar_size(uint32_t type) {
    switch (type) {
        case LM_U8: case LM_I8: case LM_BOOL: return 1u;
        case LM_U16: case LM_I16: return 2u;
        case LM_U32: case LM_I32: case LM_F32: return 4u;
        case LM_U64: case LM_I64: case LM_F64: return 8u;
        default: return 0u;
    }
}

static int lm_skip_value(FILE *f, uint32_t type) {
    const size_t scalar = lm_scalar_size(type);
    if (scalar) return lm_skip(f, scalar);
    if (type == LM_STRING) {
        uint64_t n = 0;
        return lm_u64(f, &n) && n <= LM_MAX_STRING && lm_skip(f, n);
    }
    if (type != LM_ARRAY) return 0;
    uint32_t subtype = 0;
    uint64_t count = 0;
    if (!lm_u32(f, &subtype) || !lm_u64(f, &count) || count > LM_MAX_ARRAY) return 0;
    const size_t subscalar = lm_scalar_size(subtype);
    if (subscalar) {
        if (count && count > UINT64_MAX / subscalar) return 0;
        return lm_skip(f, count * subscalar);
    }
    for (uint64_t i = 0; i < count; ++i) if (!lm_skip_value(f, subtype)) return 0;
    return 1;
}

static int lm_parse_layer(const char *name, uint32_t *layer, const char **suffix) {
    if (!name || strncmp(name, "blk.", 4) != 0) return 0;
    errno = 0;
    char *end = NULL;
    const unsigned long value = strtoul(name + 4, &end, 10);
    if (errno || end == name + 4 || !end || *end != '.' || value >= RL_LAYER_MAP_MAX_LAYERS) return 0;
    *layer = (uint32_t)value;
    *suffix = end + 1;
    return 1;
}

static int lm_recurrent_signal(const char *suffix) {
    return strncmp(suffix, "ssm_", 4) == 0 ||
        strcmp(suffix, "attn_qkv.weight") == 0 || strcmp(suffix, "attn_gate.weight") == 0;
}

static int lm_full_attention_signal(const char *suffix) {
    return strcmp(suffix, "attn_q.weight") == 0 || strcmp(suffix, "attn_k.weight") == 0 ||
        strcmp(suffix, "attn_v.weight") == 0 || strcmp(suffix, "attn_output.weight") == 0 ||
        strcmp(suffix, "attn_q_norm.weight") == 0 || strcmp(suffix, "attn_k_norm.weight") == 0;
}

const char *rl_native_layer_map_kind_name(rl_layer_map_kind kind) {
    switch (kind) {
        case RL_LAYER_MAP_RECURRENT: return "RECURRENT";
        case RL_LAYER_MAP_FULL_ATTENTION: return "FULL_ATTENTION";
        case RL_LAYER_MAP_MIXED: return "MIXED";
        default: return "UNKNOWN";
    }
}

int rl_native_layer_map_audit(const char *path, rl_native_layer_map *out,
        char *error, size_t error_cap) {
    if (!path || !out) { lm_error(error, error_cap, "invalid layer-map arguments"); return 0; }
    memset(out, 0, sizeof(*out));
    FILE *f = fopen(path, "rb");
    if (!f) { if (error && error_cap) snprintf(error, error_cap, "open failed: %s", strerror(errno)); return 0; }
    uint8_t magic[4];
    uint32_t version = 0;
    uint64_t tensor_count = 0, metadata_count = 0;
    if (!lm_read(f, magic, sizeof(magic)) || memcmp(magic, "GGUF", 4) != 0 ||
        !lm_u32(f, &version) || (version != 2u && version != 3u) ||
        !lm_u64(f, &tensor_count) || !lm_u64(f, &metadata_count)) {
        fclose(f); lm_error(error, error_cap, "invalid GGUF header for layer map"); return 0;
    }
    for (uint64_t i = 0; i < metadata_count; ++i) {
        char *key = NULL;
        uint32_t type = 0;
        if (!lm_string(f, &key) || !lm_u32(f, &type)) {
            free(key); fclose(f); lm_error(error, error_cap, "layer-map metadata parse failed"); return 0;
        }
        free(key);
        if (!lm_skip_value(f, type)) {
            fclose(f); lm_error(error, error_cap, "layer-map metadata value parse failed"); return 0;
        }
    }
    lm_summary layers[RL_LAYER_MAP_MAX_LAYERS];
    memset(layers, 0, sizeof(layers));
    for (uint64_t i = 0; i < tensor_count; ++i) {
        char *name = NULL;
        uint32_t dims = 0, type = 0;
        uint64_t ignored = 0;
        if (!lm_string(f, &name) || !lm_u32(f, &dims) || dims > 8u) {
            free(name); fclose(f); lm_error(error, error_cap, "layer-map tensor parse failed"); return 0;
        }
        for (uint32_t d = 0; d < dims; ++d) if (!lm_u64(f, &ignored)) {
            free(name); fclose(f); lm_error(error, error_cap, "layer-map shape parse failed"); return 0;
        }
        if (!lm_u32(f, &type) || !lm_u64(f, &ignored)) {
            free(name); fclose(f); lm_error(error, error_cap, "layer-map descriptor parse failed"); return 0;
        }
        (void)type;
        uint32_t layer = 0;
        const char *suffix = NULL;
        if (lm_parse_layer(name, &layer, &suffix)) {
            lm_summary *summary = &layers[layer];
            summary->seen = 1;
            if (strcmp(suffix, "attn_norm.weight") == 0) summary->attn_norm = 1;
            if (strcmp(suffix, "post_attention_norm.weight") == 0) summary->post_norm = 1;
            if (lm_recurrent_signal(suffix)) summary->recurrent = 1;
            if (lm_full_attention_signal(suffix)) summary->full_attention = 1;
        }
        free(name);
    }
    fclose(f);

    uint32_t highest = 0;
    int any = 0;
    for (uint32_t layer = 0; layer < RL_LAYER_MAP_MAX_LAYERS; ++layer) {
        if (!layers[layer].seen) continue;
        any = 1;
        highest = layer;
        if (!layers[layer].attn_norm || !layers[layer].post_norm) {
            if (error && error_cap) snprintf(error, error_cap, "layer %u is missing a norm pair", layer);
            return 0;
        }
        if (layers[layer].recurrent && layers[layer].full_attention) out->kind[layer] = RL_LAYER_MAP_MIXED;
        else if (layers[layer].recurrent) out->kind[layer] = RL_LAYER_MAP_RECURRENT;
        else if (layers[layer].full_attention) out->kind[layer] = RL_LAYER_MAP_FULL_ATTENTION;
        else out->kind[layer] = RL_LAYER_MAP_UNKNOWN;
        if (out->kind[layer] == RL_LAYER_MAP_MIXED || out->kind[layer] == RL_LAYER_MAP_UNKNOWN) {
            if (error && error_cap) snprintf(error, error_cap, "layer %u classified as %s", layer,
                rl_native_layer_map_kind_name(out->kind[layer]));
            return 0;
        }
    }
    if (!any) { lm_error(error, error_cap, "GGUF contains no decoder layers"); return 0; }
    out->layer_count = highest + 1u;
    for (uint32_t layer = 0; layer < out->layer_count; ++layer) {
        if (!layers[layer].seen) {
            if (error && error_cap) snprintf(error, error_cap, "decoder layer sequence has a gap at %u", layer);
            return 0;
        }
        if (out->kind[layer] == RL_LAYER_MAP_RECURRENT) out->recurrent_count++;
        else if (out->kind[layer] == RL_LAYER_MAP_FULL_ATTENTION) out->full_attention_count++;
    }
    if (error && error_cap) error[0] = '\0';
    return 1;
}
