#define _FILE_OFFSET_BITS 64
#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_DIMS 8u
#define MAX_LAYERS 256u
#define GGUF_DEFAULT_ALIGNMENT 32u
#define GGUF_MAX_STRING (1ull << 30)
#define GGUF_MAX_ARRAY  (1ull << 30)

enum {
    GGUF_U8 = 0, GGUF_I8, GGUF_U16, GGUF_I16, GGUF_U32, GGUF_I32,
    GGUF_F32, GGUF_BOOL, GGUF_STRING, GGUF_ARRAY, GGUF_U64, GGUF_I64, GGUF_F64
};

typedef enum {
    LAYER_UNKNOWN = 0,
    LAYER_RECURRENT,
    LAYER_FULL_ATTN,
    LAYER_MIXED,
} layer_kind;

typedef struct {
    char *name;
    uint32_t n_dims;
    uint64_t shape[MAX_DIMS];
    uint32_t ggml_type;
    uint64_t relative_offset;
    uint64_t absolute_offset;
    uint64_t span_bytes;
} raw_tensor;

typedef struct {
    uint8_t seen;
    uint8_t has_attn_norm;
    uint8_t has_attn_post_norm;
    uint8_t recurrent_signal;
    uint8_t full_attn_signal;
    uint32_t tensor_count;
    uint64_t span_bytes;
} layer_summary;

static int read_exact(FILE *f, void *dst, size_t n) {
    return n == 0 || fread(dst, 1, n, f) == n;
}

static int read_u32(FILE *f, uint32_t *v) {
    unsigned char b[4];
    if (!read_exact(f, b, 4)) return 0;
    *v = (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
    return 1;
}

static int read_u64(FILE *f, uint64_t *v) {
    unsigned char b[8];
    if (!read_exact(f, b, 8)) return 0;
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

static int read_metadata(FILE *f, uint64_t count, uint32_t *alignment) {
    for (uint64_t i = 0; i < count; ++i) {
        char *key = NULL;
        uint32_t type = 0;
        if (!read_string(f, &key) || !read_u32(f, &type)) { free(key); return 0; }
        if (strcmp(key, "general.alignment") == 0 && type == GGUF_U32) {
            uint32_t value = 0;
            free(key);
            if (!read_u32(f, &value) || !value) return 0;
            *alignment = value;
        } else {
            free(key);
            if (!skip_value(f, type)) return 0;
        }
    }
    return 1;
}

static uint64_t round_up_u64(uint64_t v, uint64_t a) {
    if (!a) return v;
    const uint64_t r = v % a;
    if (!r) return v;
    return v > UINT64_MAX - (a - r) ? 0 : v + a - r;
}

static int tensor_cmp(const void *a, const void *b) {
    const raw_tensor *ta = (const raw_tensor *)a;
    const raw_tensor *tb = (const raw_tensor *)b;
    return ta->relative_offset < tb->relative_offset ? -1 : ta->relative_offset > tb->relative_offset ? 1 : 0;
}

static void free_raw(raw_tensor *raw, uint64_t count) {
    if (!raw) return;
    for (uint64_t i = 0; i < count; ++i) free(raw[i].name);
    free(raw);
}

static const char *type_name(uint32_t t) {
    switch (t) {
        case 0u: return "F32"; case 1u: return "F16"; case 2u: return "Q4_0"; case 3u: return "Q4_1";
        case 6u: return "Q5_0"; case 7u: return "Q5_1"; case 8u: return "Q8_0"; case 9u: return "Q8_1";
        case 10u: return "Q2_K"; case 11u: return "Q3_K"; case 12u: return "Q4_K"; case 13u: return "Q5_K";
        case 14u: return "Q6_K"; case 15u: return "Q8_K"; case 16u: return "IQ2_XXS"; case 17u: return "IQ2_XS";
        case 18u: return "IQ3_XXS"; case 19u: return "IQ1_S"; case 20u: return "IQ4_NL"; case 21u: return "IQ3_S";
        case 22u: return "IQ2_S"; case 23u: return "IQ4_XS"; case 24u: return "I8"; case 25u: return "I16";
        case 26u: return "I32"; case 27u: return "I64"; case 28u: return "F64"; case 29u: return "IQ1_M";
        case 30u: return "BF16"; default: return "OTHER";
    }
}

static int parse_layer_tensor_name(const char *name, uint32_t *layer, const char **suffix) {
    if (!name || strncmp(name, "blk.", 4) != 0) return 0;
    const char *p = name + 4;
    char *end = NULL;
    errno = 0;
    const unsigned long v = strtoul(p, &end, 10);
    if (errno || end == p || !end || *end != '.' || v >= MAX_LAYERS) return 0;
    *layer = (uint32_t)v;
    *suffix = end + 1;
    return 1;
}

static int is_ffn_suffix(const char *s) {
    return strncmp(s, "ffn_", 4) == 0;
}

static int is_recurrent_signal(const char *s) {
    return strncmp(s, "ssm_", 4) == 0 ||
           strcmp(s, "attn_qkv.weight") == 0 ||
           strcmp(s, "attn_gate.weight") == 0;
}

static int is_full_attn_signal(const char *s) {
    return strcmp(s, "attn_q.weight") == 0 ||
           strcmp(s, "attn_k.weight") == 0 ||
           strcmp(s, "attn_v.weight") == 0 ||
           strcmp(s, "attn_output.weight") == 0 ||
           strcmp(s, "attn_q_norm.weight") == 0 ||
           strcmp(s, "attn_k_norm.weight") == 0;
}

static layer_kind classify(const layer_summary *s) {
    if (s->recurrent_signal && s->full_attn_signal) return LAYER_MIXED;
    if (s->recurrent_signal) return LAYER_RECURRENT;
    if (s->full_attn_signal) return LAYER_FULL_ATTN;
    return LAYER_UNKNOWN;
}

static const char *kind_name(layer_kind k) {
    switch (k) {
        case LAYER_RECURRENT: return "RECURRENT/DELTANET";
        case LAYER_FULL_ATTN: return "FULL_ATTENTION";
        case LAYER_MIXED: return "MIXED";
        default: return "UNKNOWN";
    }
}

static void print_shape(const raw_tensor *t) {
    putchar('(');
    for (uint32_t d = 0; d < t->n_dims; ++d) {
        if (d) putchar(',');
        printf("%" PRIu64, t->shape[d]);
    }
    putchar(')');
}

static int selftest(void) {
    uint32_t layer = 0;
    const char *suffix = NULL;
    if (!parse_layer_tensor_name("blk.3.attn_q.weight", &layer, &suffix) || layer != 3u || strcmp(suffix, "attn_q.weight") != 0) return 0;
    if (!is_full_attn_signal(suffix) || is_recurrent_signal(suffix)) return 0;
    if (!parse_layer_tensor_name("blk.6.ssm_conv1d.weight", &layer, &suffix) || layer != 6u || !is_recurrent_signal(suffix)) return 0;
    if (!parse_layer_tensor_name("blk.9.ffn_gate_inp.weight", &layer, &suffix) || !is_ffn_suffix(suffix)) return 0;
    return 1;
}

static void usage(const char *argv0) {
    fprintf(stderr,
        "usage: %s MODEL [--tensors]\n"
        "       %s --selftest\n\n"
        "Audit non-FFN Qwen3-Next layer tensors and classify recurrent DeltaNet vs full-attention layers.\n",
        argv0, argv0);
}

int main(int argc, char **argv) {
    if (argc == 2 && strcmp(argv[1], "--selftest") == 0) {
        if (!selftest()) { fprintf(stderr, "layer classifier selftest failed\n"); return 1; }
        printf("layer name parser  : OK\n");
        printf("FFN exclusion      : OK\n");
        printf("attention classify : OK\n");
        return 0;
    }
    if (argc < 2 || strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0) {
        usage(argv[0]);
        return argc < 2 ? 2 : 0;
    }
    const char *model = argv[1];
    int show_tensors = 0;
    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--tensors") == 0) show_tensors = 1;
        else { fprintf(stderr, "unknown option: %s\n", argv[i]); usage(argv[0]); return 2; }
    }

    FILE *f = fopen(model, "rb");
    if (!f) { fprintf(stderr, "open failed: %s\n", strerror(errno)); return 1; }
    if (fseeko(f, 0, SEEK_END) != 0) { fclose(f); return 1; }
    const off_t end = ftello(f);
    if (end <= 0 || fseeko(f, 0, SEEK_SET) != 0) { fclose(f); return 1; }
    const uint64_t file_size = (uint64_t)end;

    unsigned char magic[4];
    uint32_t version = 0;
    uint64_t tensor_count = 0, kv_count = 0;
    if (!read_exact(f, magic, 4) || memcmp(magic, "GGUF", 4) != 0 ||
        !read_u32(f, &version) || (version != 2u && version != 3u) ||
        !read_u64(f, &tensor_count) || !read_u64(f, &kv_count)) {
        fprintf(stderr, "invalid GGUF header\n"); fclose(f); return 1;
    }
    if (tensor_count > SIZE_MAX / sizeof(raw_tensor)) { fclose(f); return 1; }
    uint32_t alignment = GGUF_DEFAULT_ALIGNMENT;
    if (!read_metadata(f, kv_count, &alignment)) { fprintf(stderr, "metadata parse failed\n"); fclose(f); return 1; }

    raw_tensor *raw = (raw_tensor *)calloc((size_t)tensor_count, sizeof(*raw));
    if (!raw) { fclose(f); return 1; }
    for (uint64_t i = 0; i < tensor_count; ++i) {
        uint32_t dims = 0;
        if (!read_string(f, &raw[i].name) || !read_u32(f, &dims) || dims > MAX_DIMS) {
            fprintf(stderr, "tensor descriptor parse failed\n"); free_raw(raw, tensor_count); fclose(f); return 1;
        }
        raw[i].n_dims = dims;
        for (uint32_t d = 0; d < dims; ++d) if (!read_u64(f, &raw[i].shape[d])) {
            fprintf(stderr, "tensor shape parse failed\n"); free_raw(raw, tensor_count); fclose(f); return 1;
        }
        if (!read_u32(f, &raw[i].ggml_type) || !read_u64(f, &raw[i].relative_offset)) {
            fprintf(stderr, "tensor type/offset parse failed\n"); free_raw(raw, tensor_count); fclose(f); return 1;
        }
    }
    const off_t directory_end = ftello(f);
    fclose(f);
    if (directory_end < 0) { free_raw(raw, tensor_count); return 1; }
    const uint64_t data_base = round_up_u64((uint64_t)directory_end, alignment);
    if (!data_base || data_base > file_size) { free_raw(raw, tensor_count); return 1; }

    qsort(raw, (size_t)tensor_count, sizeof(*raw), tensor_cmp);
    layer_summary layers[MAX_LAYERS];
    memset(layers, 0, sizeof(layers));
    uint32_t non_ffn_count = 0;
    uint64_t non_ffn_span = 0;
    for (uint64_t i = 0; i < tensor_count; ++i) {
        const uint64_t rel = raw[i].relative_offset;
        const uint64_t next = i + 1 < tensor_count ? raw[i + 1].relative_offset : file_size - data_base;
        raw[i].absolute_offset = data_base + rel;
        raw[i].span_bytes = next >= rel ? next - rel : 0;
        uint32_t layer = 0;
        const char *suffix = NULL;
        if (!parse_layer_tensor_name(raw[i].name, &layer, &suffix) || is_ffn_suffix(suffix)) continue;
        layer_summary *s = &layers[layer];
        s->seen = 1;
        s->tensor_count++;
        s->span_bytes += raw[i].span_bytes;
        non_ffn_count++;
        non_ffn_span += raw[i].span_bytes;
        if (strcmp(suffix, "attn_norm.weight") == 0) s->has_attn_norm = 1;
        if (strcmp(suffix, "attn_post_norm.weight") == 0) s->has_attn_post_norm = 1;
        if (is_recurrent_signal(suffix)) s->recurrent_signal = 1;
        if (is_full_attn_signal(suffix)) s->full_attn_signal = 1;
    }

    uint32_t layer_count = 0, recurrent = 0, full = 0, mixed = 0, unknown = 0, norms_ok = 0;
    printf("runtime            : native C Qwen3-Next layer graph audit (no Python)\n");
    printf("GGUF version       : %u\n", version);
    printf("tensor count       : %" PRIu64 "\n", tensor_count);
    for (uint32_t l = 0; l < MAX_LAYERS; ++l) {
        if (!layers[l].seen) continue;
        layer_count++;
        if (layers[l].has_attn_norm && layers[l].has_attn_post_norm) norms_ok++;
        switch (classify(&layers[l])) {
            case LAYER_RECURRENT: recurrent++; break;
            case LAYER_FULL_ATTN: full++; break;
            case LAYER_MIXED: mixed++; break;
            default: unknown++; break;
        }
    }
    printf("layer tensors      : %u non-FFN tensors / %.3f MiB physical span\n", non_ffn_count, (double)non_ffn_span / (1024.0 * 1024.0));
    printf("layer count        : %u\n", layer_count);
    printf("attention pattern  : recurrent=%u full=%u mixed=%u unknown=%u\n", recurrent, full, mixed, unknown);
    printf("norm pairs         : %u/%u\n", norms_ok, layer_count);
    printf("recurrent layers   : ");
    int first = 1;
    for (uint32_t l = 0; l < MAX_LAYERS; ++l) if (layers[l].seen && classify(&layers[l]) == LAYER_RECURRENT) {
        printf("%s%u", first ? "" : ",", l); first = 0;
    }
    if (first) printf("NONE");
    putchar('\n');
    printf("full-attn layers   : ");
    first = 1;
    for (uint32_t l = 0; l < MAX_LAYERS; ++l) if (layers[l].seen && classify(&layers[l]) == LAYER_FULL_ATTN) {
        printf("%s%u", first ? "" : ",", l); first = 0;
    }
    if (first) printf("NONE");
    putchar('\n');
    printf("per layer          :\n");
    for (uint32_t l = 0; l < MAX_LAYERS; ++l) {
        if (!layers[l].seen) continue;
        printf("%5u: %-18s tensors=%u span=%.3f MiB norms=%s\n",
            l, kind_name(classify(&layers[l])), layers[l].tensor_count,
            (double)layers[l].span_bytes / (1024.0 * 1024.0),
            layers[l].has_attn_norm && layers[l].has_attn_post_norm ? "YES" : "NO");
        if (!show_tensors) continue;
        for (uint64_t i = 0; i < tensor_count; ++i) {
            uint32_t layer = 0;
            const char *suffix = NULL;
            if (!parse_layer_tensor_name(raw[i].name, &layer, &suffix) || layer != l || is_ffn_suffix(suffix)) continue;
            printf("        %-34s type=%s(%u) shape=", suffix, type_name(raw[i].ggml_type), raw[i].ggml_type);
            print_shape(&raw[i]);
            printf(" span=%.3f MiB offset=%" PRIu64 "\n",
                (double)raw[i].span_bytes / (1024.0 * 1024.0), raw[i].absolute_offset);
        }
    }

    const int ok = layer_count > 0 && mixed == 0 && unknown == 0 && norms_ok == layer_count;
    free_raw(raw, tensor_count);
    return ok ? 0 : 2;
}
