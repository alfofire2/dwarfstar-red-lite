#define _FILE_OFFSET_BITS 64
#define _POSIX_C_SOURCE 200809L

#include "redlite_native_deltanet_proj.h"
#include "redlite_native_iq2_xxs.h"
#include "redlite_native_shared_exec.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define GGUF_DEFAULT_ALIGNMENT 32u
#define GGUF_MAX_STRING (1ull << 30)
#define GGUF_MAX_ARRAY  (1ull << 30)
#define IQ2_XXS_BLOCK_BYTES 66u
#define Q8_0_BLOCK_BYTES 34u

enum {
    GGUF_U8 = 0, GGUF_I8, GGUF_U16, GGUF_I16, GGUF_U32, GGUF_I32,
    GGUF_F32, GGUF_BOOL, GGUF_STRING, GGUF_ARRAY, GGUF_U64, GGUF_I64, GGUF_F64
};

typedef struct {
    char *name;
    uint32_t n_dims;
    uint64_t shape[RL_DN_MAX_DIMS];
    uint32_t ggml_type;
    uint64_t relative_offset;
    uint64_t absolute_offset;
    uint64_t span_bytes;
} raw_tensor;

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

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
    if (!read_u64(f, &n) || n > GGUF_MAX_STRING || n > SIZE_MAX - 1) return 0;
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

static int read_metadata(FILE *f, uint64_t count, uint32_t *alignment, float *rms_eps, int *have_rms_eps) {
    for (uint64_t i = 0; i < count; ++i) {
        char *key = NULL;
        uint32_t type = 0;
        if (!read_string(f, &key) || !read_u32(f, &type)) { free(key); return 0; }
        if (strcmp(key, "general.alignment") == 0 && type == GGUF_U32) {
            uint32_t value = 0;
            free(key);
            if (!read_u32(f, &value) || !value) return 0;
            *alignment = value;
        } else if (strcmp(key, "qwen3next.attention.layer_norm_rms_epsilon") == 0 && type == GGUF_F32) {
            free(key);
            if (!read_f32(f, rms_eps) || !isfinite(*rms_eps) || *rms_eps < 0.0f) return 0;
            *have_rms_eps = 1;
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
    if (ta->relative_offset < tb->relative_offset) return -1;
    if (ta->relative_offset > tb->relative_offset) return 1;
    return 0;
}

static void free_raw(raw_tensor *raw, uint64_t count) {
    if (!raw) return;
    for (uint64_t i = 0; i < count; ++i) free(raw[i].name);
    free(raw);
}

static const char *kind_name(rl_dn_proj_kind k) {
    switch (k) {
        case RL_DN_ATTN_NORM: return "attn_norm";
        case RL_DN_QKV: return "attn_qkv";
        case RL_DN_Z_GATE: return "attn_gate";
        case RL_DN_BETA_ALPHA: return "ssm_ba";
        default: return "unknown";
    }
}

static int target_kind(const char *name, uint32_t layer, rl_dn_proj_kind *kind) {
    char expected[96];
    snprintf(expected, sizeof(expected), "blk.%u.attn_norm.weight", layer);
    if (strcmp(name, expected) == 0) { *kind = RL_DN_ATTN_NORM; return 1; }
    snprintf(expected, sizeof(expected), "blk.%u.attn_qkv.weight", layer);
    if (strcmp(name, expected) == 0) { *kind = RL_DN_QKV; return 1; }
    snprintf(expected, sizeof(expected), "blk.%u.attn_gate.weight", layer);
    if (strcmp(name, expected) == 0) { *kind = RL_DN_Z_GATE; return 1; }
    snprintf(expected, sizeof(expected), "blk.%u.ssm_ba.weight", layer);
    if (strcmp(name, expected) == 0) { *kind = RL_DN_BETA_ALPHA; return 1; }
    return 0;
}

static int audit_layer(const char *model, uint32_t layer, rl_dn_proj_tensor_info out[RL_DN_PROJ_TENSOR_COUNT],
        float *rms_eps, char *error, size_t cap) {
    memset(out, 0, RL_DN_PROJ_TENSOR_COUNT * sizeof(*out));
    FILE *f = fopen(model, "rb");
    if (!f) { snprintf(error, cap, "open failed: %s", strerror(errno)); return 0; }
    if (fseeko(f, 0, SEEK_END) != 0) { fclose(f); snprintf(error, cap, "seek failed"); return 0; }
    const off_t end = ftello(f);
    if (end <= 0 || fseeko(f, 0, SEEK_SET) != 0) { fclose(f); snprintf(error, cap, "invalid GGUF size"); return 0; }
    const uint64_t file_size = (uint64_t)end;

    unsigned char magic[4];
    uint32_t version = 0;
    uint64_t tensor_count = 0, kv_count = 0;
    if (!read_exact(f, magic, 4) || memcmp(magic, "GGUF", 4) != 0 || !read_u32(f, &version) ||
        (version != 2u && version != 3u) || !read_u64(f, &tensor_count) || !read_u64(f, &kv_count)) {
        fclose(f); snprintf(error, cap, "invalid GGUF header"); return 0;
    }
    if (tensor_count > SIZE_MAX / sizeof(raw_tensor)) { fclose(f); snprintf(error, cap, "tensor directory too large"); return 0; }
    uint32_t alignment = GGUF_DEFAULT_ALIGNMENT;
    int have_rms_eps = 0;
    if (!read_metadata(f, kv_count, &alignment, rms_eps, &have_rms_eps)) {
        fclose(f); snprintf(error, cap, "metadata parse failed"); return 0;
    }
    if (!have_rms_eps) { fclose(f); snprintf(error, cap, "missing qwen3next RMS epsilon metadata"); return 0; }

    raw_tensor *raw = (raw_tensor *)calloc((size_t)tensor_count, sizeof(*raw));
    if (!raw) { fclose(f); snprintf(error, cap, "out of memory for tensor directory"); return 0; }
    for (uint64_t i = 0; i < tensor_count; ++i) {
        uint32_t dims = 0;
        if (!read_string(f, &raw[i].name) || !read_u32(f, &dims) || dims > RL_DN_MAX_DIMS) {
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
    if (directory_end < 0) { free_raw(raw, tensor_count); snprintf(error, cap, "failed to locate GGUF data section"); return 0; }
    const uint64_t data_base = round_up_u64((uint64_t)directory_end, alignment);
    if (!data_base || data_base > file_size) { free_raw(raw, tensor_count); snprintf(error, cap, "invalid GGUF data base"); return 0; }

    qsort(raw, (size_t)tensor_count, sizeof(*raw), tensor_cmp);
    uint8_t seen[RL_DN_PROJ_TENSOR_COUNT] = {0};
    for (uint64_t i = 0; i < tensor_count; ++i) {
        const uint64_t rel = raw[i].relative_offset;
        const uint64_t next = i + 1 < tensor_count ? raw[i + 1].relative_offset : file_size - data_base;
        raw[i].absolute_offset = data_base + rel;
        raw[i].span_bytes = next >= rel ? next - rel : 0;
        rl_dn_proj_kind kind;
        if (!target_kind(raw[i].name, layer, &kind)) continue;
        if (seen[(uint32_t)kind]) { free_raw(raw, tensor_count); snprintf(error, cap, "duplicate DeltaNet projection tensor"); return 0; }
        seen[(uint32_t)kind] = 1;
        rl_dn_proj_tensor_info *t = &out[(uint32_t)kind];
        t->layer = layer;
        t->kind = kind;
        t->ggml_type = raw[i].ggml_type;
        t->n_dims = raw[i].n_dims;
        memcpy(t->shape, raw[i].shape, sizeof(t->shape));
        t->tensor_offset = raw[i].absolute_offset;
        t->tensor_span_bytes = raw[i].span_bytes;
    }
    free_raw(raw, tensor_count);
    for (uint32_t k = 0; k < RL_DN_PROJ_TENSOR_COUNT; ++k) if (!seen[k]) {
        snprintf(error, cap, "requested layer is not a complete optimized DeltaNet layer (missing %s)", kind_name((rl_dn_proj_kind)k));
        return 0;
    }
    if (error && cap) error[0] = '\0';
    return 1;
}

static size_t iq2_row_bytes(uint32_t ncols) {
    return ncols && ncols % 256u == 0 ? (size_t)(ncols / 256u) * IQ2_XXS_BLOCK_BYTES : 0;
}

static size_t q8_row_bytes(uint32_t ncols) {
    return ncols && ncols % 32u == 0 ? (size_t)(ncols / 32u) * Q8_0_BLOCK_BYTES : 0;
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

static int q8_row_dot(const uint8_t *row, size_t row_bytes, const float *x, uint32_t ncols, double *out) {
    const size_t expected = q8_row_bytes(ncols);
    if (!row || !x || !out || !expected || row_bytes < expected) return 0;
    double acc = 0.0;
    const uint32_t blocks = ncols / 32u;
    for (uint32_t ib = 0; ib < blocks; ++ib) {
        const uint8_t *bp = row + (size_t)ib * Q8_0_BLOCK_BYTES;
        const double d = (double)f16_to_f32(read_u16_le(bp));
        const int8_t *q = (const int8_t *)(bp + 2u);
        const uint32_t xb = ib * 32u;
        for (uint32_t j = 0; j < 32u; ++j) acc += (double)x[xb + j] * d * (double)q[j];
    }
    *out = acc;
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

static void rmsnorm_cpu(const float *x, const float *w, uint32_t n, float eps, double *out) {
    double mean_sq = 0.0;
    for (uint32_t i = 0; i < n; ++i) mean_sq += (double)x[i] * (double)x[i];
    mean_sq /= (double)n;
    const double scale = 1.0 / sqrt(mean_sq + (double)eps);
    for (uint32_t i = 0; i < n; ++i) out[i] = (double)x[i] * scale * (double)w[i];
}

static int validate_layout(const rl_dn_proj_tensor_info t[RL_DN_PROJ_TENSOR_COUNT], uint32_t *hidden,
        uint32_t *qkv_rows, uint32_t *z_rows, uint32_t *ba_rows, char *error, size_t cap) {
    if (t[0].ggml_type != 0u || t[0].n_dims != 1u || t[0].shape[0] == 0 || t[0].shape[0] > UINT32_MAX) {
        snprintf(error, cap, "attn_norm must be F32[hidden]"); return 0;
    }
    *hidden = (uint32_t)t[0].shape[0];
    if (t[1].ggml_type != 16u || t[2].ggml_type != 16u || t[3].ggml_type != 8u ||
        t[1].n_dims != 2u || t[2].n_dims != 2u || t[3].n_dims != 2u ||
        t[1].shape[0] != *hidden || t[2].shape[0] != *hidden || t[3].shape[0] != *hidden ||
        t[1].shape[1] == 0 || t[2].shape[1] == 0 || t[3].shape[1] == 0 ||
        t[1].shape[1] > UINT32_MAX || t[2].shape[1] > UINT32_MAX || t[3].shape[1] > UINT32_MAX) {
        snprintf(error, cap, "unexpected optimized DeltaNet projection types/shapes"); return 0;
    }
    *qkv_rows = (uint32_t)t[1].shape[1];
    *z_rows = (uint32_t)t[2].shape[1];
    *ba_rows = (uint32_t)t[3].shape[1];
    const size_t iq2_rb = iq2_row_bytes(*hidden);
    const size_t q8_rb = q8_row_bytes(*hidden);
    if (!iq2_rb || !q8_rb || t[0].tensor_span_bytes < (uint64_t)*hidden * sizeof(float) ||
        t[1].tensor_span_bytes < (uint64_t)*qkv_rows * iq2_rb ||
        t[2].tensor_span_bytes < (uint64_t)*z_rows * iq2_rb ||
        t[3].tensor_span_bytes < (uint64_t)*ba_rows * q8_rb) {
        snprintf(error, cap, "DeltaNet projection physical span/layout mismatch"); return 0;
    }
    return 1;
}

static int compare_fd(const float *gpu, const double *cpu, uint32_t n, double *max_abs, double *max_rel) {
    *max_abs = 0.0; *max_rel = 0.0;
    int ok = 1;
    for (uint32_t i = 0; i < n; ++i) {
        const double ae = fabs((double)gpu[i] - cpu[i]);
        const double re = ae / fmax(fabs(cpu[i]), 1e-12);
        if (ae > *max_abs) *max_abs = ae;
        if (re > *max_rel) *max_rel = re;
        if (ae > 1e-3 + 1e-4 * fabs(cpu[i])) ok = 0;
    }
    return ok;
}

static int selftest(void) {
    uint8_t row[Q8_0_BLOCK_BYTES];
    memset(row, 0, sizeof(row));
    row[0] = 0x00; row[1] = 0x3c; /* f16 1.0 */
    for (uint32_t i = 0; i < 32u; ++i) row[2u + i] = 1u;
    float x32[32];
    for (uint32_t i = 0; i < 32u; ++i) x32[i] = 1.0f;
    double q8 = 0.0;
    if (!q8_row_dot(row, sizeof(row), x32, 32u, &q8) || fabs(q8 - 32.0) > 1e-12) return 0;

    const float x[2] = {3.0f, 4.0f};
    const float w[2] = {2.0f, 0.5f};
    double y[2] = {0};
    rmsnorm_cpu(x, w, 2u, 0.0f, y);
    if (fabs(y[0] - 1.697056274847714) > 1e-9 || fabs(y[1] - 0.565685424949238) > 1e-9) return 0;
    return 1;
}

static int parse_u64(const char *s, uint64_t *out) {
    if (!s || !*s) return 0;
    errno = 0;
    char *end = NULL;
    const unsigned long long v = strtoull(s, &end, 10);
    if (errno || !end || *end) return 0;
    *out = (uint64_t)v;
    return 1;
}

static void usage(FILE *out) {
    fprintf(out,
        "redlite-deltanet-proj 0.3.0.dev15\n\n"
        "Usage:\n"
        "  redlite-deltanet-proj parity MODEL --layer N [--rows N]\n"
        "  redlite-deltanet-proj --selftest\n\n"
        "Validates Qwen3-Next DeltaNet input RMSNorm plus optimized QKV, Z-gate and beta/alpha projections.\n");
}

int main(int argc, char **argv) {
    if (argc == 2 && strcmp(argv[1], "--selftest") == 0) {
        if (!selftest()) { fprintf(stderr, "DeltaNet projection selftest failed\n"); return 1; }
        printf("DeltaNet Q8_0 ref : OK\n");
        printf("DeltaNet RMSNorm  : OK\n");
        return 0;
    }
    if (argc < 3 || strcmp(argv[1], "parity") != 0) { usage(argc > 1 ? stderr : stdout); return 2; }
    const char *model = argv[2];
    uint64_t layer64 = UINT64_MAX, rows64 = 8u;
    for (int i = 3; i < argc; ++i) {
        uint64_t *target = NULL;
        if (strcmp(argv[i], "--layer") == 0) target = &layer64;
        else if (strcmp(argv[i], "--rows") == 0) target = &rows64;
        else { fprintf(stderr, "unknown option: %s\n", argv[i]); return 2; }
        if (i + 1 >= argc || !parse_u64(argv[++i], target)) { fprintf(stderr, "invalid option value\n"); return 2; }
    }
    if (layer64 == UINT64_MAX || layer64 > UINT32_MAX || !rows64 || rows64 > UINT32_MAX) {
        fprintf(stderr, "layer/rows out of range\n"); return 2;
    }

    char error[512] = {0};
    rl_dn_proj_tensor_info tensors[RL_DN_PROJ_TENSOR_COUNT];
    float rms_eps = 0.0f;
    if (!audit_layer(model, (uint32_t)layer64, tensors, &rms_eps, error, sizeof(error))) {
        fprintf(stderr, "DeltaNet projection audit failed: %s\n", error); return 1;
    }
    uint32_t hidden = 0, qkv_rows = 0, z_rows = 0, ba_rows = 0;
    if (!validate_layout(tensors, &hidden, &qkv_rows, &z_rows, &ba_rows, error, sizeof(error))) {
        fprintf(stderr, "DeltaNet projection layout failed: %s\n", error); return 1;
    }
    const uint32_t rows = (uint32_t)rows64;
    if (rows > qkv_rows || rows > z_rows || rows > ba_rows) { fprintf(stderr, "--rows exceeds one projection output width\n"); return 2; }

    float *input = (float *)malloc((size_t)hidden * sizeof(float));
    float *norm_w = (float *)malloc((size_t)hidden * sizeof(float));
    double *cpu_norm = (double *)malloc((size_t)hidden * sizeof(double));
    float *gpu_norm = (float *)malloc((size_t)hidden * sizeof(float));
    double *cpu_qkv = (double *)calloc(rows, sizeof(double));
    double *cpu_z = (double *)calloc(rows, sizeof(double));
    double *cpu_ba = (double *)calloc(rows, sizeof(double));
    float *gpu_qkv = (float *)calloc(rows, sizeof(float));
    float *gpu_z = (float *)calloc(rows, sizeof(float));
    float *gpu_ba = (float *)calloc(rows, sizeof(float));
    if (!input || !norm_w || !cpu_norm || !gpu_norm || !cpu_qkv || !cpu_z || !cpu_ba || !gpu_qkv || !gpu_z || !gpu_ba) {
        fprintf(stderr, "out of memory for DeltaNet projection parity\n"); goto fail;
    }
    make_input(input, hidden);

    const size_t iq2_rb = iq2_row_bytes(hidden);
    const size_t q8_rb = q8_row_bytes(hidden);
    const size_t qkv_bytes = (size_t)rows * iq2_rb;
    const size_t z_bytes = (size_t)rows * iq2_rb;
    const size_t ba_bytes = (size_t)rows * q8_rb;
    uint8_t *qkv_buf = (uint8_t *)malloc(qkv_bytes);
    uint8_t *z_buf = (uint8_t *)malloc(z_bytes);
    uint8_t *ba_buf = (uint8_t *)malloc(ba_bytes);
    if (!qkv_buf || !z_buf || !ba_buf) { fprintf(stderr, "out of memory for CPU projection weights\n"); free(qkv_buf); free(z_buf); free(ba_buf); goto fail; }

    const int fd = open(model, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "open model for CPU projection failed: %s\n", strerror(errno)); free(qkv_buf); free(z_buf); free(ba_buf); goto fail; }
    uint64_t cpu_calls = 0;
    const double cr0 = now_ms();
    int read_ok = pread_full(fd, norm_w, (size_t)hidden * sizeof(float), tensors[0].tensor_offset, &cpu_calls) &&
                  pread_full(fd, qkv_buf, qkv_bytes, tensors[1].tensor_offset, &cpu_calls) &&
                  pread_full(fd, z_buf, z_bytes, tensors[2].tensor_offset, &cpu_calls) &&
                  pread_full(fd, ba_buf, ba_bytes, tensors[3].tensor_offset, &cpu_calls);
    const double cr1 = now_ms();
    close(fd);
    if (!read_ok) { fprintf(stderr, "CPU projection pread failed\n"); free(qkv_buf); free(z_buf); free(ba_buf); goto fail; }

    uint8_t grid[RL_IQ2_XXS_GRID_COUNT];
    if (!rl_native_iq2_xxs_build_grid(grid, error, sizeof(error))) {
        fprintf(stderr, "IQ2_XXS grid build failed: %s\n", error); free(qkv_buf); free(z_buf); free(ba_buf); goto fail;
    }
    float *cpu_norm_f = (float *)malloc((size_t)hidden * sizeof(float));
    if (!cpu_norm_f) { fprintf(stderr, "out of memory for normalized CPU input\n"); free(qkv_buf); free(z_buf); free(ba_buf); goto fail; }
    const double cc0 = now_ms();
    rmsnorm_cpu(input, norm_w, hidden, rms_eps, cpu_norm);
    for (uint32_t i = 0; i < hidden; ++i) cpu_norm_f[i] = (float)cpu_norm[i];
    for (uint32_t r = 0; r < rows; ++r) {
        if (!rl_native_shared_quant_row_dot(qkv_buf + (size_t)r * iq2_rb, iq2_rb, 16u, cpu_norm_f, hidden,
                grid, sizeof(grid), &cpu_qkv[r], error, sizeof(error)) ||
            !rl_native_shared_quant_row_dot(z_buf + (size_t)r * iq2_rb, iq2_rb, 16u, cpu_norm_f, hidden,
                grid, sizeof(grid), &cpu_z[r], error, sizeof(error)) ||
            !q8_row_dot(ba_buf + (size_t)r * q8_rb, q8_rb, cpu_norm_f, hidden, &cpu_ba[r])) {
            fprintf(stderr, "CPU DeltaNet projection compute failed: %s\n", error[0] ? error : "Q8_0 row failure");
            free(cpu_norm_f); free(qkv_buf); free(z_buf); free(ba_buf); goto fail;
        }
    }
    const double cc1 = now_ms();
    free(cpu_norm_f); free(qkv_buf); free(z_buf); free(ba_buf);

    rl_dn_proj_telemetry gt = {0};
#ifdef __APPLE__
    if (!rl_deltanet_proj_gpu_execute(model, tensors, input, hidden, rms_eps, rows,
            gpu_norm, hidden, gpu_qkv, gpu_z, gpu_ba, &gt, error, sizeof(error))) {
        fprintf(stderr, "Metal DeltaNet projection failed: %s\n", error); goto fail;
    }
#else
    fprintf(stderr, "DeltaNet projection Metal parity requires macOS\n");
    goto fail;
#endif

    double norm_abs = 0.0, norm_rel = 0.0, qkv_abs = 0.0, qkv_rel = 0.0;
    double z_abs = 0.0, z_rel = 0.0, ba_abs = 0.0, ba_rel = 0.0;
    const int norm_ok = compare_fd(gpu_norm, cpu_norm, hidden, &norm_abs, &norm_rel);
    const int qkv_ok = compare_fd(gpu_qkv, cpu_qkv, rows, &qkv_abs, &qkv_rel);
    const int z_ok = compare_fd(gpu_z, cpu_z, rows, &z_abs, &z_rel);
    const int ba_ok = compare_fd(gpu_ba, cpu_ba, rows, &ba_abs, &ba_rel);

    printf("runtime            : native C + Metal DeltaNet projection parity (no Python/ctypes)\n");
    printf("layer              : %u\n", (uint32_t)layer64);
    printf("RMS epsilon        : %.9g\n", rms_eps);
    printf("hidden             : %u\n", hidden);
    printf("QKV                : IQ2_XXS(16) (%u,%u)\n", hidden, qkv_rows);
    printf("Z gate             : IQ2_XXS(16) (%u,%u)\n", hidden, z_rows);
    printf("beta/alpha         : Q8_0(8) (%u,%u)\n", hidden, ba_rows);
    printf("rows tested        : 0..%u\n", rows - 1u);
    printf("CPU read           : %.3f ms / %" PRIu64 " calls / %.3f KiB\n", cr1 - cr0, cpu_calls,
        (double)((size_t)hidden * sizeof(float) + qkv_bytes + z_bytes + ba_bytes) / 1024.0);
    printf("GPU read           : %.3f ms / %" PRIu64 " calls / %.3f MiB\n", gt.read_ms, gt.read_calls,
        (double)gt.bytes_read / (1024.0 * 1024.0));
    printf("SSD during compute : 0 bytes / 0 calls\n");
    printf("CPU compute        : %.3f ms\n", cc1 - cc0);
    printf("GPU compute        : %.3f ms\n", gt.compute_ms);
    printf("RMSNorm max abs/rel: %.6g / %.6g parity=%s\n", norm_abs, norm_rel, norm_ok ? "YES" : "NO");
    printf("QKV max abs/rel    : %.6g / %.6g parity=%s\n", qkv_abs, qkv_rel, qkv_ok ? "YES" : "NO");
    printf("Z max abs/rel      : %.6g / %.6g parity=%s\n", z_abs, z_rel, z_ok ? "YES" : "NO");
    printf("BA max abs/rel     : %.6g / %.6g parity=%s\n", ba_abs, ba_rel, ba_ok ? "YES" : "NO");
    printf("projection parity  : %s\n", norm_ok && qkv_ok && z_ok && ba_ok ? "YES" : "NO");
    for (uint32_t r = 0; r < rows && r < 4u; ++r) {
        printf("row %u QKV/Z/BA    : gpu=%+.7f/%+.7f/%+.7f cpu=%+.7f/%+.7f/%+.7f\n",
            r, gpu_qkv[r], gpu_z[r], gpu_ba[r], cpu_qkv[r], cpu_z[r], cpu_ba[r]);
    }

    free(input); free(norm_w); free(cpu_norm); free(gpu_norm);
    free(cpu_qkv); free(cpu_z); free(cpu_ba); free(gpu_qkv); free(gpu_z); free(gpu_ba);
    return norm_ok && qkv_ok && z_ok && ba_ok ? 0 : 3;

fail:
    free(input); free(norm_w); free(cpu_norm); free(gpu_norm);
    free(cpu_qkv); free(cpu_z); free(cpu_ba); free(gpu_qkv); free(gpu_z); free(gpu_ba);
    return 1;
}
