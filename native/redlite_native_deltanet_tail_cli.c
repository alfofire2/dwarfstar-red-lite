#define _FILE_OFFSET_BITS 64
#define _POSIX_C_SOURCE 200809L

#include "redlite_native_deltanet_tail.h"

#include <errno.h>
#include <fcntl.h>
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
#define RL_MAX_DIMS 8u
#define QK_K 256u
#define Q4_K_BLOCK_BYTES 144u
#define Q4_K_TYPE 12u
#define F32_TYPE 0u

enum {
    GGUF_U8 = 0, GGUF_I8, GGUF_U16, GGUF_I16, GGUF_U32, GGUF_I32,
    GGUF_F32, GGUF_BOOL, GGUF_STRING, GGUF_ARRAY, GGUF_U64, GGUF_I64, GGUF_F64
};

typedef struct {
    uint32_t ggml_type;
    uint32_t n_dims;
    uint64_t shape[RL_MAX_DIMS];
    uint64_t relative_offset;
    uint64_t absolute_offset;
    int found;
} tensor_info;

typedef struct {
    uint32_t alignment;
    float rms_eps;
    int have_eps;
} model_meta;

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
        "redlite-deltanet-tail 0.3.0.dev15\n\n"
        "Usage:\n"
        "  %s parity MODEL --layer N\n"
        "  %s --selftest\n\n"
        "Validates Qwen3-Next gated RMSNorm + real Q4_K ssm_out on the M4.\n",
        argv0, argv0);
}

static int parse_u32(const char *s, uint32_t *out) {
    char *end = NULL;
    errno = 0;
    unsigned long v = strtoul(s, &end, 10);
    if (errno || !end || *end || v > 0xfffffffful) return 0;
    *out = (uint32_t)v;
    return 1;
}

static int read_exact(FILE *f, void *dst, size_t n) { return n == 0 || fread(dst, 1, n, f) == n; }

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
        if (strcmp(key, "general.alignment") == 0 && type == GGUF_U32) {
            free(key);
            if (!read_u32(f, &m->alignment) || !m->alignment) return 0;
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
    return m->have_eps;
}

static uint64_t round_up_u64(uint64_t v, uint64_t a) {
    if (!a) return 0;
    const uint64_t r = v % a;
    if (!r) return v;
    return v > UINT64_MAX - (a - r) ? 0 : v + a - r;
}

static int audit_tail(const char *model, uint32_t layer, model_meta *meta,
        tensor_info *norm, tensor_info *out, char *error, size_t cap) {
    memset(norm, 0, sizeof(*norm));
    memset(out, 0, sizeof(*out));
    FILE *f = fopen(model, "rb");
    if (!f) { snprintf(error, cap, "open failed: %s", strerror(errno)); return 0; }

    uint8_t magic[4];
    uint32_t version = 0;
    uint64_t tensor_count = 0, kv_count = 0;
    if (!read_exact(f, magic, 4) || memcmp(magic, "GGUF", 4) != 0 ||
        !read_u32(f, &version) || (version != 2u && version != 3u) ||
        !read_u64(f, &tensor_count) || !read_u64(f, &kv_count) || !read_metadata(f, kv_count, meta)) {
        fclose(f); snprintf(error, cap, "invalid GGUF/Qwen3-Next metadata"); return 0;
    }

    char norm_name[96], out_name[96];
    snprintf(norm_name, sizeof(norm_name), "blk.%u.ssm_norm.weight", layer);
    snprintf(out_name, sizeof(out_name), "blk.%u.ssm_out.weight", layer);

    for (uint64_t ti = 0; ti < tensor_count; ++ti) {
        char *name = NULL;
        uint32_t dims = 0, type = 0;
        uint64_t shape[RL_MAX_DIMS] = {0}, rel = 0;
        if (!read_string(f, &name) || !read_u32(f, &dims) || dims > RL_MAX_DIMS) {
            free(name); fclose(f); snprintf(error, cap, "tensor descriptor parse failed"); return 0;
        }
        for (uint32_t d = 0; d < dims; ++d) {
            if (!read_u64(f, &shape[d])) { free(name); fclose(f); snprintf(error, cap, "tensor shape parse failed"); return 0; }
        }
        if (!read_u32(f, &type) || !read_u64(f, &rel)) {
            free(name); fclose(f); snprintf(error, cap, "tensor type/offset parse failed"); return 0;
        }
        tensor_info *dst = NULL;
        if (strcmp(name, norm_name) == 0) dst = norm;
        else if (strcmp(name, out_name) == 0) dst = out;
        if (dst) {
            if (dst->found) { free(name); fclose(f); snprintf(error, cap, "duplicate DeltaNet tail tensor"); return 0; }
            dst->found = 1; dst->ggml_type = type; dst->n_dims = dims; dst->relative_offset = rel;
            memcpy(dst->shape, shape, sizeof(shape));
        }
        free(name);
    }

    const off_t directory_end = ftello(f);
    if (directory_end < 0) { fclose(f); snprintf(error, cap, "failed to locate GGUF data"); return 0; }
    const uint64_t data_base = round_up_u64((uint64_t)directory_end, meta->alignment);
    fclose(f);
    if (!data_base || !norm->found || !out->found) { snprintf(error, cap, "missing DeltaNet tail tensors"); return 0; }
    norm->absolute_offset = data_base + norm->relative_offset;
    out->absolute_offset = data_base + out->relative_offset;
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
        if (mant == 0) bits = sign;
        else {
            int shift = 0;
            while ((mant & 0x0400u) == 0) { mant <<= 1; ++shift; }
            mant &= 0x03ffu;
            bits = sign | ((uint32_t)(127 - 14 - shift) << 23) | (mant << 13);
        }
    } else if (exp == 31) bits = sign | 0x7f800000u | (mant << 13);
    else bits = sign | ((exp + 112u) << 23) | (mant << 13);
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
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

static int q4_k_row_dot(const uint8_t *row, const float *input, uint32_t ncols, double *result) {
    if (!row || !input || !result || !ncols || ncols % QK_K) return 0;
    double acc = 0.0;
    const uint32_t blocks = ncols / QK_K;
    for (uint32_t ib = 0; ib < blocks; ++ib) {
        const uint8_t *bp = row + (size_t)ib * Q4_K_BLOCK_BYTES;
        const float d = f16_to_f32(read_u16_le(bp));
        const float dmin = f16_to_f32(read_u16_le(bp + 2u));
        const uint8_t *scales = bp + 4u;
        const uint8_t *qs = bp + 16u;
        for (uint32_t g = 0; g < 8u; ++g) {
            uint8_t sc = 0, m = 0;
            get_scale_min_k4(g, scales, &sc, &m);
            const double ds = (double)d * sc;
            const double dm = (double)dmin * m;
            const uint8_t *q = qs + (g / 2u) * 32u;
            const uint32_t xb = ib * 256u + g * 32u;
            for (uint32_t l = 0; l < 32u; ++l) {
                const uint8_t quant = (g & 1u) ? (q[l] >> 4) : (q[l] & 0x0fu);
                acc += (double)input[xb + l] * (ds * quant - dm);
            }
        }
    }
    *result = acc;
    return 1;
}

static void make_core_z(float *core, float *z, uint32_t head_dim, uint32_t heads) {
    for (uint32_t h = 0; h < heads; ++h) {
        for (uint32_t i = 0; i < head_dim; ++i) {
            const size_t idx = (size_t)h * head_dim + i;
            const int cv = (int)((h * 37u + i * 19u + 5u) % 127u) - 63;
            const int zv = (int)((h * 11u + i * 23u + 17u) % 97u) - 48;
            core[idx] = (float)cv / 256.0f;
            z[idx] = (float)zv / 32.0f;
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
    for (size_t i = 0; i < n; ++i)
        if (fabsf(got[i] - ref[i]) > abs_tol + rel_tol * fabsf(ref[i])) return 0;
    return 1;
}

static int selftest(void) {
    uint8_t block[Q4_K_BLOCK_BYTES];
    memset(block, 0, sizeof(block));
    block[0] = 0x00; block[1] = 0x3c; /* d = f16(1) */
    block[2] = 0x00; block[3] = 0x00; /* dmin = 0 */
    for (uint32_t j = 0; j < 4u; ++j) block[4u + j] = 1u;
    for (uint32_t j = 8u; j < 12u; ++j) block[4u + j] = 1u;
    memset(block + 16u, 0x11, 128u);
    float x[256];
    for (uint32_t i = 0; i < 256u; ++i) x[i] = 1.0f;
    double dot = 0.0;
    if (!q4_k_row_dot(block, x, 256u, &dot) || fabs(dot - 256.0) > 1.0e-9) {
        fprintf(stderr, "DeltaNet Q4_K ref   : FAIL (dot=%.9f)\n", dot);
        return 1;
    }
    printf("DeltaNet Q4_K ref   : OK\n");

    const float core[4] = {1.0f, -1.0f, 2.0f, -2.0f};
    const float z[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const float w[2] = {1.0f, 1.0f};
    float y[4] = {1,1,1,1};
    cpu_gated_norm(core, z, w, 1.0e-6f, 2u, 2u, y);
    for (uint32_t i = 0; i < 4u; ++i) {
        if (y[i] != 0.0f) { fprintf(stderr, "DeltaNet gated norm : FAIL\n"); return 1; }
    }
    printf("DeltaNet gated norm : OK\n");
    return 0;
}

static int run_parity(const char *model, uint32_t layer) {
    model_meta meta;
    tensor_info norm, out;
    char error[512] = {0};
    if (!audit_tail(model, layer, &meta, &norm, &out, error, sizeof(error))) {
        fprintf(stderr, "tail audit failed: %s\n", error);
        return 2;
    }
    if (norm.ggml_type != F32_TYPE || norm.n_dims != 1u || norm.shape[0] != 128u ||
        out.ggml_type != Q4_K_TYPE || out.n_dims != 2u || out.shape[0] != 4096u || out.shape[1] != 2048u) {
        fprintf(stderr, "unexpected layer %u DeltaNet tail layout\n", layer);
        return 2;
    }

    const uint32_t head_dim = 128u, heads = 32u, inner = 4096u, hidden = 2048u;
    const size_t norm_bytes = (size_t)head_dim * sizeof(float);
    const size_t row_bytes = (size_t)(inner / QK_K) * Q4_K_BLOCK_BYTES;
    const size_t out_bytes = row_bytes * hidden;

    float *norm_w = (float *)malloc(norm_bytes);
    uint8_t *q4 = (uint8_t *)malloc(out_bytes);
    float *core = (float *)malloc((size_t)inner * sizeof(float));
    float *z = (float *)malloc((size_t)inner * sizeof(float));
    float *cpu_ng = (float *)malloc((size_t)inner * sizeof(float));
    float *gpu_ng = (float *)malloc((size_t)inner * sizeof(float));
    float *cpu_out = (float *)malloc((size_t)hidden * sizeof(float));
    float *gpu_out = (float *)malloc((size_t)hidden * sizeof(float));
    if (!norm_w || !q4 || !core || !z || !cpu_ng || !gpu_ng || !cpu_out || !gpu_out) {
        fprintf(stderr, "allocation failed for DeltaNet tail\n");
        free(norm_w); free(q4); free(core); free(z); free(cpu_ng); free(gpu_ng); free(cpu_out); free(gpu_out);
        return 2;
    }

    int fd = open(model, O_RDONLY);
    uint64_t read_calls = 0;
    const double r0 = now_ms();
    const int read_ok = fd >= 0 && pread_full(fd, norm_w, norm_bytes, norm.absolute_offset, &read_calls) &&
        pread_full(fd, q4, out_bytes, out.absolute_offset, &read_calls);
    const double read_ms = now_ms() - r0;
    if (fd >= 0) close(fd);
    if (!read_ok) {
        fprintf(stderr, "failed reading real DeltaNet tail tensors\n");
        free(norm_w); free(q4); free(core); free(z); free(cpu_ng); free(gpu_ng); free(cpu_out); free(gpu_out);
        return 2;
    }

    make_core_z(core, z, head_dim, heads);
    const double c0 = now_ms();
    cpu_gated_norm(core, z, norm_w, meta.rms_eps, head_dim, heads, cpu_ng);
    for (uint32_t r = 0; r < hidden; ++r) {
        double dot = 0.0;
        if (!q4_k_row_dot(q4 + (size_t)r * row_bytes, cpu_ng, inner, &dot)) {
            fprintf(stderr, "CPU Q4_K row decode failed\n");
            free(norm_w); free(q4); free(core); free(z); free(cpu_ng); free(gpu_ng); free(cpu_out); free(gpu_out);
            return 2;
        }
        cpu_out[r] = (float)dot;
    }
    const double cpu_ms = now_ms() - c0;

#ifdef __APPLE__
    rl_dn_tail_telemetry tel = {0};
    if (!rl_deltanet_tail_gpu_execute(core, z, norm_w, q4, out_bytes, meta.rms_eps,
            head_dim, heads, hidden, gpu_ng, gpu_out, &tel, error, sizeof(error))) {
        fprintf(stderr, "Metal DeltaNet tail failed: %s\n", error[0] ? error : "unknown error");
        free(norm_w); free(q4); free(core); free(z); free(cpu_ng); free(gpu_ng); free(cpu_out); free(gpu_out);
        return 2;
    }
#else
    fprintf(stderr, "parity requires macOS Metal\n");
    free(norm_w); free(q4); free(core); free(z); free(cpu_ng); free(gpu_ng); free(cpu_out); free(gpu_out);
    return 2;
#endif

    const error_stats ne = compare_arrays(gpu_ng, cpu_ng, inner);
    const error_stats oe = compare_arrays(gpu_out, cpu_out, hidden);
    const int n_ok = within(gpu_ng, cpu_ng, inner, 3.0e-6f, 3.0e-5f);
    const int o_ok = within(gpu_out, cpu_out, hidden, 1.0e-4f, 1.0e-4f);
    const int ok = n_ok && o_ok;

    printf("runtime            : native C + Metal DeltaNet gated tail parity (no Python/ctypes)\n");
    printf("layer              : %u\n", layer);
    printf("RMS epsilon        : %.9g\n", meta.rms_eps);
    printf("gated norm         : F32 weight (128) x 32 value heads\n");
    printf("ssm_out            : Q4_K(%u) (%llu,%llu) row_bytes=%zu\n", out.ggml_type,
        (unsigned long long)out.shape[0], (unsigned long long)out.shape[1], row_bytes);
    printf("weight payload      : %.3f MiB\n", (double)out_bytes / (1024.0 * 1024.0));
    printf("read               : %.3f ms / %llu calls / %.3f MiB\n", read_ms,
        (unsigned long long)read_calls, (double)(norm_bytes + out_bytes) / (1024.0 * 1024.0));
    printf("SSD during compute : 0 bytes / 0 calls\n");
    printf("CPU / GPU compute  : %.3f / %.3f ms\n", cpu_ms, tel.compute_ms);
    printf("norm+gate abs/rel  : %.6g / %.6g parity=%s\n", ne.max_abs, ne.max_rel, n_ok ? "YES" : "NO");
    printf("Q4_K out abs/rel   : %.6g / %.6g parity=%s\n", oe.max_abs, oe.max_rel, o_ok ? "YES" : "NO");
    printf("tail parity        : %s\n", ok ? "YES" : "NO");
    for (uint32_t i = 0; i < 4u; ++i)
        printf("row %u             : gpu=%+.7f cpu=%+.7f delta=%+.3e\n", i, gpu_out[i], cpu_out[i], gpu_out[i]-cpu_out[i]);

    free(norm_w); free(q4); free(core); free(z); free(cpu_ng); free(gpu_ng); free(cpu_out); free(gpu_out);
    return ok ? 0 : 1;
}

int main(int argc, char **argv) {
    if (argc == 2 && strcmp(argv[1], "--selftest") == 0) return selftest();
    if (argc == 2 && strcmp(argv[1], "--help") == 0) { usage(argv[0]); return 0; }
    if (argc < 5 || strcmp(argv[1], "parity") != 0) { usage(argv[0]); return 2; }
    uint32_t layer = UINT32_MAX;
    for (int i = 3; i < argc; ++i) {
        if (strcmp(argv[i], "--layer") == 0 && i + 1 < argc) {
            if (!parse_u32(argv[++i], &layer)) { fprintf(stderr, "invalid --layer\n"); return 2; }
        } else { fprintf(stderr, "unknown argument: %s\n", argv[i]); return 2; }
    }
    if (layer == UINT32_MAX) { fprintf(stderr, "--layer is required\n"); return 2; }
    return run_parity(argv[2], layer);
}
