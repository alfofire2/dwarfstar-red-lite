#define _FILE_OFFSET_BITS 64
#define _POSIX_C_SOURCE 200809L

#include "redlite_native_shared_exec.h"
#include "redlite_native_iq2_xxs.h"

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define QK 256u
#define Q6_K_BLOCK_BYTES 210u
#define IQ2_XXS_BLOCK_BYTES 66u

static void set_error(char *dst, size_t cap, const char *msg) {
    if (dst && cap) snprintf(dst, cap, "%s", msg ? msg : "unknown shared expert error");
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
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
        if (mant == 0) {
            bits = sign;
        } else {
            int shift = 0;
            while ((mant & 0x0400u) == 0) { mant <<= 1; ++shift; }
            mant &= 0x03ffu;
            const uint32_t exp32 = (uint32_t)(127 - 14 - shift);
            bits = sign | (exp32 << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + (127u - 15u)) << 23) | (mant << 13);
    }
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static uint32_t popcount7(uint32_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return (uint32_t)__builtin_popcount(x & 127u);
#else
    x &= 127u;
    uint32_t c = 0;
    while (x) { c += x & 1u; x >>= 1; }
    return c;
#endif
}

size_t rl_native_shared_row_bytes(uint32_t ggml_type, uint32_t ncols) {
    if (!ncols || ncols % QK != 0) return 0;
    const size_t blocks = ncols / QK;
    if (ggml_type == 14u) return blocks * Q6_K_BLOCK_BYTES;
    if (ggml_type == 16u) return blocks * IQ2_XXS_BLOCK_BYTES;
    return 0;
}

static int q6_k_row_dot(const uint8_t *row, size_t row_bytes, const float *input, uint32_t ncols, double *out) {
    const size_t expected = rl_native_shared_row_bytes(14u, ncols);
    if (!row || !input || !out || row_bytes < expected) return 0;
    double acc = 0.0;
    const uint32_t blocks = ncols / QK;
    for (uint32_t ib = 0; ib < blocks; ++ib) {
        const uint8_t *bp = row + (size_t)ib * Q6_K_BLOCK_BYTES;
        const uint8_t *ql0 = bp;
        const uint8_t *qh0 = bp + 128u;
        const int8_t *sc0 = (const int8_t *)(bp + 192u);
        const float d = f16_to_f32(read_u16_le(bp + 208u));
        for (uint32_t n = 0; n < 256u; n += 128u) {
            const uint8_t *ql = ql0 + n / 2u;
            const uint8_t *qh = qh0 + n / 4u;
            const int8_t *sc = sc0 + n / 16u;
            for (uint32_t l = 0; l < 32u; ++l) {
                const uint32_t is = l / 16u;
                const int q1 = (int)((ql[l] & 0x0fu) | (((qh[l] >> 0) & 3u) << 4)) - 32;
                const int q2 = (int)((ql[l + 32u] & 0x0fu) | (((qh[l] >> 2) & 3u) << 4)) - 32;
                const int q3 = (int)((ql[l] >> 4) | (((qh[l] >> 4) & 3u) << 4)) - 32;
                const int q4 = (int)((ql[l + 32u] >> 4) | (((qh[l] >> 6) & 3u) << 4)) - 32;
                const uint32_t xb = ib * 256u + n;
                acc += (double)input[xb + l]       * ((double)d * sc[is + 0u] * q1);
                acc += (double)input[xb + l + 32u] * ((double)d * sc[is + 2u] * q2);
                acc += (double)input[xb + l + 64u] * ((double)d * sc[is + 4u] * q3);
                acc += (double)input[xb + l + 96u] * ((double)d * sc[is + 6u] * q4);
            }
        }
    }
    *out = acc;
    return 1;
}

static int iq2_xxs_row_dot(const uint8_t *row, size_t row_bytes, const float *input, uint32_t ncols,
        const uint8_t *grid, size_t grid_count, double *out) {
    const size_t expected = rl_native_shared_row_bytes(16u, ncols);
    if (!row || !input || !grid || grid_count < RL_IQ2_XXS_GRID_COUNT || !out || row_bytes < expected) return 0;
    double acc = 0.0;
    const uint32_t blocks = ncols / QK;
    for (uint32_t ib = 0; ib < blocks; ++ib) {
        const uint8_t *bp = row + (size_t)ib * IQ2_XXS_BLOCK_BYTES;
        const float d = f16_to_f32(read_u16_le(bp));
        const uint8_t *qp = bp + 2u;
        for (uint32_t g = 0; g < 8u; ++g) {
            const uint32_t qi = 4u * g;
            const uint32_t q0 = read_u16_le(qp + 2u * (qi + 0u));
            const uint32_t q1 = read_u16_le(qp + 2u * (qi + 1u));
            const uint32_t q2 = read_u16_le(qp + 2u * (qi + 2u));
            const uint32_t q3 = read_u16_le(qp + 2u * (qi + 3u));
            const uint32_t auxg = q0 | (q1 << 16);
            const uint32_t auxs = q2 | (q3 << 16);
            const double db = (double)d * (0.5 + (double)(auxs >> 28)) * 0.25;
            for (uint32_t l = 0; l < 4u; ++l) {
                const uint32_t grid_index = (auxg >> (8u * l)) & 255u;
                const uint32_t sign7 = (auxs >> (7u * l)) & 127u;
                const uint32_t sign8 = sign7 | ((popcount7(sign7) & 1u) << 7);
                const uint8_t *gv = grid + grid_index * 8u;
                const uint32_t xb = ib * 256u + g * 32u + l * 8u;
                for (uint32_t j = 0; j < 8u; ++j) {
                    const double s = (sign8 & (1u << j)) ? -1.0 : 1.0;
                    acc += (double)input[xb + j] * (db * (double)gv[j] * s);
                }
            }
        }
    }
    *out = acc;
    return 1;
}

int rl_native_shared_quant_row_dot(
        const uint8_t *row, size_t row_bytes, uint32_t ggml_type,
        const float *input, uint32_t ncols, const uint8_t *iq2_grid, size_t iq2_grid_count,
        double *out, char *error, size_t error_cap) {
    int ok = 0;
    if (ggml_type == 14u) ok = q6_k_row_dot(row, row_bytes, input, ncols, out);
    else if (ggml_type == 16u) ok = iq2_xxs_row_dot(row, row_bytes, input, ncols, iq2_grid, iq2_grid_count, out);
    else {
        set_error(error, error_cap, "unsupported shared quant type (expected Q6_K or IQ2_XXS)");
        return 0;
    }
    if (!ok) {
        set_error(error, error_cap, "invalid shared quant row layout");
        return 0;
    }
    if (error && error_cap) error[0] = '\0';
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

static int validate_layout(const rl_shared_tensor_info t[4], uint32_t *hidden, uint32_t *ffn, size_t bytes[4], char *error, size_t cap) {
    if (!t || !hidden || !ffn || !bytes) return 0;
    if (t[RL_SHARED_GATE_INPUT].kind != RL_SHARED_GATE_INPUT ||
        t[RL_SHARED_GATE].kind != RL_SHARED_GATE || t[RL_SHARED_UP].kind != RL_SHARED_UP ||
        t[RL_SHARED_DOWN].kind != RL_SHARED_DOWN) {
        set_error(error, cap, "shared tensors must be indexed by rl_shared_kind");
        return 0;
    }
    const uint32_t layer = t[0].layer;
    for (uint32_t k = 1; k < 4; ++k) if (t[k].layer != layer) {
        set_error(error, cap, "shared tensor layer mismatch");
        return 0;
    }
    if (t[0].ggml_type != 0u || t[0].n_dims != 1u || t[0].shape[0] > UINT32_MAX ||
        t[1].n_dims != 2u || t[2].n_dims != 2u || t[3].n_dims != 2u ||
        t[1].shape[0] > UINT32_MAX || t[1].shape[1] > UINT32_MAX) {
        set_error(error, cap, "invalid shared tensor ranks/types");
        return 0;
    }
    *hidden = (uint32_t)t[0].shape[0];
    *ffn = (uint32_t)t[1].shape[1];
    if (!*hidden || !*ffn || t[1].shape[0] != *hidden || t[2].shape[0] != *hidden ||
        t[2].shape[1] != *ffn || t[3].shape[0] != *ffn || t[3].shape[1] != *hidden) {
        set_error(error, cap, "shared tensor shape mismatch");
        return 0;
    }
    bytes[0] = (size_t)(*hidden) * sizeof(float);
    const size_t gate_rb = rl_native_shared_row_bytes(t[1].ggml_type, *hidden);
    const size_t up_rb = rl_native_shared_row_bytes(t[2].ggml_type, *hidden);
    const size_t down_rb = rl_native_shared_row_bytes(t[3].ggml_type, *ffn);
    if (!gate_rb || !up_rb || !down_rb || *ffn > SIZE_MAX / gate_rb || *ffn > SIZE_MAX / up_rb || *hidden > SIZE_MAX / down_rb) {
        set_error(error, cap, "unsupported shared quant layout");
        return 0;
    }
    bytes[1] = (size_t)(*ffn) * gate_rb;
    bytes[2] = (size_t)(*ffn) * up_rb;
    bytes[3] = (size_t)(*hidden) * down_rb;
    for (uint32_t k = 0; k < 4; ++k) if (t[k].tensor_span_bytes < bytes[k]) {
        set_error(error, cap, "shared tensor physical span is smaller than expected payload");
        return 0;
    }
    return 1;
}

static double sigmoid_stable(double x) {
    if (x >= 0.0) { const double z = exp(-x); return 1.0 / (1.0 + z); }
    const double z = exp(x); return z / (1.0 + z);
}

int rl_native_shared_cpu_execute(
        const char *model_path, const rl_shared_tensor_info tensors[4],
        const float *input, uint32_t input_count, uint32_t row_start, uint32_t row_count,
        double *output, uint32_t output_count, rl_native_shared_telemetry *telemetry,
        char *error, size_t error_cap) {
    uint32_t hidden = 0, ffn = 0;
    size_t bytes[4] = {0};
    if (!model_path || !input || !output || !row_count || output_count < row_count ||
        !validate_layout(tensors, &hidden, &ffn, bytes, error, error_cap) || input_count != hidden ||
        row_start > hidden || row_count > hidden - row_start) {
        if (error && error_cap && !error[0]) set_error(error, error_cap, "invalid shared CPU execution arguments");
        return 0;
    }
    uint8_t *buf[4] = {0};
    for (uint32_t k = 0; k < 4; ++k) {
        buf[k] = (uint8_t *)malloc(bytes[k]);
        if (!buf[k]) { set_error(error, error_cap, "out of memory for shared CPU weights"); goto fail; }
    }
    const int fd = open(model_path, O_RDONLY);
    if (fd < 0) { set_error(error, error_cap, "open GGUF for shared CPU execution failed"); goto fail; }
    uint64_t read_calls = 0;
    const double read_start = now_ms();
    for (uint32_t k = 0; k < 4; ++k) {
        if (!pread_full(fd, buf[k], bytes[k], tensors[k].tensor_offset, &read_calls)) {
            close(fd); set_error(error, error_cap, "pread shared CPU weights failed"); goto fail;
        }
    }
    const double read_end = now_ms();
    close(fd);

    uint8_t grid[RL_IQ2_XXS_GRID_COUNT];
    if (!rl_native_iq2_xxs_build_grid(grid, error, error_cap)) goto fail;
    double *gate = (double *)malloc((size_t)ffn * sizeof(double));
    double *up = (double *)malloc((size_t)ffn * sizeof(double));
    float *act = (float *)malloc((size_t)ffn * sizeof(float));
    if (!gate || !up || !act) { free(gate); free(up); free(act); set_error(error, error_cap, "out of memory for shared CPU scratch"); goto fail; }

    const double compute_start = now_ms();
    double gate_logit = 0.0;
    for (uint32_t i = 0; i < hidden; ++i) {
        float w;
        memcpy(&w, buf[0] + (size_t)i * sizeof(float), sizeof(w));
        gate_logit += (double)w * input[i];
    }
    const double scalar_gate = sigmoid_stable(gate_logit);
    const size_t gate_rb = rl_native_shared_row_bytes(tensors[1].ggml_type, hidden);
    const size_t up_rb = rl_native_shared_row_bytes(tensors[2].ggml_type, hidden);
    for (uint32_t r = 0; r < ffn; ++r) {
        if (!rl_native_shared_quant_row_dot(buf[1] + (size_t)r * gate_rb, gate_rb, tensors[1].ggml_type,
                input, hidden, grid, sizeof(grid), &gate[r], error, error_cap) ||
            !rl_native_shared_quant_row_dot(buf[2] + (size_t)r * up_rb, up_rb, tensors[2].ggml_type,
                input, hidden, grid, sizeof(grid), &up[r], error, error_cap)) {
            free(gate); free(up); free(act); goto fail;
        }
        const float g = (float)gate[r];
        const float u = (float)up[r];
        act[r] = (g / (1.0f + expf(-g))) * u;
    }
    const size_t down_rb = rl_native_shared_row_bytes(tensors[3].ggml_type, ffn);
    for (uint32_t i = 0; i < row_count; ++i) {
        double v = 0.0;
        const uint32_t r = row_start + i;
        if (!rl_native_shared_quant_row_dot(buf[3] + (size_t)r * down_rb, down_rb, tensors[3].ggml_type,
                act, ffn, grid, sizeof(grid), &v, error, error_cap)) {
            free(gate); free(up); free(act); goto fail;
        }
        output[i] = v * scalar_gate;
    }
    const double compute_end = now_ms();
    free(gate); free(up); free(act);

    if (telemetry) {
        memset(telemetry, 0, sizeof(*telemetry));
        telemetry->bytes_read = (uint64_t)bytes[0] + bytes[1] + bytes[2] + bytes[3];
        telemetry->read_calls = read_calls;
        telemetry->read_ms = read_end - read_start;
        telemetry->compute_ms = compute_end - compute_start;
        telemetry->scalar_gate = scalar_gate;
    }
    for (uint32_t k = 0; k < 4; ++k) free(buf[k]);
    if (error && error_cap) error[0] = '\0';
    return 1;

fail:
    for (uint32_t k = 0; k < 4; ++k) free(buf[k]);
    return 0;
}
