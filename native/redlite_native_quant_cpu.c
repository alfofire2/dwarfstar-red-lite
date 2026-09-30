#include "redlite_native_quant_cpu.h"
#include "redlite_native_iq2_xxs.h"
#include "redlite_native_iq3.h"
#include "redlite_native_shared_exec.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#define QK_K 256u
#define Q8_0_BLOCK 34u
#define Q2_K_BLOCK 84u
#define Q4_K_BLOCK 144u
#define Q5_K_BLOCK 176u
#define Q6_K_BLOCK 210u
#define IQ2_XXS_BLOCK 66u

static uint16_t rd16(const uint8_t *p) { return (uint16_t)p[0] | ((uint16_t)p[1] << 8); }

float rl_quant_f16_to_f32(uint16_t h) {
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

/* ---- dequantizers (pinned llama.cpp ggml-quants.c semantics) ---- */

static void dequant_q8_0(const uint8_t *row, uint32_t ncols, float *out) {
    for (uint32_t ib = 0; ib < ncols / 32u; ++ib) {
        const uint8_t *bp = row + (size_t)ib * Q8_0_BLOCK;
        const float d = rl_quant_f16_to_f32(rd16(bp));
        const int8_t *q = (const int8_t *)(bp + 2u);
        for (uint32_t j = 0; j < 32u; ++j) out[ib * 32u + j] = d * (float)q[j];
    }
}

static void dequant_q2_k(const uint8_t *row, uint32_t ncols, float *out) {
    for (uint32_t ib = 0; ib < ncols / QK_K; ++ib) {
        const uint8_t *bp = row + (size_t)ib * Q2_K_BLOCK;
        const uint8_t *scales = bp;            /* 16 bytes */
        const uint8_t *q = bp + 16u;           /* 64 bytes */
        const float d = rl_quant_f16_to_f32(rd16(bp + 80u));
        const float min = rl_quant_f16_to_f32(rd16(bp + 82u));
        float *y = out + (size_t)ib * QK_K;
        uint32_t is = 0;
        for (uint32_t n = 0; n < QK_K; n += 128u) {
            uint32_t shift = 0;
            for (uint32_t j = 0; j < 4u; ++j) {
                uint8_t sc = scales[is++];
                float dl = d * (float)(sc & 0xFu), ml = min * (float)(sc >> 4);
                for (uint32_t l = 0; l < 16u; ++l) *y++ = dl * (float)((int8_t)((q[l] >> shift) & 3u)) - ml;
                sc = scales[is++];
                dl = d * (float)(sc & 0xFu); ml = min * (float)(sc >> 4);
                for (uint32_t l = 0; l < 16u; ++l) *y++ = dl * (float)((int8_t)((q[l + 16u] >> shift) & 3u)) - ml;
                shift += 2u;
            }
            q += 32u;
        }
    }
}

static void dequant_q4_k(const uint8_t *row, uint32_t ncols, float *out) {
    for (uint32_t ib = 0; ib < ncols / QK_K; ++ib) {
        const uint8_t *bp = row + (size_t)ib * Q4_K_BLOCK;
        const float d = rl_quant_f16_to_f32(rd16(bp));
        const float dmin = rl_quant_f16_to_f32(rd16(bp + 2u));
        const uint8_t *scales = bp + 4u;
        const uint8_t *qs = bp + 16u;
        float *y = out + (size_t)ib * QK_K;
        for (uint32_t g = 0; g < 8u; ++g) {
            uint8_t sc = 0, mn = 0;
            get_scale_min_k4(g, scales, &sc, &mn);
            const float ds = d * (float)sc, dm = dmin * (float)mn;
            const uint8_t *q = qs + (g / 2u) * 32u;
            for (uint32_t l = 0; l < 32u; ++l) {
                const uint8_t quant = (g & 1u) ? (q[l] >> 4) : (q[l] & 0x0fu);
                y[g * 32u + l] = ds * (float)quant - dm;
            }
        }
    }
}

static void dequant_q5_k(const uint8_t *row, uint32_t ncols, float *out) {
    for (uint32_t ib = 0; ib < ncols / QK_K; ++ib) {
        const uint8_t *bp = row + (size_t)ib * Q5_K_BLOCK;
        const float d = rl_quant_f16_to_f32(rd16(bp));
        const float min = rl_quant_f16_to_f32(rd16(bp + 2u));
        const uint8_t *scales = bp + 4u;   /* 12 bytes */
        const uint8_t *qh = bp + 16u;      /* 32 bytes */
        const uint8_t *ql = bp + 48u;      /* 128 bytes */
        float *y = out + (size_t)ib * QK_K;
        uint32_t is = 0;
        uint8_t u1 = 1u, u2 = 2u;
        for (uint32_t j = 0; j < QK_K; j += 64u) {
            uint8_t sc = 0, m = 0;
            get_scale_min_k4(is + 0u, scales, &sc, &m);
            const float d1 = d * (float)sc, m1 = min * (float)m;
            get_scale_min_k4(is + 1u, scales, &sc, &m);
            const float d2 = d * (float)sc, m2 = min * (float)m;
            for (uint32_t l = 0; l < 32u; ++l) *y++ = d1 * (float)((ql[l] & 0xFu) + ((qh[l] & u1) ? 16u : 0u)) - m1;
            for (uint32_t l = 0; l < 32u; ++l) *y++ = d2 * (float)((ql[l] >> 4) + ((qh[l] & u2) ? 16u : 0u)) - m2;
            ql += 32u; is += 2u;
            u1 <<= 2; u2 <<= 2;
        }
    }
}

static void dequant_q6_k(const uint8_t *row, uint32_t ncols, float *out) {
    for (uint32_t ib = 0; ib < ncols / QK_K; ++ib) {
        const uint8_t *bp = row + (size_t)ib * Q6_K_BLOCK;
        const uint8_t *ql = bp;             /* 128 */
        const uint8_t *qh = bp + 128u;      /* 64 */
        const int8_t *sc = (const int8_t *)(bp + 192u); /* 16 */
        const float d = rl_quant_f16_to_f32(rd16(bp + 208u));
        float *y = out + (size_t)ib * QK_K;
        for (uint32_t n = 0; n < QK_K; n += 128u) {
            for (uint32_t l = 0; l < 32u; ++l) {
                const int is = (int)(l / 16u);
                const int8_t q1 = (int8_t)((ql[l + 0u] & 0xFu) | (((qh[l] >> 0) & 3u) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32u] & 0xFu) | (((qh[l] >> 2) & 3u) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l + 0u] >> 4) | (((qh[l] >> 4) & 3u) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32u] >> 4) | (((qh[l] >> 6) & 3u) << 4)) - 32;
                y[l + 0u] = d * (float)sc[is + 0] * (float)q1;
                y[l + 32u] = d * (float)sc[is + 2] * (float)q2;
                y[l + 64u] = d * (float)sc[is + 4] * (float)q3;
                y[l + 96u] = d * (float)sc[is + 6] * (float)q4;
            }
            y += 128u; ql += 64u; qh += 32u; sc += 8;
        }
    }
}

int rl_quant_dequant_row(uint32_t ggml_type, const uint8_t *row, uint32_t ncols,
                         const uint8_t *iq2_xxs_grid, float *out) {
    if (!row || !out || !ncols) return 0;
    switch (ggml_type) {
        case 0: memcpy(out, row, (size_t)ncols * sizeof(float)); return 1;
        case 1: for (uint32_t i = 0; i < ncols; ++i) out[i] = rl_quant_f16_to_f32(rd16(row + 2u * i)); return 1;
        case 8: if (ncols % 32u) return 0; dequant_q8_0(row, ncols, out); return 1;
        case 10: if (ncols % QK_K) return 0; dequant_q2_k(row, ncols, out); return 1;
        case 12: if (ncols % QK_K) return 0; dequant_q4_k(row, ncols, out); return 1;
        case 13: if (ncols % QK_K) return 0; dequant_q5_k(row, ncols, out); return 1;
        case 14: if (ncols % QK_K) return 0; dequant_q6_k(row, ncols, out); return 1;
        case 16: {
            /* Dequantize by probing the validated IQ2_XXS dot with unit vectors is too slow;
             * decode directly following the pinned reference. */
            if (ncols % QK_K || !iq2_xxs_grid) return 0;
            for (uint32_t ib = 0; ib < ncols / QK_K; ++ib) {
                const uint8_t *bp = row + (size_t)ib * IQ2_XXS_BLOCK;
                const float d = rl_quant_f16_to_f32(rd16(bp));
                const uint8_t *q = bp + 2u;
                float *y = out + (size_t)ib * QK_K;
                for (uint32_t g = 0; g < 8u; ++g) {
                    const uint8_t *qp = q + g * 8u;
                    const uint32_t auxg = (uint32_t)qp[0] | ((uint32_t)qp[1] << 8) | ((uint32_t)qp[2] << 16) | ((uint32_t)qp[3] << 24);
                    const uint32_t auxs = (uint32_t)qp[4] | ((uint32_t)qp[5] << 8) | ((uint32_t)qp[6] << 16) | ((uint32_t)qp[7] << 24);
                    const float db = d * (0.5f + (float)(auxs >> 28)) * 0.25f;
                    for (uint32_t l = 0; l < 4u; ++l) {
                        const uint32_t gi = (auxg >> (8u * l)) & 255u;
                        const uint32_t s7 = (auxs >> (7u * l)) & 127u;
                        const uint32_t s8 = s7 | ((__builtin_popcount(s7) & 1u) << 7);
                        const uint8_t *gv = iq2_xxs_grid + gi * 8u;
                        for (uint32_t j = 0; j < 8u; ++j)
                            y[g * 32u + l * 8u + j] = db * (float)gv[j] * ((s8 & (1u << j)) ? -1.0f : 1.0f);
                    }
                }
            }
            return 1;
        }
        default:
            if (rl_iq3_supported(ggml_type)) return rl_iq3_dequant_row(ggml_type, row, ncols, out);   /* dev31 */
            return 0;
    }
}

int rl_quant_row_dot(uint32_t ggml_type, const uint8_t *row, const float *x, uint32_t ncols,
                     const uint8_t *iq2_xxs_grid, double *out) {
    if (!row || !x || !out || !ncols) return 0;
    double acc = 0.0;
    switch (ggml_type) {
        case 0: { const float *w = (const float *)row; for (uint32_t i = 0; i < ncols; ++i) acc += (double)w[i] * x[i]; *out = acc; return 1; }
        case 1: for (uint32_t i = 0; i < ncols; ++i) acc += (double)rl_quant_f16_to_f32(rd16(row + 2u * i)) * x[i]; *out = acc; return 1;
        case 8: {
            if (ncols % 32u) return 0;
            for (uint32_t ib = 0; ib < ncols / 32u; ++ib) {
                const uint8_t *bp = row + (size_t)ib * Q8_0_BLOCK;
                const double d = (double)rl_quant_f16_to_f32(rd16(bp));
                const int8_t *q = (const int8_t *)(bp + 2u);
                for (uint32_t j = 0; j < 32u; ++j) acc += (double)x[ib * 32u + j] * d * (double)q[j];
            }
            *out = acc; return 1;
        }
        case 12: {
            if (ncols % QK_K) return 0;
            for (uint32_t ib = 0; ib < ncols / QK_K; ++ib) {
                const uint8_t *bp = row + (size_t)ib * Q4_K_BLOCK;
                const double d = rl_quant_f16_to_f32(rd16(bp));
                const double dmin = rl_quant_f16_to_f32(rd16(bp + 2u));
                const uint8_t *scales = bp + 4u, *qs = bp + 16u;
                for (uint32_t g = 0; g < 8u; ++g) {
                    uint8_t sc = 0, mn = 0;
                    get_scale_min_k4(g, scales, &sc, &mn);
                    const double ds = d * sc, dm = dmin * mn;
                    const uint8_t *q = qs + (g / 2u) * 32u;
                    const uint32_t xb = ib * QK_K + g * 32u;
                    for (uint32_t l = 0; l < 32u; ++l) {
                        const uint8_t quant = (g & 1u) ? (q[l] >> 4) : (q[l] & 0x0fu);
                        acc += (double)x[xb + l] * (ds * quant - dm);
                    }
                }
            }
            *out = acc; return 1;
        }
        case 13: {
            if (ncols % QK_K) return 0;
            for (uint32_t ib = 0; ib < ncols / QK_K; ++ib) {
                const uint8_t *bp = row + (size_t)ib * Q5_K_BLOCK;
                const double d = rl_quant_f16_to_f32(rd16(bp));
                const double dmin = rl_quant_f16_to_f32(rd16(bp + 2u));
                const uint8_t *scales = bp + 4u, *qh = bp + 16u, *ql = bp + 48u;
                uint32_t is = 0; uint8_t u1 = 1u, u2 = 2u;
                const float *xb = x + (size_t)ib * QK_K;
                for (uint32_t j = 0; j < QK_K; j += 64u) {
                    uint8_t sc = 0, m = 0;
                    get_scale_min_k4(is + 0u, scales, &sc, &m);
                    const double d1 = d * sc, m1 = dmin * m;
                    get_scale_min_k4(is + 1u, scales, &sc, &m);
                    const double d2 = d * sc, m2 = dmin * m;
                    for (uint32_t l = 0; l < 32u; ++l) acc += (double)xb[j + l] * (d1 * (double)((ql[l] & 0xFu) + ((qh[l] & u1) ? 16u : 0u)) - m1);
                    for (uint32_t l = 0; l < 32u; ++l) acc += (double)xb[j + 32u + l] * (d2 * (double)((ql[l] >> 4) + ((qh[l] & u2) ? 16u : 0u)) - m2);
                    ql += 32u; is += 2u; u1 <<= 2; u2 <<= 2;
                }
            }
            *out = acc; return 1;
        }
        case 10: {
            if (ncols % QK_K) return 0;
            float tmp[QK_K];
            for (uint32_t ib = 0; ib < ncols / QK_K; ++ib) {
                dequant_q2_k(row + (size_t)ib * Q2_K_BLOCK, QK_K, tmp);
                for (uint32_t l = 0; l < QK_K; ++l) acc += (double)tmp[l] * x[ib * QK_K + l];
            }
            *out = acc; return 1;
        }
        case 14:
        case 16: {
            /* Reuse the field-validated shared-expert reference for Q6_K / IQ2_XXS. */
            const size_t rb = rl_native_shared_row_bytes(ggml_type, ncols);
            if (!rb) return 0;
            char err[128];
            return rl_native_shared_quant_row_dot(row, rb, ggml_type, x, ncols, iq2_xxs_grid,
                iq2_xxs_grid ? RL_IQ2_XXS_GRID_COUNT : 0u, out, err, sizeof(err));
        }
        default:
            if (rl_iq3_supported(ggml_type)) return rl_iq3_row_dot(ggml_type, row, x, ncols, out);   /* dev31 */
            return 0;
    }
}

/* ---- self-test ---- */

static uint16_t f32_to_f16_bits(float value) {
    /* exact for the small round values used below */
    uint32_t bits; memcpy(&bits, &value, 4);
    const uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = (int32_t)((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = bits & 0x7fffffu;
    if (exp <= 0) return (uint16_t)sign;
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

int rl_quant_selftest(char *error, size_t cap) {
    float x[QK_K];
    for (uint32_t i = 0; i < QK_K; ++i) x[i] = 1.0f;
    double dot = 0.0;

    /* Q8_0: d=1, all q=3 over 256 values -> 768 */
    {
        uint8_t row[8u * Q8_0_BLOCK];
        for (uint32_t ib = 0; ib < 8u; ++ib) {
            uint8_t *bp = row + ib * Q8_0_BLOCK;
            bp[0] = 0x00; bp[1] = 0x3c; /* 1.0 */
            memset(bp + 2u, 3, 32u);
        }
        if (!rl_quant_row_dot(8u, row, x, QK_K, NULL, &dot) || fabs(dot - 768.0) > 1e-9) {
            snprintf(error, cap, "Q8_0 selftest failed: %.9g", dot); return 0;
        }
    }
    /* Q4_K: d=1, dmin=0, scales=1 for all 8 groups, quant nibbles 0x11 -> 256 */
    {
        uint8_t block[Q4_K_BLOCK];
        memset(block, 0, sizeof(block));
        block[0] = 0x00; block[1] = 0x3c;
        for (uint32_t j = 0; j < 4u; ++j) block[4u + j] = 1u;
        for (uint32_t j = 8u; j < 12u; ++j) block[4u + j] = 1u;
        memset(block + 16u, 0x11, 128u);
        if (!rl_quant_row_dot(12u, block, x, QK_K, NULL, &dot) || fabs(dot - 256.0) > 1e-9) {
            snprintf(error, cap, "Q4_K selftest failed: %.9g", dot); return 0;
        }
    }
    /* Q5_K: d=0.5, dmin=1, scales sc=2 (groups 0..7), mins m=1, ql nibbles 0x21 (low group 1, high group 2), qh all 0xFF (+16 both)
       value low  = 0.5*2*(1+16) - 1*1 = 16
       value high = 0.5*2*(2+16) - 1 = 17
       sum over 4 x (32 low + 32 high) = 4*(32*16 + 32*17) = 4*(512+544) = 4224 */
    {
        uint8_t block[Q5_K_BLOCK];
        memset(block, 0, sizeof(block));
        const uint16_t half = f32_to_f16_bits(0.5f), one = f32_to_f16_bits(1.0f);
        block[0] = (uint8_t)(half & 0xff); block[1] = (uint8_t)(half >> 8);
        block[2] = (uint8_t)(one & 0xff); block[3] = (uint8_t)(one >> 8);
        for (uint32_t j = 0; j < 4u; ++j) { block[4u + j] = 2u; block[4u + 4u + j] = 1u; }
        /* groups 4..7: d = (q[j+4]&0xF) | ((q[j-4]>>6)<<4); m = (q[j+4]>>4) | ((q[j]>>6)<<4) -> q[8..11] = 0x12 gives d=2, m=1 */
        for (uint32_t j = 8u; j < 12u; ++j) block[4u + j] = 0x12u;
        memset(block + 16u, 0xFF, 32u);
        memset(block + 48u, 0x21, 128u);
        float y[QK_K];
        if (!rl_quant_dequant_row(13u, block, QK_K, NULL, y)) { snprintf(error, cap, "Q5_K dequant failed"); return 0; }
        for (uint32_t i = 0; i < QK_K; ++i) {
            const float expect = ((i / 32u) & 1u) ? 17.0f : 16.0f;
            if (fabsf(y[i] - expect) > 1e-6f) { snprintf(error, cap, "Q5_K dequant mismatch at %u: %g", i, y[i]); return 0; }
        }
        if (!rl_quant_row_dot(13u, block, x, QK_K, NULL, &dot) || fabs(dot - 4224.0) > 1e-9) {
            snprintf(error, cap, "Q5_K selftest failed: %.9g", dot); return 0;
        }
    }
    /* Q2_K: d=1, dmin=0.5, scales byte 0x21 (scale 1, min 2) for all 16 sub-blocks, q bits pattern 0b10 (=2) for shift 2 only:
       we set qs = 0x08 -> bits: shift0:0, shift2:2, shift4:0, shift6:0.
       Per 128 chunk, 4 shifts x 2 halves x 16 values: value = 1*q - 0.5*2 = q - 1 -> shift2 gives 1, others -1.
       Sum per block: 2 chunks * (16*2 values at +1 [shift2] + 3*32 values at -1) = 2*(32 - 96) = -128 */
    {
        uint8_t block[Q2_K_BLOCK];
        memset(block, 0, sizeof(block));
        memset(block, 0x21, 16u);
        memset(block + 16u, 0x08, 64u);
        const uint16_t one = f32_to_f16_bits(1.0f), half = f32_to_f16_bits(0.5f);
        block[80] = (uint8_t)(one & 0xff); block[81] = (uint8_t)(one >> 8);
        block[82] = (uint8_t)(half & 0xff); block[83] = (uint8_t)(half >> 8);
        if (!rl_quant_row_dot(10u, block, x, QK_K, NULL, &dot) || fabs(dot + 128.0) > 1e-9) {
            snprintf(error, cap, "Q2_K selftest failed: %.9g", dot); return 0;
        }
    }
    /* Q6_K: d=1, scales all 1, ql=0x00, qh=0x00 -> q = 0 - 32 = -32 each -> -32*256 = -8192 */
    {
        uint8_t block[Q6_K_BLOCK];
        memset(block, 0, sizeof(block));
        for (uint32_t i = 0; i < 16u; ++i) block[192u + i] = 1u;
        block[208] = 0x00; block[209] = 0x3c;
        float y[QK_K];
        if (!rl_quant_dequant_row(14u, block, QK_K, NULL, y)) { snprintf(error, cap, "Q6_K dequant failed"); return 0; }
        for (uint32_t i = 0; i < QK_K; ++i) if (fabsf(y[i] + 32.0f) > 1e-6f) { snprintf(error, cap, "Q6_K dequant mismatch at %u: %g", i, y[i]); return 0; }
        if (!rl_quant_row_dot(14u, block, x, QK_K, NULL, &dot) || fabs(dot + 8192.0) > 1e-6) {
            snprintf(error, cap, "Q6_K selftest failed: %.9g", dot); return 0;
        }
    }
    /* IQ2_XXS: dequant vs validated dot on a block with d=1, grid index 0 for all, no signs, scale nibble 0 -> db = 0.125 */
    {
        uint8_t grid[RL_IQ2_XXS_GRID_COUNT];
        char err[128];
        if (!rl_native_iq2_xxs_build_grid(grid, err, sizeof(err))) { snprintf(error, cap, "IQ2_XXS grid: %s", err); return 0; }
        uint8_t block[IQ2_XXS_BLOCK];
        memset(block, 0, sizeof(block));
        block[0] = 0x00; block[1] = 0x3c;
        float y[QK_K];
        double ref = 0.0;
        if (!rl_quant_dequant_row(16u, block, QK_K, grid, y) || !rl_quant_row_dot(16u, block, x, QK_K, grid, &ref)) {
            snprintf(error, cap, "IQ2_XXS selftest failed"); return 0;
        }
        double mine = 0.0;
        for (uint32_t i = 0; i < QK_K; ++i) mine += y[i];
        if (fabs(mine - ref) > 1e-4) { snprintf(error, cap, "IQ2_XXS dequant/dot mismatch: %.9g vs %.9g", mine, ref); return 0; }
    }
    if (error && cap) error[0] = '\0';
    return 1;
}
