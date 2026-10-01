#include "redlite_native_iq3.h"
#include "redlite_native_iq3_tables.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define QK_K 256u

static const uint8_t kmask_iq2xs[8] = {1, 2, 4, 8, 16, 32, 64, 128};

static uint16_t rd16(const uint8_t *p) { return (uint16_t)p[0] | ((uint16_t)p[1] << 8); }

/* IEEE half -> float (exact), as ggml's GGML_FP16_TO_FP32 */
static float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu, mant = h & 0x03ffu, bits;
    if (exp == 0) {
        if (!mant) bits = sign;
        else { int shift = 0; while ((mant & 0x0400u) == 0) { mant <<= 1; ++shift; } mant &= 0x03ffu; bits = sign | ((uint32_t)(127 - 14 - shift) << 23) | (mant << 13); }
    } else if (exp == 31u) bits = sign | 0x7f800000u | (mant << 13);
    else bits = sign | ((exp + 112u) << 23) | (mant << 13);
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}
static uint32_t rd32(const uint8_t *p) { return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24); }
/* ksigns_iq2xs[i] of ggml-common.h: the 7 stored sign bits plus an odd-parity eighth bit */
static uint8_t ksign(uint32_t s7) { return (uint8_t)(s7 | ((uint32_t)(__builtin_popcount(s7) & 1) << 7)); }

int rl_iq3_group8_supported(uint32_t t) { return rl_iq3_supported(t) || t == 12u; }

int rl_iq3_supported(uint32_t t) { return t == RL_GGML_IQ3_XXS || t == RL_GGML_IQ3_S || t == RL_GGML_IQ2_S || t == RL_GGML_IQ4_XS; }

uint32_t rl_iq3_block_bytes(uint32_t t) {
    switch (t) {
        case RL_GGML_IQ3_XXS: return 98u;   /* d, qs[64], scales_and_signs[32] */
        case RL_GGML_IQ3_S: return 110u;    /* d, qs[64], qh[8], signs[32], scales[4] */
        case RL_GGML_IQ2_S: return 82u;     /* d, qs[64] (32 indices + 32 signs), qh[8], scales[8] */
        case RL_GGML_IQ4_XS: return 136u;   /* d, scales_h, scales_l[4], qs[128] */
        case 12u: return 144u;              /* Q4_K (dev36: routed-expert decoder only, see rl_iq3_group8_supported) */
        default: return 0u;
    }
}

size_t rl_iq3_row_bytes(uint32_t t, uint32_t ncols) {
    const uint32_t b = rl_iq3_block_bytes(t);
    if (!b || !ncols || ncols % QK_K) return 0u;
    return (size_t)(ncols / QK_K) * b;
}

void rl_iq3_group8(uint32_t t, const uint8_t *bp, uint32_t g, float out[8]) {
    const float d = f16_to_f32(rd16(bp));
    const uint32_t ib32 = g / 4u, l = g % 4u;
    if (t == 12u) {   /* Q4_K, as ggml dequantize_row_q4_K: sub-block ib32 has scale/min pair ib32 */
        const uint8_t *sc = bp + 4u;
        uint8_t s6, m6;
        if (ib32 < 4u) { s6 = sc[ib32] & 63u; m6 = sc[ib32 + 4u] & 63u; }
        else { s6 = (uint8_t)((sc[ib32 + 4u] & 0x0fu) | ((sc[ib32 - 4u] >> 6) << 4)); m6 = (uint8_t)((sc[ib32 + 4u] >> 4) | ((sc[ib32] >> 6) << 4)); }
        const float d1 = d * (float)s6, m1 = f16_to_f32(rd16(bp + 2u)) * (float)m6;
        const uint8_t *q = bp + 16u + (ib32 / 2u) * 32u + l * 8u;
        for (uint32_t j = 0; j < 8u; ++j) out[j] = d1 * (float)((ib32 & 1u) ? (q[j] >> 4) : (q[j] & 0x0fu)) - m1;
        return;
    }
    if (t == RL_GGML_IQ3_XXS) {
        const uint8_t *qs = bp + 2u + 8u * ib32;
        const uint32_t aux32 = rd32(bp + 2u + 64u + 4u * ib32);
        const float db = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;
        const uint8_t signs = ksign((aux32 >> (7u * l)) & 127u);
        const uint32_t g1 = rl_iq3xxs_grid[qs[2u * l + 0u]], g2 = rl_iq3xxs_grid[qs[2u * l + 1u]];
        for (uint32_t j = 0; j < 4u; ++j) {
            out[j + 0u] = db * (float)((g1 >> (8u * j)) & 255u) * ((signs & kmask_iq2xs[j + 0u]) ? -1.f : 1.f);
            out[j + 4u] = db * (float)((g2 >> (8u * j)) & 255u) * ((signs & kmask_iq2xs[j + 4u]) ? -1.f : 1.f);
        }
    } else if (t == RL_GGML_IQ3_S) {
        const uint8_t *qs = bp + 2u + 8u * ib32, *qh = bp + 66u, *signs = bp + 74u + 4u * ib32, *scales = bp + 106u;
        const float db = d * (float)(1 + 2 * ((scales[ib32 / 2u] >> (4u * (ib32 % 2u))) & 0xf));
        const uint32_t i1 = qs[2u * l + 0u] | ((qh[ib32] << (8u - 2u * l)) & 256u);
        const uint32_t i2 = qs[2u * l + 1u] | ((qh[ib32] << (7u - 2u * l)) & 256u);
        const uint32_t g1 = rl_iq3s_grid[i1], g2 = rl_iq3s_grid[i2];
        for (uint32_t j = 0; j < 4u; ++j) {
            out[j + 0u] = db * (float)((g1 >> (8u * j)) & 255u) * ((signs[l] & kmask_iq2xs[j + 0u]) ? -1.f : 1.f);
            out[j + 4u] = db * (float)((g2 >> (8u * j)) & 255u) * ((signs[l] & kmask_iq2xs[j + 4u]) ? -1.f : 1.f);
        }
    } else if (t == RL_GGML_IQ2_S) {
        const uint8_t *qs = bp + 2u, *signs = bp + 2u + 32u, *qh = bp + 66u, *scales = bp + 74u;
        const float dl = d * (0.5f + (float)((l < 2u ? scales[ib32] : (uint8_t)(scales[ib32] >> 4)) & 0xf)) * 0.25f;
        const uint32_t idx = qs[4u * ib32 + l] | ((qh[ib32] << (8u - 2u * l)) & 0x300u);
        const uint64_t gv = rl_iq2s_grid[idx];
        const uint8_t s = signs[4u * ib32 + l];
        for (uint32_t j = 0; j < 8u; ++j) out[j] = dl * (float)((gv >> (8u * j)) & 255u) * ((s & kmask_iq2xs[j]) ? -1.f : 1.f);
    } else if (t == RL_GGML_IQ4_XS) {
        const uint32_t scales_h = rd16(bp + 2u);
        const uint8_t *scales_l = bp + 4u, *qs = bp + 8u + 16u * ib32;
        const int ls = ((scales_l[ib32 / 2u] >> (4u * (ib32 % 2u))) & 0xf) | (int)(((scales_h >> (2u * ib32)) & 3u) << 4);
        const float dl = d * (float)(ls - 32);
        const uint32_t base = 8u * (l % 2u);
        for (uint32_t j = 0; j < 8u; ++j)
            out[j] = dl * (float)rl_kvalues_iq4nl[l < 2u ? (qs[base + j] & 0xf) : (qs[base + j] >> 4)];
    } else {
        memset(out, 0, 8u * sizeof(float));
    }
}

int rl_iq3_dequant_row(uint32_t t, const uint8_t *row, uint32_t ncols, float *out) {
    const uint32_t bb = rl_iq3_block_bytes(t);
    if (!row || !out || !bb || !ncols || ncols % QK_K) return 0;
    for (uint32_t b = 0; b < ncols / QK_K; ++b)
        for (uint32_t g = 0; g < 32u; ++g) rl_iq3_group8(t, row + (size_t)b * bb, g, out + (size_t)b * QK_K + 8u * g);
    return 1;
}

int rl_iq3_row_dot(uint32_t t, const uint8_t *row, const float *x, uint32_t ncols, double *out) {
    const uint32_t bb = rl_iq3_block_bytes(t);
    if (!row || !x || !out || !bb || !ncols || ncols % QK_K) return 0;
    double acc = 0.0;
    float v[8];
    for (uint32_t b = 0; b < ncols / QK_K; ++b)
        for (uint32_t g = 0; g < 32u; ++g) {
            rl_iq3_group8(t, row + (size_t)b * bb, g, v);
            const float *xc = x + (size_t)b * QK_K + 8u * g;
            for (uint32_t j = 0; j < 8u; ++j) acc += (double)v[j] * (double)xc[j];
        }
    *out = acc;
    return 1;
}

int rl_iq3_row_dot_d(uint32_t t, const uint8_t *row, const double *x, uint32_t ncols, double *out) {
    const uint32_t bb = rl_iq3_block_bytes(t);
    if (!row || !x || !out || !bb || !ncols || ncols % QK_K) return 0;
    double acc = 0.0;
    float v[8];
    for (uint32_t b = 0; b < ncols / QK_K; ++b)
        for (uint32_t g = 0; g < 32u; ++g) {
            rl_iq3_group8(t, row + (size_t)b * bb, g, v);
            const double *xc = x + (size_t)b * QK_K + 8u * g;
            for (uint32_t j = 0; j < 8u; ++j) acc += (double)v[j] * xc[j];
        }
    *out = acc;
    return 1;
}

/* ---- Metal: codebooks as constant arrays + the same decoder over device memory ---- */

static int append(char **buf, size_t *len, size_t *cap, const char *s) {
    const size_t n = strlen(s);
    if (*len + n + 1u > *cap) {
        size_t c = *cap ? *cap : 65536u;
        while (c < *len + n + 1u) c *= 2u;
        char *p = (char *)realloc(*buf, c);
        if (!p) return 0;
        *buf = p; *cap = c;
    }
    memcpy(*buf + *len, s, n + 1u);
    *len += n;
    return 1;
}

char *rl_iq3_metal_source(void) {
    char *buf = NULL, tmp[64];
    size_t len = 0, cap = 0;
    int ok = append(&buf, &len, &cap, "constant uint rl_iq3xxs_grid[256] = {");
    for (int i = 0; i < 256 && ok; ++i) { snprintf(tmp, sizeof(tmp), "%s0x%08xu", i ? "," : "", (unsigned)rl_iq3xxs_grid[i]); ok = append(&buf, &len, &cap, tmp); }
    ok = ok && append(&buf, &len, &cap, "};\nconstant uint rl_iq3s_grid[512] = {");
    for (int i = 0; i < 512 && ok; ++i) { snprintf(tmp, sizeof(tmp), "%s0x%08xu", i ? "," : "", (unsigned)rl_iq3s_grid[i]); ok = append(&buf, &len, &cap, tmp); }
    ok = ok && append(&buf, &len, &cap, "};\nconstant ulong rl_iq2s_grid[1024] = {");
    for (int i = 0; i < 1024 && ok; ++i) { snprintf(tmp, sizeof(tmp), "%s0x%016llxul", i ? "," : "", (unsigned long long)rl_iq2s_grid[i]); ok = append(&buf, &len, &cap, tmp); }
    ok = ok && append(&buf, &len, &cap, "};\nconstant float rl_kvalues_iq4nl[16] = {");
    for (int i = 0; i < 16 && ok; ++i) { snprintf(tmp, sizeof(tmp), "%s%d.0f", i ? "," : "", (int)rl_kvalues_iq4nl[i]); ok = append(&buf, &len, &cap, tmp); }
    ok = ok && append(&buf, &len, &cap, "};\n");
    ok = ok && append(&buf, &len, &cap,
        /* same arithmetic as rl_iq3_group8 (redlite_native_iq3.c) */
        "static inline ushort rl3_rd16(device const uchar *p) { return ushort(p[0]) | (ushort(p[1]) << 8); }\n"
        "static inline uint rl3_rd32(device const uchar *p) { return uint(p[0]) | (uint(p[1]) << 8) | (uint(p[2]) << 16) | (uint(p[3]) << 24); }\n"
        "static inline float rl3_sgn(uint s, uint j) { return (s & (1u << j)) ? -1.0f : 1.0f; }\n"
        "static inline void rl_iq3_group8(uint t, device const uchar *bp, uint g, thread float4 &v0, thread float4 &v1) {\n"
        "    const float d = float(as_type<half>(rl3_rd16(bp))); const uint ib32 = g >> 2; const uint l = g & 3u; float v[8];\n"
        "    if (t == 18u) {\n"
        "        device const uchar *qs = bp + 2u + 8u * ib32; const uint aux32 = rl3_rd32(bp + 66u + 4u * ib32);\n"
        "        const float db = d * (0.5f + float(aux32 >> 28)) * 0.5f;\n"
        "        const uint s7 = (aux32 >> (7u * l)) & 127u; const uint s = s7 | ((popcount(s7) & 1u) << 7);\n"
        "        const uint g1 = rl_iq3xxs_grid[qs[2u * l]], g2 = rl_iq3xxs_grid[qs[2u * l + 1u]];\n"
        "        for (uint j = 0; j < 4u; ++j) { v[j] = db * float((g1 >> (8u * j)) & 255u) * rl3_sgn(s, j); v[j + 4u] = db * float((g2 >> (8u * j)) & 255u) * rl3_sgn(s, j + 4u); }\n"
        "    } else if (t == 21u) {\n"
        "        device const uchar *qs = bp + 2u + 8u * ib32; device const uchar *qh = bp + 66u; const uint s = (bp + 74u + 4u * ib32)[l];\n"
        "        const float db = d * float(1 + 2 * int(((bp + 106u)[ib32 >> 1] >> (4u * (ib32 & 1u))) & 15u));\n"
        "        const uint i1 = uint(qs[2u * l]) | ((uint(qh[ib32]) << (8u - 2u * l)) & 256u);\n"
        "        const uint i2 = uint(qs[2u * l + 1u]) | ((uint(qh[ib32]) << (7u - 2u * l)) & 256u);\n"
        "        const uint g1 = rl_iq3s_grid[i1], g2 = rl_iq3s_grid[i2];\n"
        "        for (uint j = 0; j < 4u; ++j) { v[j] = db * float((g1 >> (8u * j)) & 255u) * rl3_sgn(s, j); v[j + 4u] = db * float((g2 >> (8u * j)) & 255u) * rl3_sgn(s, j + 4u); }\n"
        "    } else if (t == 22u) {\n"
        "        const uint sc = (bp + 74u)[ib32]; const float dl = d * (0.5f + float((l < 2u ? sc : (sc >> 4)) & 15u)) * 0.25f;\n"
        "        const uint idx = uint((bp + 2u)[4u * ib32 + l]) | ((uint((bp + 66u)[ib32]) << (8u - 2u * l)) & 0x300u);\n"
        "        const ulong gv = rl_iq2s_grid[idx]; const uint s = (bp + 34u)[4u * ib32 + l];\n"
        "        for (uint j = 0; j < 8u; ++j) v[j] = dl * float(uint(gv >> (8u * j)) & 255u) * rl3_sgn(s, j);\n"
        "    } else if (t == 12u) {\n"
        "        device const uchar *sc = bp + 4u; uint s6, m6;\n"
        "        if (ib32 < 4u) { s6 = sc[ib32] & 63u; m6 = sc[ib32 + 4u] & 63u; }\n"
        "        else { s6 = (sc[ib32 + 4u] & 15u) | ((sc[ib32 - 4u] >> 6) << 4); m6 = (sc[ib32 + 4u] >> 4) | ((sc[ib32] >> 6) << 4); }\n"
        "        const float d1 = d * float(s6), m1 = float(as_type<half>(rl3_rd16(bp + 2u))) * float(m6);\n"
        "        device const uchar *q = bp + 16u + (ib32 >> 1) * 32u + l * 8u;\n"
        "        for (uint j = 0; j < 8u; ++j) v[j] = d1 * float((ib32 & 1u) ? (q[j] >> 4) : (q[j] & 15u)) - m1;\n"
        "    } else {\n"
        "        const uint sh = rl3_rd16(bp + 2u); device const uchar *qs = bp + 8u + 16u * ib32;\n"
        "        const int ls = int(((bp + 4u)[ib32 >> 1] >> (4u * (ib32 & 1u))) & 15u) | int(((sh >> (2u * ib32)) & 3u) << 4);\n"
        "        const float dl = d * float(ls - 32); const uint base = 8u * (l & 1u);\n"
        "        for (uint j = 0; j < 8u; ++j) v[j] = dl * rl_kvalues_iq4nl[l < 2u ? (qs[base + j] & 15u) : (qs[base + j] >> 4)];\n"
        "    }\n"
        "    v0 = float4(v[0], v[1], v[2], v[3]); v1 = float4(v[4], v[5], v[6], v[7]);\n"
        "}\n"
        );
    ok = ok && append(&buf, &len, &cap,
        /* dev33: rl_iq3_group8 with 16-bit loads and uchar4 unpacking; per value the same product db * grid * sign */
        "static inline float4 rl3_sg4f(uint s, uint shift) { return select(float4(1.0f), float4(-1.0f), ((uint4(s) >> (uint4(0u, 1u, 2u, 3u) + shift)) & 1u) != 0u); }\n"
        "static inline void rl_iq3_group8f(uint t, device const uchar *bp, uint g, thread float4 &v0, thread float4 &v1) {\n"
        "    const float d = float(as_type<half>(*(device const ushort *)bp)); const uint sb = g >> 2; const uint l = g & 3u;\n"
        "    if (t == 18u) {\n"
        "        device const ushort *a16 = (device const ushort *)(bp + 66u + 4u * sb); const uint aux32 = uint(a16[0]) | (uint(a16[1]) << 16);\n"
        "        const uint qq = ((device const ushort *)(bp + 2u + 8u * sb))[l];\n"
        "        const float db = d * (0.5f + float(aux32 >> 28)) * 0.5f;\n"
        "        const uint s7 = (aux32 >> (7u * l)) & 127u; const uint sg = s7 | ((popcount(s7) & 1u) << 7);\n"
        "        v0 = db * float4(as_type<uchar4>(rl_iq3xxs_grid[qq & 255u])) * rl3_sg4f(sg, 0u);\n"
        "        v1 = db * float4(as_type<uchar4>(rl_iq3xxs_grid[qq >> 8])) * rl3_sg4f(sg, 4u);\n"
        "        return;\n"
        "    }\n"
        "    if (t == 21u) {\n"
        "        const uint qq = ((device const ushort *)(bp + 2u + 8u * sb))[l]; const uint qh = (bp + 66u)[sb]; const uint sg = (bp + 74u + 4u * sb)[l];\n"
        "        const float db = d * float(1 + 2 * int(((bp + 106u)[sb >> 1] >> (4u * (sb & 1u))) & 15u));\n"
        "        v0 = db * float4(as_type<uchar4>(rl_iq3s_grid[(qq & 255u) | ((qh << (8u - 2u * l)) & 256u)])) * rl3_sg4f(sg, 0u);\n"
        "        v1 = db * float4(as_type<uchar4>(rl_iq3s_grid[(qq >> 8) | ((qh << (7u - 2u * l)) & 256u)])) * rl3_sg4f(sg, 4u);\n"
        "        return;\n"
        "    }\n"
        "    rl_iq3_group8(t, bp, g, v0, v1);\n"
        "}\n"
        );
    ok = ok && append(&buf, &len, &cap,
        /* dev33: sum over the 32 values of sub-block sb (0..7) of one block times x[0..31]: the scale and sign words are
         * decoded once per sub-block and the codebook entries unpacked as uchar4 (same values as rl_iq3_group8; the
         * block scale is applied to the sub-block sum, the order of the dev22 IQ2_XXS sub-block kernel) */
        "static inline float4 rl3_sg4(uint s, uint shift) { return select(float4(1.0f), float4(-1.0f), ((uint4(s) >> (uint4(0u, 1u, 2u, 3u) + shift)) & 1u) != 0u); }\n"
        "static inline float rl_iq3_dot32(uint t, device const uchar *bp, uint sb, device const float *x) {\n"
        "    const float d = float(as_type<half>(rl3_rd16(bp))); float acc = 0.0f;\n"
        "    device const float4 *x4 = (device const float4 *)x;\n"
        "    if (t == 18u) {\n"
        "        device const ushort *q16 = (device const ushort *)(bp + 2u + 8u * sb); device const ushort *a16 = (device const ushort *)(bp + 66u + 4u * sb);\n"
        "        const uint aux32 = uint(a16[0]) | (uint(a16[1]) << 16);\n"
        "        for (uint l = 0; l < 4u; ++l) {\n"
        "            const uint qq = q16[l]; const uint s7 = (aux32 >> (7u * l)) & 127u; const uint sg = s7 | ((popcount(s7) & 1u) << 7);\n"
        "            const float4 w1 = float4(as_type<uchar4>(rl_iq3xxs_grid[qq & 255u])) * rl3_sg4(sg, 0u);\n"
        "            const float4 w2 = float4(as_type<uchar4>(rl_iq3xxs_grid[qq >> 8])) * rl3_sg4(sg, 4u);\n"
        "            acc += dot(x4[2u * l], w1) + dot(x4[2u * l + 1u], w2);\n"
        "        }\n"
        "        return d * (0.5f + float(aux32 >> 28)) * 0.5f * acc;\n"
        "    }\n"
        "    if (t == 21u) {\n"
        "        device const ushort *q16 = (device const ushort *)(bp + 2u + 8u * sb); const uint qh = (bp + 66u)[sb];\n"
        "        device const uchar *sgn = bp + 74u + 4u * sb;\n"
        "        for (uint l = 0; l < 4u; ++l) {\n"
        "            const uint qq = q16[l]; const uint sg = sgn[l];\n"
        "            const uint i1 = (qq & 255u) | ((qh << (8u - 2u * l)) & 256u), i2 = (qq >> 8) | ((qh << (7u - 2u * l)) & 256u);\n"
        "            const float4 w1 = float4(as_type<uchar4>(rl_iq3s_grid[i1])) * rl3_sg4(sg, 0u);\n"
        "            const float4 w2 = float4(as_type<uchar4>(rl_iq3s_grid[i2])) * rl3_sg4(sg, 4u);\n"
        "            acc += dot(x4[2u * l], w1) + dot(x4[2u * l + 1u], w2);\n"
        "        }\n"
        "        return d * float(1 + 2 * int(((bp + 106u)[sb >> 1] >> (4u * (sb & 1u))) & 15u)) * acc;\n"
        "    }\n"
        "    if (t == 12u) {   /* Q4_K sub-block: d*sc * sum(q*x) - dmin*m * sum(x) */\n"
        "        device const uchar *sc = bp + 4u; uint s6, m6;\n"
        "        if (sb < 4u) { s6 = sc[sb] & 63u; m6 = sc[sb + 4u] & 63u; }\n"
        "        else { s6 = (sc[sb + 4u] & 15u) | ((sc[sb - 4u] >> 6) << 4); m6 = (sc[sb + 4u] >> 4) | ((sc[sb] >> 6) << 4); }\n"
        "        device const uchar *q = bp + 16u + (sb >> 1) * 32u; const uint sh = (sb & 1u) * 4u; float xs = 0.0f;\n"
        "        for (uint i = 0; i < 8u; ++i) { const float4 xv = x4[i]; const uchar4 qb = uchar4(q[4u*i], q[4u*i+1u], q[4u*i+2u], q[4u*i+3u]);\n"
        "            acc += dot(xv, float4((uint4(qb) >> sh) & 15u)); xs += (xv.x + xv.y) + (xv.z + xv.w); }\n"
        "        return d * float(s6) * acc - float(as_type<half>(rl3_rd16(bp + 2u))) * float(m6) * xs;\n"
        "    }\n"
        );
    ok = ok && append(&buf, &len, &cap,
        "    if (t == 22u) {\n"
        "        const uint sc = (bp + 74u)[sb]; const uint qh = (bp + 66u)[sb]; float a2[2] = {0.0f, 0.0f};\n"
        "        for (uint l = 0; l < 4u; ++l) {\n"
        "            const uint idx = uint((bp + 2u)[4u * sb + l]) | ((qh << (8u - 2u * l)) & 0x300u); const uint sg = (bp + 34u)[4u * sb + l];\n"
        "            const ulong gv = rl_iq2s_grid[idx];\n"
        "            const float4 w1 = float4(as_type<uchar4>(uint(gv))) * rl3_sg4(sg, 0u);\n"
        "            const float4 w2 = float4(as_type<uchar4>(uint(gv >> 32))) * rl3_sg4(sg, 4u);\n"
        "            a2[l >> 1] += dot(x4[2u * l], w1) + dot(x4[2u * l + 1u], w2);\n"
        "        }\n"
        "        return d * (0.5f + float(sc & 15u)) * 0.25f * a2[0] + d * (0.5f + float(sc >> 4)) * 0.25f * a2[1];\n"
        "    }\n"
        "    {\n"
        "        const uint sh = rl3_rd16(bp + 2u); device const uchar *qs = bp + 8u + 16u * sb;\n"
        "        const int ls = int(((bp + 4u)[sb >> 1] >> (4u * (sb & 1u))) & 15u) | int(((sh >> (2u * sb)) & 3u) << 4);\n"
        "        for (uint j = 0; j < 16u; j += 4u) {\n"
        "            const uchar4 q = uchar4(qs[j], qs[j + 1u], qs[j + 2u], qs[j + 3u]);\n"
        "            const float4 lo = float4(rl_kvalues_iq4nl[q.x & 15u], rl_kvalues_iq4nl[q.y & 15u], rl_kvalues_iq4nl[q.z & 15u], rl_kvalues_iq4nl[q.w & 15u]);\n"
        "            const float4 hi = float4(rl_kvalues_iq4nl[q.x >> 4], rl_kvalues_iq4nl[q.y >> 4], rl_kvalues_iq4nl[q.z >> 4], rl_kvalues_iq4nl[q.w >> 4]);\n"
        "            acc += dot(x4[j >> 2], lo) + dot(x4[4u + (j >> 2)], hi);\n"
        "        }\n"
        "        return d * float(ls - 32) * acc;\n"
        "    }\n"
        "}\n"
        "static inline uint rl_iq3_block_bytes(uint t) { return t == 18u ? 98u : t == 21u ? 110u : t == 22u ? 82u : t == 12u ? 144u : 136u; }\n");
    if (!ok) { free(buf); return NULL; }
    return buf;
}

/* ---- selftest ---- */

int rl_iq3_selftest(char *error, size_t cap) {
    if (rl_iq3_block_bytes(18u) != 98u || rl_iq3_block_bytes(21u) != 110u || rl_iq3_block_bytes(22u) != 82u || rl_iq3_block_bytes(23u) != 136u ||
        rl_iq3_row_bytes(18u, 2048u) != 8u * 98u || rl_iq3_row_bytes(18u, 100u) != 0u || rl_iq3_supported(17u)) {
        snprintf(error, cap, "IQ3 block sizes wrong"); return 0;
    }
    /* codebook spot values of ggml-common.h */
    if (rl_iq3xxs_grid[0] != 0x04040404u || rl_iq3xxs_grid[255] != 0x3e341c04u || rl_iq3s_grid[0] != 0x01010101u || rl_iq3s_grid[511] != 0x0f0f0101u || rl_iq2s_grid[1023] != 0x2b2b2b2b2b2b2b2bull ||
        rl_iq2s_grid[0] != 0x0808080808080808ull || rl_kvalues_iq4nl[0] != -127 || rl_kvalues_iq4nl[15] != 113) {
        snprintf(error, cap, "IQ3 codebook spot values differ from the pinned ggml-common.h"); return 0;
    }
    float v[8];
    /* IQ3_XXS: d = 1, all indices 0 (grid 0x04040404), aux32 = 0 -> db = 0.25, signs 0 -> every value 1.0 */
    {
        uint8_t b[98] = {0};
        b[1] = 0x3c;
        rl_iq3_group8(18u, b, 5u, v);
        for (int j = 0; j < 8; ++j) if (v[j] != 1.0f) { snprintf(error, cap, "IQ3_XXS group decode wrong (%g)", (double)v[j]); return 0; }
        /* scale nibble 1 -> db = 0.75, sign bit 0 of group 0 -> first value negative */
        b[66 + 3] = 0x10; b[66] = 0x01;
        rl_iq3_group8(18u, b, 0u, v);
        if (v[0] != -3.0f || v[1] != 3.0f || v[7] != -3.0f) { snprintf(error, cap, "IQ3_XXS scale/sign decode wrong (%g %g %g)", (double)v[0], (double)v[1], (double)v[7]); return 0; }
    }
    /* IQ4_XS: d = 1, ls = 32 + 1 (scales_l low nibble 1, scales_h bits 2) -> dl = 1; nibble 8 -> kvalues[8] = 1 */
    {
        uint8_t b[136] = {0};
        b[1] = 0x3c; b[2] = 0x02; b[4] = 0x01; memset(b + 8, 0x88, 16);
        rl_iq3_group8(23u, b, 0u, v);
        for (int j = 0; j < 8; ++j) if (v[j] != (float)rl_kvalues_iq4nl[8]) { snprintf(error, cap, "IQ4_XS group decode wrong (%g)", (double)v[j]); return 0; }
    }
    if (error && cap) error[0] = '\0';
    return 1;
}
