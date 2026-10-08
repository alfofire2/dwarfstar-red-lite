#define _FILE_OFFSET_BITS 64
#define _POSIX_C_SOURCE 200809L
#define _DARWIN_C_SOURCE 1

/*
 * dev20: batched prompt ingestion for the persistent Metal engine.
 *
 * A chunk of B tokens goes through every layer with one command buffer for
 * the dense work (norms, DeltaNet, full attention, router, shared expert) and
 * per-token routed-expert dispatches through the same bounded LRU pool that
 * decode uses. Kernel arithmetic mirrors the validated single-token kernels in
 * redmetal_engine.m; the batched row kernels decode each quantized weight
 * block once and apply it to eight token vectors (functor-style decoders), the
 * DeltaNet conv/recurrent updates run the B tokens sequentially inside one
 * dispatch, and full attention appends the B keys/values first and then runs
 * causal GQA per (token, head).
 *
 * Validation: redlite-engine prefill MODEL --tokens ... compares the batched
 * path against the token-by-token Metal path and the CPU oracle.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "redmetal_engine_private.h"
#include "redlite_native_quant_cpu.h"
#include "redlite_native_iq3.h"
#include "redlite_native_router_exec.h"
#include "redmetal_topk.h"
#include "redlite_native_tables.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

static void set_error(char *dst, size_t cap, const char *msg) {
    if (dst && cap) snprintf(dst, cap, "%s", msg ? msg : "unknown Metal prefill error");
}

static NSString * const kPrefillSource = @
"#include <metal_stdlib>\n"
"#include <metal_simdgroup_matrix>\n"
"using namespace metal;\n"
/* dev74: KV cache rows into float simdgroup matrices; a half cache is loaded as half and widened element by element
 * (the 8x8 layout per thread does not depend on the element type), exact */
"#if RL_KV_HALF\n"
"inline void rl_kv_load(thread simdgroup_float8x8 &m, device const half *p, ulong stride) {\n"
"    simdgroup_half8x8 h; simdgroup_load(h, p, stride); m.thread_elements()[0] = float(h.thread_elements()[0]); m.thread_elements()[1] = float(h.thread_elements()[1]); }\n"
"inline void rl_kv_load_t(thread simdgroup_float8x8 &m, device const half *p, ulong stride) {\n"
"    simdgroup_half8x8 h; simdgroup_load(h, p, stride, ulong2(0, 0), true); m.thread_elements()[0] = float(h.thread_elements()[0]); m.thread_elements()[1] = float(h.thread_elements()[1]); }\n"
"#else\n"
"inline void rl_kv_load(thread simdgroup_float8x8 &m, device const float *p, ulong stride) { simdgroup_load(m, p, stride); }\n"
"inline void rl_kv_load_t(thread simdgroup_float8x8 &m, device const float *p, ulong stride) { simdgroup_load(m, p, stride, ulong2(0, 0), true); }\n"
"#endif\n"
"static inline ushort rd16(device const uchar *p) { return ushort(p[0]) | (ushort(p[1]) << 8); }\n"
"static inline float fp16(device const uchar *p) { return float(as_type<half>(rd16(p))); }\n"
"static inline uchar2 scale_min(uint j, device const uchar *s) {\n"
"    if (j < 4u) return uchar2(s[j] & 63u, s[j + 4u] & 63u);\n"
"    return uchar2((s[j + 4u] & 15u) | ((s[j - 4u] >> 6) << 4), (s[j + 4u] >> 4) | ((s[j] >> 6) << 4));\n"
"}\n"
/* eight token vectors accumulated against one decoded weight value */
"struct acc8 {\n"
"    device const float *x[8]; float a[8]; uint base;\n"
"    inline void add(uint i, float v) thread { const uint idx = base + i; for (uint n = 0; n < 8u; ++n) a[n] = fma(x[n][idx], v, a[n]); }\n"
"};\n"
/* dequantize one block into a dense f32 row */
"struct storer { device float *dst; uint base; inline void add(uint i, float v) thread { dst[base + i] = v; } };\n"
"template <typename F> inline void dq_q8(device const uchar *bp, thread F &f) {\n"
"    const float d = fp16(bp); device const int8_t *q = (device const int8_t *)(bp + 2ul);\n"
"    for (uint j = 0; j < 32u; ++j) f.add(j, d * float(q[j]));\n"
"}\n"
"template <typename F> inline void dq_q4k(device const uchar *bp, thread F &f) {\n"
"    const float d = fp16(bp); const float dmin = fp16(bp + 2u);\n"
"    device const uchar *scales = bp + 4u; device const uchar *qs = bp + 16u;\n"
"    for (uint g = 0; g < 8u; ++g) {\n"
"        const uchar2 sm = scale_min(g, scales);\n"
"        const float ds = d * float(sm.x); const float dm = dmin * float(sm.y);\n"
"        device const uchar *q = qs + (g >> 1) * 32u; const uint xb = g * 32u;\n"
"        for (uint l = 0; l < 32u; ++l) { const uchar quant = (g & 1u) ? (q[l] >> 4) : (q[l] & 15u); f.add(xb + l, ds * float(quant) - dm); }\n"
"    }\n"
"}\n"
"template <typename F> inline void dq_q5k(device const uchar *bp, thread F &f) {\n"
"    const float d = fp16(bp); const float dmin = fp16(bp + 2u);\n"
"    device const uchar *scales = bp + 4u; device const uchar *qh = bp + 16u; device const uchar *ql = bp + 48u;\n"
"    uint is = 0u; uchar u1 = 1u; uchar u2 = 2u;\n"
"    for (uint j = 0; j < 256u; j += 64u) {\n"
"        const uchar2 sm1 = scale_min(is, scales); const uchar2 sm2 = scale_min(is + 1u, scales);\n"
"        const float d1 = d * float(sm1.x); const float m1 = dmin * float(sm1.y);\n"
"        const float d2 = d * float(sm2.x); const float m2 = dmin * float(sm2.y);\n"
"        for (uint l = 0; l < 32u; ++l) {\n"
"            f.add(j + l, d1 * float((ql[l] & 15u) + ((qh[l] & u1) ? 16u : 0u)) - m1);\n"
"            f.add(j + 32u + l, d2 * float((ql[l] >> 4) + ((qh[l] & u2) ? 16u : 0u)) - m2);\n"
"        }\n"
"        ql += 32u; is += 2u; u1 <<= 2; u2 <<= 2;\n"
"    }\n"
"}\n"
"template <typename F> inline void dq_q6k(device const uchar *bp, thread F &f) {\n"
"    device const ushort *ql0 = (device const ushort *)(bp + 0ul);\n"
"    device const ushort *qh0 = (device const ushort *)(bp + 128ul);\n"
"    device const int8_t *scales = (device const int8_t *)(bp + 192ul);\n"
"    const float d_all = float(as_type<half>(*(device const ushort *)(bp + 208ul)));\n"
"    for (short il0 = 0; il0 < 16; ++il0) {\n"
"        short il = il0;\n"
"        device const ushort *ql = ql0 + 32 * (il / 8) + 16 * ((il / 2) & 1) + 8 * (il & 1);\n"
"        device const ushort *qh = qh0 + 16 * (il / 8) + 8 * (il & 1);\n"
"        const float sc = float(scales[(il % 2) + 2 * (il / 2)]);\n"
"        il = (il / 2) & 3;\n"
"        const uint kmask1 = il > 1 ? (il > 2 ? 0xC0C0C0C0u : 0x30303030u) : (il > 0 ? 0x0C0C0C0Cu : 0x03030303u);\n"
"        const uint kmask2 = il > 1 ? 0xF0F0F0F0u : 0x0F0F0F0Fu;\n"
"        const float ml = d_all * sc * 32.0f; const float dl0 = d_all * sc; const float dl1 = dl0 / 256.0f;\n"
"        const float dl2 = dl1 / 256.0f; const float dl3 = dl2 / 256.0f;\n"
"        const uchar shr_h = il > 2 ? 2 : 0; const uchar shl_h = il > 1 ? 0 : (il > 0 ? 2 : 4); const uchar shr_l = il > 1 ? 4 : 0;\n"
"        for (uint i = 0; i < 4u; ++i) {\n"
"            const uint low = (uint(ql[2u*i]) | (uint(ql[2u*i + 1u]) << 16)) & kmask2;\n"
"            const uint high = (uint(qh[2u*i]) | (uint(qh[2u*i + 1u]) << 16)) & kmask1;\n"
"            const uint q = ((high << shl_h) >> shr_h) | (low >> shr_l);\n"
"            const uint xb = uint(il0) * 16u + i * 4u;\n"
"            f.add(xb + 0u, dl0 * float(q & 0xFFu) - ml);\n"
"            f.add(xb + 1u, dl1 * float(q & 0xFF00u) - ml);\n"
"            f.add(xb + 2u, dl2 * float(q & 0xFF0000u) - ml);\n"
"            f.add(xb + 3u, dl3 * float(q & 0xFF000000u) - ml);\n"
"        }\n"
"    }\n"
"}\n"
"template <typename F> inline void dq_iq2xxs(device const uchar *bp, thread F &f, device const uchar *grid) {\n"
"    const float d = fp16(bp); device const ushort *q = (device const ushort *)(bp + 2ul);\n"
"    for (uint g = 0; g < 8u; ++g) {\n"
"        const uint qi = 4u * g;\n"
"        const uint auxg = uint(q[qi]) | (uint(q[qi + 1u]) << 16);\n"
"        const uint auxs = uint(q[qi + 2u]) | (uint(q[qi + 3u]) << 16);\n"
"        const float db = d * (0.5f + float(auxs >> 28)) * 0.25f;\n"
"        for (uint l = 0; l < 4u; ++l) {\n"
"            const uint grid_index = (auxg >> (8u * l)) & 255u;\n"
"            const uint sign7 = (auxs >> (7u * l)) & 127u;\n"
"            const uint sign8 = sign7 | ((popcount(sign7) & 1u) << 7);\n"
"            device const uchar *gv = grid + grid_index * 8u;\n"
"            const uint xb = g * 32u + l * 8u;\n"
"            for (uint j = 0; j < 8u; ++j) { const float s = (sign8 & (1u << j)) ? -1.0f : 1.0f; f.add(xb + j, db * float(gv[j]) * s); }\n"
"        }\n"
"    }\n"
"}\n"
/* out[t][row] = W[row] . x[t] for t < ntok; lanes SIMD lanes per row, eight tokens per pass */
#define RL_ROWSB_KERNEL(NAME, BLOCK, BYTES, DQCALL, EXTRA_PARAM) \
"kernel void " NAME "(device const uchar *weights [[buffer(0)]], constant uint &ncols [[buffer(1)]], device const float *x [[buffer(2)]],\n" \
"    device float *out [[buffer(3)]], constant uint &nrows [[buffer(4)]], constant uint &lanes [[buffer(5)]], constant uint &ntok [[buffer(6)]]" EXTRA_PARAM ",\n" \
"    uint tid [[thread_position_in_grid]], ushort simd_lane [[thread_index_in_simdgroup]]) {\n" \
"    const uint row = tid / lanes; const uint lane = uint(simd_lane) % lanes; const bool active = row < nrows;\n" \
"    const uint blocks = ncols / " BLOCK "u; const ulong row_bytes = ulong(blocks) * " BYTES "ul;\n" \
"    device const uchar *rp = weights + ulong(active ? row : 0u) * row_bytes;\n" \
"    for (uint t0 = 0; t0 < ntok; t0 += 8u) {\n" \
"        acc8 f; for (uint n = 0; n < 8u; ++n) { const uint t = min(t0 + n, ntok - 1u); f.x[n] = x + ulong(t) * ncols; f.a[n] = 0.0f; }\n" \
"        if (active) { for (uint b = lane; b < blocks; b += lanes) { device const uchar *bp = rp + ulong(b) * " BYTES "ul; f.base = b * " BLOCK "u; " DQCALL "; } }\n" \
"        for (uint n = 0; n < 8u; ++n) { float a = f.a[n]; for (uint off = lanes >> 1; off > 0u; off >>= 1) a += simd_shuffle_xor(a, ushort(off));\n" \
"            if (active && lane == 0u && t0 + n < ntok) out[(t0 + n) * nrows + row] = a; }\n" \
"    }\n" \
"}\n"
RL_ROWSB_KERNEL("rl_rowsb_q8", "32", "34", "dq_q8(bp, f)", "")
RL_ROWSB_KERNEL("rl_rowsb_q4k", "256", "144", "dq_q4k(bp, f)", "")
RL_ROWSB_KERNEL("rl_rowsb_q5k", "256", "176", "dq_q5k(bp, f)", "")
RL_ROWSB_KERNEL("rl_rowsb_q6k", "256", "210", "dq_q6k(bp, f)", "")
RL_ROWSB_KERNEL("rl_rowsb_iq2xxs", "256", "66", "dq_iq2xxs(bp, f, grid)", ", device const uchar *grid [[buffer(7)]]")
"kernel void rl_rowsb_f32(device const float *weights [[buffer(0)]], constant uint &ncols [[buffer(1)]], device const float *x [[buffer(2)]],\n"
"    device float *out [[buffer(3)]], constant uint &nrows [[buffer(4)]], constant uint &lanes [[buffer(5)]], constant uint &ntok [[buffer(6)]],\n"
"    uint tid [[thread_position_in_grid]], ushort simd_lane [[thread_index_in_simdgroup]]) {\n"
"    const uint row = tid / lanes; const uint lane = uint(simd_lane) % lanes; const bool active = row < nrows;\n"
"    device const float *rp = weights + ulong(active ? row : 0u) * ulong(ncols);\n"
"    for (uint t0 = 0; t0 < ntok; t0 += 8u) {\n"
"        float a[8]; device const float *xs[8];\n"
"        for (uint n = 0; n < 8u; ++n) { const uint t = min(t0 + n, ntok - 1u); xs[n] = x + ulong(t) * ncols; a[n] = 0.0f; }\n"
"        if (active) { for (uint i = lane; i < ncols; i += lanes) { const float w = rp[i]; for (uint n = 0; n < 8u; ++n) a[n] = fma(w, xs[n][i], a[n]); } }\n"
"        for (uint n = 0; n < 8u; ++n) { float v = a[n]; for (uint off = lanes >> 1; off > 0u; off >>= 1) v += simd_shuffle_xor(v, ushort(off));\n"
"            if (active && lane == 0u && t0 + n < ntok) out[(t0 + n) * nrows + row] = v; }\n"
"    }\n"
"}\n"
/* ---- dequantize a whole weight matrix to f32 (one thread per row block) ---- */
#define RL_DQ_KERNEL(NAME, BLOCK, BYTES, DQCALL, EXTRA_PARAM) \
"kernel void " NAME "(device const uchar *weights [[buffer(0)]], constant uint &ncols [[buffer(1)]], device float *out [[buffer(2)]],\n" \
"    constant uint &nrows [[buffer(3)]]" EXTRA_PARAM ", uint2 gid [[thread_position_in_grid]]) {\n" \
"    const uint row = gid.y; const uint b = gid.x; const uint blocks = ncols / " BLOCK "u; if (row >= nrows || b >= blocks) return;\n" \
"    storer f; f.dst = out + ulong(row) * ncols; f.base = b * " BLOCK "u;\n" \
"    device const uchar *bp = weights + ulong(row) * ulong(blocks) * " BYTES "ul + ulong(b) * " BYTES "ul; " DQCALL ";\n" \
"}\n"
RL_DQ_KERNEL("rl_dq_q8", "32", "34", "dq_q8(bp, f)", "")
RL_DQ_KERNEL("rl_dq_q4k", "256", "144", "dq_q4k(bp, f)", "")
RL_DQ_KERNEL("rl_dq_q5k", "256", "176", "dq_q5k(bp, f)", "")
RL_DQ_KERNEL("rl_dq_q6k", "256", "210", "dq_q6k(bp, f)", "")
RL_DQ_KERNEL("rl_dq_iq2xxs", "256", "66", "dq_iq2xxs(bp, f, grid)", ", device const uchar *grid [[buffer(4)]]")
/* dev31: IQ3_XXS / IQ3_S / IQ2_S / IQ4_XS: dequantize one 8-value group per thread; batched rows (GEMM fallback) */
"kernel void rl_dq_iq3(device const uchar *weights [[buffer(0)]], constant uint &ncols [[buffer(1)]], device float *out [[buffer(2)]],\n"
"    constant uint &nrows [[buffer(3)]], constant uint &type [[buffer(5)]], uint2 gid [[thread_position_in_grid]]) {\n"
"    const uint row = gid.y; const uint gi = gid.x; const uint groups = ncols / 8u; if (row >= nrows || gi >= groups) return;\n"
"    const uint bb = rl_iq3_block_bytes(type); const uint b = gi >> 5;\n"
"    device const uchar *bp = weights + ulong(row) * ulong(ncols / 256u) * bb + ulong(b) * bb;\n"
"    float4 v0, v1; rl_iq3_group8f(type, bp, gi & 31u, v0, v1);\n"
"    device float4 *o = (device float4 *)(out + ulong(row) * ncols + gi * 8u); o[0] = v0; o[1] = v1;\n"
"}\n"
"kernel void rl_rowsb_iq3(device const uchar *weights [[buffer(0)]], constant uint &ncols [[buffer(1)]], device const float *x [[buffer(2)]],\n"
"    device float *out [[buffer(3)]], constant uint &nrows [[buffer(4)]], constant uint &lanes [[buffer(5)]], constant uint &ntok [[buffer(6)]],\n"
"    constant uint &type [[buffer(8)]], uint tid [[thread_position_in_grid]], ushort simd_lane [[thread_index_in_simdgroup]]) {\n"
"    const uint row = tid / lanes; const uint lane = uint(simd_lane) % lanes; const bool active = row < nrows;\n"
"    const uint bb = rl_iq3_block_bytes(type); const uint blocks = ncols / 256u; const ulong row_bytes = ulong(blocks) * bb;\n"
"    device const uchar *rp = weights + ulong(active ? row : 0u) * row_bytes;\n"
"    for (uint t0 = 0; t0 < ntok; t0 += 8u) {\n"
"        float a[8]; for (uint n = 0; n < 8u; ++n) a[n] = 0.0f;\n"
"        if (active) { for (uint i = lane; i < blocks * 32u; i += lanes) { const uint b = i >> 5; const uint g = i & 31u; float4 v0, v1;\n"
"            rl_iq3_group8(type, rp + ulong(b) * bb, g, v0, v1);\n"
"            for (uint n = 0; n < 8u; ++n) { const uint t = min(t0 + n, ntok - 1u); device const float *xc = x + ulong(t) * ncols + i * 8u;\n"
"                a[n] += dot(*(device const float4 *)xc, v0) + dot(*(device const float4 *)(xc + 4u), v1); } } }\n"
"        for (uint n = 0; n < 8u; ++n) { float v = a[n]; for (uint off = lanes >> 1; off > 0u; off >>= 1) v += simd_shuffle_xor(v, ushort(off));\n"
"            if (active && lane == 0u && t0 + n < ntok) out[(t0 + n) * nrows + row] = v; }\n"
"    }\n"
"}\n"
/* out[t][row] = sum_k X[t][k] * W[row][k]: 32 rows x 32 tokens per threadgroup, four simdgroups of 8 rows, f32 simdgroup matrices */
"kernel void rl_gemm_wt(device const float *W [[buffer(0)]], device const float *X [[buffer(1)]], device float *out [[buffer(2)]],\n"
"    constant uint &rows [[buffer(3)]], constant uint &K [[buffer(4)]], constant uint &ntok_pad [[buffer(5)]],\n"
"    uint2 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]]) {\n"
"    const uint row0 = tg.x * 32u + uint(sg) * 8u; const uint t0 = tg.y * 32u;\n"
"    if (row0 >= rows || t0 >= ntok_pad) return;\n"
"    simdgroup_float8x8 acc[4]; for (uint j = 0; j < 4u; ++j) acc[j] = simdgroup_float8x8(0.0f);\n"
"    simdgroup_float8x8 a, b;\n"
"    device const float *wp = W + ulong(row0) * K;\n"
"    for (uint k = 0; k < K; k += 8u) {\n"
"        simdgroup_load(a, wp + k, K);\n"
"        for (uint j = 0; j < 4u; ++j) { simdgroup_load(b, X + ulong(t0 + j * 8u) * K + k, K, ulong2(0, 0), true); simdgroup_multiply_accumulate(acc[j], a, b, acc[j]); }\n"
"    }\n"
"    for (uint j = 0; j < 4u; ++j) simdgroup_store(acc[j], out + ulong(t0 + j * 8u) * rows + row0, rows, ulong2(0, 0), true);\n"
"}\n"
/* dev30: the same product staged through threadgroup memory: 64 rows x 32 tokens per threadgroup, K steps of 32;
 * each simdgroup owns 16 rows x 32 tokens (8 accumulators) and reads its 8x8 tiles from threadgroup memory.
 * The k order inside every accumulator is that of rl_gemm_wt. */
"kernel void rl_gemm_tg(device const float *W [[buffer(0)]], device const float *X [[buffer(1)]], device float *out [[buffer(2)]],\n"
"    constant uint &rows [[buffer(3)]], constant uint &K [[buffer(4)]], constant uint &ntok_pad [[buffer(5)]],\n"
"    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {\n"
"    threadgroup float wa[64 * 32]; threadgroup float xb[32 * 32];\n"
"    const uint row0 = tg.x * 64u; const uint t0 = tg.y * 32u;\n"
"    simdgroup_float8x8 acc[2][4]; for (uint r = 0; r < 2u; ++r) for (uint j = 0; j < 4u; ++j) acc[r][j] = simdgroup_float8x8(0.0f);\n"
"    simdgroup_float8x8 a[2], b;\n"
"    for (uint k0 = 0; k0 < K; k0 += 32u) {\n"
"        for (uint i = tid; i < 64u * 8u; i += 128u) { const uint r = i >> 3; const uint c = (i & 7u) * 4u;\n"
"            *(threadgroup float4 *)(wa + r * 32u + c) = *(device const float4 *)(W + ulong(row0 + r) * K + k0 + c); }\n"
"        for (uint i = tid; i < 32u * 8u; i += 128u) { const uint r = i >> 3; const uint c = (i & 7u) * 4u;\n"
"            *(threadgroup float4 *)(xb + r * 32u + c) = *(device const float4 *)(X + ulong(t0 + r) * K + k0 + c); }\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"        for (uint kk = 0; kk < 32u; kk += 8u) {\n"
"            simdgroup_load(a[0], wa + (uint(sg) * 16u) * 32u + kk, 32); simdgroup_load(a[1], wa + (uint(sg) * 16u + 8u) * 32u + kk, 32);\n"
"            for (uint j = 0; j < 4u; ++j) { simdgroup_load(b, xb + (j * 8u) * 32u + kk, 32, ulong2(0, 0), true);\n"
"                simdgroup_multiply_accumulate(acc[0][j], a[0], b, acc[0][j]); simdgroup_multiply_accumulate(acc[1][j], a[1], b, acc[1][j]); }\n"
"        }\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    for (uint r = 0; r < 2u; ++r) for (uint j = 0; j < 4u; ++j)\n"
"        simdgroup_store(acc[r][j], out + ulong(t0 + j * 8u) * rows + row0 + uint(sg) * 16u + r * 8u, rows, ulong2(0, 0), true);\n"
"}\n"
/* ---- norms / residuals: one threadgroup per token ---- */
"kernel void rl_rms_b(device const float *x [[buffer(0)]], device const float *w [[buffer(1)]], device float *y [[buffer(2)]],\n"
"    constant uint &n [[buffer(3)]], constant float &eps [[buffer(4)]], uint t [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],\n"
"    ushort simd_lane [[thread_index_in_simdgroup]], ushort simd_id [[simdgroup_index_in_threadgroup]]) {\n"
"    threadgroup float partial[8]; float ss = 0.0f; x += ulong(t) * n; y += ulong(t) * n;\n"
"    for (uint i = tid; i < n; i += 256u) ss = fma(x[i], x[i], ss);\n"
"    ss = simd_sum(ss); if (simd_lane == 0) partial[simd_id] = ss; threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    float total = 0.0f; for (uint k = 0; k < 8u; ++k) total += partial[k];\n"
"    const float inv = rsqrt(total / float(n) + eps);\n"
"    for (uint i = tid; i < n; i += 256u) y[i] = x[i] * inv * w[i];\n"
"}\n"
"kernel void rl_resid_rms_b(device const float *a [[buffer(0)]], device const float *b [[buffer(1)]], device const float *w [[buffer(2)]],\n"
"    device float *sum [[buffer(3)]], device float *norm [[buffer(4)]], constant uint &n [[buffer(5)]], constant float &eps [[buffer(6)]],\n"
"    uint t [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]], ushort simd_lane [[thread_index_in_simdgroup]], ushort simd_id [[simdgroup_index_in_threadgroup]]) {\n"
"    threadgroup float partial[8]; float ss = 0.0f; a += ulong(t) * n; b += ulong(t) * n; sum += ulong(t) * n; norm += ulong(t) * n;\n"
"    for (uint i = tid; i < n; i += 256u) { const float v = a[i] + b[i]; sum[i] = v; ss = fma(v, v, ss); }\n"
"    ss = simd_sum(ss); if (simd_lane == 0) partial[simd_id] = ss; threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    float total = 0.0f; for (uint k = 0; k < 8u; ++k) total += partial[k];\n"
"    const float inv = rsqrt(total / float(n) + eps);\n"
"    for (uint i = tid; i < n; i += 256u) norm[i] = sum[i] * inv * w[i];\n"
"}\n"
"kernel void rl_scale_add_b(device const float *resid [[buffer(0)]], device const float *routed [[buffer(1)]], device const float *shared [[buffer(2)]],\n"
"    device const float *scalar [[buffer(3)]], device float *out [[buffer(4)]], constant uint &n [[buffer(5)]], constant uint &ntok [[buffer(6)]],\n"
"    uint gid [[thread_position_in_grid]]) {\n"
"    if (gid < n * ntok) { const uint t = gid / n; out[gid] = resid[gid] + (routed[gid] + shared[gid] * scalar[t]); }\n"
"}\n"
/* ---- Gated DeltaNet, B tokens ---- */
"kernel void dn_ba_params_b(device const float *ba [[buffer(0)]], device const float *dt [[buffer(1)]], device const float *avec [[buffer(2)]],\n"
"    device float *beta [[buffer(3)]], device float *gate [[buffer(4)]], constant uint &dt_rank [[buffer(5)]], constant uint &n_group [[buffer(6)]],\n"
"    constant uint &ntok [[buffer(7)]], uint gid [[thread_position_in_grid]]) {\n"
"    if (gid >= dt_rank * ntok) return; const uint t = gid / dt_rank; const uint h = gid - t * dt_rank;\n"
"    ba += ulong(t) * 2u * dt_rank; beta += ulong(t) * dt_rank; gate += ulong(t) * dt_rank;\n"
"    const uint width = dt_rank / n_group; const uint group = h / width; const uint local = h - group * width;\n"
"    const uint stride = 2u * width; const float b = ba[group * stride + local]; const float alpha = ba[group * stride + width + local] + dt[h];\n"
"    beta[h] = 1.0f / (1.0f + exp(-b)); const float sp = alpha > 20.0f ? alpha : log(1.0f + exp(alpha)); gate[h] = sp * avec[h];\n"
"}\n"
/* causal conv over the B tokens of one channel; the state holds the last dconv-1 inputs and is updated in place */
"kernel void dn_conv_seq(device float *state [[buffer(0)]], device const float *qkv [[buffer(1)]], device const float *conv_w [[buffer(2)]],\n"
"    device float *out [[buffer(3)]], constant uint &channels [[buffer(4)]], constant uint &dconv [[buffer(5)]], constant uint &ntok [[buffer(6)]],\n"
"    uint gid [[thread_position_in_grid]]) {\n"
"    if (gid >= channels) return; const uint ns = dconv - 1u; const uint sb = gid * ns; const uint kb = gid * dconv;\n"
"    float hist[8]; for (uint j = 0; j < ns; ++j) hist[j] = state[sb + j];\n"
"    for (uint t = 0; t < ntok; ++t) {\n"
"        const float xin = qkv[ulong(t) * channels + gid]; float acc = 0.0f;\n"
"        for (uint j = 0; j < ns; ++j) acc = fma(hist[j], conv_w[kb + j], acc);\n"
"        acc = fma(xin, conv_w[kb + ns], acc); out[ulong(t) * channels + gid] = acc / (1.0f + exp(-acc));\n"
"        for (uint j = 0; j + 1u < ns; ++j) hist[j] = hist[j + 1u];\n"
"        hist[ns - 1u] = xin;\n"
"    }\n"
"    for (uint j = 0; j < ns; ++j) state[sb + j] = hist[j];\n"
"}\n"
"kernel void dn_qk_l2_b(device const float *src [[buffer(0)]], device float *q [[buffer(1)]], device float *k [[buffer(2)]],\n"
"    constant uint &head_dim [[buffer(3)]], constant uint &heads [[buffer(4)]], constant float &eps [[buffer(5)]], constant uint &src_stride [[buffer(6)]],\n"
"    uint tg [[threadgroup_position_in_grid]], uint i [[thread_position_in_threadgroup]],\n"
"    ushort simd_lane [[thread_index_in_simdgroup]], ushort simd_id [[simdgroup_index_in_threadgroup]]) {\n"
"    threadgroup float red[32]; const uint per = 2u * heads; const uint t = tg / per; const uint local = tg - t * per;\n"
"    const uint which = local / heads; const uint head = local - which * heads;\n"
"    src += ulong(t) * src_stride; q += ulong(t) * heads * head_dim; k += ulong(t) * heads * head_dim;\n"
"    const uint base = which * heads * head_dim + head * head_dim; const float x = i < head_dim ? src[base + i] : 0.0f;\n"
"    float ss = simd_sum(x * x); if (simd_lane == 0) red[simd_id] = ss; threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    float total = 0.0f; for (uint s = 0; s < (head_dim + 31u) / 32u; ++s) total += red[s];\n"
"    const float scale = 1.0f / max(sqrt(total), eps);\n"
"    if (i < head_dim) { if (which == 0u) q[head * head_dim + i] = x * scale; else k[head * head_dim + i] = x * scale; }\n"
"}\n"
/* gated delta rule for one (value head, state row j) over B tokens in sequence; state updated in place */
"kernel void dn_state_seq(device float *state [[buffer(0)]], device const float *q [[buffer(1)]], device const float *k [[buffer(2)]],\n"
"    device const float *v [[buffer(3)]], device const float *gate [[buffer(4)]], device const float *beta [[buffer(5)]],\n"
"    device float *out [[buffer(6)]], constant uint &state_size [[buffer(7)]], constant uint &kv_ratio [[buffer(8)]], constant uint &value_heads [[buffer(9)]],\n"
"    constant uint &ntok [[buffer(10)]], constant uint &qk_stride [[buffer(11)]], constant uint &v_stride [[buffer(12)]], constant uint &out_stride [[buffer(13)]],\n"
"    uint tg [[threadgroup_position_in_grid]], uint i [[thread_position_in_threadgroup]],\n"
"    ushort simd_lane [[thread_index_in_simdgroup]], ushort simd_id [[simdgroup_index_in_threadgroup]]) {\n"
"    threadgroup float red[32]; const uint h = tg / state_size; const uint j = tg - h * state_size; if (h >= value_heads) return;\n"
"    const uint kh = h / kv_ratio; const uint nsimd = state_size / 32u; const uint idx = (h * state_size + j) * state_size + i;\n"
"    float s = state[idx];\n"
"    for (uint t = 0; t < ntok; ++t) {\n"
"        s *= exp(gate[t * value_heads + h]); const float ki = k[t * qk_stride + kh * state_size + i];\n"
"        float part = simd_sum(s * ki); if (simd_lane == 0) red[simd_id] = part; threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"        float sum = 0.0f; for (uint n = 0; n < nsimd; ++n) sum += red[n];\n"
"        const float delta = (v[t * v_stride + h * state_size + j] - sum) * beta[t * value_heads + h];\n"
"        s += ki * delta;\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"        float o = simd_sum(s * q[t * qk_stride + kh * state_size + i]); if (simd_lane == 0) red[simd_id] = o; threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"        if (i == 0u) { float total = 0.0f; for (uint n = 0; n < nsimd; ++n) total += red[n]; out[t * out_stride + h * state_size + j] = total * (1.0f / sqrt((float)state_size)); }\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    state[idx] = s;\n"
"}\n"
/* dev30: same recurrence, one simdgroup per (value head, state row j), four state columns per lane: the
 * per-token dot products are simd_sum reductions with no threadgroup barrier (state_size == 128) */
"kernel void dn_state_seq_sg(device float *state [[buffer(0)]], device const float *q [[buffer(1)]], device const float *k [[buffer(2)]],\n"
"    device const float *v [[buffer(3)]], device const float *gate [[buffer(4)]], device const float *beta [[buffer(5)]],\n"
"    device float *out [[buffer(6)]], constant uint &state_size [[buffer(7)]], constant uint &kv_ratio [[buffer(8)]], constant uint &value_heads [[buffer(9)]],\n"
"    constant uint &ntok [[buffer(10)]], constant uint &qk_stride [[buffer(11)]], constant uint &v_stride [[buffer(12)]], constant uint &out_stride [[buffer(13)]],\n"
"    uint tg [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {\n"
"    const uint row = tg * 4u + uint(sg); const uint h = row / state_size; const uint j = row - h * state_size; if (h >= value_heads) return;\n"
"    const uint kh = h / kv_ratio; const uint i0 = uint(lane) * 4u;\n"
"    device float4 *sp = (device float4 *)(state + (ulong(h) * state_size + j) * state_size + i0);\n"
"    float4 s = *sp; const float norm = 1.0f / sqrt((float)state_size);\n"
"    for (uint t = 0; t < ntok; ++t) {\n"
"        s *= exp(gate[t * value_heads + h]);\n"
"        const float4 kv = *(device const float4 *)(k + t * qk_stride + kh * state_size + i0);\n"
"        const float sum = simd_sum(dot(s, kv));\n"
"        const float delta = (v[t * v_stride + h * state_size + j] - sum) * beta[t * value_heads + h];\n"
"        s += kv * delta;\n"
"        const float o = simd_sum(dot(s, *(device const float4 *)(q + t * qk_stride + kh * state_size + i0)));\n"
"        if (lane == 0) out[t * out_stride + h * state_size + j] = o * norm;\n"
"    }\n"
"    *sp = s;\n"
"}\n"
"kernel void dn_tail_norm_b(device const float *core [[buffer(0)]], device const float *z [[buffer(1)]], device const float *w [[buffer(2)]],\n"
"    device float *out [[buffer(3)]], constant float &eps [[buffer(4)]], constant uint &head_dim [[buffer(5)]], constant uint &heads [[buffer(6)]],\n"
"    uint tg [[threadgroup_position_in_grid]], uint i [[thread_position_in_threadgroup]],\n"
"    ushort simd_lane [[thread_index_in_simdgroup]], ushort simd_id [[simdgroup_index_in_threadgroup]]) {\n"
"    threadgroup float red[32]; const uint t = tg / heads; const uint h = tg - t * heads;\n"
"    core += ulong(t) * heads * head_dim; z += ulong(t) * heads * head_dim; out += ulong(t) * heads * head_dim;\n"
"    const uint base = h * head_dim; const float x = i < head_dim ? core[base + i] : 0.0f;\n"
"    float ss = simd_sum(x * x); if (simd_lane == 0) red[simd_id] = ss; threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    float total = 0.0f; for (uint n = 0; n < (head_dim + 31u) / 32u; ++n) total += red[n];\n"
"    const float inv = rsqrt(total / float(head_dim) + eps);\n"
"    if (i < head_dim) { const float g = z[base + i]; const float silu = g / (1.0f + exp(-g)); out[base + i] = x * inv * w[i] * silu; }\n"
"}\n"
/* ---- full attention, B tokens: append all keys/values, then causal GQA per (token, head) ---- */
"kernel void attn_qk_prep_b(device const float *qgate_raw [[buffer(0)]], device const float *k_raw [[buffer(1)]], device const float *value [[buffer(2)]],\n"
"    device const float *qw [[buffer(3)]], device const float *kw [[buffer(4)]], device float *query_rope [[buffer(5)]], device float *gate [[buffer(6)]],\n"
"    device RL_KV *key_cache [[buffer(7)]], device RL_KV *value_cache [[buffer(8)]], constant uint &head_dim [[buffer(9)]],\n"
"    constant uint &query_heads [[buffer(10)]], constant uint &kv_heads [[buffer(11)]], constant uint &position0 [[buffer(12)]],\n"
"    constant uint &rope_dims [[buffer(13)]], constant float &freq_base [[buffer(14)]], constant float &eps [[buffer(15)]],\n"
"    uint tg [[threadgroup_position_in_grid]], uint i [[thread_position_in_threadgroup]],\n"
"    ushort simd_lane [[thread_index_in_simdgroup]], ushort simd_id [[simdgroup_index_in_threadgroup]]) {\n"
"    threadgroup float red[32]; threadgroup float tmp[256];\n"
"    const uint per = query_heads + kv_heads; const uint t = tg / per; const uint local = tg - t * per; const uint position = position0 + t;\n"
"    const uint qcount = query_heads * head_dim; const uint kvcount = kv_heads * head_dim;\n"
"    qgate_raw += ulong(t) * 2u * qcount; k_raw += ulong(t) * kvcount; value += ulong(t) * kvcount; query_rope += ulong(t) * qcount; gate += ulong(t) * qcount;\n"
"    const bool is_q = local < query_heads; const uint head = is_q ? local : local - query_heads; if (!is_q && head >= kv_heads) return;\n"
"    const uint src = is_q ? head * head_dim * 2u : head * head_dim;\n"
"    const float x = i < head_dim ? (is_q ? qgate_raw[src + i] : k_raw[src + i]) : 0.0f;\n"
"    float ss = simd_sum(x * x); if (simd_lane == 0) red[simd_id] = ss; threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    float total = 0.0f; for (uint n = 0; n < (head_dim + 31u) / 32u; ++n) total += red[n];\n"
"    const float inv = rsqrt(total / float(head_dim) + eps);\n"
"    if (i < head_dim) tmp[i] = x * inv * (is_q ? qw[i] : kw[i]);\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    if (i >= head_dim) return;\n"
"    const uint rhalf = rope_dims / 2u; float r = tmp[i];\n"
"    if (i < rope_dims) {\n"
"        const uint pidx = i < rhalf ? i : i - rhalf;\n"
"        const float theta = float(position) * pow(freq_base, -2.0f * float(pidx) / float(rope_dims));\n"
"        const float c = cos(theta); const float sn = sin(theta);\n"
"        const float x0 = tmp[pidx]; const float x1 = tmp[pidx + rhalf];\n"
"        r = i < rhalf ? (x0 * c - x1 * sn) : (x0 * sn + x1 * c);\n"
"    }\n"
"    if (is_q) { query_rope[head * head_dim + i] = r; gate[head * head_dim + i] = qgate_raw[src + head_dim + i]; }\n"
"    else { const uint dst = (position * kv_heads + head) * head_dim + i; key_cache[dst] = RL_KV(r); value_cache[dst] = RL_KV(value[head * head_dim + i]); }\n"
"}\n"
"kernel void attn_gqa_b(device const float *query [[buffer(0)]], device const RL_KV *key_cache [[buffer(1)]], device const RL_KV *value_cache [[buffer(2)]],\n"
"    device const float *gate [[buffer(3)]], device float *gated [[buffer(4)]], constant uint &head_dim [[buffer(5)]],\n"
"    constant uint &query_heads [[buffer(6)]], constant uint &kv_heads [[buffer(7)]], constant uint &position0 [[buffer(8)]],\n"
"    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],\n"
"    ushort simd_lane [[thread_index_in_simdgroup]], ushort simd_id [[simdgroup_index_in_threadgroup]]) {\n"
"    threadgroup float scores[1024]; threadgroup float red_max[8]; threadgroup float red_sum[8];\n"
"    const uint t = tg / query_heads; const uint head = tg - t * query_heads; const uint seq_len = position0 + t + 1u;\n"
"    const uint qcount = query_heads * head_dim; query += ulong(t) * qcount; gate += ulong(t) * qcount; gated += ulong(t) * qcount;\n"
"    const uint qbase = head * head_dim; const uint kv_head = head / (query_heads / kv_heads); const float scale = rsqrt(float(head_dim));\n"
"    float acc = 0.0f; float m = -INFINITY; float l = 0.0f;\n"
"    for (uint chunk_start = 0; chunk_start < seq_len; chunk_start += 1024u) {\n"
"        const uint chunk = min(1024u, seq_len - chunk_start);\n"
"        for (uint pos = tid; pos < chunk; pos += 256u) {\n"
"            const uint kbase = ((chunk_start + pos) * kv_heads + kv_head) * head_dim; float dot = 0.0f;\n"
"            for (uint i = 0; i < head_dim; ++i) dot = fma(query[qbase + i], float(key_cache[kbase + i]), dot);\n"
"            scores[pos] = dot * scale;\n"
"        }\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"        float cm = -INFINITY; for (uint pos = tid; pos < chunk; pos += 256u) cm = max(cm, scores[pos]);\n"
"        cm = simd_max(cm); if (simd_lane == 0) red_max[simd_id] = cm; threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"        cm = red_max[0]; for (uint k = 1; k < 8u; ++k) cm = max(cm, red_max[k]);\n"
"        const float new_m = max(m, cm); const float alpha = (m == -INFINITY) ? 0.0f : exp(m - new_m);\n"
"        float cs = 0.0f; for (uint pos = tid; pos < chunk; pos += 256u) { const float p = exp(scores[pos] - new_m); scores[pos] = p; cs += p; }\n"
"        cs = simd_sum(cs); if (simd_lane == 0) red_sum[simd_id] = cs; threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"        float total = 0.0f; for (uint k = 0; k < 8u; ++k) total += red_sum[k];\n"
"        l = l * alpha + total;\n"
"        if (tid < head_dim) {\n"
"            acc *= alpha;\n"
"            for (uint pos = 0; pos < chunk; ++pos) acc = fma(scores[pos], float(value_cache[((chunk_start + pos) * kv_heads + kv_head) * head_dim + tid]), acc);\n"
"        }\n"
"        m = new_m;\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    if (tid < head_dim) gated[qbase + tid] = (acc / l) / (1.0f + exp(-gate[qbase + tid]));\n"
"}\n"
/* dev30: tiled causal GQA for the batched prefill (flash-attention style, f32 simdgroup matrices).
 * Threadgroup = (query head, 8 consecutive tokens), 4 simdgroups. Key blocks of 32 positions:
 * S = Q K^T (each simdgroup one 8x8 block of positions), online softmax by simdgroup 0 (lane = position),
 * O = diag(alpha) O + P V (each simdgroup 64 of the 256 head dimensions). head_dim == 256 only. */
"kernel void attn_fa_b(device const float *query [[buffer(0)]], device const RL_KV *key_cache [[buffer(1)]], device const RL_KV *value_cache [[buffer(2)]],\n"
"    device const float *gate [[buffer(3)]], device float *gated [[buffer(4)]], constant uint &head_dim [[buffer(5)]],\n"
"    constant uint &query_heads [[buffer(6)]], constant uint &kv_heads [[buffer(7)]], constant uint &position0 [[buffer(8)]], constant uint &ntok [[buffer(9)]],\n"
"    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],\n"
"    ushort simd_lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {\n"
"    threadgroup float qs[8 * 256]; threadgroup float ss[8 * 32]; threadgroup float dg[64]; threadgroup float mrow[8]; threadgroup float lrow[8];\n"
"    const uint head = tg.x; const uint t0 = tg.y * 8u; const uint qcount = query_heads * head_dim;\n"
"    const uint kvh = head / (query_heads / kv_heads); const uint kstride = kv_heads * head_dim; const float scale = rsqrt(float(head_dim));\n"
"    for (uint i = tid; i < 8u * 256u; i += 128u) { const uint r = i >> 8; const uint d = i & 255u;\n"
"        qs[i] = t0 + r < ntok ? query[ulong(t0 + r) * qcount + head * 256u + d] * scale : 0.0f; }\n"
"    if (tid < 64u) dg[tid] = 0.0f;\n"
"    if (tid < 8u) { mrow[tid] = -INFINITY; lrow[tid] = 0.0f; }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    simdgroup_float8x8 o[8]; for (uint i = 0; i < 8u; ++i) o[i] = simdgroup_float8x8(0.0f);\n"
"    const uint last = position0 + min(t0 + 8u, ntok) - 1u;   /* last key position any row of the tile sees */\n"
"    for (uint kb = 0; kb <= last; kb += 32u) {\n"
"        {   /* S block: 8 tokens x positions kb + sg*8 .. +8 */\n"
"            simdgroup_float8x8 acc = simdgroup_float8x8(0.0f), a, b;\n"
"            device const RL_KV *kp = key_cache + ulong(kb + uint(sg) * 8u) * kstride + kvh * 256u;\n"
"            for (uint k = 0; k < 256u; k += 8u) {\n"
"                simdgroup_load(a, qs + k, 256);\n"
"                rl_kv_load_t(b, kp + k, kstride);\n"
"                simdgroup_multiply_accumulate(acc, a, b, acc);\n"
"            }\n"
"            simdgroup_store(acc, ss + uint(sg) * 8u, 32);\n"
"        }\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"        if (sg == 0) {\n"
"            const uint pos = kb + uint(simd_lane);\n"
"            for (uint r = 0; r < 8u; ++r) {\n"
"                const bool valid = t0 + r < ntok && pos <= position0 + t0 + r;\n"
"                const float s = valid ? ss[r * 32u + simd_lane] : -INFINITY;\n"
"                const float mx = simd_max(s); const float m_old = mrow[r]; const float m_new = max(m_old, mx);\n"
"                const float p = valid ? exp(s - m_new) : 0.0f;\n"
"                const float sum = simd_sum(p);\n"
"                const float alpha = m_old == -INFINITY ? 0.0f : exp(m_old - m_new);\n"
"                ss[r * 32u + simd_lane] = p;\n"
"                if (simd_lane == 0) { lrow[r] = lrow[r] * alpha + sum; mrow[r] = m_new; dg[r * 9u] = alpha; }\n"
"            }\n"
"        }\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"        {\n"
"            simdgroup_float8x8 d, p, v;\n"
"            simdgroup_load(d, dg, 8);\n"
"            for (uint i = 0; i < 8u; ++i) simdgroup_multiply(o[i], d, o[i]);\n"
"            for (uint kk = 0; kk < 4u; ++kk) {\n"
"                simdgroup_load(p, ss + kk * 8u, 32);\n"
"                device const RL_KV *vp = value_cache + ulong(kb + kk * 8u) * kstride + kvh * 256u + uint(sg) * 64u;\n"
"                for (uint i = 0; i < 8u; ++i) { rl_kv_load(v, vp + i * 8u, kstride); simdgroup_multiply_accumulate(o[i], p, v, o[i]); }\n"
"            }\n"
"        }\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    for (uint i = 0; i < 8u; ++i) simdgroup_store(o[i], qs + uint(sg) * 64u + i * 8u, 256);\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint i = tid; i < 8u * 256u; i += 128u) { const uint r = i >> 8; const uint d = i & 255u; if (t0 + r >= ntok) continue;\n"
"        const ulong idx = ulong(t0 + r) * qcount + head * 256u + d;\n"
"        gated[idx] = (qs[i] / lrow[r]) / (1.0f + exp(-gate[idx])); }\n"
"}\n"
/* dev65: attn_fa_b with 16-token tiles and the online softmax spread over the 4 simdgroups (4 rows each) instead of
 * simdgroup 0 alone. Every element is computed in the same order as attn_fa_b, so the output is bit-identical; at long
 * contexts the key/value blocks serve twice the tokens and the serial softmax section is a quarter as long. */
"kernel void attn_fa_b2(device const float *query [[buffer(0)]], device const RL_KV *key_cache [[buffer(1)]], device const RL_KV *value_cache [[buffer(2)]],\n"
"    device const float *gate [[buffer(3)]], device float *gated [[buffer(4)]], constant uint &head_dim [[buffer(5)]],\n"
"    constant uint &query_heads [[buffer(6)]], constant uint &kv_heads [[buffer(7)]], constant uint &position0 [[buffer(8)]], constant uint &ntok [[buffer(9)]],\n"
"    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],\n"
"    ushort simd_lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {\n"
"    threadgroup float qs[16 * 256]; threadgroup float ss[16 * 32]; threadgroup float dg[128]; threadgroup float mrow[16]; threadgroup float lrow[16];\n"
"    const uint head = tg.x; const uint t0 = tg.y * 16u; const uint qcount = query_heads * head_dim;\n"
"    const uint kvh = head / (query_heads / kv_heads); const uint kstride = kv_heads * head_dim; const float scale = rsqrt(float(head_dim));\n"
"    for (uint i = tid; i < 16u * 256u; i += 128u) { const uint r = i >> 8; const uint d = i & 255u;\n"
"        qs[i] = t0 + r < ntok ? query[ulong(t0 + r) * qcount + head * 256u + d] * scale : 0.0f; }\n"
"    dg[tid] = 0.0f;\n"
"    if (tid < 16u) { mrow[tid] = -INFINITY; lrow[tid] = 0.0f; }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    simdgroup_float8x8 o0[8], o1[8]; for (uint i = 0; i < 8u; ++i) { o0[i] = simdgroup_float8x8(0.0f); o1[i] = simdgroup_float8x8(0.0f); }\n"
"    const uint last = position0 + min(t0 + 16u, ntok) - 1u;\n"
"    for (uint kb = 0; kb <= last; kb += 32u) {\n"
"        {   /* S block: 16 tokens x positions kb + sg*8 .. +8 */\n"
"            simdgroup_float8x8 acc0 = simdgroup_float8x8(0.0f), acc1 = simdgroup_float8x8(0.0f), a0, a1, b;\n"
"            device const RL_KV *kp = key_cache + ulong(kb + uint(sg) * 8u) * kstride + kvh * 256u;\n"
"            for (uint k = 0; k < 256u; k += 8u) {\n"
"                rl_kv_load_t(b, kp + k, kstride);\n"
"                simdgroup_load(a0, qs + k, 256);\n"
"                simdgroup_load(a1, qs + 8u * 256u + k, 256);\n"
"                simdgroup_multiply_accumulate(acc0, a0, b, acc0);\n"
"                simdgroup_multiply_accumulate(acc1, a1, b, acc1);\n"
"            }\n"
"            simdgroup_store(acc0, ss + uint(sg) * 8u, 32);\n"
"            simdgroup_store(acc1, ss + 8u * 32u + uint(sg) * 8u, 32);\n"
"        }\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"        {   /* online softmax: simdgroup sg takes rows sg, sg+4, sg+8, sg+12 */\n"
"            const uint pos = kb + uint(simd_lane);\n"
"            for (uint r = uint(sg); r < 16u; r += 4u) {\n"
"                const bool valid = t0 + r < ntok && pos <= position0 + t0 + r;\n"
"                const float s = valid ? ss[r * 32u + simd_lane] : -INFINITY;\n"
"                const float mx = simd_max(s); const float m_old = mrow[r]; const float m_new = max(m_old, mx);\n"
"                const float p = valid ? exp(s - m_new) : 0.0f;\n"
"                const float sum = simd_sum(p);\n"
"                const float alpha = m_old == -INFINITY ? 0.0f : exp(m_old - m_new);\n"
"                ss[r * 32u + simd_lane] = p;\n"
"                if (simd_lane == 0) { lrow[r] = lrow[r] * alpha + sum; mrow[r] = m_new; dg[(r >> 3) * 64u + (r & 7u) * 9u] = alpha; }\n"
"            }\n"
"        }\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"        {\n"
"            simdgroup_float8x8 d0, d1, p0, p1, v;\n"
"            simdgroup_load(d0, dg, 8); simdgroup_load(d1, dg + 64u, 8);\n"
"            for (uint i = 0; i < 8u; ++i) { simdgroup_multiply(o0[i], d0, o0[i]); simdgroup_multiply(o1[i], d1, o1[i]); }\n"
"            for (uint kk = 0; kk < 4u; ++kk) {\n"
"                simdgroup_load(p0, ss + kk * 8u, 32);\n"
"                simdgroup_load(p1, ss + 8u * 32u + kk * 8u, 32);\n"
"                device const RL_KV *vp = value_cache + ulong(kb + kk * 8u) * kstride + kvh * 256u + uint(sg) * 64u;\n"
"                for (uint i = 0; i < 8u; ++i) {\n"
"                    rl_kv_load(v, vp + i * 8u, kstride);\n"
"                    simdgroup_multiply_accumulate(o0[i], p0, v, o0[i]);\n"
"                    simdgroup_multiply_accumulate(o1[i], p1, v, o1[i]);\n"
"                }\n"
"            }\n"
"        }\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    for (uint i = 0; i < 8u; ++i) { simdgroup_store(o0[i], qs + uint(sg) * 64u + i * 8u, 256); simdgroup_store(o1[i], qs + 8u * 256u + uint(sg) * 64u + i * 8u, 256); }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint i = tid; i < 16u * 256u; i += 128u) { const uint r = i >> 8; const uint d = i & 255u; if (t0 + r >= ntok) continue;\n"
"        const ulong idx = ulong(t0 + r) * qcount + head * 256u + d;\n"
"        gated[idx] = (qs[i] / lrow[r]) / (1.0f + exp(-gate[idx])); }\n"
"}\n"
/* ---- shared expert: per-token scalar gate ---- */
"kernel void sh_scalar_gate_b(device const float *w [[buffer(0)]], device const float *x [[buffer(1)]], device float *scalar [[buffer(2)]],\n"
"    constant uint &hidden [[buffer(3)]], uint t [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],\n"
"    ushort simd_lane [[thread_index_in_simdgroup]], ushort simd_id [[simdgroup_index_in_threadgroup]]) {\n"
"    threadgroup float partial[8]; float acc = 0.0f; x += ulong(t) * hidden;\n"
"    for (uint i = tid; i < hidden; i += 256u) acc = fma(w[i], x[i], acc);\n"
"    acc = simd_sum(acc); if (simd_lane == 0) partial[simd_id] = acc; threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    if (tid == 0u) { float total = 0.0f; for (uint k = 0; k < 8u; ++k) total += partial[k]; scalar[t] = 1.0f / (1.0f + exp(-total)); }\n"
"}\n"
"kernel void sh_silu_mul_b(device const float *gate [[buffer(0)]], device const float *up [[buffer(1)]], device float *act [[buffer(2)]],\n"
"    constant uint &count [[buffer(3)]], uint gid [[thread_position_in_grid]]) {\n"
"    if (gid >= count) return; const float g = gate[gid]; act[gid] = (g / (1.0f + exp(-g))) * up[gid];\n"
"}\n";

struct rl_metal_prefill {
    uint32_t cap;
    id<MTLLibrary> lib;
    uint32_t cap_pad;        /* token capacity rounded up to the 32-token GEMM tile */
    id<MTLComputePipelineState> p_rowsb_f32, p_rowsb_q8, p_rowsb_q4k, p_rowsb_q5k, p_rowsb_q6k, p_rowsb_iq2xxs, p_rowsb_iq3, p_dq_iq3;
    id<MTLComputePipelineState> p_dq_q8, p_dq_q4k, p_dq_q5k, p_dq_q6k, p_dq_iq2xxs, p_gemm, p_gemm_tg;
    int gemm_tg;             /* dev30: threadgroup-staged dense GEMM (RL_PREFILL_GEMM_TG=0 -> rl_gemm_wt) */
    id<MTLBuffer> wdq;       /* dequantized f32 copy of the current dense matrix */
    size_t wdq_floats;
    /* mapped experts: the routed gate/up/down tensors of every layer wrapped in place from the mmap (no copy) */
    int mapped;
    __unsafe_unretained id<MTLBuffer> *exp_gate, *exp_up, *exp_down;
    uint64_t *exp_gate_addr0, *exp_up_addr0, *exp_down_addr0;   /* GPU address of expert 0 in each */
    uint64_t *exp_gate_bytes, *exp_up_bytes, *exp_down_bytes;   /* per-expert stride */
    uint32_t *exp_type;
    id<MTLComputePipelineState> p_rms, p_resid_rms, p_scale_add;
    id<MTLComputePipelineState> p_dn_ba, p_dn_conv, p_dn_l2, p_dn_state, p_dn_state_sg, p_dn_tail;
    int state_sg;            /* dev30: simdgroup-per-row DeltaNet recurrence (RL_PREFILL_STATE_SG=0 -> dn_state_seq) */
    id<MTLComputePipelineState> p_attn_prep, p_attn_gqa, p_attn_fa, p_attn_fa2, p_sh_scalar, p_sh_silu;
    int attn_tiled;          /* dev30: tiled attention (RL_PREFILL_ATTN_TILE=0 -> the dev20 per-(token, head) kernel) */
    int attn_fa2;            /* dev65: 16-token tiles, softmax on every simdgroup (RL_PREFILL_FA2) */
    int profile;             /* dev30: RL_PREFILL_PROFILE=1 commits at every stage boundary and sums GPU time per stage */
    double prof_ms[10];
    id<MTLBuffer> xb, normed, branch, resid, ffn_in;
    id<MTLBuffer> qkv, z, ba, beta, gate, conv_silu, q, k, core, ng;
    id<MTLBuffer> qgate_raw, k_raw, value, query_rope, agate, gated;
    id<MTLBuffer> router_logits, sh_gate, sh_up, sh_act, sh_out, scalar, routed;
    id<MTLBuffer> pred_logits[2];   /* dev24: the next layer's router on this layer's FFN input, by layer parity (the
                                     * prefetch thread reads one while the next dense pass writes the other) */
    NSMutableArray *keep;
};

void rl_metal_prefill_destroy(struct rl_metal_prefill *pf) {
    if (!pf) return;
    pf->p_rowsb_f32 = pf->p_rowsb_q8 = pf->p_rowsb_q4k = pf->p_rowsb_q5k = pf->p_rowsb_q6k = pf->p_rowsb_iq2xxs = pf->p_rowsb_iq3 = pf->p_dq_iq3 = nil;
    pf->p_rms = pf->p_resid_rms = pf->p_scale_add = pf->p_dn_ba = pf->p_dn_conv = pf->p_dn_l2 = pf->p_dn_state = pf->p_dn_state_sg = pf->p_dn_tail = nil;
        pf->p_attn_prep = pf->p_attn_gqa = pf->p_attn_fa = pf->p_attn_fa2 = pf->p_sh_scalar = pf->p_sh_silu = nil;
    pf->p_dq_q8 = pf->p_dq_q4k = pf->p_dq_q5k = pf->p_dq_q6k = pf->p_dq_iq2xxs = pf->p_gemm = pf->p_gemm_tg = nil; pf->wdq = nil;
    pf->xb = pf->normed = pf->branch = pf->resid = pf->ffn_in = pf->qkv = pf->z = pf->ba = pf->beta = pf->gate = pf->conv_silu = nil;
    pf->q = pf->k = pf->core = pf->ng = pf->qgate_raw = pf->k_raw = pf->value = pf->query_rope = pf->agate = pf->gated = nil;
    pf->router_logits = pf->sh_gate = pf->sh_up = pf->sh_act = pf->sh_out = pf->scalar = pf->routed = nil;
    pf->pred_logits[0] = pf->pred_logits[1] = nil;
    pf->keep = nil; pf->lib = nil;
    free(pf->exp_gate); free(pf->exp_up); free(pf->exp_down);
    free(pf->exp_gate_addr0); free(pf->exp_up_addr0); free(pf->exp_down_addr0);
    free(pf->exp_gate_bytes); free(pf->exp_up_bytes); free(pf->exp_down_bytes); free(pf->exp_type);
    free(pf);
}

static id<MTLBuffer> pf_buf(rl_metal_engine *m, struct rl_metal_prefill *pf, size_t bytes) {
    id<MTLBuffer> b = [m->dev newBufferWithLength:bytes ? bytes : 16u options:MTLResourceStorageModeShared];
    if (b) [pf->keep addObject:b];
    return b;
}

/* No-copy MTLBuffer over the page-aligned window [off, off+len) of the mmap; *addr0 is the GPU address of `off`. */
static id<MTLBuffer> wrap_window(rl_metal_engine *m, struct rl_metal_prefill *pf, const rl_gguf_model *g, uint64_t off, uint64_t len, uint64_t *addr0) {
    if (!g->map || off + len > g->map_len) return nil;
    const size_t page = (size_t)sysconf(_SC_PAGESIZE);
    const uintptr_t base = (uintptr_t)g->map + (uintptr_t)off;
    const uintptr_t aligned = base & ~(uintptr_t)(page - 1u);
    size_t wlen = (size_t)(base + len - aligned);
    wlen = (wlen + page - 1u) & ~(page - 1u);
    const uintptr_t map_end = ((uintptr_t)g->map + g->map_len + page - 1u) & ~(uintptr_t)(page - 1u);
    if (aligned + wlen > map_end) wlen = (size_t)(map_end - aligned);
    id<MTLBuffer> b = [m->dev newBufferWithBytesNoCopy:(void *)aligned length:wlen options:MTLResourceStorageModeShared deallocator:nil];
    if (!b) return nil;
    uint64_t gpu = 0;
    if (@available(macOS 13.0, *)) gpu = (uint64_t)[b gpuAddress];
    if (!gpu) return nil;
    *addr0 = gpu + (uint64_t)(base - aligned);
    [pf->keep addObject:b];   /* the per-layer tables are unretained; keep owns the buffer */
    return b;
}

/* Wrap every layer's routed expert tensors; on any failure the slot-pool path stays in use. */
static void prefill_map_experts(rl_engine *e, rl_metal_engine *m, struct rl_metal_prefill *pf) {
    const rl_engine_info *in = &e->info;
    const uint32_t L = in->n_layer, E = in->n_expert;
    pf->exp_gate = (__unsafe_unretained id<MTLBuffer> *)calloc(L, sizeof(id));
    pf->exp_up = (__unsafe_unretained id<MTLBuffer> *)calloc(L, sizeof(id));
    pf->exp_down = (__unsafe_unretained id<MTLBuffer> *)calloc(L, sizeof(id));
    pf->exp_gate_addr0 = (uint64_t *)calloc(L, sizeof(uint64_t)); pf->exp_up_addr0 = (uint64_t *)calloc(L, sizeof(uint64_t)); pf->exp_down_addr0 = (uint64_t *)calloc(L, sizeof(uint64_t));
    pf->exp_gate_bytes = (uint64_t *)calloc(L, sizeof(uint64_t)); pf->exp_up_bytes = (uint64_t *)calloc(L, sizeof(uint64_t)); pf->exp_down_bytes = (uint64_t *)calloc(L, sizeof(uint64_t));
    pf->exp_type = (uint32_t *)calloc(L, sizeof(uint32_t));
    if (!pf->exp_gate || !pf->exp_up || !pf->exp_down || !pf->exp_gate_addr0 || !pf->exp_up_addr0 || !pf->exp_down_addr0 ||
        !pf->exp_gate_bytes || !pf->exp_up_bytes || !pf->exp_down_bytes || !pf->exp_type) return;
    char err[256];
    for (uint32_t l = 0; l < L; ++l) {
        rl_expert_layout first, last;
        if (!rl_native_expert_layout(&e->expert_map, l, 0u, &first, err, sizeof(err)) ||
            !rl_native_expert_layout(&e->expert_map, l, E - 1u, &last, err, sizeof(err))) return;
        if ((first.ggml_type != 17u && first.ggml_type != 29u) || first.down_type != first.ggml_type) return;
        /* per-expert strides from consecutive experts; contiguous tensors give stride == bytes */
        rl_expert_layout second;
        if (E > 1u && !rl_native_expert_layout(&e->expert_map, l, 1u, &second, err, sizeof(err))) return;
        const uint64_t gs = E > 1u ? second.gate_offset - first.gate_offset : first.gate_bytes;
        const uint64_t us = E > 1u ? second.up_offset - first.up_offset : first.up_bytes;
        const uint64_t ds = E > 1u ? second.down_offset - first.down_offset : first.down_bytes;
        if (gs < first.gate_bytes || us < first.up_bytes || ds < first.down_bytes) return;
        pf->exp_gate[l] = wrap_window(m, pf, &e->gguf, first.gate_offset, last.gate_offset + last.gate_bytes - first.gate_offset, &pf->exp_gate_addr0[l]);
        pf->exp_up[l] = wrap_window(m, pf, &e->gguf, first.up_offset, last.up_offset + last.up_bytes - first.up_offset, &pf->exp_up_addr0[l]);
        pf->exp_down[l] = wrap_window(m, pf, &e->gguf, first.down_offset, last.down_offset + last.down_bytes - first.down_offset, &pf->exp_down_addr0[l]);
        if (!pf->exp_gate[l] || !pf->exp_up[l] || !pf->exp_down[l]) return;
        pf->exp_gate_bytes[l] = gs; pf->exp_up_bytes[l] = us; pf->exp_down_bytes[l] = ds;
        pf->exp_type[l] = first.ggml_type;
    }
    pf->mapped = 1;
}

static struct rl_metal_prefill *prefill_create(rl_engine *e, rl_metal_engine *m, uint32_t cap, char *error, size_t ecap) {
    const rl_engine_info *in = &e->info;
    if (in->d_conv > 8u) { set_error(error, ecap, "batched prefill supports d_conv <= 8"); return NULL; }
    struct rl_metal_prefill *pf = (struct rl_metal_prefill *)calloc(1, sizeof(*pf));
    if (!pf) { set_error(error, ecap, "prefill allocation failed"); return NULL; }
    pf->cap = cap;
    pf->cap_pad = (cap + 31u) & ~31u;
    pf->keep = [NSMutableArray array];
    NSError *le = nil;
    char *iq3 = rl_iq3_metal_source();   /* dev31: codebooks + rl_iq3_group8 */
    if (!iq3) { set_error(error, ecap, "IQ3 Metal source allocation failed"); rl_metal_prefill_destroy(pf); return NULL; }
    NSString *src = [NSString stringWithFormat:@"#include <metal_stdlib>\nusing namespace metal;\n%s%s\n%@", rl_kv_defines(), iq3, kPrefillSource];
    free(iq3);
    pf->lib = [m->dev newLibraryWithSource:src options:nil error:&le];
    if (!pf->lib) {
        snprintf(error, ecap, "Metal prefill library compile failed: %s", le.localizedDescription.UTF8String ?: "unknown");
        rl_metal_prefill_destroy(pf); return NULL;
    }
    struct { __strong id<MTLComputePipelineState> *slot; NSString *name; } pipes[] = {
        {&pf->p_rowsb_f32, @"rl_rowsb_f32"}, {&pf->p_rowsb_q8, @"rl_rowsb_q8"}, {&pf->p_rowsb_q4k, @"rl_rowsb_q4k"},
        {&pf->p_rowsb_q5k, @"rl_rowsb_q5k"}, {&pf->p_rowsb_q6k, @"rl_rowsb_q6k"}, {&pf->p_rowsb_iq2xxs, @"rl_rowsb_iq2xxs"},
        {&pf->p_rowsb_iq3, @"rl_rowsb_iq3"}, {&pf->p_dq_iq3, @"rl_dq_iq3"},
        {&pf->p_rms, @"rl_rms_b"}, {&pf->p_resid_rms, @"rl_resid_rms_b"}, {&pf->p_scale_add, @"rl_scale_add_b"},
        {&pf->p_dn_ba, @"dn_ba_params_b"}, {&pf->p_dn_conv, @"dn_conv_seq"}, {&pf->p_dn_l2, @"dn_qk_l2_b"},
        {&pf->p_dn_state, @"dn_state_seq"}, {&pf->p_dn_state_sg, @"dn_state_seq_sg"}, {&pf->p_dn_tail, @"dn_tail_norm_b"},
        {&pf->p_attn_prep, @"attn_qk_prep_b"}, {&pf->p_attn_gqa, @"attn_gqa_b"}, {&pf->p_attn_fa, @"attn_fa_b"}, {&pf->p_attn_fa2, @"attn_fa_b2"},
        {&pf->p_sh_scalar, @"sh_scalar_gate_b"}, {&pf->p_sh_silu, @"sh_silu_mul_b"},
        {&pf->p_dq_q8, @"rl_dq_q8"}, {&pf->p_dq_q4k, @"rl_dq_q4k"}, {&pf->p_dq_q5k, @"rl_dq_q5k"}, {&pf->p_dq_q6k, @"rl_dq_q6k"},
        {&pf->p_dq_iq2xxs, @"rl_dq_iq2xxs"}, {&pf->p_gemm, @"rl_gemm_wt"}, {&pf->p_gemm_tg, @"rl_gemm_tg"},
    };
    for (size_t i = 0; i < sizeof(pipes) / sizeof(pipes[0]); ++i) {
        *pipes[i].slot = make_pipe(m->dev, pf->lib, pipes[i].name, error, ecap);
        if (!*pipes[i].slot) { rl_metal_prefill_destroy(pf); return NULL; }
        if (getenv("RL_PIPELINE_STATS"))   /* dev72: register pressure shows as fewer threads per threadgroup than 1024 */
            fprintf(stderr, "pipeline %-18s max threads/threadgroup %4lu  static threadgroup memory %5lu bytes\n", pipes[i].name.UTF8String,
                    (unsigned long)(*pipes[i].slot).maxTotalThreadsPerThreadgroup, (unsigned long)(*pipes[i].slot).staticThreadgroupMemoryLength);
    }
    const size_t B = pf->cap_pad;
    const size_t hb = (size_t)in->hidden * sizeof(float) * B;
    /* dequant scratch: the largest dense matrix of any layer (plus the LM head is never batched) */
    size_t wmax = 0;
    for (uint32_t l = 0; l < in->n_layer; ++l) {
        const mlayer *w = &m->layers[l];
        const mweight *all[] = {&w->qkv, &w->z, &w->ba, &w->ssm_out, &w->q, &w->k, &w->v, &w->o, &w->router, &w->sh_gate, &w->sh_up, &w->sh_down};
        for (size_t i = 0; i < sizeof(all) / sizeof(all[0]); ++i) {
            const size_t n = (size_t)all[i]->rows * all[i]->cols;
            if (all[i]->buf && all[i]->type != 0u && n > wmax) wmax = n;
        }
    }
    pf->wdq_floats = wmax;
    pf->wdq = pf_buf(m, pf, wmax * sizeof(float));
    const uint32_t qcount = in->n_head * in->head_dim, kvcount = in->n_head_kv * in->head_dim;
    const uint32_t ffn_sh = e->gguf.n_ff_shexp;
    pf->xb = pf_buf(m, pf, hb); pf->normed = pf_buf(m, pf, hb); pf->branch = pf_buf(m, pf, hb); pf->resid = pf_buf(m, pf, hb); pf->ffn_in = pf_buf(m, pf, hb);
    pf->qkv = pf_buf(m, pf, (size_t)in->channels * 4u * B); pf->z = pf_buf(m, pf, (size_t)in->d_inner * 4u * B);
    pf->ba = pf_buf(m, pf, (size_t)2u * in->dt_rank * 4u * B); pf->beta = pf_buf(m, pf, (size_t)in->dt_rank * 4u * B); pf->gate = pf_buf(m, pf, (size_t)in->dt_rank * 4u * B);
    pf->conv_silu = pf_buf(m, pf, (size_t)in->channels * 4u * B);
    pf->q = pf_buf(m, pf, (size_t)in->n_group * in->d_state * 4u * B); pf->k = pf_buf(m, pf, (size_t)in->n_group * in->d_state * 4u * B);
    pf->core = pf_buf(m, pf, (size_t)in->d_inner * 4u * B); pf->ng = pf_buf(m, pf, (size_t)in->d_inner * 4u * B);
    pf->qgate_raw = pf_buf(m, pf, (size_t)2u * qcount * 4u * B); pf->k_raw = pf_buf(m, pf, (size_t)kvcount * 4u * B); pf->value = pf_buf(m, pf, (size_t)kvcount * 4u * B);
    pf->query_rope = pf_buf(m, pf, (size_t)qcount * 4u * B); pf->agate = pf_buf(m, pf, (size_t)qcount * 4u * B); pf->gated = pf_buf(m, pf, (size_t)qcount * 4u * B);
    pf->router_logits = pf_buf(m, pf, (size_t)in->n_expert * 4u * B);
    pf->pred_logits[0] = pf_buf(m, pf, (size_t)in->n_expert * 4u * B);
    pf->pred_logits[1] = pf_buf(m, pf, (size_t)in->n_expert * 4u * B);
    pf->sh_gate = pf_buf(m, pf, (size_t)ffn_sh * 4u * B); pf->sh_up = pf_buf(m, pf, (size_t)ffn_sh * 4u * B); pf->sh_act = pf_buf(m, pf, (size_t)ffn_sh * 4u * B);
    pf->sh_out = pf_buf(m, pf, hb); pf->scalar = pf_buf(m, pf, 4u * B); pf->routed = pf_buf(m, pf, hb);
    if (!pf->wdq || !pf->xb || !pf->normed || !pf->branch || !pf->resid || !pf->ffn_in || !pf->qkv || !pf->z || !pf->ba || !pf->beta || !pf->gate ||
        !pf->conv_silu || !pf->q || !pf->k || !pf->core || !pf->ng || !pf->qgate_raw || !pf->k_raw || !pf->value || !pf->query_rope ||
        !pf->agate || !pf->gated || !pf->router_logits || !pf->pred_logits[0] || !pf->pred_logits[1] || !pf->sh_gate || !pf->sh_up || !pf->sh_act || !pf->sh_out || !pf->scalar || !pf->routed) {
        set_error(error, ecap, "Metal prefill scratch allocation failed"); rl_metal_prefill_destroy(pf); return NULL;
    }
    /* Opt-in only: reading the experts in place from the mmap is numerically identical but measured 4-10x
     * slower than the slot pool on macOS 26 / M4 Max, because Metal re-establishes residency of each
     * layer's whole expert window (~350 MB) for every command buffer (~3.3 s per 48-layer chunk). */
    if (getenv("RL_PREFILL_MAPPED_EXPERTS")) prefill_map_experts(e, m, pf);
    { const char *at = getenv("RL_PREFILL_ATTN_TILE"); pf->attn_tiled = (!at || atoi(at) != 0) && in->head_dim == 256u; }
    /* dev65: on by default (bit-identical to attn_fa_b, prefill attention -20 % at 32K, -29 % at 64K); RL_PREFILL_FA2=0 for A/B */
    { const char *f2 = getenv("RL_PREFILL_FA2"); pf->attn_fa2 = !f2 || atoi(f2) != 0; }
    pf->profile = getenv("RL_PREFILL_PROFILE") != NULL;
    if (m->engine_rs) {
        if (@available(macOS 15.0, *)) {
            id<MTLResidencySet> rs = (id<MTLResidencySet>)m->engine_rs;
            for (id<MTLBuffer> b in pf->keep) [rs addAllocation:b];
            [rs commit];
        }
    }
    { const char *gt = getenv("RL_PREFILL_GEMM_TG"); pf->gemm_tg = !gt || atoi(gt) != 0; }
    { const char *ss = getenv("RL_PREFILL_STATE_SG"); pf->state_sg = (!ss || atoi(ss) != 0) && in->d_state == 128u; }
    return pf;
}

/* dev30 prefill stage profile (development only: every boundary becomes a commit + wait) */
enum { PF_NORM, PF_REC_IN, PF_REC_CORE, PF_REC_OUT, PF_ATT_IN, PF_ATT_CORE, PF_ATT_OUT, PF_RESID, PF_ROUTER, PF_SHARED, PF_NSTAGE };
static const char *const kPfStage[PF_NSTAGE] = {"norm", "rec qkv/z/ba", "rec conv/state/tail", "rec ssm_out", "attn q/k/v",
                                                "attn rope+gqa", "attn o", "resid+norm", "router(+pred)", "shared expert"};
static void pf_split(rl_metal_engine *m, struct rl_metal_prefill *pf, __strong id<MTLCommandBuffer> *cbp, int stage) {
    if (!pf->profile) return;
    [*cbp commit];
    [*cbp waitUntilCompleted];
    pf->prof_ms[stage] += ((*cbp).GPUEndTime - (*cbp).GPUStartTime) * 1000.0;
    *cbp = [m->queue commandBuffer];
}

static id<MTLComputePipelineState> rowsb_pipe(struct rl_metal_prefill *pf, uint32_t type) {
    switch (type) {
        case 0: return pf->p_rowsb_f32;
        case 8: return pf->p_rowsb_q8;
        case 12: return pf->p_rowsb_q4k;
        case 13: return pf->p_rowsb_q5k;
        case 14: return pf->p_rowsb_q6k;
        case 16: return pf->p_rowsb_iq2xxs;
        case 18: case 21: case 22: case 23: return pf->p_rowsb_iq3;   /* dev31 */
        default: return nil;
    }
}

/* out[t][row] = W[row] . x[t] for t < ntok */
static void enc_rowsb(rl_metal_engine *m, struct rl_metal_prefill *pf, id<MTLCommandBuffer> cb, const mweight *w, id<MTLBuffer> x, id<MTLBuffer> out, uint32_t ntok) {
    id<MTLComputePipelineState> p = rowsb_pipe(pf, w->type);
    const uint32_t lanes = lanes_for(w->type, w->cols);
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:p];
    [enc setBuffer:w->buf offset:w->off atIndex:0];
    [enc setBytes:&w->cols length:sizeof(w->cols) atIndex:1];
    [enc setBuffer:x offset:0 atIndex:2];
    [enc setBuffer:out offset:0 atIndex:3];
    [enc setBytes:&w->rows length:sizeof(w->rows) atIndex:4];
    [enc setBytes:&lanes length:sizeof(lanes) atIndex:5];
    [enc setBytes:&ntok length:sizeof(ntok) atIndex:6];
    if (w->type == 16u) [enc setBuffer:m->grid offset:0 atIndex:7];
    if (rl_iq3_supported(w->type)) [enc setBytes:&w->type length:4 atIndex:8];
    const NSUInteger threads = (((NSUInteger)w->rows * lanes) + 31u) & ~(NSUInteger)31u;
    [enc dispatchThreads:MTLSizeMake(threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [enc endEncoding];
}

static id<MTLComputePipelineState> dq_pipe(struct rl_metal_prefill *pf, uint32_t type) {
    switch (type) {
        case 8: return pf->p_dq_q8;
        case 12: return pf->p_dq_q4k;
        case 13: return pf->p_dq_q5k;
        case 14: return pf->p_dq_q6k;
        case 16: return pf->p_dq_iq2xxs;
        case 18: case 21: case 22: case 23: return pf->p_dq_iq3;   /* dev31 */
        default: return nil;
    }
}

/* out[t][row] = W[row] . x[t]: dequantize W to f32 (unless it already is) and run the simdgroup GEMM over the padded chunk */
static void enc_gemm(rl_metal_engine *m, struct rl_metal_prefill *pf, id<MTLCommandBuffer> cb, const mweight *w, id<MTLBuffer> x, id<MTLBuffer> out, uint32_t ntok) {
    if (w->rows % 8u || w->cols % 8u || (w->type != 0u && !dq_pipe(pf, w->type)) || (size_t)w->rows * w->cols > pf->wdq_floats + (w->type == 0u ? SIZE_MAX / 2u : 0u)) {
        enc_rowsb(m, pf, cb, w, x, out, ntok);
        return;
    }
    id<MTLBuffer> wbuf = w->buf;
    NSUInteger woff = w->off;
    if (w->type != 0u) {
        id<MTLComputePipelineState> p = dq_pipe(pf, w->type);
        const uint32_t block = w->type == 8u ? 32u : 256u;
        const uint32_t blocks = w->cols / block;
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:p];
        [enc setBuffer:w->buf offset:w->off atIndex:0];
        [enc setBytes:&w->cols length:4 atIndex:1];
        [enc setBuffer:pf->wdq offset:0 atIndex:2];
        [enc setBytes:&w->rows length:4 atIndex:3];
        if (w->type == 16u) [enc setBuffer:m->grid offset:0 atIndex:4];
        if (rl_iq3_supported(w->type)) {   /* dev31: one thread per 8-value group */
            [enc setBytes:&w->type length:4 atIndex:5];
            [enc dispatchThreads:MTLSizeMake(w->cols / 8u, w->rows, 1) threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
        } else {
            const NSUInteger tgx = blocks < 32u ? blocks : 32u;
            [enc dispatchThreads:MTLSizeMake(blocks, w->rows, 1) threadsPerThreadgroup:MTLSizeMake(tgx, 256u / tgx, 1)];
        }
        [enc endEncoding];
        wbuf = pf->wdq; woff = 0;
    }
    const uint32_t ntok_pad = (ntok + 31u) & ~31u;
    const int tgk = pf->gemm_tg && w->rows % 64u == 0u && w->cols % 32u == 0u;
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:tgk ? pf->p_gemm_tg : pf->p_gemm];
    [enc setBuffer:wbuf offset:woff atIndex:0];
    [enc setBuffer:x offset:0 atIndex:1];
    [enc setBuffer:out offset:0 atIndex:2];
    [enc setBytes:&w->rows length:4 atIndex:3];
    [enc setBytes:&w->cols length:4 atIndex:4];
    [enc setBytes:&ntok_pad length:4 atIndex:5];
    if (tgk) [enc dispatchThreadgroups:MTLSizeMake(w->rows / 64u, ntok_pad / 32u, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    else [enc dispatchThreadgroups:MTLSizeMake((w->rows + 31u) / 32u, ntok_pad / 32u, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [enc endEncoding];
}

static void enc_tg_norm(id<MTLCommandBuffer> cb, id<MTLComputePipelineState> p, id<MTLBuffer> x, const mweight *w, id<MTLBuffer> y, uint32_t n, float eps, uint32_t ntok) {
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:p];
    [enc setBuffer:x offset:0 atIndex:0]; [enc setBuffer:w->buf offset:w->off atIndex:1]; [enc setBuffer:y offset:0 atIndex:2];
    [enc setBytes:&n length:4 atIndex:3]; [enc setBytes:&eps length:4 atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake(ntok, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc endEncoding];
}

static void enc_recurrent_b(rl_engine *e, rl_metal_engine *m, struct rl_metal_prefill *pf, __strong id<MTLCommandBuffer> *cbp,
                            const rl_layer_tensors *t, const mlayer *w, uint32_t ntok) {
    id<MTLCommandBuffer> cb = *cbp;
    const rl_engine_info *in = &e->info;
    const uint32_t r = t->recurrent_index;
    const uint32_t channels = in->channels, dconv = in->d_conv, rank = in->dt_rank, groups = in->n_group, S = in->d_state;
    const uint32_t qk_each = S * groups, d_inner = in->d_inner, head_v = in->head_v;
    const uint32_t kv_ratio = rank / groups;
    const float eps = in->rms_eps;
    enc_gemm(m, pf, cb, &w->qkv, pf->normed, pf->qkv, ntok);
    enc_gemm(m, pf, cb, &w->z, pf->normed, pf->z, ntok);
    enc_gemm(m, pf, cb, &w->ba, pf->normed, pf->ba, ntok);
    pf_split(m, pf, cbp, PF_REC_IN); cb = *cbp;

    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pf->p_dn_ba];
    [enc setBuffer:pf->ba offset:0 atIndex:0]; [enc setBuffer:w->dt.buf offset:w->dt.off atIndex:1]; [enc setBuffer:w->a.buf offset:w->a.off atIndex:2];
    [enc setBuffer:pf->beta offset:0 atIndex:3]; [enc setBuffer:pf->gate offset:0 atIndex:4];
    [enc setBytes:&rank length:4 atIndex:5]; [enc setBytes:&groups length:4 atIndex:6]; [enc setBytes:&ntok length:4 atIndex:7];
    enc_1d(enc, pf->p_dn_ba, (NSUInteger)rank * ntok, 64u); [enc endEncoding];

    enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pf->p_dn_conv];
    [enc setBuffer:m->conv_state[r] offset:0 atIndex:0]; [enc setBuffer:pf->qkv offset:0 atIndex:1]; [enc setBuffer:w->conv.buf offset:w->conv.off atIndex:2];
    [enc setBuffer:pf->conv_silu offset:0 atIndex:3]; [enc setBytes:&channels length:4 atIndex:4]; [enc setBytes:&dconv length:4 atIndex:5]; [enc setBytes:&ntok length:4 atIndex:6];
    enc_1d(enc, pf->p_dn_conv, channels, 64u); [enc endEncoding];

    enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pf->p_dn_l2];
    [enc setBuffer:pf->conv_silu offset:0 atIndex:0]; [enc setBuffer:pf->q offset:0 atIndex:1]; [enc setBuffer:pf->k offset:0 atIndex:2];
    [enc setBytes:&S length:4 atIndex:3]; [enc setBytes:&groups length:4 atIndex:4]; [enc setBytes:&eps length:4 atIndex:5]; [enc setBytes:&channels length:4 atIndex:6];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)2u * groups * ntok, 1, 1) threadsPerThreadgroup:MTLSizeMake(S, 1, 1)]; [enc endEncoding];

    const NSUInteger v_offset = (NSUInteger)(2u * qk_each) * sizeof(float);
    enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pf->state_sg ? pf->p_dn_state_sg : pf->p_dn_state];
    [enc setBuffer:m->rec_state[r] offset:0 atIndex:0]; [enc setBuffer:pf->q offset:0 atIndex:1]; [enc setBuffer:pf->k offset:0 atIndex:2];
    [enc setBuffer:pf->conv_silu offset:v_offset atIndex:3]; [enc setBuffer:pf->gate offset:0 atIndex:4]; [enc setBuffer:pf->beta offset:0 atIndex:5];
    [enc setBuffer:pf->core offset:0 atIndex:6]; [enc setBytes:&S length:4 atIndex:7]; [enc setBytes:&kv_ratio length:4 atIndex:8]; [enc setBytes:&rank length:4 atIndex:9];
    [enc setBytes:&ntok length:4 atIndex:10]; [enc setBytes:&qk_each length:4 atIndex:11]; [enc setBytes:&channels length:4 atIndex:12]; [enc setBytes:&d_inner length:4 atIndex:13];
    if (pf->state_sg) [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)rank * S / 4u, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    else [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)rank * S, 1, 1) threadsPerThreadgroup:MTLSizeMake(S, 1, 1)];
    [enc endEncoding];

    enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pf->p_dn_tail];
    [enc setBuffer:pf->core offset:0 atIndex:0]; [enc setBuffer:pf->z offset:0 atIndex:1]; [enc setBuffer:w->ssm_norm.buf offset:w->ssm_norm.off atIndex:2];
    [enc setBuffer:pf->ng offset:0 atIndex:3]; [enc setBytes:&eps length:4 atIndex:4]; [enc setBytes:&head_v length:4 atIndex:5]; [enc setBytes:&rank length:4 atIndex:6];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)rank * ntok, 1, 1) threadsPerThreadgroup:MTLSizeMake(head_v, 1, 1)]; [enc endEncoding];
    pf_split(m, pf, cbp, PF_REC_CORE); cb = *cbp;

    enc_gemm(m, pf, cb, &w->ssm_out, pf->ng, pf->branch, ntok);
    pf_split(m, pf, cbp, PF_REC_OUT);
}

static void enc_attention_b(rl_engine *e, rl_metal_engine *m, struct rl_metal_prefill *pf, __strong id<MTLCommandBuffer> *cbp,
                            const rl_layer_tensors *t, const mlayer *w, uint32_t position0, uint32_t ntok) {
    id<MTLCommandBuffer> cb = *cbp;
    const rl_engine_info *in = &e->info;
    const uint32_t a = t->attention_index;
    const uint32_t head_dim = in->head_dim, qheads = in->n_head, kvheads = in->n_head_kv, rope_dims = in->rope_dims;
    const float eps = in->rms_eps, base = in->rope_freq_base;
    enc_gemm(m, pf, cb, &w->q, pf->normed, pf->qgate_raw, ntok);
    enc_gemm(m, pf, cb, &w->k, pf->normed, pf->k_raw, ntok);
    enc_gemm(m, pf, cb, &w->v, pf->normed, pf->value, ntok);
    pf_split(m, pf, cbp, PF_ATT_IN); cb = *cbp;
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pf->p_attn_prep];
    [enc setBuffer:pf->qgate_raw offset:0 atIndex:0]; [enc setBuffer:pf->k_raw offset:0 atIndex:1]; [enc setBuffer:pf->value offset:0 atIndex:2];
    [enc setBuffer:w->q_norm.buf offset:w->q_norm.off atIndex:3]; [enc setBuffer:w->k_norm.buf offset:w->k_norm.off atIndex:4];
    [enc setBuffer:pf->query_rope offset:0 atIndex:5]; [enc setBuffer:pf->agate offset:0 atIndex:6];
    [enc setBuffer:m->kcache[a] offset:0 atIndex:7]; [enc setBuffer:m->vcache[a] offset:0 atIndex:8];
    [enc setBytes:&head_dim length:4 atIndex:9]; [enc setBytes:&qheads length:4 atIndex:10]; [enc setBytes:&kvheads length:4 atIndex:11];
    [enc setBytes:&position0 length:4 atIndex:12]; [enc setBytes:&rope_dims length:4 atIndex:13]; [enc setBytes:&base length:4 atIndex:14]; [enc setBytes:&eps length:4 atIndex:15];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)(qheads + kvheads) * ntok, 1, 1) threadsPerThreadgroup:MTLSizeMake(head_dim, 1, 1)]; [enc endEncoding];

    enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pf->attn_tiled ? (pf->attn_fa2 ? pf->p_attn_fa2 : pf->p_attn_fa) : pf->p_attn_gqa];
    [enc setBuffer:pf->query_rope offset:0 atIndex:0]; [enc setBuffer:m->kcache[a] offset:0 atIndex:1]; [enc setBuffer:m->vcache[a] offset:0 atIndex:2];
    [enc setBuffer:pf->agate offset:0 atIndex:3]; [enc setBuffer:pf->gated offset:0 atIndex:4];
    [enc setBytes:&head_dim length:4 atIndex:5]; [enc setBytes:&qheads length:4 atIndex:6]; [enc setBytes:&kvheads length:4 atIndex:7]; [enc setBytes:&position0 length:4 atIndex:8];
    if (pf->attn_tiled) {
        /* dev30: (head, 8-token tile) threadgroups; key blocks of 32 may read up to 31 positions past the last
         * token, which the KV cache allocation pads (RL_ENGINE_KV_PAD) and whose weights are zero */
        [enc setBytes:&ntok length:4 atIndex:9];
        const uint32_t tile = pf->attn_fa2 ? 16u : 8u;
        [enc dispatchThreadgroups:MTLSizeMake(qheads, (ntok + tile - 1u) / tile, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    } else {
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)qheads * ntok, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
    [enc endEncoding];
    pf_split(m, pf, cbp, PF_ATT_CORE); cb = *cbp;

    enc_gemm(m, pf, cb, &w->o, pf->gated, pf->branch, ntok);
    pf_split(m, pf, cbp, PF_ATT_OUT);
}

/* experts per plan: bounded by the plan table and by half the cache so a group never evicts its own experts */
static uint32_t prefill_plan_limit(rl_metal_engine *m) {
    const uint32_t capacity = rl_native_metal_slot_capacity(m->experts);
    uint32_t limit = 512u;
    if (capacity / 2u < limit) limit = capacity / 2u;
    return limit < 10u ? 10u : limit;
}

/* dev24: softmax top-k selection of B tokens' router logits, spread over the CPU cores (tokens are independent;
 * each selection is the unchanged single-token routine) */
static int route_tokens(const float *logits, uint32_t B, uint32_t experts, uint32_t topk, uint32_t *ids, float *weights, char *error, size_t cap) {
    if (experts > 1024u) { set_error(error, cap, "batched router selection supports at most 1024 experts"); return 0; }
    enum { RL_ROUTE_STRIDE = 32 };
    const size_t n_blocks = ((size_t)B + RL_ROUTE_STRIDE - 1u) / RL_ROUTE_STRIDE;
    uint8_t *bad = (uint8_t *)calloc(n_blocks, 1);
    if (!bad) { set_error(error, cap, "router selection scratch allocation failed"); return 0; }
    dispatch_apply(n_blocks, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t blk) {
        float probs[1024];
        const uint32_t t0 = (uint32_t)(blk * RL_ROUTE_STRIDE), t1 = t0 + RL_ROUTE_STRIDE < B ? t0 + RL_ROUTE_STRIDE : B;
        for (uint32_t tk = t0; tk < t1; ++tk) {
            if (!rl_native_router_select_softmax_topk(logits + (size_t)tk * experts, experts, topk,
                    ids + (size_t)tk * RL_ENGINE_MAX_TOPK, weights + (size_t)tk * RL_ENGINE_MAX_TOPK, probs, NULL, 0)) { bad[blk] = 1; return; }
        }
    });
    int ok = 1;
    for (size_t blk = 0; blk < n_blocks && ok; ++blk) {
        if (!bad[blk]) continue;
        /* repeat the failing block serially for its error message */
        float probs[1024];
        for (uint32_t tk = (uint32_t)(blk * RL_ROUTE_STRIDE); tk < B && ok; ++tk)
            ok = rl_native_router_select_softmax_topk(logits + (size_t)tk * experts, experts, topk,
                    ids + (size_t)tk * RL_ENGINE_MAX_TOPK, weights + (size_t)tk * RL_ENGINE_MAX_TOPK, probs, error, cap);
        if (ok) { set_error(error, cap, "batched router selection failed"); ok = 0; }
    }
    free(bad);
    return ok;
}

/* dev24: loads (prepare, then immediate release) the union of the experts layer l_next is predicted to select
 * for the chunk, while the current layer's experts run. Skipped when the cache cannot hold both sets. */
static int prefill_prefetch(rl_engine *e, rl_metal_engine *m, const float *pred_logits, uint32_t l_next, uint32_t B, uint32_t in_flight,
                            uint32_t *ids, float *weights, uint16_t *umap, uint32_t *uid, float *zero_w, char *error, size_t cap) {
    const uint32_t experts = e->info.n_expert, topk = e->info.top_k;
    uint32_t U = 0;
    if (!route_tokens(pred_logits, B, experts, topk, ids, weights, error, cap)) return 0;
    for (uint32_t tk = 0; tk < B; ++tk) {
        const uint32_t *tid = ids + (size_t)tk * RL_ENGINE_MAX_TOPK;
        for (uint32_t k = 0; k < topk; ++k) if (umap[tid[k]] == UINT16_MAX) { umap[tid[k]] = (uint16_t)U; uid[U++] = tid[k]; }
    }
    for (uint32_t u = 0; u < U; ++u) { umap[uid[u]] = UINT16_MAX; zero_w[u] = 0.0f; }
    if ((uint64_t)in_flight + U + topk > rl_native_metal_layer_capacity(m->experts, l_next)) return 1;   /* dev43: per size class */
    rl_native_topk_plan pplan;
    memset(&pplan, 0, sizeof(pplan));
    return rl_native_metal_prepare_topk(m->experts, &e->expert_map, l_next, uid, zero_w, U, &pplan, error, cap) &&
           rl_native_metal_release_topk(m->experts, &pplan, NULL, error, cap);
}

/* waits for the background prefetch, then releases the previous layer's deferred expert plan */
static int join_prefetch(dispatch_group_t group, int *pending, const int *prefetch_ok, const char *prefetch_error, rl_native_topk_plan *plan,
                         rl_metal_engine *m, rl_engine_step_stats *stats, const double *prefetch_ms, char *error, size_t cap) {
    dispatch_group_wait(group, DISPATCH_TIME_FOREVER);
    *pending = 0;
    stats->prefetch_ms += *prefetch_ms;
    if (!*prefetch_ok) { snprintf(error, cap, "prefill expert prefetch failed: %s", prefetch_error); return 0; }
    return rl_native_metal_release_topk(m->experts, plan, NULL, error, cap);
}

static int prefill_chunk(rl_engine *e, rl_metal_engine *m, struct rl_metal_prefill *pf, const uint32_t *tokens, uint32_t B,
                         int want_logits, float *logits, rl_engine_step_stats *stats, char *error, size_t cap) {
    rl_backend_state *s = &e->gpu;
    const rl_engine_info *in = &e->info;
    const uint32_t hidden = in->hidden, experts = in->n_expert, topk = in->top_k;
    const float eps = in->rms_eps;
    const size_t hb = (size_t)hidden * sizeof(float);
    const uint32_t p0 = s->position;
    const double start = rl_engine_now_ms();
    float *probs = (float *)malloc((size_t)experts * sizeof(float));
    uint32_t *ids = (uint32_t *)malloc((size_t)B * RL_ENGINE_MAX_TOPK * sizeof(uint32_t));
    float *weights = (float *)malloc((size_t)B * RL_ENGINE_MAX_TOPK * sizeof(float));
    const size_t max_pairs = (size_t)B * topk;
    uint16_t *umap = (uint16_t *)malloc((size_t)experts * sizeof(uint16_t));
    uint32_t *uid = (uint32_t *)malloc((size_t)experts * sizeof(uint32_t));
    uint32_t *expert_start = (uint32_t *)malloc(((size_t)experts + 1u) * sizeof(uint32_t));
    uint32_t *fill = (uint32_t *)malloc((size_t)experts * sizeof(uint32_t));
    float *dummy_w = (float *)malloc((size_t)experts * sizeof(float));
    uint32_t *pair_token = (uint32_t *)malloc(max_pairs * sizeof(uint32_t));
    float *pair_weight = (float *)malloc(max_pairs * sizeof(float));
    uint32_t *tok_pair = (uint32_t *)malloc(max_pairs * sizeof(uint32_t));
    if (!probs || !ids || !weights || !umap || !uid || !expert_start || !fill || !dummy_w || !pair_token || !pair_weight || !tok_pair) {
        free(probs); free(ids); free(weights); free(umap); free(uid); free(expert_start); free(fill); free(dummy_w); free(pair_token); free(pair_weight); free(tok_pair);
        set_error(error, cap, "prefill scratch allocation failed"); return 0;
    }
    for (uint32_t i = 0; i < experts; ++i) umap[i] = UINT16_MAX;
    rl_native_topk_plan plan;
    memset(&plan, 0, sizeof(plan));
    rl_native_metal_telemetry tel;
    memset(&tel, 0, sizeof(tel));
    dispatch_group_t prefetch_group = dispatch_group_create();
    __block int prefetch_ok = 1;
    __block double prefetch_ms = 0.0;
    char prefetch_error[256] = { 0 };
    int prefetch_pending = 0;
    int ok = 0;
    const double read_before = m->last_read_ms;
    double p_lru0, p_load0, p_commit0;
    rl_native_metal_prepare_profile(m->experts, &p_lru0, &p_load0, &p_commit0);
    @autoreleasepool {
        for (uint32_t t = 0; t < B; ++t) {
            if (!rl_engine_embed_token(e, tokens[t], (float *)pf->xb.contents + (size_t)t * hidden, error, cap)) goto done;
        }
        memcpy(s->embed, (const float *)pf->xb.contents + (size_t)(B - 1u) * hidden, hb);
        stats->embed_ms += rl_engine_now_ms() - start;

        /* dev72 (dev only): RL_GPU_CAPTURE=FILE.gputrace records layer RL_GPU_CAPTURE_LAYER (default 3) of chunk
         * RL_GPU_CAPTURE_SKIP (default 0) as a Metal GPU trace for Xcode; needs MTL_CAPTURE_ENABLED=1. A capture holds every
         * live buffer of the device (here the whole model), so Xcode profiles it in lite mode only: for the full shader
         * profiler use redlite-engine prefill-attn-bench, which loads no model. */
        static int captured = 0, chunks_seen = 0;
        const int cap_skip = getenv("RL_GPU_CAPTURE_SKIP") ? atoi(getenv("RL_GPU_CAPTURE_SKIP")) : 0;
        const char *cap_path = captured || chunks_seen++ < cap_skip || getenv("RL_GPU_CAPTURE_STEP") ? NULL : getenv("RL_GPU_CAPTURE");
        const uint32_t cap_layer = getenv("RL_GPU_CAPTURE_LAYER") ? (uint32_t)atoi(getenv("RL_GPU_CAPTURE_LAYER")) : 3u;
        id<MTLCaptureScope> cap_scope = nil;
        for (uint32_t l = 0; l < in->n_layer; ++l) {
            const rl_layer_tensors *t = &e->layers[l];
            const mlayer *w = &m->layers[l];
            const double l0 = rl_engine_now_ms();
            if (cap_path && l == cap_layer) {
                cap_scope = [[MTLCaptureManager sharedCaptureManager] newCaptureScopeWithCommandQueue:m->queue];
                MTLCaptureDescriptor *d = [[MTLCaptureDescriptor alloc] init];
                d.captureObject = cap_scope;
                d.destination = MTLCaptureDestinationGPUTraceDocument;
                d.outputURL = [NSURL fileURLWithPath:[NSString stringWithUTF8String:cap_path]];
                NSError *ce = nil;
                if ([[MTLCaptureManager sharedCaptureManager] startCaptureWithDescriptor:d error:&ce]) [cap_scope beginScope];
                else { fprintf(stderr, "GPU capture failed: %s\n", ce.localizedDescription.UTF8String ?: "unknown"); cap_scope = nil; }
            }
            id<MTLCommandBuffer> cb = [m->queue commandBuffer];
            enc_tg_norm(cb, pf->p_rms, pf->xb, &w->attn_norm, pf->normed, hidden, eps, B);
            pf_split(m, pf, &cb, PF_NORM);
            if (t->kind == RL_LAYER_MAP_RECURRENT) enc_recurrent_b(e, m, pf, &cb, t, w, B);
            else enc_attention_b(e, m, pf, &cb, t, w, p0, B);
            {
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pf->p_resid_rms];
                [enc setBuffer:pf->xb offset:0 atIndex:0]; [enc setBuffer:pf->branch offset:0 atIndex:1];
                [enc setBuffer:w->post_norm.buf offset:w->post_norm.off atIndex:2];
                [enc setBuffer:pf->resid offset:0 atIndex:3]; [enc setBuffer:pf->ffn_in offset:0 atIndex:4];
                [enc setBytes:&hidden length:4 atIndex:5]; [enc setBytes:&eps length:4 atIndex:6];
                [enc dispatchThreadgroups:MTLSizeMake(B, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)]; [enc endEncoding];
            }
            pf_split(m, pf, &cb, PF_RESID);
            enc_gemm(m, pf, cb, &w->router, pf->ffn_in, pf->router_logits, B);
            /* dev24 pre-gating: layer l+1's router on layer l's FFN input predicts the experts to load during experts_l */
            /* dev30: with every expert preloaded there is nothing to prefetch (no predictor GEMM, no loader thread) */
            const int predict = m->prefetch && !m->preloaded && !pf->mapped && l + 1u < in->n_layer;
            if (predict) enc_gemm(m, pf, cb, &m->layers[l + 1u].router, pf->ffn_in, pf->pred_logits[l & 1u], B);
            pf_split(m, pf, &cb, PF_ROUTER);
            {
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pf->p_sh_scalar];
                [enc setBuffer:w->sh_gate_inp.buf offset:w->sh_gate_inp.off atIndex:0]; [enc setBuffer:pf->ffn_in offset:0 atIndex:1];
                [enc setBuffer:pf->scalar offset:0 atIndex:2]; [enc setBytes:&hidden length:4 atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(B, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)]; [enc endEncoding];
            }
            enc_gemm(m, pf, cb, &w->sh_gate, pf->ffn_in, pf->sh_gate, B);
            enc_gemm(m, pf, cb, &w->sh_up, pf->ffn_in, pf->sh_up, B);
            {
                const uint32_t count = w->sh_gate.rows * B;
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pf->p_sh_silu];
                [enc setBuffer:pf->sh_gate offset:0 atIndex:0]; [enc setBuffer:pf->sh_up offset:0 atIndex:1]; [enc setBuffer:pf->sh_act offset:0 atIndex:2];
                [enc setBytes:&count length:4 atIndex:3]; enc_1d(enc, pf->p_sh_silu, count, 64u); [enc endEncoding];
            }
            enc_gemm(m, pf, cb, &w->sh_down, pf->sh_act, pf->sh_out, B);
            if (pf->profile) {
                const double g0 = stats->gpu_ms;
                if (!commit_wait(cb, "prefill layer dense", &stats->gpu_ms, error, cap)) goto done;
                pf->prof_ms[PF_SHARED] += stats->gpu_ms - g0;
            } else if (!commit_wait(cb, "prefill layer dense", &stats->gpu_ms, error, cap)) goto done;
            const double l1 = rl_engine_now_ms();
            if (t->kind == RL_LAYER_MAP_RECURRENT) stats->recurrent_ms += l1 - l0; else stats->attention_ms += l1 - l0;

            if (prefetch_pending && !join_prefetch(prefetch_group, &prefetch_pending, &prefetch_ok, prefetch_error, &plan, m, stats, &prefetch_ms, error, cap)) goto done;
            /* routing: every token of the chunk */
            if (!route_tokens((const float *)pf->router_logits.contents, B, experts, topk, ids, weights, error, cap)) goto done;
            memcpy(s->router_ids + (size_t)l * RL_ENGINE_MAX_TOPK, ids + (size_t)(B - 1u) * RL_ENGINE_MAX_TOPK, (size_t)topk * sizeof(uint32_t));
            const double l2 = rl_engine_now_ms();
            stats->router_ms += l2 - l1;

            /* routed experts (dev20b): the chunk's tokens are split into consecutive groups whose union of
             * selected experts fits one pool plan; each group runs three batched dispatches (gate/up, down,
             * per-token weighted sum). The residual add of the whole chunk is encoded after the last group. */
            const uint32_t plan_limit = pf->mapped ? (experts < 512u ? experts : 512u) : prefill_plan_limit(m);
            for (uint32_t g0 = 0; g0 < B;) {
                uint32_t g1 = g0, U = 0;
                /* grow the group while the union stays within the plan limit (always at least one token) */
                while (g1 < B) {
                    uint32_t added = 0;
                    for (uint32_t k = 0; k < topk; ++k) {
                        const uint32_t ex = ids[(size_t)g1 * RL_ENGINE_MAX_TOPK + k];
                        if (umap[ex] == UINT16_MAX) { umap[ex] = (uint16_t)(U + added); uid[U + added] = ex; added++; }
                    }
                    if (U + added > plan_limit && g1 > g0) {
                        for (uint32_t k = 0; k < added; ++k) umap[uid[U + k]] = UINT16_MAX;
                        break;
                    }
                    U += added; g1++;
                }
                const uint32_t ntok_g = g1 - g0, P = ntok_g * topk;
                /* pairs sorted by expert: counts -> prefix sums -> placement in token/selection order */
                for (uint32_t u = 0; u <= U; ++u) expert_start[u] = 0;
                for (uint32_t t = g0; t < g1; ++t) for (uint32_t k = 0; k < topk; ++k) expert_start[umap[ids[(size_t)t * RL_ENGINE_MAX_TOPK + k]] + 1u]++;
                for (uint32_t u = 0; u < U; ++u) expert_start[u + 1u] += expert_start[u];
                for (uint32_t u = 0; u < U; ++u) fill[u] = expert_start[u];
                for (uint32_t t = g0; t < g1; ++t) {
                    for (uint32_t k = 0; k < topk; ++k) {
                        const uint32_t u = umap[ids[(size_t)t * RL_ENGINE_MAX_TOPK + k]];
                        const uint32_t pidx = fill[u]++;
                        pair_token[pidx] = t;
                        pair_weight[pidx] = weights[(size_t)t * RL_ENGINE_MAX_TOPK + k];
                        tok_pair[(size_t)(t - g0) * topk + k] = pidx;
                    }
                }
                for (uint32_t u = 0; u < U; ++u) { umap[uid[u]] = UINT16_MAX; dummy_w[u] = 0.0f; }
                cb = [m->queue commandBuffer];
                if (pf->mapped) {
                    if (!redmetal_topk_pool_encode_mapped(rl_native_metal_pool_handle(m->experts), (__bridge void *)cb,
                            (__bridge void *)pf->exp_gate[l], pf->exp_gate_addr0[l], (__bridge void *)pf->exp_up[l], pf->exp_up_addr0[l],
                            (__bridge void *)pf->exp_down[l], pf->exp_down_addr0[l], pf->exp_gate_bytes[l], pf->exp_up_bytes[l], pf->exp_down_bytes[l],
                            uid, U, pf->exp_type[l], hidden, e->gguf.n_ff_exp, g0, ntok_g, topk, P, pair_token, pair_weight, expert_start, tok_pair,
                            (__bridge void *)pf->ffn_in, 0u, (__bridge void *)pf->routed, 0u)) {
                        snprintf(error, cap, "mapped expert encode failed: %s", redmetal_topk_last_error());
                        goto done;
                    }
                } else {
                    if (!rl_native_metal_prepare_topk(m->experts, &e->expert_map, l, uid, dummy_w, U, &plan, error, cap)) goto done;
                    if (!rl_native_metal_encode_topk_batched(m->experts, &plan, (__bridge void *)cb, g0, ntok_g, topk,
                            pair_token, pair_weight, expert_start, tok_pair, (__bridge void *)pf->ffn_in, 0u,
                            (__bridge void *)pf->routed, 0u, error, cap)) goto done;
                }
                if (g1 == B) {
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    [enc setComputePipelineState:pf->p_scale_add];
                    [enc setBuffer:pf->resid offset:0 atIndex:0]; [enc setBuffer:pf->routed offset:0 atIndex:1]; [enc setBuffer:pf->sh_out offset:0 atIndex:2];
                    [enc setBuffer:pf->scalar offset:0 atIndex:3]; [enc setBuffer:pf->xb offset:0 atIndex:4]; [enc setBytes:&hidden length:4 atIndex:5]; [enc setBytes:&B length:4 atIndex:6];
                    enc_1d(enc, pf->p_scale_add, (NSUInteger)hidden * B, 64u); [enc endEncoding];
                }
                const double w0 = rl_engine_now_ms();
                if (predict && g1 == B) {
                    /* the experts run on the GPU while the CPU loads the next layer's predicted union */
                    [cb commit];
                    /* ... and on through the next layer's dense pass: the prefetch is joined before that layer routes.
                     * This layer's plan stays pinned (release deferred) so the prefetch can never evict it. */
                    const uint32_t l_next = l + 1u;
                    /* pinned slots of this layer only compete with the prefetch inside one size class */
                    const uint32_t in_flight = rl_native_metal_layer_class(m->experts, l) == rl_native_metal_layer_class(m->experts, l_next) ? U : 0u;
                    uint32_t *p_ids = ids; float *p_w = weights, *p_zero = dummy_w;
                    const float *p_pred = (const float *)pf->pred_logits[l & 1u].contents; uint16_t *p_umap = umap; uint32_t *p_uid = uid;
                    char *p_err = prefetch_error;
                    prefetch_pending = 1;
                    dispatch_group_async(prefetch_group, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
                        const double p0 = rl_engine_now_ms();
                        prefetch_ok = prefill_prefetch(e, m, p_pred, l_next, B, in_flight, p_ids, p_w, p_umap, p_uid, p_zero, p_err, 256);
                        prefetch_ms = rl_engine_now_ms() - p0;
                    });
                    [cb waitUntilCompleted];
                    if (cb.status != MTLCommandBufferStatusCompleted || cb.error) {
                        snprintf(error, cap, "prefill routed experts command buffer failed: %s", cb.error.localizedDescription.UTF8String ?: "unknown");
                        goto done;
                    }
                    stats->routed_gpu_ms += (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
                } else if (!commit_wait(cb, "prefill routed experts", &stats->routed_gpu_ms, error, cap)) goto done;
                stats->expert_wait_ms += rl_engine_now_ms() - w0;
                if (!pf->mapped && !prefetch_pending) {
                    const int last_plan = l + 1u == in->n_layer && g1 == B;
                    if (!rl_native_metal_release_topk(m->experts, &plan, last_plan ? &tel : NULL, error, cap)) goto done;
                }
                stats->expert_plans++;
                g0 = g1;
            }
            const double l3 = rl_engine_now_ms();
            stats->routed_ms += l3 - l2;
            memcpy(s->layer_out + (size_t)l * hidden, (const float *)pf->xb.contents + (size_t)(B - 1u) * hidden, hb);
            if (cap_scope && l == cap_layer) {
                [cap_scope endScope];
                [[MTLCaptureManager sharedCaptureManager] stopCapture];
                fprintf(stderr, "GPU capture of layer %u written to %s\n", l, cap_path);
                cap_scope = nil; captured = 1;
            }
        }
        stats->layers_ms += rl_engine_now_ms() - start - stats->embed_ms;
        {
            /* pool counters are read per chunk, after every prefetch has joined (the prefetch thread updates them);
             * they include the prefetch's own lookups and loads */
            double p_lru1, p_load1, p_commit1;
            rl_native_metal_prepare_profile(m->experts, &p_lru1, &p_load1, &p_commit1);
            stats->prep_lru_ms += p_lru1 - p_lru0; stats->prep_load_ms += p_load1 - p_load0; stats->prep_commit_ms += p_commit1 - p_commit0;
            const double read_now = rl_native_metal_read_ms(m->experts);
            stats->routed_load_ms += read_now - read_before;
            m->last_read_ms = read_now;
        }

        /* output head on the last token of the chunk; m->x keeps that hidden state like a decode step would */
        const double o0 = rl_engine_now_ms();
        memcpy(m->x.contents, (const float *)pf->xb.contents + (size_t)(B - 1u) * hidden, hb);
        id<MTLCommandBuffer> cb = [m->queue commandBuffer];
        enc_rms(m, cb, m->x, &m->output_norm, m->final_norm, hidden, eps);
        if (want_logits) enc_rows(m, cb, &m->output, m->final_norm, m->logits);
        if (!commit_wait(cb, "prefill output head", &stats->gpu_ms, error, cap)) goto done;
        stats->expert_loads = tel.expert_loads; stats->cache_hits = tel.cache_hits; stats->cache_misses = tel.cache_misses;
        stats->ssd_bytes = tel.bytes_read_total; stats->ssd_reads = tel.read_calls_total;
        stats->resident_slots = tel.resident_slots; stats->slot_capacity = tel.slot_capacity;
        memcpy(s->final_norm, m->final_norm.contents, hb);
        if (want_logits && logits) memcpy(logits, m->logits.contents, (size_t)in->vocab * sizeof(float));
        stats->output_ms += rl_engine_now_ms() - o0;
        s->position += B;
        ok = 1;
    }
done:
    if (prefetch_pending) {
        dispatch_group_wait(prefetch_group, DISPATCH_TIME_FOREVER);
        prefetch_pending = 0;
    }
    if (plan.active) {
        id<MTLCommandBuffer> drain = [m->queue commandBuffer];
        [drain commit];
        [drain waitUntilCompleted];
        rl_native_metal_release_topk(m->experts, &plan, NULL, NULL, 0);
    }
    stats->total_ms += rl_engine_now_ms() - start;
    free(probs); free(ids); free(weights); free(umap); free(uid); free(expert_start); free(fill); free(dummy_w); free(pair_token); free(pair_weight); free(tok_pair);
    return ok;
}

int rl_metal_engine_prefill(rl_engine *e, rl_metal_engine *m, const uint32_t *tokens, uint32_t count, float *logits,
                            rl_engine_step_stats *stats, char *error, size_t cap) {
    if (!e || !m || !tokens || !count) { set_error(error, cap, "invalid prefill request"); return 0; }
    const uint32_t batch = rl_engine_prefill_batch(e);
    if (!m->pf || m->pf->cap < batch) {
        if (m->pf) {
            if (m->engine_rs) {
                if (@available(macOS 15.0, *)) {
                    id<MTLResidencySet> rs = (id<MTLResidencySet>)m->engine_rs;
                    for (id<MTLBuffer> b in m->pf->keep) [rs removeAllocation:b];
                    [rs commit];
                }
            }
            rl_metal_prefill_destroy(m->pf); m->pf = NULL;
        }
        m->pf = prefill_create(e, m, batch, error, cap);
        if (!m->pf) return 0;
    }
    for (uint32_t done = 0; done < count;) {
        const uint32_t B = count - done < batch ? count - done : batch;
        const int last = done + B == count;
        if (!prefill_chunk(e, m, m->pf, tokens + done, B, last && logits != NULL, last ? logits : NULL, stats, error, cap)) return 0;
        done += B;
    }
    if (m->pf->profile) {
        fprintf(stderr, "--- prefill GPU stage profile (cumulative ms, RL_PREFILL_PROFILE) ---\n");
        for (int i = 0; i < PF_NSTAGE; ++i) fprintf(stderr, "  %-20s %9.1f ms\n", kPfStage[i], m->pf->prof_ms[i]);
    }
    if (error && cap) error[0] = '\0';
    return 1;
}

static id<MTLBuffer> rnd_buf(id<MTLDevice> dev, size_t n) {
    id<MTLBuffer> b = [dev newBufferWithLength:n * 4u options:MTLResourceStorageModeShared];
    if (b) { float *p = (float *)b.contents; for (size_t i = 0; i < n; ++i) p[i] = (float)rand() / (float)RAND_MAX - 0.5f; }
    return b;
}
/* dev74: n random KV cache elements of the engine's KV type (half or float) */
static id<MTLBuffer> rnd_kv_buf(id<MTLDevice> dev, size_t n) {
    if (rl_kv_elem_bytes() == 4u) return rnd_buf(dev, n);
    id<MTLBuffer> b = [dev newBufferWithLength:n * 2u options:MTLResourceStorageModeShared];
    if (b) { _Float16 *p = (_Float16 *)b.contents; for (size_t i = 0; i < n; ++i) p[i] = (_Float16)((float)rand() / (float)RAND_MAX - 0.5f); }
    return b;
}

/* dev72 (dev only): the batched-prefill attention (attn_fa_b2) and routed-expert kernels of layer `layer` on buffers of
 * their own (B tokens at position0 over a random KV cache; B x top_k uniformly random experts on the layer's mapped
 * expert window), for timing. */
int rl_metal_prefill_kernel_bench(rl_engine *e, rl_metal_engine *m, uint32_t B, uint32_t position0, uint32_t layer,
                                  char *error, size_t cap) {
    const rl_engine_info *in = &e->info;
    if (!m->pf || m->pf->cap < B) {
        if (m->pf) { rl_metal_prefill_destroy(m->pf); m->pf = NULL; }
        m->pf = prefill_create(e, m, B, error, cap);
        if (!m->pf) return 0;
    }
    struct rl_metal_prefill *pf = m->pf;
    if (!pf->mapped) prefill_map_experts(e, m, pf);   /* the same expert kernels as the pool path, on the layer's mmap window */
    if (!pf->mapped) { set_error(error, cap, "prefill-kernel-bench could not map the routed experts"); return 0; }
    if (layer >= in->n_layer) { set_error(error, cap, "layer out of range"); return 0; }
    const uint32_t head_dim = in->head_dim, qheads = in->n_head, kvheads = in->n_head_kv, hidden = in->hidden;
    const uint32_t experts = in->n_expert, topk = in->top_k, ffn = e->gguf.n_ff_exp, kvlen = position0 + B + 64u;
    srand(72);
    id<MTLBuffer> q = rnd_buf(m->dev, (size_t)B * qheads * head_dim), g = rnd_buf(m->dev, (size_t)B * qheads * head_dim);
    id<MTLBuffer> kc = rnd_buf(m->dev, (size_t)kvlen * kvheads * head_dim), vc = rnd_buf(m->dev, (size_t)kvlen * kvheads * head_dim);
    id<MTLBuffer> o = [m->dev newBufferWithLength:(size_t)B * qheads * head_dim * 4u options:MTLResourceStorageModeShared];
    id<MTLBuffer> xin = rnd_buf(m->dev, (size_t)B * hidden);
    id<MTLBuffer> xout = [m->dev newBufferWithLength:(size_t)B * hidden * 4u options:MTLResourceStorageModeShared];
    if (!q || !g || !kc || !vc || !o || !xin || !xout) { set_error(error, cap, "bench buffer allocation failed"); return 0; }
    const uint32_t P = B * topk;
    uint32_t *ids = (uint32_t *)malloc((size_t)P * 4u), *pair_token = (uint32_t *)malloc((size_t)P * 4u), *tok_pair = (uint32_t *)malloc((size_t)P * 4u);
    uint32_t *uid = (uint32_t *)malloc((size_t)experts * 4u), *expert_start = (uint32_t *)calloc(experts + 1u, 4u), *fill = (uint32_t *)malloc((size_t)experts * 4u);
    int32_t *umap = (int32_t *)malloc((size_t)experts * 4u);
    float *pair_weight = (float *)malloc((size_t)P * 4u);
    int ok = 0;
    if (!ids || !pair_token || !tok_pair || !uid || !expert_start || !fill || !umap || !pair_weight) { set_error(error, cap, "bench allocation failed"); goto out; }
    for (uint32_t t = 0; t < B; ++t)
        for (uint32_t k = 0; k < topk; ++k) {
            uint32_t x, dup;
            do { x = (uint32_t)rand() % experts; dup = 0; for (uint32_t j = 0; j < k; ++j) dup |= ids[t * topk + j] == x; } while (dup);
            ids[t * topk + k] = x;
        }
    for (uint32_t x = 0; x < experts; ++x) umap[x] = -1;
    uint32_t U = 0;
    for (uint32_t i = 0; i < P; ++i) if (umap[ids[i]] < 0) { umap[ids[i]] = (int32_t)U; uid[U++] = ids[i]; }
    for (uint32_t i = 0; i < P; ++i) expert_start[umap[ids[i]] + 1]++;
    for (uint32_t u = 0; u < U; ++u) expert_start[u + 1u] += expert_start[u];
    for (uint32_t u = 0; u < U; ++u) fill[u] = expert_start[u];
    for (uint32_t t = 0; t < B; ++t)
        for (uint32_t k = 0; k < topk; ++k) {
            const uint32_t pidx = fill[umap[ids[t * topk + k]]]++;
            pair_token[pidx] = t; pair_weight[pidx] = 1.0f / (float)topk; tok_pair[t * topk + k] = pidx;
        }
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [m->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pf->p_attn_fa2];
        [enc setBuffer:q offset:0 atIndex:0]; [enc setBuffer:kc offset:0 atIndex:1]; [enc setBuffer:vc offset:0 atIndex:2];
        [enc setBuffer:g offset:0 atIndex:3]; [enc setBuffer:o offset:0 atIndex:4];
        [enc setBytes:&head_dim length:4 atIndex:5]; [enc setBytes:&qheads length:4 atIndex:6]; [enc setBytes:&kvheads length:4 atIndex:7];
        [enc setBytes:&position0 length:4 atIndex:8]; [enc setBytes:&B length:4 atIndex:9];
        [enc dispatchThreadgroups:MTLSizeMake(qheads, (B + 15u) / 16u, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [enc endEncoding];
        double attn_ms = 0.0, exp_ms = 0.0;
        if (!commit_wait(cb, "bench attention", &attn_ms, error, cap)) goto out;
        cb = [m->queue commandBuffer];
        if (!redmetal_topk_pool_encode_mapped(rl_native_metal_pool_handle(m->experts), (__bridge void *)cb,
                (__bridge void *)pf->exp_gate[layer], pf->exp_gate_addr0[layer], (__bridge void *)pf->exp_up[layer], pf->exp_up_addr0[layer],
                (__bridge void *)pf->exp_down[layer], pf->exp_down_addr0[layer], pf->exp_gate_bytes[layer], pf->exp_up_bytes[layer], pf->exp_down_bytes[layer],
                uid, U, pf->exp_type[layer], hidden, ffn, 0u, B, topk, P, pair_token, pair_weight, expert_start, tok_pair,
                (__bridge void *)xin, 0u, (__bridge void *)xout, 0u)) {
            snprintf(error, cap, "bench expert encode failed: %s", redmetal_topk_last_error()); goto out;
        }
        if (!commit_wait(cb, "bench experts", &exp_ms, error, cap)) goto out;
        printf("prefill-kernel-bench: layer %u, %u tokens at position %u: attention %.2f ms, routed experts %.2f ms (%u experts, %u pairs)\n",
               layer, B, position0, attn_ms, exp_ms, U, P);
        ok = 1;
    }
out:
    free(ids); free(pair_token); free(tok_pair); free(uid); free(expert_start); free(fill); free(umap); free(pair_weight);
    return ok;
}

/* dev72 (dev only): attn_fa_b2 in a process with no model loaded (a Metal GPU capture holds every live buffer of the
 * device, so this is the way to a small trace for Xcode's full shader profiler). Qwen3-Next shapes: 16 query heads,
 * 2 KV heads, head_dim 256; B tokens at position0 over a random KV cache. RL_GPU_CAPTURE=FILE captures the third run. */
int rl_metal_prefill_attn_bench_standalone(uint32_t B, uint32_t position0, char *error, size_t cap) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [dev newCommandQueue];
        char *iq3 = rl_iq3_metal_source();
        if (!dev || !queue || !iq3) { set_error(error, cap, "Metal setup failed"); free(iq3); return 0; }
        NSString *src = [NSString stringWithFormat:@"#include <metal_stdlib>\nusing namespace metal;\n%s%s\n%@", rl_kv_defines(), iq3, kPrefillSource];
        free(iq3);
        NSError *le = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&le];
        id<MTLComputePipelineState> pipe = lib ? make_pipe(dev, lib, @"attn_fa_b2", error, cap) : nil;
        if (!pipe) { if (!lib) set_error(error, cap, "prefill library compile failed"); return 0; }
        const uint32_t head_dim = 256u, qheads = 16u, kvheads = 2u, kvlen = position0 + B + 64u;
        srand(72);
        id<MTLBuffer> q = rnd_buf(dev, (size_t)B * qheads * head_dim), g = rnd_buf(dev, (size_t)B * qheads * head_dim);
        id<MTLBuffer> kc = rnd_kv_buf(dev, (size_t)kvlen * kvheads * head_dim), vc = rnd_kv_buf(dev, (size_t)kvlen * kvheads * head_dim);
        id<MTLBuffer> o = [dev newBufferWithLength:(size_t)B * qheads * head_dim * 4u options:MTLResourceStorageModeShared];
        const char *cap_path = getenv("RL_GPU_CAPTURE");
        int capturing = 0;
        double ms[12];
        for (int rep = 0; rep < 12; ++rep) {
            if (rep == 2 && cap_path) {
                MTLCaptureDescriptor *d = [[MTLCaptureDescriptor alloc] init];
                d.captureObject = queue; d.destination = MTLCaptureDestinationGPUTraceDocument;
                d.outputURL = [NSURL fileURLWithPath:[NSString stringWithUTF8String:cap_path]];
                NSError *ce = nil;
                capturing = [[MTLCaptureManager sharedCaptureManager] startCaptureWithDescriptor:d error:&ce];
                if (!capturing) fprintf(stderr, "GPU capture failed: %s\n", ce.localizedDescription.UTF8String ?: "unknown");
            }
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pipe];
            [enc setBuffer:q offset:0 atIndex:0]; [enc setBuffer:kc offset:0 atIndex:1]; [enc setBuffer:vc offset:0 atIndex:2];
            [enc setBuffer:g offset:0 atIndex:3]; [enc setBuffer:o offset:0 atIndex:4];
            [enc setBytes:&head_dim length:4 atIndex:5]; [enc setBytes:&qheads length:4 atIndex:6]; [enc setBytes:&kvheads length:4 atIndex:7];
            [enc setBytes:&position0 length:4 atIndex:8]; [enc setBytes:&B length:4 atIndex:9];
            [enc dispatchThreadgroups:MTLSizeMake(qheads, (B + 15u) / 16u, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [enc endEncoding];
            [cb commit]; [cb waitUntilCompleted];
            ms[rep] = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
            if (capturing) { [[MTLCaptureManager sharedCaptureManager] stopCapture]; fprintf(stderr, "GPU capture written to %s\n", cap_path); capturing = 0; }
        }
        for (int i = 2; i < 12; ++i) for (int j = i + 1; j < 12; ++j) if (ms[j] < ms[i]) { const double t = ms[i]; ms[i] = ms[j]; ms[j] = t; }
        printf("attn_fa_b2: %u tokens at position %u: min %.2f ms, median %.2f ms (10 warm runs)\n", B, position0, ms[2], (ms[6] + ms[7]) / 2.0);
    }
    return 1;
}

/* dev74 (dev only): one dense prefill GEMM kernel (rl_gemm_tg or rl_gemm_wt) on random f32 data, no model:
 * ROWS x K weights, B tokens. RL_GPU_CAPTURE=FILE captures the third run. */
int rl_metal_prefill_gemm_bench_standalone(uint32_t rows, uint32_t K, uint32_t B, const char *kernel, char *error, size_t cap) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [dev newCommandQueue];
        char *iq3 = rl_iq3_metal_source();
        if (!dev || !queue || !iq3) { set_error(error, cap, "Metal setup failed"); free(iq3); return 0; }
        NSString *src = [NSString stringWithFormat:@"#include <metal_stdlib>\nusing namespace metal;\n%s%s\n%@", rl_kv_defines(), iq3, kPrefillSource];
        free(iq3);
        NSError *le = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&le];
        id<MTLComputePipelineState> pipe = lib ? make_pipe(dev, lib, [NSString stringWithUTF8String:kernel], error, cap) : nil;
        if (!pipe) { if (!lib) set_error(error, cap, "prefill library compile failed"); return 0; }
        const int tg = strcmp(kernel, "rl_gemm_tg") == 0;
        const uint32_t ntok_pad = (B + 31u) & ~31u;
        srand(74);
        id<MTLBuffer> w = rnd_buf(dev, (size_t)rows * K), x = rnd_buf(dev, (size_t)ntok_pad * K);
        id<MTLBuffer> o = [dev newBufferWithLength:(size_t)ntok_pad * rows * 4u options:MTLResourceStorageModeShared];
        const char *cap_path = getenv("RL_GPU_CAPTURE");
        int capturing = 0;
        double ms[12];
        for (int rep = 0; rep < 12; ++rep) {
            if (rep == 2 && cap_path) {
                MTLCaptureDescriptor *d = [[MTLCaptureDescriptor alloc] init];
                d.captureObject = queue; d.destination = MTLCaptureDestinationGPUTraceDocument;
                d.outputURL = [NSURL fileURLWithPath:[NSString stringWithUTF8String:cap_path]];
                NSError *ce = nil;
                capturing = [[MTLCaptureManager sharedCaptureManager] startCaptureWithDescriptor:d error:&ce];
                if (!capturing) fprintf(stderr, "GPU capture failed: %s\n", ce.localizedDescription.UTF8String ?: "unknown");
            }
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pipe];
            [enc setBuffer:w offset:0 atIndex:0]; [enc setBuffer:x offset:0 atIndex:1]; [enc setBuffer:o offset:0 atIndex:2];
            [enc setBytes:&rows length:4 atIndex:3]; [enc setBytes:&K length:4 atIndex:4]; [enc setBytes:&ntok_pad length:4 atIndex:5];
            if (tg) [enc dispatchThreadgroups:MTLSizeMake(rows / 64u, ntok_pad / 32u, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            else [enc dispatchThreadgroups:MTLSizeMake((rows + 31u) / 32u, ntok_pad / 32u, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [enc endEncoding];
            [cb commit]; [cb waitUntilCompleted];
            ms[rep] = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
            if (capturing) { [[MTLCaptureManager sharedCaptureManager] stopCapture]; fprintf(stderr, "GPU capture written to %s\n", cap_path); capturing = 0; }
        }
        for (int i = 2; i < 12; ++i) for (int j = i + 1; j < 12; ++j) if (ms[j] < ms[i]) { const double t = ms[i]; ms[i] = ms[j]; ms[j] = t; }
        const double med = (ms[6] + ms[7]) / 2.0;
        printf("%s: %u x %u weights, %u tokens: min %.3f ms, median %.3f ms, %.2f TFLOPS (10 warm runs)\n", kernel, rows, K, B, ms[2], med,
               2.0 * rows * K * (double)B / (med * 1e9));
    }
    return 1;
}

/* dev72 (dev only): the batched routed-expert kernels (redmetal_topk_gateup_mm / down_mm / sum) of layer `layer` with no
 * model buffer on the GPU: `e` is a CPU-only engine (GGUF and expert map), the layer's expert tensors are copied into
 * three buffers of their own, the pool is a one-slot pool used for its pipelines. B tokens x top_k uniformly random
 * experts. RL_GPU_CAPTURE=FILE captures the third run: a trace small enough for Xcode's full shader profiler. */
int rl_metal_expert_bench_standalone(rl_engine *e, const char *model_path, uint32_t layer, uint32_t B, char *error, size_t cap) {
    const rl_engine_info *in = &e->info;
    const uint32_t E = in->n_expert, topk = in->top_k, hidden = in->hidden, ffn = e->gguf.n_ff_exp;
    if (layer >= in->n_layer) { set_error(error, cap, "layer out of range"); return 0; }
    rl_expert_layout first, second, last;
    if (!rl_native_expert_layout(&e->expert_map, layer, 0u, &first, error, cap) || !rl_native_expert_layout(&e->expert_map, layer, 1u, &second, error, cap) ||
        !rl_native_expert_layout(&e->expert_map, layer, E - 1u, &last, error, cap)) return 0;
    if ((first.ggml_type != 17u && first.ggml_type != 29u) || first.down_type != first.ggml_type) { set_error(error, cap, "layer experts are not IQ2_XS / IQ1_M"); return 0; }
    rl_native_quant_tables tables;
    if (!rl_native_quant_tables_init(&tables, error, cap)) return 0;
    redmetal_topk_pool_t pool = redmetal_topk_pool_create(model_path, e->expert_map.max_expert_triplet_bytes, e->expert_map.max_expert_triplet_bytes, 1u,
                                                          tables.iq2_xs, RL_IQ2_XS_GRID_COUNT, tables.iq1_m, RL_IQ1_M_GRID_COUNT);
    rl_native_quant_tables_free(&tables);
    if (!pool) { snprintf(error, cap, "pool: %s", redmetal_topk_last_error()); return 0; }
    int ok = 0;
    const uint32_t P = B * topk;
    uint32_t *ids = (uint32_t *)malloc((size_t)P * 4u), *pair_token = (uint32_t *)malloc((size_t)P * 4u), *tok_pair = (uint32_t *)malloc((size_t)P * 4u);
    uint32_t *uid = (uint32_t *)malloc((size_t)E * 4u), *expert_start = (uint32_t *)calloc(E + 1u, 4u), *fill = (uint32_t *)malloc((size_t)E * 4u);
    int32_t *umap = (int32_t *)malloc((size_t)E * 4u);
    float *pair_weight = (float *)malloc((size_t)P * 4u);
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [dev newCommandQueue];
        const uint8_t *base = (const uint8_t *)e->gguf.map;
        id<MTLBuffer> bg = [dev newBufferWithBytes:base + first.gate_offset length:(NSUInteger)(last.gate_offset + last.gate_bytes - first.gate_offset) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bu = [dev newBufferWithBytes:base + first.up_offset length:(NSUInteger)(last.up_offset + last.up_bytes - first.up_offset) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bd = [dev newBufferWithBytes:base + first.down_offset length:(NSUInteger)(last.down_offset + last.down_bytes - first.down_offset) options:MTLResourceStorageModeShared];
        srand(72);
        id<MTLBuffer> xin = rnd_buf(dev, (size_t)B * hidden);
        id<MTLBuffer> xout = [dev newBufferWithLength:(size_t)B * hidden * 4u options:MTLResourceStorageModeShared];
        if (!queue || !bg || !bu || !bd || !xin || !xout || !ids || !pair_token || !tok_pair || !uid || !expert_start || !fill || !umap || !pair_weight) {
            set_error(error, cap, "bench allocation failed"); goto done;
        }
        for (uint32_t t = 0; t < B; ++t)
            for (uint32_t k = 0; k < topk; ++k) {
                uint32_t x, dup;
                do { x = (uint32_t)rand() % E; dup = 0; for (uint32_t j = 0; j < k; ++j) dup |= ids[t * topk + j] == x; } while (dup);
                ids[t * topk + k] = x;
            }
        for (uint32_t x = 0; x < E; ++x) umap[x] = -1;
        uint32_t U = 0;
        for (uint32_t i = 0; i < P; ++i) if (umap[ids[i]] < 0) { umap[ids[i]] = (int32_t)U; uid[U++] = ids[i]; }
        for (uint32_t i = 0; i < P; ++i) expert_start[umap[ids[i]] + 1]++;
        for (uint32_t u = 0; u < U; ++u) expert_start[u + 1u] += expert_start[u];
        for (uint32_t u = 0; u < U; ++u) fill[u] = expert_start[u];
        for (uint32_t t = 0; t < B; ++t)
            for (uint32_t k = 0; k < topk; ++k) {
                const uint32_t pidx = fill[umap[ids[t * topk + k]]]++;
                pair_token[pidx] = t; pair_weight[pidx] = 1.0f / (float)topk; tok_pair[t * topk + k] = pidx;
            }
        const char *cap_path = getenv("RL_GPU_CAPTURE");
        int capturing = 0;
        double ms[12];
        for (int rep = 0; rep < 12; ++rep) {
            if (rep == 2 && cap_path) {
                MTLCaptureDescriptor *d = [[MTLCaptureDescriptor alloc] init];
                d.captureObject = queue; d.destination = MTLCaptureDestinationGPUTraceDocument;
                d.outputURL = [NSURL fileURLWithPath:[NSString stringWithUTF8String:cap_path]];
                NSError *ce = nil;
                capturing = [[MTLCaptureManager sharedCaptureManager] startCaptureWithDescriptor:d error:&ce];
                if (!capturing) fprintf(stderr, "GPU capture failed: %s\n", ce.localizedDescription.UTF8String ?: "unknown");
            }
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            if (!redmetal_topk_pool_encode_mapped(pool, (__bridge void *)cb, (__bridge void *)bg, (uint64_t)bg.gpuAddress, (__bridge void *)bu, (uint64_t)bu.gpuAddress,
                    (__bridge void *)bd, (uint64_t)bd.gpuAddress, second.gate_offset - first.gate_offset, second.up_offset - first.up_offset,
                    second.down_offset - first.down_offset, uid, U, first.ggml_type, hidden, ffn, 0u, B, topk, P, pair_token, pair_weight,
                    expert_start, tok_pair, (__bridge void *)xin, 0u, (__bridge void *)xout, 0u)) {
                snprintf(error, cap, "expert encode failed: %s", redmetal_topk_last_error()); goto done;
            }
            [cb commit]; [cb waitUntilCompleted];
            ms[rep] = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
            if (capturing) { [[MTLCaptureManager sharedCaptureManager] stopCapture]; fprintf(stderr, "GPU capture written to %s\n", cap_path); capturing = 0; }
        }
        for (int i = 2; i < 12; ++i) for (int j = i + 1; j < 12; ++j) if (ms[j] < ms[i]) { const double t = ms[i]; ms[i] = ms[j]; ms[j] = t; }
        printf("routed experts, layer %u (%s), %u tokens, %u experts, %u pairs: min %.2f ms, median %.2f ms (10 warm runs)\n", layer,
               first.ggml_type == 17u ? "IQ2_XS" : "IQ1_M", B, U, P, ms[2], (ms[6] + ms[7]) / 2.0);
        ok = 1;
    }
done:
    redmetal_topk_pool_destroy(pool);
    free(ids); free(pair_token); free(tok_pair); free(uid); free(expert_start); free(fill); free(umap); free(pair_weight);
    return ok;
}

/* dev74 (dev only): the decode routed-expert kernels (redmetal_topk_pool_encode_device_into: gate/up, down, weighted
 * sum) for one simulated token: 48 calls of top_k experts of layer `layer`, all distinct experts copied from the GGUF
 * into one packed buffer (gate | up | down per expert, as in a pool slot). `e` is a CPU-only engine. */
int rl_metal_decode_expert_bench_standalone(rl_engine *e, const char *model_path, uint32_t layer, char *error, size_t cap) {
    const rl_engine_info *in = &e->info;
    /* RL_BENCH_K=K: K experts per call and 480 / K calls (same total work), to separate per-call cost from per-expert cost */
    const uint32_t topk = getenv("RL_BENCH_K") ? (uint32_t)atoi(getenv("RL_BENCH_K")) : in->top_k;
    const uint32_t E = in->n_expert, hidden = in->hidden, ffn = e->gguf.n_ff_exp, calls = topk ? 480u / topk : 0u;
    if (!topk || topk > 64u) { set_error(error, cap, "RL_BENCH_K must be 1..64"); return 0; }
    if (layer >= in->n_layer || calls * topk > E) { set_error(error, cap, "layer out of range"); return 0; }
    rl_native_layer_info li;
    rl_expert_layout lay0;
    if (!rl_native_get_layer_info(&e->expert_map, layer, &li, error, cap) || !rl_native_expert_layout(&e->expert_map, layer, 0u, &lay0, error, cap)) return 0;
    const uint64_t slot = (lay0.gate_bytes + lay0.up_bytes + lay0.down_bytes + 4095u) & ~(uint64_t)4095u;
    rl_native_quant_tables tables;
    if (!rl_native_quant_tables_init(&tables, error, cap)) return 0;
    redmetal_topk_pool_t pool = redmetal_topk_pool_create(model_path, e->expert_map.max_expert_triplet_bytes, e->expert_map.max_expert_triplet_bytes, 1u,
                                                          tables.iq2_xs, RL_IQ2_XS_GRID_COUNT, tables.iq1_m, RL_IQ1_M_GRID_COUNT);
    rl_native_quant_tables_free(&tables);
    if (!pool) { snprintf(error, cap, "pool: %s", redmetal_topk_last_error()); return 0; }
    int ok = 0;
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [dev newCommandQueue];
        id<MTLBuffer> packed = [dev newBufferWithLength:(NSUInteger)(slot * calls * topk) options:MTLResourceStorageModeShared];
        id<MTLBuffer> table = [dev newBufferWithLength:calls * topk * 8u options:MTLResourceStorageModeShared];
        id<MTLBuffer> wts = [dev newBufferWithLength:calls * topk * 4u options:MTLResourceStorageModeShared];
        srand(74);
        id<MTLBuffer> xin = rnd_buf(dev, hidden);
        id<MTLBuffer> xout = [dev newBufferWithLength:(size_t)hidden * 4u options:MTLResourceStorageModeShared];
        if (!queue || !packed || !table || !wts || !xin || !xout) { set_error(error, cap, "bench allocation failed"); goto done; }
        const uint8_t *base = (const uint8_t *)e->gguf.map;
        uint8_t *dst = (uint8_t *)packed.contents;
        uint64_t *tab = (uint64_t *)table.contents;
        float *w = (float *)wts.contents;
        uint32_t perm[512];
        for (uint32_t i = 0; i < E && i < 512u; ++i) perm[i] = i;
        for (uint32_t i = E - 1u; i > 0; --i) { const uint32_t j = (uint32_t)rand() % (i + 1u); const uint32_t t = perm[i]; perm[i] = perm[j]; perm[j] = t; }
        for (uint32_t i = 0; i < calls * topk; ++i) {
            rl_expert_layout l;
            if (!rl_native_expert_layout(&e->expert_map, layer, perm[i], &l, error, cap)) goto done;
            memcpy(dst + i * slot, base + l.gate_offset, l.gate_bytes);
            memcpy(dst + i * slot + l.gate_bytes, base + l.up_offset, l.up_bytes);
            memcpy(dst + i * slot + l.gate_bytes + l.up_bytes, base + l.down_offset, l.down_bytes);
            tab[i] = (uint64_t)packed.gpuAddress + i * slot;
            w[i] = 1.0f / (float)topk;
        }
        const uint32_t tw = rl_native_expert_type_word(li.ggml_type, li.down_type);
        const char *cap_path = getenv("RL_GPU_CAPTURE");
        int capturing = 0;
        double ms[12];
        for (int rep = 0; rep < 12; ++rep) {
            if (rep == 2 && cap_path) {
                MTLCaptureDescriptor *d = [[MTLCaptureDescriptor alloc] init];
                d.captureObject = queue; d.destination = MTLCaptureDestinationGPUTraceDocument;
                d.outputURL = [NSURL fileURLWithPath:[NSString stringWithUTF8String:cap_path]];
                NSError *ce = nil;
                capturing = [[MTLCaptureManager sharedCaptureManager] startCaptureWithDescriptor:d error:&ce];
            }
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
            [enc useResource:packed usage:MTLResourceUsageRead];
            for (uint32_t c = 0; c < calls; ++c)
                if (!redmetal_topk_pool_encode_device_into(pool, (__bridge void *)enc, (__bridge void *)table, (uint64_t)c * topk * 8u,
                        (__bridge void *)wts, (uint64_t)c * topk * 4u, topk, tw, hidden, ffn, lay0.gate_bytes, lay0.up_bytes,
                        (__bridge void *)xin, 0u, (__bridge void *)xout, 0u)) { [enc endEncoding]; snprintf(error, cap, "encode: %s", redmetal_topk_last_error()); goto done; }
            [enc endEncoding];
            [cb commit]; [cb waitUntilCompleted];
            ms[rep] = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
            if (capturing) { [[MTLCaptureManager sharedCaptureManager] stopCapture]; fprintf(stderr, "GPU capture written to %s\n", cap_path); capturing = 0; }
        }
        for (int i = 2; i < 12; ++i) for (int j = i + 1; j < 12; ++j) if (ms[j] < ms[i]) { const double t = ms[i]; ms[i] = ms[j]; ms[j] = t; }
        const double med = (ms[6] + ms[7]) / 2.0, bytes = (double)calls * topk * (double)(lay0.gate_bytes + lay0.up_bytes + lay0.down_bytes);
        printf("decode experts, layer %u (%s), %u calls x %u experts: min %.3f ms, median %.3f ms per token, %.1f GB/s\n", layer,
               li.ggml_type == 17u ? "IQ2_XS" : li.ggml_type == 29u ? "IQ1_M" : "other", calls, topk, ms[2], med, bytes / (med * 1e6));
        ok = 1;
    }
done:
    redmetal_topk_pool_destroy(pool);
    return ok;
}
