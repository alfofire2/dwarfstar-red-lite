#define _FILE_OFFSET_BITS 64
#define _POSIX_C_SOURCE 200809L
#define _DARWIN_C_SOURCE 1

/*
 * Persistent Metal backend for the Qwen3-Next engine.
 *
 * - one device/queue/library/pipeline set for the whole session;
 * - dense tensors are wrapped in place from the read-only mmap
 *   (newBufferWithBytesNoCopy on page-aligned windows, copy fallback);
 * - DeltaNet conv/recurrent states and full-attention KV caches live in
 *   per-layer shared MTLBuffers and are mutated in place every token;
 * - routed experts run through the field-validated top-k LRU pool
 *   (rl_native_metal_runtime) with one bounded cache shared by all layers.
 *
 * Kernel arithmetic is copied from the field-validated stage kernels
 * (redmetal_deltanet_*.m, redmetal_attention_proj.m, redmetal_shared.m,
 * redmetal_router.m, redmetal_block_ops.m). New kernels: Q5_K rows for the
 * LM head, a context-independent GQA kernel and the fused shared-expert
 * scale/residual add. The delta-rule key/value head pairing follows the
 * pinned llama.cpp repeat-interleave (kv_ratio) semantics.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "redmetal_engine_private.h"
#include "redlite_native_metal.h"
#include "redlite_native_quant_cpu.h"
#include "redlite_native_iq3.h"
#include "redlite_native_router_exec.h"
#include "redmetal_topk.h"
#include "redlite_native_model.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

void rl_metal_engine_profile_report(rl_metal_engine *m);

static void set_error(char *dst, size_t cap, const char *msg) {
    if (dst && cap) snprintf(dst, cap, "%s", msg ? msg : "unknown Metal engine error");
}

static double redmetal_topk_pool_read_ms_total(struct rl_metal_engine *m);

static NSString * const kEngineSource = @
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"static inline ushort rd16(device const uchar *p) { return ushort(p[0]) | (ushort(p[1]) << 8); }\n"
"static inline float fp16(device const uchar *p) { return float(as_type<half>(rd16(p))); }\n"
"static inline uchar2 scale_min(uint j, device const uchar *s) {\n"
"    if (j < 4u) return uchar2(s[j] & 63u, s[j + 4u] & 63u);\n"
"    return uchar2((s[j + 4u] & 15u) | ((s[j - 4u] >> 6) << 4), (s[j + 4u] >> 4) | ((s[j] >> 6) << 4));\n"
"}\n"
/* ---- norms / residuals (validated block ops) ---- */
/* dev39: weighted expert sum + residual add + layer-output copy in one dispatch (was the pool's sum, rl_scale_add and
 * rl_copy_f32): routed = sum_e weights[e] * tmp[e][i] in the pool's order, x = resid + (routed + shared * scalar) */
"kernel void rl_moe_tail(device uint *rl_abort [[buffer(30)]], device const float *tmp [[buffer(0)]], device const float *weights [[buffer(1)]],\n"
"    device const float *resid [[buffer(2)]], device const float *shared [[buffer(3)]], device const float *scalar [[buffer(4)]], device float *x [[buffer(5)]],\n"
"    device float *layer_out [[buffer(6)]], constant uint &n [[buffer(7)]], constant uint &top_k [[buffer(8)]], uint i [[thread_position_in_grid]]) {\n"
"    if (rl_abort[0] != 0u || i >= n) return;\n"
"    float routed = 0.0f;\n"
"    for (uint e = 0; e < top_k; ++e) routed += weights[e] * tmp[e * n + i];\n"
"    const float v = resid[i] + (routed + shared[i] * scalar[0]);\n"
"    x[i] = v; layer_out[i] = v;\n"
"}\n"
/* dev45: rl_moe_tail for the two verify rows: row r sums the experts r * per_row .. (same order as rl_moe_tail) */
"kernel void rl_moe_tail2(device uint *rl_abort [[buffer(30)]], device const float *tmp [[buffer(0)]], device const float *weights [[buffer(1)]],\n"
"    device const float *resid [[buffer(2)]], device const float *shared [[buffer(3)]], device const float *scalar [[buffer(4)]], device float *x [[buffer(5)]],\n"
"    device float *layer_out [[buffer(6)]], constant uint &n [[buffer(7)]], constant uint &per_row [[buffer(8)]],\n"
"    device const float *resid1 [[buffer(9)]], device const float *shared1 [[buffer(10)]], device const float *scalar1 [[buffer(11)]], device float *x1 [[buffer(12)]],\n"
"    uint gid [[thread_position_in_grid]]) {\n"
"    if (rl_abort[0] != 0u || gid >= 2u * n) return;\n"
"    const uint row = gid / n; const uint i = gid - row * n;\n"
"    device const float *rs = row ? resid1 : resid; device const float *sh = row ? shared1 : shared; device const float *sc = row ? scalar1 : scalar;\n"
"    float routed = 0.0f;\n"
"    for (uint e = 0; e < per_row; ++e) routed += weights[row * per_row + e] * tmp[(row * per_row + e) * n + i];\n"
"    const float v = rs[i] + (routed + sh[i] * sc[0]);\n"
"    if (row) x1[i] = v; else { x[i] = v; layer_out[i] = v; }\n"
"}\n"
"kernel void rl_rms(device uint *rl_abort [[buffer(30)]], device const float *x [[buffer(0)]], device const float *w [[buffer(1)]], device float *y [[buffer(2)]],\n"
"    constant uint &n [[buffer(3)]], constant float &eps [[buffer(4)]], uint tid [[thread_position_in_threadgroup]],\n"
"    ushort simd_lane [[thread_index_in_simdgroup]], ushort simd_id [[simdgroup_index_in_threadgroup]]) {\n"
"    if (rl_abort[0] != 0u) return;\n"
"    threadgroup float partial[8]; float ss = 0.0f;\n"
"    for (uint i = tid; i < n; i += 256u) ss = fma(x[i], x[i], ss);\n"
"    ss = simd_sum(ss); if (simd_lane == 0) partial[simd_id] = ss; threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    float total = 0.0f; for (uint k = 0; k < 8u; ++k) total += partial[k];\n"
"    const float inv = rsqrt(total / float(n) + eps);\n"
"    for (uint i = tid; i < n; i += 256u) y[i] = x[i] * inv * w[i];\n"
"}\n"
"kernel void rl_resid_rms(device uint *rl_abort [[buffer(30)]], device const float *a [[buffer(0)]], device const float *b [[buffer(1)]], device const float *w [[buffer(2)]],\n"
"    device float *sum [[buffer(3)]], device float *norm [[buffer(4)]], constant uint &n [[buffer(5)]], constant float &eps [[buffer(6)]],\n"
"    uint tid [[thread_position_in_threadgroup]], ushort simd_lane [[thread_index_in_simdgroup]], ushort simd_id [[simdgroup_index_in_threadgroup]]) {\n"
"    if (rl_abort[0] != 0u) return;\n"
"    threadgroup float partial[8]; float ss = 0.0f;\n"
"    for (uint i = tid; i < n; i += 256u) { const float v = a[i] + b[i]; sum[i] = v; ss = fma(v, v, ss); }\n"
"    ss = simd_sum(ss); if (simd_lane == 0) partial[simd_id] = ss; threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    float total = 0.0f; for (uint k = 0; k < 8u; ++k) total += partial[k];\n"
"    const float inv = rsqrt(total / float(n) + eps);\n"
"    for (uint i = tid; i < n; i += 256u) norm[i] = sum[i] * inv * w[i];\n"
"}\n"
"kernel void rl_steer_add(device uint *rl_abort [[buffer(30)]], device float *x [[buffer(0)]], device const float *v [[buffer(1)]],\n"
"    constant float &s [[buffer(2)]], constant uint &n [[buffer(3)]], uint gid [[thread_position_in_grid]]) {\n"
"    if (rl_abort[0] != 0u) return;\n"
"    if (gid < n) x[gid] = fma(s, v[gid], x[gid]);   /* dev52: activation steering */\n"
"}\n"
"kernel void rl_scale_add(device uint *rl_abort [[buffer(30)]], device const float *resid [[buffer(0)]], device const float *routed [[buffer(1)]], device const float *shared [[buffer(2)]],\n"
"    device const float *scalar [[buffer(3)]], device float *out [[buffer(4)]], constant uint &n [[buffer(5)]], uint gid [[thread_position_in_grid]]) {\n"
"    if (rl_abort[0] != 0u) return;\n"
"    if (gid < n) out[gid] = resid[gid] + (routed[gid] + shared[gid] * scalar[0]);\n"
"}\n"
/* ---- quantized row dot products: lanes_per_row SIMD lanes per row, one block per lane per step ---- */
"inline float rl_q8_block(device const uchar *bp, device const float *x) {\n"
"    const float d = fp16(bp); device const int8_t *q = (device const int8_t *)(bp + 2ul); float acc = 0.0f;\n"
"    for (uint j = 0; j < 32u; ++j) acc += x[j] * (d * float(q[j]));\n"
"    return acc;\n"
"}\n"
/* dev45: two-vector versions (MTP verify rows): weights decoded once, each sum in the 1-vector order */
"inline float2 rl_q8_block2(device const uchar *bp, device const float *x, device const float *x1) {\n"
"    const float d = fp16(bp); device const int8_t *q = (device const int8_t *)(bp + 2ul); float acc = 0.0f, acc1 = 0.0f;\n"
"    for (uint j = 0; j < 32u; ++j) { const float w = d * float(q[j]); acc += x[j] * w; acc1 += x1[j] * w; }\n"
"    return float2(acc, acc1);\n"
"}\n"
"inline float rl_q4k_block(device const uchar *bp, device const float *x) {\n"
"    const float d = fp16(bp); const float dmin = fp16(bp + 2u);\n"
"    device const uchar *scales = bp + 4u; device const uchar *qs = bp + 16u; float acc = 0.0f;\n"
"    for (uint g = 0; g < 8u; ++g) {\n"
"        const uchar2 sm = scale_min(g, scales);\n"
"        const float ds = d * float(sm.x); const float dm = dmin * float(sm.y);\n"
"        device const uchar *q = qs + (g >> 1) * 32u; const uint xb = g * 32u;\n"
"        for (uint l = 0; l < 32u; ++l) { const uchar quant = (g & 1u) ? (q[l] >> 4) : (q[l] & 15u); acc = fma(x[xb + l], ds * float(quant) - dm, acc); }\n"
"    }\n"
"    return acc;\n"
"}\n"
"inline float rl_q5k_block(device const uchar *bp, device const float *x) {\n"
"    const float d = fp16(bp); const float dmin = fp16(bp + 2u);\n"
"    device const uchar *scales = bp + 4u; device const uchar *qh = bp + 16u; device const uchar *ql = bp + 48u;\n"
"    uint is = 0u; uchar u1 = 1u; uchar u2 = 2u; float acc = 0.0f;\n"
"    for (uint j = 0; j < 256u; j += 64u) {\n"
"        const uchar2 sm1 = scale_min(is, scales); const uchar2 sm2 = scale_min(is + 1u, scales);\n"
"        const float d1 = d * float(sm1.x); const float m1 = dmin * float(sm1.y);\n"
"        const float d2 = d * float(sm2.x); const float m2 = dmin * float(sm2.y);\n"
"        for (uint l = 0; l < 32u; ++l) {\n"
"            acc = fma(x[j + l], d1 * float((ql[l] & 15u) + ((qh[l] & u1) ? 16u : 0u)) - m1, acc);\n"
"            acc = fma(x[j + 32u + l], d2 * float((ql[l] >> 4) + ((qh[l] & u2) ? 16u : 0u)) - m2, acc);\n"
"        }\n"
"        ql += 32u; is += 2u; u1 <<= 2; u2 <<= 2;\n"
"    }\n"
"    return acc;\n"
"}\n"
"inline float2 rl_q5k_block2(device const uchar *bp, device const float *x, device const float *x1) {\n"
"    const float d = fp16(bp); const float dmin = fp16(bp + 2u);\n"
"    device const uchar *scales = bp + 4u; device const uchar *qh = bp + 16u; device const uchar *ql = bp + 48u;\n"
"    uint is = 0u; uchar u1 = 1u; uchar u2 = 2u; float acc = 0.0f, acc1 = 0.0f;\n"
"    for (uint j = 0; j < 256u; j += 64u) {\n"
"        const uchar2 sm1 = scale_min(is, scales); const uchar2 sm2 = scale_min(is + 1u, scales);\n"
"        const float d1 = d * float(sm1.x); const float m1 = dmin * float(sm1.y);\n"
"        const float d2 = d * float(sm2.x); const float m2 = dmin * float(sm2.y);\n"
"        for (uint l = 0; l < 32u; ++l) {\n"
"            const float wa = d1 * float((ql[l] & 15u) + ((qh[l] & u1) ? 16u : 0u)) - m1;\n"
"            const float wb = d2 * float((ql[l] >> 4) + ((qh[l] & u2) ? 16u : 0u)) - m2;\n"
"            acc = fma(x[j + l], wa, acc); acc = fma(x[j + 32u + l], wb, acc);\n"
"            acc1 = fma(x1[j + l], wa, acc1); acc1 = fma(x1[j + 32u + l], wb, acc1);\n"
"        }\n"
"        ql += 32u; is += 2u; u1 <<= 2; u2 <<= 2;\n"
"    }\n"
"    return float2(acc, acc1);\n"
"}\n"
"inline float rl_q6k_block(device const uchar *bp, device const float *x) {\n"
"    device const ushort *ql0 = (device const ushort *)(bp + 0ul);\n"
"    device const ushort *qh0 = (device const ushort *)(bp + 128ul);\n"
"    device const int8_t *scales = (device const int8_t *)(bp + 192ul);\n"
"    const float d_all = float(as_type<half>(*(device const ushort *)(bp + 208ul))); float acc = 0.0f;\n"
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
"            acc += x[xb + 0u] * (dl0 * float(q & 0xFFu) - ml);\n"
"            acc += x[xb + 1u] * (dl1 * float(q & 0xFF00u) - ml);\n"
"            acc += x[xb + 2u] * (dl2 * float(q & 0xFF0000u) - ml);\n"
"            acc += x[xb + 3u] * (dl3 * float(q & 0xFF000000u) - ml);\n"
"        }\n"
"    }\n"
"    return acc;\n"
"}\n"
"inline float rl_iq2xxs_block(device const uchar *bp, device const float *x, device const uchar *grid) {\n"
"    const float d = fp16(bp); device const ushort *q = (device const ushort *)(bp + 2ul); float acc = 0.0f;\n"
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
"            for (uint j = 0; j < 8u; ++j) { const float s = (sign8 & (1u << j)) ? -1.0f : 1.0f; acc += x[xb + j] * (db * float(gv[j]) * s); }\n"
"        }\n"
"    }\n"
"    return acc;\n"
"}\n"
/* generic lane-parallel row kernel body: BLOCK values per block, BYTES bytes per block, DOT(bp, xchunk) */
#define RL_ROWS_KERNEL(NAME, BLOCK, BYTES, DOTEXPR, EXTRA_PARAM) \
"kernel void " NAME "(device uint *rl_abort [[buffer(30)]], device const uchar *weights [[buffer(0)]], constant uint &ncols [[buffer(1)]], device const float *x [[buffer(2)]],\n" \
"    device float *out [[buffer(3)]], constant uint &nrows [[buffer(4)]], constant uint &lanes [[buffer(5)]]" EXTRA_PARAM ",\n" \
"    uint tid [[thread_position_in_grid]], ushort simd_lane [[thread_index_in_simdgroup]]) {\n" \
"    if (rl_abort[0] != 0u) return;\n" \
"    const uint row = tid / lanes; const uint lane = uint(simd_lane) % lanes; const bool active = row < nrows;\n" \
"    const uint blocks = ncols / " BLOCK "u; const ulong row_bytes = ulong(blocks) * " BYTES "ul; float acc = 0.0f;\n" \
"    if (active) { device const uchar *rp = weights + ulong(row) * row_bytes;\n" \
"        for (uint b = lane; b < blocks; b += lanes) { device const uchar *bp = rp + ulong(b) * " BYTES "ul; device const float *xc = x + b * " BLOCK "u; acc += " DOTEXPR "; } }\n" \
"    for (uint off = lanes >> 1; off > 0u; off >>= 1) acc += simd_shuffle_xor(acc, ushort(off));\n" \
"    if (active && lane == 0u) out[row] = acc;\n" \
"}\n"
RL_ROWS_KERNEL("rl_rows_q8", "32", "34", "rl_q8_block(bp, xc)", "")
#define RL_ROWS_KERNEL_R2(NAME, BLOCK, BYTES, DOT2) \
"kernel void " NAME "(device uint *rl_abort [[buffer(30)]], device const uchar *weights [[buffer(0)]], constant uint &ncols [[buffer(1)]], device const float *x [[buffer(2)]],\n" \
"    device float *out [[buffer(3)]], constant uint &nrows [[buffer(4)]], constant uint &lanes [[buffer(5)]], device const float *x1 [[buffer(8)]], device float *out1 [[buffer(9)]],\n" \
"    uint tid [[thread_position_in_grid]], ushort simd_lane [[thread_index_in_simdgroup]]) {\n" \
"    if (rl_abort[0] != 0u) return;\n" \
"    const uint row = tid / lanes; const uint lane = uint(simd_lane) % lanes; const bool active = row < nrows;\n" \
"    const uint blocks = ncols / " BLOCK "u; const ulong row_bytes = ulong(blocks) * " BYTES "ul; float acc = 0.0f, acc1 = 0.0f;\n" \
"    if (active) { device const uchar *rp = weights + ulong(row) * row_bytes;\n" \
"        for (uint b = lane; b < blocks; b += lanes) { device const uchar *bp = rp + ulong(b) * " BYTES "ul; const float2 d2 = " DOT2 "(bp, x + b * " BLOCK "u, x1 + b * " BLOCK "u); acc += d2.x; acc1 += d2.y; } }\n" \
"    for (uint off = lanes >> 1; off > 0u; off >>= 1) { acc += simd_shuffle_xor(acc, ushort(off)); acc1 += simd_shuffle_xor(acc1, ushort(off)); }\n" \
"    if (active && lane == 0u) { out[row] = acc; out1[row] = acc1; }\n" \
"}\n"
RL_ROWS_KERNEL_R2("rl_rowsr2_q8", "32", "34", "rl_q8_block2")
RL_ROWS_KERNEL_R2("rl_rowsr2_q5k", "256", "176", "rl_q5k_block2")
RL_ROWS_KERNEL("rl_rows_q4k", "256", "144", "rl_q4k_block(bp, xc)", "")
RL_ROWS_KERNEL("rl_rows_q5k", "256", "176", "rl_q5k_block(bp, xc)", "")
RL_ROWS_KERNEL("rl_rows_q6k", "256", "210", "rl_q6k_block(bp, xc)", "")
RL_ROWS_KERNEL("rl_rows_iq2xxs", "256", "66", "rl_iq2xxs_block(bp, xc, grid)", ", device const uchar *grid [[buffer(6)]]")
/* ---- dev22 decode GEMV: one lane per sub-block (32 values for Q4_K/IQ2_XXS, 16 for Q6_K), up to 32 lanes per row,
 *      float4 activation loads; same dequantization as the block kernels above, summed in a different order ---- */
"inline float rl_q4k_sub(device const uchar *bp, uint g, device const float *x) {\n"
"    const float d = fp16(bp); const float dmin = fp16(bp + 2u);\n"
"    const uchar2 sm = scale_min(g, bp + 4u);\n"
"    device const uchar *q = bp + 16u + (g >> 1) * 32u; const uint sh = (g & 1u) * 4u;\n"
"    float acc = 0.0f; float xs = 0.0f;\n"
"    for (uint l = 0; l < 32u; l += 4u) {\n"
"        const float4 xv = *(device const float4 *)(x + l);\n"
"        const float4 qf = float4((uint4(q[l], q[l + 1u], q[l + 2u], q[l + 3u]) >> sh) & 15u);\n"
"        acc += dot(xv, qf); xs += (xv.x + xv.y) + (xv.z + xv.w);\n"
"    }\n"
"    return (d * float(sm.x)) * acc - (dmin * float(sm.y)) * xs;\n"
"}\n"
"inline float2 rl_q4k_sub2(device const uchar *bp, uint g, device const float *x, device const float *x1) {\n"
"    const float d = fp16(bp); const float dmin = fp16(bp + 2u);\n"
"    const uchar2 sm = scale_min(g, bp + 4u);\n"
"    device const uchar *q = bp + 16u + (g >> 1) * 32u; const uint sh = (g & 1u) * 4u;\n"
"    float acc = 0.0f, xs = 0.0f, acc1 = 0.0f, xs1 = 0.0f;\n"
"    for (uint l = 0; l < 32u; l += 4u) {\n"
"        const float4 qf = float4((uint4(q[l], q[l + 1u], q[l + 2u], q[l + 3u]) >> sh) & 15u);\n"
"        const float4 xv = *(device const float4 *)(x + l); acc += dot(xv, qf); xs += (xv.x + xv.y) + (xv.z + xv.w);\n"
"        const float4 xw = *(device const float4 *)(x1 + l); acc1 += dot(xw, qf); xs1 += (xw.x + xw.y) + (xw.z + xw.w);\n"
"    }\n"
"    return float2((d * float(sm.x)) * acc - (dmin * float(sm.y)) * xs, (d * float(sm.x)) * acc1 - (dmin * float(sm.y)) * xs1);\n"
"}\n"
"inline float rl_iq2xxs_sub(device const uchar *bp, uint g, device const float *x, device const uchar *grid) {\n"
"    const float d = fp16(bp); device const ushort *q = (device const ushort *)(bp + 2ul) + 4u * g;\n"
"    const uint auxg = uint(q[0]) | (uint(q[1]) << 16); const uint auxs = uint(q[2]) | (uint(q[3]) << 16);\n"
"    const float db = d * (0.5f + float(auxs >> 28)) * 0.25f; float acc = 0.0f;\n"
"    for (uint l = 0; l < 4u; ++l) {\n"
"        const uint gi = (auxg >> (8u * l)) & 255u; const uint s7 = (auxs >> (7u * l)) & 127u;\n"
"        const uint s8 = s7 | ((popcount(s7) & 1u) << 7);\n"
"        device const uchar *gv = grid + gi * 8u;\n"
"        const float4 g0 = float4(gv[0], gv[1], gv[2], gv[3]); const float4 g1 = float4(gv[4], gv[5], gv[6], gv[7]);\n"
"        const float4 s0 = select(float4(1.0f), float4(-1.0f), ((uint4(s8) >> uint4(0u, 1u, 2u, 3u)) & 1u) != 0u);\n"
"        const float4 s1 = select(float4(1.0f), float4(-1.0f), ((uint4(s8) >> uint4(4u, 5u, 6u, 7u)) & 1u) != 0u);\n"
"        acc += dot(*(device const float4 *)(x + l * 8u), g0 * s0) + dot(*(device const float4 *)(x + l * 8u + 4u), g1 * s1);\n"
"    }\n"
"    return db * acc;\n"
"}\n"
/* dev45: rl_iq2xxs_sub over two activation vectors, the codebook and signs decoded once; each sum in the 1-vector order */
"inline float2 rl_iq2xxs_sub2(device const uchar *bp, uint g, device const float *x, device const float *x1, device const uchar *grid) {\n"
"    const float d = fp16(bp); device const ushort *q = (device const ushort *)(bp + 2ul) + 4u * g;\n"
"    const uint auxg = uint(q[0]) | (uint(q[1]) << 16); const uint auxs = uint(q[2]) | (uint(q[3]) << 16);\n"
"    const float db = d * (0.5f + float(auxs >> 28)) * 0.25f; float acc = 0.0f, acc1 = 0.0f;\n"
"    for (uint l = 0; l < 4u; ++l) {\n"
"        const uint gi = (auxg >> (8u * l)) & 255u; const uint s7 = (auxs >> (7u * l)) & 127u;\n"
"        const uint s8 = s7 | ((popcount(s7) & 1u) << 7);\n"
"        device const uchar *gv = grid + gi * 8u;\n"
"        const float4 g0 = float4(gv[0], gv[1], gv[2], gv[3]); const float4 g1 = float4(gv[4], gv[5], gv[6], gv[7]);\n"
"        const float4 s0 = select(float4(1.0f), float4(-1.0f), ((uint4(s8) >> uint4(0u, 1u, 2u, 3u)) & 1u) != 0u);\n"
"        const float4 s1 = select(float4(1.0f), float4(-1.0f), ((uint4(s8) >> uint4(4u, 5u, 6u, 7u)) & 1u) != 0u);\n"
"        const float4 w0 = g0 * s0, w1 = g1 * s1;\n"
"        acc += dot(*(device const float4 *)(x + l * 8u), w0) + dot(*(device const float4 *)(x + l * 8u + 4u), w1);\n"
"        acc1 += dot(*(device const float4 *)(x1 + l * 8u), w0) + dot(*(device const float4 *)(x1 + l * 8u + 4u), w1);\n"
"    }\n"
"    return float2(db * acc, db * acc1);\n"
"}\n"
"inline float rl_q6k_sub(device const uchar *bp, uint il0, device const float *x) {\n"
"    device const ushort *ql0 = (device const ushort *)(bp + 0ul);\n"
"    device const ushort *qh0 = (device const ushort *)(bp + 128ul);\n"
"    device const int8_t *scales = (device const int8_t *)(bp + 192ul);\n"
"    const float d_all = float(as_type<half>(*(device const ushort *)(bp + 208ul))); float acc = 0.0f;\n"
"    uint il = il0;\n"
"    device const ushort *ql = ql0 + 32u * (il / 8u) + 16u * ((il / 2u) & 1u) + 8u * (il & 1u);\n"
"    device const ushort *qh = qh0 + 16u * (il / 8u) + 8u * (il & 1u);\n"
"    const float sc = float(scales[(il % 2u) + 2u * (il / 2u)]);\n"
"    il = (il / 2u) & 3u;\n"
"    const uint kmask1 = il > 1u ? (il > 2u ? 0xC0C0C0C0u : 0x30303030u) : (il > 0u ? 0x0C0C0C0Cu : 0x03030303u);\n"
"    const uint kmask2 = il > 1u ? 0xF0F0F0F0u : 0x0F0F0F0Fu;\n"
"    const float ml = d_all * sc * 32.0f; const float dl0 = d_all * sc; const float dl1 = dl0 / 256.0f;\n"
"    const float dl2 = dl1 / 256.0f; const float dl3 = dl2 / 256.0f;\n"
"    const uint shr_h = il > 2u ? 2u : 0u; const uint shl_h = il > 1u ? 0u : (il > 0u ? 2u : 4u); const uint shr_l = il > 1u ? 4u : 0u;\n"
"    for (uint i = 0; i < 4u; ++i) {\n"
"        const uint low = (uint(ql[2u*i]) | (uint(ql[2u*i + 1u]) << 16)) & kmask2;\n"
"        const uint high = (uint(qh[2u*i]) | (uint(qh[2u*i + 1u]) << 16)) & kmask1;\n"
"        const uint q = ((high << shl_h) >> shr_h) | (low >> shr_l);\n"
"        const float4 xv = *(device const float4 *)(x + i * 4u);\n"
"        acc += xv.x * (dl0 * float(q & 0xFFu) - ml);\n"
"        acc += xv.y * (dl1 * float(q & 0xFF00u) - ml);\n"
"        acc += xv.z * (dl2 * float(q & 0xFF0000u) - ml);\n"
"        acc += xv.w * (dl3 * float(q & 0xFF000000u) - ml);\n"
"    }\n"
"    return acc;\n"
"}\n"
"inline float2 rl_q6k_sub2(device const uchar *bp, uint il0, device const float *x, device const float *x1) {\n"
"    device const ushort *ql0 = (device const ushort *)(bp + 0ul);\n"
"    device const ushort *qh0 = (device const ushort *)(bp + 128ul);\n"
"    device const int8_t *scales = (device const int8_t *)(bp + 192ul);\n"
"    const float d_all = float(as_type<half>(*(device const ushort *)(bp + 208ul))); float acc = 0.0f, acc1 = 0.0f;\n"
"    uint il = il0;\n"
"    device const ushort *ql = ql0 + 32u * (il / 8u) + 16u * ((il / 2u) & 1u) + 8u * (il & 1u);\n"
"    device const ushort *qh = qh0 + 16u * (il / 8u) + 8u * (il & 1u);\n"
"    const float sc = float(scales[(il % 2u) + 2u * (il / 2u)]);\n"
"    il = (il / 2u) & 3u;\n"
"    const uint kmask1 = il > 1u ? (il > 2u ? 0xC0C0C0C0u : 0x30303030u) : (il > 0u ? 0x0C0C0C0Cu : 0x03030303u);\n"
"    const uint kmask2 = il > 1u ? 0xF0F0F0F0u : 0x0F0F0F0Fu;\n"
"    const float ml = d_all * sc * 32.0f; const float dl0 = d_all * sc; const float dl1 = dl0 / 256.0f;\n"
"    const float dl2 = dl1 / 256.0f; const float dl3 = dl2 / 256.0f;\n"
"    const uint shr_h = il > 2u ? 2u : 0u; const uint shl_h = il > 1u ? 0u : (il > 0u ? 2u : 4u); const uint shr_l = il > 1u ? 4u : 0u;\n"
"    for (uint i = 0; i < 4u; ++i) {\n"
"        const uint low = (uint(ql[2u*i]) | (uint(ql[2u*i + 1u]) << 16)) & kmask2;\n"
"        const uint high = (uint(qh[2u*i]) | (uint(qh[2u*i + 1u]) << 16)) & kmask1;\n"
"        const uint q = ((high << shl_h) >> shr_h) | (low >> shr_l);\n"
"        const float4 xv = *(device const float4 *)(x + i * 4u); const float4 xw = *(device const float4 *)(x1 + i * 4u);\n"
"        const float w0 = dl0 * float(q & 0xFFu) - ml, w1 = dl1 * float(q & 0xFF00u) - ml, w2 = dl2 * float(q & 0xFF0000u) - ml, w3 = dl3 * float(q & 0xFF000000u) - ml;\n"
"        acc += xv.x * w0; acc += xv.y * w1; acc += xv.z * w2; acc += xv.w * w3;\n"
"        acc1 += xw.x * w0; acc1 += xw.y * w1; acc1 += xw.z * w2; acc1 += xw.w * w3;\n"
"    }\n"
"    return float2(acc, acc1);\n"
"}\n"
#define RL_ROWS_SUB_KERNEL(NAME, BLOCK, BYTES, SUB, SUBVALS, DOTEXPR, EXTRA_PARAM) \
"kernel void " NAME "(device uint *rl_abort [[buffer(30)]], device const uchar *weights [[buffer(0)]], constant uint &ncols [[buffer(1)]], device const float *x [[buffer(2)]],\n" \
"    device float *out [[buffer(3)]], constant uint &nrows [[buffer(4)]], constant uint &lanes [[buffer(5)]]" EXTRA_PARAM ",\n" \
"    uint tid [[thread_position_in_grid]], ushort simd_lane [[thread_index_in_simdgroup]]) {\n" \
"    if (rl_abort[0] != 0u) return;\n" \
"    const uint row = tid / lanes; const uint lane = uint(simd_lane) % lanes; const bool active = row < nrows;\n" \
"    const uint blocks = ncols / " BLOCK "u; const uint items = blocks * " SUB "u; const ulong row_bytes = ulong(blocks) * " BYTES "ul; float acc = 0.0f;\n" \
"    if (active) { device const uchar *rp = weights + ulong(row) * row_bytes;\n" \
"        for (uint i = lane; i < items; i += lanes) { const uint b = i / " SUB "u; const uint s = i % " SUB "u;\n" \
"            device const uchar *bp = rp + ulong(b) * " BYTES "ul; device const float *xc = x + b * " BLOCK "u + s * " SUBVALS "u; acc += " DOTEXPR "; } }\n" \
"    for (uint off = lanes >> 1; off > 0u; off >>= 1) acc += simd_shuffle_xor(acc, ushort(off));\n" \
"    if (active && lane == 0u) out[row] = acc;\n" \
"}\n"
/* dev45: the same kernels over two activation vectors (x, out at 2/3 and x1, out1 at 8/9) for the 2-row MTP verify:
 * each row's arithmetic is the 1-row kernel's (the compiler shares the dequantization between the two dots) */
#define RL_ROWS_SUB_KERNEL_R2(NAME, BLOCK, BYTES, SUB, SUBVALS, DOT0, DOT1, EXTRA_PARAM) \
"kernel void " NAME "(device uint *rl_abort [[buffer(30)]], device const uchar *weights [[buffer(0)]], constant uint &ncols [[buffer(1)]], device const float *x [[buffer(2)]],\n" \
"    device float *out [[buffer(3)]], constant uint &nrows [[buffer(4)]], constant uint &lanes [[buffer(5)]], device const float *x1 [[buffer(8)]], device float *out1 [[buffer(9)]]" EXTRA_PARAM ",\n" \
"    uint tid [[thread_position_in_grid]], ushort simd_lane [[thread_index_in_simdgroup]]) {\n" \
"    if (rl_abort[0] != 0u) return;\n" \
"    const uint row = tid / lanes; const uint lane = uint(simd_lane) % lanes; const bool active = row < nrows;\n" \
"    const uint blocks = ncols / " BLOCK "u; const uint items = blocks * " SUB "u; const ulong row_bytes = ulong(blocks) * " BYTES "ul; float acc = 0.0f, acc1 = 0.0f;\n" \
"    if (active) { device const uchar *rp = weights + ulong(row) * row_bytes;\n" \
"        for (uint i = lane; i < items; i += lanes) { const uint b = i / " SUB "u; const uint s = i % " SUB "u;\n" \
"            device const uchar *bp = rp + ulong(b) * " BYTES "ul; device const float *xc = x + b * " BLOCK "u + s * " SUBVALS "u;\n" \
"            device const float *xc1 = x1 + b * " BLOCK "u + s * " SUBVALS "u; const float2 d2 = " DOT0 "; acc += d2.x; acc1 += d2.y; } }\n" \
"    for (uint off = lanes >> 1; off > 0u; off >>= 1) { acc += simd_shuffle_xor(acc, ushort(off)); acc1 += simd_shuffle_xor(acc1, ushort(off)); }\n" \
"    if (active && lane == 0u) { out[row] = acc; out1[row] = acc1; }\n" \
"}\n"
RL_ROWS_SUB_KERNEL_R2("rl_rows2r2_iq2xxs", "256", "66", "8", "32", "rl_iq2xxs_sub2(bp, s, xc, xc1, grid)", "", ", device const uchar *grid [[buffer(6)]]")
RL_ROWS_SUB_KERNEL_R2("rl_rows2r2_q4k", "256", "144", "8", "32", "rl_q4k_sub2(bp, s, xc, xc1)", "", "")
RL_ROWS_SUB_KERNEL_R2("rl_rows2r2_q6k", "256", "210", "16", "16", "rl_q6k_sub2(bp, s, xc, xc1)", "", "")
RL_ROWS_SUB_KERNEL("rl_rows2_q4k", "256", "144", "8", "32", "rl_q4k_sub(bp, s, xc)", "")
RL_ROWS_SUB_KERNEL("rl_rows2_q6k", "256", "210", "16", "16", "rl_q6k_sub(bp, s, xc)", "")
RL_ROWS_SUB_KERNEL("rl_rows2_iq2xxs", "256", "66", "8", "32", "rl_iq2xxs_sub(bp, s, xc, grid)", ", device const uchar *grid [[buffer(6)]]")
/* dev31: IQ3_XXS / IQ3_S / IQ2_S / IQ4_XS rows: one lane per 32-value item (4 groups of rl_iq3_group8) */
"kernel void rl_rows_iq3(device uint *rl_abort [[buffer(30)]], device const uchar *weights [[buffer(0)]], constant uint &ncols [[buffer(1)]], device const float *x [[buffer(2)]],\n"
"    device float *out [[buffer(3)]], constant uint &nrows [[buffer(4)]], constant uint &lanes [[buffer(5)]], constant uint &type [[buffer(7)]],\n"
"    uint tid [[thread_position_in_grid]], ushort simd_lane [[thread_index_in_simdgroup]]) {\n"
"    if (rl_abort[0] != 0u) return;\n"
"    const uint row = tid / lanes; const uint lane = uint(simd_lane) % lanes; const bool active = row < nrows;\n"
"    const uint bb = rl_iq3_block_bytes(type); const uint blocks = ncols / 256u; const uint items = blocks * 8u; const ulong row_bytes = ulong(blocks) * bb; float acc = 0.0f;\n"
"    if (active) { device const uchar *rp = weights + ulong(row) * row_bytes;\n"
"        for (uint i = lane; i < items; i += lanes) { const uint b = i >> 3; const uint s = i & 7u; device const uchar *bp = rp + ulong(b) * bb;\n"
"            acc += rl_iq3_dot32(type, bp, s, x + b * 256u + s * 32u); } }\n"
"    for (uint off = lanes >> 1; off > 0u; off >>= 1) acc += simd_shuffle_xor(acc, ushort(off));\n"
"    if (active && lane == 0u) out[row] = acc;\n"
"}\n"
"kernel void rl_rows2_f32(device uint *rl_abort [[buffer(30)]], device const float *weights [[buffer(0)]], constant uint &ncols [[buffer(1)]], device const float *x [[buffer(2)]],\n"
"    device float *out [[buffer(3)]], constant uint &nrows [[buffer(4)]], constant uint &lanes [[buffer(5)]],\n"
"    uint tid [[thread_position_in_grid]], ushort simd_lane [[thread_index_in_simdgroup]]) {\n"
"    if (rl_abort[0] != 0u) return;\n"
"    const uint row = tid / lanes; const uint lane = uint(simd_lane) % lanes; const bool active = row < nrows; float acc = 0.0f;\n"
"    if (active) { device const float4 *rp = (device const float4 *)(weights + ulong(row) * ulong(ncols)); device const float4 *xv = (device const float4 *)x;\n"
"        for (uint i = lane; i < ncols / 4u; i += lanes) acc += dot(rp[i], xv[i]); }\n"
"    for (uint off = lanes >> 1; off > 0u; off >>= 1) acc += simd_shuffle_xor(acc, ushort(off));\n"
"    if (active && lane == 0u) out[row] = acc;\n"
"}\n"
"kernel void rl_rows_iq3_r2(device uint *rl_abort [[buffer(30)]], device const uchar *weights [[buffer(0)]], constant uint &ncols [[buffer(1)]], device const float *x [[buffer(2)]],\n"
"    device float *out [[buffer(3)]], constant uint &nrows [[buffer(4)]], constant uint &lanes [[buffer(5)]], constant uint &type [[buffer(7)]],\n"
"    device const float *x1 [[buffer(8)]], device float *out1 [[buffer(9)]],\n"
"    uint tid [[thread_position_in_grid]], ushort simd_lane [[thread_index_in_simdgroup]]) {\n"
"    if (rl_abort[0] != 0u) return;\n"
"    const uint row = tid / lanes; const uint lane = uint(simd_lane) % lanes; const bool active = row < nrows;\n"
"    const uint bb = rl_iq3_block_bytes(type); const uint blocks = ncols / 256u; const uint items = blocks * 8u; const ulong row_bytes = ulong(blocks) * bb; float acc = 0.0f, acc1 = 0.0f;\n"
"    if (active) { device const uchar *rp = weights + ulong(row) * row_bytes;\n"
"        for (uint i = lane; i < items; i += lanes) { const uint b = i >> 3; const uint s = i & 7u; device const uchar *bp = rp + ulong(b) * bb;\n"
"            const float2 d2 = rl_iq3_dot32_2(type, bp, s, x + b * 256u + s * 32u, x1 + b * 256u + s * 32u); acc += d2.x; acc1 += d2.y; } }\n"
"    for (uint off = lanes >> 1; off > 0u; off >>= 1) { acc += simd_shuffle_xor(acc, ushort(off)); acc1 += simd_shuffle_xor(acc1, ushort(off)); }\n"
"    if (active && lane == 0u) { out[row] = acc; out1[row] = acc1; }\n"
"}\n"
"kernel void rl_rows2r2_f32(device uint *rl_abort [[buffer(30)]], device const float *weights [[buffer(0)]], constant uint &ncols [[buffer(1)]], device const float *x [[buffer(2)]],\n"
"    device float *out [[buffer(3)]], constant uint &nrows [[buffer(4)]], constant uint &lanes [[buffer(5)]], device const float *x1 [[buffer(8)]], device float *out1 [[buffer(9)]],\n"
"    uint tid [[thread_position_in_grid]], ushort simd_lane [[thread_index_in_simdgroup]]) {\n"
"    if (rl_abort[0] != 0u) return;\n"
"    const uint row = tid / lanes; const uint lane = uint(simd_lane) % lanes; const bool active = row < nrows; float acc = 0.0f, acc1 = 0.0f;\n"
"    if (active) { device const float4 *rp = (device const float4 *)(weights + ulong(row) * ulong(ncols)); device const float4 *xv = (device const float4 *)x; device const float4 *xw = (device const float4 *)x1;\n"
"        for (uint i = lane; i < ncols / 4u; i += lanes) { const float4 w = rp[i]; acc += dot(w, xv[i]); acc1 += dot(w, xw[i]); } }\n"
"    for (uint off = lanes >> 1; off > 0u; off >>= 1) { acc += simd_shuffle_xor(acc, ushort(off)); acc1 += simd_shuffle_xor(acc1, ushort(off)); }\n"
"    if (active && lane == 0u) { out[row] = acc; out1[row] = acc1; }\n"
"}\n"
"kernel void rl_rows_f32(device uint *rl_abort [[buffer(30)]], device const float *weights [[buffer(0)]], constant uint &ncols [[buffer(1)]], device const float *x [[buffer(2)]],\n"
"    device float *out [[buffer(3)]], constant uint &nrows [[buffer(4)]], constant uint &lanes [[buffer(5)]],\n"
"    uint tid [[thread_position_in_grid]], ushort simd_lane [[thread_index_in_simdgroup]]) {\n"
"    if (rl_abort[0] != 0u) return;\n"
"    const uint row = tid / lanes; const uint lane = uint(simd_lane) % lanes; const bool active = row < nrows; float acc = 0.0f;\n"
"    if (active) { device const float *rp = weights + ulong(row) * ulong(ncols); for (uint i = lane; i < ncols; i += lanes) acc = fma(rp[i], x[i], acc); }\n"
"    for (uint off = lanes >> 1; off > 0u; off >>= 1) acc += simd_shuffle_xor(acc, ushort(off));\n"
"    if (active && lane == 0u) out[row] = acc;\n"
"}\n"
/* ---- Gated DeltaNet (validated dev15 kernels, kv_ratio pairing) ---- */
"kernel void dn_ba_params(device uint *rl_abort [[buffer(30)]], device const float *ba [[buffer(0)]], device const float *dt [[buffer(1)]], device const float *avec [[buffer(2)]],\n"
"    device float *beta [[buffer(3)]], device float *gate [[buffer(4)]], constant uint &dt_rank [[buffer(5)]], constant uint &n_group [[buffer(6)]],\n"
"    uint gid [[thread_position_in_grid]]) {\n"
"    if (rl_abort[0] != 0u) return;\n"
"    if (gid >= dt_rank) return; const uint width = dt_rank / n_group; const uint group = gid / width; const uint local = gid - group * width;\n"
"    const uint stride = 2u * width; const float b = ba[group * stride + local]; const float alpha = ba[group * stride + width + local] + dt[gid];\n"
"    beta[gid] = 1.0f / (1.0f + exp(-b)); const float sp = alpha > 20.0f ? alpha : log(1.0f + exp(alpha)); gate[gid] = sp * avec[gid];\n"
"}\n"
"kernel void dn_conv_silu(device uint *rl_abort [[buffer(30)]], device const float *state [[buffer(0)]], device const float *qkv [[buffer(1)]], device const float *conv_w [[buffer(2)]],\n"
"    device float *out [[buffer(3)]], constant uint &channels [[buffer(4)]], constant uint &dconv [[buffer(5)]], uint gid [[thread_position_in_grid]]) {\n"
"    if (rl_abort[0] != 0u) return;\n"
"    if (gid >= channels) return; float acc = 0.0f; const uint ns = dconv - 1u; const uint sb = gid * ns; const uint kb = gid * dconv;\n"
"    for (uint j = 0; j < ns; ++j) acc = fma(state[sb + j], conv_w[kb + j], acc);\n"
"    acc = fma(qkv[gid], conv_w[kb + ns], acc); out[gid] = acc / (1.0f + exp(-acc));\n"
"}\n"
"kernel void dn_qk_l2_tg(device uint *rl_abort [[buffer(30)]], device const float *src [[buffer(0)]], device float *q [[buffer(1)]], device float *k [[buffer(2)]],\n"
"    constant uint &head_dim [[buffer(3)]], constant uint &heads [[buffer(4)]], constant float &eps [[buffer(5)]],\n"
"    uint tg [[threadgroup_position_in_grid]], uint i [[thread_position_in_threadgroup]],\n"
"    ushort simd_lane [[thread_index_in_simdgroup]], ushort simd_id [[simdgroup_index_in_threadgroup]]) {\n"
"    if (rl_abort[0] != 0u) return;\n"
"    threadgroup float red[32]; const uint which = tg / heads; const uint head = tg - which * heads;\n"
"    const uint base = which * heads * head_dim + head * head_dim; const float x = i < head_dim ? src[base + i] : 0.0f;\n"
"    float ss = simd_sum(x * x); if (simd_lane == 0) red[simd_id] = ss; threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    float total = 0.0f; for (uint s = 0; s < (head_dim + 31u) / 32u; ++s) total += red[s];\n"
"    const float scale = 1.0f / max(sqrt(total), eps);\n"
"    if (i < head_dim) { if (which == 0u) q[head * head_dim + i] = x * scale; else k[head * head_dim + i] = x * scale; }\n"
"}\n"
"kernel void dn_shift_state(device uint *rl_abort [[buffer(30)]], device const float *state [[buffer(0)]], device const float *qkv [[buffer(1)]], device float *next_state [[buffer(2)]],\n"
"    constant uint &channels [[buffer(3)]], constant uint &dconv [[buffer(4)]], uint gid [[thread_position_in_grid]]) {\n"
"    if (rl_abort[0] != 0u) return;\n"
"    if (gid >= channels) return; const uint ns = dconv - 1u; const uint b = gid * ns;\n"
"    for (uint j = 0; j + 1u < ns; ++j) next_state[b + j] = state[b + j + 1u];\n"
"    next_state[b + ns - 1u] = qkv[gid];\n"
"}\n"
/* fused gated delta rule for one (value head, state row j): decay, delta, in-place update, output */
"kernel void dn_state_fused(device uint *rl_abort [[buffer(30)]], device float *state [[buffer(0)]], device const float *q [[buffer(1)]], device const float *k [[buffer(2)]],\n"
"    device const float *v [[buffer(3)]], device const float *gate [[buffer(4)]], device const float *beta [[buffer(5)]],\n"
"    device float *out [[buffer(6)]], constant uint &state_size [[buffer(7)]], constant uint &kv_ratio [[buffer(8)]], constant uint &value_heads [[buffer(9)]],\n"
"    uint tg [[threadgroup_position_in_grid]], uint i [[thread_position_in_threadgroup]],\n"
"    ushort simd_lane [[thread_index_in_simdgroup]], ushort simd_id [[simdgroup_index_in_threadgroup]]) {\n"
"    if (rl_abort[0] != 0u) return;\n"
"    threadgroup float red[32]; const uint h = tg / state_size; const uint j = tg - h * state_size; if (h >= value_heads) return;\n"
"    const uint kh = h / kv_ratio; const uint nsimd = state_size / 32u; const uint idx = (h * state_size + j) * state_size + i;\n"
"    float s = state[idx] * exp(gate[h]); const float ki = k[kh * state_size + i];\n"
"    float part = simd_sum(s * ki); if (simd_lane == 0) red[simd_id] = part; threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    float sum = 0.0f; for (uint n = 0; n < nsimd; ++n) sum += red[n];\n"
"    const float delta = (v[h * state_size + j] - sum) * beta[h];\n"
"    s += ki * delta; state[idx] = s;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    float o = simd_sum(s * q[kh * state_size + i]); if (simd_lane == 0) red[simd_id] = o; threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    if (i == 0u) { float total = 0.0f; for (uint n = 0; n < nsimd; ++n) total += red[n]; out[h * state_size + j] = total * (1.0f / sqrt((float)state_size)); }\n"
"}\n"
/* ---- dev45: the DeltaNet kernels for the two verify rows, one dispatch each; every row's arithmetic is the 1-row
 *      kernel's, row 1 starts from row 0's state, and the state after row 0 is written to the snapshot buffers ---- */
"kernel void dn_ba_params2(device uint *rl_abort [[buffer(30)]], device const float *ba [[buffer(0)]], device const float *dt [[buffer(1)]], device const float *avec [[buffer(2)]],\n"
"    device float *beta [[buffer(3)]], device float *gate [[buffer(4)]], constant uint &dt_rank [[buffer(5)]], constant uint &n_group [[buffer(6)]],\n"
"    device const float *ba1 [[buffer(7)]], device float *beta1 [[buffer(8)]], device float *gate1 [[buffer(9)]], uint g2 [[thread_position_in_grid]]) {\n"
"    if (rl_abort[0] != 0u) return;\n"
"    if (g2 >= 2u * dt_rank) return; const uint row = g2 / dt_rank; const uint gid = g2 - row * dt_rank;\n"
"    device const float *b_ = row ? ba1 : ba; device float *be = row ? beta1 : beta; device float *ga = row ? gate1 : gate;\n"
"    const uint width = dt_rank / n_group; const uint group = gid / width; const uint local = gid - group * width;\n"
"    const uint stride = 2u * width; const float b = b_[group * stride + local]; const float alpha = b_[group * stride + width + local] + dt[gid];\n"
"    be[gid] = 1.0f / (1.0f + exp(-b)); const float sp = alpha > 20.0f ? alpha : log(1.0f + exp(alpha)); ga[gid] = sp * avec[gid];\n"
"}\n"
"kernel void dn_conv_shift2(device uint *rl_abort [[buffer(30)]], device float *state [[buffer(0)]], device const float *qkv [[buffer(1)]], device const float *conv_w [[buffer(2)]],\n"
"    device float *out [[buffer(3)]], constant uint &channels [[buffer(4)]], constant uint &dconv [[buffer(5)]], device const float *qkv1 [[buffer(6)]],\n"
"    device float *out1 [[buffer(7)]], device float *snap [[buffer(8)]], uint gid [[thread_position_in_grid]]) {\n"
"    if (rl_abort[0] != 0u) return;\n"
"    if (gid >= channels) return; const uint ns = dconv - 1u; const uint sb = gid * ns; const uint kb = gid * dconv;\n"
"    float s[8]; for (uint j = 0; j < ns; ++j) s[j] = state[sb + j];\n"
"    float acc = 0.0f; for (uint j = 0; j < ns; ++j) acc = fma(s[j], conv_w[kb + j], acc);\n"
"    acc = fma(qkv[gid], conv_w[kb + ns], acc); out[gid] = acc / (1.0f + exp(-acc));\n"
"    for (uint j = 0; j + 1u < ns; ++j) s[j] = s[j + 1u]; s[ns - 1u] = qkv[gid];\n"
"    for (uint j = 0; j < ns; ++j) snap[sb + j] = s[j];\n"
"    float acc1 = 0.0f; for (uint j = 0; j < ns; ++j) acc1 = fma(s[j], conv_w[kb + j], acc1);\n"
"    acc1 = fma(qkv1[gid], conv_w[kb + ns], acc1); out1[gid] = acc1 / (1.0f + exp(-acc1));\n"
"    for (uint j = 0; j + 1u < ns; ++j) s[j] = s[j + 1u]; s[ns - 1u] = qkv1[gid];\n"
"    for (uint j = 0; j < ns; ++j) state[sb + j] = s[j];\n"
"}\n"
"kernel void dn_qk_l2_2(device uint *rl_abort [[buffer(30)]], device const float *src [[buffer(0)]], device float *q [[buffer(1)]], device float *k [[buffer(2)]],\n"
"    constant uint &head_dim [[buffer(3)]], constant uint &heads [[buffer(4)]], constant float &eps [[buffer(5)]],\n"
"    device const float *src1 [[buffer(6)]], device float *q1 [[buffer(7)]], device float *k1 [[buffer(8)]],\n"
"    uint tg2 [[threadgroup_position_in_grid]], uint i [[thread_position_in_threadgroup]],\n"
"    ushort simd_lane [[thread_index_in_simdgroup]], ushort simd_id [[simdgroup_index_in_threadgroup]]) {\n"
"    if (rl_abort[0] != 0u) return;\n"
"    threadgroup float red[32]; const uint row = tg2 / (2u * heads); const uint tg = tg2 - row * 2u * heads;\n"
"    device const float *sr = row ? src1 : src; device float *qq = row ? q1 : q; device float *kk = row ? k1 : k;\n"
"    const uint which = tg / heads; const uint head = tg - which * heads;\n"
"    const uint base = which * heads * head_dim + head * head_dim; const float x = i < head_dim ? sr[base + i] : 0.0f;\n"
"    float ss = simd_sum(x * x); if (simd_lane == 0) red[simd_id] = ss; threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    float total = 0.0f; for (uint s = 0; s < (head_dim + 31u) / 32u; ++s) total += red[s];\n"
"    const float scale = 1.0f / max(sqrt(total), eps);\n"
"    if (i < head_dim) { if (which == 0u) qq[head * head_dim + i] = x * scale; else kk[head * head_dim + i] = x * scale; }\n"
"}\n"
"kernel void dn_state2(device uint *rl_abort [[buffer(30)]], device float *state [[buffer(0)]], device const float *q [[buffer(1)]], device const float *k [[buffer(2)]],\n"
"    device const float *v [[buffer(3)]], device const float *gate [[buffer(4)]], device const float *beta [[buffer(5)]],\n"
"    device float *out [[buffer(6)]], constant uint &state_size [[buffer(7)]], constant uint &kv_ratio [[buffer(8)]], constant uint &value_heads [[buffer(9)]],\n"
"    device const float *q1 [[buffer(10)]], device const float *k1 [[buffer(11)]], device const float *v1 [[buffer(12)]], device const float *gate1 [[buffer(13)]],\n"
"    device const float *beta1 [[buffer(14)]], device float *out1 [[buffer(15)]], device float *snap [[buffer(16)]],\n"
"    uint tg [[threadgroup_position_in_grid]], uint i [[thread_position_in_threadgroup]],\n"
"    ushort simd_lane [[thread_index_in_simdgroup]], ushort simd_id [[simdgroup_index_in_threadgroup]]) {\n"
"    if (rl_abort[0] != 0u) return;\n"
"    threadgroup float red[32]; const uint h = tg / state_size; const uint j = tg - h * state_size; if (h >= value_heads) return;\n"
"    const uint kh = h / kv_ratio; const uint nsimd = state_size / 32u; const uint idx = (h * state_size + j) * state_size + i;\n"
"    float s = state[idx];\n"
"    for (uint row = 0; row < 2u; ++row) {\n"
"        device const float *qq = row ? q1 : q; device const float *kk = row ? k1 : k; device const float *vv = row ? v1 : v;\n"
"        device const float *gg = row ? gate1 : gate; device const float *bb = row ? beta1 : beta; device float *oo = row ? out1 : out;\n"
"        s = s * exp(gg[h]); const float ki = kk[kh * state_size + i];\n"
"        float part = simd_sum(s * ki); if (simd_lane == 0) red[simd_id] = part; threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"        float sum = 0.0f; for (uint n = 0; n < nsimd; ++n) sum += red[n];\n"
"        const float delta = (vv[h * state_size + j] - sum) * bb[h];\n"
"        s += ki * delta;\n"
"        if (row == 0u) snap[idx] = s; else state[idx] = s;\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"        float o = simd_sum(s * qq[kh * state_size + i]); if (simd_lane == 0) red[simd_id] = o; threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"        if (i == 0u) { float total = 0.0f; for (uint n = 0; n < nsimd; ++n) total += red[n]; oo[h * state_size + j] = total * (1.0f / sqrt((float)state_size)); }\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"}\n"
"kernel void dn_tail_norm2(device uint *rl_abort [[buffer(30)]], device const float *core [[buffer(0)]], device const float *z [[buffer(1)]], device const float *w [[buffer(2)]],\n"
"    device float *out [[buffer(3)]], constant float &eps [[buffer(4)]], constant uint &head_dim [[buffer(5)]], constant uint &heads [[buffer(6)]],\n"
"    device const float *core1 [[buffer(7)]], device const float *z1 [[buffer(8)]], device float *out1 [[buffer(9)]],\n"
"    uint h2 [[threadgroup_position_in_grid]], uint i [[thread_position_in_threadgroup]],\n"
"    ushort simd_lane [[thread_index_in_simdgroup]], ushort simd_id [[simdgroup_index_in_threadgroup]]) {\n"
"    if (rl_abort[0] != 0u) return;\n"
"    threadgroup float red[32]; const uint row = h2 / heads; const uint h = h2 - row * heads; if (row >= 2u) return;\n"
"    device const float *cc = row ? core1 : core; device const float *zz = row ? z1 : z; device float *oo = row ? out1 : out;\n"
"    const uint base = h * head_dim; const float x = i < head_dim ? cc[base + i] : 0.0f;\n"
"    float ss = simd_sum(x * x); if (simd_lane == 0) red[simd_id] = ss; threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    float total = 0.0f; for (uint n = 0; n < (head_dim + 31u) / 32u; ++n) total += red[n];\n"
"    const float inv = rsqrt(total / float(head_dim) + eps);\n"
"    if (i < head_dim) { const float g = zz[base + i]; const float silu = g / (1.0f + exp(-g)); oo[base + i] = x * inv * w[i] * silu; }\n"
"}\n"
"kernel void dn_tail_norm(device uint *rl_abort [[buffer(30)]], device const float *core [[buffer(0)]], device const float *z [[buffer(1)]], device const float *w [[buffer(2)]],\n"
"    device float *out [[buffer(3)]], constant float &eps [[buffer(4)]], constant uint &head_dim [[buffer(5)]], constant uint &heads [[buffer(6)]],\n"
"    uint h [[threadgroup_position_in_grid]], uint i [[thread_position_in_threadgroup]],\n"
"    ushort simd_lane [[thread_index_in_simdgroup]], ushort simd_id [[simdgroup_index_in_threadgroup]]) {\n"
"    if (rl_abort[0] != 0u) return;\n"
"    threadgroup float red[32]; if (h >= heads) return; const uint base = h * head_dim; const float x = i < head_dim ? core[base + i] : 0.0f;\n"
"    float ss = simd_sum(x * x); if (simd_lane == 0) red[simd_id] = ss; threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    float total = 0.0f; for (uint n = 0; n < (head_dim + 31u) / 32u; ++n) total += red[n];\n"
"    const float inv = rsqrt(total / float(head_dim) + eps);\n"
"    if (i < head_dim) { const float g = z[base + i]; const float silu = g / (1.0f + exp(-g)); out[base + i] = x * inv * w[i] * silu; }\n"
"}\n"
/* ---- full attention (validated dev16 kernels + general GQA) ---- */
/* per head: RMSNorm (q or k weight), NeoX partial RoPE, write query_rope / key+value cache; threadgroup per head */
"kernel void attn_qk_prep(device uint *rl_abort [[buffer(30)]], device const float *qgate_raw [[buffer(0)]], device const float *k_raw [[buffer(1)]], device const float *value [[buffer(2)]],\n"
"    device const float *qw [[buffer(3)]], device const float *kw [[buffer(4)]], device float *query_rope [[buffer(5)]], device float *gate [[buffer(6)]],\n"
"    device float *key_cache [[buffer(7)]], device float *value_cache [[buffer(8)]], constant uint &head_dim [[buffer(9)]],\n"
"    constant uint &query_heads [[buffer(10)]], constant uint &kv_heads [[buffer(11)]], constant uint &position [[buffer(12)]],\n"
"    constant uint &rope_dims [[buffer(13)]], constant float &freq_base [[buffer(14)]], constant float &eps [[buffer(15)]],\n"
"    uint tg [[threadgroup_position_in_grid]], uint i [[thread_position_in_threadgroup]],\n"
"    ushort simd_lane [[thread_index_in_simdgroup]], ushort simd_id [[simdgroup_index_in_threadgroup]]) {\n"
"    if (rl_abort[0] != 0u) return;\n"
"    threadgroup float red[32]; threadgroup float tmp[256];\n"
"    const bool is_q = tg < query_heads; const uint head = is_q ? tg : tg - query_heads; if (!is_q && head >= kv_heads) return;\n"
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
"    else { const uint dst = (position * kv_heads + head) * head_dim + i; key_cache[dst] = r; value_cache[dst] = value[head * head_dim + i]; }\n"
"}\n"
"kernel void attn_gqa(device uint *rl_abort [[buffer(30)]], device const float *query [[buffer(0)]], device const float *key_cache [[buffer(1)]], device const float *value_cache [[buffer(2)]],\n"
"    device const float *gate [[buffer(3)]], device float *gated [[buffer(4)]], constant uint &head_dim [[buffer(5)]],\n"
"    constant uint &query_heads [[buffer(6)]], constant uint &kv_heads [[buffer(7)]], constant uint &seq_len [[buffer(8)]],\n"
"    uint head [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],\n"
"    ushort simd_lane [[thread_index_in_simdgroup]], ushort simd_id [[simdgroup_index_in_threadgroup]]) {\n"
"    if (rl_abort[0] != 0u) return;\n"
"    threadgroup float scores[1024]; threadgroup float red_max[8]; threadgroup float red_sum[8];\n"
"    if (head >= query_heads || seq_len == 0u) return;\n"
"    const uint qbase = head * head_dim; const uint kv_head = head / (query_heads / kv_heads); const float scale = rsqrt(float(head_dim));\n"
"    float acc = 0.0f; float m = -INFINITY; float l = 0.0f;\n"
"    for (uint chunk_start = 0; chunk_start < seq_len; chunk_start += 1024u) {\n"
"        const uint chunk = min(1024u, seq_len - chunk_start);\n"
"        for (uint pos = tid; pos < chunk; pos += 256u) {\n"
"            const uint kbase = ((chunk_start + pos) * kv_heads + kv_head) * head_dim; float dot = 0.0f;\n"
"            for (uint i = 0; i < head_dim; ++i) dot = fma(query[qbase + i], key_cache[kbase + i], dot);\n"
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
"            for (uint pos = 0; pos < chunk; ++pos) acc = fma(scores[pos], value_cache[((chunk_start + pos) * kv_heads + kv_head) * head_dim + tid], acc);\n"
"        }\n"
"        m = new_m;\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    if (tid < head_dim) gated[qbase + tid] = (acc / l) / (1.0f + exp(-gate[qbase + tid]));\n"
"}\n"
/* ---- dev26: split-K decode attention (flash decoding) ----
 * attn_gqa_split: one threadgroup per (query head, block of up to 256 positions). Scores are simdgroup-cooperative
 * dot products (lanes stride the head dimension: coalesced key reads); the block's max, exp-sum and unnormalized
 * value accumulation are written as a partial. attn_gqa_merge combines a head's partials with the online-softmax
 * algebra of attn_gqa (rescale by exp(m_b - M)), normalizes and applies the output gate. head_dim <= 256. */
"kernel void attn_gqa_split(device uint *rl_abort [[buffer(30)]], device const float *query [[buffer(0)]], device const float *key_cache [[buffer(1)]],\n"
"    device const float *value_cache [[buffer(2)]], device float2 *part_ml [[buffer(3)]], device float *part_acc [[buffer(4)]],\n"
"    constant uint &head_dim [[buffer(5)]], constant uint &query_heads [[buffer(6)]], constant uint &kv_heads [[buffer(7)]], constant uint &seq_len [[buffer(8)]],\n"
"    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],\n"
"    ushort simd_lane [[thread_index_in_simdgroup]], ushort simd_id [[simdgroup_index_in_threadgroup]]) {\n"
"    if (rl_abort[0] != 0u) return;\n"
"    threadgroup float scores[256]; threadgroup float qv[256]; threadgroup float red[8];\n"
"    const uint block = tg.x, head = tg.y, nblocks = (seq_len + 255u) / 256u;\n"
"    if (head >= query_heads || block >= nblocks) return;\n"
"    const uint start = block * 256u, len = min(256u, seq_len - start);\n"
"    const uint kv_head = head / (query_heads / kv_heads); const float scale = rsqrt(float(head_dim));\n"
"    if (tid < head_dim) qv[tid] = query[head * head_dim + tid];\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint p = simd_id; p < len; p += 8u) {\n"
"        device const float *kr = key_cache + ulong((start + p) * kv_heads + kv_head) * head_dim; float dot = 0.0f;\n"
"        for (uint i = simd_lane; i < head_dim; i += 32u) dot = fma(qv[i], kr[i], dot);\n"
"        dot = simd_sum(dot); if (simd_lane == 0) scores[p] = dot * scale;\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    float bm = tid < len ? scores[tid] : -INFINITY;\n"
"    bm = simd_max(bm); if (simd_lane == 0) red[simd_id] = bm; threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    bm = red[0]; for (uint k = 1; k < 8u; ++k) bm = max(bm, red[k]);\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    float e = 0.0f; if (tid < len) { e = exp(scores[tid] - bm); scores[tid] = e; }\n"
"    e = simd_sum(e); if (simd_lane == 0) red[simd_id] = e; threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    const ulong part = ulong(head) * nblocks + block;\n"
"    if (tid == 0u) { float bl = 0.0f; for (uint k = 0; k < 8u; ++k) bl += red[k]; part_ml[part] = float2(bm, bl); }\n"
"    if (tid < head_dim) {\n"
"        device const float *vc = value_cache + ulong(start * kv_heads + kv_head) * head_dim + tid; const ulong stride = ulong(kv_heads) * head_dim;\n"
"        float acc = 0.0f; for (uint p = 0; p < len; ++p) acc = fma(scores[p], vc[ulong(p) * stride], acc);\n"
"        part_acc[part * head_dim + tid] = acc;\n"
"    }\n"
"}\n"
/* dev35: the same partials, one threadgroup per (block of blk <= 256 positions, KV head) for all the query heads sharing
 * that KV head (8 here): each K and V row is read once instead of once per query head. Scores: one thread per position,
 * float4 K loads, one accumulator per query head. Values: one thread per dimension. head_dim == 256, <= 8 heads per group. */
"kernel void attn_gqa_split_g(device uint *rl_abort [[buffer(30)]], device const float *query [[buffer(0)]], device const float *key_cache [[buffer(1)]],\n"
"    device const float *value_cache [[buffer(2)]], device float2 *part_ml [[buffer(3)]], device float *part_acc [[buffer(4)]],\n"
"    constant uint &head_dim [[buffer(5)]], constant uint &query_heads [[buffer(6)]], constant uint &kv_heads [[buffer(7)]], constant uint &seq_len [[buffer(8)]],\n"
"    constant uint &blk [[buffer(9)]], uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],\n"
"    ushort simd_lane [[thread_index_in_simdgroup]], ushort simd_id [[simdgroup_index_in_threadgroup]]) {\n"
"    if (rl_abort[0] != 0u) return;\n"
"    threadgroup float4 qv[8 * 64]; threadgroup float sc[8 * 256];\n"
"    const uint block = tg.x, kvh = tg.y, nblocks = (seq_len + blk - 1u) / blk, G = query_heads / kv_heads;\n"
"    if (kvh >= kv_heads || block >= nblocks) return;\n"
"    const uint start = block * blk, len = min(blk, seq_len - start); const float scale = rsqrt(float(head_dim));\n"
"    for (uint i = tid; i < G * 64u; i += 256u) qv[i] = ((device const float4 *)(query + (kvh * G + i / 64u) * 256u))[i % 64u];\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    if (tid < len) {\n"
"        device const float4 *kr = (device const float4 *)(key_cache + ulong((start + tid) * kv_heads + kvh) * 256u);\n"
"        float a[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};\n"
"        for (uint i = 0; i < 64u; ++i) { const float4 k = kr[i]; for (uint h = 0; h < 8u; ++h) if (h < G) a[h] += dot(qv[h * 64u + i], k); }\n"
"        for (uint h = 0; h < 8u; ++h) if (h < G) sc[h * 256u + tid] = a[h] * scale;\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    if (simd_id < G) {   /* simdgroup h: softmax statistics of query head h over the block */\n"
"        const uint h = simd_id; float bm = -INFINITY;\n"
"        for (uint p = simd_lane; p < len; p += 32u) bm = max(bm, sc[h * 256u + p]);\n"
"        bm = simd_max(bm); float e = 0.0f;\n"
"        for (uint p = simd_lane; p < len; p += 32u) { const float v = exp(sc[h * 256u + p] - bm); sc[h * 256u + p] = v; e += v; }\n"
"        e = simd_sum(e);\n"
"        if (simd_lane == 0) part_ml[ulong(kvh * G + h) * nblocks + block] = float2(bm, e);\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    if (tid < head_dim) {\n"
"        device const float *vc = value_cache + ulong(start * kv_heads + kvh) * 256u + tid; const ulong stride = ulong(kv_heads) * 256u;\n"
"        float a[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};\n"
"        for (uint p = 0; p < len; ++p) { const float v = vc[ulong(p) * stride]; for (uint h = 0; h < 8u; ++h) if (h < G) a[h] = fma(sc[h * 256u + p], v, a[h]); }\n"
"        for (uint h = 0; h < 8u; ++h) if (h < G) part_acc[(ulong(kvh * G + h) * nblocks + block) * 256u + tid] = a[h];\n"
"    }\n"
"}\n"
"kernel void attn_gqa_merge(device uint *rl_abort [[buffer(30)]], device const float2 *part_ml [[buffer(0)]], device const float *part_acc [[buffer(1)]],\n"
"    device const float *gate [[buffer(2)]], device float *gated [[buffer(3)]], constant uint &head_dim [[buffer(4)]], constant uint &seq_len [[buffer(5)]],\n"
"    constant uint &blk [[buffer(6)]], uint head [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {\n"
"    if (rl_abort[0] != 0u) return;\n"
"    const uint nblocks = (seq_len + blk - 1u) / blk; if (tid >= head_dim || seq_len == 0u) return;\n"
"    device const float2 *ml = part_ml + ulong(head) * nblocks;\n"
"    float M = -INFINITY; for (uint b = 0; b < nblocks; ++b) M = max(M, ml[b].x);\n"
"    float l = 0.0f, acc = 0.0f;\n"
"    for (uint b = 0; b < nblocks; ++b) { const float w = exp(ml[b].x - M); l = fma(ml[b].y, w, l); acc = fma(part_acc[(ulong(head) * nblocks + b) * head_dim + tid], w, acc); }\n"
"    const uint o = head * head_dim + tid; gated[o] = (acc / l) / (1.0f + exp(-gate[o]));\n"
"}\n"
/* ---- dev21: router selection on the GPU ----
 * Same selection as rl_native_router_select_softmax_topk: top_k experts by probability, ties to the
 * lower index (softmax is monotone, so the logits order the candidates exactly); weights are the
 * softmax probabilities of the selected experts renormalized to sum 1. Looks each selected expert up
 * in the layer's residency table and writes the slot addresses / weights the expert kernels read.
 * plan_miss counts selected experts that are not resident (the token must then be redone). */
"kernel void rl_route(device uint *rl_abort [[buffer(30)]], device const float *logits [[buffer(0)]], device const ulong *resident [[buffer(1)]],\n"
"    device ulong *plan_slots [[buffer(2)]], device float *plan_weights [[buffer(3)]], device uint *plan_ids [[buffer(4)]],\n"
"    device uint *plan_miss [[buffer(5)]], constant uint &n_expert [[buffer(6)]], constant uint &top_k [[buffer(7)]],\n"
"    constant float &cache_bias [[buffer(8)]],\n"
"    uint tid [[thread_position_in_threadgroup]], ushort simd_lane [[thread_index_in_simdgroup]], ushort simd_id [[simdgroup_index_in_threadgroup]]) {\n"
"    if (rl_abort[0] != 0u) return;\n"
"    threadgroup float lg[512]; threadgroup float red_v[16]; threadgroup uint red_i[16]; threadgroup uint chosen[64];\n"
"    /* dev46: cached experts rank cache_bias higher (0 = off); the weights below use the unbiased logits */\n"
"    for (uint i = tid; i < n_expert; i += 256u) lg[i] = logits[i] + (resident[i] != 0ul ? cache_bias : 0.0f);\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    float gmax = -INFINITY; for (uint i = tid; i < n_expert; i += 256u) gmax = max(gmax, lg[i]);\n"
"    gmax = simd_max(gmax); if (simd_lane == 0) red_v[simd_id] = gmax; threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    gmax = red_v[0]; for (uint k = 1; k < 8u; ++k) gmax = max(gmax, red_v[k]);\n"
"    for (uint k = 0; k < top_k; ++k) {\n"
"        /* each thread: best (value, lowest index) among its experts not yet chosen */\n"
"        float bv = -INFINITY; uint bi = 0xFFFFFFFFu;\n"
"        for (uint i = tid; i < n_expert; i += 256u) { const float v = lg[i]; if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; } }\n"
"        const float sv = simd_max(bv);\n"
"        const uint si = simd_min(bv == sv ? bi : 0xFFFFFFFFu);\n"
"        if (simd_lane == 0) { red_v[simd_id] = sv; red_i[simd_id] = si; }\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"        if (tid == 0u) {\n"
"            float tv = red_v[0]; uint ti = red_i[0];\n"
"            for (uint q = 1; q < 8u; ++q) { if (red_v[q] > tv || (red_v[q] == tv && red_i[q] < ti)) { tv = red_v[q]; ti = red_i[q]; } }\n"
"            chosen[k] = ti; lg[ti] = -INFINITY;\n"
"        }\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    if (tid == 0u) {\n"
"        float sum = 0.0f; uint miss = 0u;\n"
"        for (uint k = 0; k < top_k; ++k) { const float pk = exp(logits[chosen[k]] - gmax); plan_weights[k] = pk; sum += pk; }\n"
"        for (uint k = 0; k < top_k; ++k) {\n"
"            plan_weights[k] = plan_weights[k] / sum; plan_ids[k] = chosen[k];\n"
"            const ulong a = resident[chosen[k]]; plan_slots[k] = a; if (a == 0ul) miss++;\n"
"        }\n"
"        plan_miss[0] = miss; if (miss) rl_abort[0] = 1u;\n"
"    }\n"
"}\n"
/* dev45: rl_route for the two verify rows (one threadgroup each); row 1 writes its plan right after row 0's */
"kernel void rl_route2(device uint *rl_abort [[buffer(30)]], device const float *logits_r0 [[buffer(0)]], device const ulong *resident [[buffer(1)]],\n"
"    device ulong *plan_slots_r0 [[buffer(2)]], device float *plan_weights_r0 [[buffer(3)]], device uint *plan_ids_r0 [[buffer(4)]],\n"
"    device uint *plan_miss_r0 [[buffer(5)]], constant uint &n_expert [[buffer(6)]], constant uint &top_k [[buffer(7)]],\n"
"    device const float *logits_r1 [[buffer(9)]], uint row [[threadgroup_position_in_grid]],\n"
"    uint tid [[thread_position_in_threadgroup]], ushort simd_lane [[thread_index_in_simdgroup]], ushort simd_id [[simdgroup_index_in_threadgroup]]) {\n"
"    if (rl_abort[0] != 0u) return;\n"
"    device const float *logits = row ? logits_r1 : logits_r0; device ulong *plan_slots = plan_slots_r0 + row * top_k;\n"
"    device float *plan_weights = plan_weights_r0 + row * top_k; device uint *plan_ids = plan_ids_r0 + row * top_k; device uint *plan_miss = plan_miss_r0 + row;\n"
"    threadgroup float lg[512]; threadgroup float red_v[16]; threadgroup uint red_i[16]; threadgroup uint chosen[64];\n"
"    for (uint i = tid; i < n_expert; i += 256u) lg[i] = logits[i];\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    float gmax = -INFINITY; for (uint i = tid; i < n_expert; i += 256u) gmax = max(gmax, lg[i]);\n"
"    gmax = simd_max(gmax); if (simd_lane == 0) red_v[simd_id] = gmax; threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    gmax = red_v[0]; for (uint k = 1; k < 8u; ++k) gmax = max(gmax, red_v[k]);\n"
"    for (uint k = 0; k < top_k; ++k) {\n"
"        /* each thread: best (value, lowest index) among its experts not yet chosen */\n"
"        float bv = -INFINITY; uint bi = 0xFFFFFFFFu;\n"
"        for (uint i = tid; i < n_expert; i += 256u) { const float v = lg[i]; if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; } }\n"
"        const float sv = simd_max(bv);\n"
"        const uint si = simd_min(bv == sv ? bi : 0xFFFFFFFFu);\n"
"        if (simd_lane == 0) { red_v[simd_id] = sv; red_i[simd_id] = si; }\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"        if (tid == 0u) {\n"
"            float tv = red_v[0]; uint ti = red_i[0];\n"
"            for (uint q = 1; q < 8u; ++q) { if (red_v[q] > tv || (red_v[q] == tv && red_i[q] < ti)) { tv = red_v[q]; ti = red_i[q]; } }\n"
"            chosen[k] = ti; lg[ti] = -INFINITY;\n"
"        }\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    if (tid == 0u) {\n"
"        float sum = 0.0f; uint miss = 0u;\n"
"        for (uint k = 0; k < top_k; ++k) { const float pk = exp(logits[chosen[k]] - gmax); plan_weights[k] = pk; sum += pk; }\n"
"        for (uint k = 0; k < top_k; ++k) {\n"
"            plan_weights[k] = plan_weights[k] / sum; plan_ids[k] = chosen[k];\n"
"            const ulong a = resident[chosen[k]]; plan_slots[k] = a; if (a == 0ul) miss++;\n"
"        }\n"
"        plan_miss[0] = miss; if (miss) rl_abort[0] = 1u;\n"
"    }\n"
"}\n"
/* ---- shared expert (validated dev14 kernels) ---- */
"kernel void sh_scalar_gate(device uint *rl_abort [[buffer(30)]], device const float *w [[buffer(0)]], device const float *x [[buffer(1)]], device float *scalar [[buffer(2)]],\n"
"    constant uint &hidden [[buffer(3)]], uint tid [[thread_position_in_threadgroup]], ushort simd_lane [[thread_index_in_simdgroup]],\n"
"    ushort simd_id [[simdgroup_index_in_threadgroup]]) {\n"
"    if (rl_abort[0] != 0u) return;\n"
"    threadgroup float partial[8]; float acc = 0.0f;\n"
"    for (uint i = tid; i < hidden; i += 256u) acc = fma(w[i], x[i], acc);\n"
"    acc = simd_sum(acc); if (simd_lane == 0) partial[simd_id] = acc; threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    if (tid == 0u) { float total = 0.0f; for (uint k = 0; k < 8u; ++k) total += partial[k]; scalar[0] = 1.0f / (1.0f + exp(-total)); }\n"
"}\n"
"kernel void sh_silu_mul(device uint *rl_abort [[buffer(30)]], device const float *gate [[buffer(0)]], device const float *up [[buffer(1)]], device float *act [[buffer(2)]],\n"
"    constant uint &count [[buffer(3)]], uint gid [[thread_position_in_grid]]) {\n"
"    if (rl_abort[0] != 0u) return;\n"
"    if (gid >= count) return; const float g = gate[gid]; act[gid] = (g / (1.0f + exp(-g))) * up[gid];\n"
"}\n"
"kernel void rl_copy_f32(device uint *rl_abort [[buffer(30)]], device const float *src [[buffer(0)]], device float *dst [[buffer(1)]], constant uint &count [[buffer(2)]],\n"
"    uint gid [[thread_position_in_grid]]) {\n"
"    if (rl_abort[0] != 0u) return;\n"
"    if (gid < count) dst[gid] = src[gid];\n"
"}\n";

/* the engine's kernel library (also compiled by the model-free kernel self-test, redmetal_engine_selftest.m) */
id<MTLLibrary> rl_metal_engine_library(id<MTLDevice> dev, NSError **err) {
    char *iq3 = rl_iq3_metal_source();   /* dev31: codebooks + rl_iq3_group8 */
    if (!iq3) return nil;
    NSString *src = [NSString stringWithFormat:@"#include <metal_stdlib>\nusing namespace metal;\n%s\n%@", iq3, kEngineSource];
    free(iq3);
    return [dev newLibraryWithSource:src options:nil error:err];
}

id<MTLComputePipelineState> make_pipe(id<MTLDevice> dev, id<MTLLibrary> lib, NSString *name, char *error, size_t cap) {
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    if (!fn) { snprintf(error, cap, "missing Metal function %s", name.UTF8String); return nil; }
    NSError *e = nil;
    id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:fn error:&e];
    if (!p) snprintf(error, cap, "pipeline %s failed: %s", name.UTF8String, e.localizedDescription.UTF8String ?: "unknown");
    return p;
}

id<MTLBuffer> new_buf(rl_metal_engine *m, size_t bytes) {
    id<MTLBuffer> b = [m->dev newBufferWithLength:bytes ? bytes : 16u options:MTLResourceStorageModeShared];
    if (b) { m->resident_bytes += bytes; [m->keep addObject:b]; }
    return b;
}

static int wrap_tensor(rl_metal_engine *m, const rl_gguf_model *g, const rl_gguf_tensor *t, mweight *out, char *error, size_t cap) {
    out->buf = nil; out->off = 0; out->type = 0; out->rows = 0; out->cols = 0;
    if (!t) return 1;
    const uint8_t *data = rl_gguf_tensor_data(g, t);
    if (!data) { snprintf(error, cap, "tensor %s is not mapped", t->name); return 0; }
    const size_t page = (size_t)sysconf(_SC_PAGESIZE);
    const uintptr_t base = (uintptr_t)data;
    const uintptr_t aligned = base & ~(uintptr_t)(page - 1u);
    size_t len = (size_t)(base + t->payload_bytes - aligned);
    len = (len + page - 1u) & ~(page - 1u);
    const uintptr_t map_end = ((uintptr_t)g->map + g->map_len + page - 1u) & ~(uintptr_t)(page - 1u);
    if (aligned + len > map_end) len = (size_t)(map_end - aligned);
    id<MTLBuffer> b = [m->dev newBufferWithBytesNoCopy:(void *)aligned length:len options:MTLResourceStorageModeShared deallocator:nil];
    if (b) {
        out->buf = b;
        out->off = (NSUInteger)(base - aligned);
    } else {
        b = [m->dev newBufferWithBytes:data length:(NSUInteger)t->payload_bytes options:MTLResourceStorageModeShared];
        if (!b) { snprintf(error, cap, "failed to create Metal buffer for %s", t->name); return 0; }
        out->buf = b;
        out->off = 0;
    }
    [m->keep addObject:b];
    out->type = t->ggml_type;
    out->cols = (uint32_t)t->shape[0];
    out->rows = t->n_dims > 1 ? (uint32_t)t->shape[1] : 1u;
    m->resident_bytes += t->payload_bytes;
    return 1;
}

id<MTLComputePipelineState> rows_pipe(rl_metal_engine *m, uint32_t type) {
    switch (type) {
        case 0: return m->p_rows_f32;
        case 8: return m->p_rows_q8;
        case 12: return m->p_rows_q4k;
        case 13: return m->p_rows_q5k;
        case 14: return m->p_rows_q6k;
        case 16: return m->p_rows_iq2xxs;
        case 18: case 21: case 22: case 23: return m->p_rows_iq3;   /* dev31 */
        default: return nil;
    }
}

static double redmetal_topk_pool_read_ms_total(struct rl_metal_engine *m) {
    return rl_native_metal_read_ms(m->experts);
}

/* dev38: decode encoders may be concurrent (m->concurrent). Every dispatch is then followed by a buffer barrier, except
 * inside an em_group of dispatches that neither read nor write each other's outputs (one barrier at the group's end).
 * Serial encoders (prefill, profile, RL_ENGINE_CONCURRENT=0) are unaffected. */
static _Thread_local int g_rl_group;
static inline void rl_after_dispatch(id<MTLComputeCommandEncoder> enc) {
    if (!g_rl_group && enc.dispatchType == MTLDispatchTypeConcurrent) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

void enc_1d(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p, NSUInteger n, NSUInteger tg_max) {
    const NSUInteger tg = MIN(tg_max, p.maxTotalThreadsPerThreadgroup);
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg ? tg : 1, 1, 1)]; rl_after_dispatch(enc);
}

uint32_t lanes_for(uint32_t type, uint32_t ncols) {
    uint32_t blocks;
    if (type == 0u) blocks = ncols / 64u;          /* F32: strided lanes */
    else if (type == 8u) blocks = ncols / 32u;
    else blocks = ncols / 256u;
    uint32_t lanes = 1u;
    while (lanes < blocks && lanes < 32u) lanes <<= 1;
    return lanes;
}

/* out[row] = W[row] . x for all rows of the weight, as one dispatch in an open encoder */
/* dev22: sub-block decode kernel for a type (nil: use the block kernel) and its lanes per row */
static id<MTLComputePipelineState> rows2_pipe(rl_metal_engine *m, uint32_t type, uint32_t ncols, uint32_t *lanes) {
    uint32_t items;
    id<MTLComputePipelineState> p;
    switch (type) {
        case 0:  p = m->p_rows2_f32;    items = ncols / 4u; break;
        case 12: p = m->p_rows2_q4k;    items = (ncols / 256u) * 8u; break;
        case 14: p = m->p_rows2_q6k;    items = (ncols / 256u) * 16u; break;
        case 16: p = m->p_rows2_iq2xxs; items = (ncols / 256u) * 8u; break;
        case 18: case 21: case 22: case 23: p = m->p_rows_iq3; items = (ncols / 256u) * 8u; break;   /* dev31: already sub-block */
        default: return nil;
    }
    if (!m->rows2 || (type == 0u && ncols % 4u)) return nil;
    uint32_t l = 1u;
    while (l < items && l < 32u) l <<= 1;
    *lanes = l;
    return p;
}

void emit_rows(rl_metal_engine *m, id<MTLComputeCommandEncoder> enc, const mweight *w, id<MTLBuffer> x, id<MTLBuffer> out) {
    uint32_t lanes = 0;
    id<MTLComputePipelineState> p = rows2_pipe(m, w->type, w->cols, &lanes);
    if (!p) { p = rows_pipe(m, w->type); lanes = lanes_for(w->type, w->cols); }
    [enc setComputePipelineState:p];
    [enc setBuffer:w->buf offset:w->off atIndex:0];
    [enc setBytes:&w->cols length:sizeof(w->cols) atIndex:1];
    [enc setBuffer:x offset:0 atIndex:2];
    [enc setBuffer:out offset:0 atIndex:3];
    [enc setBytes:&w->rows length:sizeof(w->rows) atIndex:4];
    [enc setBytes:&lanes length:sizeof(lanes) atIndex:5];
    if (w->type == 16u) [enc setBuffer:m->grid offset:0 atIndex:6];
    if (rl_iq3_supported(w->type)) [enc setBytes:&w->type length:sizeof(w->type) atIndex:7];
    const NSUInteger threads = (((NSUInteger)w->rows * lanes) + 31u) & ~(NSUInteger)31u;
    [enc dispatchThreads:MTLSizeMake(threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)]; rl_after_dispatch(enc);
}

/* dev45: the same product for two activation vectors in one dispatch where a two-vector kernel exists (each row's
 * result equals emit_rows' for that row: same lanes, same per-row order); two emit_rows otherwise */
static void emit_rows_r2(rl_metal_engine *m, id<MTLComputeCommandEncoder> enc, const mweight *w, id<MTLBuffer> x0, id<MTLBuffer> x1,
                         id<MTLBuffer> out0, id<MTLBuffer> out1) {
    uint32_t lanes = 0;
    id<MTLComputePipelineState> p = nil;
    if (rows2_pipe(m, w->type, w->cols, &lanes)) {
        p = w->type == 0u ? m->p_r2_f32 : w->type == 12u ? m->p_r2_q4k : w->type == 14u ? m->p_r2_q6k : w->type == 16u ? m->p_r2_iq2xxs :
            rl_iq3_supported(w->type) ? m->p_r2_iq3 : nil;
    } else if (w->type == 8u || w->type == 13u) {
        p = w->type == 8u ? m->p_r2_q8 : m->p_r2_q5k;
        lanes = lanes_for(w->type, w->cols);
    }
    if (!p) { emit_rows(m, enc, w, x0, out0); emit_rows(m, enc, w, x1, out1); return; }
    [enc setComputePipelineState:p];
    [enc setBuffer:w->buf offset:w->off atIndex:0];
    [enc setBytes:&w->cols length:sizeof(w->cols) atIndex:1];
    [enc setBuffer:x0 offset:0 atIndex:2]; [enc setBuffer:out0 offset:0 atIndex:3];
    [enc setBytes:&w->rows length:sizeof(w->rows) atIndex:4];
    [enc setBytes:&lanes length:sizeof(lanes) atIndex:5];
    if (w->type == 16u) [enc setBuffer:m->grid offset:0 atIndex:6];
    if (rl_iq3_supported(w->type)) [enc setBytes:&w->type length:sizeof(w->type) atIndex:7];
    [enc setBuffer:x1 offset:0 atIndex:8]; [enc setBuffer:out1 offset:0 atIndex:9];
    const NSUInteger threads = (((NSUInteger)w->rows * lanes) + 31u) & ~(NSUInteger)31u;
    [enc dispatchThreads:MTLSizeMake(threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)]; rl_after_dispatch(enc);
}

void enc_rows(rl_metal_engine *m, id<MTLCommandBuffer> cb, const mweight *w, id<MTLBuffer> x, id<MTLBuffer> out) {
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setBuffer:m->abort_zero offset:0 atIndex:30];
    emit_rows(m, enc, w, x, out);
    [enc endEncoding];
}

void emit_rms(rl_metal_engine *m, id<MTLComputeCommandEncoder> enc, id<MTLBuffer> x, const mweight *w, id<MTLBuffer> y, uint32_t n, float eps) {
    [enc setComputePipelineState:m->p_rms];
    [enc setBuffer:x offset:0 atIndex:0];
    [enc setBuffer:w->buf offset:w->off atIndex:1];
    [enc setBuffer:y offset:0 atIndex:2];
    [enc setBytes:&n length:sizeof(n) atIndex:3];
    [enc setBytes:&eps length:sizeof(eps) atIndex:4];
    [enc dispatchThreads:MTLSizeMake(256, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)]; rl_after_dispatch(enc);
}

void enc_rms(rl_metal_engine *m, id<MTLCommandBuffer> cb, id<MTLBuffer> x, const mweight *w, id<MTLBuffer> y, uint32_t n, float eps) {
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setBuffer:m->abort_zero offset:0 atIndex:30];
    emit_rms(m, enc, x, w, y, n, eps);
    [enc endEncoding];
}

static const char *kStageNames[16] = {"rms", "dn_proj", "dn_prestate", "dn_state", "dn_tail", "attn_proj", "attn_norm_rope", "attn_gqa", "attn_out", "resid_rms", "router", "shared", "experts", "head", "", ""};

static id<MTLCommandBuffer> stage_end(rl_metal_engine *m, id<MTLCommandBuffer> cb, int stage) {
    if (!m->profile) return cb;
    [cb commit];
    [cb waitUntilCompleted];
    m->prof_ms[stage] += (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
    return [m->queue commandBuffer];
}

void rl_metal_engine_profile_report(rl_metal_engine *m) {
    if (!m || !m->profile) return;
    fprintf(stderr, "--- GPU stage profile (cumulative ms) ---\n");
    for (int i = 0; i < 14; ++i) fprintf(stderr, "  %-16s %.2f ms\n", kStageNames[i], m->prof_ms[i]);
}

int commit_wait(id<MTLCommandBuffer> cb, const char *what, double *gpu_ms, char *error, size_t cap) {
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted || cb.error) {
        snprintf(error, cap, "%s command buffer failed: %s", what, cb.error.localizedDescription.UTF8String ?: "unknown");
        return 0;
    }
    if (gpu_ms) *gpu_ms += (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
    return 1;
}

rl_metal_engine *rl_metal_engine_create(rl_engine *e, char *error, size_t cap) {
    rl_metal_engine *m = (rl_metal_engine *)calloc(1, sizeof(*m));
    if (!m) { set_error(error, cap, "Metal engine allocation failed"); return NULL; }
    const rl_engine_info *in = &e->info;
    @autoreleasepool {
        m->keep = [NSMutableArray array];
        m->profile = getenv("RL_ENGINE_PROFILE") != NULL;
        { const char *r2 = getenv("RL_ENGINE_ROWS2"); m->rows2 = !r2 || atoi(r2) != 0; }
        m->dev = MTLCreateSystemDefaultDevice();
        if (!m->dev || !m->dev.hasUnifiedMemory) { set_error(error, cap, "Apple unified-memory Metal device required"); rl_metal_engine_destroy(m); return NULL; }
        m->queue = [m->dev newCommandQueue];
        NSError *le = nil;
        m->lib = rl_metal_engine_library(m->dev, &le);
        if (!m->queue || !m->lib) {
            snprintf(error, cap, "Metal engine library compile failed: %s", le.localizedDescription.UTF8String ?: "unknown");
            rl_metal_engine_destroy(m); return NULL;
        }
        struct { __strong id<MTLComputePipelineState> *slot; NSString *name; } pipes[] = {
            {&m->p_rms, @"rl_rms"}, {&m->p_resid_rms, @"rl_resid_rms"}, {&m->p_scale_add, @"rl_scale_add"}, {&m->p_steer, @"rl_steer_add"}, {&m->p_moe_tail, @"rl_moe_tail"},
            {&m->p_rows_f32, @"rl_rows_f32"}, {&m->p_rows_q8, @"rl_rows_q8"}, {&m->p_rows_q4k, @"rl_rows_q4k"},
            {&m->p_rows_q5k, @"rl_rows_q5k"}, {&m->p_rows_q6k, @"rl_rows_q6k"}, {&m->p_rows_iq2xxs, @"rl_rows_iq2xxs"}, {&m->p_rows_iq3, @"rl_rows_iq3"},
            {&m->p_dn_ba, @"dn_ba_params"}, {&m->p_dn_conv, @"dn_conv_silu"}, {&m->p_dn_l2, @"dn_qk_l2_tg"},
            {&m->p_dn_shift, @"dn_shift_state"}, {&m->p_dn_state, @"dn_state_fused"}, {&m->p_dn_tail, @"dn_tail_norm"},
            {&m->p_attn_prep, @"attn_qk_prep"}, {&m->p_attn_gqa, @"attn_gqa"}, {&m->p_sh_scalar, @"sh_scalar_gate"}, {&m->p_sh_silu, @"sh_silu_mul"},
            {&m->p_route, @"rl_route"}, {&m->p_copy, @"rl_copy_f32"},
            {&m->p_rows2_q4k, @"rl_rows2_q4k"}, {&m->p_rows2_q6k, @"rl_rows2_q6k"}, {&m->p_rows2_iq2xxs, @"rl_rows2_iq2xxs"}, {&m->p_rows2_f32, @"rl_rows2_f32"},
            {&m->p_attn_split, @"attn_gqa_split"}, {&m->p_attn_merge, @"attn_gqa_merge"}, {&m->p_attn_split_g, @"attn_gqa_split_g"},
            {&m->p_r2_f32, @"rl_rows2r2_f32"}, {&m->p_r2_q8, @"rl_rowsr2_q8"}, {&m->p_r2_q4k, @"rl_rows2r2_q4k"},
            {&m->p_r2_q5k, @"rl_rowsr2_q5k"}, {&m->p_r2_q6k, @"rl_rows2r2_q6k"}, {&m->p_r2_iq2xxs, @"rl_rows2r2_iq2xxs"}, {&m->p_r2_iq3, @"rl_rows_iq3_r2"},
            {&m->p_dn_ba2, @"dn_ba_params2"}, {&m->p_dn_convshift2, @"dn_conv_shift2"}, {&m->p_dn_l2_2, @"dn_qk_l2_2"},
            {&m->p_dn_state2, @"dn_state2"}, {&m->p_dn_tail2, @"dn_tail_norm2"},
            {&m->p_route2, @"rl_route2"}, {&m->p_moe_tail2, @"rl_moe_tail2"},
        };
        for (size_t i = 0; i < sizeof(pipes) / sizeof(pipes[0]); ++i) {
            *pipes[i].slot = make_pipe(m->dev, m->lib, pipes[i].name, error, cap);
            if (!*pipes[i].slot) { rl_metal_engine_destroy(m); return NULL; }
        }
        m->grid = [m->dev newBufferWithBytes:e->iq2_grid length:RL_IQ2_XXS_GRID_COUNT options:MTLResourceStorageModeShared];
        { const char *pf = getenv("RL_ENGINE_PREFETCH"); m->prefetch = !pf || atoi(pf) != 0; }
        m->pred_logits = new_buf(m, (size_t)in->n_expert * sizeof(float));
        m->abort = new_buf(m, 16u);
        m->abort_zero = new_buf(m, 16u);
        if (!m->abort || !m->abort_zero) { set_error(error, cap, "Metal early-out flag allocation failed"); rl_metal_engine_destroy(m); return NULL; }
        memset(m->abort.contents, 0, 16u); memset(m->abort_zero.contents, 0, 16u);

        /* dense weights */
        m->layers = (mlayer *)calloc(in->n_layer, sizeof(*m->layers));
        if (!m->layers || !m->grid) { set_error(error, cap, "Metal layer table allocation failed"); rl_metal_engine_destroy(m); return NULL; }
        for (uint32_t l = 0; l < in->n_layer; ++l) {
            const rl_layer_tensors *t = &e->layers[l];
            mlayer *w = &m->layers[l];
            struct { mweight *dst; const rl_gguf_tensor *src; } pairs[] = {
                {&w->attn_norm, t->attn_norm}, {&w->post_norm, t->post_norm}, {&w->qkv, t->qkv}, {&w->z, t->z}, {&w->ba, t->ba},
                {&w->conv, t->conv}, {&w->dt, t->dt}, {&w->a, t->a}, {&w->ssm_norm, t->ssm_norm}, {&w->ssm_out, t->ssm_out},
                {&w->q, t->q}, {&w->k, t->k}, {&w->v, t->v}, {&w->q_norm, t->q_norm}, {&w->k_norm, t->k_norm}, {&w->o, t->o},
                {&w->router, t->router}, {&w->sh_gate_inp, t->sh_gate_inp}, {&w->sh_gate, t->sh_gate}, {&w->sh_up, t->sh_up}, {&w->sh_down, t->sh_down},
            };
            for (size_t i = 0; i < sizeof(pairs) / sizeof(pairs[0]); ++i) {
                if (!wrap_tensor(m, &e->gguf, pairs[i].src, pairs[i].dst, error, cap)) { rl_metal_engine_destroy(m); return NULL; }
                if (pairs[i].src && pairs[i].src->n_dims == 2u && !rows_pipe(m, pairs[i].src->ggml_type)) {
                    snprintf(error, cap, "tensor %s type %s has no Metal row kernel", pairs[i].src->name, rl_gguf_type_name(pairs[i].src->ggml_type));
                    rl_metal_engine_destroy(m); return NULL;
                }
            }
        }
        if (!wrap_tensor(m, &e->gguf, e->output_norm, &m->output_norm, error, cap) ||
            !wrap_tensor(m, &e->gguf, e->output, &m->output, error, cap) || !rows_pipe(m, e->output->ggml_type)) {
            if (error && cap && !error[0]) set_error(error, cap, "LM head tensor has no Metal row kernel");
            rl_metal_engine_destroy(m); return NULL;
        }

        /* states */
        const size_t conv_bytes = rl_engine_conv_count(e) * sizeof(float);
        const size_t rec_bytes = rl_engine_rec_count(e) * sizeof(float);
        /* + RL_ENGINE_KV_PAD positions: the dev30 tiled prefill attention reads whole 32-position key blocks */
        const size_t kv_bytes = rl_engine_kv_row_count(e) * ((size_t)in->context + RL_ENGINE_KV_PAD) * sizeof(float);
        m->conv_state = (__unsafe_unretained id<MTLBuffer> *)calloc(in->n_recurrent ? in->n_recurrent : 1u, sizeof(id));
        m->rec_state = (__unsafe_unretained id<MTLBuffer> *)calloc(in->n_recurrent ? in->n_recurrent : 1u, sizeof(id));
        /* + 1: dev45 MTP block KV rows at index n_attention (m->n_attention stays the trunk's count) */
        m->kcache = (__unsafe_unretained id<MTLBuffer> *)calloc(in->n_attention + 1u, sizeof(id));
        m->vcache = (__unsafe_unretained id<MTLBuffer> *)calloc(in->n_attention + 1u, sizeof(id));
        if (!m->conv_state || !m->rec_state || !m->kcache || !m->vcache) { set_error(error, cap, "state table allocation failed"); rl_metal_engine_destroy(m); return NULL; }
        m->n_recurrent = in->n_recurrent; m->n_attention = in->n_attention;
        for (uint32_t r = 0; r < in->n_recurrent; ++r) {
            m->conv_state[r] = new_buf(m, conv_bytes);
            m->rec_state[r] = new_buf(m, rec_bytes);
            if (!m->conv_state[r] || !m->rec_state[r]) { set_error(error, cap, "DeltaNet state allocation failed"); rl_metal_engine_destroy(m); return NULL; }
        }
        for (uint32_t a = 0; a < in->n_attention; ++a) {
            m->kcache[a] = new_buf(m, kv_bytes);
            m->vcache[a] = new_buf(m, kv_bytes);
            if (!m->kcache[a] || !m->vcache[a]) { set_error(error, cap, "KV cache allocation failed"); rl_metal_engine_destroy(m); return NULL; }
        }
        /* dev26 split-K decode attention partials, sized for the whole context; RL_ENGINE_ATTN_SPLIT=0 for A/B runs */
        { const char *as = getenv("RL_ENGINE_ATTN_SPLIT"); m->attn_split = (!as || atoi(as) != 0) && in->head_dim <= 256u && in->n_attention; }
        { const char *ag = getenv("RL_ENGINE_ATTN_GROUP"); m->attn_group = !ag || atoi(ag) != 0; }
        { const char *ft = getenv("RL_ENGINE_FUSE_TAIL"); m->fuse_tail = !ft || atoi(ft) != 0; }
        { const char *rb = getenv("RL_ROUTE_CACHE_BIAS"); m->route_bias = rb ? (float)atof(rb) : 0.0f; if (!(m->route_bias > 0.0f)) m->route_bias = 0.0f; }
        { const char *cc = getenv("RL_ENGINE_CONCURRENT"); m->concurrent = !m->profile && (!cc || atoi(cc) != 0); }
        { const char *ab = getenv("RL_ENGINE_ATTN_BLK"); m->attn_blk = ab && atoi(ab) >= 32 && atoi(ab) <= 256 ? (uint32_t)atoi(ab) : 128u; }
        if (m->attn_split) {
            const size_t nblocks = ((size_t)in->context + 31u) / 32u;   /* dev35: blocks of down to 32 positions */
            m->attn_ml = new_buf(m, (size_t)in->n_head * nblocks * 2u * sizeof(float));
            m->attn_acc = new_buf(m, (size_t)in->n_head * nblocks * in->head_dim * sizeof(float));
            if (!m->attn_ml || !m->attn_acc) { set_error(error, cap, "attention partial allocation failed"); rl_metal_engine_destroy(m); return NULL; }
        }

        /* scratch */
        const size_t hb = (size_t)in->hidden * sizeof(float);
        const uint32_t qcount = in->n_head * in->head_dim, kvcount = in->n_head_kv * in->head_dim;
        const uint32_t ffn_sh = e->gguf.n_ff_shexp;
        m->x = new_buf(m, hb); m->normed = new_buf(m, hb); m->branch = new_buf(m, hb); m->resid = new_buf(m, hb); m->ffn_in = new_buf(m, hb);
        m->qkv = new_buf(m, (size_t)in->channels * 4u); m->z = new_buf(m, (size_t)in->d_inner * 4u); m->ba = new_buf(m, (size_t)2u * in->dt_rank * 4u);
        m->beta = new_buf(m, (size_t)in->dt_rank * 4u); m->gate = new_buf(m, (size_t)in->dt_rank * 4u);
        m->conv_silu = new_buf(m, (size_t)in->channels * 4u);
        m->q = new_buf(m, (size_t)in->n_group * in->d_state * 4u); m->k = new_buf(m, (size_t)in->n_group * in->d_state * 4u);
        m->delta = new_buf(m, (size_t)in->d_inner * 4u); m->core = new_buf(m, (size_t)in->d_inner * 4u); m->ng = new_buf(m, (size_t)in->d_inner * 4u);
        m->next_conv = new_buf(m, conv_bytes); m->rec_scratch = new_buf(m, rec_bytes);
        m->qgate_raw = new_buf(m, (size_t)2u * qcount * 4u); m->k_raw = new_buf(m, (size_t)kvcount * 4u); m->value = new_buf(m, (size_t)kvcount * 4u);
        m->query = new_buf(m, (size_t)qcount * 4u); m->agate = new_buf(m, (size_t)qcount * 4u); m->key = new_buf(m, (size_t)kvcount * 4u);
        m->query_rope = new_buf(m, (size_t)qcount * 4u); m->gated = new_buf(m, (size_t)qcount * 4u);
        m->router_logits = new_buf(m, (size_t)in->n_expert * 4u);
        m->sh_gate = new_buf(m, (size_t)ffn_sh * 4u); m->sh_up = new_buf(m, (size_t)ffn_sh * 4u); m->sh_act = new_buf(m, (size_t)ffn_sh * 4u);
        m->sh_out = new_buf(m, hb); m->scalar = new_buf(m, 16u); m->routed = new_buf(m, hb); m->final_norm = new_buf(m, hb);
        m->logits = new_buf(m, (size_t)in->vocab * 4u);
        m->routed_host = (float *)calloc(in->hidden, sizeof(float));
        if (!m->x || !m->normed || !m->branch || !m->resid || !m->ffn_in || !m->qkv || !m->z || !m->ba || !m->beta || !m->gate ||
            !m->conv_silu || !m->q || !m->k || !m->delta || !m->core || !m->ng || !m->next_conv || !m->rec_scratch ||
            !m->qgate_raw || !m->k_raw || !m->value || !m->query || !m->agate || !m->key || !m->query_rope || !m->gated ||
            !m->router_logits || !m->sh_gate || !m->sh_up || !m->sh_act || !m->sh_out || !m->scalar || !m->routed ||
            !m->final_norm || !m->logits || !m->routed_host) {
            set_error(error, cap, "Metal scratch allocation failed"); rl_metal_engine_destroy(m); return NULL;
        }

        /* routed experts: one bounded LRU shared by every layer */
        m->experts = rl_native_metal_create(e->gguf.path, &e->expert_map, e->cfg.cache_mib * 1024ull * 1024ull, 64u, error, cap);
        if (!m->experts) { rl_metal_engine_destroy(m); return NULL; }

        /* expert slabs stay resident for this queue without per-encoder useResource (macOS 15+) */
        if (@available(macOS 15.0, *)) {
            id<MTLResidencySet> rs = (__bridge id<MTLResidencySet>)redmetal_topk_pool_slab_residency_set(rl_native_metal_pool_handle(m->experts));
            if (rs) [m->queue addResidencySet:rs];
        }
        /* dev45: MTP block: dense weights wrapped from its file, all 512 experts resident in a pool of their own */
        if (e->mtp_enabled) {
            const rl_layer_tensors *t = &e->mtp;
            mlayer *w = &m->mtp_w;
            struct { mweight *dst; const rl_gguf_tensor *src; } pairs[] = {
                {&w->attn_norm, t->attn_norm}, {&w->post_norm, t->post_norm}, {&w->q, t->q}, {&w->k, t->k}, {&w->v, t->v},
                {&w->q_norm, t->q_norm}, {&w->k_norm, t->k_norm}, {&w->o, t->o}, {&w->router, t->router},
                {&w->sh_gate_inp, t->sh_gate_inp}, {&w->sh_gate, t->sh_gate}, {&w->sh_up, t->sh_up}, {&w->sh_down, t->sh_down},
                {&m->mtp_eh_proj, e->mtp_eh_proj}, {&m->mtp_enorm, e->mtp_enorm}, {&m->mtp_hnorm, e->mtp_hnorm},
                {&m->mtp_head_norm, e->mtp_head_norm}, {&m->mtp_head, e->mtp_head},
            };
            for (size_t i = 0; i < sizeof(pairs) / sizeof(pairs[0]); ++i) {
                if (!wrap_tensor(m, &e->mtp_gguf, pairs[i].src, pairs[i].dst, error, cap)) { rl_metal_engine_destroy(m); return NULL; }
                if (pairs[i].src->n_dims == 2u && !rows_pipe(m, pairs[i].src->ggml_type)) {
                    snprintf(error, cap, "MTP tensor %s has no Metal row kernel", pairs[i].src->name); rl_metal_engine_destroy(m); return NULL;
                }
            }
            const size_t kvb = rl_engine_kv_row_count(e) * ((size_t)in->context + RL_ENGINE_KV_PAD) * sizeof(float);
            m->kcache[in->n_attention] = new_buf(m, kvb);
            m->vcache[in->n_attention] = new_buf(m, kvb);
            m->mtp_emb = new_buf(m, hb); m->mtp_ea = new_buf(m, hb); m->mtp_eb = new_buf(m, hb); m->mtp_cat = new_buf(m, 2u * hb);
            m->mtp_slots = new_buf(m, 512u * sizeof(uint64_t)); m->mtp_weights = new_buf(m, 512u * sizeof(float));
            m->mtp_ids = new_buf(m, 64u * sizeof(uint32_t)); m->mtp_miss = new_buf(m, 16u);
            const uint64_t slot = (e->mtp_expert_map.max_expert_triplet_bytes + 4095u) & ~(uint64_t)4095u;
            m->mtp_experts = rl_native_metal_create(e->mtp_gguf.path, &e->mtp_expert_map, slot * in->n_expert + slot, 64u, error, cap);
            char perr[256] = "";
            uint32_t *all_ids = (uint32_t *)malloc((size_t)in->n_expert * sizeof(uint32_t));
            float *zero_w = (float *)calloc(in->n_expert, sizeof(float));
            int ok = m->kcache[in->n_attention] && m->vcache[in->n_attention] && m->mtp_emb && m->mtp_ea && m->mtp_eb && m->mtp_cat &&
                     m->mtp_slots && m->mtp_weights && m->mtp_ids && m->mtp_miss && m->mtp_experts && all_ids && zero_w &&
                     rl_native_metal_residency_enable(m->mtp_experts, e->mtp_layer + 1u, in->n_expert, perr, sizeof(perr));
            if (ok) {
                for (uint32_t x = 0; x < in->n_expert; ++x) all_ids[x] = x;
                rl_native_topk_plan plan; memset(&plan, 0, sizeof(plan));
                ok = rl_native_metal_prepare_topk(m->mtp_experts, &e->mtp_expert_map, e->mtp_layer, all_ids, zero_w, in->n_expert, &plan, perr, sizeof(perr)) &&
                     rl_native_metal_release_topk(m->mtp_experts, &plan, NULL, perr, sizeof(perr));
            }
            free(all_ids); free(zero_w);
            if (!ok) { snprintf(error, cap, "MTP block setup failed: %s", perr[0] ? perr : (error && error[0] ? error : "allocation")); rl_metal_engine_destroy(m); return NULL; }
            if (@available(macOS 15.0, *)) {
                id<MTLResidencySet> rs = (__bridge id<MTLResidencySet>)redmetal_topk_pool_slab_residency_set(rl_native_metal_pool_handle(m->mtp_experts));
                if (rs) [m->queue addResidencySet:rs];
            }
            m->mtp = 1;
        }
        /* dev30: the dense weights are no-copy windows of the mmap; without a residency set every command buffer
         * re-establishes their residency (~3.8 ms per batched-prefill layer measured on the M4 Max) */
        {
            const char *re = getenv("RL_ENGINE_RESIDENCY");
            if (!re || atoi(re) != 0) {
                if (@available(macOS 15.0, *)) {
                    MTLResidencySetDescriptor *rd = [[MTLResidencySetDescriptor alloc] init];
                    rd.label = @"redlite_engine";
                    rd.initialCapacity = m->keep.count + 64u;
                    NSError *rerr = nil;
                    id<MTLResidencySet> rs = [m->dev newResidencySetWithDescriptor:rd error:&rerr];
                    if (rs) {
                        for (id<MTLBuffer> b in m->keep) [rs addAllocation:b];
                        [rs commit];
                        [m->queue addResidencySet:rs];
                        m->engine_rs = rs;
                    }
                }
            }
        }
        /* dev21/dev23: GPU-routed decode (residency table, per-layer plans, early-out) */
        const char *spec_env = getenv("RL_ENGINE_SPECULATIVE");
        m->spec_enabled = (!spec_env || atoi(spec_env) != 0) && !m->profile && in->n_expert <= 512u && in->top_k <= 64u;
        m->last_token_missed = 1;
        if (m->spec_enabled) {
            char serr[256];
            if (!rl_native_metal_residency_enable(m->experts, in->n_layer, in->n_expert, serr, sizeof(serr))) m->spec_enabled = 0;
        }
        if (m->spec_enabled) {
            m->plan_slots = new_buf(m, (size_t)in->n_layer * 512u * sizeof(uint64_t));
            m->plan_weights = new_buf(m, (size_t)in->n_layer * 512u * sizeof(float));
            m->plan_ids = new_buf(m, (size_t)in->n_layer * 64u * sizeof(uint32_t));
            m->plan_miss = new_buf(m, (size_t)in->n_layer * sizeof(uint32_t));
            m->layer_out_gpu = new_buf(m, (size_t)in->n_layer * hb);
            /* dev23: no state backups; the early-out path never runs a layer past its first miss */
            const int ok = m->plan_slots && m->plan_weights && m->plan_ids && m->plan_miss && m->layer_out_gpu;
            if (!ok) m->spec_enabled = 0;
        }
        /* full residency: when the cache can hold every routed expert, load them all now (page cache -> pool)
         * so every token from the first one takes the GPU-routed path (RL_ENGINE_PRELOAD=0 disables) */
        const char *preload_env = getenv("RL_ENGINE_PRELOAD");
        if (m->spec_enabled && (!preload_env || atoi(preload_env) != 0) &&
            rl_native_metal_slot_capacity(m->experts) >= (uint64_t)in->n_layer * in->n_expert) {
            uint32_t *all_ids = (uint32_t *)malloc((size_t)in->n_expert * sizeof(uint32_t));
            float *zero_w = (float *)calloc(in->n_expert, sizeof(float));
            if (all_ids && zero_w) {
                for (uint32_t x = 0; x < in->n_expert; ++x) all_ids[x] = x;
                const double t0 = rl_engine_now_ms();
                int ok = 1;
                for (uint32_t l = 0; ok && l < in->n_layer; ++l) {
                    rl_native_topk_plan plan;
                    memset(&plan, 0, sizeof(plan));
                    char perr[256];
                    ok = rl_native_metal_prepare_topk(m->experts, &e->expert_map, l, all_ids, zero_w, in->n_expert, &plan, perr, sizeof(perr));
                }
                if (ok) {
                    rl_native_topk_plan none; memset(&none, 0, sizeof(none));
                    rl_native_metal_telemetry tel;
                    if (rl_native_metal_release_topk(m->experts, &none, &tel, NULL, 0)) m->misses_seen = tel.cache_misses;
                    m->last_token_missed = 0;
                    m->preloaded = 1;
                    /* the first command buffer after the preload pays the residency of the freshly written slabs
                     * (~3 s for 22 GiB); take it here rather than on the first prompt */
                    id<MTLCommandBuffer> warm = [m->queue commandBuffer];
                    id<MTLBlitCommandEncoder> blit = [warm blitCommandEncoder];
                    [blit fillBuffer:m->scalar range:NSMakeRange(0, 4) value:0];
                    [blit endEncoding];
                    [warm commit];
                    [warm waitUntilCompleted];
                    m->preload_ms = rl_engine_now_ms() - t0;
                }
            }
            free(all_ids); free(zero_w);
        }
    }
    if (error && cap) error[0] = '\0';
    return m;
}

void rl_metal_engine_destroy(rl_metal_engine *m) {
    if (!m) return;
    rl_metal_engine_profile_report(m);
    if (m->experts) rl_native_metal_destroy(m->experts);
    if (m->mtp_experts) rl_native_metal_destroy(m->mtp_experts);
    if (m->pf) rl_metal_prefill_destroy(m->pf);
    if (m->engine_rs) {
        if (@available(macOS 15.0, *)) { [m->queue removeResidencySet:(id<MTLResidencySet>)m->engine_rs]; [(id<MTLResidencySet>)m->engine_rs removeAllAllocations]; }
        m->engine_rs = nil;
    }
    free(m->layers); free(m->conv_state); free(m->rec_state); free(m->kcache); free(m->vcache); free(m->routed_host);
    free(m->conv_park); free(m->rec_park); free(m->k_park); free(m->v_park);
    free(m->snap_conv); free(m->snap_rec);
    m->p_route = nil; m->plan_slots = m->plan_weights = m->plan_ids = m->plan_miss = m->layer_out_gpu = nil;
    m->x = m->normed = m->branch = m->resid = m->ffn_in = m->qkv = m->z = m->ba = m->beta = m->gate = m->conv_silu = nil;
    m->q = m->k = m->delta = m->core = m->ng = m->next_conv = m->rec_scratch = m->qgate_raw = m->k_raw = m->value = nil;
    m->query = m->agate = m->key = m->query_rope = m->gated = m->router_logits = m->sh_gate = m->sh_up = m->sh_act = nil;
    m->sh_out = m->scalar = m->routed = m->final_norm = m->logits = m->grid = nil;
    m->p_moe_tail = nil;
    m->steer = nil; m->p_steer = nil;
    m->p_rms = m->p_resid_rms = m->p_scale_add = m->p_rows_f32 = m->p_rows_q8 = m->p_rows_q4k = m->p_rows_q5k = m->p_rows_q6k = m->p_rows_iq2xxs = m->p_rows_iq3 = nil;
    m->p_dn_ba = m->p_dn_conv = m->p_dn_l2 = m->p_dn_shift = m->p_dn_state = m->p_dn_tail = nil;
    m->p_attn_prep = m->p_attn_gqa = m->p_sh_scalar = m->p_sh_silu = nil;
    m->p_attn_split = m->p_attn_merge = m->p_attn_split_g = nil; m->attn_ml = m->attn_acc = nil;
    m->keep = nil; m->dev = nil; m->queue = nil; m->lib = nil;
    free(m);
}

int rl_metal_engine_reset(rl_metal_engine *m, char *error, size_t cap) {
    (void)error; (void)cap;
    if (!m) return 0;
    /* zero states (host-visible shared buffers) */
    for (uint32_t r = 0; r < m->n_recurrent; ++r) {
        memset(m->conv_state[r].contents, 0, m->conv_state[r].length);
        memset(m->rec_state[r].contents, 0, m->rec_state[r].length);
    }
    for (uint32_t a = 0; a < m->n_attention + (m->mtp ? 1u : 0u); ++a) {
        memset(m->kcache[a].contents, 0, m->kcache[a].length);
        memset(m->vcache[a].contents, 0, m->vcache[a].length);
    }
    return 1;
}

/* dev43: raw state I/O on the shared buffers (the engine is idle between calls: every step waits for its GPU work) */
int rl_metal_engine_state_io(rl_metal_engine *m, FILE *f, size_t kv_bytes, int save) {
    if (!m || !f) return 0;
#define RL_IO(buf, n) (save ? fwrite((buf).contents, 1, (n), f) == (n) : fread((buf).contents, 1, (n), f) == (n))
    for (uint32_t r = 0; r < m->n_recurrent; ++r)
        if (!RL_IO(m->conv_state[r], m->conv_state[r].length) || !RL_IO(m->rec_state[r], m->rec_state[r].length)) return 0;
    for (uint32_t a = 0; a < m->n_attention; ++a) {
        if (kv_bytes > m->kcache[a].length) return 0;
        if (!RL_IO(m->kcache[a], kv_bytes) || !RL_IO(m->vcache[a], kv_bytes)) return 0;
    }
#undef RL_IO
    return 1;
}

uint64_t rl_metal_engine_resident_bytes(const rl_metal_engine *m) { return m ? m->resident_bytes : 0u; }

int rl_metal_engine_preloaded(const rl_metal_engine *m, double *preload_ms) {
    if (preload_ms) *preload_ms = m ? m->preload_ms : 0.0;
    return m ? m->preloaded : 0;
}

/* dev22: one compute encoder carries a whole token (GPU-routed path) or a whole layer (synchronous path).
 * The encoder is serial, so each dispatch sees the previous one's writes exactly as with one encoder per
 * dispatch. Under RL_ENGINE_PROFILE the encoder is closed and the command buffer committed at every stage
 * boundary, so the per-stage profile keeps its meaning. */
typedef struct {
    rl_metal_engine *m;
    __strong id<MTLCommandBuffer> cb;
    __strong id<MTLComputeCommandEncoder> enc;
    __unsafe_unretained id<MTLBuffer> abort;   /* dev23 early-out flag bound at index 30 (m->abort_zero: never set) */
} emitter;

static id<MTLComputeCommandEncoder> em_enc(emitter *em) {
    if (!em->enc) {
        em->enc = em->m->concurrent ? [em->cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent] : [em->cb computeCommandEncoder];
        g_rl_group = 0;   /* a group never spans encoders (an error return may have left one open) */
        [em->enc setBuffer:em->abort offset:0 atIndex:30];
    }
    return em->enc;
}

/* dev38: open (1) / close (0) a group of mutually independent dispatches; closing issues the barrier they skipped */
static void em_group(emitter *em, int open) {
    g_rl_group = open;
    if (!open && em->enc && em->enc.dispatchType == MTLDispatchTypeConcurrent) [em->enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

static void em_close(emitter *em) {
    if (em->enc) { [em->enc endEncoding]; em->enc = nil; }
}

static void em_stage(emitter *em, int stage) {
    if (!em->m->profile) return;
    em_close(em);
    em->cb = stage_end(em->m, em->cb, stage);
}

/* dst[doff..] = src[soff..] for count floats, inside the open encoder (replaces a blit) */
static void em_copy(emitter *em, id<MTLBuffer> src, NSUInteger soff, id<MTLBuffer> dst, NSUInteger doff, uint32_t count) {
    id<MTLComputeCommandEncoder> enc = em_enc(em);
    [enc setComputePipelineState:em->m->p_copy];
    [enc setBuffer:src offset:soff atIndex:0]; [enc setBuffer:dst offset:doff atIndex:1]; [enc setBytes:&count length:4 atIndex:2];
    enc_1d(enc, em->m->p_copy, count, 256u);
}

/* dev45: the per-row DeltaNet kernels of a layer (between the input and output projections) */
static void emit_recurrent_mid(rl_engine *e, emitter *em, const rl_layer_tensors *t, const mlayer *w) {
    rl_metal_engine *m = em->m;
    const rl_engine_info *in = &e->info;
    const uint32_t r = t->recurrent_index;
    const uint32_t channels = in->channels, dconv = in->d_conv, rank = in->dt_rank, groups = in->n_group, S = in->d_state;
    const uint32_t qk_each = S * groups, head_v = in->head_v;
    const uint32_t kv_ratio = rank / groups;
    const float eps = in->rms_eps;
    em_stage(em, 1);
    id<MTLComputeCommandEncoder> enc = em_enc(em);
    em_group(em, 1);   /* dn_ba (ba -> beta/gate), dn_conv (qkv, conv_state -> conv_silu) */
    [enc setComputePipelineState:m->p_dn_ba];
    [enc setBuffer:m->ba offset:0 atIndex:0]; [enc setBuffer:w->dt.buf offset:w->dt.off atIndex:1]; [enc setBuffer:w->a.buf offset:w->a.off atIndex:2];
    [enc setBuffer:m->beta offset:0 atIndex:3]; [enc setBuffer:m->gate offset:0 atIndex:4];
    [enc setBytes:&rank length:4 atIndex:5]; [enc setBytes:&groups length:4 atIndex:6];
    enc_1d(enc, m->p_dn_ba, rank, 64u);

    [enc setComputePipelineState:m->p_dn_conv];
    [enc setBuffer:m->conv_state[r] offset:0 atIndex:0]; [enc setBuffer:m->qkv offset:0 atIndex:1]; [enc setBuffer:w->conv.buf offset:w->conv.off atIndex:2];
    [enc setBuffer:m->conv_silu offset:0 atIndex:3]; [enc setBytes:&channels length:4 atIndex:4]; [enc setBytes:&dconv length:4 atIndex:5];
    enc_1d(enc, m->p_dn_conv, channels, 64u);
    em_group(em, 0);
    em_group(em, 1);   /* dn_l2 (conv_silu -> q/k), dn_shift (conv_state, qkv -> next_conv) */

    [enc setComputePipelineState:m->p_dn_l2];
    [enc setBuffer:m->conv_silu offset:0 atIndex:0]; [enc setBuffer:m->q offset:0 atIndex:1]; [enc setBuffer:m->k offset:0 atIndex:2];
    [enc setBytes:&S length:4 atIndex:3]; [enc setBytes:&groups length:4 atIndex:4]; [enc setBytes:&eps length:4 atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake(2u * groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(S, 1, 1)]; rl_after_dispatch(enc);

    [enc setComputePipelineState:m->p_dn_shift];
    [enc setBuffer:m->conv_state[r] offset:0 atIndex:0]; [enc setBuffer:m->qkv offset:0 atIndex:1]; [enc setBuffer:m->next_conv offset:0 atIndex:2];
    [enc setBytes:&channels length:4 atIndex:3]; [enc setBytes:&dconv length:4 atIndex:4];
    enc_1d(enc, m->p_dn_shift, channels, 64u);
    em_group(em, 0);
    em_stage(em, 2);

    const NSUInteger v_offset = (NSUInteger)(2u * qk_each) * sizeof(float);
    enc = em_enc(em);
    [enc setComputePipelineState:m->p_dn_state];
    [enc setBuffer:m->rec_state[r] offset:0 atIndex:0]; [enc setBuffer:m->q offset:0 atIndex:1]; [enc setBuffer:m->k offset:0 atIndex:2];
    [enc setBuffer:m->conv_silu offset:v_offset atIndex:3]; [enc setBuffer:m->gate offset:0 atIndex:4]; [enc setBuffer:m->beta offset:0 atIndex:5];
    [enc setBuffer:m->core offset:0 atIndex:6]; [enc setBytes:&S length:4 atIndex:7]; [enc setBytes:&kv_ratio length:4 atIndex:8]; [enc setBytes:&rank length:4 atIndex:9];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)rank * S, 1, 1) threadsPerThreadgroup:MTLSizeMake(S, 1, 1)]; rl_after_dispatch(enc);
    em_stage(em, 3);

    enc = em_enc(em);
    [enc setComputePipelineState:m->p_dn_tail];
    [enc setBuffer:m->core offset:0 atIndex:0]; [enc setBuffer:m->z offset:0 atIndex:1]; [enc setBuffer:w->ssm_norm.buf offset:w->ssm_norm.off atIndex:2];
    [enc setBuffer:m->ng offset:0 atIndex:3]; [enc setBytes:&eps length:4 atIndex:4]; [enc setBytes:&head_v length:4 atIndex:5]; [enc setBytes:&rank length:4 atIndex:6];
    [enc dispatchThreadgroups:MTLSizeMake(rank, 1, 1) threadsPerThreadgroup:MTLSizeMake(head_v, 1, 1)]; rl_after_dispatch(enc);

}

static void emit_recurrent(rl_engine *e, emitter *em, const rl_layer_tensors *t, const mlayer *w) {
    rl_metal_engine *m = em->m;
    const uint32_t r = t->recurrent_index;
    em_group(em, 1);
    emit_rows(m, em_enc(em), &w->qkv, m->normed, m->qkv);
    emit_rows(m, em_enc(em), &w->z, m->normed, m->z);
    emit_rows(m, em_enc(em), &w->ba, m->normed, m->ba);
    em_group(em, 0);
    emit_recurrent_mid(e, em, t, w);
    emit_rows(m, em_enc(em), &w->ssm_out, m->ng, m->branch);
    em_copy(em, m->next_conv, 0, m->conv_state[r], 0, (uint32_t)rl_engine_conv_count(e));
    em_stage(em, 4);
}

/* dev45: the per-row attention kernels (norm/rope/KV append at `position`, then attention over position + 1 keys) */
static void emit_attention_mid(rl_engine *e, emitter *em, const rl_layer_tensors *t, const mlayer *w, uint32_t position) {
    rl_metal_engine *m = em->m;
    const rl_engine_info *in = &e->info;
    const uint32_t a = t->attention_index;
    const uint32_t head_dim = in->head_dim, qheads = in->n_head, kvheads = in->n_head_kv, rope_dims = in->rope_dims;
    const uint32_t seq_len = position + 1u;
    const float eps = in->rms_eps, base = in->rope_freq_base;
    em_stage(em, 5);
    id<MTLComputeCommandEncoder> enc = em_enc(em);
    [enc setComputePipelineState:m->p_attn_prep];
    [enc setBuffer:m->qgate_raw offset:0 atIndex:0]; [enc setBuffer:m->k_raw offset:0 atIndex:1]; [enc setBuffer:m->value offset:0 atIndex:2];
    [enc setBuffer:w->q_norm.buf offset:w->q_norm.off atIndex:3]; [enc setBuffer:w->k_norm.buf offset:w->k_norm.off atIndex:4];
    [enc setBuffer:m->query_rope offset:0 atIndex:5]; [enc setBuffer:m->agate offset:0 atIndex:6];
    [enc setBuffer:m->kcache[a] offset:0 atIndex:7]; [enc setBuffer:m->vcache[a] offset:0 atIndex:8];
    [enc setBytes:&head_dim length:4 atIndex:9]; [enc setBytes:&qheads length:4 atIndex:10]; [enc setBytes:&kvheads length:4 atIndex:11];
    [enc setBytes:&position length:4 atIndex:12]; [enc setBytes:&rope_dims length:4 atIndex:13]; [enc setBytes:&base length:4 atIndex:14]; [enc setBytes:&eps length:4 atIndex:15];
    [enc dispatchThreadgroups:MTLSizeMake(qheads + kvheads, 1, 1) threadsPerThreadgroup:MTLSizeMake(head_dim, 1, 1)]; rl_after_dispatch(enc);
    em_stage(em, 6);

    enc = em_enc(em);
    if (m->attn_split && seq_len > 256u) {
        /* dev26: one threadgroup per (head, 256-position block), then a per-head merge. Up to one block the
         * single-threadgroup kernel has the same parallelism without the merge dispatch, and measured faster. */
        /* dev35: grouped kernel (one threadgroup per KV head and block, blocks of m->attn_blk positions) */
        const int grouped = m->attn_group && m->p_attn_split_g && head_dim == 256u && qheads % kvheads == 0u && qheads / kvheads <= 8u;
        const uint32_t blk = grouped ? m->attn_blk : 256u;
        const uint32_t nblocks = (seq_len + blk - 1u) / blk;
        [enc setComputePipelineState:grouped ? m->p_attn_split_g : m->p_attn_split];
        [enc setBuffer:m->query_rope offset:0 atIndex:0]; [enc setBuffer:m->kcache[a] offset:0 atIndex:1]; [enc setBuffer:m->vcache[a] offset:0 atIndex:2];
        [enc setBuffer:m->attn_ml offset:0 atIndex:3]; [enc setBuffer:m->attn_acc offset:0 atIndex:4];
        [enc setBytes:&head_dim length:4 atIndex:5]; [enc setBytes:&qheads length:4 atIndex:6]; [enc setBytes:&kvheads length:4 atIndex:7]; [enc setBytes:&seq_len length:4 atIndex:8];
        [enc setBytes:&blk length:4 atIndex:9];
        [enc dispatchThreadgroups:MTLSizeMake(nblocks, grouped ? kvheads : qheads, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)]; rl_after_dispatch(enc);
        [enc setComputePipelineState:m->p_attn_merge];
        [enc setBuffer:m->attn_ml offset:0 atIndex:0]; [enc setBuffer:m->attn_acc offset:0 atIndex:1];
        [enc setBuffer:m->agate offset:0 atIndex:2]; [enc setBuffer:m->gated offset:0 atIndex:3];
        [enc setBytes:&head_dim length:4 atIndex:4]; [enc setBytes:&seq_len length:4 atIndex:5]; [enc setBytes:&blk length:4 atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake(qheads, 1, 1) threadsPerThreadgroup:MTLSizeMake(head_dim, 1, 1)]; rl_after_dispatch(enc);
    } else {
        [enc setComputePipelineState:m->p_attn_gqa];
        [enc setBuffer:m->query_rope offset:0 atIndex:0]; [enc setBuffer:m->kcache[a] offset:0 atIndex:1]; [enc setBuffer:m->vcache[a] offset:0 atIndex:2];
        [enc setBuffer:m->agate offset:0 atIndex:3]; [enc setBuffer:m->gated offset:0 atIndex:4];
        [enc setBytes:&head_dim length:4 atIndex:5]; [enc setBytes:&qheads length:4 atIndex:6]; [enc setBytes:&kvheads length:4 atIndex:7]; [enc setBytes:&seq_len length:4 atIndex:8];
        [enc dispatchThreadgroups:MTLSizeMake(qheads, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)]; rl_after_dispatch(enc);
    }
    em_stage(em, 7);
}

static void emit_attention(rl_engine *e, emitter *em, const rl_layer_tensors *t, const mlayer *w, uint32_t position) {
    rl_metal_engine *m = em->m;
    em_group(em, 1);
    emit_rows(m, em_enc(em), &w->q, m->normed, m->qgate_raw);
    emit_rows(m, em_enc(em), &w->k, m->normed, m->k_raw);
    emit_rows(m, em_enc(em), &w->v, m->normed, m->value);
    em_group(em, 0);
    emit_attention_mid(e, em, t, w, position);
    emit_rows(m, em_enc(em), &w->o, m->gated, m->branch);
    em_stage(em, 8);
}

/* post-attention residual + RMSNorm, router logits, shared expert (gate scalar, gate/up, SiLU*up, down) */
static void emit_ffn_pre(rl_engine *e, emitter *em, const mlayer *w) {
    rl_metal_engine *m = em->m;
    const uint32_t hidden = e->info.hidden;
    const float eps = e->info.rms_eps;
    id<MTLComputeCommandEncoder> enc = em_enc(em);
    [enc setComputePipelineState:m->p_resid_rms];
    [enc setBuffer:m->x offset:0 atIndex:0]; [enc setBuffer:m->branch offset:0 atIndex:1];
    [enc setBuffer:w->post_norm.buf offset:w->post_norm.off atIndex:2];
    [enc setBuffer:m->resid offset:0 atIndex:3]; [enc setBuffer:m->ffn_in offset:0 atIndex:4];
    [enc setBytes:&hidden length:4 atIndex:5]; [enc setBytes:&eps length:4 atIndex:6];
    [enc dispatchThreads:MTLSizeMake(256, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)]; rl_after_dispatch(enc);
    em_stage(em, 9);
    em_group(em, 1);   /* router, shared scalar, shared gate, shared up: all read ffn_in only */
    emit_rows(m, em_enc(em), &w->router, m->ffn_in, m->router_logits);
    em_stage(em, 10);
    enc = em_enc(em);
    [enc setComputePipelineState:m->p_sh_scalar];
    [enc setBuffer:w->sh_gate_inp.buf offset:w->sh_gate_inp.off atIndex:0]; [enc setBuffer:m->ffn_in offset:0 atIndex:1];
    [enc setBuffer:m->scalar offset:0 atIndex:2]; [enc setBytes:&hidden length:4 atIndex:3];
    [enc dispatchThreads:MTLSizeMake(256, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)]; rl_after_dispatch(enc);
    emit_rows(m, enc, &w->sh_gate, m->ffn_in, m->sh_gate);
    emit_rows(m, enc, &w->sh_up, m->ffn_in, m->sh_up);
    em_group(em, 0);
    {
        const uint32_t ffn = w->sh_gate.rows;
        [enc setComputePipelineState:m->p_sh_silu];
        [enc setBuffer:m->sh_gate offset:0 atIndex:0]; [enc setBuffer:m->sh_up offset:0 atIndex:1]; [enc setBuffer:m->sh_act offset:0 atIndex:2];
        [enc setBytes:&ffn length:4 atIndex:3]; enc_1d(enc, m->p_sh_silu, ffn, 64u);
    }
    emit_rows(m, enc, &w->sh_down, m->sh_act, m->sh_out);
    em_stage(em, 11);
}

/* dev52: x += strength * steering vector at the input of layer l, when that layer is steered */
static void emit_steer(rl_engine *e, emitter *em, uint32_t l, id<MTLBuffer> x) {
    rl_metal_engine *m = em->m;
    if (!rl_engine_steers(e, l) || !m->steer) return;
    const uint32_t hidden = e->info.hidden; const float s = e->steer_strength;
    id<MTLComputeCommandEncoder> enc = em_enc(em);
    [enc setComputePipelineState:m->p_steer];
    [enc setBuffer:x offset:0 atIndex:0]; [enc setBuffer:m->steer offset:0 atIndex:1];
    [enc setBytes:&s length:4 atIndex:2]; [enc setBytes:&hidden length:4 atIndex:3];
    enc_1d(enc, m->p_steer, hidden, 256u);
}

int rl_metal_engine_set_steering(rl_metal_engine *m, const float *vector, uint32_t hidden, char *error, size_t cap) {
    id<MTLBuffer> b = [m->dev newBufferWithBytes:vector length:(NSUInteger)hidden * sizeof(float) options:MTLResourceStorageModeShared];
    if (!b) { snprintf(error, cap, "steering buffer allocation failed"); return 0; }
    m->steer = b;
    return 1;
}

static void emit_scale_add(rl_engine *e, emitter *em) {
    rl_metal_engine *m = em->m;
    const uint32_t hidden = e->info.hidden;
    id<MTLComputeCommandEncoder> enc = em_enc(em);
    [enc setComputePipelineState:m->p_scale_add];
    [enc setBuffer:m->resid offset:0 atIndex:0]; [enc setBuffer:m->routed offset:0 atIndex:1]; [enc setBuffer:m->sh_out offset:0 atIndex:2];
    [enc setBuffer:m->scalar offset:0 atIndex:3]; [enc setBuffer:m->x offset:0 atIndex:4]; [enc setBytes:&hidden length:4 atIndex:5];
    enc_1d(enc, m->p_scale_add, hidden, 64u);
}

/* GPU-routed layers first..n_layer-1 plus the output head: router selection and expert lookup on the GPU
 * (dev21); with the early-out flag bound, everything after the first layer that selects a non-resident
 * expert returns immediately (dev23). */
/* GPU routing, routed experts and the end of layer l for the current row buffers (the layer's FFN input is ready) */
static int emit_layer_experts(rl_engine *e, emitter *em, uint32_t l, id<MTLBuffer> resident, char *error, size_t cap) {
    rl_metal_engine *m = em->m;
    const rl_engine_info *in = &e->info;
    const uint32_t hidden = in->hidden, experts = in->n_expert, topk = in->top_k;
    const size_t hb = (size_t)hidden * sizeof(float);
    const int fused = m->fuse_tail && m->p_moe_tail;
    const NSUInteger slots_off = (NSUInteger)l * 512u * sizeof(uint64_t), weights_off = (NSUInteger)l * 512u * sizeof(float);
    const NSUInteger ids_off = (NSUInteger)l * 64u * sizeof(uint32_t), miss_off = (NSUInteger)l * sizeof(uint32_t);
    {
        id<MTLComputeCommandEncoder> enc = em_enc(em);
        [enc setComputePipelineState:m->p_route];
        [enc setBuffer:m->router_logits offset:0 atIndex:0];
        [enc setBuffer:resident offset:(NSUInteger)l * experts * sizeof(uint64_t) atIndex:1];
        [enc setBuffer:m->plan_slots offset:slots_off atIndex:2];
        [enc setBuffer:m->plan_weights offset:weights_off atIndex:3];
        [enc setBuffer:m->plan_ids offset:ids_off atIndex:4];
        [enc setBuffer:m->plan_miss offset:miss_off atIndex:5];
        [enc setBytes:&experts length:4 atIndex:6]; [enc setBytes:&topk length:4 atIndex:7]; [enc setBytes:&m->route_bias length:4 atIndex:8];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)]; rl_after_dispatch(enc);
    }
    rl_native_layer_info li;
    rl_expert_layout lay;
    if (!rl_native_get_layer_info(&e->expert_map, l, &li, error, cap) ||
        !rl_native_expert_layout(&e->expert_map, l, 0u, &lay, error, cap)) return 0;
    if (!redmetal_topk_pool_encode_device_into(rl_native_metal_pool_handle(m->experts), (__bridge void *)em_enc(em),
            (__bridge void *)m->plan_slots, slots_off, (__bridge void *)m->plan_weights, weights_off, topk, rl_native_expert_type_word(li.ggml_type, li.down_type),
            li.hidden_size, li.ffn_size, lay.gate_bytes, lay.up_bytes,
            (__bridge void *)m->ffn_in, 0u, fused ? NULL : (__bridge void *)m->routed, 0u)) {
        snprintf(error, cap, "GPU-routed expert encode failed: %s", redmetal_topk_last_error()); return 0;
    }
    if (fused) {
        id<MTLComputeCommandEncoder> enc = em_enc(em);
        [enc setComputePipelineState:m->p_moe_tail];
        [enc setBuffer:(__bridge id<MTLBuffer>)redmetal_topk_pool_tmp_buffer(rl_native_metal_pool_handle(m->experts)) offset:0 atIndex:0];
        [enc setBuffer:m->plan_weights offset:weights_off atIndex:1];
        [enc setBuffer:m->resid offset:0 atIndex:2]; [enc setBuffer:m->sh_out offset:0 atIndex:3]; [enc setBuffer:m->scalar offset:0 atIndex:4];
        [enc setBuffer:m->x offset:0 atIndex:5]; [enc setBuffer:m->layer_out_gpu offset:(NSUInteger)l * hb atIndex:6];
        [enc setBytes:&hidden length:4 atIndex:7]; [enc setBytes:&topk length:4 atIndex:8];
        enc_1d(enc, m->p_moe_tail, hidden, 64u);
        return 1;
    }
    emit_scale_add(e, em);
    em_copy(em, m->x, 0, m->layer_out_gpu, (NSUInteger)l * hb, hidden);
    return 1;
}

static int emit_routed_layers(rl_engine *e, emitter *em, uint32_t first, id<MTLBuffer> resident, uint32_t position,
                              int want_logits, char *error, size_t cap) {
    rl_metal_engine *m = em->m;
    const rl_engine_info *in = &e->info;
    const uint32_t hidden = in->hidden;
    const float eps = in->rms_eps;
    for (uint32_t l = first; l < in->n_layer; ++l) {
        const rl_layer_tensors *t = &e->layers[l];
        const mlayer *w = &m->layers[l];
        emit_steer(e, em, l, m->x);
        emit_rms(m, em_enc(em), m->x, &w->attn_norm, m->normed, hidden, eps);
        if (t->kind == RL_LAYER_MAP_RECURRENT) emit_recurrent(e, em, t, w);
        else emit_attention(e, em, t, w, position);
        emit_ffn_pre(e, em, w);
        if (!emit_layer_experts(e, em, l, resident, error, cap)) return 0;
    }
    emit_rms(m, em_enc(em), m->x, &m->output_norm, m->final_norm, hidden, eps);
    if (want_logits) emit_rows(m, em_enc(em), &m->output, m->final_norm, m->logits);
    return 1;
}

/* ---- dev45: 2-row verify (token t at position p and draft d at p + 1 in one pass) ----
 * Every per-token buffer has a row-1 twin in m->alt; swap_rows exchanges them so the 1-row emit helpers serve either
 * row. Dense products use emit_rows_r2 (weights decoded once for both rows); per-row kernels run row 0 then row 1
 * in each layer. The DeltaNet conv/recurrent states after row 0 are copied to snapshots: a rejected draft is undone
 * by swapping the state pointers with them. */
#define RL_ROW_FIELDS(X) X(x) X(normed) X(branch) X(resid) X(ffn_in) X(qkv) X(z) X(ba) X(beta) X(gate) X(conv_silu) X(q) X(k) \
    X(delta) X(core) X(ng) X(next_conv) X(rec_scratch) X(qgate_raw) X(k_raw) X(value) X(query) X(agate) X(key) X(query_rope) \
    X(gated) X(router_logits) X(sh_gate) X(sh_up) X(sh_act) X(sh_out) X(scalar) X(routed) X(final_norm) X(logits)
#define RL_ROW_ENUM(f) RB_##f,
enum { RL_ROW_FIELDS(RL_ROW_ENUM) RB_COUNT };
#define ALT(f) (m->alt[RB_##f])

static void swap_rows(rl_metal_engine *m) {
#define RL_ROW_SWAP(f) { id<MTLBuffer> t_ = m->f; m->f = m->alt[RB_##f]; m->alt[RB_##f] = t_; }
    RL_ROW_FIELDS(RL_ROW_SWAP)
#undef RL_ROW_SWAP
}

static int verify_alloc(rl_engine *e, rl_metal_engine *m) {
    if (m->verify_ready) return 1;
    _Static_assert(RB_COUNT <= 40, "alt[] too small");
#define RL_ROW_ALLOC(f) { if (!m->f) return 0; m->alt[RB_##f] = new_buf(m, m->f.length); if (!m->alt[RB_##f]) return 0; }
    RL_ROW_FIELDS(RL_ROW_ALLOC)
#undef RL_ROW_ALLOC
    m->snap_conv = (__unsafe_unretained id<MTLBuffer> *)calloc(m->n_recurrent ? m->n_recurrent : 1u, sizeof(id));
    m->snap_rec = (__unsafe_unretained id<MTLBuffer> *)calloc(m->n_recurrent ? m->n_recurrent : 1u, sizeof(id));
    if (!m->snap_conv || !m->snap_rec) return 0;
    for (uint32_t r = 0; r < m->n_recurrent; ++r) {
        m->snap_conv[r] = new_buf(m, m->conv_state[r].length);
        m->snap_rec[r] = new_buf(m, m->rec_state[r].length);
        if (!m->snap_conv[r] || !m->snap_rec[r]) return 0;
    }
    (void)e;
    m->verify_miss = new_buf(m, 16u);
    if (!m->verify_miss) return 0;
    m->verify_ready = 1;
    return 1;
}

static void emit_resid_rms(rl_metal_engine *m, emitter *em, const mlayer *w, uint32_t hidden, float eps) {
    id<MTLComputeCommandEncoder> enc = em_enc(em);
    [enc setComputePipelineState:m->p_resid_rms];
    [enc setBuffer:m->x offset:0 atIndex:0]; [enc setBuffer:m->branch offset:0 atIndex:1];
    [enc setBuffer:w->post_norm.buf offset:w->post_norm.off atIndex:2];
    [enc setBuffer:m->resid offset:0 atIndex:3]; [enc setBuffer:m->ffn_in offset:0 atIndex:4];
    [enc setBytes:&hidden length:4 atIndex:5]; [enc setBytes:&eps length:4 atIndex:6];
    [enc dispatchThreads:MTLSizeMake(256, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)]; rl_after_dispatch(enc);
}

static void emit_sh_scalar(rl_metal_engine *m, emitter *em, const mlayer *w, uint32_t hidden) {
    id<MTLComputeCommandEncoder> enc = em_enc(em);
    [enc setComputePipelineState:m->p_sh_scalar];
    [enc setBuffer:w->sh_gate_inp.buf offset:w->sh_gate_inp.off atIndex:0]; [enc setBuffer:m->ffn_in offset:0 atIndex:1];
    [enc setBuffer:m->scalar offset:0 atIndex:2]; [enc setBytes:&hidden length:4 atIndex:3];
    [enc dispatchThreads:MTLSizeMake(256, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)]; rl_after_dispatch(enc);
}

static void emit_sh_silu(rl_metal_engine *m, emitter *em, const mlayer *w) {
    id<MTLComputeCommandEncoder> enc = em_enc(em);
    const uint32_t ffn = w->sh_gate.rows;
    [enc setComputePipelineState:m->p_sh_silu];
    [enc setBuffer:m->sh_gate offset:0 atIndex:0]; [enc setBuffer:m->sh_up offset:0 atIndex:1]; [enc setBuffer:m->sh_act offset:0 atIndex:2];
    [enc setBytes:&ffn length:4 atIndex:3]; enc_1d(enc, m->p_sh_silu, ffn, 64u);
}

/* the DeltaNet kernels of one layer for both verify rows: 5 dispatches; conv_state/rec_state end after row 1, the
 * snapshots hold the state after row 0 */
static void emit_recurrent_mid2(rl_engine *e, emitter *em, const rl_layer_tensors *t, const mlayer *w) {
    rl_metal_engine *m = em->m;
    const rl_engine_info *in = &e->info;
    const uint32_t r = t->recurrent_index;
    const uint32_t channels = in->channels, dconv = in->d_conv, rank = in->dt_rank, groups = in->n_group, S = in->d_state;
    const uint32_t qk_each = S * groups, head_v = in->head_v;
    const uint32_t kv_ratio = rank / groups;
    const float eps = in->rms_eps;
    id<MTLComputeCommandEncoder> enc = em_enc(em);
    em_group(em, 1);   /* ba (both rows) and conv + shift (both rows) are independent */
    [enc setComputePipelineState:m->p_dn_ba2];
    [enc setBuffer:m->ba offset:0 atIndex:0]; [enc setBuffer:w->dt.buf offset:w->dt.off atIndex:1]; [enc setBuffer:w->a.buf offset:w->a.off atIndex:2];
    [enc setBuffer:m->beta offset:0 atIndex:3]; [enc setBuffer:m->gate offset:0 atIndex:4];
    [enc setBytes:&rank length:4 atIndex:5]; [enc setBytes:&groups length:4 atIndex:6];
    [enc setBuffer:ALT(ba) offset:0 atIndex:7]; [enc setBuffer:ALT(beta) offset:0 atIndex:8]; [enc setBuffer:ALT(gate) offset:0 atIndex:9];
    enc_1d(enc, m->p_dn_ba2, 2u * rank, 64u);
    [enc setComputePipelineState:m->p_dn_convshift2];
    [enc setBuffer:m->conv_state[r] offset:0 atIndex:0]; [enc setBuffer:m->qkv offset:0 atIndex:1]; [enc setBuffer:w->conv.buf offset:w->conv.off atIndex:2];
    [enc setBuffer:m->conv_silu offset:0 atIndex:3]; [enc setBytes:&channels length:4 atIndex:4]; [enc setBytes:&dconv length:4 atIndex:5];
    [enc setBuffer:ALT(qkv) offset:0 atIndex:6]; [enc setBuffer:ALT(conv_silu) offset:0 atIndex:7]; [enc setBuffer:m->snap_conv[r] offset:0 atIndex:8];
    enc_1d(enc, m->p_dn_convshift2, channels, 64u);
    em_group(em, 0);
    [enc setComputePipelineState:m->p_dn_l2_2];
    [enc setBuffer:m->conv_silu offset:0 atIndex:0]; [enc setBuffer:m->q offset:0 atIndex:1]; [enc setBuffer:m->k offset:0 atIndex:2];
    [enc setBytes:&S length:4 atIndex:3]; [enc setBytes:&groups length:4 atIndex:4]; [enc setBytes:&eps length:4 atIndex:5];
    [enc setBuffer:ALT(conv_silu) offset:0 atIndex:6]; [enc setBuffer:ALT(q) offset:0 atIndex:7]; [enc setBuffer:ALT(k) offset:0 atIndex:8];
    [enc dispatchThreadgroups:MTLSizeMake(4u * groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(S, 1, 1)]; rl_after_dispatch(enc);
    const NSUInteger v_offset = (NSUInteger)(2u * qk_each) * sizeof(float);
    [enc setComputePipelineState:m->p_dn_state2];
    [enc setBuffer:m->rec_state[r] offset:0 atIndex:0]; [enc setBuffer:m->q offset:0 atIndex:1]; [enc setBuffer:m->k offset:0 atIndex:2];
    [enc setBuffer:m->conv_silu offset:v_offset atIndex:3]; [enc setBuffer:m->gate offset:0 atIndex:4]; [enc setBuffer:m->beta offset:0 atIndex:5];
    [enc setBuffer:m->core offset:0 atIndex:6]; [enc setBytes:&S length:4 atIndex:7]; [enc setBytes:&kv_ratio length:4 atIndex:8]; [enc setBytes:&rank length:4 atIndex:9];
    [enc setBuffer:ALT(q) offset:0 atIndex:10]; [enc setBuffer:ALT(k) offset:0 atIndex:11]; [enc setBuffer:ALT(conv_silu) offset:v_offset atIndex:12];
    [enc setBuffer:ALT(gate) offset:0 atIndex:13]; [enc setBuffer:ALT(beta) offset:0 atIndex:14]; [enc setBuffer:ALT(core) offset:0 atIndex:15];
    [enc setBuffer:m->snap_rec[r] offset:0 atIndex:16];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)rank * S, 1, 1) threadsPerThreadgroup:MTLSizeMake(S, 1, 1)]; rl_after_dispatch(enc);
    [enc setComputePipelineState:m->p_dn_tail2];
    [enc setBuffer:m->core offset:0 atIndex:0]; [enc setBuffer:m->z offset:0 atIndex:1]; [enc setBuffer:w->ssm_norm.buf offset:w->ssm_norm.off atIndex:2];
    [enc setBuffer:m->ng offset:0 atIndex:3]; [enc setBytes:&eps length:4 atIndex:4]; [enc setBytes:&head_v length:4 atIndex:5]; [enc setBytes:&rank length:4 atIndex:6];
    [enc setBuffer:ALT(core) offset:0 atIndex:7]; [enc setBuffer:ALT(z) offset:0 atIndex:8]; [enc setBuffer:ALT(ng) offset:0 atIndex:9];
    [enc dispatchThreadgroups:MTLSizeMake(2u * rank, 1, 1) threadsPerThreadgroup:MTLSizeMake(head_v, 1, 1)]; rl_after_dispatch(enc);
}

/* routing, experts and layer end for both verify rows: one router dispatch, the 2 x top_k experts in one pool
 * encode (row 1's read its FFN input from x_split = top_k), one tail */
static int emit_layer_experts2(rl_engine *e, emitter *em, uint32_t l, id<MTLBuffer> resident, char *error, size_t cap) {
    rl_metal_engine *m = em->m;
    const rl_engine_info *in = &e->info;
    const uint32_t hidden = in->hidden, experts = in->n_expert, topk = in->top_k, both = 2u * in->top_k;
    const size_t hb = (size_t)hidden * sizeof(float);
    const NSUInteger slots_off = (NSUInteger)l * 512u * sizeof(uint64_t), weights_off = (NSUInteger)l * 512u * sizeof(float);
    const NSUInteger ids_off = (NSUInteger)l * 64u * sizeof(uint32_t);
    id<MTLComputeCommandEncoder> enc = em_enc(em);
    [enc setComputePipelineState:m->p_route2];
    [enc setBuffer:m->router_logits offset:0 atIndex:0];
    [enc setBuffer:resident offset:(NSUInteger)l * experts * sizeof(uint64_t) atIndex:1];
    [enc setBuffer:m->plan_slots offset:slots_off atIndex:2]; [enc setBuffer:m->plan_weights offset:weights_off atIndex:3];
    [enc setBuffer:m->plan_ids offset:ids_off atIndex:4]; [enc setBuffer:m->verify_miss offset:0 atIndex:5];
    [enc setBytes:&experts length:4 atIndex:6]; [enc setBytes:&topk length:4 atIndex:7];
    [enc setBuffer:ALT(router_logits) offset:0 atIndex:9];
    [enc dispatchThreadgroups:MTLSizeMake(2, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)]; rl_after_dispatch(enc);
    rl_native_layer_info li;
    rl_expert_layout lay;
    if (!rl_native_get_layer_info(&e->expert_map, l, &li, error, cap) ||
        !rl_native_expert_layout(&e->expert_map, l, 0u, &lay, error, cap)) return 0;
    if (!redmetal_topk_pool_encode_device_into2(rl_native_metal_pool_handle(m->experts), (__bridge void *)em_enc(em),
            (__bridge void *)m->plan_slots, slots_off, (__bridge void *)m->plan_weights, weights_off, both, rl_native_expert_type_word(li.ggml_type, li.down_type),
            li.hidden_size, li.ffn_size, lay.gate_bytes, lay.up_bytes, (__bridge void *)m->ffn_in, 0u, NULL, 0u,
            (__bridge void *)ALT(ffn_in), 0u, topk)) {
        snprintf(error, cap, "verify expert encode failed: %s", redmetal_topk_last_error()); return 0;
    }
    enc = em_enc(em);
    [enc setComputePipelineState:m->p_moe_tail2];
    [enc setBuffer:(__bridge id<MTLBuffer>)redmetal_topk_pool_tmp_buffer(rl_native_metal_pool_handle(m->experts)) offset:0 atIndex:0];
    [enc setBuffer:m->plan_weights offset:weights_off atIndex:1];
    [enc setBuffer:m->resid offset:0 atIndex:2]; [enc setBuffer:m->sh_out offset:0 atIndex:3]; [enc setBuffer:m->scalar offset:0 atIndex:4];
    [enc setBuffer:m->x offset:0 atIndex:5]; [enc setBuffer:m->layer_out_gpu offset:(NSUInteger)l * hb atIndex:6];
    [enc setBytes:&hidden length:4 atIndex:7]; [enc setBytes:&topk length:4 atIndex:8];
    [enc setBuffer:ALT(resid) offset:0 atIndex:9]; [enc setBuffer:ALT(sh_out) offset:0 atIndex:10]; [enc setBuffer:ALT(scalar) offset:0 atIndex:11];
    [enc setBuffer:ALT(x) offset:0 atIndex:12];
    enc_1d(enc, m->p_moe_tail2, 2u * hidden, 64u);
    return 1;
}

/* dev56: the parked slot's sequence state takes the place of the selected one's (encode time only) */
static void swap_state(rl_metal_engine *m) {
    __unsafe_unretained id<MTLBuffer> *t;
    t = m->conv_state; m->conv_state = m->conv_park; m->conv_park = t;
    t = m->rec_state; m->rec_state = m->rec_park; m->rec_park = t;
    t = m->kcache; m->kcache = m->k_park; m->k_park = t;
    t = m->vcache; m->vcache = m->v_park; m->v_park = t;
}

/* Two rows in one pass. Verify (pair = 0): row 1 is the next position of the same sequence. Pair (dev56): row 1 is
 * the parked slot's sequence at position1, with its own DeltaNet and KV state. */
static int emit_verify2(rl_engine *e, emitter *em, id<MTLBuffer> resident, uint32_t position, uint32_t position1, int pair,
                        char *error, size_t cap) {
    rl_metal_engine *m = em->m;
    const rl_engine_info *in = &e->info;
    const uint32_t hidden = in->hidden;
    const float eps = in->rms_eps;
    const uint32_t conv_n = (uint32_t)rl_engine_conv_count(e), rec_n = (uint32_t)rl_engine_rec_count(e);
    for (uint32_t l = 0; l < in->n_layer; ++l) {
        const rl_layer_tensors *t = &e->layers[l];
        const mlayer *w = &m->layers[l];
        emit_steer(e, em, l, m->x); emit_steer(e, em, l, ALT(x));
        em_group(em, 1);   /* the two rows' independent kernels share one barrier */
        emit_rms(m, em_enc(em), m->x, &w->attn_norm, m->normed, hidden, eps);
        emit_rms(m, em_enc(em), ALT(x), &w->attn_norm, ALT(normed), hidden, eps);
        em_group(em, 0);
        if (t->kind == RL_LAYER_MAP_RECURRENT) {
            const uint32_t r = t->recurrent_index;
            em_group(em, 1);
            emit_rows_r2(m, em_enc(em), &w->qkv, m->normed, ALT(normed), m->qkv, ALT(qkv));
            emit_rows_r2(m, em_enc(em), &w->z, m->normed, ALT(normed), m->z, ALT(z));
            emit_rows_r2(m, em_enc(em), &w->ba, m->normed, ALT(normed), m->ba, ALT(ba));
            em_group(em, 0);
            if (pair) {
                emit_recurrent_mid(e, em, t, w);
                em_copy(em, m->next_conv, 0, m->conv_state[r], 0, conv_n);
                swap_rows(m); swap_state(m);
                emit_recurrent_mid(e, em, t, w);
                em_copy(em, m->next_conv, 0, m->conv_state[r], 0, conv_n);
                swap_state(m); swap_rows(m);
            } else if (e->info.d_conv <= 9u && m->p_dn_state2) {
                emit_recurrent_mid2(e, em, t, w);
            } else {
                emit_recurrent_mid(e, em, t, w);
                em_copy(em, m->next_conv, 0, m->conv_state[r], 0, conv_n);
                em_group(em, 1);   /* snapshot of the state after row 0 */
                em_copy(em, m->rec_state[r], 0, m->snap_rec[r], 0, rec_n);
                em_copy(em, m->conv_state[r], 0, m->snap_conv[r], 0, conv_n);
                em_group(em, 0);
                swap_rows(m);
                emit_recurrent_mid(e, em, t, w);
                em_copy(em, m->next_conv, 0, m->conv_state[r], 0, conv_n);
                swap_rows(m);
            }
            emit_rows_r2(m, em_enc(em), &w->ssm_out, m->ng, ALT(ng), m->branch, ALT(branch));
        } else {
            em_group(em, 1);
            emit_rows_r2(m, em_enc(em), &w->q, m->normed, ALT(normed), m->qgate_raw, ALT(qgate_raw));
            emit_rows_r2(m, em_enc(em), &w->k, m->normed, ALT(normed), m->k_raw, ALT(k_raw));
            emit_rows_r2(m, em_enc(em), &w->v, m->normed, ALT(normed), m->value, ALT(value));
            em_group(em, 0);
            emit_attention_mid(e, em, t, w, position);
            swap_rows(m); if (pair) swap_state(m);
            emit_attention_mid(e, em, t, w, position1);
            if (pair) swap_state(m);
            swap_rows(m);
            emit_rows_r2(m, em_enc(em), &w->o, m->gated, ALT(gated), m->branch, ALT(branch));
        }
        em_group(em, 1);
        emit_resid_rms(m, em, w, hidden, eps);
        swap_rows(m); emit_resid_rms(m, em, w, hidden, eps); swap_rows(m);
        em_group(em, 0);
        em_group(em, 1);   /* router, shared scalar, shared gate, shared up: all read ffn_in only */
        emit_rows_r2(m, em_enc(em), &w->router, m->ffn_in, ALT(ffn_in), m->router_logits, ALT(router_logits));
        emit_sh_scalar(m, em, w, hidden);
        swap_rows(m); emit_sh_scalar(m, em, w, hidden); swap_rows(m);
        emit_rows_r2(m, em_enc(em), &w->sh_gate, m->ffn_in, ALT(ffn_in), m->sh_gate, ALT(sh_gate));
        emit_rows_r2(m, em_enc(em), &w->sh_up, m->ffn_in, ALT(ffn_in), m->sh_up, ALT(sh_up));
        em_group(em, 0);
        em_group(em, 1);
        emit_sh_silu(m, em, w);
        swap_rows(m); emit_sh_silu(m, em, w); swap_rows(m);
        em_group(em, 0);
        emit_rows_r2(m, em_enc(em), &w->sh_down, m->sh_act, ALT(sh_act), m->sh_out, ALT(sh_out));
        if (m->fuse_tail && m->p_moe_tail2 && 2u * in->top_k <= 64u) {
            if (!emit_layer_experts2(e, em, l, resident, error, cap)) return 0;
        } else {
            if (!emit_layer_experts(e, em, l, resident, error, cap)) return 0;
            swap_rows(m);
            const int ok = emit_layer_experts(e, em, l, resident, error, cap);
            swap_rows(m);
            if (!ok) return 0;
        }
    }
    em_group(em, 1);
    emit_rms(m, em_enc(em), m->x, &m->output_norm, m->final_norm, hidden, eps);
    emit_rms(m, em_enc(em), ALT(x), &m->output_norm, ALT(final_norm), hidden, eps);
    em_group(em, 0);
    emit_rows_r2(m, em_enc(em), &m->output, m->final_norm, ALT(final_norm), m->logits, ALT(logits));
    return 1;
}

int rl_metal_engine_verify2(rl_engine *e, rl_metal_engine *m, uint32_t t0, uint32_t t1, float *logits0, float *logits1,
                            char *error, size_t cap) {
    rl_backend_state *s = &e->gpu;
    const rl_engine_info *in = &e->info;
    if (!m->spec_enabled || !m->preloaded) { set_error(error, cap, "verify needs every expert resident (full residency)"); return 0; }
    if (m->verify_pending) { set_error(error, cap, "previous verify not committed"); return 0; }
    if (s->position + 2u > in->context) { set_error(error, cap, "context capacity exhausted"); return 0; }
    if (!verify_alloc(e, m)) { set_error(error, cap, "verify buffer allocation failed"); return 0; }
    id<MTLBuffer> resident = (__bridge id<MTLBuffer>)rl_native_metal_residency_table(m->experts);
    @autoreleasepool {
        if (!rl_engine_embed_token(e, t0, (float *)m->x.contents, error, cap) ||
            !rl_engine_embed_token(e, t1, (float *)ALT(x).contents, error, cap)) return 0;
        memset(m->abort.contents, 0, 16u);
        emitter em = { m, [m->queue commandBuffer], nil, m->abort };
        if (!emit_verify2(e, &em, resident, s->position, s->position + 1u, 0, error, cap)) { em_close(&em); return 0; }
        em_close(&em);
        double gpu_ms = 0.0;
        if (!commit_wait(em.cb, "verify", &gpu_ms, error, cap)) return 0;
        if (((volatile uint32_t *)m->abort.contents)[0]) { set_error(error, cap, "verify hit a non-resident expert"); return 0; }
        if (logits0) memcpy(logits0, m->logits.contents, (size_t)in->vocab * sizeof(float));
        if (logits1) memcpy(logits1, ALT(logits).contents, (size_t)in->vocab * sizeof(float));
    }
    m->verify_pending = 1;
    return 1;
}

int rl_metal_engine_slots_enable(rl_engine *e, rl_metal_engine *m, char *error, size_t cap) {
    if (!m->spec_enabled || !m->preloaded) { set_error(error, cap, "two slots need every expert resident (full residency)"); return 0; }
    if (!verify_alloc(e, m)) { set_error(error, cap, "row buffer allocation failed"); return 0; }
    const uint32_t na = m->n_attention + 1u, nr = m->n_recurrent ? m->n_recurrent : 1u;
    m->conv_park = (__unsafe_unretained id<MTLBuffer> *)calloc(nr, sizeof(id));
    m->rec_park = (__unsafe_unretained id<MTLBuffer> *)calloc(nr, sizeof(id));
    m->k_park = (__unsafe_unretained id<MTLBuffer> *)calloc(na, sizeof(id));
    m->v_park = (__unsafe_unretained id<MTLBuffer> *)calloc(na, sizeof(id));
    if (!m->conv_park || !m->rec_park || !m->k_park || !m->v_park) { set_error(error, cap, "slot table allocation failed"); return 0; }
#define RL_PARK(dst, src) { dst = new_buf(m, (src).length); if (!dst) { set_error(error, cap, "second slot allocation failed"); return 0; } \
                            memset(dst.contents, 0, dst.length); }
    for (uint32_t r = 0; r < m->n_recurrent; ++r) { RL_PARK(m->conv_park[r], m->conv_state[r]); RL_PARK(m->rec_park[r], m->rec_state[r]); }
    for (uint32_t a = 0; a < m->n_attention + (m->mtp ? 1u : 0u); ++a) { RL_PARK(m->k_park[a], m->kcache[a]); RL_PARK(m->v_park[a], m->vcache[a]); }
    RL_PARK(m->final_park, m->final_norm);
#undef RL_PARK
    return 1;
}

void rl_metal_engine_swap_slot(rl_metal_engine *m) {
    swap_state(m);
    id<MTLBuffer> t = m->final_norm; m->final_norm = m->final_park; m->final_park = t;
}

int rl_metal_engine_step_pair(rl_engine *e, rl_metal_engine *m, uint32_t t0, uint32_t t1, float *logits0, float *logits1,
                              char *error, size_t cap) {
    const rl_engine_info *in = &e->info;
    const uint32_t p0 = e->gpu.position, p1 = e->gpu_parked.position;
    if (m->verify_pending) { set_error(error, cap, "previous verify not committed"); return 0; }
    if (p0 + 1u > in->context || p1 + 1u > in->context) { set_error(error, cap, "context capacity exhausted"); return 0; }
    const size_t hb = (size_t)in->hidden * sizeof(float);
    id<MTLBuffer> resident = (__bridge id<MTLBuffer>)rl_native_metal_residency_table(m->experts);
    @autoreleasepool {
        if (!rl_engine_embed_token(e, t0, (float *)m->x.contents, error, cap) ||
            !rl_engine_embed_token(e, t1, (float *)ALT(x).contents, error, cap)) return 0;
        memset(m->abort.contents, 0, 16u);
        emitter em = { m, [m->queue commandBuffer], nil, m->abort };
        if (!emit_verify2(e, &em, resident, p0, p1, 1, error, cap)) { em_close(&em); return 0; }
        em_close(&em);
        double gpu_ms = 0.0;
        if (!commit_wait(em.cb, "pair step", &gpu_ms, error, cap)) return 0;
        if (((volatile uint32_t *)m->abort.contents)[0]) { set_error(error, cap, "pair step hit a non-resident expert"); return 0; }
        if (logits0) memcpy(logits0, m->logits.contents, (size_t)in->vocab * sizeof(float));
        if (logits1) memcpy(logits1, ALT(logits).contents, (size_t)in->vocab * sizeof(float));
        memcpy(m->final_park.contents, ALT(final_norm).contents, hb);   /* row 1's final norm belongs to the parked slot */
        memcpy(e->gpu.final_norm, m->final_norm.contents, hb);
        memcpy(e->gpu_parked.final_norm, ALT(final_norm).contents, hb);
    }
    return 1;
}

/* accepted: both rows stand (position + 2, row 1's final norm becomes the current one); otherwise row 0 only:
 * the DeltaNet states return to the row-0 snapshots by pointer swap (position + 1; row 1's KV rows are overwritten later) */
int rl_metal_engine_verify_commit(rl_engine *e, rl_metal_engine *m, int accepted, char *error, size_t cap) {
    rl_backend_state *s = &e->gpu;
    if (!m->verify_pending) { set_error(error, cap, "no verify to commit"); return 0; }
    m->verify_pending = 0;
    if (accepted) {
        memcpy(m->final_norm.contents, ALT(final_norm).contents, (size_t)e->info.hidden * sizeof(float));
        s->position += 2u;
        return 1;
    }
    for (uint32_t r = 0; r < m->n_recurrent; ++r) {
        __unsafe_unretained id<MTLBuffer> t = m->rec_state[r]; m->rec_state[r] = m->snap_rec[r]; m->snap_rec[r] = t;
        t = m->conv_state[r]; m->conv_state[r] = m->snap_conv[r]; m->snap_conv[r] = t;
    }
    s->position += 1u;
    return 1;
}

/* dev45: one MTP block pass. Input: the trunk's hidden state of the last step (m->x before its final norm, or
 * m->final_norm with RL_MTP_H=post) and the next token's embedding; output: argmax of the MTP head. */
int rl_metal_engine_mtp_draft(rl_engine *e, rl_metal_engine *m, const float *embedding, uint32_t position, uint32_t *draft,
                              float *logits, char *error, size_t cap) {
    if (!m || !m->mtp) { set_error(error, cap, "MTP block not loaded"); return 0; }
    const rl_engine_info *in = &e->info;
    const uint32_t hidden = in->hidden, experts = in->n_expert, topk = in->top_k;
    const float eps = in->rms_eps, no_bias = 0.0f;
    memcpy(m->mtp_emb.contents, embedding, (size_t)hidden * sizeof(float));
    memset(m->abort.contents, 0, 16u);
    const char *hsel = getenv("RL_MTP_H");
    id<MTLBuffer> hsrc = hsel && strcmp(hsel, "pre") == 0 ? m->x : m->final_norm;   /* post-norm: 0.742 vs 0.672 acceptance (dev45) */
    emitter em = {m, [m->queue commandBuffer], nil, m->abort};
    emit_rms(m, em_enc(&em), m->mtp_emb, &m->mtp_enorm, m->mtp_ea, hidden, eps);
    emit_rms(m, em_enc(&em), hsrc, &m->mtp_hnorm, m->mtp_eb, hidden, eps);
    em_copy(&em, m->mtp_ea, 0, m->mtp_cat, 0, hidden);
    em_copy(&em, m->mtp_eb, 0, m->mtp_cat, (NSUInteger)hidden * sizeof(float), hidden);
    emit_rows(m, em_enc(&em), &m->mtp_eh_proj, m->mtp_cat, m->x);
    emit_rms(m, em_enc(&em), m->x, &m->mtp_w.attn_norm, m->normed, hidden, eps);
    emit_attention(e, &em, &e->mtp, &m->mtp_w, position);
    emit_ffn_pre(e, &em, &m->mtp_w);
    {
        id<MTLComputeCommandEncoder> enc = em_enc(&em);
        id<MTLBuffer> resident = (__bridge id<MTLBuffer>)rl_native_metal_residency_table(m->mtp_experts);
        [enc setComputePipelineState:m->p_route];
        [enc setBuffer:m->router_logits offset:0 atIndex:0];
        [enc setBuffer:resident offset:(NSUInteger)e->mtp_layer * experts * sizeof(uint64_t) atIndex:1];
        [enc setBuffer:m->mtp_slots offset:0 atIndex:2]; [enc setBuffer:m->mtp_weights offset:0 atIndex:3];
        [enc setBuffer:m->mtp_ids offset:0 atIndex:4]; [enc setBuffer:m->mtp_miss offset:0 atIndex:5];
        [enc setBytes:&experts length:4 atIndex:6]; [enc setBytes:&topk length:4 atIndex:7]; [enc setBytes:&no_bias length:4 atIndex:8];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)]; rl_after_dispatch(enc);
    }
    rl_native_layer_info li;
    rl_expert_layout lay;
    if (!rl_native_get_layer_info(&e->mtp_expert_map, e->mtp_layer, &li, error, cap) ||
        !rl_native_expert_layout(&e->mtp_expert_map, e->mtp_layer, 0u, &lay, error, cap)) { em_close(&em); return 0; }
    if (!redmetal_topk_pool_encode_device_into(rl_native_metal_pool_handle(m->mtp_experts), (__bridge void *)em_enc(&em),
            (__bridge void *)m->mtp_slots, 0u, (__bridge void *)m->mtp_weights, 0u, topk, rl_native_expert_type_word(li.ggml_type, li.down_type),
            li.hidden_size, li.ffn_size, lay.gate_bytes, lay.up_bytes, (__bridge void *)m->ffn_in, 0u, (__bridge void *)m->routed, 0u)) {
        em_close(&em); snprintf(error, cap, "MTP expert encode failed: %s", redmetal_topk_last_error()); return 0;
    }
    emit_scale_add(e, &em);
    emit_rms(m, em_enc(&em), m->x, &m->mtp_head_norm, m->final_norm, hidden, eps);
    emit_rows(m, em_enc(&em), &m->mtp_head, m->final_norm, m->logits);
    em_close(&em);
    [em.cb commit];
    [em.cb waitUntilCompleted];
    if (em.cb.status != MTLCommandBufferStatusCompleted || em.cb.error) {
        snprintf(error, cap, "MTP command buffer failed: %s", em.cb.error.localizedDescription.UTF8String ?: "unknown"); return 0;
    }
    if (((const uint32_t *)m->mtp_miss.contents)[0]) { set_error(error, cap, "MTP expert not resident"); return 0; }
    const float *lg = (const float *)m->logits.contents;
    uint32_t best = 0;
    for (uint32_t v = 1; v < in->vocab; ++v) if (lg[v] > lg[best]) best = v;
    *draft = best;
    if (logits) memcpy(logits, lg, (size_t)in->vocab * sizeof(float));
    return 1;
}

#define RL_ROUTED_MAX_RESTARTS 4u

/* dev23: GPU-routed decode with per-layer early-out. The token runs as one command buffer; when layer f selects
 * a non-resident expert, the router kernel raises the early-out flag and every later dispatch returns at once.
 * Layers before f are complete and layer f has run up to its router exactly once, so no state is restored:
 * the CPU loads layer f's missing experts (with the ids/weights the GPU selected), runs them, and resumes the
 * GPU-routed chain at layer f+1 in the same command buffer. Only layers that miss cost a CPU round trip.
 * (dev21 restored every DeltaNet state and redid the whole token synchronously on the first miss.) */
static int step_routed(rl_engine *e, rl_metal_engine *m, uint32_t token, float *logits,
                       rl_engine_step_stats *stats, char *error, size_t cap) {
    rl_backend_state *s = &e->gpu;
    const rl_engine_info *in = &e->info;
    const uint32_t hidden = in->hidden, topk = in->top_k;
    const size_t hb = (size_t)hidden * sizeof(float);
    const double start = rl_engine_now_ms();
    id<MTLBuffer> resident = (__bridge id<MTLBuffer>)rl_native_metal_residency_table(m->experts);
    if (!resident) { set_error(error, cap, "residency table unavailable"); return 0; }
    rl_native_topk_plan plan;
    memset(&plan, 0, sizeof(plan));
    int result = 0;
    @autoreleasepool {
        if (!rl_engine_embed_token(e, token, (float *)m->x.contents, error, cap)) return 0;
        memcpy(s->embed, m->x.contents, hb);
        stats->embed_ms = rl_engine_now_ms() - start;
        uint32_t *miss = (uint32_t *)m->plan_miss.contents;
        const uint32_t *ids = (const uint32_t *)m->plan_ids.contents;
        const float *weights = (const float *)m->plan_weights.contents;
        volatile uint32_t *abort_flag = (volatile uint32_t *)m->abort.contents;
        memset(miss, 0, (size_t)in->n_layer * sizeof(uint32_t));
        abort_flag[0] = 0u;
        uint32_t first = 0, restarts = 0;
        emitter em = { m, [m->queue commandBuffer], nil, m->abort };
        for (;;) {
            if (!emit_routed_layers(e, &em, first, resident, s->position, logits != NULL, error, cap)) { em_close(&em); goto done; }
            em_close(&em);
            if (!commit_wait(em.cb, "GPU-routed token", &stats->gpu_ms, error, cap)) goto done;
            if (plan.active && !rl_native_metal_release_topk(m->experts, &plan, NULL, error, cap)) goto done;
            uint32_t f = first;
            if (abort_flag[0] != 0u) while (f < in->n_layer && miss[f] == 0u) ++f;
            else f = in->n_layer;
            /* LRU bookkeeping for the layers the GPU routed, before a load can evict one of their experts */
            for (uint32_t l = first; l < f; ++l)
                if (!rl_native_metal_touch_resident(m->experts, l, ids + (size_t)l * 64u, topk, error, cap)) goto done;
            if (abort_flag[0] == 0u) break;
            if (f >= in->n_layer) { set_error(error, cap, "early-out raised without a missed layer"); goto done; }
            /* layer f: load its missing experts (GPU-selected ids and renormalized weights) and run them */
            if (!rl_native_metal_prepare_topk(m->experts, &e->expert_map, f, ids + (size_t)f * 64u, weights + (size_t)f * 512u,
                    topk, &plan, error, cap)) goto done;
            abort_flag[0] = 0u;   /* the previous command buffer has completed: nothing reads the flag now */
            restarts++;
            em.cb = [m->queue commandBuffer];
            if (!rl_native_metal_encode_topk(m->experts, &plan, (__bridge void *)em.cb, (__bridge void *)m->ffn_in, 0u,
                    (__bridge void *)m->routed, 0u, error, cap)) goto done;
            emit_scale_add(e, &em);
            em_copy(&em, m->x, 0, m->layer_out_gpu, (NSUInteger)f * hb, hidden);
            first = f + 1u;
        }
        stats->layers_ms = rl_engine_now_ms() - start - stats->embed_ms;
        for (uint32_t l = 0; l < in->n_layer; ++l)
            memcpy(s->router_ids + (size_t)l * RL_ENGINE_MAX_TOPK, ids + (size_t)l * 64u, (size_t)topk * sizeof(uint32_t));
        memcpy(s->layer_out, m->layer_out_gpu.contents, (size_t)in->n_layer * hb);
        memcpy(s->final_norm, m->final_norm.contents, hb);
        if (logits) memcpy(logits, m->logits.contents, (size_t)in->vocab * sizeof(float));
        {
            rl_native_topk_plan none; memset(&none, 0, sizeof(none));
            rl_native_metal_telemetry tel;
            /* an inactive plan release only reports counters */
            if (!rl_native_metal_release_topk(m->experts, &none, &tel, error, cap)) goto done;
            stats->expert_loads = tel.expert_loads; stats->cache_hits = tel.cache_hits; stats->cache_misses = tel.cache_misses;
            stats->ssd_bytes = tel.bytes_read_total; stats->ssd_reads = tel.read_calls_total;
            stats->resident_slots = tel.resident_slots; stats->slot_capacity = tel.slot_capacity;
        }
        s->position++;
        stats->speculative = 1;
        stats->speculative_fallback = restarts;   /* dev23: number of per-layer early-outs in this token */
        result = 1;
    }
done:
    if (plan.active) {
        id<MTLCommandBuffer> drain = [m->queue commandBuffer];
        [drain commit];
        [drain waitUntilCompleted];
        rl_native_metal_release_topk(m->experts, &plan, NULL, NULL, 0);
    }
    stats->total_ms = rl_engine_now_ms() - start;
    return result;
}

int rl_metal_engine_step(rl_engine *e, rl_metal_engine *m, uint32_t token, float *logits,
                         rl_engine_step_stats *stats, char *error, size_t cap) {
    /* dev23 policy: the early-out path pays one resume (and the skipped dispatches of the rest of the token)
     * per layer that misses, so it only wins when misses are rare. As in dev21 it is tried after a token that
     * needed no expert load; a token with more than RL_ROUTED_MAX_RESTARTS early-outs sends the next tokens
     * back to the synchronous path until one of them loads nothing. */
    if (m->spec_enabled && !m->last_token_missed) {
        const int r = step_routed(e, m, token, logits, stats, error, cap);
        if (r) m->last_token_missed = stats->speculative_fallback > RL_ROUTED_MAX_RESTARTS;
        return r;
    }
    return rl_metal_engine_step_sync(e, m, token, logits, stats, error, cap);
}

int rl_metal_engine_step_sync(rl_engine *e, rl_metal_engine *m, uint32_t token, float *logits,
                              rl_engine_step_stats *stats, char *error, size_t cap) {
    rl_backend_state *s = &e->gpu;
    const rl_engine_info *in = &e->info;
    const uint32_t hidden = in->hidden, experts = in->n_expert, topk = in->top_k;
    const float eps = in->rms_eps;
    const double start = rl_engine_now_ms();
    uint32_t ids[RL_ENGINE_MAX_TOPK];
    float weights[RL_ENGINE_MAX_TOPK];
    float *probs = (float *)malloc((size_t)experts * sizeof(float));
    if (!probs) { set_error(error, cap, "router scratch allocation failed"); return 0; }
    rl_native_topk_plan plan;
    memset(&plan, 0, sizeof(plan));
    uint32_t pred_ids[RL_ENGINE_MAX_TOPK];
    float pred_w[RL_ENGINE_MAX_TOPK], pred_zero[RL_ENGINE_MAX_TOPK];
    memset(pred_zero, 0, sizeof(pred_zero));
    int have_pred = 0;
    int ok = 0;
    @autoreleasepool {
        id<MTLCommandBuffer> pending = nil; /* expert + residual command buffer of the previous layer */
        if (!rl_engine_embed_token(e, token, (float *)m->x.contents, error, cap)) goto done;
        memcpy(s->embed, m->x.contents, (size_t)hidden * sizeof(float));
        stats->embed_ms = rl_engine_now_ms() - start;

        for (uint32_t l = 0; l < in->n_layer; ++l) {
            const rl_layer_tensors *t = &e->layers[l];
            const mlayer *w = &m->layers[l];
            const double l0 = rl_engine_now_ms();
            emitter em = { m, [m->queue commandBuffer], nil, m->abort_zero };
            emit_steer(e, &em, l, m->x);
            emit_rms(m, em_enc(&em), m->x, &w->attn_norm, m->normed, hidden, eps);
            em_stage(&em, 0);
            if (t->kind == RL_LAYER_MAP_RECURRENT) emit_recurrent(e, &em, t, w);
            else emit_attention(e, &em, t, w, s->position);
            emit_ffn_pre(e, &em, w);
            /* dev23 pre-gating: the next layer's router applied to this layer's FFN input predicts its experts */
            const int predict = m->prefetch && l + 1u < in->n_layer;
            if (predict) emit_rows(m, em_enc(&em), &m->layers[l + 1u].router, m->ffn_in, m->pred_logits);
            em_close(&em);
            id<MTLCommandBuffer> cb = em.cb;
            /* the queue executes the previous layer's expert/residual buffer before this one */
            [cb commit];
            if (have_pred) {
                /* prefetch this layer's predicted experts while the GPU runs the previous experts and this layer */
                rl_native_topk_plan pplan;
                memset(&pplan, 0, sizeof(pplan));
                const double p0 = rl_engine_now_ms();
                if (!rl_native_metal_prepare_topk(m->experts, &e->expert_map, l, pred_ids, pred_zero, topk, &pplan, error, cap) ||
                    !rl_native_metal_release_topk(m->experts, &pplan, NULL, error, cap)) { [cb waitUntilCompleted]; goto done; }
                m->prefetch_ms += rl_engine_now_ms() - p0;
                have_pred = 0;
            }
            [cb waitUntilCompleted];
            if (cb.status != MTLCommandBufferStatusCompleted || cb.error) {
                snprintf(error, cap, "layer attention/router/shared command buffer failed: %s", cb.error.localizedDescription.UTF8String ?: "unknown");
                goto done;
            }
            stats->gpu_ms += (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
            if (predict) {
                if (!rl_native_router_select_softmax_topk((const float *)m->pred_logits.contents, experts, topk, pred_ids, pred_w, probs, error, cap)) goto done;
                have_pred = 1;
            }
            const double l1 = rl_engine_now_ms();
            if (t->kind == RL_LAYER_MAP_RECURRENT) stats->recurrent_ms += l1 - l0; else stats->attention_ms += l1 - l0;
            if (pending) {
                stats->routed_gpu_ms += (pending.GPUEndTime - pending.GPUStartTime) * 1000.0;
                m->prof_ms[12] += (pending.GPUEndTime - pending.GPUStartTime) * 1000.0;
                pending = nil;
                memcpy(s->layer_out + (size_t)(l - 1u) * hidden, m->x.contents, (size_t)hidden * sizeof(float));
                if (!rl_native_metal_release_topk(m->experts, &plan, NULL, error, cap)) goto done;
            }

            if (m->route_bias > 0.0f) {   /* dev46: select on biased logits, weigh with the unbiased ones */
                const float *z = (const float *)m->router_logits.contents;
                for (uint32_t x = 0; x < experts; ++x) probs[x] = z[x] + (rl_native_metal_is_cached(m->experts, l, x) ? m->route_bias : 0.0f);
                float *zb = (float *)malloc((size_t)experts * sizeof(float));
                if (!zb) { set_error(error, cap, "router bias allocation failed"); goto done; }
                memcpy(zb, probs, (size_t)experts * sizeof(float));
                const int sel = rl_native_router_select_softmax_topk(zb, experts, topk, ids, weights, probs, error, cap);
                free(zb);
                if (!sel) goto done;
                double sum = 0.0; float zmax = z[ids[0]];
                for (uint32_t k = 1; k < topk; ++k) if (z[ids[k]] > zmax) zmax = z[ids[k]];
                for (uint32_t k = 0; k < topk; ++k) { weights[k] = expf(z[ids[k]] - zmax); sum += weights[k]; }
                for (uint32_t k = 0; k < topk; ++k) weights[k] = (float)(weights[k] / sum);
            } else if (!rl_native_router_select_softmax_topk((const float *)m->router_logits.contents, experts, topk, ids, weights, probs, error, cap)) goto done;
            memcpy(s->router_ids + (size_t)l * RL_ENGINE_MAX_TOPK, ids, (size_t)topk * sizeof(uint32_t));
            const double l2 = rl_engine_now_ms();
            stats->router_ms += l2 - l1;

            const double read_before = m->last_read_ms;
            if (!rl_native_metal_prepare_topk(m->experts, &e->expert_map, l, ids, weights, topk, &plan, error, cap)) goto done;
            cb = [m->queue commandBuffer];
            if (!rl_native_metal_encode_topk(m->experts, &plan, (__bridge void *)cb, (__bridge void *)m->ffn_in, 0u,
                    (__bridge void *)m->routed, 0u, error, cap)) goto done;
            {
                emitter xm = { m, cb, nil, m->abort_zero };
                emit_scale_add(e, &xm);
                em_close(&xm);
            }
            [cb commit];
            pending = cb;
            const double l3 = rl_engine_now_ms();
            stats->routed_ms += l3 - l2;
            {
                const double read_now = redmetal_topk_pool_read_ms_total(m);
                stats->routed_load_ms += read_now - read_before;
                m->last_read_ms = read_now;
            }
        }
        stats->layers_ms = rl_engine_now_ms() - start - stats->embed_ms;

        const double o0 = rl_engine_now_ms();
        id<MTLCommandBuffer> cb = [m->queue commandBuffer];
        enc_rms(m, cb, m->x, &m->output_norm, m->final_norm, hidden, eps);
        if (logits) enc_rows(m, cb, &m->output, m->final_norm, m->logits);
        if (!commit_wait(cb, "output head", &stats->gpu_ms, error, cap)) goto done;
        m->prof_ms[13] += (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
        if (pending) {
            stats->routed_gpu_ms += (pending.GPUEndTime - pending.GPUStartTime) * 1000.0;
            m->prof_ms[12] += (pending.GPUEndTime - pending.GPUStartTime) * 1000.0;
            pending = nil;
        }
        memcpy(s->layer_out + (size_t)(in->n_layer - 1u) * hidden, m->x.contents, (size_t)hidden * sizeof(float));
        {
            rl_native_metal_telemetry tel;
            if (!rl_native_metal_release_topk(m->experts, &plan, &tel, error, cap)) goto done;
            stats->expert_loads = tel.expert_loads; stats->cache_hits = tel.cache_hits; stats->cache_misses = tel.cache_misses;
            stats->ssd_bytes = tel.bytes_read_total; stats->ssd_reads = tel.read_calls_total;
            stats->resident_slots = tel.resident_slots; stats->slot_capacity = tel.slot_capacity;
            m->last_token_missed = tel.cache_misses != m->misses_seen;
            m->misses_seen = tel.cache_misses;
        }
        memcpy(s->final_norm, m->final_norm.contents, (size_t)hidden * sizeof(float));
        if (logits) memcpy(logits, m->logits.contents, (size_t)in->vocab * sizeof(float));
        stats->output_ms = rl_engine_now_ms() - o0;
        s->position++;
        ok = 1;
    }
done:
    if (plan.active) {
        /* a failure left experts in flight: wait for the queue to drain before releasing */
        id<MTLCommandBuffer> drain = [m->queue commandBuffer];
        [drain commit];
        [drain waitUntilCompleted];
        rl_native_metal_release_topk(m->experts, &plan, NULL, NULL, 0);
    }
    stats->total_ms = rl_engine_now_ms() - start;
    free(probs);
    return ok;
}
