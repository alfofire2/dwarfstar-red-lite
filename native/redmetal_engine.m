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

#include "redlite_native_engine_internal.h"
#include "redlite_native_metal.h"
#include "redlite_native_quant_cpu.h"
#include "redlite_native_router_exec.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void set_error(char *dst, size_t cap, const char *msg) {
    if (dst && cap) snprintf(dst, cap, "%s", msg ? msg : "unknown Metal engine error");
}

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
"kernel void rl_rms(device const float *x [[buffer(0)]], device const float *w [[buffer(1)]], device float *y [[buffer(2)]],\n"
"    constant uint &n [[buffer(3)]], constant float &eps [[buffer(4)]], uint tid [[thread_position_in_threadgroup]],\n"
"    ushort simd_lane [[thread_index_in_simdgroup]], ushort simd_id [[simdgroup_index_in_threadgroup]]) {\n"
"    threadgroup float partial[8]; float ss = 0.0f;\n"
"    for (uint i = tid; i < n; i += 256u) ss = fma(x[i], x[i], ss);\n"
"    ss = simd_sum(ss); if (simd_lane == 0) partial[simd_id] = ss; threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    float total = 0.0f; for (uint k = 0; k < 8u; ++k) total += partial[k];\n"
"    const float inv = rsqrt(total / float(n) + eps);\n"
"    for (uint i = tid; i < n; i += 256u) y[i] = x[i] * inv * w[i];\n"
"}\n"
"kernel void rl_resid_rms(device const float *a [[buffer(0)]], device const float *b [[buffer(1)]], device const float *w [[buffer(2)]],\n"
"    device float *sum [[buffer(3)]], device float *norm [[buffer(4)]], constant uint &n [[buffer(5)]], constant float &eps [[buffer(6)]],\n"
"    uint tid [[thread_position_in_threadgroup]], ushort simd_lane [[thread_index_in_simdgroup]], ushort simd_id [[simdgroup_index_in_threadgroup]]) {\n"
"    threadgroup float partial[8]; float ss = 0.0f;\n"
"    for (uint i = tid; i < n; i += 256u) { const float v = a[i] + b[i]; sum[i] = v; ss = fma(v, v, ss); }\n"
"    ss = simd_sum(ss); if (simd_lane == 0) partial[simd_id] = ss; threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    float total = 0.0f; for (uint k = 0; k < 8u; ++k) total += partial[k];\n"
"    const float inv = rsqrt(total / float(n) + eps);\n"
"    for (uint i = tid; i < n; i += 256u) norm[i] = sum[i] * inv * w[i];\n"
"}\n"
"kernel void rl_scale_add(device const float *resid [[buffer(0)]], device const float *routed [[buffer(1)]], device const float *shared [[buffer(2)]],\n"
"    device const float *scalar [[buffer(3)]], device float *out [[buffer(4)]], constant uint &n [[buffer(5)]], uint gid [[thread_position_in_grid]]) {\n"
"    if (gid < n) out[gid] = resid[gid] + (routed[gid] + shared[gid] * scalar[0]);\n"
"}\n"
/* ---- quantized row dot products: lanes_per_row SIMD lanes per row, one block per lane per step ---- */
"inline float rl_q8_block(device const uchar *bp, device const float *x) {\n"
"    const float d = fp16(bp); device const int8_t *q = (device const int8_t *)(bp + 2ul); float acc = 0.0f;\n"
"    for (uint j = 0; j < 32u; ++j) acc += x[j] * (d * float(q[j]));\n"
"    return acc;\n"
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
"kernel void " NAME "(device const uchar *weights [[buffer(0)]], constant uint &ncols [[buffer(1)]], device const float *x [[buffer(2)]],\n" \
"    device float *out [[buffer(3)]], constant uint &nrows [[buffer(4)]], constant uint &lanes [[buffer(5)]]" EXTRA_PARAM ",\n" \
"    uint tid [[thread_position_in_grid]], ushort simd_lane [[thread_index_in_simdgroup]]) {\n" \
"    const uint row = tid / lanes; const uint lane = uint(simd_lane) % lanes; const bool active = row < nrows;\n" \
"    const uint blocks = ncols / " BLOCK "u; const ulong row_bytes = ulong(blocks) * " BYTES "ul; float acc = 0.0f;\n" \
"    if (active) { device const uchar *rp = weights + ulong(row) * row_bytes;\n" \
"        for (uint b = lane; b < blocks; b += lanes) { device const uchar *bp = rp + ulong(b) * " BYTES "ul; device const float *xc = x + b * " BLOCK "u; acc += " DOTEXPR "; } }\n" \
"    for (uint off = lanes >> 1; off > 0u; off >>= 1) acc += simd_shuffle_xor(acc, ushort(off));\n" \
"    if (active && lane == 0u) out[row] = acc;\n" \
"}\n"
RL_ROWS_KERNEL("rl_rows_q8", "32", "34", "rl_q8_block(bp, xc)", "")
RL_ROWS_KERNEL("rl_rows_q4k", "256", "144", "rl_q4k_block(bp, xc)", "")
RL_ROWS_KERNEL("rl_rows_q5k", "256", "176", "rl_q5k_block(bp, xc)", "")
RL_ROWS_KERNEL("rl_rows_q6k", "256", "210", "rl_q6k_block(bp, xc)", "")
RL_ROWS_KERNEL("rl_rows_iq2xxs", "256", "66", "rl_iq2xxs_block(bp, xc, grid)", ", device const uchar *grid [[buffer(6)]]")
"kernel void rl_rows_f32(device const float *weights [[buffer(0)]], constant uint &ncols [[buffer(1)]], device const float *x [[buffer(2)]],\n"
"    device float *out [[buffer(3)]], constant uint &nrows [[buffer(4)]], constant uint &lanes [[buffer(5)]],\n"
"    uint tid [[thread_position_in_grid]], ushort simd_lane [[thread_index_in_simdgroup]]) {\n"
"    const uint row = tid / lanes; const uint lane = uint(simd_lane) % lanes; const bool active = row < nrows; float acc = 0.0f;\n"
"    if (active) { device const float *rp = weights + ulong(row) * ulong(ncols); for (uint i = lane; i < ncols; i += lanes) acc = fma(rp[i], x[i], acc); }\n"
"    for (uint off = lanes >> 1; off > 0u; off >>= 1) acc += simd_shuffle_xor(acc, ushort(off));\n"
"    if (active && lane == 0u) out[row] = acc;\n"
"}\n"
/* ---- Gated DeltaNet (validated dev15 kernels, kv_ratio pairing) ---- */
"kernel void dn_ba_params(device const float *ba [[buffer(0)]], device const float *dt [[buffer(1)]], device const float *avec [[buffer(2)]],\n"
"    device float *beta [[buffer(3)]], device float *gate [[buffer(4)]], constant uint &dt_rank [[buffer(5)]], constant uint &n_group [[buffer(6)]],\n"
"    uint gid [[thread_position_in_grid]]) {\n"
"    if (gid >= dt_rank) return; const uint width = dt_rank / n_group; const uint group = gid / width; const uint local = gid - group * width;\n"
"    const uint stride = 2u * width; const float b = ba[group * stride + local]; const float alpha = ba[group * stride + width + local] + dt[gid];\n"
"    beta[gid] = 1.0f / (1.0f + exp(-b)); const float sp = alpha > 20.0f ? alpha : log(1.0f + exp(alpha)); gate[gid] = sp * avec[gid];\n"
"}\n"
"kernel void dn_conv_silu(device const float *state [[buffer(0)]], device const float *qkv [[buffer(1)]], device const float *conv_w [[buffer(2)]],\n"
"    device float *out [[buffer(3)]], constant uint &channels [[buffer(4)]], constant uint &dconv [[buffer(5)]], uint gid [[thread_position_in_grid]]) {\n"
"    if (gid >= channels) return; float acc = 0.0f; const uint ns = dconv - 1u; const uint sb = gid * ns; const uint kb = gid * dconv;\n"
"    for (uint j = 0; j < ns; ++j) acc = fma(state[sb + j], conv_w[kb + j], acc);\n"
"    acc = fma(qkv[gid], conv_w[kb + ns], acc); out[gid] = acc / (1.0f + exp(-acc));\n"
"}\n"
"kernel void dn_qk_l2(device const float *src [[buffer(0)]], device float *dst [[buffer(1)]], constant uint &offset [[buffer(2)]],\n"
"    constant uint &head_dim [[buffer(3)]], constant uint &heads [[buffer(4)]], constant float &eps [[buffer(5)]], uint gid [[thread_position_in_grid]]) {\n"
"    if (gid >= heads) return; const uint base = offset + gid * head_dim; float ss = 0.0f;\n"
"    for (uint j = 0; j < head_dim; ++j) ss = fma(src[base + j], src[base + j], ss);\n"
"    const float scale = 1.0f / max(sqrt(ss), eps);\n"
"    for (uint j = 0; j < head_dim; ++j) dst[gid * head_dim + j] = src[base + j] * scale;\n"
"}\n"
"kernel void dn_shift_state(device const float *state [[buffer(0)]], device const float *qkv [[buffer(1)]], device float *next_state [[buffer(2)]],\n"
"    constant uint &channels [[buffer(3)]], constant uint &dconv [[buffer(4)]], uint gid [[thread_position_in_grid]]) {\n"
"    if (gid >= channels) return; const uint ns = dconv - 1u; const uint b = gid * ns;\n"
"    for (uint j = 0; j + 1u < ns; ++j) next_state[b + j] = state[b + j + 1u];\n"
"    next_state[b + ns - 1u] = qkv[gid];\n"
"}\n"
"kernel void dn_state_decay(device const float *prev [[buffer(0)]], device const float *gate [[buffer(1)]], device float *state [[buffer(2)]],\n"
"    constant uint &state_size [[buffer(3)]], constant uint &value_heads [[buffer(4)]], uint gid [[thread_position_in_grid]]) {\n"
"    const uint matrix = state_size * state_size; const uint total = value_heads * matrix; if (gid >= total) return;\n"
"    const uint h = gid / matrix; state[gid] = prev[gid] * exp(gate[h]);\n"
"}\n"
"kernel void dn_state_delta(device const float *state [[buffer(0)]], device const float *k [[buffer(1)]], device const float *v [[buffer(2)]],\n"
"    device const float *beta [[buffer(3)]], device float *delta [[buffer(4)]], constant uint &state_size [[buffer(5)]],\n"
"    constant uint &kv_ratio [[buffer(6)]], constant uint &value_heads [[buffer(7)]], uint gid [[thread_position_in_grid]]) {\n"
"    const uint total = value_heads * state_size; if (gid >= total) return;\n"
"    const uint h = gid / state_size; const uint j = gid - h * state_size; const uint kh = h / kv_ratio;\n"
"    const uint sb = h * state_size * state_size + j * state_size; const uint kb = kh * state_size; float sum = 0.0f;\n"
"    for (uint i = 0; i < state_size; ++i) sum = fma(state[sb + i], k[kb + i], sum);\n"
"    delta[gid] = (v[gid] - sum) * beta[h];\n"
"}\n"
"kernel void dn_state_update(device float *state [[buffer(0)]], device const float *k [[buffer(1)]], device const float *delta [[buffer(2)]],\n"
"    constant uint &state_size [[buffer(3)]], constant uint &kv_ratio [[buffer(4)]], constant uint &value_heads [[buffer(5)]], uint gid [[thread_position_in_grid]]) {\n"
"    const uint matrix = state_size * state_size; const uint total = value_heads * matrix; if (gid >= total) return;\n"
"    const uint h = gid / matrix; const uint rem = gid - h * matrix; const uint j = rem / state_size; const uint i = rem - j * state_size; const uint kh = h / kv_ratio;\n"
"    state[gid] += k[kh * state_size + i] * delta[h * state_size + j];\n"
"}\n"
"kernel void dn_state_output(device const float *state [[buffer(0)]], device const float *q [[buffer(1)]], device float *out [[buffer(2)]],\n"
"    constant uint &state_size [[buffer(3)]], constant uint &kv_ratio [[buffer(4)]], constant uint &value_heads [[buffer(5)]], uint gid [[thread_position_in_grid]]) {\n"
"    const uint total = value_heads * state_size; if (gid >= total) return;\n"
"    const uint h = gid / state_size; const uint j = gid - h * state_size; const uint kh = h / kv_ratio;\n"
"    const uint sb = h * state_size * state_size + j * state_size; const uint qb = kh * state_size;\n"
"    float sum = 0.0f; for (uint i = 0; i < state_size; ++i) sum = fma(state[sb + i], q[qb + i], sum);\n"
"    out[gid] = sum * (1.0f / sqrt((float)state_size));\n"
"}\n"
"kernel void dn_tail_norm(device const float *core [[buffer(0)]], device const float *z [[buffer(1)]], device const float *w [[buffer(2)]],\n"
"    device float *out [[buffer(3)]], constant float &eps [[buffer(4)]], constant uint &head_dim [[buffer(5)]], constant uint &heads [[buffer(6)]],\n"
"    uint h [[thread_position_in_grid]]) {\n"
"    if (h >= heads) return; const uint base = h * head_dim; float ss = 0.0f;\n"
"    for (uint i = 0; i < head_dim; ++i) ss = fma(core[base+i], core[base+i], ss);\n"
"    const float inv = rsqrt(ss / float(head_dim) + eps);\n"
"    for (uint i = 0; i < head_dim; ++i) { const float g = z[base+i]; const float silu = g / (1.0f + exp(-g)); out[base+i] = core[base+i] * inv * w[i] * silu; }\n"
"}\n"
/* ---- full attention (validated dev16 kernels + general GQA) ---- */
"kernel void attn_qgate_norm(device const float *qgate [[buffer(0)]], device const float *w [[buffer(1)]], device float *query [[buffer(2)]],\n"
"    device float *gate [[buffer(3)]], constant uint &head_dim [[buffer(4)]], constant float &eps [[buffer(5)]], constant uint &heads [[buffer(6)]],\n"
"    uint head [[thread_position_in_grid]]) {\n"
"    if (head >= heads) return; const uint src = head * head_dim * 2u; const uint dst = head * head_dim; float ss = 0.0f;\n"
"    for (uint i = 0; i < head_dim; ++i) ss = fma(qgate[src + i], qgate[src + i], ss);\n"
"    const float inv = rsqrt(ss / float(head_dim) + eps);\n"
"    for (uint i = 0; i < head_dim; ++i) { query[dst + i] = qgate[src + i] * inv * w[i]; gate[dst + i] = qgate[src + head_dim + i]; }\n"
"}\n"
"kernel void attn_k_norm(device const float *key_raw [[buffer(0)]], device const float *w [[buffer(1)]], device float *key [[buffer(2)]],\n"
"    constant uint &head_dim [[buffer(3)]], constant float &eps [[buffer(4)]], constant uint &heads [[buffer(5)]], uint head [[thread_position_in_grid]]) {\n"
"    if (head >= heads) return; const uint base = head * head_dim; float ss = 0.0f;\n"
"    for (uint i = 0; i < head_dim; ++i) ss = fma(key_raw[base + i], key_raw[base + i], ss);\n"
"    const float inv = rsqrt(ss / float(head_dim) + eps);\n"
"    for (uint i = 0; i < head_dim; ++i) key[base + i] = key_raw[base + i] * inv * w[i];\n"
"}\n"
"kernel void attn_rope_cache(device const float *query [[buffer(0)]], device const float *key [[buffer(1)]], device const float *value [[buffer(2)]],\n"
"    device float *query_rope [[buffer(3)]], device float *key_cache [[buffer(4)]], device float *value_cache [[buffer(5)]],\n"
"    constant uint &head_dim [[buffer(6)]], constant uint &query_heads [[buffer(7)]], constant uint &kv_heads [[buffer(8)]],\n"
"    constant uint &position [[buffer(9)]], constant uint &rope_dims [[buffer(10)]], constant float &freq_base [[buffer(11)]],\n"
"    uint head [[thread_position_in_grid]]) {\n"
"    const uint rope_half = rope_dims / 2u;\n"
"    if (head < query_heads) {\n"
"        const uint base = head * head_dim;\n"
"        for (uint i = 0; i < head_dim; ++i) query_rope[base + i] = query[base + i];\n"
"        for (uint i = 0; i < rope_half; ++i) {\n"
"            const float theta = float(position) * pow(freq_base, -2.0f * float(i) / float(rope_dims));\n"
"            const float c = cos(theta); const float s = sin(theta);\n"
"            const float x0 = query[base + i]; const float x1 = query[base + rope_half + i];\n"
"            query_rope[base + i] = x0 * c - x1 * s; query_rope[base + rope_half + i] = x0 * s + x1 * c;\n"
"        }\n"
"    }\n"
"    if (head < kv_heads) {\n"
"        const uint src = head * head_dim; const uint dst = (position * kv_heads + head) * head_dim;\n"
"        for (uint i = 0; i < head_dim; ++i) { key_cache[dst + i] = key[src + i]; value_cache[dst + i] = value[src + i]; }\n"
"        for (uint i = 0; i < rope_half; ++i) {\n"
"            const float theta = float(position) * pow(freq_base, -2.0f * float(i) / float(rope_dims));\n"
"            const float c = cos(theta); const float s = sin(theta);\n"
"            const float x0 = key[src + i]; const float x1 = key[src + rope_half + i];\n"
"            key_cache[dst + i] = x0 * c - x1 * s; key_cache[dst + rope_half + i] = x0 * s + x1 * c;\n"
"        }\n"
"    }\n"
"}\n"
"kernel void attn_gqa(device const float *query [[buffer(0)]], device const float *key_cache [[buffer(1)]], device const float *value_cache [[buffer(2)]],\n"
"    device const float *gate [[buffer(3)]], device float *gated [[buffer(4)]], constant uint &head_dim [[buffer(5)]],\n"
"    constant uint &query_heads [[buffer(6)]], constant uint &kv_heads [[buffer(7)]], constant uint &seq_len [[buffer(8)]],\n"
"    uint head [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],\n"
"    ushort simd_lane [[thread_index_in_simdgroup]], ushort simd_id [[simdgroup_index_in_threadgroup]]) {\n"
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
/* ---- shared expert (validated dev14 kernels) ---- */
"kernel void sh_scalar_gate(device const float *w [[buffer(0)]], device const float *x [[buffer(1)]], device float *scalar [[buffer(2)]],\n"
"    constant uint &hidden [[buffer(3)]], uint tid [[thread_position_in_threadgroup]], ushort simd_lane [[thread_index_in_simdgroup]],\n"
"    ushort simd_id [[simdgroup_index_in_threadgroup]]) {\n"
"    threadgroup float partial[8]; float acc = 0.0f;\n"
"    for (uint i = tid; i < hidden; i += 256u) acc = fma(w[i], x[i], acc);\n"
"    acc = simd_sum(acc); if (simd_lane == 0) partial[simd_id] = acc; threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    if (tid == 0u) { float total = 0.0f; for (uint k = 0; k < 8u; ++k) total += partial[k]; scalar[0] = 1.0f / (1.0f + exp(-total)); }\n"
"}\n"
"kernel void sh_silu_mul(device const float *gate [[buffer(0)]], device const float *up [[buffer(1)]], device float *act [[buffer(2)]],\n"
"    constant uint &count [[buffer(3)]], uint gid [[thread_position_in_grid]]) {\n"
"    if (gid >= count) return; const float g = gate[gid]; act[gid] = (g / (1.0f + exp(-g))) * up[gid];\n"
"}\n";

typedef struct {
    __unsafe_unretained id<MTLBuffer> buf;
    NSUInteger off;
    uint32_t type;
    uint32_t rows;
    uint32_t cols;
} mweight;

typedef struct {
    mweight attn_norm, post_norm;
    mweight qkv, z, ba, conv, dt, a, ssm_norm, ssm_out;
    mweight q, k, v, q_norm, k_norm, o;
    mweight router, sh_gate_inp, sh_gate, sh_up, sh_down;
} mlayer;

struct rl_metal_engine {
    id<MTLDevice> dev;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> lib;
    id<MTLComputePipelineState> p_rms, p_resid_rms, p_scale_add;
    id<MTLComputePipelineState> p_rows_f32, p_rows_q8, p_rows_q4k, p_rows_q5k, p_rows_q6k, p_rows_iq2xxs;
    id<MTLComputePipelineState> p_dn_ba, p_dn_conv, p_dn_l2, p_dn_shift, p_dn_decay, p_dn_delta, p_dn_update, p_dn_output, p_dn_tail;
    id<MTLComputePipelineState> p_attn_qgate, p_attn_knorm, p_attn_rope, p_attn_gqa;
    id<MTLComputePipelineState> p_sh_scalar, p_sh_silu;
    id<MTLBuffer> grid;
    mlayer *layers;
    mweight output_norm, output;
    __unsafe_unretained id<MTLBuffer> *conv_state, *rec_state, *kcache, *vcache;
    NSMutableArray *keep;
    uint32_t n_recurrent, n_attention;
    id<MTLBuffer> x, normed, branch, resid, ffn_in;
    id<MTLBuffer> qkv, z, ba, beta, gate, conv_silu, q, k, delta, core, ng, next_conv, rec_scratch;
    id<MTLBuffer> qgate_raw, k_raw, value, query, agate, key, query_rope, gated;
    id<MTLBuffer> router_logits, sh_gate, sh_up, sh_act, sh_out, scalar, routed, final_norm, logits;
    rl_native_metal_runtime *experts;
    float *routed_host;
    uint64_t resident_bytes;
    uint64_t last_bytes_read, last_calls, last_hits, last_misses, last_loads;
    double last_read_ms;
};

static id<MTLComputePipelineState> make_pipe(id<MTLDevice> dev, id<MTLLibrary> lib, NSString *name, char *error, size_t cap) {
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    if (!fn) { snprintf(error, cap, "missing Metal function %s", name.UTF8String); return nil; }
    NSError *e = nil;
    id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:fn error:&e];
    if (!p) snprintf(error, cap, "pipeline %s failed: %s", name.UTF8String, e.localizedDescription.UTF8String ?: "unknown");
    return p;
}

static id<MTLBuffer> new_buf(rl_metal_engine *m, size_t bytes) {
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

static id<MTLComputePipelineState> rows_pipe(rl_metal_engine *m, uint32_t type) {
    switch (type) {
        case 0: return m->p_rows_f32;
        case 8: return m->p_rows_q8;
        case 12: return m->p_rows_q4k;
        case 13: return m->p_rows_q5k;
        case 14: return m->p_rows_q6k;
        case 16: return m->p_rows_iq2xxs;
        default: return nil;
    }
}

static void enc_1d(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p, NSUInteger n, NSUInteger tg_max) {
    const NSUInteger tg = MIN(tg_max, p.maxTotalThreadsPerThreadgroup);
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg ? tg : 1, 1, 1)];
}

static uint32_t lanes_for(uint32_t type, uint32_t ncols) {
    uint32_t blocks;
    if (type == 0u) blocks = ncols / 64u;          /* F32: strided lanes */
    else if (type == 8u) blocks = ncols / 32u;
    else blocks = ncols / 256u;
    uint32_t lanes = 1u;
    while (lanes < blocks && lanes < 32u) lanes <<= 1;
    return lanes;
}

/* out[row] = W[row] . x for all rows of the weight */
static void enc_rows(rl_metal_engine *m, id<MTLCommandBuffer> cb, const mweight *w, id<MTLBuffer> x, id<MTLBuffer> out) {
    id<MTLComputePipelineState> p = rows_pipe(m, w->type);
    const uint32_t lanes = lanes_for(w->type, w->cols);
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:p];
    [enc setBuffer:w->buf offset:w->off atIndex:0];
    [enc setBytes:&w->cols length:sizeof(w->cols) atIndex:1];
    [enc setBuffer:x offset:0 atIndex:2];
    [enc setBuffer:out offset:0 atIndex:3];
    [enc setBytes:&w->rows length:sizeof(w->rows) atIndex:4];
    [enc setBytes:&lanes length:sizeof(lanes) atIndex:5];
    if (w->type == 16u) [enc setBuffer:m->grid offset:0 atIndex:6];
    const NSUInteger threads = (((NSUInteger)w->rows * lanes) + 31u) & ~(NSUInteger)31u;
    [enc dispatchThreads:MTLSizeMake(threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [enc endEncoding];
}

static void enc_rms(rl_metal_engine *m, id<MTLCommandBuffer> cb, id<MTLBuffer> x, const mweight *w, id<MTLBuffer> y, uint32_t n, float eps) {
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:m->p_rms];
    [enc setBuffer:x offset:0 atIndex:0];
    [enc setBuffer:w->buf offset:w->off atIndex:1];
    [enc setBuffer:y offset:0 atIndex:2];
    [enc setBytes:&n length:sizeof(n) atIndex:3];
    [enc setBytes:&eps length:sizeof(eps) atIndex:4];
    [enc dispatchThreads:MTLSizeMake(256, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc endEncoding];
}

static void enc_blit(id<MTLCommandBuffer> cb, id<MTLBuffer> src, id<MTLBuffer> dst, size_t bytes) {
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit copyFromBuffer:src sourceOffset:0 toBuffer:dst destinationOffset:0 size:bytes];
    [blit endEncoding];
}

static int commit_wait(id<MTLCommandBuffer> cb, const char *what, double *gpu_ms, char *error, size_t cap) {
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
        m->dev = MTLCreateSystemDefaultDevice();
        if (!m->dev || !m->dev.hasUnifiedMemory) { set_error(error, cap, "Apple unified-memory Metal device required"); rl_metal_engine_destroy(m); return NULL; }
        m->queue = [m->dev newCommandQueue];
        NSError *le = nil;
        m->lib = [m->dev newLibraryWithSource:kEngineSource options:nil error:&le];
        if (!m->queue || !m->lib) {
            snprintf(error, cap, "Metal engine library compile failed: %s", le.localizedDescription.UTF8String ?: "unknown");
            rl_metal_engine_destroy(m); return NULL;
        }
        struct { __strong id<MTLComputePipelineState> *slot; NSString *name; } pipes[] = {
            {&m->p_rms, @"rl_rms"}, {&m->p_resid_rms, @"rl_resid_rms"}, {&m->p_scale_add, @"rl_scale_add"},
            {&m->p_rows_f32, @"rl_rows_f32"}, {&m->p_rows_q8, @"rl_rows_q8"}, {&m->p_rows_q4k, @"rl_rows_q4k"},
            {&m->p_rows_q5k, @"rl_rows_q5k"}, {&m->p_rows_q6k, @"rl_rows_q6k"}, {&m->p_rows_iq2xxs, @"rl_rows_iq2xxs"},
            {&m->p_dn_ba, @"dn_ba_params"}, {&m->p_dn_conv, @"dn_conv_silu"}, {&m->p_dn_l2, @"dn_qk_l2"},
            {&m->p_dn_shift, @"dn_shift_state"}, {&m->p_dn_decay, @"dn_state_decay"}, {&m->p_dn_delta, @"dn_state_delta"},
            {&m->p_dn_update, @"dn_state_update"}, {&m->p_dn_output, @"dn_state_output"}, {&m->p_dn_tail, @"dn_tail_norm"},
            {&m->p_attn_qgate, @"attn_qgate_norm"}, {&m->p_attn_knorm, @"attn_k_norm"}, {&m->p_attn_rope, @"attn_rope_cache"},
            {&m->p_attn_gqa, @"attn_gqa"}, {&m->p_sh_scalar, @"sh_scalar_gate"}, {&m->p_sh_silu, @"sh_silu_mul"},
        };
        for (size_t i = 0; i < sizeof(pipes) / sizeof(pipes[0]); ++i) {
            *pipes[i].slot = make_pipe(m->dev, m->lib, pipes[i].name, error, cap);
            if (!*pipes[i].slot) { rl_metal_engine_destroy(m); return NULL; }
        }
        m->grid = [m->dev newBufferWithBytes:e->iq2_grid length:RL_IQ2_XXS_GRID_COUNT options:MTLResourceStorageModeShared];

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
        const size_t kv_bytes = rl_engine_kv_row_count(e) * (size_t)in->context * sizeof(float);
        m->conv_state = (__unsafe_unretained id<MTLBuffer> *)calloc(in->n_recurrent ? in->n_recurrent : 1u, sizeof(id));
        m->rec_state = (__unsafe_unretained id<MTLBuffer> *)calloc(in->n_recurrent ? in->n_recurrent : 1u, sizeof(id));
        m->kcache = (__unsafe_unretained id<MTLBuffer> *)calloc(in->n_attention ? in->n_attention : 1u, sizeof(id));
        m->vcache = (__unsafe_unretained id<MTLBuffer> *)calloc(in->n_attention ? in->n_attention : 1u, sizeof(id));
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
    }
    if (error && cap) error[0] = '\0';
    return m;
}

void rl_metal_engine_destroy(rl_metal_engine *m) {
    if (!m) return;
    if (m->experts) rl_native_metal_destroy(m->experts);
    free(m->layers); free(m->conv_state); free(m->rec_state); free(m->kcache); free(m->vcache); free(m->routed_host);
    m->x = m->normed = m->branch = m->resid = m->ffn_in = m->qkv = m->z = m->ba = m->beta = m->gate = m->conv_silu = nil;
    m->q = m->k = m->delta = m->core = m->ng = m->next_conv = m->rec_scratch = m->qgate_raw = m->k_raw = m->value = nil;
    m->query = m->agate = m->key = m->query_rope = m->gated = m->router_logits = m->sh_gate = m->sh_up = m->sh_act = nil;
    m->sh_out = m->scalar = m->routed = m->final_norm = m->logits = m->grid = nil;
    m->p_rms = m->p_resid_rms = m->p_scale_add = m->p_rows_f32 = m->p_rows_q8 = m->p_rows_q4k = m->p_rows_q5k = m->p_rows_q6k = m->p_rows_iq2xxs = nil;
    m->p_dn_ba = m->p_dn_conv = m->p_dn_l2 = m->p_dn_shift = m->p_dn_decay = m->p_dn_delta = m->p_dn_update = m->p_dn_output = m->p_dn_tail = nil;
    m->p_attn_qgate = m->p_attn_knorm = m->p_attn_rope = m->p_attn_gqa = m->p_sh_scalar = m->p_sh_silu = nil;
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
    for (uint32_t a = 0; a < m->n_attention; ++a) {
        memset(m->kcache[a].contents, 0, m->kcache[a].length);
        memset(m->vcache[a].contents, 0, m->vcache[a].length);
    }
    return 1;
}

uint64_t rl_metal_engine_resident_bytes(const rl_metal_engine *m) { return m ? m->resident_bytes : 0u; }

static void enc_recurrent(rl_engine *e, rl_metal_engine *m, id<MTLCommandBuffer> cb, const rl_layer_tensors *t, const mlayer *w) {
    const rl_engine_info *in = &e->info;
    const uint32_t r = t->recurrent_index;
    const uint32_t channels = in->channels, dconv = in->d_conv, rank = in->dt_rank, groups = in->n_group, S = in->d_state;
    const uint32_t qk_each = S * groups, d_inner = in->d_inner, head_v = in->head_v;
    const uint32_t kv_ratio = rank / groups;
    const float eps = in->rms_eps;
    enc_rows(m, cb, &w->qkv, m->normed, m->qkv);
    enc_rows(m, cb, &w->z, m->normed, m->z);
    enc_rows(m, cb, &w->ba, m->normed, m->ba);
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:m->p_dn_ba];
    [enc setBuffer:m->ba offset:0 atIndex:0]; [enc setBuffer:w->dt.buf offset:w->dt.off atIndex:1]; [enc setBuffer:w->a.buf offset:w->a.off atIndex:2];
    [enc setBuffer:m->beta offset:0 atIndex:3]; [enc setBuffer:m->gate offset:0 atIndex:4];
    [enc setBytes:&rank length:4 atIndex:5]; [enc setBytes:&groups length:4 atIndex:6];
    enc_1d(enc, m->p_dn_ba, rank, 64u); [enc endEncoding];

    enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:m->p_dn_conv];
    [enc setBuffer:m->conv_state[r] offset:0 atIndex:0]; [enc setBuffer:m->qkv offset:0 atIndex:1]; [enc setBuffer:w->conv.buf offset:w->conv.off atIndex:2];
    [enc setBuffer:m->conv_silu offset:0 atIndex:3]; [enc setBytes:&channels length:4 atIndex:4]; [enc setBytes:&dconv length:4 atIndex:5];
    enc_1d(enc, m->p_dn_conv, channels, 64u); [enc endEncoding];

    uint32_t offset = 0;
    enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:m->p_dn_l2];
    [enc setBuffer:m->conv_silu offset:0 atIndex:0]; [enc setBuffer:m->q offset:0 atIndex:1]; [enc setBytes:&offset length:4 atIndex:2];
    [enc setBytes:&S length:4 atIndex:3]; [enc setBytes:&groups length:4 atIndex:4]; [enc setBytes:&eps length:4 atIndex:5];
    enc_1d(enc, m->p_dn_l2, groups, 64u); [enc endEncoding];
    offset = qk_each;
    enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:m->p_dn_l2];
    [enc setBuffer:m->conv_silu offset:0 atIndex:0]; [enc setBuffer:m->k offset:0 atIndex:1]; [enc setBytes:&offset length:4 atIndex:2];
    [enc setBytes:&S length:4 atIndex:3]; [enc setBytes:&groups length:4 atIndex:4]; [enc setBytes:&eps length:4 atIndex:5];
    enc_1d(enc, m->p_dn_l2, groups, 64u); [enc endEncoding];

    enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:m->p_dn_shift];
    [enc setBuffer:m->conv_state[r] offset:0 atIndex:0]; [enc setBuffer:m->qkv offset:0 atIndex:1]; [enc setBuffer:m->next_conv offset:0 atIndex:2];
    [enc setBytes:&channels length:4 atIndex:3]; [enc setBytes:&dconv length:4 atIndex:4];
    enc_1d(enc, m->p_dn_shift, channels, 64u); [enc endEncoding];

    const NSUInteger state_total = (NSUInteger)rank * S * S;
    const NSUInteger vec_total = (NSUInteger)rank * S;
    const NSUInteger v_offset = (NSUInteger)(2u * qk_each) * sizeof(float);
    enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:m->p_dn_decay];
    [enc setBuffer:m->rec_state[r] offset:0 atIndex:0]; [enc setBuffer:m->gate offset:0 atIndex:1]; [enc setBuffer:m->rec_scratch offset:0 atIndex:2];
    [enc setBytes:&S length:4 atIndex:3]; [enc setBytes:&rank length:4 atIndex:4];
    enc_1d(enc, m->p_dn_decay, state_total, 128u); [enc endEncoding];

    enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:m->p_dn_delta];
    [enc setBuffer:m->rec_scratch offset:0 atIndex:0]; [enc setBuffer:m->k offset:0 atIndex:1]; [enc setBuffer:m->conv_silu offset:v_offset atIndex:2];
    [enc setBuffer:m->beta offset:0 atIndex:3]; [enc setBuffer:m->delta offset:0 atIndex:4];
    [enc setBytes:&S length:4 atIndex:5]; [enc setBytes:&kv_ratio length:4 atIndex:6]; [enc setBytes:&rank length:4 atIndex:7];
    enc_1d(enc, m->p_dn_delta, vec_total, 128u); [enc endEncoding];

    enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:m->p_dn_update];
    [enc setBuffer:m->rec_scratch offset:0 atIndex:0]; [enc setBuffer:m->k offset:0 atIndex:1]; [enc setBuffer:m->delta offset:0 atIndex:2];
    [enc setBytes:&S length:4 atIndex:3]; [enc setBytes:&kv_ratio length:4 atIndex:4]; [enc setBytes:&rank length:4 atIndex:5];
    enc_1d(enc, m->p_dn_update, state_total, 128u); [enc endEncoding];

    enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:m->p_dn_output];
    [enc setBuffer:m->rec_scratch offset:0 atIndex:0]; [enc setBuffer:m->q offset:0 atIndex:1]; [enc setBuffer:m->core offset:0 atIndex:2];
    [enc setBytes:&S length:4 atIndex:3]; [enc setBytes:&kv_ratio length:4 atIndex:4]; [enc setBytes:&rank length:4 atIndex:5];
    enc_1d(enc, m->p_dn_output, vec_total, 128u); [enc endEncoding];

    enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:m->p_dn_tail];
    [enc setBuffer:m->core offset:0 atIndex:0]; [enc setBuffer:m->z offset:0 atIndex:1]; [enc setBuffer:w->ssm_norm.buf offset:w->ssm_norm.off atIndex:2];
    [enc setBuffer:m->ng offset:0 atIndex:3]; [enc setBytes:&eps length:4 atIndex:4]; [enc setBytes:&head_v length:4 atIndex:5]; [enc setBytes:&rank length:4 atIndex:6];
    enc_1d(enc, m->p_dn_tail, rank, 128u); [enc endEncoding];

    enc_rows(m, cb, &w->ssm_out, m->ng, m->branch);
    (void)d_inner;
    enc_blit(cb, m->next_conv, m->conv_state[r], rl_engine_conv_count(e) * sizeof(float));
    enc_blit(cb, m->rec_scratch, m->rec_state[r], rl_engine_rec_count(e) * sizeof(float));
}

static void enc_attention(rl_engine *e, rl_metal_engine *m, id<MTLCommandBuffer> cb, const rl_layer_tensors *t, const mlayer *w, uint32_t position) {
    const rl_engine_info *in = &e->info;
    const uint32_t a = t->attention_index;
    const uint32_t head_dim = in->head_dim, qheads = in->n_head, kvheads = in->n_head_kv, rope_dims = in->rope_dims;
    const uint32_t seq_len = position + 1u;
    const float eps = in->rms_eps, base = in->rope_freq_base;
    enc_rows(m, cb, &w->q, m->normed, m->qgate_raw);
    enc_rows(m, cb, &w->k, m->normed, m->k_raw);
    enc_rows(m, cb, &w->v, m->normed, m->value);
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:m->p_attn_qgate];
    [enc setBuffer:m->qgate_raw offset:0 atIndex:0]; [enc setBuffer:w->q_norm.buf offset:w->q_norm.off atIndex:1];
    [enc setBuffer:m->query offset:0 atIndex:2]; [enc setBuffer:m->agate offset:0 atIndex:3];
    [enc setBytes:&head_dim length:4 atIndex:4]; [enc setBytes:&eps length:4 atIndex:5]; [enc setBytes:&qheads length:4 atIndex:6];
    enc_1d(enc, m->p_attn_qgate, qheads, 32u); [enc endEncoding];

    enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:m->p_attn_knorm];
    [enc setBuffer:m->k_raw offset:0 atIndex:0]; [enc setBuffer:w->k_norm.buf offset:w->k_norm.off atIndex:1]; [enc setBuffer:m->key offset:0 atIndex:2];
    [enc setBytes:&head_dim length:4 atIndex:3]; [enc setBytes:&eps length:4 atIndex:4]; [enc setBytes:&kvheads length:4 atIndex:5];
    enc_1d(enc, m->p_attn_knorm, kvheads, 32u); [enc endEncoding];

    enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:m->p_attn_rope];
    [enc setBuffer:m->query offset:0 atIndex:0]; [enc setBuffer:m->key offset:0 atIndex:1]; [enc setBuffer:m->value offset:0 atIndex:2];
    [enc setBuffer:m->query_rope offset:0 atIndex:3]; [enc setBuffer:m->kcache[a] offset:0 atIndex:4]; [enc setBuffer:m->vcache[a] offset:0 atIndex:5];
    [enc setBytes:&head_dim length:4 atIndex:6]; [enc setBytes:&qheads length:4 atIndex:7]; [enc setBytes:&kvheads length:4 atIndex:8];
    [enc setBytes:&position length:4 atIndex:9]; [enc setBytes:&rope_dims length:4 atIndex:10]; [enc setBytes:&base length:4 atIndex:11];
    enc_1d(enc, m->p_attn_rope, qheads, 32u); [enc endEncoding];

    enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:m->p_attn_gqa];
    [enc setBuffer:m->query_rope offset:0 atIndex:0]; [enc setBuffer:m->kcache[a] offset:0 atIndex:1]; [enc setBuffer:m->vcache[a] offset:0 atIndex:2];
    [enc setBuffer:m->agate offset:0 atIndex:3]; [enc setBuffer:m->gated offset:0 atIndex:4];
    [enc setBytes:&head_dim length:4 atIndex:5]; [enc setBytes:&qheads length:4 atIndex:6]; [enc setBytes:&kvheads length:4 atIndex:7]; [enc setBytes:&seq_len length:4 atIndex:8];
    [enc dispatchThreadgroups:MTLSizeMake(qheads, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)]; [enc endEncoding];

    enc_rows(m, cb, &w->o, m->gated, m->branch);
}

int rl_metal_engine_step(rl_engine *e, rl_metal_engine *m, uint32_t token, float *logits,
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
    int ok = 0;
    @autoreleasepool {
        if (!rl_engine_embed_token(e, token, (float *)m->x.contents, error, cap)) goto done;
        memcpy(s->embed, m->x.contents, (size_t)hidden * sizeof(float));
        stats->embed_ms = rl_engine_now_ms() - start;

        const uint64_t bytes0 = 0; (void)bytes0;
        for (uint32_t l = 0; l < in->n_layer; ++l) {
            const rl_layer_tensors *t = &e->layers[l];
            const mlayer *w = &m->layers[l];
            const double l0 = rl_engine_now_ms();
            id<MTLCommandBuffer> cb = [m->queue commandBuffer];
            enc_rms(m, cb, m->x, &w->attn_norm, m->normed, hidden, eps);
            if (t->kind == RL_LAYER_MAP_RECURRENT) enc_recurrent(e, m, cb, t, w);
            else enc_attention(e, m, cb, t, w, s->position);
            {
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:m->p_resid_rms];
                [enc setBuffer:m->x offset:0 atIndex:0]; [enc setBuffer:m->branch offset:0 atIndex:1];
                [enc setBuffer:w->post_norm.buf offset:w->post_norm.off atIndex:2];
                [enc setBuffer:m->resid offset:0 atIndex:3]; [enc setBuffer:m->ffn_in offset:0 atIndex:4];
                [enc setBytes:&hidden length:4 atIndex:5]; [enc setBytes:&eps length:4 atIndex:6];
                [enc dispatchThreads:MTLSizeMake(256, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)]; [enc endEncoding];
            }
            enc_rows(m, cb, &w->router, m->ffn_in, m->router_logits);
            {
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:m->p_sh_scalar];
                [enc setBuffer:w->sh_gate_inp.buf offset:w->sh_gate_inp.off atIndex:0]; [enc setBuffer:m->ffn_in offset:0 atIndex:1];
                [enc setBuffer:m->scalar offset:0 atIndex:2]; [enc setBytes:&hidden length:4 atIndex:3];
                [enc dispatchThreads:MTLSizeMake(256, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)]; [enc endEncoding];
            }
            enc_rows(m, cb, &w->sh_gate, m->ffn_in, m->sh_gate);
            enc_rows(m, cb, &w->sh_up, m->ffn_in, m->sh_up);
            {
                const uint32_t ffn = w->sh_gate.rows;
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:m->p_sh_silu];
                [enc setBuffer:m->sh_gate offset:0 atIndex:0]; [enc setBuffer:m->sh_up offset:0 atIndex:1]; [enc setBuffer:m->sh_act offset:0 atIndex:2];
                [enc setBytes:&ffn length:4 atIndex:3]; enc_1d(enc, m->p_sh_silu, ffn, 64u); [enc endEncoding];
            }
            enc_rows(m, cb, &w->sh_down, m->sh_act, m->sh_out);
            if (!commit_wait(cb, "layer attention/router", &stats->gpu_ms, error, cap)) goto done;
            const double l1 = rl_engine_now_ms();
            if (t->kind == RL_LAYER_MAP_RECURRENT) stats->recurrent_ms += l1 - l0; else stats->attention_ms += l1 - l0;

            if (!rl_native_router_select_softmax_topk((const float *)m->router_logits.contents, experts, topk, ids, weights, probs, error, cap)) goto done;
            memcpy(s->router_ids + (size_t)l * RL_ENGINE_MAX_TOPK, ids, (size_t)topk * sizeof(uint32_t));
            const double l2 = rl_engine_now_ms();
            stats->router_ms += l2 - l1;

            rl_native_metal_telemetry tel;
            const double read_before = m->last_read_ms;
            if (!rl_native_metal_execute_topk(m->experts, &e->expert_map, l, ids, weights, topk, 0u, hidden,
                    (const float *)m->ffn_in.contents, hidden, m->routed_host, hidden, &tel, error, cap)) goto done;
            memcpy(m->routed.contents, m->routed_host, (size_t)hidden * sizeof(float));
            stats->expert_loads = tel.expert_loads; stats->cache_hits = tel.cache_hits; stats->cache_misses = tel.cache_misses;
            stats->ssd_bytes = tel.bytes_read_total; stats->ssd_reads = tel.read_calls_total;
            stats->resident_slots = tel.resident_slots; stats->slot_capacity = tel.slot_capacity;
            stats->routed_gpu_ms += tel.gpu_ms;
            stats->routed_load_ms += tel.read_ms_total - read_before;
            m->last_read_ms = tel.read_ms_total;
            const double l3 = rl_engine_now_ms();
            stats->routed_ms += l3 - l2;

            cb = [m->queue commandBuffer];
            {
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:m->p_scale_add];
                [enc setBuffer:m->resid offset:0 atIndex:0]; [enc setBuffer:m->routed offset:0 atIndex:1]; [enc setBuffer:m->sh_out offset:0 atIndex:2];
                [enc setBuffer:m->scalar offset:0 atIndex:3]; [enc setBuffer:m->x offset:0 atIndex:4]; [enc setBytes:&hidden length:4 atIndex:5];
                enc_1d(enc, m->p_scale_add, hidden, 64u); [enc endEncoding];
            }
            if (!commit_wait(cb, "layer shared/residual", &stats->gpu_ms, error, cap)) goto done;
            stats->shared_ms += rl_engine_now_ms() - l3;
            memcpy(s->layer_out + (size_t)l * hidden, m->x.contents, (size_t)hidden * sizeof(float));
        }
        stats->layers_ms = rl_engine_now_ms() - start - stats->embed_ms;

        const double o0 = rl_engine_now_ms();
        id<MTLCommandBuffer> cb = [m->queue commandBuffer];
        enc_rms(m, cb, m->x, &m->output_norm, m->final_norm, hidden, eps);
        if (logits) enc_rows(m, cb, &m->output, m->final_norm, m->logits);
        if (!commit_wait(cb, "output head", &stats->gpu_ms, error, cap)) goto done;
        memcpy(s->final_norm, m->final_norm.contents, (size_t)hidden * sizeof(float));
        if (logits) memcpy(logits, m->logits.contents, (size_t)in->vocab * sizeof(float));
        stats->output_ms = rl_engine_now_ms() - o0;
        s->position++;
        ok = 1;
    }
done:
    stats->total_ms = rl_engine_now_ms() - start;
    free(probs);
    return ok;
}
