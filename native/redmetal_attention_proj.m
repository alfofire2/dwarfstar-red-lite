#define _FILE_OFFSET_BITS 64
#define _POSIX_C_SOURCE 200809L

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "redlite_native_attention_proj.h"
#include "redlite_native_iq2_xxs.h"

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define IQ2_XXS_BLOCK_BYTES 66u
#define Q4_K_BLOCK_BYTES 144u

static void set_error(char *dst, size_t cap, const char *msg) {
    if (dst && cap) snprintf(dst, cap, "%s", msg ? msg : "unknown full-attention Metal error");
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

static int pread_full(int fd, void *dst, size_t bytes, uint64_t offset, uint64_t *calls) {
    unsigned char *p = (unsigned char *)dst;
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

static size_t iq2_row_bytes(uint32_t ncols) {
    return ncols && ncols % 256u == 0 ? (size_t)(ncols / 256u) * IQ2_XXS_BLOCK_BYTES : 0;
}

static size_t q4_row_bytes(uint32_t ncols) {
    return ncols && ncols % 256u == 0 ? (size_t)(ncols / 256u) * Q4_K_BLOCK_BYTES : 0;
}

static NSString * const kAttentionProjSource = @
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"static inline ushort attn_rd16(device const uchar *p) { return ushort(p[0]) | (ushort(p[1]) << 8); }\n"
"static inline float attn_fp16(device const uchar *p) { return float(as_type<half>(attn_rd16(p))); }\n"
"static inline uchar2 attn_scale_min(uint j, device const uchar *s) {\n"
"    if (j < 4u) return uchar2(s[j] & 63u, s[j + 4u] & 63u);\n"
"    return uchar2((s[j + 4u] & 15u) | ((s[j - 4u] >> 6) << 4),\n"
"                  (s[j + 4u] >> 4) | ((s[j] >> 6) << 4));\n"
"}\n"
"kernel void redlite_attn_input_rms(\n"
"    device const float *x [[buffer(0)]], device const float *w [[buffer(1)]],\n"
"    device float *y [[buffer(2)]], constant uint &n [[buffer(3)]],\n"
"    constant float &eps [[buffer(4)]], uint gid [[thread_position_in_grid]]) {\n"
"    if (gid != 0u) return;\n"
"    float ss = 0.0f;\n"
"    for (uint i = 0; i < n; ++i) ss = fma(x[i], x[i], ss);\n"
"    const float inv = rsqrt(ss / float(n) + eps);\n"
"    for (uint i = 0; i < n; ++i) y[i] = x[i] * inv * w[i];\n"
"}\n"
"kernel void redlite_attn_iq2_rows(\n"
"    device const uchar *weights [[buffer(0)]], constant uint &ncols [[buffer(1)]],\n"
"    device const float *x [[buffer(2)]], device float *out [[buffer(3)]],\n"
"    device const uchar *grid [[buffer(4)]], uint row [[thread_position_in_grid]]) {\n"
"    const uint blocks = ncols / 256u;\n"
"    const ulong row_bytes = ulong(blocks) * 66ul;\n"
"    device const uchar *rowp = weights + ulong(row) * row_bytes;\n"
"    float acc = 0.0f;\n"
"    for (uint ib = 0; ib < blocks; ++ib) {\n"
"        device const uchar *bp = rowp + ulong(ib) * 66ul;\n"
"        const float d = attn_fp16(bp);\n"
"        device const ushort *q = (device const ushort *)(bp + 2ul);\n"
"        for (uint g = 0; g < 8u; ++g) {\n"
"            const uint qi = 4u * g;\n"
"            const uint auxg = uint(q[qi]) | (uint(q[qi + 1u]) << 16);\n"
"            const uint auxs = uint(q[qi + 2u]) | (uint(q[qi + 3u]) << 16);\n"
"            const float db = d * (0.5f + float(auxs >> 28)) * 0.25f;\n"
"            for (uint l = 0; l < 4u; ++l) {\n"
"                const uint grid_index = (auxg >> (8u * l)) & 255u;\n"
"                const uint sign7 = (auxs >> (7u * l)) & 127u;\n"
"                const uint sign8 = sign7 | ((popcount(sign7) & 1u) << 7);\n"
"                device const uchar *gv = grid + grid_index * 8u;\n"
"                const uint xb = ib * 256u + g * 32u + l * 8u;\n"
"                for (uint j = 0; j < 8u; ++j) {\n"
"                    const float sign = (sign8 & (1u << j)) ? -1.0f : 1.0f;\n"
"                    acc += x[xb + j] * (db * float(gv[j]) * sign);\n"
"                }\n"
"            }\n"
"        }\n"
"    }\n"
"    out[row] = acc;\n"
"}\n"
"kernel void redlite_attn_q4_rows(\n"
"    device const uchar *weights [[buffer(0)]], constant uint &ncols [[buffer(1)]],\n"
"    device const float *x [[buffer(2)]], device float *out [[buffer(3)]],\n"
"    uint row [[thread_position_in_grid]]) {\n"
"    const uint blocks = ncols / 256u; const uint row_bytes = blocks * 144u;\n"
"    device const uchar *rp = weights + row * row_bytes; float acc = 0.0f;\n"
"    for (uint ib = 0; ib < blocks; ++ib) {\n"
"        device const uchar *bp = rp + ib * 144u;\n"
"        const float d = attn_fp16(bp); const float dmin = attn_fp16(bp + 2u);\n"
"        device const uchar *scales = bp + 4u; device const uchar *qs = bp + 16u;\n"
"        for (uint g = 0; g < 8u; ++g) {\n"
"            const uchar2 sm = attn_scale_min(g, scales);\n"
"            const float ds = d * float(sm.x); const float dm = dmin * float(sm.y);\n"
"            device const uchar *q = qs + (g >> 1) * 32u; const uint xb = ib * 256u + g * 32u;\n"
"            for (uint l = 0; l < 32u; ++l) {\n"
"                const uchar quant = (g & 1u) ? (q[l] >> 4) : (q[l] & 15u);\n"
"                acc = fma(x[xb + l], ds * float(quant) - dm, acc);\n"
"            }\n"
"        }\n"
"    }\n"
"    out[row] = acc;\n"
"}\n"
"kernel void redlite_attn_qgate_norm(\n"
"    device const float *qgate [[buffer(0)]], device const float *w [[buffer(1)]],\n"
"    device float *query [[buffer(2)]], device float *gate [[buffer(3)]],\n"
"    constant uint &head_dim [[buffer(4)]], constant float &eps [[buffer(5)]],\n"
"    uint head [[thread_position_in_grid]]) {\n"
"    const uint src = head * head_dim * 2u; const uint dst = head * head_dim;\n"
"    float ss = 0.0f;\n"
"    for (uint i = 0; i < head_dim; ++i) ss = fma(qgate[src + i], qgate[src + i], ss);\n"
"    const float inv = rsqrt(ss / float(head_dim) + eps);\n"
"    for (uint i = 0; i < head_dim; ++i) {\n"
"        query[dst + i] = qgate[src + i] * inv * w[i];\n"
"        gate[dst + i] = qgate[src + head_dim + i];\n"
"    }\n"
"}\n"
"kernel void redlite_attn_k_norm(\n"
"    device const float *key_raw [[buffer(0)]], device const float *w [[buffer(1)]],\n"
"    device float *key [[buffer(2)]], constant uint &head_dim [[buffer(3)]],\n"
"    constant float &eps [[buffer(4)]], uint head [[thread_position_in_grid]]) {\n"
"    const uint base = head * head_dim; float ss = 0.0f;\n"
"    for (uint i = 0; i < head_dim; ++i) ss = fma(key_raw[base + i], key_raw[base + i], ss);\n"
"    const float inv = rsqrt(ss / float(head_dim) + eps);\n"
"    for (uint i = 0; i < head_dim; ++i) key[base + i] = key_raw[base + i] * inv * w[i];\n"
"}\n"
"kernel void redlite_attn_rope_cache(\n"
"    device const float *query [[buffer(0)]], device const float *key [[buffer(1)]],\n"
"    device const float *value [[buffer(2)]], device float *query_rope [[buffer(3)]],\n"
"    device float *key_cache [[buffer(4)]], device float *value_cache [[buffer(5)]],\n"
"    constant uint &head_dim [[buffer(6)]], constant uint &query_heads [[buffer(7)]],\n"
"    constant uint &kv_heads [[buffer(8)]], constant uint &position [[buffer(9)]],\n"
"    constant uint &rope_dims [[buffer(10)]], constant float &freq_base [[buffer(11)]],\n"
"    uint head [[thread_position_in_grid]]) {\n"
"    const uint rope_half = rope_dims / 2u;\n"
"    if (head < query_heads) {\n"
"        const uint base = head * head_dim;\n"
"        for (uint i = 0; i < head_dim; ++i) query_rope[base + i] = query[base + i];\n"
"        for (uint i = 0; i < rope_half; ++i) {\n"
"            const float theta = float(position) * pow(freq_base, -2.0f * float(i) / float(rope_dims));\n"
"            const float c = cos(theta); const float s = sin(theta);\n"
"            const float x0 = query[base + i]; const float x1 = query[base + rope_half + i];\n"
"            query_rope[base + i] = x0 * c - x1 * s;\n"
"            query_rope[base + rope_half + i] = x0 * s + x1 * c;\n"
"        }\n"
"    }\n"
"    if (head < kv_heads) {\n"
"        const uint src = head * head_dim;\n"
"        const uint dst = (position * kv_heads + head) * head_dim;\n"
"        for (uint i = 0; i < head_dim; ++i) { key_cache[dst + i] = key[src + i]; value_cache[dst + i] = value[src + i]; }\n"
"        for (uint i = 0; i < rope_half; ++i) {\n"
"            const float theta = float(position) * pow(freq_base, -2.0f * float(i) / float(rope_dims));\n"
"            const float c = cos(theta); const float s = sin(theta);\n"
"            const float x0 = key[src + i]; const float x1 = key[src + rope_half + i];\n"
"            key_cache[dst + i] = x0 * c - x1 * s;\n"
"            key_cache[dst + rope_half + i] = x0 * s + x1 * c;\n"
"        }\n"
"    }\n"
"}\n"
"kernel void redlite_attn_gqa(\n"
"    device const float *query [[buffer(0)]], device const float *key_cache [[buffer(1)]],\n"
"    device const float *value_cache [[buffer(2)]], device const float *gate [[buffer(3)]],\n"
"    device float *attention [[buffer(4)]], device float *gated [[buffer(5)]],\n"
"    constant uint &head_dim [[buffer(6)]], constant uint &query_heads [[buffer(7)]],\n"
"    constant uint &kv_heads [[buffer(8)]], constant uint &seq_len [[buffer(9)]],\n"
"    uint head [[thread_position_in_grid]]) {\n"
"    if (head >= query_heads || seq_len == 0u || seq_len > 64u) return;\n"
"    const uint qbase = head * head_dim; const uint kv_head = head / (query_heads / kv_heads);\n"
"    const float scale = rsqrt(float(head_dim)); float scores[64]; float max_score = -INFINITY;\n"
"    for (uint pos = 0; pos < seq_len; ++pos) {\n"
"        const uint kbase = (pos * kv_heads + kv_head) * head_dim; float dot = 0.0f;\n"
"        for (uint i = 0; i < head_dim; ++i) dot = fma(query[qbase + i], key_cache[kbase + i], dot);\n"
"        scores[pos] = dot * scale; max_score = max(max_score, scores[pos]);\n"
"    }\n"
"    float denom = 0.0f;\n"
"    for (uint pos = 0; pos < seq_len; ++pos) { scores[pos] = exp(scores[pos] - max_score); denom += scores[pos]; }\n"
"    const float inv_denom = 1.0f / denom;\n"
"    for (uint i = 0; i < head_dim; ++i) {\n"
"        float acc = 0.0f;\n"
"        for (uint pos = 0; pos < seq_len; ++pos) {\n"
"            const uint vbase = (pos * kv_heads + kv_head) * head_dim;\n"
"            acc = fma(scores[pos] * inv_denom, value_cache[vbase + i], acc);\n"
"        }\n"
"        attention[qbase + i] = acc;\n"
"        gated[qbase + i] = acc / (1.0f + exp(-gate[qbase + i]));\n"
"    }\n"
"}\n";

static id<MTLComputePipelineState> make_pipeline(id<MTLDevice> device, id<MTLLibrary> lib,
        NSString *name, char *error, size_t cap) {
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    if (!fn) { snprintf(error, cap, "missing Metal function %s", name.UTF8String); return nil; }
    NSError *pe = nil;
    id<MTLComputePipelineState> pipe = [device newComputePipelineStateWithFunction:fn error:&pe];
    if (!pipe) {
        const char *detail = pe.localizedDescription.UTF8String;
        snprintf(error, cap, "pipeline %s failed: %s", name.UTF8String, detail ? detail : "unknown");
    }
    return pipe;
}

static void encode_input_rms(id<MTLCommandBuffer> cb, id<MTLComputePipelineState> pipe,
        id<MTLBuffer> x, id<MTLBuffer> w, id<MTLBuffer> out, uint32_t n, float eps) {
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pipe];
    [enc setBuffer:x offset:0 atIndex:0];
    [enc setBuffer:w offset:0 atIndex:1];
    [enc setBuffer:out offset:0 atIndex:2];
    [enc setBytes:&n length:sizeof(n) atIndex:3];
    [enc setBytes:&eps length:sizeof(eps) atIndex:4];
    [enc dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [enc endEncoding];
}

static void encode_rows(id<MTLCommandBuffer> cb, id<MTLComputePipelineState> pipe,
        id<MTLBuffer> weights, uint32_t ncols, id<MTLBuffer> x, id<MTLBuffer> out,
        uint32_t rows, id<MTLBuffer> grid) {
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pipe];
    [enc setBuffer:weights offset:0 atIndex:0];
    [enc setBytes:&ncols length:sizeof(ncols) atIndex:1];
    [enc setBuffer:x offset:0 atIndex:2];
    [enc setBuffer:out offset:0 atIndex:3];
    if (grid) [enc setBuffer:grid offset:0 atIndex:4];
    const NSUInteger tg = MIN((NSUInteger)64, pipe.maxTotalThreadsPerThreadgroup);
    [enc dispatchThreads:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg ? tg : 1, 1, 1)];
    [enc endEncoding];
}

static void encode_qgate_norm(id<MTLCommandBuffer> cb, id<MTLComputePipelineState> pipe,
        id<MTLBuffer> qgate, id<MTLBuffer> w, id<MTLBuffer> query, id<MTLBuffer> gate,
        uint32_t head_dim, uint32_t heads, float eps) {
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pipe];
    [enc setBuffer:qgate offset:0 atIndex:0];
    [enc setBuffer:w offset:0 atIndex:1];
    [enc setBuffer:query offset:0 atIndex:2];
    [enc setBuffer:gate offset:0 atIndex:3];
    [enc setBytes:&head_dim length:sizeof(head_dim) atIndex:4];
    [enc setBytes:&eps length:sizeof(eps) atIndex:5];
    const NSUInteger tg = MIN((NSUInteger)32, pipe.maxTotalThreadsPerThreadgroup);
    [enc dispatchThreads:MTLSizeMake(heads, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg ? tg : 1, 1, 1)];
    [enc endEncoding];
}

static void encode_k_norm(id<MTLCommandBuffer> cb, id<MTLComputePipelineState> pipe,
        id<MTLBuffer> raw, id<MTLBuffer> w, id<MTLBuffer> out,
        uint32_t head_dim, uint32_t heads, float eps) {
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pipe];
    [enc setBuffer:raw offset:0 atIndex:0];
    [enc setBuffer:w offset:0 atIndex:1];
    [enc setBuffer:out offset:0 atIndex:2];
    [enc setBytes:&head_dim length:sizeof(head_dim) atIndex:3];
    [enc setBytes:&eps length:sizeof(eps) atIndex:4];
    const NSUInteger tg = MIN((NSUInteger)32, pipe.maxTotalThreadsPerThreadgroup);
    [enc dispatchThreads:MTLSizeMake(heads, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg ? tg : 1, 1, 1)];
    [enc endEncoding];
}

static void encode_rope_cache(id<MTLCommandBuffer> cb, id<MTLComputePipelineState> pipe,
        id<MTLBuffer> query, id<MTLBuffer> key, id<MTLBuffer> value, id<MTLBuffer> query_rope,
        id<MTLBuffer> key_cache, id<MTLBuffer> value_cache, uint32_t head_dim,
        uint32_t query_heads, uint32_t kv_heads, uint32_t position,
        uint32_t rope_dims, float rope_freq_base) {
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pipe];
    [enc setBuffer:query offset:0 atIndex:0];
    [enc setBuffer:key offset:0 atIndex:1];
    [enc setBuffer:value offset:0 atIndex:2];
    [enc setBuffer:query_rope offset:0 atIndex:3];
    [enc setBuffer:key_cache offset:0 atIndex:4];
    [enc setBuffer:value_cache offset:0 atIndex:5];
    [enc setBytes:&head_dim length:sizeof(head_dim) atIndex:6];
    [enc setBytes:&query_heads length:sizeof(query_heads) atIndex:7];
    [enc setBytes:&kv_heads length:sizeof(kv_heads) atIndex:8];
    [enc setBytes:&position length:sizeof(position) atIndex:9];
    [enc setBytes:&rope_dims length:sizeof(rope_dims) atIndex:10];
    [enc setBytes:&rope_freq_base length:sizeof(rope_freq_base) atIndex:11];
    const NSUInteger tg = MIN((NSUInteger)32, pipe.maxTotalThreadsPerThreadgroup);
    [enc dispatchThreads:MTLSizeMake(query_heads, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg ? tg : 1, 1, 1)];
    [enc endEncoding];
}

static void encode_gqa(id<MTLCommandBuffer> cb, id<MTLComputePipelineState> pipe,
        id<MTLBuffer> query, id<MTLBuffer> key_cache, id<MTLBuffer> value_cache,
        id<MTLBuffer> gate, id<MTLBuffer> attention, id<MTLBuffer> gated,
        uint32_t head_dim, uint32_t query_heads, uint32_t kv_heads, uint32_t seq_len) {
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pipe];
    [enc setBuffer:query offset:0 atIndex:0];
    [enc setBuffer:key_cache offset:0 atIndex:1];
    [enc setBuffer:value_cache offset:0 atIndex:2];
    [enc setBuffer:gate offset:0 atIndex:3];
    [enc setBuffer:attention offset:0 atIndex:4];
    [enc setBuffer:gated offset:0 atIndex:5];
    [enc setBytes:&head_dim length:sizeof(head_dim) atIndex:6];
    [enc setBytes:&query_heads length:sizeof(query_heads) atIndex:7];
    [enc setBytes:&kv_heads length:sizeof(kv_heads) atIndex:8];
    [enc setBytes:&seq_len length:sizeof(seq_len) atIndex:9];
    const NSUInteger tg = MIN((NSUInteger)32, pipe.maxTotalThreadsPerThreadgroup);
    [enc dispatchThreads:MTLSizeMake(query_heads, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg ? tg : 1, 1, 1)];
    [enc endEncoding];
}

int rl_attention_gpu_execute(
        const char *model_path,
        const rl_attn_proj_tensor_info t[RL_ATTN_PROJ_TENSOR_COUNT],
        const float *input, uint32_t input_count, float rms_eps,
        uint32_t position, uint32_t rope_dims, float rope_freq_base,
        float *key_cache, float *value_cache, uint32_t cache_float_count,
        float *input_norm_out, uint32_t input_norm_out_count,
        float *query_out, uint32_t query_out_count,
        float *gate_out, uint32_t gate_out_count,
        float *key_out, uint32_t key_out_count,
        float *value_out, uint32_t value_out_count,
        float *attention_out, uint32_t attention_out_count,
        float *gated_out, uint32_t gated_out_count,
        float *output, uint32_t output_count,
        rl_attn_proj_telemetry *telemetry, char *error, size_t error_cap) {
    if (!model_path || !t || !input || !key_cache || !value_cache || !input_norm_out || !query_out ||
        !gate_out || !key_out || !value_out || !attention_out || !gated_out || !output ||
        t[0].kind != RL_ATTN_PROJ_INPUT_NORM || t[1].kind != RL_ATTN_PROJ_Q_GATE ||
        t[2].kind != RL_ATTN_PROJ_K || t[3].kind != RL_ATTN_PROJ_V ||
        t[4].kind != RL_ATTN_PROJ_Q_NORM || t[5].kind != RL_ATTN_PROJ_K_NORM || t[6].kind != RL_ATTN_PROJ_OUTPUT ||
        t[0].ggml_type != 0u || t[1].ggml_type != 16u || t[2].ggml_type != 12u ||
        t[3].ggml_type != 12u || t[4].ggml_type != 0u || t[5].ggml_type != 0u || t[6].ggml_type != 12u ||
        t[0].n_dims != 1u || t[1].n_dims != 2u || t[2].n_dims != 2u ||
        t[3].n_dims != 2u || t[4].n_dims != 1u || t[5].n_dims != 1u || t[6].n_dims != 2u ||
        !t[0].shape[0] || t[0].shape[0] > UINT32_MAX || !t[4].shape[0] || t[4].shape[0] > UINT32_MAX) {
        set_error(error, error_cap, "invalid full-attention Metal arguments/layout"); return 0;
    }
    const uint32_t hidden = (uint32_t)t[0].shape[0];
    const uint32_t head_dim = (uint32_t)t[4].shape[0];
    if (t[5].shape[0] != head_dim || t[1].shape[0] != hidden || t[2].shape[0] != hidden ||
        t[3].shape[0] != hidden || !t[1].shape[1] || t[1].shape[1] > UINT32_MAX ||
        !t[2].shape[1] || t[2].shape[1] > UINT32_MAX || t[3].shape[1] != t[2].shape[1] ||
        t[1].shape[1] % ((uint64_t)head_dim * 2u) != 0 || t[2].shape[1] % head_dim != 0) {
        set_error(error, error_cap, "full-attention projection Metal shape mismatch"); return 0;
    }
    const uint32_t qgate_rows = (uint32_t)t[1].shape[1];
    const uint32_t kv_rows = (uint32_t)t[2].shape[1];
    const uint32_t query_count = qgate_rows / 2u;
    const uint32_t query_heads = query_count / head_dim;
    const uint32_t kv_heads = kv_rows / head_dim;
    if (!query_heads || !kv_heads || query_heads % kv_heads != 0 || !rope_dims || rope_dims > head_dim ||
        rope_dims % 2u != 0 || position >= RL_ATTN_MAX_CONTEXT || !isfinite(rope_freq_base) || rope_freq_base <= 0.0f ||
        t[6].shape[0] != query_count || t[6].shape[1] != hidden) {
        set_error(error, error_cap, "unsupported full-attention head/RoPE/output geometry"); return 0;
    }
    const uint64_t required_cache = (uint64_t)(position + 1u) * kv_rows;
    if (input_count != hidden || input_norm_out_count < hidden || query_out_count < query_count ||
        gate_out_count < query_count || key_out_count < kv_rows || value_out_count < kv_rows ||
        attention_out_count < query_count || gated_out_count < query_count || output_count < hidden ||
        required_cache > cache_float_count) {
        set_error(error, error_cap, "full-attention Metal buffer mismatch"); return 0;
    }
    const size_t iq2_rb = iq2_row_bytes(hidden);
    const size_t q4_in_rb = q4_row_bytes(hidden);
    const size_t q4_out_rb = q4_row_bytes(query_count);
    if (!iq2_rb || !q4_in_rb || !q4_out_rb || qgate_rows > SIZE_MAX / iq2_rb ||
        kv_rows > SIZE_MAX / q4_in_rb || hidden > SIZE_MAX / q4_out_rb) {
        set_error(error, error_cap, "unsupported full-attention quantized row layout"); return 0;
    }
    const size_t weight_bytes[RL_ATTN_PROJ_TENSOR_COUNT] = {
        (size_t)hidden * sizeof(float), (size_t)qgate_rows * iq2_rb,
        (size_t)kv_rows * q4_in_rb, (size_t)kv_rows * q4_in_rb,
        (size_t)head_dim * sizeof(float), (size_t)head_dim * sizeof(float),
        (size_t)hidden * q4_out_rb,
    };
    for (uint32_t i = 0; i < RL_ATTN_PROJ_TENSOR_COUNT; ++i) if (t[i].tensor_span_bytes < weight_bytes[i]) {
        set_error(error, error_cap, "full-attention tensor span is smaller than its payload"); return 0;
    }

    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device || !device.hasUnifiedMemory) { set_error(error, error_cap, "full-attention projection requires Apple unified memory"); return 0; }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) { set_error(error, error_cap, "failed to create full-attention Metal queue"); return 0; }
        NSError *le = nil;
        id<MTLLibrary> lib = [device newLibraryWithSource:kAttentionProjSource options:nil error:&le];
        if (!lib) {
            const char *detail = le.localizedDescription.UTF8String;
            snprintf(error, error_cap, "full-attention Metal source compile failed: %s", detail ? detail : "unknown"); return 0;
        }
        id<MTLComputePipelineState> input_rms = make_pipeline(device, lib, @"redlite_attn_input_rms", error, error_cap);
        id<MTLComputePipelineState> iq2 = make_pipeline(device, lib, @"redlite_attn_iq2_rows", error, error_cap);
        id<MTLComputePipelineState> q4 = make_pipeline(device, lib, @"redlite_attn_q4_rows", error, error_cap);
        id<MTLComputePipelineState> qgate_norm = make_pipeline(device, lib, @"redlite_attn_qgate_norm", error, error_cap);
        id<MTLComputePipelineState> k_norm = make_pipeline(device, lib, @"redlite_attn_k_norm", error, error_cap);
        id<MTLComputePipelineState> rope_cache = make_pipeline(device, lib, @"redlite_attn_rope_cache", error, error_cap);
        id<MTLComputePipelineState> gqa = make_pipeline(device, lib, @"redlite_attn_gqa", error, error_cap);
        if (!input_rms || !iq2 || !q4 || !qgate_norm || !k_norm || !rope_cache || !gqa) return 0;

        id<MTLBuffer> weights[RL_ATTN_PROJ_TENSOR_COUNT];
        for (uint32_t i = 0; i < RL_ATTN_PROJ_TENSOR_COUNT; ++i) {
            weights[i] = [device newBufferWithLength:weight_bytes[i] options:MTLResourceStorageModeShared];
            if (!weights[i]) { set_error(error, error_cap, "failed to allocate full-attention weight buffer"); return 0; }
        }
        id<MTLBuffer> x = [device newBufferWithBytes:input length:(size_t)hidden * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> input_norm = [device newBufferWithLength:(size_t)hidden * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> qgate_raw = [device newBufferWithLength:(size_t)qgate_rows * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> k_raw = [device newBufferWithLength:(size_t)kv_rows * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> query = [device newBufferWithLength:(size_t)query_count * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> gate = [device newBufferWithLength:(size_t)query_count * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> key = [device newBufferWithLength:(size_t)kv_rows * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> value = [device newBufferWithLength:(size_t)kv_rows * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> query_rope = [device newBufferWithLength:(size_t)query_count * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> key_cache_buf = [device newBufferWithBytes:key_cache length:(size_t)cache_float_count * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> value_cache_buf = [device newBufferWithBytes:value_cache length:(size_t)cache_float_count * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> attention = [device newBufferWithLength:(size_t)query_count * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> gated = [device newBufferWithLength:(size_t)query_count * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> output_buf = [device newBufferWithLength:(size_t)hidden * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> grid = [device newBufferWithLength:RL_IQ2_XXS_GRID_COUNT options:MTLResourceStorageModeShared];
        if (!x || !input_norm || !qgate_raw || !k_raw || !query || !gate || !key || !value || !query_rope ||
            !key_cache_buf || !value_cache_buf || !attention || !gated || !output_buf || !grid) {
            set_error(error, error_cap, "failed to allocate full-attention scratch buffer"); return 0;
        }
        if (!rl_native_iq2_xxs_build_grid((uint8_t *)grid.contents, error, error_cap)) return 0;

        const int fd = open(model_path, O_RDONLY);
        if (fd < 0) { snprintf(error, error_cap, "open GGUF for full-attention Metal failed: %s", strerror(errno)); return 0; }
        uint64_t calls = 0, bytes = 0;
        const double r0 = now_ms();
        for (uint32_t i = 0; i < RL_ATTN_PROJ_TENSOR_COUNT; ++i) {
            if (!pread_full(fd, weights[i].contents, weight_bytes[i], t[i].tensor_offset, &calls)) {
                close(fd); set_error(error, error_cap, "pread full-attention Metal weights failed"); return 0;
            }
            bytes += weight_bytes[i];
        }
        const double r1 = now_ms();
        close(fd);

        id<MTLCommandBuffer> cb = [queue commandBuffer];
        if (!cb) { set_error(error, error_cap, "failed to create full-attention command buffer"); return 0; }
        const double c0 = now_ms();
        encode_input_rms(cb, input_rms, x, weights[0], input_norm, hidden, rms_eps);
        encode_rows(cb, iq2, weights[1], hidden, input_norm, qgate_raw, qgate_rows, grid);
        encode_rows(cb, q4, weights[2], hidden, input_norm, k_raw, kv_rows, nil);
        encode_rows(cb, q4, weights[3], hidden, input_norm, value, kv_rows, nil);
        encode_qgate_norm(cb, qgate_norm, qgate_raw, weights[4], query, gate, head_dim, query_heads, rms_eps);
        encode_k_norm(cb, k_norm, k_raw, weights[5], key, head_dim, kv_heads, rms_eps);
        encode_rope_cache(cb, rope_cache, query, key, value, query_rope, key_cache_buf, value_cache_buf,
            head_dim, query_heads, kv_heads, position, rope_dims, rope_freq_base);
        encode_gqa(cb, gqa, query_rope, key_cache_buf, value_cache_buf, gate, attention, gated,
            head_dim, query_heads, kv_heads, position + 1u);
        encode_rows(cb, q4, weights[6], query_count, gated, output_buf, hidden, nil);
        [cb commit];
        [cb waitUntilCompleted];
        const double c1 = now_ms();
        if (cb.status != MTLCommandBufferStatusCompleted || cb.error) {
            const char *detail = cb.error.localizedDescription.UTF8String;
            snprintf(error, error_cap, "full-attention Metal command failed: %s", detail ? detail : "unknown"); return 0;
        }

        memcpy(input_norm_out, input_norm.contents, (size_t)hidden * sizeof(float));
        memcpy(query_out, query_rope.contents, (size_t)query_count * sizeof(float));
        memcpy(gate_out, gate.contents, (size_t)query_count * sizeof(float));
        memcpy(key_out, (const uint8_t *)key_cache_buf.contents + (size_t)position * kv_rows * sizeof(float),
            (size_t)kv_rows * sizeof(float));
        memcpy(value_out, value.contents, (size_t)kv_rows * sizeof(float));
        memcpy(key_cache, key_cache_buf.contents, (size_t)cache_float_count * sizeof(float));
        memcpy(value_cache, value_cache_buf.contents, (size_t)cache_float_count * sizeof(float));
        memcpy(attention_out, attention.contents, (size_t)query_count * sizeof(float));
        memcpy(gated_out, gated.contents, (size_t)query_count * sizeof(float));
        memcpy(output, output_buf.contents, (size_t)hidden * sizeof(float));
        if (telemetry) {
            telemetry->bytes_read = bytes;
            telemetry->read_calls = calls;
            telemetry->read_ms = r1 - r0;
            telemetry->compute_ms = c1 - c0;
        }
    }
    if (error && error_cap) error[0] = '\0';
    return 1;
}
