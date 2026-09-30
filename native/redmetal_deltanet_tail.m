#define _POSIX_C_SOURCE 200809L

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "redlite_native_deltanet_tail.h"

#include <stdio.h>
#include <string.h>
#include <time.h>

static void set_error(char *dst, size_t cap, const char *msg) {
    if (dst && cap) snprintf(dst, cap, "%s", msg ? msg : "unknown DeltaNet tail Metal error");
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

static NSString * const kSource = @
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"static inline ushort rd16(device const uchar *p) { return ushort(p[0]) | (ushort(p[1]) << 8); }\n"
"static inline float fp16(device const uchar *p) { return float(as_type<half>(rd16(p))); }\n"
"static inline uchar2 scale_min(uint j, device const uchar *s) {\n"
"    if (j < 4) return uchar2(s[j] & 63, s[j + 4] & 63);\n"
"    return uchar2((s[j + 4] & 15) | ((s[j - 4] >> 6) << 4),\n"
"                  (s[j + 4] >> 4) | ((s[j] >> 6) << 4));\n"
"}\n"
"kernel void dn_tail_norm(device const float *core [[buffer(0)]],\n"
"    device const float *z [[buffer(1)]], device const float *w [[buffer(2)]],\n"
"    device float *out [[buffer(3)]], constant float &eps [[buffer(4)]],\n"
"    constant uint &head_dim [[buffer(5)]], constant uint &heads [[buffer(6)]],\n"
"    uint h [[thread_position_in_grid]]) {\n"
"    if (h >= heads) return; const uint base = h * head_dim; float ss = 0.0f;\n"
"    for (uint i = 0; i < head_dim; ++i) ss = fma(core[base+i], core[base+i], ss);\n"
"    const float inv = rsqrt(ss / float(head_dim) + eps);\n"
"    for (uint i = 0; i < head_dim; ++i) {\n"
"        const float g = z[base+i]; const float silu = g / (1.0f + exp(-g));\n"
"        out[base+i] = core[base+i] * inv * w[i] * silu;\n"
"    }\n"
"}\n"
"kernel void dn_tail_q4(device const uchar *weights [[buffer(0)]],\n"
"    device const float *input [[buffer(1)]], device float *out [[buffer(2)]],\n"
"    constant uint &ncols [[buffer(3)]], constant uint &nrows [[buffer(4)]],\n"
"    uint row [[thread_position_in_grid]]) {\n"
"    if (row >= nrows) return; const uint blocks = ncols / 256; const uint row_bytes = blocks * 144;\n"
"    device const uchar *rp = weights + row * row_bytes; float acc = 0.0f;\n"
"    for (uint ib = 0; ib < blocks; ++ib) {\n"
"        device const uchar *bp = rp + ib * 144; const float d = fp16(bp); const float dmin = fp16(bp+2);\n"
"        device const uchar *scales = bp + 4; device const uchar *qs = bp + 16;\n"
"        for (uint g = 0; g < 8; ++g) {\n"
"            const uchar2 sm = scale_min(g, scales); const float ds = d * float(sm.x); const float dm = dmin * float(sm.y);\n"
"            device const uchar *q = qs + (g >> 1) * 32; const uint xb = ib * 256 + g * 32;\n"
"            for (uint l = 0; l < 32; ++l) {\n"
"                const uchar quant = (g & 1) ? (q[l] >> 4) : (q[l] & 15);\n"
"                acc = fma(input[xb+l], ds * float(quant) - dm, acc);\n"
"            }\n"
"        }\n"
"    }\n"
"    out[row] = acc;\n"
"}\n";

static id<MTLComputePipelineState> make_pipeline(id<MTLDevice> dev, id<MTLLibrary> lib, NSString *name,
        char *error, size_t cap) {
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    if (!fn) { snprintf(error, cap, "missing Metal function %s", name.UTF8String); return nil; }
    NSError *pe = nil;
    id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:fn error:&pe];
    if (!p) snprintf(error, cap, "pipeline %s failed: %s", name.UTF8String,
        pe.localizedDescription.UTF8String ?: "unknown");
    return p;
}

static void dispatch_1d(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p, NSUInteger n) {
    NSUInteger tg = MIN((NSUInteger)128, p.maxTotalThreadsPerThreadgroup);
    if (!tg) tg = 1;
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
}

int rl_deltanet_tail_gpu_execute(
        const float *core,
        const float *z,
        const float *norm_w,
        const uint8_t *q4_out,
        size_t q4_out_bytes,
        float rms_eps,
        uint32_t head_dim,
        uint32_t value_heads,
        uint32_t hidden,
        float *normalized_gated,
        float *output,
        rl_dn_tail_telemetry *telemetry,
        char *error,
        size_t error_cap) {
    if (!core || !z || !norm_w || !q4_out || !q4_out_bytes || !head_dim || !value_heads || !hidden ||
        !normalized_gated || !output) {
        set_error(error, error_cap, "invalid DeltaNet tail Metal arguments");
        return 0;
    }
    const uint32_t inner = head_dim * value_heads;
    if (inner % 256u) { set_error(error, error_cap, "DeltaNet tail inner dimension must be divisible by 256"); return 0; }
    const size_t row_bytes = (size_t)(inner / 256u) * 144u;
    if (q4_out_bytes < row_bytes * hidden) { set_error(error, error_cap, "Q4_K output buffer is too small"); return 0; }

    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device || !device.hasUnifiedMemory) { set_error(error, error_cap, "DeltaNet tail requires Apple unified memory"); return 0; }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        NSError *le = nil;
        id<MTLLibrary> lib = [device newLibraryWithSource:kSource options:nil error:&le];
        if (!lib) {
            snprintf(error, error_cap, "DeltaNet tail Metal compile failed: %s", le.localizedDescription.UTF8String ?: "unknown");
            return 0;
        }
        id<MTLComputePipelineState> p_norm = make_pipeline(device, lib, @"dn_tail_norm", error, error_cap);
        id<MTLComputePipelineState> p_q4 = make_pipeline(device, lib, @"dn_tail_q4", error, error_cap);
        if (!p_norm || !p_q4) return 0;

        const size_t inner_bytes = (size_t)inner * sizeof(float);
        id<MTLBuffer> b_core = [device newBufferWithBytes:core length:inner_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_z = [device newBufferWithBytes:z length:inner_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_w = [device newBufferWithBytes:norm_w length:(size_t)head_dim*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_q4 = [device newBufferWithBytes:q4_out length:q4_out_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_ng = [device newBufferWithLength:inner_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_out = [device newBufferWithLength:(size_t)hidden*sizeof(float) options:MTLResourceStorageModeShared];
        if (!b_core || !b_z || !b_w || !b_q4 || !b_ng || !b_out) {
            set_error(error, error_cap, "out of memory for DeltaNet tail Metal buffers");
            return 0;
        }

        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:p_norm];
        [enc setBuffer:b_core offset:0 atIndex:0];
        [enc setBuffer:b_z offset:0 atIndex:1];
        [enc setBuffer:b_w offset:0 atIndex:2];
        [enc setBuffer:b_ng offset:0 atIndex:3];
        [enc setBytes:&rms_eps length:sizeof(rms_eps) atIndex:4];
        [enc setBytes:&head_dim length:sizeof(head_dim) atIndex:5];
        [enc setBytes:&value_heads length:sizeof(value_heads) atIndex:6];
        dispatch_1d(enc, p_norm, value_heads);
        [enc endEncoding];

        enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:p_q4];
        [enc setBuffer:b_q4 offset:0 atIndex:0];
        [enc setBuffer:b_ng offset:0 atIndex:1];
        [enc setBuffer:b_out offset:0 atIndex:2];
        [enc setBytes:&inner length:sizeof(inner) atIndex:3];
        [enc setBytes:&hidden length:sizeof(hidden) atIndex:4];
        dispatch_1d(enc, p_q4, hidden);
        [enc endEncoding];

        const double t0 = now_ms();
        [cb commit];
        [cb waitUntilCompleted];
        const double t1 = now_ms();
        if (cb.status != MTLCommandBufferStatusCompleted) {
            snprintf(error, error_cap, "DeltaNet tail command buffer failed: %s", cb.error.localizedDescription.UTF8String ?: "unknown");
            return 0;
        }
        memcpy(normalized_gated, b_ng.contents, inner_bytes);
        memcpy(output, b_out.contents, (size_t)hidden*sizeof(float));
        if (telemetry) telemetry->compute_ms = t1 - t0;
    }
    if (error && error_cap) error[0] = '\0';
    return 1;
}
