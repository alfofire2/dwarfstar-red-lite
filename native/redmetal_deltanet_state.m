#define _POSIX_C_SOURCE 200809L

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "redlite_native_deltanet_state.h"

#include <stdio.h>
#include <string.h>
#include <time.h>

static void set_error(char *dst, size_t cap, const char *msg) {
    if (dst && cap) snprintf(dst, cap, "%s", msg ? msg : "unknown DeltaNet state Metal error");
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

static NSString * const kSource = @
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"kernel void dn_state_decay(device const float *prev [[buffer(0)]],\n"
"    device const float *gate [[buffer(1)]], device float *state [[buffer(2)]],\n"
"    constant uint &state_size [[buffer(3)]], constant uint &value_heads [[buffer(4)]],\n"
"    uint gid [[thread_position_in_grid]]) {\n"
"    const uint matrix = state_size * state_size; const uint total = value_heads * matrix;\n"
"    if (gid >= total) return; const uint h = gid / matrix;\n"
"    state[gid] = prev[gid] * exp(gate[h]);\n"
"}\n"
"kernel void dn_state_delta(device const float *state [[buffer(0)]],\n"
"    device const float *k [[buffer(1)]], device const float *v [[buffer(2)]],\n"
"    device const float *beta [[buffer(3)]], device float *delta [[buffer(4)]],\n"
"    constant uint &state_size [[buffer(5)]], constant uint &key_heads [[buffer(6)]],\n"
"    constant uint &value_heads [[buffer(7)]], uint gid [[thread_position_in_grid]]) {\n"
"    const uint total = value_heads * state_size; if (gid >= total) return;\n"
"    const uint h = gid / state_size; const uint j = gid - h * state_size;\n"
"    const uint kh = h / (value_heads / key_heads); const uint sb = h * state_size * state_size + j * state_size;\n"
"    const uint kb = kh * state_size; float sum = 0.0f;\n"
"    for (uint i = 0; i < state_size; ++i) sum = fma(state[sb + i], k[kb + i], sum);\n"
"    delta[gid] = (v[gid] - sum) * beta[h];\n"
"}\n"
"kernel void dn_state_update(device float *state [[buffer(0)]],\n"
"    device const float *k [[buffer(1)]], device const float *delta [[buffer(2)]],\n"
"    constant uint &state_size [[buffer(3)]], constant uint &key_heads [[buffer(4)]],\n"
"    constant uint &value_heads [[buffer(5)]], uint gid [[thread_position_in_grid]]) {\n"
"    const uint matrix = state_size * state_size; const uint total = value_heads * matrix;\n"
"    if (gid >= total) return; const uint h = gid / matrix; const uint rem = gid - h * matrix;\n"
"    const uint j = rem / state_size; const uint i = rem - j * state_size; const uint kh = h / (value_heads / key_heads);\n"
"    state[gid] += k[kh * state_size + i] * delta[h * state_size + j];\n"
"}\n"
"kernel void dn_state_output(device const float *state [[buffer(0)]],\n"
"    device const float *q [[buffer(1)]], device float *out [[buffer(2)]],\n"
"    constant uint &state_size [[buffer(3)]], constant uint &key_heads [[buffer(4)]],\n"
"    constant uint &value_heads [[buffer(5)]], uint gid [[thread_position_in_grid]]) {\n"
"    const uint total = value_heads * state_size; if (gid >= total) return;\n"
"    const uint h = gid / state_size; const uint j = gid - h * state_size; const uint kh = h / (value_heads / key_heads);\n"
"    const uint sb = h * state_size * state_size + j * state_size; const uint qb = kh * state_size;\n"
"    float sum = 0.0f; for (uint i = 0; i < state_size; ++i) sum = fma(state[sb + i], q[qb + i], sum);\n"
"    out[gid] = sum * (1.0f / sqrt((float)state_size));\n"
"}\n";

static id<MTLComputePipelineState> pipeline(id<MTLDevice> dev, id<MTLLibrary> lib, NSString *name, char *error, size_t cap) {
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    if (!fn) {
        snprintf(error, cap, "missing Metal function %s", name.UTF8String);
        return nil;
    }
    NSError *pe = nil;
    id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:fn error:&pe];
    if (!p) {
        const char *d = pe.localizedDescription.UTF8String;
        snprintf(error, cap, "pipeline %s failed: %s", name.UTF8String, d ? d : "unknown");
    }
    return p;
}

static void dispatch_1d(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p, NSUInteger n) {
    const NSUInteger tg = MIN((NSUInteger)128, p.maxTotalThreadsPerThreadgroup);
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg ? tg : 1, 1, 1)];
}

int rl_deltanet_state_gpu_execute(
        const float *q,
        const float *k,
        const float *v,
        const float *gate,
        const float *beta,
        const float *prev_state,
        uint32_t state_size,
        uint32_t key_heads,
        uint32_t value_heads,
        float *delta,
        float *next_state,
        float *output,
        rl_dn_state_telemetry *telemetry,
        char *error,
        size_t error_cap) {
    if (!q || !k || !v || !gate || !beta || !prev_state || !delta || !next_state || !output ||
        !state_size || !key_heads || !value_heads || value_heads % key_heads != 0u) {
        set_error(error, error_cap, "invalid DeltaNet state Metal arguments");
        return 0;
    }

    const size_t qk_count = (size_t)key_heads * state_size;
    const size_t vec_count = (size_t)value_heads * state_size;
    const size_t state_count = (size_t)value_heads * state_size * state_size;

    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device || !device.hasUnifiedMemory) {
            set_error(error, error_cap, "DeltaNet state requires Apple unified memory");
            return 0;
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        NSError *le = nil;
        id<MTLLibrary> lib = [device newLibraryWithSource:kSource options:nil error:&le];
        if (!lib) {
            const char *d = le.localizedDescription.UTF8String;
            snprintf(error, error_cap, "DeltaNet state Metal compile failed: %s", d ? d : "unknown");
            return 0;
        }
        id<MTLComputePipelineState> p_decay = pipeline(device, lib, @"dn_state_decay", error, error_cap);
        id<MTLComputePipelineState> p_delta = pipeline(device, lib, @"dn_state_delta", error, error_cap);
        id<MTLComputePipelineState> p_update = pipeline(device, lib, @"dn_state_update", error, error_cap);
        id<MTLComputePipelineState> p_output = pipeline(device, lib, @"dn_state_output", error, error_cap);
        if (!p_decay || !p_delta || !p_update || !p_output) return 0;

        id<MTLBuffer> b_q = [device newBufferWithBytes:q length:qk_count * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_k = [device newBufferWithBytes:k length:qk_count * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_v = [device newBufferWithBytes:v length:vec_count * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_gate = [device newBufferWithBytes:gate length:(size_t)value_heads * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_beta = [device newBufferWithBytes:beta length:(size_t)value_heads * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_prev = [device newBufferWithBytes:prev_state length:state_count * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_delta = [device newBufferWithLength:vec_count * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_state = [device newBufferWithLength:state_count * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_out = [device newBufferWithLength:vec_count * sizeof(float) options:MTLResourceStorageModeShared];
        if (!b_q || !b_k || !b_v || !b_gate || !b_beta || !b_prev || !b_delta || !b_state || !b_out) {
            set_error(error, error_cap, "out of memory for DeltaNet state Metal buffers");
            return 0;
        }

        id<MTLCommandBuffer> cb = [queue commandBuffer];
        if (!cb) { set_error(error, error_cap, "could not create DeltaNet state command buffer"); return 0; }

        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:p_decay];
        [enc setBuffer:b_prev offset:0 atIndex:0];
        [enc setBuffer:b_gate offset:0 atIndex:1];
        [enc setBuffer:b_state offset:0 atIndex:2];
        [enc setBytes:&state_size length:sizeof(state_size) atIndex:3];
        [enc setBytes:&value_heads length:sizeof(value_heads) atIndex:4];
        dispatch_1d(enc, p_decay, state_count);
        [enc endEncoding];

        enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:p_delta];
        [enc setBuffer:b_state offset:0 atIndex:0];
        [enc setBuffer:b_k offset:0 atIndex:1];
        [enc setBuffer:b_v offset:0 atIndex:2];
        [enc setBuffer:b_beta offset:0 atIndex:3];
        [enc setBuffer:b_delta offset:0 atIndex:4];
        [enc setBytes:&state_size length:sizeof(state_size) atIndex:5];
        [enc setBytes:&key_heads length:sizeof(key_heads) atIndex:6];
        [enc setBytes:&value_heads length:sizeof(value_heads) atIndex:7];
        dispatch_1d(enc, p_delta, vec_count);
        [enc endEncoding];

        enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:p_update];
        [enc setBuffer:b_state offset:0 atIndex:0];
        [enc setBuffer:b_k offset:0 atIndex:1];
        [enc setBuffer:b_delta offset:0 atIndex:2];
        [enc setBytes:&state_size length:sizeof(state_size) atIndex:3];
        [enc setBytes:&key_heads length:sizeof(key_heads) atIndex:4];
        [enc setBytes:&value_heads length:sizeof(value_heads) atIndex:5];
        dispatch_1d(enc, p_update, state_count);
        [enc endEncoding];

        enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:p_output];
        [enc setBuffer:b_state offset:0 atIndex:0];
        [enc setBuffer:b_q offset:0 atIndex:1];
        [enc setBuffer:b_out offset:0 atIndex:2];
        [enc setBytes:&state_size length:sizeof(state_size) atIndex:3];
        [enc setBytes:&key_heads length:sizeof(key_heads) atIndex:4];
        [enc setBytes:&value_heads length:sizeof(value_heads) atIndex:5];
        dispatch_1d(enc, p_output, vec_count);
        [enc endEncoding];

        const double t0 = now_ms();
        [cb commit];
        [cb waitUntilCompleted];
        const double t1 = now_ms();
        if (cb.status != MTLCommandBufferStatusCompleted) {
            const char *d = cb.error.localizedDescription.UTF8String;
            snprintf(error, error_cap, "DeltaNet state command buffer failed: %s", d ? d : "unknown");
            return 0;
        }

        memcpy(delta, b_delta.contents, vec_count * sizeof(float));
        memcpy(next_state, b_state.contents, state_count * sizeof(float));
        memcpy(output, b_out.contents, vec_count * sizeof(float));
        if (telemetry) telemetry->compute_ms = t1 - t0;
    }
    return 1;
}
