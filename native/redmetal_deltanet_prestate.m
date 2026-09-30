#define _FILE_OFFSET_BITS 64
#define _POSIX_C_SOURCE 200809L

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "redlite_native_deltanet_prestate.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static void set_error(char *dst, size_t cap, const char *msg) {
    if (dst && cap) snprintf(dst, cap, "%s", msg ? msg : "unknown DeltaNet prestate Metal error");
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

static NSString * const kSource = @
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"kernel void dn_ba_params(device const float *ba [[buffer(0)]],\n"
"    device const float *dt [[buffer(1)]], device const float *avec [[buffer(2)]],\n"
"    device float *beta [[buffer(3)]], device float *gate [[buffer(4)]],\n"
"    constant uint &dt_rank [[buffer(5)]], constant uint &n_group [[buffer(6)]],\n"
"    uint gid [[thread_position_in_grid]]) {\n"
"    if (gid >= dt_rank) return;\n"
"    const uint width = dt_rank / n_group;\n"
"    const uint group = gid / width; const uint local = gid - group * width;\n"
"    const uint stride = 2u * width;\n"
"    const float b = ba[group * stride + local];\n"
"    const float alpha = ba[group * stride + width + local] + dt[gid];\n"
"    beta[gid] = 1.0f / (1.0f + exp(-b));\n"
"    const float sp = alpha > 20.0f ? alpha : log(1.0f + exp(alpha));\n"
"    gate[gid] = sp * avec[gid];\n"
"}\n"
"kernel void dn_conv_silu(device const float *state [[buffer(0)]],\n"
"    device const float *qkv [[buffer(1)]], device const float *conv_w [[buffer(2)]],\n"
"    device float *out [[buffer(3)]], constant uint &channels [[buffer(4)]],\n"
"    constant uint &dconv [[buffer(5)]], uint gid [[thread_position_in_grid]]) {\n"
"    if (gid >= channels) return; float acc = 0.0f;\n"
"    const uint ns = dconv - 1u; const uint sb = gid * ns; const uint kb = gid * dconv;\n"
"    for (uint j = 0; j < ns; ++j) acc = fma(state[sb + j], conv_w[kb + j], acc);\n"
"    acc = fma(qkv[gid], conv_w[kb + ns], acc);\n"
"    out[gid] = acc / (1.0f + exp(-acc));\n"
"}\n"
"kernel void dn_qk_l2(device const float *src [[buffer(0)]], device float *dst [[buffer(1)]],\n"
"    constant uint &offset [[buffer(2)]], constant uint &head_dim [[buffer(3)]],\n"
"    constant uint &heads [[buffer(4)]], constant float &eps [[buffer(5)]],\n"
"    uint gid [[thread_position_in_grid]]) {\n"
"    if (gid >= heads) return; const uint base = offset + gid * head_dim; float ss = 0.0f;\n"
"    for (uint j = 0; j < head_dim; ++j) ss = fma(src[base + j], src[base + j], ss);\n"
"    const float scale = 1.0f / max(sqrt(ss), eps);\n"
"    for (uint j = 0; j < head_dim; ++j) dst[gid * head_dim + j] = src[base + j] * scale;\n"
"}\n"
"kernel void dn_copy_slice(device const float *src [[buffer(0)]], device float *dst [[buffer(1)]],\n"
"    constant uint &offset [[buffer(2)]], constant uint &count [[buffer(3)]], uint gid [[thread_position_in_grid]]) {\n"
"    if (gid < count) dst[gid] = src[offset + gid];\n"
"}\n"
"kernel void dn_shift_state(device const float *state [[buffer(0)]], device const float *qkv [[buffer(1)]],\n"
"    device float *next_state [[buffer(2)]], constant uint &channels [[buffer(3)]],\n"
"    constant uint &dconv [[buffer(4)]], uint gid [[thread_position_in_grid]]) {\n"
"    if (gid >= channels) return; const uint ns = dconv - 1u; const uint b = gid * ns;\n"
"    for (uint j = 0; j + 1u < ns; ++j) next_state[b + j] = state[b + j + 1u];\n"
"    next_state[b + ns - 1u] = qkv[gid];\n"
"}\n";

static id<MTLComputePipelineState> pipeline(id<MTLDevice> dev, id<MTLLibrary> lib, NSString *name, char *error, size_t cap) {
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    if (!fn) { snprintf(error, cap, "missing Metal function %s", name.UTF8String); return nil; }
    NSError *pe = nil;
    id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:fn error:&pe];
    if (!p) {
        const char *d = pe.localizedDescription.UTF8String;
        snprintf(error, cap, "pipeline %s failed: %s", name.UTF8String, d ? d : "unknown");
    }
    return p;
}

static void dispatch_1d(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p, NSUInteger n) {
    const NSUInteger tg = MIN((NSUInteger)64, p.maxTotalThreadsPerThreadgroup);
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg ? tg : 1, 1, 1)];
}

int rl_deltanet_prestate_gpu_execute(
        const char *model_path,
        const rl_dn_prestate_tensor_info tensors[RL_DN_PRESTATE_TENSOR_COUNT],
        const float *qkv_mixed, uint32_t qkv_count,
        const float *ba, uint32_t ba_count,
        const float *conv_state, uint32_t conv_state_count,
        uint32_t d_conv, uint32_t d_inner, uint32_t d_state, uint32_t dt_rank, uint32_t n_group,
        float eps,
        float *beta, float *gate, float *conv_silu, float *q_norm, float *k_norm, float *v, float *next_conv_state,
        rl_dn_prestate_telemetry *telemetry, char *error, size_t error_cap) {
    if (!model_path || !tensors || !qkv_mixed || !ba || !conv_state || !beta || !gate || !conv_silu ||
        !q_norm || !k_norm || !v || !next_conv_state || d_conv < 2u || !d_inner || !d_state || !dt_rank || !n_group ||
        dt_rank % n_group != 0u || d_inner % dt_rank != 0u) {
        set_error(error, error_cap, "invalid DeltaNet prestate Metal arguments"); return 0;
    }
    const uint32_t head_v_dim = d_inner / dt_rank;
    const uint32_t qk_each = d_state * n_group;
    const uint32_t channels = 2u * qk_each + d_inner;
    const uint32_t state_count = (d_conv - 1u) * channels;
    if (qkv_count != channels || ba_count != 2u * dt_rank || conv_state_count != state_count || head_v_dim == 0u) {
        set_error(error, error_cap, "DeltaNet prestate Metal shape mismatch"); return 0;
    }
    if (tensors[RL_DN_CONV1D].ggml_type != 0u || tensors[RL_DN_CONV1D].n_dims != 2u ||
        tensors[RL_DN_CONV1D].shape[0] != d_conv || tensors[RL_DN_CONV1D].shape[1] != channels ||
        tensors[RL_DN_DT].ggml_type != 0u || tensors[RL_DN_DT].n_dims != 1u || tensors[RL_DN_DT].shape[0] != dt_rank ||
        tensors[RL_DN_A].ggml_type != 0u || tensors[RL_DN_A].n_dims != 1u || tensors[RL_DN_A].shape[0] != dt_rank) {
        set_error(error, error_cap, "unexpected real DeltaNet F32 tensor layout"); return 0;
    }
    const size_t conv_bytes = (size_t)d_conv * channels * sizeof(float);
    const size_t vec_bytes = (size_t)dt_rank * sizeof(float);
    if (tensors[RL_DN_CONV1D].tensor_span_bytes < conv_bytes || tensors[RL_DN_DT].tensor_span_bytes < vec_bytes ||
        tensors[RL_DN_A].tensor_span_bytes < vec_bytes) {
        set_error(error, error_cap, "DeltaNet prestate tensor span too small"); return 0;
    }

    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device || !device.hasUnifiedMemory) { set_error(error, error_cap, "DeltaNet prestate requires Apple unified memory"); return 0; }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        NSError *le = nil;
        id<MTLLibrary> lib = [device newLibraryWithSource:kSource options:nil error:&le];
        if (!lib) {
            const char *d = le.localizedDescription.UTF8String;
            snprintf(error, error_cap, "DeltaNet prestate Metal compile failed: %s", d ? d : "unknown"); return 0;
        }
        id<MTLComputePipelineState> p_ba = pipeline(device, lib, @"dn_ba_params", error, error_cap);
        id<MTLComputePipelineState> p_conv = pipeline(device, lib, @"dn_conv_silu", error, error_cap);
        id<MTLComputePipelineState> p_l2 = pipeline(device, lib, @"dn_qk_l2", error, error_cap);
        id<MTLComputePipelineState> p_copy = pipeline(device, lib, @"dn_copy_slice", error, error_cap);
        id<MTLComputePipelineState> p_shift = pipeline(device, lib, @"dn_shift_state", error, error_cap);
        if (!p_ba || !p_conv || !p_l2 || !p_copy || !p_shift) return 0;

        id<MTLBuffer> b_convw = [device newBufferWithLength:conv_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_dt = [device newBufferWithLength:vec_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_a = [device newBufferWithLength:vec_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_qkv = [device newBufferWithBytes:qkv_mixed length:(size_t)channels*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_ba = [device newBufferWithBytes:ba length:(size_t)ba_count*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_state = [device newBufferWithBytes:conv_state length:(size_t)state_count*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_beta = [device newBufferWithLength:vec_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_gate = [device newBufferWithLength:vec_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_conv = [device newBufferWithLength:(size_t)channels*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_q = [device newBufferWithLength:(size_t)qk_each*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_k = [device newBufferWithLength:(size_t)qk_each*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_v = [device newBufferWithLength:(size_t)d_inner*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_next = [device newBufferWithLength:(size_t)state_count*sizeof(float) options:MTLResourceStorageModeShared];
        if (!b_convw || !b_dt || !b_a || !b_qkv || !b_ba || !b_state || !b_beta || !b_gate || !b_conv || !b_q || !b_k || !b_v || !b_next) {
            set_error(error, error_cap, "out of memory for DeltaNet prestate Metal buffers"); return 0;
        }

        const int fd = open(model_path, O_RDONLY);
        if (fd < 0) { snprintf(error, error_cap, "open GGUF failed: %s", strerror(errno)); return 0; }
        uint64_t calls = 0;
        const double r0 = now_ms();
        const int ok = pread_full(fd, b_convw.contents, conv_bytes, tensors[RL_DN_CONV1D].tensor_offset, &calls) &&
                       pread_full(fd, b_dt.contents, vec_bytes, tensors[RL_DN_DT].tensor_offset, &calls) &&
                       pread_full(fd, b_a.contents, vec_bytes, tensors[RL_DN_A].tensor_offset, &calls);
        const double r1 = now_ms();
        close(fd);
        if (!ok) { set_error(error, error_cap, "pread DeltaNet prestate Metal weights failed"); return 0; }

        const double c0 = now_ms();
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        if (!cb) { set_error(error, error_cap, "failed to create DeltaNet prestate command buffer"); return 0; }

        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:p_ba]; [enc setBuffer:b_ba offset:0 atIndex:0]; [enc setBuffer:b_dt offset:0 atIndex:1];
        [enc setBuffer:b_a offset:0 atIndex:2]; [enc setBuffer:b_beta offset:0 atIndex:3]; [enc setBuffer:b_gate offset:0 atIndex:4];
        [enc setBytes:&dt_rank length:sizeof(dt_rank) atIndex:5]; [enc setBytes:&n_group length:sizeof(n_group) atIndex:6];
        dispatch_1d(enc, p_ba, dt_rank); [enc endEncoding];

        enc = [cb computeCommandEncoder]; [enc setComputePipelineState:p_conv]; [enc setBuffer:b_state offset:0 atIndex:0];
        [enc setBuffer:b_qkv offset:0 atIndex:1]; [enc setBuffer:b_convw offset:0 atIndex:2]; [enc setBuffer:b_conv offset:0 atIndex:3];
        [enc setBytes:&channels length:sizeof(channels) atIndex:4]; [enc setBytes:&d_conv length:sizeof(d_conv) atIndex:5];
        dispatch_1d(enc, p_conv, channels); [enc endEncoding];

        uint32_t offset = 0;
        enc = [cb computeCommandEncoder]; [enc setComputePipelineState:p_l2]; [enc setBuffer:b_conv offset:0 atIndex:0]; [enc setBuffer:b_q offset:0 atIndex:1];
        [enc setBytes:&offset length:sizeof(offset) atIndex:2]; [enc setBytes:&d_state length:sizeof(d_state) atIndex:3];
        [enc setBytes:&n_group length:sizeof(n_group) atIndex:4]; [enc setBytes:&eps length:sizeof(eps) atIndex:5]; dispatch_1d(enc, p_l2, n_group); [enc endEncoding];

        offset = qk_each;
        enc = [cb computeCommandEncoder]; [enc setComputePipelineState:p_l2]; [enc setBuffer:b_conv offset:0 atIndex:0]; [enc setBuffer:b_k offset:0 atIndex:1];
        [enc setBytes:&offset length:sizeof(offset) atIndex:2]; [enc setBytes:&d_state length:sizeof(d_state) atIndex:3];
        [enc setBytes:&n_group length:sizeof(n_group) atIndex:4]; [enc setBytes:&eps length:sizeof(eps) atIndex:5]; dispatch_1d(enc, p_l2, n_group); [enc endEncoding];

        offset = 2u * qk_each;
        enc = [cb computeCommandEncoder]; [enc setComputePipelineState:p_copy]; [enc setBuffer:b_conv offset:0 atIndex:0]; [enc setBuffer:b_v offset:0 atIndex:1];
        [enc setBytes:&offset length:sizeof(offset) atIndex:2]; [enc setBytes:&d_inner length:sizeof(d_inner) atIndex:3]; dispatch_1d(enc, p_copy, d_inner); [enc endEncoding];

        enc = [cb computeCommandEncoder]; [enc setComputePipelineState:p_shift]; [enc setBuffer:b_state offset:0 atIndex:0]; [enc setBuffer:b_qkv offset:0 atIndex:1];
        [enc setBuffer:b_next offset:0 atIndex:2]; [enc setBytes:&channels length:sizeof(channels) atIndex:3]; [enc setBytes:&d_conv length:sizeof(d_conv) atIndex:4];
        dispatch_1d(enc, p_shift, channels); [enc endEncoding];

        [cb commit]; [cb waitUntilCompleted];
        const double c1 = now_ms();
        if (cb.status != MTLCommandBufferStatusCompleted) {
            const char *d = cb.error.localizedDescription.UTF8String;
            snprintf(error, error_cap, "DeltaNet prestate Metal execution failed: %s", d ? d : "unknown"); return 0;
        }

        memcpy(beta, b_beta.contents, vec_bytes); memcpy(gate, b_gate.contents, vec_bytes);
        memcpy(conv_silu, b_conv.contents, (size_t)channels*sizeof(float));
        memcpy(q_norm, b_q.contents, (size_t)qk_each*sizeof(float)); memcpy(k_norm, b_k.contents, (size_t)qk_each*sizeof(float));
        memcpy(v, b_v.contents, (size_t)d_inner*sizeof(float)); memcpy(next_conv_state, b_next.contents, (size_t)state_count*sizeof(float));
        if (telemetry) {
            telemetry->bytes_read = conv_bytes + 2u * vec_bytes; telemetry->read_calls = calls;
            telemetry->read_ms = r1 - r0; telemetry->compute_ms = c1 - c0;
            telemetry->ssd_during_compute_bytes = 0; telemetry->ssd_during_compute_calls = 0;
        }
        if (error && error_cap) error[0] = '\0';
        return 1;
    }
}
