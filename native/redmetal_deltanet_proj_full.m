#define _FILE_OFFSET_BITS 64
#define _POSIX_C_SOURCE 200809L

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "redlite_native_deltanet_layer.h"
#include "redlite_native_iq2_xxs.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define IQ2_XXS_BLOCK_BYTES 66u
#define Q8_0_BLOCK_BYTES 34u

static void set_error(char *dst, size_t cap, const char *msg) {
    if (dst && cap) snprintf(dst, cap, "%s", msg ? msg : "unknown DeltaNet full projection Metal error");
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

static size_t q8_row_bytes(uint32_t ncols) {
    return ncols && ncols % 32u == 0 ? (size_t)(ncols / 32u) * Q8_0_BLOCK_BYTES : 0;
}

static NSString * const kDeltaFullProjSource = @
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"kernel void redlite_dn_full_rmsnorm(\n"
"    device const float *x [[buffer(0)]], device const float *w [[buffer(1)]],\n"
"    device float *y [[buffer(2)]], constant uint &n [[buffer(3)]],\n"
"    constant float &eps [[buffer(4)]], uint gid [[thread_position_in_grid]]) {\n"
"    if (gid != 0u) return;\n"
"    float ss = 0.0f;\n"
"    for (uint i = 0; i < n; ++i) ss = fma(x[i], x[i], ss);\n"
"    const float scale = rsqrt(ss / float(n) + eps);\n"
"    for (uint i = 0; i < n; ++i) y[i] = x[i] * scale * w[i];\n"
"}\n"
"kernel void redlite_dn_full_iq2(\n"
"    device const uchar *weights [[buffer(0)]], constant uint &ncols [[buffer(1)]],\n"
"    device const float *x [[buffer(2)]], device float *out [[buffer(3)]],\n"
"    device const uchar *grid [[buffer(4)]], uint row [[thread_position_in_grid]]) {\n"
"    const uint blocks = ncols / 256u;\n"
"    const ulong row_bytes = ulong(blocks) * 66ul;\n"
"    device const uchar *rowp = weights + ulong(row) * row_bytes;\n"
"    float acc = 0.0f;\n"
"    for (uint ib = 0; ib < blocks; ++ib) {\n"
"        device const uchar *bp = rowp + ulong(ib) * 66ul;\n"
"        const float d = float(as_type<half>(*(device const ushort *)bp));\n"
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
"                    const float s = (sign8 & (1u << j)) ? -1.0f : 1.0f;\n"
"                    acc += x[xb + j] * (db * float(gv[j]) * s);\n"
"                }\n"
"            }\n"
"        }\n"
"    }\n"
"    out[row] = acc;\n"
"}\n"
"kernel void redlite_dn_full_q8(\n"
"    device const uchar *weights [[buffer(0)]], constant uint &ncols [[buffer(1)]],\n"
"    device const float *x [[buffer(2)]], device float *out [[buffer(3)]],\n"
"    uint row [[thread_position_in_grid]]) {\n"
"    const uint blocks = ncols / 32u;\n"
"    const ulong row_bytes = ulong(blocks) * 34ul;\n"
"    device const uchar *rowp = weights + ulong(row) * row_bytes;\n"
"    float acc = 0.0f;\n"
"    for (uint ib = 0; ib < blocks; ++ib) {\n"
"        device const uchar *bp = rowp + ulong(ib) * 34ul;\n"
"        const float d = float(as_type<half>(*(device const ushort *)bp));\n"
"        device const int8_t *q = (device const int8_t *)(bp + 2ul);\n"
"        const uint xb = ib * 32u;\n"
"        for (uint j = 0; j < 32u; ++j) acc += x[xb + j] * (d * float(q[j]));\n"
"    }\n"
"    out[row] = acc;\n"
"}\n";

static id<MTLComputePipelineState> make_pipeline(id<MTLDevice> device, id<MTLLibrary> lib, NSString *name,
        char *error, size_t cap) {
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    if (!fn) { snprintf(error, cap, "missing Metal function %s", name.UTF8String); return nil; }
    NSError *pe = nil;
    id<MTLComputePipelineState> pipe = [device newComputePipelineStateWithFunction:fn error:&pe];
    if (!pipe) {
        const char *d = pe.localizedDescription.UTF8String;
        snprintf(error, cap, "pipeline %s failed: %s", name.UTF8String, d ? d : "unknown");
    }
    return pipe;
}

static void encode_rms(id<MTLCommandBuffer> cb, id<MTLComputePipelineState> pipe,
        id<MTLBuffer> x, id<MTLBuffer> w, id<MTLBuffer> y, uint32_t n, float eps) {
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pipe];
    [enc setBuffer:x offset:0 atIndex:0];
    [enc setBuffer:w offset:0 atIndex:1];
    [enc setBuffer:y offset:0 atIndex:2];
    [enc setBytes:&n length:sizeof(n) atIndex:3];
    [enc setBytes:&eps length:sizeof(eps) atIndex:4];
    [enc dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [enc endEncoding];
}

static void encode_iq2(id<MTLCommandBuffer> cb, id<MTLComputePipelineState> pipe,
        id<MTLBuffer> weights, uint32_t ncols, id<MTLBuffer> x, id<MTLBuffer> out,
        uint32_t rows, id<MTLBuffer> grid) {
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pipe];
    [enc setBuffer:weights offset:0 atIndex:0];
    [enc setBytes:&ncols length:sizeof(ncols) atIndex:1];
    [enc setBuffer:x offset:0 atIndex:2];
    [enc setBuffer:out offset:0 atIndex:3];
    [enc setBuffer:grid offset:0 atIndex:4];
    const NSUInteger tg = MIN((NSUInteger)64, pipe.maxTotalThreadsPerThreadgroup);
    [enc dispatchThreads:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg ? tg : 1, 1, 1)];
    [enc endEncoding];
}

static void encode_q8(id<MTLCommandBuffer> cb, id<MTLComputePipelineState> pipe,
        id<MTLBuffer> weights, uint32_t ncols, id<MTLBuffer> x, id<MTLBuffer> out, uint32_t rows) {
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pipe];
    [enc setBuffer:weights offset:0 atIndex:0];
    [enc setBytes:&ncols length:sizeof(ncols) atIndex:1];
    [enc setBuffer:x offset:0 atIndex:2];
    [enc setBuffer:out offset:0 atIndex:3];
    const NSUInteger tg = MIN((NSUInteger)64, pipe.maxTotalThreadsPerThreadgroup);
    [enc dispatchThreads:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg ? tg : 1, 1, 1)];
    [enc endEncoding];
}

int rl_deltanet_proj_gpu_execute_full(
        const char *model_path,
        const rl_dn_proj_tensor_info t[RL_DN_PROJ_TENSOR_COUNT],
        const float *input,
        uint32_t input_count,
        float rms_eps,
        float *norm_out,
        uint32_t norm_out_count,
        float *qkv_out,
        uint32_t qkv_out_count,
        float *z_out,
        uint32_t z_out_count,
        float *ba_out,
        uint32_t ba_out_count,
        rl_dn_proj_telemetry *telemetry,
        char *error,
        size_t error_cap) {
    if (!model_path || !t || !input || !norm_out || !qkv_out || !z_out || !ba_out ||
        t[0].kind != RL_DN_ATTN_NORM || t[1].kind != RL_DN_QKV || t[2].kind != RL_DN_Z_GATE ||
        t[3].kind != RL_DN_BETA_ALPHA || t[0].ggml_type != 0u || t[1].ggml_type != 16u ||
        t[2].ggml_type != 16u || t[3].ggml_type != 8u || t[0].n_dims != 1u ||
        t[1].n_dims != 2u || t[2].n_dims != 2u || t[3].n_dims != 2u ||
        t[0].shape[0] == 0 || t[0].shape[0] > UINT32_MAX) {
        set_error(error, error_cap, "invalid full DeltaNet projection Metal arguments/layout");
        return 0;
    }
    const uint32_t hidden = (uint32_t)t[0].shape[0];
    if (t[1].shape[0] != hidden || t[2].shape[0] != hidden || t[3].shape[0] != hidden ||
        t[1].shape[1] == 0 || t[2].shape[1] == 0 || t[3].shape[1] == 0 ||
        t[1].shape[1] > UINT32_MAX || t[2].shape[1] > UINT32_MAX || t[3].shape[1] > UINT32_MAX) {
        set_error(error, error_cap, "full DeltaNet projection Metal shape mismatch");
        return 0;
    }
    const uint32_t qkv_rows = (uint32_t)t[1].shape[1];
    const uint32_t z_rows = (uint32_t)t[2].shape[1];
    const uint32_t ba_rows = (uint32_t)t[3].shape[1];
    if (input_count != hidden || norm_out_count < hidden || qkv_out_count < qkv_rows ||
        z_out_count < z_rows || ba_out_count < ba_rows) {
        set_error(error, error_cap, "full DeltaNet projection output buffer too small");
        return 0;
    }

    const size_t iq2_rb = iq2_row_bytes(hidden);
    const size_t q8_rb = q8_row_bytes(hidden);
    if (!iq2_rb || !q8_rb || qkv_rows > SIZE_MAX / iq2_rb ||
        z_rows > SIZE_MAX / iq2_rb || ba_rows > SIZE_MAX / q8_rb) {
        set_error(error, error_cap, "unsupported full DeltaNet projection row layout");
        return 0;
    }
    const size_t weight_bytes[RL_DN_PROJ_TENSOR_COUNT] = {
        (size_t)hidden * sizeof(float),
        (size_t)qkv_rows * iq2_rb,
        (size_t)z_rows * iq2_rb,
        (size_t)ba_rows * q8_rb,
    };
    for (uint32_t k = 0; k < RL_DN_PROJ_TENSOR_COUNT; ++k) {
        if (t[k].tensor_span_bytes < weight_bytes[k]) {
            set_error(error, error_cap, "full DeltaNet projection tensor span too small");
            return 0;
        }
    }

    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device || !device.hasUnifiedMemory) {
            set_error(error, error_cap, "full DeltaNet projection requires Apple unified memory");
            return 0;
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) {
            set_error(error, error_cap, "failed to create full DeltaNet projection queue");
            return 0;
        }
        NSError *le = nil;
        id<MTLLibrary> lib = [device newLibraryWithSource:kDeltaFullProjSource options:nil error:&le];
        if (!lib) {
            const char *d = le.localizedDescription.UTF8String;
            snprintf(error, error_cap, "full DeltaNet Metal source compile failed: %s", d ? d : "unknown");
            return 0;
        }
        id<MTLComputePipelineState> rms_pipe = make_pipeline(device, lib, @"redlite_dn_full_rmsnorm", error, error_cap);
        id<MTLComputePipelineState> iq2_pipe = make_pipeline(device, lib, @"redlite_dn_full_iq2", error, error_cap);
        id<MTLComputePipelineState> q8_pipe = make_pipeline(device, lib, @"redlite_dn_full_q8", error, error_cap);
        if (!rms_pipe || !iq2_pipe || !q8_pipe) return 0;

        id<MTLBuffer> weights[RL_DN_PROJ_TENSOR_COUNT];
        for (uint32_t k = 0; k < RL_DN_PROJ_TENSOR_COUNT; ++k) {
            weights[k] = [device newBufferWithLength:weight_bytes[k] options:MTLResourceStorageModeShared];
            if (!weights[k]) {
                set_error(error, error_cap, "failed to allocate full DeltaNet projection weight buffer");
                return 0;
            }
        }
        id<MTLBuffer> xbuf = [device newBufferWithLength:(size_t)hidden * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> nbuf = [device newBufferWithLength:(size_t)hidden * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> qbuf = [device newBufferWithLength:(size_t)qkv_rows * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> zbuf = [device newBufferWithLength:(size_t)z_rows * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bbuf = [device newBufferWithLength:(size_t)ba_rows * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> gridbuf = [device newBufferWithLength:RL_IQ2_XXS_GRID_COUNT options:MTLResourceStorageModeShared];
        if (!xbuf || !nbuf || !qbuf || !zbuf || !bbuf || !gridbuf) {
            set_error(error, error_cap, "failed to allocate full DeltaNet projection scratch");
            return 0;
        }
        memcpy(xbuf.contents, input, (size_t)hidden * sizeof(float));
        if (!rl_native_iq2_xxs_build_grid((uint8_t *)gridbuf.contents, error, error_cap)) return 0;

        const int fd = open(model_path, O_RDONLY);
        if (fd < 0) {
            snprintf(error, error_cap, "open GGUF for full DeltaNet projection failed: %s", strerror(errno));
            return 0;
        }
        uint64_t calls = 0, bytes = 0;
        const double r0 = now_ms();
        for (uint32_t k = 0; k < RL_DN_PROJ_TENSOR_COUNT; ++k) {
            if (!pread_full(fd, weights[k].contents, weight_bytes[k], t[k].tensor_offset, &calls)) {
                close(fd);
                set_error(error, error_cap, "pread full DeltaNet projection weights failed");
                return 0;
            }
            bytes += weight_bytes[k];
        }
        const double r1 = now_ms();
        close(fd);

        id<MTLCommandBuffer> cb = [queue commandBuffer];
        if (!cb) {
            set_error(error, error_cap, "failed to create full DeltaNet projection command buffer");
            return 0;
        }
        const double c0 = now_ms();
        encode_rms(cb, rms_pipe, xbuf, weights[0], nbuf, hidden, rms_eps);
        encode_iq2(cb, iq2_pipe, weights[1], hidden, nbuf, qbuf, qkv_rows, gridbuf);
        encode_iq2(cb, iq2_pipe, weights[2], hidden, nbuf, zbuf, z_rows, gridbuf);
        encode_q8(cb, q8_pipe, weights[3], hidden, nbuf, bbuf, ba_rows);
        [cb commit];
        [cb waitUntilCompleted];
        const double c1 = now_ms();
        if (cb.status != MTLCommandBufferStatusCompleted || cb.error) {
            const char *d = cb.error.localizedDescription.UTF8String;
            snprintf(error, error_cap, "full DeltaNet projection command failed: %s", d ? d : "unknown");
            return 0;
        }

        memcpy(norm_out, nbuf.contents, (size_t)hidden * sizeof(float));
        memcpy(qkv_out, qbuf.contents, (size_t)qkv_rows * sizeof(float));
        memcpy(z_out, zbuf.contents, (size_t)z_rows * sizeof(float));
        memcpy(ba_out, bbuf.contents, (size_t)ba_rows * sizeof(float));
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
