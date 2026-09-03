#define _FILE_OFFSET_BITS 64
#define _POSIX_C_SOURCE 200809L

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <TargetConditionals.h>

#include "redlite_native_shared_exec.h"
#include "redlite_native_iq2_xxs.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static void set_error(char *dst, size_t cap, const char *msg) {
    if (dst && cap) snprintf(dst, cap, "%s", msg ? msg : "unknown shared Metal error");
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

static NSString * const kSharedSource = @
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"kernel void redlite_shared_q6_k_rows(\n"
"    device const uchar *weights [[buffer(0)]], constant uint &ncols [[buffer(1)]],\n"
"    constant uint &row_start [[buffer(2)]], device const float *x [[buffer(3)]],\n"
"    device float *out [[buffer(4)]], uint gid [[thread_position_in_grid]]) {\n"
"    const uint row = row_start + gid;\n"
"    const uint blocks = ncols / 256u;\n"
"    const ulong row_bytes = ulong(blocks) * 210ul;\n"
"    device const uchar *rowp = weights + ulong(row) * row_bytes;\n"
"    float acc = 0.0f;\n"
"    for (uint ib = 0; ib < blocks; ++ib) {\n"
"        device const uchar *bp = rowp + ulong(ib) * 210ul;\n"
"        device const ushort *ql0 = (device const ushort *)(bp + 0ul);\n"
"        device const ushort *qh0 = (device const ushort *)(bp + 128ul);\n"
"        device const int8_t *scales = (device const int8_t *)(bp + 192ul);\n"
"        const half d_all_h = as_type<half>(*(device const ushort *)(bp + 208ul));\n"
"        const float d_all = float(d_all_h);\n"
"        for (short il0 = 0; il0 < 16; ++il0) {\n"
"            short il = il0;\n"
"            device const ushort *ql = ql0 + 32 * (il / 8) + 16 * ((il / 2) & 1) + 8 * (il & 1);\n"
"            device const ushort *qh = qh0 + 16 * (il / 8) + 8 * (il & 1);\n"
"            const float sc = float(scales[(il % 2) + 2 * (il / 2)]);\n"
"            il = (il / 2) & 3;\n"
"            const uint kmask1 = il > 1 ? (il > 2 ? 0xC0C0C0C0u : 0x30303030u) : (il > 0 ? 0x0C0C0C0Cu : 0x03030303u);\n"
"            const uint kmask2 = il > 1 ? 0xF0F0F0F0u : 0x0F0F0F0Fu;\n"
"            const float ml = d_all * sc * 32.0f;\n"
"            const float dl0 = d_all * sc;\n"
"            const float dl1 = dl0 / 256.0f;\n"
"            const float dl2 = dl1 / 256.0f;\n"
"            const float dl3 = dl2 / 256.0f;\n"
"            const uchar shr_h = il > 2 ? 2 : 0;\n"
"            const uchar shl_h = il > 1 ? 0 : (il > 0 ? 2 : 4);\n"
"            const uchar shr_l = il > 1 ? 4 : 0;\n"
"            for (uint i = 0; i < 4u; ++i) {\n"
"                const uint low = (uint(ql[2u*i]) | (uint(ql[2u*i + 1u]) << 16)) & kmask2;\n"
"                const uint high = (uint(qh[2u*i]) | (uint(qh[2u*i + 1u]) << 16)) & kmask1;\n"
"                const uint q = ((high << shl_h) >> shr_h) | (low >> shr_l);\n"
"                const uint xb = ib * 256u + uint(il0) * 16u + i * 4u;\n"
"                acc += x[xb + 0u] * (dl0 * float(q & 0xFFu) - ml);\n"
"                acc += x[xb + 1u] * (dl1 * float(q & 0xFF00u) - ml);\n"
"                acc += x[xb + 2u] * (dl2 * float(q & 0xFF0000u) - ml);\n"
"                acc += x[xb + 3u] * (dl3 * float(q & 0xFF000000u) - ml);\n"
"            }\n"
"        }\n"
"    }\n"
"    out[gid] = acc;\n"
"}\n"
"kernel void redlite_shared_iq2_xxs_rows(\n"
"    device const uchar *weights [[buffer(0)]], constant uint &ncols [[buffer(1)]],\n"
"    constant uint &row_start [[buffer(2)]], device const float *x [[buffer(3)]],\n"
"    device float *out [[buffer(4)]], device const uchar *grid [[buffer(5)]],\n"
"    uint gid [[thread_position_in_grid]]) {\n"
"    const uint row = row_start + gid;\n"
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
"    out[gid] = acc;\n"
"}\n"
"kernel void redlite_shared_silu_mul(device const float *gate [[buffer(0)]],\n"
"    device const float *up [[buffer(1)]], device float *act [[buffer(2)]],\n"
"    constant uint &count [[buffer(3)]], uint gid [[thread_position_in_grid]]) {\n"
"    if (gid >= count) return; const float g = gate[gid]; act[gid] = (g / (1.0f + exp(-g))) * up[gid];\n"
"}\n"
"kernel void redlite_shared_scalar_gate(device const float *w [[buffer(0)]],\n"
"    device const float *x [[buffer(1)]], device float *scalar [[buffer(2)]],\n"
"    constant uint &hidden [[buffer(3)]], uint gid [[thread_position_in_grid]]) {\n"
"    if (gid != 0u) return; float acc = 0.0f; for (uint i = 0; i < hidden; ++i) acc = fma(w[i], x[i], acc);\n"
"    scalar[0] = 1.0f / (1.0f + exp(-acc));\n"
"}\n"
"kernel void redlite_shared_scale(device float *v [[buffer(0)]], device const float *scalar [[buffer(1)]],\n"
"    constant uint &count [[buffer(2)]], uint gid [[thread_position_in_grid]]) {\n"
"    if (gid < count) v[gid] *= scalar[0];\n"
"}\n";

static int validate(const rl_shared_tensor_info t[4], uint32_t *hidden, uint32_t *ffn, size_t bytes[4], char *error, size_t cap) {
    if (!t || t[0].kind != RL_SHARED_GATE_INPUT || t[1].kind != RL_SHARED_GATE || t[2].kind != RL_SHARED_UP || t[3].kind != RL_SHARED_DOWN ||
        t[0].ggml_type != 0u || t[0].n_dims != 1u || t[1].n_dims != 2u || t[2].n_dims != 2u || t[3].n_dims != 2u ||
        t[0].shape[0] > UINT32_MAX || t[1].shape[1] > UINT32_MAX) {
        set_error(error, cap, "invalid shared Metal tensor layout"); return 0;
    }
    *hidden = (uint32_t)t[0].shape[0]; *ffn = (uint32_t)t[1].shape[1];
    if (!*hidden || !*ffn || t[1].shape[0] != *hidden || t[2].shape[0] != *hidden || t[2].shape[1] != *ffn ||
        t[3].shape[0] != *ffn || t[3].shape[1] != *hidden) { set_error(error, cap, "shared Metal shape mismatch"); return 0; }
    bytes[0] = (size_t)*hidden * sizeof(float);
    const size_t rb1 = rl_native_shared_row_bytes(t[1].ggml_type, *hidden);
    const size_t rb2 = rl_native_shared_row_bytes(t[2].ggml_type, *hidden);
    const size_t rb3 = rl_native_shared_row_bytes(t[3].ggml_type, *ffn);
    if (!rb1 || !rb2 || !rb3) { set_error(error, cap, "unsupported shared Metal quant type"); return 0; }
    bytes[1] = (size_t)*ffn * rb1; bytes[2] = (size_t)*ffn * rb2; bytes[3] = (size_t)*hidden * rb3;
    for (uint32_t k = 0; k < 4; ++k) if (t[k].tensor_span_bytes < bytes[k]) { set_error(error, cap, "shared Metal tensor span mismatch"); return 0; }
    return 1;
}

static id<MTLComputePipelineState> make_pipeline(id<MTLDevice> device, id<MTLLibrary> lib, NSString *name, char *error, size_t cap) {
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    if (!fn) { snprintf(error, cap, "missing Metal function %s", name.UTF8String); return nil; }
    NSError *pe = nil;
    id<MTLComputePipelineState> p = [device newComputePipelineStateWithFunction:fn error:&pe];
    if (!p) { const char *d = pe.localizedDescription.UTF8String; snprintf(error, cap, "pipeline %s failed: %s", name.UTF8String, d ? d : "unknown"); }
    return p;
}

static void encode_rows(id<MTLCommandBuffer> cb, id<MTLComputePipelineState> pipe, id<MTLBuffer> weights,
        uint32_t ncols, uint32_t row_start, id<MTLBuffer> x, id<MTLBuffer> out, uint32_t rows, id<MTLBuffer> grid, int iq2) {
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pipe];
    [enc setBuffer:weights offset:0 atIndex:0];
    [enc setBytes:&ncols length:sizeof(ncols) atIndex:1];
    [enc setBytes:&row_start length:sizeof(row_start) atIndex:2];
    [enc setBuffer:x offset:0 atIndex:3];
    [enc setBuffer:out offset:0 atIndex:4];
    if (iq2) [enc setBuffer:grid offset:0 atIndex:5];
    const NSUInteger tg = MIN((NSUInteger)64, pipe.maxTotalThreadsPerThreadgroup);
    [enc dispatchThreads:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg ? tg : 1, 1, 1)];
    [enc endEncoding];
}

int rl_native_shared_gpu_execute(
        const char *model_path, const rl_shared_tensor_info tensors[4], const float *input, uint32_t input_count,
        uint32_t row_start, uint32_t row_count, float *output, uint32_t output_count,
        rl_native_shared_telemetry *telemetry, char *error, size_t error_cap) {
    uint32_t hidden = 0, ffn = 0; size_t bytes[4] = {0};
    if (!model_path || !input || !output || !row_count || output_count < row_count ||
        !validate(tensors, &hidden, &ffn, bytes, error, error_cap) || input_count != hidden ||
        row_start > hidden || row_count > hidden - row_start) {
        if (error && error_cap && !error[0]) set_error(error, error_cap, "invalid shared Metal execution arguments");
        return 0;
    }
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device || !device.hasUnifiedMemory) { set_error(error, error_cap, "shared Metal requires Apple unified memory"); return 0; }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) { set_error(error, error_cap, "failed to create shared Metal queue"); return 0; }
        NSError *le = nil;
        id<MTLLibrary> lib = [device newLibraryWithSource:kSharedSource options:nil error:&le];
        if (!lib) { const char *d = le.localizedDescription.UTF8String; snprintf(error, error_cap, "shared Metal compile failed: %s", d ? d : "unknown"); return 0; }
        id<MTLComputePipelineState> q6 = make_pipeline(device, lib, @"redlite_shared_q6_k_rows", error, error_cap);
        id<MTLComputePipelineState> iq2 = make_pipeline(device, lib, @"redlite_shared_iq2_xxs_rows", error, error_cap);
        id<MTLComputePipelineState> silu = make_pipeline(device, lib, @"redlite_shared_silu_mul", error, error_cap);
        id<MTLComputePipelineState> scalar = make_pipeline(device, lib, @"redlite_shared_scalar_gate", error, error_cap);
        id<MTLComputePipelineState> scale = make_pipeline(device, lib, @"redlite_shared_scale", error, error_cap);
        if (!q6 || !iq2 || !silu || !scalar || !scale) return 0;

        id<MTLBuffer> wb[4];
        for (uint32_t k = 0; k < 4; ++k) {
            wb[k] = [device newBufferWithLength:bytes[k] options:MTLResourceStorageModeShared];
            if (!wb[k]) { set_error(error, error_cap, "failed to allocate shared Metal weight buffer"); return 0; }
        }
        id<MTLBuffer> xbuf = [device newBufferWithBytes:input length:(NSUInteger)hidden * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> gatebuf = [device newBufferWithLength:(NSUInteger)ffn * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> upbuf = [device newBufferWithLength:(NSUInteger)ffn * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> actbuf = [device newBufferWithLength:(NSUInteger)ffn * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> outbuf = [device newBufferWithLength:(NSUInteger)row_count * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> scalarbuf = [device newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];
        uint8_t grid_data[RL_IQ2_XXS_GRID_COUNT];
        if (!rl_native_iq2_xxs_build_grid(grid_data, error, error_cap)) return 0;
        id<MTLBuffer> grid = [device newBufferWithBytes:grid_data length:sizeof(grid_data) options:MTLResourceStorageModeShared];
        if (!xbuf || !gatebuf || !upbuf || !actbuf || !outbuf || !scalarbuf || !grid) { set_error(error, error_cap, "failed to allocate shared Metal scratch"); return 0; }

        const int fd = open(model_path, O_RDONLY);
        if (fd < 0) { set_error(error, error_cap, "open GGUF for shared Metal failed"); return 0; }
        uint64_t calls = 0;
        const double rs = now_ms();
        for (uint32_t k = 0; k < 4; ++k) {
            if (!pread_full(fd, wb[k].contents, bytes[k], tensors[k].tensor_offset, &calls)) { close(fd); set_error(error, error_cap, "pread shared Metal weights failed"); return 0; }
#if TARGET_OS_OSX
            [wb[k] didModifyRange:NSMakeRange(0, bytes[k])];
#endif
        }
        const double re = now_ms();
        close(fd);
#if TARGET_OS_OSX
        [xbuf didModifyRange:NSMakeRange(0, (NSUInteger)hidden * sizeof(float))];
#endif

        id<MTLCommandBuffer> cb = [queue commandBuffer];
        if (!cb) { set_error(error, error_cap, "failed to create shared Metal command buffer"); return 0; }
        {
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:scalar]; [enc setBuffer:wb[0] offset:0 atIndex:0]; [enc setBuffer:xbuf offset:0 atIndex:1];
            [enc setBuffer:scalarbuf offset:0 atIndex:2]; [enc setBytes:&hidden length:sizeof(hidden) atIndex:3];
            [enc dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)]; [enc endEncoding];
        }
        id<MTLComputePipelineState> gp = tensors[1].ggml_type == 14u ? q6 : iq2;
        id<MTLComputePipelineState> up = tensors[2].ggml_type == 14u ? q6 : iq2;
        id<MTLComputePipelineState> dp = tensors[3].ggml_type == 14u ? q6 : iq2;
        encode_rows(cb, gp, wb[1], hidden, 0u, xbuf, gatebuf, ffn, grid, tensors[1].ggml_type == 16u);
        encode_rows(cb, up, wb[2], hidden, 0u, xbuf, upbuf, ffn, grid, tensors[2].ggml_type == 16u);
        {
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:silu]; [enc setBuffer:gatebuf offset:0 atIndex:0]; [enc setBuffer:upbuf offset:0 atIndex:1];
            [enc setBuffer:actbuf offset:0 atIndex:2]; [enc setBytes:&ffn length:sizeof(ffn) atIndex:3];
            const NSUInteger tg = MIN((NSUInteger)64, silu.maxTotalThreadsPerThreadgroup);
            [enc dispatchThreads:MTLSizeMake(ffn,1,1) threadsPerThreadgroup:MTLSizeMake(tg ? tg : 1,1,1)]; [enc endEncoding];
        }
        encode_rows(cb, dp, wb[3], ffn, row_start, actbuf, outbuf, row_count, grid, tensors[3].ggml_type == 16u);
        {
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:scale]; [enc setBuffer:outbuf offset:0 atIndex:0]; [enc setBuffer:scalarbuf offset:0 atIndex:1];
            [enc setBytes:&row_count length:sizeof(row_count) atIndex:2]; const NSUInteger tg = MIN((NSUInteger)64, scale.maxTotalThreadsPerThreadgroup);
            [enc dispatchThreads:MTLSizeMake(row_count,1,1) threadsPerThreadgroup:MTLSizeMake(tg ? tg : 1,1,1)]; [enc endEncoding];
        }
        const double gs = now_ms(); [cb commit]; [cb waitUntilCompleted]; const double ge = now_ms();
        if (cb.status == MTLCommandBufferStatusError) { const char *d = cb.error.localizedDescription.UTF8String; snprintf(error, error_cap, "shared Metal command failed: %s", d ? d : "unknown"); return 0; }
        memcpy(output, outbuf.contents, (size_t)row_count * sizeof(float));
        const float scalar_value = *(float *)scalarbuf.contents;
        if (telemetry) {
            memset(telemetry, 0, sizeof(*telemetry)); telemetry->bytes_read = (uint64_t)bytes[0] + bytes[1] + bytes[2] + bytes[3];
            telemetry->read_calls = calls; telemetry->read_ms = re - rs; telemetry->compute_ms = ge - gs; telemetry->scalar_gate = scalar_value;
        }
        if (error && error_cap) error[0] = '\0';
        return 1;
    }
}
