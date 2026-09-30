#define _FILE_OFFSET_BITS 64
#define _POSIX_C_SOURCE 200809L

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <TargetConditionals.h>

#include "redlite_native_q6_probe.h"
#include "redlite_native_shared_exec.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

static void set_error(char *dst, size_t cap, const char *msg) {
    if (dst && cap) snprintf(dst, cap, "%s", msg ? msg : "unknown Q6 probe error");
}

static int pread_full(int fd, void *dst, size_t bytes, uint64_t offset) {
    unsigned char *p = (unsigned char *)dst;
    size_t done = 0;
    while (done < bytes) {
        const ssize_t n = pread(fd, p + done, bytes - done, (off_t)(offset + done));
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return 0;
        done += (size_t)n;
    }
    return 1;
}

static NSString * const kQ6ProbeSource = @
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"kernel void redlite_q6_probe_rows(\n"
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
"        device const uchar *ql0 = bp;\n"
"        device const uchar *qh0 = bp + 128ul;\n"
"        device const int8_t *sc0 = (device const int8_t *)(bp + 192ul);\n"
"        const float d = float(as_type<half>(*(device const ushort *)(bp + 208ul)));\n"
"        for (uint n = 0; n < 256u; n += 128u) {\n"
"            device const uchar *ql = ql0 + n / 2u;\n"
"            device const uchar *qh = qh0 + n / 4u;\n"
"            device const int8_t *sc = sc0 + n / 16u;\n"
"            for (uint l = 0; l < 32u; ++l) {\n"
"                const uint is = l / 16u;\n"
"                const int q1 = int((ql[l] & 15u) | (((qh[l] >> 0) & 3u) << 4)) - 32;\n"
"                const int q2 = int((ql[l + 32u] & 15u) | (((qh[l] >> 2) & 3u) << 4)) - 32;\n"
"                const int q3 = int((ql[l] >> 4) | (((qh[l] >> 4) & 3u) << 4)) - 32;\n"
"                const int q4 = int((ql[l + 32u] >> 4) | (((qh[l] >> 6) & 3u) << 4)) - 32;\n"
"                const uint xb = ib * 256u + n;\n"
"                acc += x[xb + l]       * (d * float(sc[is + 0u]) * float(q1));\n"
"                acc += x[xb + l + 32u] * (d * float(sc[is + 2u]) * float(q2));\n"
"                acc += x[xb + l + 64u] * (d * float(sc[is + 4u]) * float(q3));\n"
"                acc += x[xb + l + 96u] * (d * float(sc[is + 6u]) * float(q4));\n"
"            }\n"
"        }\n"
"    }\n"
"    out[gid] = acc;\n"
"}\n";

int rl_native_q6_gpu_rows(
        const char *model_path,
        const rl_shared_tensor_info *tensor,
        const float *input,
        uint32_t input_count,
        uint32_t row_start,
        uint32_t row_count,
        float *output,
        uint32_t output_count,
        char *error,
        size_t error_cap) {
    if (!model_path || !tensor || !input || !output || tensor->ggml_type != 14u || tensor->n_dims != 2u ||
        tensor->shape[0] > UINT32_MAX || tensor->shape[1] > UINT32_MAX || !row_count || output_count < row_count) {
        set_error(error, error_cap, "invalid Q6 GPU probe arguments"); return 0;
    }
    const uint32_t ncols = (uint32_t)tensor->shape[0];
    const uint32_t nrows = (uint32_t)tensor->shape[1];
    if (input_count != ncols || row_start > nrows || row_count > nrows - row_start) {
        set_error(error, error_cap, "Q6 GPU probe shape mismatch"); return 0;
    }
    const size_t row_bytes = rl_native_shared_row_bytes(14u, ncols);
    const size_t matrix_bytes = (size_t)nrows * row_bytes;
    if (!row_bytes || tensor->tensor_span_bytes < matrix_bytes) {
        set_error(error, error_cap, "Q6 GPU probe tensor span mismatch"); return 0;
    }

    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device || !device.hasUnifiedMemory) { set_error(error, error_cap, "Q6 probe requires unified-memory Metal"); return 0; }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        NSError *le = nil;
        id<MTLLibrary> lib = [device newLibraryWithSource:kQ6ProbeSource options:nil error:&le];
        if (!lib) { const char *d = le.localizedDescription.UTF8String; snprintf(error, error_cap, "Q6 probe Metal compile failed: %s", d ? d : "unknown"); return 0; }
        id<MTLFunction> fn = [lib newFunctionWithName:@"redlite_q6_probe_rows"];
        NSError *pe = nil;
        id<MTLComputePipelineState> pipe = fn ? [device newComputePipelineStateWithFunction:fn error:&pe] : nil;
        if (!pipe) { const char *d = pe.localizedDescription.UTF8String; snprintf(error, error_cap, "Q6 probe pipeline failed: %s", d ? d : "unknown"); return 0; }

        id<MTLBuffer> weights = [device newBufferWithLength:matrix_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> xbuf = [device newBufferWithBytes:input length:(NSUInteger)ncols * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> outbuf = [device newBufferWithLength:(NSUInteger)row_count * sizeof(float) options:MTLResourceStorageModeShared];
        if (!weights || !xbuf || !outbuf) { set_error(error, error_cap, "Q6 probe buffer allocation failed"); return 0; }
        const int fd = open(model_path, O_RDONLY);
        if (fd < 0) { set_error(error, error_cap, "Q6 probe model open failed"); return 0; }
        const int ok = pread_full(fd, weights.contents, matrix_bytes, tensor->tensor_offset);
        close(fd);
        if (!ok) { set_error(error, error_cap, "Q6 probe matrix pread failed"); return 0; }
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pipe];
        [enc setBuffer:weights offset:0 atIndex:0];
        [enc setBytes:&ncols length:sizeof(ncols) atIndex:1];
        [enc setBytes:&row_start length:sizeof(row_start) atIndex:2];
        [enc setBuffer:xbuf offset:0 atIndex:3];
        [enc setBuffer:outbuf offset:0 atIndex:4];
        const NSUInteger tg = MIN((NSUInteger)64, pipe.maxTotalThreadsPerThreadgroup);
        [enc dispatchThreads:MTLSizeMake(row_count,1,1) threadsPerThreadgroup:MTLSizeMake(tg ? tg : 1,1,1)];
        [enc endEncoding];
        [cb commit]; [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) { const char *d = cb.error.localizedDescription.UTF8String; snprintf(error, error_cap, "Q6 probe command failed: %s", d ? d : "unknown"); return 0; }
        memcpy(output, outbuf.contents, (size_t)row_count * sizeof(float));
        if (error && error_cap) error[0] = '\0';
        return 1;
    }
}
