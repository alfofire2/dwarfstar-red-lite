#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "redmetal_quant.h"

#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/stat.h>

#define REDMETAL_QUANT_ABI_VERSION 1u
#define GGML_TYPE_IQ2_XS 17u
#define GGML_TYPE_IQ1_M 29u
#define QK_IQ 256u

static __thread char g_quant_error[512];

static void quant_set_error(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_quant_error, sizeof(g_quant_error), fmt, ap);
    va_end(ap);
}

static double quant_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static int quant_pread_all(int fd, uint64_t file_size, uint64_t offset, uint64_t len, uint8_t *dst) {
    if (!dst || offset > file_size || len > file_size - offset) {
        quant_set_error("invalid quant pread range offset=%llu len=%llu file_size=%llu",
            (unsigned long long)offset,
            (unsigned long long)len,
            (unsigned long long)file_size);
        return 0;
    }
    uint64_t pos = 0;
    while (pos < len) {
        const uint64_t rem = len - pos;
        const size_t want = rem > (uint64_t)SSIZE_MAX ? (size_t)SSIZE_MAX : (size_t)rem;
        ssize_t nread;
        do {
            nread = pread(fd, dst + pos, want, (off_t)(offset + pos));
        } while (nread < 0 && errno == EINTR);
        if (nread <= 0) {
            quant_set_error("quant pread failed at offset=%llu: %s",
                (unsigned long long)(offset + pos),
                nread < 0 ? strerror(errno) : "unexpected EOF");
            return 0;
        }
        pos += (uint64_t)nread;
    }
    return 1;
}

static NSString * const kMixedQuantSource = @
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"\n"
"inline uint redlite_sign8(uint sign7) {\n"
"    sign7 &= 127u;\n"
"    return sign7 | ((popcount(sign7) & 1u) << 7);\n"
"}\n"
"\n"
"kernel void redmetal_iq2_xs_rows(\n"
"    device const uchar *matrix [[buffer(0)]],\n"
"    constant uint &ncols [[buffer(1)]],\n"
"    constant uint &row_start [[buffer(2)]],\n"
"    device const float *x [[buffer(3)]],\n"
"    device float *out [[buffer(4)]],\n"
"    device const char *grid [[buffer(5)]],\n"
"    uint gid [[thread_position_in_grid]]) {\n"
"    const uint row = row_start + gid;\n"
"    const uint blocks = ncols / 256u;\n"
"    const ulong row_bytes = ulong(blocks) * 74ul;\n"
"    device const uchar *rowp = matrix + ulong(row) * row_bytes;\n"
"    float acc = 0.0f;\n"
"    for (uint ib = 0; ib < blocks; ++ib) {\n"
"        device const uchar *bp = rowp + ulong(ib) * 74ul;\n"
"        const ushort dbits = *(device const ushort *)bp;\n"
"        const float d = float(as_type<half>(dbits));\n"
"        device const ushort *qs = (device const ushort *)(bp + 2);\n"
"        device const uchar *scales = bp + 66;\n"
"        for (uint g = 0; g < 16u; ++g) {\n"
"            const uint scale = (uint(scales[g >> 1]) >> (4u * (g & 1u))) & 15u;\n"
"            const float db = d * (0.5f + float(scale)) * 0.25f;\n"
"            for (uint part = 0; part < 2u; ++part) {\n"
"                const uint q = uint(qs[2u * g + part]);\n"
"                const uint grid_index = q & 511u;\n"
"                const uint signs = redlite_sign8(q >> 9);\n"
"                device const char *gv = grid + grid_index * 8u;\n"
"                const uint xbase = ib * 256u + g * 16u + part * 8u;\n"
"                for (uint j = 0; j < 8u; ++j) {\n"
"                    const float sign = (signs & (1u << j)) ? -1.0f : 1.0f;\n"
"                    acc += x[xbase + j] * (db * float(gv[j]) * sign);\n"
"                }\n"
"            }\n"
"        }\n"
"    }\n"
"    out[gid] = acc;\n"
"}\n"
"\n"
"kernel void redmetal_iq1_m_rows(\n"
"    device const uchar *matrix [[buffer(0)]],\n"
"    constant uint &ncols [[buffer(1)]],\n"
"    constant uint &row_start [[buffer(2)]],\n"
"    device const float *x [[buffer(3)]],\n"
"    device float *out [[buffer(4)]],\n"
"    device const char *grid [[buffer(5)]],\n"
"    uint gid [[thread_position_in_grid]]) {\n"
"    const uint row = row_start + gid;\n"
"    const uint blocks = ncols / 256u;\n"
"    const ulong row_bytes = ulong(blocks) * 56ul;\n"
"    device const uchar *rowp = matrix + ulong(row) * row_bytes;\n"
"    float acc = 0.0f;\n"
"    for (uint ib = 0; ib < blocks; ++ib) {\n"
"        device const uchar *bp = rowp + ulong(ib) * 56ul;\n"
"        device const uchar *qs = bp;\n"
"        device const uchar *qh = bp + 32;\n"
"        device const ushort *sc = (device const ushort *)(bp + 48);\n"
"        const ushort dbits = ushort(sc[0] >> 12) | ushort((sc[1] >> 8) & 0x00f0u) |\n"
"                             ushort((sc[2] >> 4) & 0x0f00u) | ushort(sc[3] & 0xf000u);\n"
"        const float d = float(as_type<half>(dbits));\n"
"        for (uint g = 0; g < 32u; ++g) {\n"
"            const uint nibble = (uint(qh[g >> 1]) >> (4u * (g & 1u))) & 15u;\n"
"            const uint grid_index = uint(qs[g]) | ((nibble & 7u) << 8);\n"
"            const float delta = (nibble & 8u) ? -0.125f : 0.125f;\n"
"            const uint scale_index = g >> 1;\n"
"            const uint scale = (uint(sc[scale_index >> 2]) >> (3u * (scale_index & 3u))) & 7u;\n"
"            const float dl = d * float(2u * scale + 1u);\n"
"            device const char *gv = grid + grid_index * 8u;\n"
"            const uint xbase = ib * 256u + g * 8u;\n"
"            for (uint j = 0; j < 8u; ++j) {\n"
"                acc += x[xbase + j] * (dl * (float(gv[j]) + delta));\n"
"            }\n"
"        }\n"
"    }\n"
"    out[gid] = acc;\n"
"}\n";

uint32_t redmetal_quant_abi_version(void) { return REDMETAL_QUANT_ABI_VERSION; }

const char *redmetal_quant_last_error(void) {
    return g_quant_error[0] ? g_quant_error : "unknown Red Metal quant error";
}

int redmetal_quant_rows(
        const char *model_path,
        uint64_t matrix_file_offset,
        uint64_t matrix_bytes,
        uint32_t ggml_type,
        uint32_t ncols,
        uint32_t nrows,
        uint32_t row_start,
        uint32_t row_count,
        const float *input,
        uint32_t input_count,
        const int8_t *grid,
        uint32_t grid_count,
        float *output,
        uint32_t output_count,
        double *io_ms,
        double *gpu_ms) {
    @autoreleasepool {
        g_quant_error[0] = '\0';
        if (!model_path || !model_path[0] || !input || !grid || !output) {
            quant_set_error("invalid Red Metal mixed-quant arguments");
            return 0;
        }
        if (ggml_type != GGML_TYPE_IQ2_XS && ggml_type != GGML_TYPE_IQ1_M) {
            quant_set_error("unsupported routed GGML type %u", ggml_type);
            return 0;
        }
        if (!ncols || ncols % QK_IQ || !nrows || !row_count || row_start >= nrows ||
            row_count > nrows - row_start || input_count != ncols || output_count < row_count) {
            quant_set_error("invalid mixed-quant dimensions cols=%u rows=%u start=%u count=%u",
                ncols, nrows, row_start, row_count);
            return 0;
        }
        const uint64_t block_bytes = ggml_type == GGML_TYPE_IQ2_XS ? 74u : 56u;
        const uint32_t expected_grid = ggml_type == GGML_TYPE_IQ2_XS ? 512u * 8u : 2048u * 8u;
        if (grid_count != expected_grid) {
            quant_set_error("GGML type %u grid has %u values, expected %u", ggml_type, grid_count, expected_grid);
            return 0;
        }
        const uint64_t row_bytes = (uint64_t)(ncols / QK_IQ) * block_bytes;
        if (!row_bytes || (uint64_t)nrows > UINT64_MAX / row_bytes || matrix_bytes != (uint64_t)nrows * row_bytes) {
            quant_set_error("matrix byte size mismatch: got %llu expected %llu",
                (unsigned long long)matrix_bytes,
                (unsigned long long)((uint64_t)nrows * row_bytes));
            return 0;
        }
        if (matrix_bytes > (uint64_t)NSUIntegerMax) {
            quant_set_error("matrix is too large for one parity MTLBuffer");
            return 0;
        }

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device || ![device hasUnifiedMemory]) {
            quant_set_error("mixed quant parity requires Apple unified memory");
            return 0;
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) {
            quant_set_error("failed to create mixed quant command queue");
            return 0;
        }

        NSError *library_error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:kMixedQuantSource options:nil error:&library_error];
        if (!library) {
            quant_set_error("failed to compile mixed quant kernels: %s",
                library_error.localizedDescription.UTF8String ?: "unknown Metal error");
            return 0;
        }
        NSString *function_name = ggml_type == GGML_TYPE_IQ2_XS ? @"redmetal_iq2_xs_rows" : @"redmetal_iq1_m_rows";
        id<MTLFunction> function = [library newFunctionWithName:function_name];
        if (!function) {
            quant_set_error("mixed quant Metal function not found");
            return 0;
        }
        NSError *pipeline_error = nil;
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&pipeline_error];
        if (!pipeline) {
            quant_set_error("failed to create mixed quant pipeline: %s",
                pipeline_error.localizedDescription.UTF8String ?: "unknown Metal error");
            return 0;
        }

        id<MTLBuffer> matrix = [device newBufferWithLength:(NSUInteger)matrix_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> in_buffer = [device newBufferWithBytes:input
                                                     length:(NSUInteger)ncols * sizeof(float)
                                                    options:MTLResourceStorageModeShared];
        id<MTLBuffer> grid_buffer = [device newBufferWithBytes:grid
                                                       length:(NSUInteger)grid_count * sizeof(int8_t)
                                                      options:MTLResourceStorageModeShared];
        id<MTLBuffer> out_buffer = [device newBufferWithLength:(NSUInteger)row_count * sizeof(float)
                                                        options:MTLResourceStorageModeShared];
        if (!matrix || !in_buffer || !grid_buffer || !out_buffer) {
            quant_set_error("failed to allocate mixed quant Metal buffers");
            return 0;
        }

        int fd = open(model_path, O_RDONLY);
        if (fd < 0) {
            quant_set_error("open(%s) failed: %s", model_path, strerror(errno));
            return 0;
        }
        struct stat st;
        if (fstat(fd, &st) != 0 || st.st_size <= 0) {
            quant_set_error("failed to stat model file: %s", strerror(errno));
            close(fd);
            return 0;
        }
        const double t_io = quant_now_ms();
        const int read_ok = quant_pread_all(fd, (uint64_t)st.st_size, matrix_file_offset, matrix_bytes,
                                           (uint8_t *)[matrix contents]);
        const double local_io_ms = quant_now_ms() - t_io;
        close(fd);
        if (!read_ok) return 0;
        memset([out_buffer contents], 0, (size_t)row_count * sizeof(float));

        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (!cb || !enc) {
            quant_set_error("failed to create mixed quant Metal command encoder");
            return 0;
        }
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:matrix offset:0 atIndex:0];
        const uint32_t cols = ncols;
        const uint32_t start = row_start;
        [enc setBytes:&cols length:sizeof(cols) atIndex:1];
        [enc setBytes:&start length:sizeof(start) atIndex:2];
        [enc setBuffer:in_buffer offset:0 atIndex:3];
        [enc setBuffer:out_buffer offset:0 atIndex:4];
        [enc setBuffer:grid_buffer offset:0 atIndex:5];
        const NSUInteger tg = MIN((NSUInteger)row_count, (NSUInteger)64u);
        [enc dispatchThreads:MTLSizeMake(row_count, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];

        const double t_gpu = quant_now_ms();
        [cb commit];
        [cb waitUntilCompleted];
        const double local_gpu_ms = quant_now_ms() - t_gpu;
        if (cb.status == MTLCommandBufferStatusError) {
            quant_set_error("mixed quant Metal command failed: %s",
                cb.error.localizedDescription.UTF8String ?: "unknown Metal error");
            return 0;
        }
        memcpy(output, [out_buffer contents], (size_t)row_count * sizeof(float));
        if (io_ms) *io_ms = local_io_ms;
        if (gpu_ms) *gpu_ms = local_gpu_ms;
        return 1;
    }
}
