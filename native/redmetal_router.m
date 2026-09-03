#define _FILE_OFFSET_BITS 64
#define _POSIX_C_SOURCE 200809L

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "redlite_native_router_exec.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static void set_error(char *dst, size_t cap, const char *msg) {
    if (dst && cap) snprintf(dst, cap, "%s", msg ? msg : "unknown Metal router error");
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
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

int rl_native_router_gpu_f32(
        const char *model_path,
        const rl_router_tensor_info *router,
        const float *input,
        uint32_t input_count,
        float *logits,
        uint32_t logits_count,
        rl_native_router_telemetry *telemetry,
        char *error,
        size_t error_cap) {
    if (!model_path || !router || !input || !logits || router->ggml_type != 0u || router->n_dims != 2u ||
        router->shape[0] > UINT32_MAX || router->shape[1] > UINT32_MAX) {
        set_error(error, error_cap, "invalid F32 Metal router arguments");
        return 0;
    }
    const uint32_t hidden = (uint32_t)router->shape[0];
    const uint32_t experts = (uint32_t)router->shape[1];
    const uint64_t weight_bytes64 = (uint64_t)hidden * experts * sizeof(float);
    if (!hidden || !experts || input_count != hidden || logits_count < experts ||
        router->tensor_span_bytes < weight_bytes64 || weight_bytes64 > SIZE_MAX) {
        set_error(error, error_cap, "F32 Metal router shape/span mismatch");
        return 0;
    }

    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            set_error(error, error_cap, "Metal device unavailable");
            return 0;
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) {
            set_error(error, error_cap, "failed to create Metal command queue");
            return 0;
        }

        static NSString *source = @
            "#include <metal_stdlib>\n"
            "using namespace metal;\n"
            "kernel void redlite_router_f32(\n"
            "    device const float *weights [[buffer(0)]],\n"
            "    device const float *x [[buffer(1)]],\n"
            "    device float *out [[buffer(2)]],\n"
            "    constant uint &hidden [[buffer(3)]],\n"
            "    constant uint &experts [[buffer(4)]],\n"
            "    uint gid [[thread_position_in_grid]]) {\n"
            "    if (gid >= experts) return;\n"
            "    const ulong base = (ulong)gid * (ulong)hidden;\n"
            "    float sum = 0.0f;\n"
            "    for (uint i = 0; i < hidden; ++i) {\n"
            "        sum = fma(weights[base + i], x[i], sum);\n"
            "    }\n"
            "    out[gid] = sum;\n"
            "}\n";

        NSError *libraryError = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&libraryError];
        if (!library) {
            const char *detail = libraryError.localizedDescription.UTF8String;
            snprintf(error, error_cap, "failed to compile F32 router Metal kernel: %s", detail ? detail : "unknown error");
            return 0;
        }
        id<MTLFunction> fn = [library newFunctionWithName:@"redlite_router_f32"];
        if (!fn) {
            set_error(error, error_cap, "missing F32 router Metal function");
            return 0;
        }
        NSError *pipelineError = nil;
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:fn error:&pipelineError];
        if (!pipeline) {
            const char *detail = pipelineError.localizedDescription.UTF8String;
            snprintf(error, error_cap, "failed to create F32 router pipeline: %s", detail ? detail : "unknown error");
            return 0;
        }

        id<MTLBuffer> weights = [device newBufferWithLength:(NSUInteger)weight_bytes64 options:MTLResourceStorageModeShared];
        id<MTLBuffer> xbuf = [device newBufferWithLength:(NSUInteger)hidden * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> outbuf = [device newBufferWithLength:(NSUInteger)experts * sizeof(float) options:MTLResourceStorageModeShared];
        if (!weights || !xbuf || !outbuf) {
            set_error(error, error_cap, "failed to allocate F32 router Metal buffers");
            return 0;
        }

        const int fd = open(model_path, O_RDONLY);
        if (fd < 0) {
            set_error(error, error_cap, "open GGUF for Metal router failed");
            return 0;
        }
        const double read_start = now_ms();
        const int read_ok = pread_full(fd, weights.contents, (size_t)weight_bytes64, router->tensor_offset);
        const double read_end = now_ms();
        close(fd);
        if (!read_ok) {
            set_error(error, error_cap, "pread F32 router into shared Metal buffer failed");
            return 0;
        }
#if TARGET_OS_OSX
        [weights didModifyRange:NSMakeRange(0, (NSUInteger)weight_bytes64)];
#endif
        memcpy(xbuf.contents, input, (size_t)hidden * sizeof(float));
#if TARGET_OS_OSX
        [xbuf didModifyRange:NSMakeRange(0, (NSUInteger)hidden * sizeof(float))];
#endif

        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (!cb || !enc) {
            set_error(error, error_cap, "failed to create Metal router command encoder");
            return 0;
        }
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:weights offset:0 atIndex:0];
        [enc setBuffer:xbuf offset:0 atIndex:1];
        [enc setBuffer:outbuf offset:0 atIndex:2];
        [enc setBytes:&hidden length:sizeof(hidden) atIndex:3];
        [enc setBytes:&experts length:sizeof(experts) atIndex:4];
        const NSUInteger tg = MIN((NSUInteger)64, pipeline.maxTotalThreadsPerThreadgroup);
        [enc dispatchThreads:MTLSizeMake(experts, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg ? tg : 1, 1, 1)];
        [enc endEncoding];

        const double gpu_start = now_ms();
        [cb commit];
        [cb waitUntilCompleted];
        const double gpu_end = now_ms();
        if (cb.status == MTLCommandBufferStatusError) {
            const char *detail = cb.error.localizedDescription.UTF8String;
            snprintf(error, error_cap, "F32 router Metal command failed: %s", detail ? detail : "unknown error");
            return 0;
        }
        memcpy(logits, outbuf.contents, (size_t)experts * sizeof(float));

        if (telemetry) {
            telemetry->read_ms = read_end - read_start;
            telemetry->compute_ms = gpu_end - gpu_start;
        }
        if (error && error_cap) error[0] = '\0';
        return 1;
    }
}
