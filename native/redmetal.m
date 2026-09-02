#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "redmetal.h"

#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/stat.h>

#define REDMETAL_ABI_VERSION 1u

static __thread char g_redmetal_error[512];

static void redmetal_set_error(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_redmetal_error, sizeof(g_redmetal_error), fmt, ap);
    va_end(ap);
}

static double redmetal_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static int redmetal_pread_all(
        int fd,
        uint64_t file_size,
        uint64_t offset,
        uint64_t len,
        uint8_t *dst,
        uint64_t *bytes_read,
        uint64_t *read_calls) {
    if (!dst || offset > file_size || len > file_size - offset) {
        redmetal_set_error(
            "invalid pread range offset=%llu len=%llu file_size=%llu",
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

        if (read_calls) (*read_calls)++;
        if (nread <= 0) {
            redmetal_set_error(
                "pread failed at offset=%llu: %s",
                (unsigned long long)(offset + pos),
                nread < 0 ? strerror(errno) : "unexpected EOF");
            return 0;
        }
        pos += (uint64_t)nread;
        if (bytes_read) *bytes_read += (uint64_t)nread;
    }
    return 1;
}

static NSString * const kProbeSource = @
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"kernel void redmetal_probe(device const uchar *data [[buffer(0)]],\n"
"                           constant ulong &length [[buffer(1)]],\n"
"                           device uint *out [[buffer(2)]],\n"
"                           uint gid [[thread_position_in_grid]]) {\n"
"    if (gid != 0) return;\n"
"    uint h = 2166136261u;\n"
"    if (length == 0) { out[0] = h; return; }\n"
"    const ulong step = 4096ul;\n"
"    for (ulong i = 0; i < length; i += step) {\n"
"        h = (h ^ uint(data[i])) * 16777619u;\n"
"    }\n"
"    h = (h ^ uint(data[length - 1])) * 16777619u;\n"
"    out[0] = h;\n"
"}\n";

@interface RMMetalPool : NSObject {
@public
    id<MTLDevice> _device;
    id<MTLCommandQueue> _queue;
    id<MTLComputePipelineState> _probePipeline;
    id<MTLBuffer> _probeOutput;
    NSMutableArray<id<MTLBuffer>> *_slabs;

    int _fd;
    uint64_t _fileSize;
    uint64_t _budgetBytes;
    uint64_t _slotBytes;
    uint32_t _capacity;
    uint32_t _slotsPerSlab;

    uint64_t _allocatedBytes;
    uint64_t _bytesRead;
    uint64_t _readCalls;
    double _readMs;
}

- (instancetype)initWithModelPath:(const char *)path
                       budgetBytes:(uint64_t)budget
                         slotBytes:(uint64_t)slotBytes
                      slotsPerSlab:(uint32_t)slotsPerSlab;
- (id<MTLBuffer>)bufferForSlot:(uint32_t)slot offset:(NSUInteger *)offset;
@end

@implementation RMMetalPool

- (instancetype)initWithModelPath:(const char *)path
                       budgetBytes:(uint64_t)budget
                         slotBytes:(uint64_t)slotBytes
                      slotsPerSlab:(uint32_t)slotsPerSlab {
    self = [super init];
    if (!self) return nil;

    _fd = -1;
    if (!path || !path[0] || budget == 0 || slotBytes == 0) {
        redmetal_set_error("invalid Red Metal pool arguments");
        return nil;
    }

    _capacity = (uint32_t)(budget / slotBytes);
    if (_capacity == 0) {
        redmetal_set_error("cache budget cannot fit one Metal expert slot");
        return nil;
    }

    _device = MTLCreateSystemDefaultDevice();
    if (!_device) {
        redmetal_set_error("Metal device is unavailable");
        return nil;
    }
    if (![_device hasUnifiedMemory]) {
        redmetal_set_error("Red Metal requires Apple unified memory");
        return nil;
    }

    _queue = [_device newCommandQueue];
    if (!_queue) {
        redmetal_set_error("failed to create Metal command queue");
        return nil;
    }

    NSError *libraryError = nil;
    id<MTLLibrary> library = [_device newLibraryWithSource:kProbeSource
                                                   options:nil
                                                     error:&libraryError];
    if (!library) {
        redmetal_set_error(
            "failed to compile Red Metal probe kernel: %s",
            libraryError.localizedDescription.UTF8String ?: "unknown error");
        return nil;
    }

    id<MTLFunction> fn = [library newFunctionWithName:@"redmetal_probe"];
    if (!fn) {
        redmetal_set_error("Red Metal probe function not found");
        return nil;
    }

    NSError *pipelineError = nil;
    _probePipeline = [_device newComputePipelineStateWithFunction:fn error:&pipelineError];
    if (!_probePipeline) {
        redmetal_set_error(
            "failed to create Red Metal probe pipeline: %s",
            pipelineError.localizedDescription.UTF8String ?: "unknown error");
        return nil;
    }

    _probeOutput = [_device newBufferWithLength:sizeof(uint32_t)
                                         options:MTLResourceStorageModeShared];
    if (!_probeOutput) {
        redmetal_set_error("failed to allocate Red Metal probe output buffer");
        return nil;
    }

    _fd = open(path, O_RDONLY);
    if (_fd < 0) {
        redmetal_set_error("open(%s) failed: %s", path, strerror(errno));
        return nil;
    }

    struct stat st;
    if (fstat(_fd, &st) != 0 || st.st_size <= 0) {
        redmetal_set_error("failed to stat model file: %s", strerror(errno));
        close(_fd);
        _fd = -1;
        return nil;
    }

    _fileSize = (uint64_t)st.st_size;
    _budgetBytes = budget;
    _slotBytes = slotBytes;
    _slotsPerSlab = slotsPerSlab ? slotsPerSlab : 64u;
    if (_slotsPerSlab > _capacity) _slotsPerSlab = _capacity;
    _slabs = [NSMutableArray array];
    return self;
}

- (void)dealloc {
    if (_fd >= 0) {
        close(_fd);
        _fd = -1;
    }
}

- (id<MTLBuffer>)bufferForSlot:(uint32_t)slot offset:(NSUInteger *)offset {
    if (slot >= _capacity) {
        redmetal_set_error("slot %u exceeds capacity %u", slot, _capacity);
        return nil;
    }

    const uint32_t slabIndex = slot / _slotsPerSlab;
    const uint32_t localSlot = slot % _slotsPerSlab;

    @synchronized (self) {
        while (_slabs.count <= slabIndex) {
            const uint32_t newSlabIndex = (uint32_t)_slabs.count;
            const uint32_t firstSlot = newSlabIndex * _slotsPerSlab;
            const uint32_t remaining = _capacity - firstSlot;
            const uint32_t count = remaining < _slotsPerSlab ? remaining : _slotsPerSlab;
            const uint64_t bytes64 = (uint64_t)count * _slotBytes;
            if (bytes64 == 0 || bytes64 > (uint64_t)NSUIntegerMax) {
                redmetal_set_error("invalid Metal slab size");
                return nil;
            }
            if (_allocatedBytes > _budgetBytes - bytes64) {
                redmetal_set_error("Metal slab allocation would exceed hard cache budget");
                return nil;
            }

            id<MTLBuffer> slab = [_device newBufferWithLength:(NSUInteger)bytes64
                                                      options:MTLResourceStorageModeShared];
            if (!slab) {
                redmetal_set_error(
                    "failed to allocate Metal expert slab %.2f MiB",
                    (double)bytes64 / (1024.0 * 1024.0));
                return nil;
            }
            slab.label = [NSString stringWithFormat:@"redlite_expert_slab_%u", newSlabIndex];
            [_slabs addObject:slab];
            _allocatedBytes += bytes64;
        }

        id<MTLBuffer> slab = _slabs[slabIndex];
        const uint64_t inner64 = (uint64_t)localSlot * _slotBytes;
        if (inner64 > (uint64_t)NSUIntegerMax) {
            redmetal_set_error("Metal slot offset exceeds NSUInteger");
            return nil;
        }
        if (offset) *offset = (NSUInteger)inner64;
        return slab;
    }
}

@end

uint32_t redmetal_abi_version(void) {
    return REDMETAL_ABI_VERSION;
}

const char * redmetal_last_error(void) {
    return g_redmetal_error[0] ? g_redmetal_error : "unknown Red Metal error";
}

int redmetal_system_device_name(char *dst, size_t capacity) {
    @autoreleasepool {
        if (!dst || capacity == 0) return 0;
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device || ![device hasUnifiedMemory]) {
            redmetal_set_error("compatible Apple Metal device unavailable");
            return 0;
        }
        const char *name = device.name.UTF8String ?: "Apple Metal device";
        snprintf(dst, capacity, "%s", name);
        return 1;
    }
}

redmetal_pool_t redmetal_pool_create(
        const char *model_path,
        uint64_t budget_bytes,
        uint64_t slot_bytes,
        uint32_t slots_per_slab) {
    @autoreleasepool {
        g_redmetal_error[0] = '\0';
        RMMetalPool *pool = [[RMMetalPool alloc] initWithModelPath:model_path
                                                      budgetBytes:budget_bytes
                                                        slotBytes:slot_bytes
                                                     slotsPerSlab:slots_per_slab];
        if (!pool) return NULL;
        return (__bridge_retained void *)pool;
    }
}

void redmetal_pool_destroy(redmetal_pool_t handle) {
    if (!handle) return;
    @autoreleasepool {
        RMMetalPool *pool = (__bridge_transfer RMMetalPool *)handle;
        (void)pool;
    }
}

static RMMetalPool * redmetal_pool_obj(redmetal_pool_t handle) {
    if (!handle) {
        redmetal_set_error("Red Metal pool is null");
        return nil;
    }
    return (__bridge RMMetalPool *)handle;
}

uint32_t redmetal_pool_capacity(redmetal_pool_t handle) {
    RMMetalPool *pool = redmetal_pool_obj(handle);
    return pool ? pool->_capacity : 0;
}

uint32_t redmetal_pool_slab_count(redmetal_pool_t handle) {
    RMMetalPool *pool = redmetal_pool_obj(handle);
    return pool ? (uint32_t)pool->_slabs.count : 0;
}

uint64_t redmetal_pool_allocated_bytes(redmetal_pool_t handle) {
    RMMetalPool *pool = redmetal_pool_obj(handle);
    return pool ? pool->_allocatedBytes : 0;
}

uint64_t redmetal_pool_bytes_read(redmetal_pool_t handle) {
    RMMetalPool *pool = redmetal_pool_obj(handle);
    if (!pool) return 0;
    @synchronized (pool) { return pool->_bytesRead; }
}

uint64_t redmetal_pool_read_calls(redmetal_pool_t handle) {
    RMMetalPool *pool = redmetal_pool_obj(handle);
    if (!pool) return 0;
    @synchronized (pool) { return pool->_readCalls; }
}

double redmetal_pool_read_ms(redmetal_pool_t handle) {
    RMMetalPool *pool = redmetal_pool_obj(handle);
    if (!pool) return 0.0;
    @synchronized (pool) { return pool->_readMs; }
}

int redmetal_pool_load_expert(
        redmetal_pool_t handle,
        uint32_t slot_id,
        uint64_t gate_file_offset,
        uint64_t gate_bytes,
        uint64_t up_file_offset,
        uint64_t up_bytes,
        uint64_t down_file_offset,
        uint64_t down_bytes,
        uint64_t *bytes_read,
        double *elapsed_ms) {
    @autoreleasepool {
        RMMetalPool *pool = redmetal_pool_obj(handle);
        if (!pool) return 0;

        if (gate_bytes > UINT64_MAX - up_bytes ||
            gate_bytes + up_bytes > UINT64_MAX - down_bytes) {
            redmetal_set_error("expert payload size overflow");
            return 0;
        }
        const uint64_t payload = gate_bytes + up_bytes + down_bytes;
        if (payload > pool->_slotBytes) {
            redmetal_set_error(
                "expert payload %llu exceeds slot size %llu",
                (unsigned long long)payload,
                (unsigned long long)pool->_slotBytes);
            return 0;
        }

        NSUInteger slotBase = 0;
        id<MTLBuffer> slab = [pool bufferForSlot:slot_id offset:&slotBase];
        if (!slab) return 0;
        uint8_t *base = (uint8_t *)[slab contents];
        if (!base) {
            redmetal_set_error("Metal shared slab has no CPU-visible contents");
            return 0;
        }

        uint64_t localBytes = 0;
        uint64_t localCalls = 0;
        const double started = redmetal_now_ms();
        int ok = redmetal_pread_all(
                     pool->_fd, pool->_fileSize,
                     gate_file_offset, gate_bytes,
                     base + slotBase,
                     &localBytes, &localCalls) &&
                 redmetal_pread_all(
                     pool->_fd, pool->_fileSize,
                     up_file_offset, up_bytes,
                     base + slotBase + gate_bytes,
                     &localBytes, &localCalls) &&
                 redmetal_pread_all(
                     pool->_fd, pool->_fileSize,
                     down_file_offset, down_bytes,
                     base + slotBase + gate_bytes + up_bytes,
                     &localBytes, &localCalls);
        const double ms = redmetal_now_ms() - started;

        if (!ok) return 0;
        @synchronized (pool) {
            pool->_bytesRead += localBytes;
            pool->_readCalls += localCalls;
            pool->_readMs += ms;
        }
        if (bytes_read) *bytes_read = localBytes;
        if (elapsed_ms) *elapsed_ms = ms;
        return 1;
    }
}

int redmetal_pool_gpu_probe(
        redmetal_pool_t handle,
        uint32_t slot_id,
        uint64_t payload_bytes,
        uint32_t *checksum,
        double *elapsed_ms) {
    @autoreleasepool {
        RMMetalPool *pool = redmetal_pool_obj(handle);
        if (!pool) return 0;
        if (payload_bytes == 0 || payload_bytes > pool->_slotBytes) {
            redmetal_set_error("invalid GPU probe payload size");
            return 0;
        }

        NSUInteger slotBase = 0;
        id<MTLBuffer> slab = [pool bufferForSlot:slot_id offset:&slotBase];
        if (!slab) return 0;

        *((uint32_t *)[pool->_probeOutput contents]) = 0u;
        id<MTLCommandBuffer> cb = [pool->_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (!cb || !enc) {
            redmetal_set_error("failed to create Metal probe command buffer");
            return 0;
        }

        [enc setComputePipelineState:pool->_probePipeline];
        [enc setBuffer:slab offset:slotBase atIndex:0];
        [enc setBytes:&payload_bytes length:sizeof(payload_bytes) atIndex:1];
        [enc setBuffer:pool->_probeOutput offset:0 atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [enc endEncoding];

        const double started = redmetal_now_ms();
        [cb commit];
        [cb waitUntilCompleted];
        const double ms = redmetal_now_ms() - started;

        if (cb.status == MTLCommandBufferStatusError) {
            redmetal_set_error(
                "Metal probe command failed: %s",
                cb.error.localizedDescription.UTF8String ?: "unknown error");
            return 0;
        }

        if (checksum) *checksum = *((uint32_t *)[pool->_probeOutput contents]);
        if (elapsed_ms) *elapsed_ms = ms;
        return 1;
    }
}
