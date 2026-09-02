#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "redmetal_ffn.h"

#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/stat.h>

#define REDMETAL_FFN_ABI_VERSION 1u
#define GGML_TYPE_IQ2_XS 17u
#define GGML_TYPE_IQ1_M 29u
#define QK_IQ 256u

static __thread char g_ffn_error[512];

static void ffn_set_error(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_ffn_error, sizeof(g_ffn_error), fmt, ap);
    va_end(ap);
}

static double ffn_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static uint64_t round_up_u64(uint64_t value, uint64_t alignment) {
    if (!alignment) return value;
    const uint64_t rem = value % alignment;
    if (!rem) return value;
    if (value > UINT64_MAX - (alignment - rem)) return 0;
    return value + alignment - rem;
}

static int ffn_pread_all(
        int fd,
        uint64_t file_size,
        uint64_t offset,
        uint64_t len,
        uint8_t *dst,
        uint64_t *bytes_read,
        uint64_t *read_calls) {
    if (!dst || offset > file_size || len > file_size - offset) {
        ffn_set_error("invalid FFN pread range offset=%llu len=%llu file_size=%llu",
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
            ffn_set_error("FFN pread failed at offset=%llu: %s",
                (unsigned long long)(offset + pos),
                nread < 0 ? strerror(errno) : "unexpected EOF");
            return 0;
        }
        pos += (uint64_t)nread;
        if (bytes_read) *bytes_read += (uint64_t)nread;
    }
    return 1;
}

static NSString * const kFfnSource = @
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"\n"
"inline uint redlite_sign8(uint sign7) {\n"
"    sign7 &= 127u;\n"
"    return sign7 | ((popcount(sign7) & 1u) << 7);\n"
"}\n"
"\n"
"kernel void redmetal_ffn_iq2_xs_rows(\n"
"    device const uchar *slot [[buffer(0)]],\n"
"    constant ulong &matrix_offset [[buffer(1)]],\n"
"    constant uint &ncols [[buffer(2)]],\n"
"    constant uint &row_start [[buffer(3)]],\n"
"    device const float *x [[buffer(4)]],\n"
"    device float *out [[buffer(5)]],\n"
"    device const char *grid [[buffer(6)]],\n"
"    uint gid [[thread_position_in_grid]]) {\n"
"    const uint row = row_start + gid;\n"
"    const uint blocks = ncols / 256u;\n"
"    const ulong row_bytes = ulong(blocks) * 74ul;\n"
"    device const uchar *rowp = slot + matrix_offset + ulong(row) * row_bytes;\n"
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
"kernel void redmetal_ffn_iq1_m_rows(\n"
"    device const uchar *slot [[buffer(0)]],\n"
"    constant ulong &matrix_offset [[buffer(1)]],\n"
"    constant uint &ncols [[buffer(2)]],\n"
"    constant uint &row_start [[buffer(3)]],\n"
"    device const float *x [[buffer(4)]],\n"
"    device float *out [[buffer(5)]],\n"
"    device const char *grid [[buffer(6)]],\n"
"    uint gid [[thread_position_in_grid]]) {\n"
"    const uint row = row_start + gid;\n"
"    const uint blocks = ncols / 256u;\n"
"    const ulong row_bytes = ulong(blocks) * 56ul;\n"
"    device const uchar *rowp = slot + matrix_offset + ulong(row) * row_bytes;\n"
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
"}\n"
"\n"
"kernel void redmetal_ffn_silu_mul(\n"
"    device const float *gate [[buffer(0)]],\n"
"    device const float *up [[buffer(1)]],\n"
"    device float *out [[buffer(2)]],\n"
"    uint gid [[thread_position_in_grid]]) {\n"
"    const float g = gate[gid];\n"
"    out[gid] = (g / (1.0f + exp(-g))) * up[gid];\n"
"}\n";

@interface RMFfnPool : NSObject {
@public
    id<MTLDevice> _device;
    id<MTLCommandQueue> _queue;
    id<MTLComputePipelineState> _iq2Pipeline;
    id<MTLComputePipelineState> _iq1Pipeline;
    id<MTLComputePipelineState> _siluPipeline;
    id<MTLBuffer> _iq2Grid;
    id<MTLBuffer> _iq1Grid;
    NSMutableArray<id<MTLBuffer>> *_slabs;
    int _fd;
    uint64_t _fileSize;
    uint64_t _budgetBytes;
    uint64_t _slotBytes;
    uint32_t _capacity;
    uint32_t _slotsPerSlab;
    uint32_t *_slotInflight;
    uint64_t _allocatedBytes;
    uint64_t _bytesRead;
    uint64_t _readCalls;
    double _readMs;
}
- (instancetype)initWithModelPath:(const char *)path
                       budgetBytes:(uint64_t)budget
                         slotBytes:(uint64_t)slotBytes
                      slotsPerSlab:(uint32_t)slotsPerSlab
                          iq2Grid:(const int8_t *)iq2Grid
                       iq2GridCount:(uint32_t)iq2GridCount
                          iq1Grid:(const int8_t *)iq1Grid
                       iq1GridCount:(uint32_t)iq1GridCount;
- (id<MTLBuffer>)bufferForSlot:(uint32_t)slot offset:(NSUInteger *)offset;
- (void)markSlot:(uint32_t)slot delta:(int)delta;
@end

@implementation RMFfnPool

- (instancetype)initWithModelPath:(const char *)path
                       budgetBytes:(uint64_t)budget
                         slotBytes:(uint64_t)slotBytes
                      slotsPerSlab:(uint32_t)slotsPerSlab
                          iq2Grid:(const int8_t *)iq2Grid
                       iq2GridCount:(uint32_t)iq2GridCount
                          iq1Grid:(const int8_t *)iq1Grid
                       iq1GridCount:(uint32_t)iq1GridCount {
    self = [super init];
    if (!self) return nil;
    _fd = -1;
    if (!path || !path[0] || !budget || !slotBytes || !iq2Grid || !iq1Grid ||
        iq2GridCount != 512u * 8u || iq1GridCount != 2048u * 8u) {
        ffn_set_error("invalid Red Metal FFN pool arguments");
        return nil;
    }
    const uint64_t aligned = round_up_u64(slotBytes, 4096u);
    if (!aligned || aligned > budget) {
        ffn_set_error("FFN cache budget cannot fit one aligned expert slot");
        return nil;
    }
    _slotBytes = aligned;
    _budgetBytes = budget;
    _capacity = (uint32_t)(budget / aligned);
    _slotsPerSlab = slotsPerSlab ? slotsPerSlab : 64u;
    if (_slotsPerSlab > _capacity) _slotsPerSlab = _capacity;

    _device = MTLCreateSystemDefaultDevice();
    if (!_device || ![_device hasUnifiedMemory]) {
        ffn_set_error("Red Metal FFN requires Apple unified memory");
        return nil;
    }
    _queue = [_device newCommandQueue];
    if (!_queue) {
        ffn_set_error("failed to create FFN Metal command queue");
        return nil;
    }
    NSError *libraryError = nil;
    id<MTLLibrary> library = [_device newLibraryWithSource:kFfnSource options:nil error:&libraryError];
    if (!library) {
        ffn_set_error("failed to compile FFN kernels: %s",
            libraryError.localizedDescription.UTF8String ?: "unknown Metal error");
        return nil;
    }
    id<MTLFunction> iq2 = [library newFunctionWithName:@"redmetal_ffn_iq2_xs_rows"];
    id<MTLFunction> iq1 = [library newFunctionWithName:@"redmetal_ffn_iq1_m_rows"];
    id<MTLFunction> silu = [library newFunctionWithName:@"redmetal_ffn_silu_mul"];
    if (!iq2 || !iq1 || !silu) {
        ffn_set_error("one or more FFN Metal functions were not found");
        return nil;
    }
    NSError *pipelineError = nil;
    _iq2Pipeline = [_device newComputePipelineStateWithFunction:iq2 error:&pipelineError];
    if (!_iq2Pipeline) {
        ffn_set_error("failed to create IQ2_XS FFN pipeline: %s",
            pipelineError.localizedDescription.UTF8String ?: "unknown Metal error");
        return nil;
    }
    pipelineError = nil;
    _iq1Pipeline = [_device newComputePipelineStateWithFunction:iq1 error:&pipelineError];
    if (!_iq1Pipeline) {
        ffn_set_error("failed to create IQ1_M FFN pipeline: %s",
            pipelineError.localizedDescription.UTF8String ?: "unknown Metal error");
        return nil;
    }
    pipelineError = nil;
    _siluPipeline = [_device newComputePipelineStateWithFunction:silu error:&pipelineError];
    if (!_siluPipeline) {
        ffn_set_error("failed to create SiLU FFN pipeline: %s",
            pipelineError.localizedDescription.UTF8String ?: "unknown Metal error");
        return nil;
    }
    _iq2Grid = [_device newBufferWithBytes:iq2Grid length:iq2GridCount options:MTLResourceStorageModeShared];
    _iq1Grid = [_device newBufferWithBytes:iq1Grid length:iq1GridCount options:MTLResourceStorageModeShared];
    if (!_iq2Grid || !_iq1Grid) {
        ffn_set_error("failed to allocate FFN quant grid buffers");
        return nil;
    }

    _fd = open(path, O_RDONLY);
    if (_fd < 0) {
        ffn_set_error("open(%s) failed: %s", path, strerror(errno));
        return nil;
    }
    struct stat st;
    if (fstat(_fd, &st) != 0 || st.st_size <= 0) {
        ffn_set_error("failed to stat model file: %s", strerror(errno));
        close(_fd);
        _fd = -1;
        return nil;
    }
    _fileSize = (uint64_t)st.st_size;
    _slotInflight = calloc(_capacity, sizeof(uint32_t));
    if (!_slotInflight) {
        ffn_set_error("failed to allocate FFN in-flight table");
        close(_fd);
        _fd = -1;
        return nil;
    }
    _slabs = [NSMutableArray array];
    return self;
}

- (void)dealloc {
    if (_fd >= 0) close(_fd);
    free(_slotInflight);
    _slotInflight = NULL;
}

- (id<MTLBuffer>)bufferForSlot:(uint32_t)slot offset:(NSUInteger *)offset {
    if (slot >= _capacity) {
        ffn_set_error("FFN slot %u exceeds capacity %u", slot, _capacity);
        return nil;
    }
    const uint32_t slabIndex = slot / _slotsPerSlab;
    const uint32_t localSlot = slot % _slotsPerSlab;
    @synchronized (self) {
        while (_slabs.count <= slabIndex) {
            const uint32_t newSlab = (uint32_t)_slabs.count;
            const uint32_t first = newSlab * _slotsPerSlab;
            const uint32_t remaining = _capacity - first;
            const uint32_t count = remaining < _slotsPerSlab ? remaining : _slotsPerSlab;
            const uint64_t bytes64 = (uint64_t)count * _slotBytes;
            if (!bytes64 || bytes64 > (uint64_t)NSUIntegerMax ||
                _allocatedBytes > _budgetBytes - bytes64) {
                ffn_set_error("FFN slab allocation would exceed hard budget");
                return nil;
            }
            id<MTLBuffer> slab = [_device newBufferWithLength:(NSUInteger)bytes64
                                                       options:MTLResourceStorageModeShared];
            if (!slab) {
                ffn_set_error("failed to allocate FFN Metal slab %.2f MiB",
                    (double)bytes64 / (1024.0 * 1024.0));
                return nil;
            }
            slab.label = [NSString stringWithFormat:@"redlite_ffn_slab_%u", newSlab];
            [_slabs addObject:slab];
            _allocatedBytes += bytes64;
        }
        id<MTLBuffer> slab = _slabs[slabIndex];
        const uint64_t inner = (uint64_t)localSlot * _slotBytes;
        if (inner > (uint64_t)NSUIntegerMax) {
            ffn_set_error("FFN slot offset exceeds NSUInteger");
            return nil;
        }
        if (offset) *offset = (NSUInteger)inner;
        return slab;
    }
}

- (void)markSlot:(uint32_t)slot delta:(int)delta {
    if (slot >= _capacity || !_slotInflight) return;
    @synchronized (self) {
        if (delta > 0) {
            _slotInflight[slot] += (uint32_t)delta;
        } else if (delta < 0) {
            const uint32_t dec = (uint32_t)(-delta);
            _slotInflight[slot] = _slotInflight[slot] > dec ? _slotInflight[slot] - dec : 0u;
        }
    }
}

@end

static RMFfnPool *ffn_obj(redmetal_ffn_pool_t handle) {
    if (!handle) {
        ffn_set_error("Red Metal FFN pool is null");
        return nil;
    }
    return (__bridge RMFfnPool *)handle;
}

static id<MTLComputePipelineState> pipeline_for_type(RMFfnPool *p, uint32_t type) {
    if (type == GGML_TYPE_IQ2_XS) return p->_iq2Pipeline;
    if (type == GGML_TYPE_IQ1_M) return p->_iq1Pipeline;
    ffn_set_error("unsupported FFN GGML type %u", type);
    return nil;
}

static id<MTLBuffer> grid_for_type(RMFfnPool *p, uint32_t type) {
    if (type == GGML_TYPE_IQ2_XS) return p->_iq2Grid;
    if (type == GGML_TYPE_IQ1_M) return p->_iq1Grid;
    return nil;
}

static int encode_quant_rows(
        id<MTLComputeCommandEncoder> enc,
        id<MTLComputePipelineState> pipeline,
        id<MTLBuffer> slab,
        NSUInteger slotBase,
        uint64_t matrixOffset,
        uint32_t ncols,
        uint32_t rowStart,
        uint32_t rowCount,
        id<MTLBuffer> input,
        id<MTLBuffer> output,
        id<MTLBuffer> grid) {
    if (!enc || !pipeline || !slab || !input || !output || !grid || !rowCount) return 0;
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:slab offset:slotBase atIndex:0];
    [enc setBytes:&matrixOffset length:sizeof(matrixOffset) atIndex:1];
    [enc setBytes:&ncols length:sizeof(ncols) atIndex:2];
    [enc setBytes:&rowStart length:sizeof(rowStart) atIndex:3];
    [enc setBuffer:input offset:0 atIndex:4];
    [enc setBuffer:output offset:0 atIndex:5];
    [enc setBuffer:grid offset:0 atIndex:6];
    const NSUInteger tg = MIN((NSUInteger)rowCount, (NSUInteger)64u);
    [enc dispatchThreads:MTLSizeMake(rowCount, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    return 1;
}

uint32_t redmetal_ffn_abi_version(void) { return REDMETAL_FFN_ABI_VERSION; }
const char *redmetal_ffn_last_error(void) {
    return g_ffn_error[0] ? g_ffn_error : "unknown Red Metal FFN error";
}

redmetal_ffn_pool_t redmetal_ffn_pool_create(
        const char *model_path,
        uint64_t budget_bytes,
        uint64_t slot_bytes,
        uint32_t slots_per_slab,
        const int8_t *iq2_xs_grid,
        uint32_t iq2_xs_grid_count,
        const int8_t *iq1_m_grid,
        uint32_t iq1_m_grid_count) {
    @autoreleasepool {
        g_ffn_error[0] = '\0';
        RMFfnPool *pool = [[RMFfnPool alloc] initWithModelPath:model_path
                                                   budgetBytes:budget_bytes
                                                     slotBytes:slot_bytes
                                                  slotsPerSlab:slots_per_slab
                                                      iq2Grid:iq2_xs_grid
                                                   iq2GridCount:iq2_xs_grid_count
                                                      iq1Grid:iq1_m_grid
                                                   iq1GridCount:iq1_m_grid_count];
        return pool ? (__bridge_retained void *)pool : NULL;
    }
}

void redmetal_ffn_pool_destroy(redmetal_ffn_pool_t handle) {
    if (!handle) return;
    @autoreleasepool {
        RMFfnPool *pool = (__bridge_transfer RMFfnPool *)handle;
        (void)pool;
    }
}

uint32_t redmetal_ffn_pool_capacity(redmetal_ffn_pool_t handle) {
    RMFfnPool *p = ffn_obj(handle); return p ? p->_capacity : 0;
}
uint32_t redmetal_ffn_pool_slab_count(redmetal_ffn_pool_t handle) {
    RMFfnPool *p = ffn_obj(handle); return p ? (uint32_t)p->_slabs.count : 0;
}
uint64_t redmetal_ffn_pool_allocated_bytes(redmetal_ffn_pool_t handle) {
    RMFfnPool *p = ffn_obj(handle); return p ? p->_allocatedBytes : 0;
}
uint64_t redmetal_ffn_pool_bytes_read(redmetal_ffn_pool_t handle) {
    RMFfnPool *p = ffn_obj(handle); if (!p) return 0; @synchronized (p) { return p->_bytesRead; }
}
uint64_t redmetal_ffn_pool_read_calls(redmetal_ffn_pool_t handle) {
    RMFfnPool *p = ffn_obj(handle); if (!p) return 0; @synchronized (p) { return p->_readCalls; }
}
double redmetal_ffn_pool_read_ms(redmetal_ffn_pool_t handle) {
    RMFfnPool *p = ffn_obj(handle); if (!p) return 0.0; @synchronized (p) { return p->_readMs; }
}
int redmetal_ffn_pool_slot_inflight(redmetal_ffn_pool_t handle, uint32_t slot_id) {
    RMFfnPool *p = ffn_obj(handle);
    if (!p || slot_id >= p->_capacity) return 0;
    @synchronized (p) { return p->_slotInflight[slot_id] != 0; }
}

int redmetal_ffn_pool_load_expert(
        redmetal_ffn_pool_t handle,
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
        RMFfnPool *p = ffn_obj(handle);
        if (!p || slot_id >= p->_capacity) return 0;
        if (redmetal_ffn_pool_slot_inflight(handle, slot_id)) {
            ffn_set_error("refusing to overwrite in-flight FFN slot %u", slot_id);
            return 0;
        }
        if (gate_bytes > UINT64_MAX - up_bytes || gate_bytes + up_bytes > UINT64_MAX - down_bytes) {
            ffn_set_error("FFN expert payload size overflow");
            return 0;
        }
        const uint64_t payload = gate_bytes + up_bytes + down_bytes;
        if (payload > p->_slotBytes) {
            ffn_set_error("FFN expert payload exceeds slot size");
            return 0;
        }
        NSUInteger baseOffset = 0;
        id<MTLBuffer> slab = [p bufferForSlot:slot_id offset:&baseOffset];
        if (!slab) return 0;
        uint8_t *base = (uint8_t *)[slab contents];
        if (!base) {
            ffn_set_error("FFN shared Metal slab is not CPU visible");
            return 0;
        }
        uint64_t localBytes = 0, localCalls = 0;
        const double t0 = ffn_now_ms();
        const int ok =
            ffn_pread_all(p->_fd, p->_fileSize, gate_file_offset, gate_bytes,
                          base + baseOffset, &localBytes, &localCalls) &&
            ffn_pread_all(p->_fd, p->_fileSize, up_file_offset, up_bytes,
                          base + baseOffset + gate_bytes, &localBytes, &localCalls) &&
            ffn_pread_all(p->_fd, p->_fileSize, down_file_offset, down_bytes,
                          base + baseOffset + gate_bytes + up_bytes, &localBytes, &localCalls);
        const double ms = ffn_now_ms() - t0;
        if (!ok) return 0;
        [slab didModifyRange:NSMakeRange(baseOffset, (NSUInteger)payload)];
        @synchronized (p) {
            p->_bytesRead += localBytes;
            p->_readCalls += localCalls;
            p->_readMs += ms;
        }
        if (bytes_read) *bytes_read = localBytes;
        if (elapsed_ms) *elapsed_ms = ms;
        return 1;
    }
}

int redmetal_ffn_pool_slot_addresses(
        redmetal_ffn_pool_t handle,
        uint32_t slot_id,
        uint64_t gate_bytes,
        uint64_t up_bytes,
        uint64_t down_bytes,
        uint64_t *gate_address,
        uint64_t *up_address,
        uint64_t *down_address) {
    @autoreleasepool {
        RMFfnPool *p = ffn_obj(handle);
        if (!p || slot_id >= p->_capacity) return 0;
        if (gate_bytes > UINT64_MAX - up_bytes || gate_bytes + up_bytes > UINT64_MAX - down_bytes ||
            gate_bytes + up_bytes + down_bytes > p->_slotBytes) return 0;
        NSUInteger inner = 0;
        id<MTLBuffer> slab = [p bufferForSlot:slot_id offset:&inner];
        if (!slab) return 0;
        uint64_t base = 0;
        if (@available(macOS 13.0, *)) base = (uint64_t)[slab gpuAddress] + (uint64_t)inner;
        if (!base) {
            ffn_set_error("FFN slab did not expose a GPU virtual address");
            return 0;
        }
        if (gate_address) *gate_address = base;
        if (up_address) *up_address = base + gate_bytes;
        if (down_address) *down_address = base + gate_bytes + up_bytes;
        return 1;
    }
}

int redmetal_ffn_pool_execute_expert(
        redmetal_ffn_pool_t handle,
        uint32_t slot_id,
        uint32_t ggml_type,
        uint64_t gate_bytes,
        uint64_t up_bytes,
        uint64_t down_bytes,
        uint32_t hidden_size,
        uint32_t ffn_size,
        uint32_t output_row_start,
        uint32_t output_row_count,
        const float *input,
        uint32_t input_count,
        float *output,
        uint32_t output_count,
        double *elapsed_ms) {
    @autoreleasepool {
        RMFfnPool *p = ffn_obj(handle);
        if (!p || !input || !output || slot_id >= p->_capacity) return 0;
        id<MTLComputePipelineState> quantPipeline = pipeline_for_type(p, ggml_type);
        id<MTLBuffer> grid = grid_for_type(p, ggml_type);
        if (!quantPipeline || !grid) return 0;
        if (!hidden_size || hidden_size % QK_IQ || !ffn_size || ffn_size % QK_IQ ||
            input_count != hidden_size || !output_row_count || output_row_start >= hidden_size ||
            output_row_count > hidden_size - output_row_start || output_count < output_row_count) {
            ffn_set_error("invalid FFN dimensions hidden=%u ffn=%u start=%u count=%u",
                hidden_size, ffn_size, output_row_start, output_row_count);
            return 0;
        }
        const uint64_t blockBytes = ggml_type == GGML_TYPE_IQ2_XS ? 74u : 56u;
        const uint64_t gateRowBytes = (uint64_t)(hidden_size / QK_IQ) * blockBytes;
        const uint64_t downRowBytes = (uint64_t)(ffn_size / QK_IQ) * blockBytes;
        const uint64_t expectedGate = (uint64_t)ffn_size * gateRowBytes;
        const uint64_t expectedDown = (uint64_t)hidden_size * downRowBytes;
        if (gate_bytes != expectedGate || up_bytes != expectedGate || down_bytes != expectedDown) {
            ffn_set_error("FFN matrix byte mismatch gate=%llu/%llu up=%llu/%llu down=%llu/%llu",
                (unsigned long long)gate_bytes, (unsigned long long)expectedGate,
                (unsigned long long)up_bytes, (unsigned long long)expectedGate,
                (unsigned long long)down_bytes, (unsigned long long)expectedDown);
            return 0;
        }
        if (gate_bytes > UINT64_MAX - up_bytes || gate_bytes + up_bytes > UINT64_MAX - down_bytes ||
            gate_bytes + up_bytes + down_bytes > p->_slotBytes) {
            ffn_set_error("FFN resident expert exceeds slot bounds");
            return 0;
        }

        NSUInteger slotBase = 0;
        id<MTLBuffer> slab = [p bufferForSlot:slot_id offset:&slotBase];
        if (!slab) return 0;
        id<MTLBuffer> inBuffer = [p->_device newBufferWithBytes:input
                                                        length:(NSUInteger)hidden_size * sizeof(float)
                                                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> gateBuffer = [p->_device newBufferWithLength:(NSUInteger)ffn_size * sizeof(float)
                                                            options:MTLResourceStorageModeShared];
        id<MTLBuffer> upBuffer = [p->_device newBufferWithLength:(NSUInteger)ffn_size * sizeof(float)
                                                          options:MTLResourceStorageModeShared];
        id<MTLBuffer> actBuffer = [p->_device newBufferWithLength:(NSUInteger)ffn_size * sizeof(float)
                                                           options:MTLResourceStorageModeShared];
        id<MTLBuffer> outBuffer = [p->_device newBufferWithLength:(NSUInteger)output_row_count * sizeof(float)
                                                           options:MTLResourceStorageModeShared];
        if (!inBuffer || !gateBuffer || !upBuffer || !actBuffer || !outBuffer) {
            ffn_set_error("failed to allocate FFN activation buffers");
            return 0;
        }
        memset([outBuffer contents], 0, (size_t)output_row_count * sizeof(float));

        id<MTLCommandBuffer> cb = [p->_queue commandBuffer];
        if (!cb) {
            ffn_set_error("failed to create FFN command buffer");
            return 0;
        }
        id<MTLComputeCommandEncoder> qenc = [cb computeCommandEncoder];
        if (!qenc) {
            ffn_set_error("failed to create gate/up FFN encoder");
            return 0;
        }
        if (!encode_quant_rows(qenc, quantPipeline, slab, slotBase, 0u,
                               hidden_size, 0u, ffn_size, inBuffer, gateBuffer, grid) ||
            !encode_quant_rows(qenc, quantPipeline, slab, slotBase, gate_bytes,
                               hidden_size, 0u, ffn_size, inBuffer, upBuffer, grid)) {
            [qenc endEncoding];
            ffn_set_error("failed to encode gate/up FFN matvec");
            return 0;
        }
        [qenc endEncoding];

        id<MTLComputeCommandEncoder> aenc = [cb computeCommandEncoder];
        if (!aenc) {
            ffn_set_error("failed to create SiLU FFN encoder");
            return 0;
        }
        [aenc setComputePipelineState:p->_siluPipeline];
        [aenc setBuffer:gateBuffer offset:0 atIndex:0];
        [aenc setBuffer:upBuffer offset:0 atIndex:1];
        [aenc setBuffer:actBuffer offset:0 atIndex:2];
        const NSUInteger atg = MIN((NSUInteger)ffn_size, (NSUInteger)64u);
        [aenc dispatchThreads:MTLSizeMake(ffn_size, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(atg, 1, 1)];
        [aenc endEncoding];

        id<MTLComputeCommandEncoder> denc = [cb computeCommandEncoder];
        if (!denc) {
            ffn_set_error("failed to create down FFN encoder");
            return 0;
        }
        const uint64_t downOffset = gate_bytes + up_bytes;
        if (!encode_quant_rows(denc, quantPipeline, slab, slotBase, downOffset,
                               ffn_size, output_row_start, output_row_count,
                               actBuffer, outBuffer, grid)) {
            [denc endEncoding];
            ffn_set_error("failed to encode down FFN matvec");
            return 0;
        }
        [denc endEncoding];

        [p markSlot:slot_id delta:1];
        RMFfnPool *keepPool = p;
        [cb addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
            (void)buffer;
            [keepPool markSlot:slot_id delta:-1];
        }];
        const double t0 = ffn_now_ms();
        [cb commit];
        [cb waitUntilCompleted];
        const double ms = ffn_now_ms() - t0;
        if (cb.status == MTLCommandBufferStatusError) {
            ffn_set_error("FFN Metal command failed: %s",
                cb.error.localizedDescription.UTF8String ?: "unknown Metal error");
            return 0;
        }
        memcpy(output, [outBuffer contents], (size_t)output_row_count * sizeof(float));
        if (elapsed_ms) *elapsed_ms = ms;
        return 1;
    }
}
