#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "redmetal_exec.h"

#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/stat.h>

#define REDMETAL_EXEC_ABI_VERSION 1u
#define REDMETAL_MAX_LAYERS 128u
#define REDMETAL_MAX_EXPERTS 512u

static __thread char g_exec_error[512];

static void exec_set_error(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_exec_error, sizeof(g_exec_error), fmt, ap);
    va_end(ap);
}

static double exec_now_ms(void) {
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

static int exec_pread_all(
        int fd,
        uint64_t file_size,
        uint64_t offset,
        uint64_t len,
        uint8_t *dst,
        uint64_t *bytes_read,
        uint64_t *read_calls) {
    if (!dst || offset > file_size || len > file_size - offset) {
        exec_set_error(
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
            exec_set_error(
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

/* Canonical IQ2_XXS grid from ggml, stored in the same compact 2-bit form used
 * by gguf-py. Each decoded element maps 0->0x08, 1->0x19, 2->0x2b. */
static const char kIQ2GridHex[] =
    "00000200050008000a00110014002000220028002a0041004400500058006100"
    "6400800082008a00a20001010401100115014001840198010002020222028202"
    "010404041004210424044004420448046004810484049004a404000502050805"
    "200546056905800591050906100640068406a406000805080808140828084108"
    "440850085208880804094009020a140a01100410101021104010601084109010"
    "951000110811201150115a118011241245120014081420142514491480141815"
    "6215001616160118041810184018811800190519a019511a002002200a204420"
    "6120802082202921482100220222012404241024402456240025412564259026"
    "082820289428442a014004401040184021402440404048405640604081408440"
    "9040004120416141804185410142104248425642684200440844204480449944"
    "124524450046014804481048404845480049584961498249454a904a00500850"
    "1150195020508050885004514251a4519152905492540a550156545600581158"
    "195864584059085a046010604060686000615561186260620064056410651265"
    "84654268008002800a8041808280048118814081118201840484108415844084"
    "608400854685948509864086608602880489118a0490109024904090a1901691"
    "8091459200942294449451958198209902a050a085a009a100a218a450a804a9";

static int hex_nibble(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static int build_iq2_grid(uint8_t out[256 * 8]) {
    const size_t hex_len = strlen(kIQ2GridHex);
    if (hex_len != 1024u) {
        exec_set_error("internal IQ2 grid hex has unexpected length %zu", hex_len);
        return 0;
    }
    static const uint8_t map[3] = {0x08u, 0x19u, 0x2bu};
    size_t out_pos = 0;
    for (size_t i = 0; i < hex_len; i += 2) {
        const int hi = hex_nibble(kIQ2GridHex[i]);
        const int lo = hex_nibble(kIQ2GridHex[i + 1]);
        if (hi < 0 || lo < 0) {
            exec_set_error("invalid IQ2 grid hex");
            return 0;
        }
        const uint8_t packed = (uint8_t)((hi << 4) | lo);
        for (uint32_t shift = 0; shift < 8; shift += 2) {
            const uint8_t code = (packed >> shift) & 3u;
            if (code >= 3u || out_pos >= 256u * 8u) {
                exec_set_error("invalid IQ2 grid code");
                return 0;
            }
            out[out_pos++] = map[code];
        }
    }
    if (out_pos != 256u * 8u) {
        exec_set_error("IQ2 grid expansion produced %zu bytes", out_pos);
        return 0;
    }
    return 1;
}

static NSString * const kIQ2Source = @
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"kernel void redmetal_iq2_xxs_rows(\n"
"    device const uchar *slot [[buffer(0)]],\n"
"    constant ulong &matrix_offset [[buffer(1)]],\n"
"    constant uint &ncols [[buffer(2)]],\n"
"    constant uint &row_start [[buffer(3)]],\n"
"    device const float *x [[buffer(4)]],\n"
"    device float *out [[buffer(5)]],\n"
"    device const uchar *grid_table [[buffer(6)]],\n"
"    uint gid [[thread_position_in_grid]]) {\n"
"    const uint row = row_start + gid;\n"
"    const uint blocks = ncols / 256u;\n"
"    const ulong row_bytes = ulong(blocks) * 66ul;\n"
"    device const uchar *rowp = slot + matrix_offset + ulong(row) * row_bytes;\n"
"    float acc = 0.0f;\n"
"    for (uint ib = 0; ib < blocks; ++ib) {\n"
"        device const uchar *bp = rowp + ulong(ib) * 66ul;\n"
"        const ushort dbits = *(device const ushort *)bp;\n"
"        const float d = float(as_type<half>(dbits));\n"
"        device const ushort *q = (device const ushort *)(bp + 2);\n"
"        for (uint g = 0; g < 8u; ++g) {\n"
"            const uint qi = 4u * g;\n"
"            const uint auxg = uint(q[qi]) | (uint(q[qi + 1]) << 16);\n"
"            const uint auxs = uint(q[qi + 2]) | (uint(q[qi + 3]) << 16);\n"
"            const float db = d * (0.5f + float(auxs >> 28)) * 0.25f;\n"
"            for (uint l = 0; l < 4u; ++l) {\n"
"                const uint grid_index = (auxg >> (8u * l)) & 255u;\n"
"                const uint sign7 = (auxs >> (7u * l)) & 127u;\n"
"                const uint sign8 = sign7 | ((popcount(sign7) & 1u) << 7);\n"
"                device const uchar *gv = grid_table + grid_index * 8u;\n"
"                const uint xbase = ib * 256u + g * 32u + l * 8u;\n"
"                for (uint j = 0; j < 8u; ++j) {\n"
"                    const float s = (sign8 & (1u << j)) ? -1.0f : 1.0f;\n"
"                    acc += x[xbase + j] * (db * float(gv[j]) * s);\n"
"                }\n"
"            }\n"
"        }\n"
"    }\n"
"    out[gid] = acc;\n"
"}\n";

@interface RMExecPool : NSObject {
@public
    id<MTLDevice> _device;
    id<MTLCommandQueue> _queue;
    id<MTLComputePipelineState> _iq2Pipeline;
    id<MTLBuffer> _gridBuffer;
    NSMutableArray<id<MTLBuffer>> *_slabs;
    __strong id<MTLBuffer> _gateTables[REDMETAL_MAX_LAYERS];
    __strong id<MTLBuffer> _upTables[REDMETAL_MAX_LAYERS];
    __strong id<MTLBuffer> _downTables[REDMETAL_MAX_LAYERS];

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
                      slotsPerSlab:(uint32_t)slotsPerSlab;
- (id<MTLBuffer>)bufferForSlot:(uint32_t)slot offset:(NSUInteger *)offset;
- (int)ensureAddressTables:(uint32_t)layer;
- (void)markSlot:(uint32_t)slot inflightDelta:(int)delta;
@end

@implementation RMExecPool

- (instancetype)initWithModelPath:(const char *)path
                       budgetBytes:(uint64_t)budget
                         slotBytes:(uint64_t)slotBytes
                      slotsPerSlab:(uint32_t)slotsPerSlab {
    self = [super init];
    if (!self) return nil;
    _fd = -1;
    if (!path || !path[0] || budget == 0 || slotBytes == 0) {
        exec_set_error("invalid Red Metal execution pool arguments");
        return nil;
    }
    const uint64_t page = 4096u;
    const uint64_t aligned = round_up_u64(slotBytes, page);
    if (!aligned || aligned > budget) {
        exec_set_error("cache budget cannot fit one aligned Metal expert slot");
        return nil;
    }
    _slotBytes = aligned;
    _budgetBytes = budget;
    _capacity = (uint32_t)(budget / aligned);
    if (!_capacity) {
        exec_set_error("execution pool has zero slots");
        return nil;
    }
    _slotsPerSlab = slotsPerSlab ? slotsPerSlab : 64u;
    if (_slotsPerSlab > _capacity) _slotsPerSlab = _capacity;

    _device = MTLCreateSystemDefaultDevice();
    if (!_device || ![_device hasUnifiedMemory]) {
        exec_set_error("Red Metal execution requires Apple unified memory");
        return nil;
    }
    _queue = [_device newCommandQueue];
    if (!_queue) {
        exec_set_error("failed to create Red Metal execution queue");
        return nil;
    }

    NSError *libraryError = nil;
    id<MTLLibrary> library = [_device newLibraryWithSource:kIQ2Source options:nil error:&libraryError];
    if (!library) {
        exec_set_error("failed to compile IQ2 kernel: %s",
            libraryError.localizedDescription.UTF8String ?: "unknown error");
        return nil;
    }
    id<MTLFunction> function = [library newFunctionWithName:@"redmetal_iq2_xxs_rows"];
    if (!function) {
        exec_set_error("IQ2 Metal function not found");
        return nil;
    }
    NSError *pipelineError = nil;
    _iq2Pipeline = [_device newComputePipelineStateWithFunction:function error:&pipelineError];
    if (!_iq2Pipeline) {
        exec_set_error("failed to create IQ2 pipeline: %s",
            pipelineError.localizedDescription.UTF8String ?: "unknown error");
        return nil;
    }

    uint8_t grid[256 * 8];
    if (!build_iq2_grid(grid)) return nil;
    _gridBuffer = [_device newBufferWithBytes:grid
                                       length:sizeof(grid)
                                      options:MTLResourceStorageModeShared];
    if (!_gridBuffer) {
        exec_set_error("failed to allocate IQ2 grid buffer");
        return nil;
    }

    _fd = open(path, O_RDONLY);
    if (_fd < 0) {
        exec_set_error("open(%s) failed: %s", path, strerror(errno));
        return nil;
    }
    struct stat st;
    if (fstat(_fd, &st) != 0 || st.st_size <= 0) {
        exec_set_error("failed to stat model file: %s", strerror(errno));
        close(_fd);
        _fd = -1;
        return nil;
    }
    _fileSize = (uint64_t)st.st_size;
    _slotInflight = calloc(_capacity, sizeof(uint32_t));
    if (!_slotInflight) {
        exec_set_error("failed to allocate slot in-flight table");
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
        exec_set_error("slot %u exceeds capacity %u", slot, _capacity);
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
                exec_set_error("Metal slab allocation would exceed hard budget");
                return nil;
            }
            id<MTLBuffer> slab = [_device newBufferWithLength:(NSUInteger)bytes64
                                                       options:MTLResourceStorageModeShared];
            if (!slab) {
                exec_set_error("failed to allocate execution slab %.2f MiB",
                    (double)bytes64 / (1024.0 * 1024.0));
                return nil;
            }
            slab.label = [NSString stringWithFormat:@"redlite_exec_slab_%u", newSlab];
            [_slabs addObject:slab];
            _allocatedBytes += bytes64;
        }
        id<MTLBuffer> slab = _slabs[slabIndex];
        const uint64_t inner = (uint64_t)localSlot * _slotBytes;
        if (inner > (uint64_t)NSUIntegerMax) {
            exec_set_error("slot offset exceeds NSUInteger");
            return nil;
        }
        if (offset) *offset = (NSUInteger)inner;
        return slab;
    }
}

- (int)ensureAddressTables:(uint32_t)layer {
    if (layer >= REDMETAL_MAX_LAYERS) {
        exec_set_error("layer %u exceeds address-table limit", layer);
        return 0;
    }
    @synchronized (self) {
        if (_gateTables[layer]) return 1;
        const NSUInteger bytes = REDMETAL_MAX_EXPERTS * sizeof(uint64_t);
        _gateTables[layer] = [_device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        _upTables[layer] = [_device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        _downTables[layer] = [_device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (!_gateTables[layer] || !_upTables[layer] || !_downTables[layer]) {
            _gateTables[layer] = nil;
            _upTables[layer] = nil;
            _downTables[layer] = nil;
            exec_set_error("failed to allocate layer %u GPU address tables", layer);
            return 0;
        }
        memset([_gateTables[layer] contents], 0, bytes);
        memset([_upTables[layer] contents], 0, bytes);
        memset([_downTables[layer] contents], 0, bytes);
        return 1;
    }
}

- (void)markSlot:(uint32_t)slot inflightDelta:(int)delta {
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

static RMExecPool *exec_obj(redmetal_exec_pool_t handle) {
    if (!handle) {
        exec_set_error("Red Metal execution pool is null");
        return nil;
    }
    return (__bridge RMExecPool *)handle;
}

uint32_t redmetal_exec_abi_version(void) { return REDMETAL_EXEC_ABI_VERSION; }
const char *redmetal_exec_last_error(void) {
    return g_exec_error[0] ? g_exec_error : "unknown Red Metal execution error";
}

redmetal_exec_pool_t redmetal_exec_pool_create(
        const char *model_path,
        uint64_t budget_bytes,
        uint64_t slot_bytes,
        uint32_t slots_per_slab) {
    @autoreleasepool {
        g_exec_error[0] = '\0';
        RMExecPool *pool = [[RMExecPool alloc] initWithModelPath:model_path
                                                     budgetBytes:budget_bytes
                                                       slotBytes:slot_bytes
                                                    slotsPerSlab:slots_per_slab];
        return pool ? (__bridge_retained void *)pool : NULL;
    }
}

void redmetal_exec_pool_destroy(redmetal_exec_pool_t handle) {
    if (!handle) return;
    @autoreleasepool {
        RMExecPool *pool = (__bridge_transfer RMExecPool *)handle;
        (void)pool;
    }
}

uint32_t redmetal_exec_pool_capacity(redmetal_exec_pool_t handle) {
    RMExecPool *p = exec_obj(handle); return p ? p->_capacity : 0;
}
uint32_t redmetal_exec_pool_slab_count(redmetal_exec_pool_t handle) {
    RMExecPool *p = exec_obj(handle); return p ? (uint32_t)p->_slabs.count : 0;
}
uint64_t redmetal_exec_pool_allocated_bytes(redmetal_exec_pool_t handle) {
    RMExecPool *p = exec_obj(handle); return p ? p->_allocatedBytes : 0;
}
uint64_t redmetal_exec_pool_bytes_read(redmetal_exec_pool_t handle) {
    RMExecPool *p = exec_obj(handle); if (!p) return 0; @synchronized (p) { return p->_bytesRead; }
}
uint64_t redmetal_exec_pool_read_calls(redmetal_exec_pool_t handle) {
    RMExecPool *p = exec_obj(handle); if (!p) return 0; @synchronized (p) { return p->_readCalls; }
}
double redmetal_exec_pool_read_ms(redmetal_exec_pool_t handle) {
    RMExecPool *p = exec_obj(handle); if (!p) return 0.0; @synchronized (p) { return p->_readMs; }
}

int redmetal_exec_pool_slot_inflight(redmetal_exec_pool_t handle, uint32_t slot_id) {
    RMExecPool *p = exec_obj(handle);
    if (!p || slot_id >= p->_capacity) return 0;
    @synchronized (p) { return p->_slotInflight[slot_id] != 0; }
}

int redmetal_exec_pool_load_expert(
        redmetal_exec_pool_t handle,
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
        RMExecPool *p = exec_obj(handle);
        if (!p) return 0;
        if (slot_id >= p->_capacity) {
            exec_set_error("slot %u exceeds capacity %u", slot_id, p->_capacity);
            return 0;
        }
        if (redmetal_exec_pool_slot_inflight(handle, slot_id)) {
            exec_set_error("refusing to overwrite in-flight Metal slot %u", slot_id);
            return 0;
        }
        if (gate_bytes > UINT64_MAX - up_bytes || gate_bytes + up_bytes > UINT64_MAX - down_bytes) {
            exec_set_error("expert payload size overflow");
            return 0;
        }
        const uint64_t payload = gate_bytes + up_bytes + down_bytes;
        if (payload > p->_slotBytes) {
            exec_set_error("expert payload exceeds aligned Metal slot");
            return 0;
        }
        NSUInteger baseOffset = 0;
        id<MTLBuffer> slab = [p bufferForSlot:slot_id offset:&baseOffset];
        if (!slab) return 0;
        uint8_t *base = (uint8_t *)[slab contents];
        if (!base) {
            exec_set_error("shared Metal slab is not CPU visible");
            return 0;
        }
        uint64_t localBytes = 0, localCalls = 0;
        const double t0 = exec_now_ms();
        const int ok =
            exec_pread_all(p->_fd, p->_fileSize, gate_file_offset, gate_bytes,
                           base + baseOffset, &localBytes, &localCalls) &&
            exec_pread_all(p->_fd, p->_fileSize, up_file_offset, up_bytes,
                           base + baseOffset + gate_bytes, &localBytes, &localCalls) &&
            exec_pread_all(p->_fd, p->_fileSize, down_file_offset, down_bytes,
                           base + baseOffset + gate_bytes + up_bytes, &localBytes, &localCalls);
        const double ms = exec_now_ms() - t0;
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

int redmetal_exec_pool_bind_expert(
        redmetal_exec_pool_t handle,
        uint32_t layer,
        uint32_t expert,
        uint32_t slot_id,
        uint64_t gate_bytes,
        uint64_t up_bytes,
        uint64_t down_bytes) {
    @autoreleasepool {
        RMExecPool *p = exec_obj(handle);
        if (!p || expert >= REDMETAL_MAX_EXPERTS || slot_id >= p->_capacity) {
            exec_set_error("invalid expert binding layer=%u expert=%u slot=%u", layer, expert, slot_id);
            return 0;
        }
        if (![p ensureAddressTables:layer]) return 0;
        NSUInteger inner = 0;
        id<MTLBuffer> slab = [p bufferForSlot:slot_id offset:&inner];
        if (!slab) return 0;
        if (gate_bytes > UINT64_MAX - up_bytes || gate_bytes + up_bytes > UINT64_MAX - down_bytes ||
            gate_bytes + up_bytes + down_bytes > p->_slotBytes) {
            exec_set_error("invalid expert binding sizes");
            return 0;
        }
        uint64_t gpuBase = 0;
        if (@available(macOS 13.0, *)) {
            gpuBase = (uint64_t)[slab gpuAddress] + (uint64_t)inner;
        } else {
            exec_set_error("GPU virtual addresses require macOS 13 or newer");
            return 0;
        }
        if (!gpuBase) {
            exec_set_error("Metal slab did not expose a GPU virtual address");
            return 0;
        }
        uint64_t *gate = (uint64_t *)[p->_gateTables[layer] contents];
        uint64_t *up = (uint64_t *)[p->_upTables[layer] contents];
        uint64_t *down = (uint64_t *)[p->_downTables[layer] contents];
        gate[expert] = gpuBase;
        up[expert] = gpuBase + gate_bytes;
        down[expert] = gpuBase + gate_bytes + up_bytes;
        const NSRange r = NSMakeRange((NSUInteger)expert * sizeof(uint64_t), sizeof(uint64_t));
        [p->_gateTables[layer] didModifyRange:r];
        [p->_upTables[layer] didModifyRange:r];
        [p->_downTables[layer] didModifyRange:r];
        return 1;
    }
}

int redmetal_exec_pool_unbind_expert(
        redmetal_exec_pool_t handle,
        uint32_t layer,
        uint32_t expert) {
    @autoreleasepool {
        RMExecPool *p = exec_obj(handle);
        if (!p || layer >= REDMETAL_MAX_LAYERS || expert >= REDMETAL_MAX_EXPERTS) return 0;
        if (!p->_gateTables[layer]) return 1;
        ((uint64_t *)[p->_gateTables[layer] contents])[expert] = 0;
        ((uint64_t *)[p->_upTables[layer] contents])[expert] = 0;
        ((uint64_t *)[p->_downTables[layer] contents])[expert] = 0;
        const NSRange r = NSMakeRange((NSUInteger)expert * sizeof(uint64_t), sizeof(uint64_t));
        [p->_gateTables[layer] didModifyRange:r];
        [p->_upTables[layer] didModifyRange:r];
        [p->_downTables[layer] didModifyRange:r];
        return 1;
    }
}

int redmetal_exec_pool_bound_addresses(
        redmetal_exec_pool_t handle,
        uint32_t layer,
        uint32_t expert,
        uint64_t *gate_address,
        uint64_t *up_address,
        uint64_t *down_address) {
    RMExecPool *p = exec_obj(handle);
    if (!p || layer >= REDMETAL_MAX_LAYERS || expert >= REDMETAL_MAX_EXPERTS ||
        !p->_gateTables[layer]) return 0;
    const uint64_t g = ((uint64_t *)[p->_gateTables[layer] contents])[expert];
    const uint64_t u = ((uint64_t *)[p->_upTables[layer] contents])[expert];
    const uint64_t d = ((uint64_t *)[p->_downTables[layer] contents])[expert];
    if (!g || !u || !d) return 0;
    if (gate_address) *gate_address = g;
    if (up_address) *up_address = u;
    if (down_address) *down_address = d;
    return 1;
}

int redmetal_exec_pool_iq2_xxs_rows(
        redmetal_exec_pool_t handle,
        uint32_t slot_id,
        uint64_t matrix_slot_offset,
        uint32_t ncols,
        uint32_t nrows,
        uint32_t row_start,
        uint32_t row_count,
        const float *input,
        uint32_t input_count,
        float *output,
        uint32_t output_count,
        double *elapsed_ms) {
    @autoreleasepool {
        RMExecPool *p = exec_obj(handle);
        if (!p || !input || !output || slot_id >= p->_capacity) return 0;
        if (!ncols || ncols % 256u != 0 || !nrows || !row_count || row_start >= nrows ||
            row_count > nrows - row_start || input_count != ncols || output_count < row_count) {
            exec_set_error("invalid IQ2 matvec dimensions cols=%u rows=%u start=%u count=%u", ncols, nrows, row_start, row_count);
            return 0;
        }
        const uint64_t rowBytes = (uint64_t)(ncols / 256u) * 66u;
        if (matrix_slot_offset > p->_slotBytes ||
            (uint64_t)nrows > (p->_slotBytes - matrix_slot_offset) / rowBytes) {
            exec_set_error("IQ2 matrix exceeds expert slot bounds");
            return 0;
        }
        NSUInteger slotBase = 0;
        id<MTLBuffer> slab = [p bufferForSlot:slot_id offset:&slotBase];
        if (!slab) return 0;
        id<MTLBuffer> inBuffer = [p->_device newBufferWithBytes:input
                                                        length:(NSUInteger)ncols * sizeof(float)
                                                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> outBuffer = [p->_device newBufferWithLength:(NSUInteger)row_count * sizeof(float)
                                                           options:MTLResourceStorageModeShared];
        if (!inBuffer || !outBuffer) {
            exec_set_error("failed to allocate IQ2 input/output Metal buffers");
            return 0;
        }
        memset([outBuffer contents], 0, (size_t)row_count * sizeof(float));
        id<MTLCommandBuffer> cb = [p->_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (!cb || !enc) {
            exec_set_error("failed to create IQ2 Metal command encoder");
            return 0;
        }
        [p markSlot:slot_id inflightDelta:1];
        RMExecPool *keepPool = p;
        [cb addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
            (void)buffer;
            [keepPool markSlot:slot_id inflightDelta:-1];
        }];
        [enc setComputePipelineState:p->_iq2Pipeline];
        [enc setBuffer:slab offset:slotBase atIndex:0];
        const uint64_t matrixOffset = matrix_slot_offset;
        const uint32_t cols = ncols;
        const uint32_t start = row_start;
        [enc setBytes:&matrixOffset length:sizeof(matrixOffset) atIndex:1];
        [enc setBytes:&cols length:sizeof(cols) atIndex:2];
        [enc setBytes:&start length:sizeof(start) atIndex:3];
        [enc setBuffer:inBuffer offset:0 atIndex:4];
        [enc setBuffer:outBuffer offset:0 atIndex:5];
        [enc setBuffer:p->_gridBuffer offset:0 atIndex:6];
        const NSUInteger tg = MIN((NSUInteger)row_count, (NSUInteger)64u);
        [enc dispatchThreads:MTLSizeMake(row_count, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        const double t0 = exec_now_ms();
        [cb commit];
        [cb waitUntilCompleted];
        const double ms = exec_now_ms() - t0;
        if (cb.status == MTLCommandBufferStatusError) {
            exec_set_error("IQ2 Metal command failed: %s",
                cb.error.localizedDescription.UTF8String ?: "unknown Metal error");
            return 0;
        }
        memcpy(output, [outBuffer contents], (size_t)row_count * sizeof(float));
        if (elapsed_ms) *elapsed_ms = ms;
        return 1;
    }
}
