#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "redlite_native_iq3.h"

#include "redmetal_topk.h"

#include <errno.h>
#include <fcntl.h>
#ifndef F_NOCACHE
#define F_NOCACHE 48   /* <sys/fcntl.h> hides it under _POSIX_C_SOURCE */
#endif
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/stat.h>

/*
 * Resident top-k routed-expert pool.
 *
 * dev18 execution model: every selected expert of a layer is executed by three
 * batched dispatches (gate+up+SiLU, down, weighted sum) instead of ten
 * sequential per-expert encoders. Each matrix row is reduced by a group of
 * SIMD lanes (one 256-value quant block per lane, simd_shuffle_xor tree). The
 * per-block IQ2_XS / IQ1_M decode arithmetic is unchanged from the
 * field-validated dev10/dev11 kernels; only the parallel decomposition and the
 * float summation order differ (validated by routed/FFN parity and the
 * engine's CPU-oracle and pinned-llama.cpp comparisons).
 *
 * Slot pointers are passed through a Metal 3 argument buffer filled with GPU
 * virtual addresses (buffer.gpuAddress + slot offset); the referenced slabs
 * are made resident with useResource.
 */

#define REDMETAL_TOPK_ABI_VERSION 1u
#define GGML_TYPE_IQ2_XS 17u
#define GGML_TYPE_IQ1_M 29u
/* dev31 type words (see rm_gate_type / rm_down_type in the kernel source) */
static uint32_t tw_gate(uint32_t tw) { return tw & 255u; }
static uint32_t tw_down(uint32_t tw) { return (tw >> 8) ? (tw >> 8) : (tw & 255u); }
static uint64_t tw_block_bytes(uint32_t t) { return t == GGML_TYPE_IQ2_XS ? 74u : t == GGML_TYPE_IQ1_M ? 56u : rl_iq3_block_bytes(t); }
static int tw_ok(uint32_t tw) {
    const uint32_t g = tw_gate(tw), d = tw_down(tw);
    if (tw >> 16 || !tw_block_bytes(g) || !tw_block_bytes(d)) return 0;
    /* one grid buffer serves every dispatch of a plan: two different legacy grids cannot be mixed */
    const int lg = g == GGML_TYPE_IQ2_XS || g == GGML_TYPE_IQ1_M, ld = d == GGML_TYPE_IQ2_XS || d == GGML_TYPE_IQ1_M;
    return !(lg && ld && g != d);
}
#define QK_IQ 256u
#define REDMETAL_TOPK_SLICE 4u   /* must match RM_S in the kernel source */
#define REDMETAL_TOPK_MM_SLICE 16u  /* pairs per tile of the dev30 matrix kernels (their xs[16 * 32]) */
#define REDMETAL_TOPK_MAX 512u   /* experts per encoded plan (a batched-prefill union of several tokens) */

static __thread char g_topk_error[512];

static void topk_set_error(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_topk_error, sizeof(g_topk_error), fmt, ap);
    va_end(ap);
}

static double topk_now_ms(void) {
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

static int topk_pread_all(
        int fd,
        uint64_t file_size,
        uint64_t offset,
        uint64_t len,
        uint8_t *dst,
        uint64_t *bytes_read,
        uint64_t *read_calls) {
    if (!dst || offset > file_size || len > file_size - offset) {
        topk_set_error("invalid top-k pread range offset=%llu len=%llu file_size=%llu",
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
            topk_set_error("top-k pread failed at offset=%llu: %s",
                (unsigned long long)(offset + pos),
                nread < 0 ? strerror(errno) : "unexpected EOF");
            return 0;
        }
        pos += (uint64_t)nread;
        if (bytes_read) *bytes_read += (uint64_t)nread;
    }
    return 1;
}

static NSString * const kTopKSource = @
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"constant bool redmetal_guard [[function_constant(0)]];   /* dev23: early-out variant (flag at index 30) */\n"
/* dev45: two activation vectors (2-row MTP verify): experts from x_split on read x1 */
"constant bool redmetal_two_x_in [[function_constant(1)]];\n"
"constant bool redmetal_two_x = is_function_constant_defined(redmetal_two_x_in) && redmetal_two_x_in;\n"
"\n"
"struct redmetal_topk_slots { device const uchar *slot[512]; };\n"
"#define RM_S 4u   /* pairs per slice: one decoded weight serves RM_S tokens */\n"
"struct redmetal_topk_slots3 { device const uchar *gate[512]; device const uchar *up[512]; device const uchar *down[512]; };\n"
"\n"
"inline uint redlite_sign8(uint sign7) {\n"
"    sign7 &= 127u;\n"
"    return sign7 | ((popcount(sign7) & 1u) << 7);\n"
"}\n"
/* dev31: the type argument is a type word: gate/up type in bits 0-7, the down type in bits 8-15 when it differs
 * (the IQ3_XXS GGUF has IQ3_S down projections in some layers); IQ3 types decode through rl_iq3_group8 */
"inline uint rm_gate_type(uint tw) { return tw & 255u; }\n"
"inline uint rm_down_type(uint tw) { return (tw >> 8) != 0u ? (tw >> 8) : (tw & 255u); }\n"
"inline ulong rm_block_bytes(uint t) { return t == 17u ? 74ul : t == 29u ? 56ul : ulong(rl_iq3_block_bytes(t)); }\n"
"\n"
/* one IQ2_XS 256-value block dot (validated dev10 arithmetic) */
"inline float redmetal_topk_iq2_xs_block(device const uchar *bp, device const float *x, device const char *grid, uint part, uint nparts) {\n"
"    const float d = float(as_type<half>(*(device const ushort *)bp));\n"
"    device const ushort *qs = (device const ushort *)(bp + 2);\n"
"    device const uchar *scales = bp + 66;\n"
"    float acc = 0.0f;\n"
"    const uint gper = 16u / nparts;\n"
"    for (uint g = part * gper; g < (part + 1u) * gper; ++g) {\n"
"        const uint scale = (uint(scales[g >> 1]) >> (4u * (g & 1u))) & 15u;\n"
"        float a = 0.0f;\n"
"        for (uint h = 0; h < 2u; ++h) {\n"
"            const uint q = uint(qs[2u * g + h]);\n"
"            const uint signs = redlite_sign8(q >> 9);\n"
"            device const char4 *gv = (device const char4 *)(grid + (q & 511u) * 8u);\n"
"            device const float4 *xv = (device const float4 *)(x + g * 16u + h * 8u);\n"
"            const float4 s0 = select(float4(1.0f), float4(-1.0f), ((uint4(signs) >> uint4(0u, 1u, 2u, 3u)) & 1u) != 0u);\n"
"            const float4 s1 = select(float4(1.0f), float4(-1.0f), ((uint4(signs) >> uint4(4u, 5u, 6u, 7u)) & 1u) != 0u);\n"
"            a += dot(xv[0], float4(gv[0]) * s0) + dot(xv[1], float4(gv[1]) * s1);\n"
"        }\n"
"        acc += (d * (0.5f + float(scale)) * 0.25f) * a;\n"
"    }\n"
"    return acc;\n"
"}\n"
"\n"
"inline float redmetal_topk_iq1_m_block(device const uchar *bp, device const float *x, device const char *grid, uint part, uint nparts) {\n"
"    device const uchar *qs = bp;\n"
"    device const uchar *qh = bp + 32;\n"
"    device const ushort *sc = (device const ushort *)(bp + 48);\n"
"    const ushort dbits = ushort(sc[0] >> 12) | ushort((sc[1] >> 8) & 0x00f0u) |\n"
"                         ushort((sc[2] >> 4) & 0x0f00u) | ushort(sc[3] & 0xf000u);\n"
"    const float d = float(as_type<half>(dbits));\n"
"    float acc = 0.0f;\n"
"    const uint gper = 32u / nparts;\n"
"    for (uint g = part * gper; g < (part + 1u) * gper; ++g) {\n"
"        const uint nibble = (uint(qh[g >> 1]) >> (4u * (g & 1u))) & 15u;\n"
"        const uint grid_index = uint(qs[g]) | ((nibble & 7u) << 8);\n"
"        const float delta = (nibble & 8u) ? -0.125f : 0.125f;\n"
"        const uint scale_index = g >> 1;\n"
"        const uint scale = (uint(sc[scale_index >> 2]) >> (3u * (scale_index & 3u))) & 7u;\n"
"        device const char4 *gv = (device const char4 *)(grid + grid_index * 8u);\n"
"        device const float4 *xv = (device const float4 *)(x + g * 8u);\n"
"        acc += (d * float(2u * scale + 1u)) * (dot(xv[0], float4(gv[0]) + delta) + dot(xv[1], float4(gv[1]) + delta));\n"
"    }\n"
"    return acc;\n"
"}\n"
"\n"
"inline float redmetal_topk_iq3_block(uint type, device const uchar *bp, device const float *x, uint part, uint nparts) {\n"
"    const uint gper = 32u / nparts; float acc = 0.0f;\n"
"    if ((gper & 3u) == 0u) {   /* dev33: whole 32-value sub-blocks: specialized dot */\n"
"        for (uint sb = part * (gper >> 2); sb < (part + 1u) * (gper >> 2); ++sb) acc += rl_iq3_dot32(type, bp, sb, x + sb * 32u);\n"
"        return acc;\n"
"    }\n"
"    for (uint g = part * gper; g < (part + 1u) * gper; ++g) {\n"
"        float4 v0, v1; rl_iq3_group8(type, bp, g, v0, v1);\n"
"        acc += dot(*(device const float4 *)(x + g * 8u), v0) + dot(*(device const float4 *)(x + g * 8u + 4u), v1);\n"
"    }\n"
"    return acc;\n"
"}\n"
"inline float redmetal_topk_block(uint type, device const uchar *bp, device const float *x, device const char *grid, uint part, uint nparts) {\n"
"    if (type == 17u) return redmetal_topk_iq2_xs_block(bp, x, grid, part, nparts);\n"
"    if (type == 29u) return redmetal_topk_iq1_m_block(bp, x, grid, part, nparts);\n"
"    return redmetal_topk_iq3_block(type, bp, x, part, nparts);\n"
"}\n"
"\n"
/* act[e][row] = SiLU(gate_e[row].x) * up_e[row].x for every selected expert; lanes_per_row lanes per row */
"kernel void redmetal_topk_gateup_lanes(\n"
"    device const redmetal_topk_slots &slots [[buffer(0)]],\n"
"    constant uint &ncols [[buffer(1)]],\n"
"    constant uint &rows [[buffer(2)]],\n"
"    constant ulong &up_offset [[buffer(3)]],\n"
"    device const float *x [[buffer(4)]],\n"
"    device float *act [[buffer(5)]],\n"
"    device const char *grid [[buffer(6)]],\n"
"    constant uint &top_k [[buffer(7)]],\n"
"    constant uint &type [[buffer(8)]],\n"
"    constant uint &lanes_per_row [[buffer(9)]],\n"
"    device const uint *rl_abort [[buffer(30), function_constant(redmetal_guard)]],\n"
"    device const float *x1 [[buffer(11), function_constant(redmetal_two_x)]],\n"
"    constant uint &x_split [[buffer(12), function_constant(redmetal_two_x)]],\n"
"    uint2 tid [[thread_position_in_grid]],\n"
"    ushort simd_lane [[thread_index_in_simdgroup]]) {\n"
"    if (redmetal_guard) { if (rl_abort[0] != 0u) return; }\n"
"    const uint expert = tid.y;\n"
"    device const float *xin = x; if (redmetal_two_x) { if (expert >= x_split) xin = x1; }\n"
"    const uint row = tid.x / lanes_per_row;\n"
"    const uint lane = uint(simd_lane) % lanes_per_row;\n"
"    const bool active = expert < top_k && row < rows;\n"
"    const uint blocks = ncols / 256u; const uint gt = rm_gate_type(type);\n"
"    const ulong block_bytes = rm_block_bytes(gt);\n"
"    const ulong row_bytes = ulong(blocks) * block_bytes;\n"
"    const uint nparts = max(1u, lanes_per_row / blocks); const uint part = lane % nparts; const uint lane_block = lane / nparts;\n"
"    const uint block_lanes = max(1u, lanes_per_row / nparts);\n"
"    float g = 0.0f, u = 0.0f;\n"
"    if (active) {\n"
"        device const uchar *slot = slots.slot[expert];\n"
"        device const uchar *grow = slot + ulong(row) * row_bytes;\n"
"        device const uchar *urow = slot + up_offset + ulong(row) * row_bytes;\n"
"        for (uint b = lane_block; b < blocks; b += block_lanes) {\n"
"            g += redmetal_topk_block(gt, grow + ulong(b) * block_bytes, xin + b * 256u, grid, part, nparts);\n"
"            u += redmetal_topk_block(gt, urow + ulong(b) * block_bytes, xin + b * 256u, grid, part, nparts);\n"
"        }\n"
"    }\n"
"    for (uint off = lanes_per_row >> 1; off > 0u; off >>= 1) { g += simd_shuffle_xor(g, ushort(off)); u += simd_shuffle_xor(u, ushort(off)); }\n"
"    if (active && lane == 0u) act[expert * rows + row] = (g / (1.0f + exp(-g))) * u;\n"
"}\n"
"\n"
/* tmp[e][r] = down_e[row_start + r] . act[e] */
"kernel void redmetal_topk_down_lanes(\n"
"    device const redmetal_topk_slots &slots [[buffer(0)]],\n"
"    constant uint &ncols [[buffer(1)]],\n"
"    constant uint &row_start [[buffer(2)]],\n"
"    constant uint &row_count [[buffer(3)]],\n"
"    constant ulong &down_offset [[buffer(4)]],\n"
"    device const float *act [[buffer(5)]],\n"
"    device float *tmp [[buffer(6)]],\n"
"    device const char *grid [[buffer(7)]],\n"
"    constant uint &top_k [[buffer(8)]],\n"
"    constant uint &type [[buffer(9)]],\n"
"    constant uint &lanes_per_row [[buffer(10)]],\n"
"    device const uint *rl_abort [[buffer(30), function_constant(redmetal_guard)]],\n"
"    uint2 tid [[thread_position_in_grid]],\n"
"    ushort simd_lane [[thread_index_in_simdgroup]]) {\n"
"    if (redmetal_guard) { if (rl_abort[0] != 0u) return; }\n"
"    const uint expert = tid.y;\n"
"    const uint r = tid.x / lanes_per_row;\n"
"    const uint lane = uint(simd_lane) % lanes_per_row;\n"
"    const bool active = expert < top_k && r < row_count;\n"
"    const uint blocks = ncols / 256u; const uint dt = rm_down_type(type);\n"
"    const ulong block_bytes = rm_block_bytes(dt);\n"
"    const ulong row_bytes = ulong(blocks) * block_bytes;\n"
"    const uint nparts = max(1u, lanes_per_row / blocks); const uint part = lane % nparts; const uint lane_block = lane / nparts;\n"
"    const uint block_lanes = max(1u, lanes_per_row / nparts);\n"
"    float acc = 0.0f;\n"
"    if (active) {\n"
"        device const uchar *drow = slots.slot[expert] + down_offset + ulong(row_start + r) * row_bytes;\n"
"        device const float *xe = act + expert * ncols;\n"
"        for (uint b = lane_block; b < blocks; b += block_lanes) acc += redmetal_topk_block(dt, drow + ulong(b) * block_bytes, xe + b * 256u, grid, part, nparts);\n"
"    }\n"
"    for (uint off = lanes_per_row >> 1; off > 0u; off >>= 1) acc += simd_shuffle_xor(acc, ushort(off));\n"
"    if (active && lane == 0u) tmp[expert * row_count + r] = acc;\n"
"}\n"
"\n"
/* ---- batched prefill (dev20b/d): U unique experts serve P (expert, token) pairs sorted by expert ---- */
/* Eight consecutive dequantized values (columns gi*8 .. gi*8+7) of one IQ2_XS / IQ1_M block; same
 * arithmetic as redmetal_topk_iq2_xs_block / redmetal_topk_iq1_m_block, one 8-value group at a time
 * so several rows can be decoded in lockstep and share the activation loads. */
"inline void rm_group8(uint type, device const uchar *bp, uint gi, device const char *grid, thread float4 &v0, thread float4 &v1) {\n"
"    if (type != 17u && type != 29u) { rl_iq3_group8f(type, bp, gi, v0, v1); return; }\n"
"    if (type == 17u) {\n"
"        const uint g = gi >> 1; const uint half_ = gi & 1u;\n"
"        const float d = float(as_type<half>(*(device const ushort *)bp));\n"
"        device const ushort *qs = (device const ushort *)(bp + 2);\n"
"        device const uchar *scales = bp + 66;\n"
"        const uint scale = (uint(scales[g >> 1]) >> (4u * (g & 1u))) & 15u;\n"
"        const float db = d * (0.5f + float(scale)) * 0.25f;\n"
"        const uint q = uint(qs[2u * g + half_]);\n"
"        const uint grid_index = q & 511u;\n"
"        const uint signs = redlite_sign8(q >> 9);\n"
"        device const char4 *gv = (device const char4 *)(grid + grid_index * 8u);\n"
"        const bool4 n0 = bool4(uint4(signs) & uint4(1u, 2u, 4u, 8u)); const bool4 n1 = bool4(uint4(signs) & uint4(16u, 32u, 64u, 128u));\n"
"        v0 = db * float4(gv[0]) * select(float4(1.0f), float4(-1.0f), n0); v1 = db * float4(gv[1]) * select(float4(1.0f), float4(-1.0f), n1);\n"
"        return;\n"
"    } else {\n"
"        device const uchar *qs = bp; device const uchar *qh = bp + 32;\n"
"        device const ushort *sc = (device const ushort *)(bp + 48);\n"
"        const ushort dbits = ushort(sc[0] >> 12) | ushort((sc[1] >> 8) & 0x00f0u) | ushort((sc[2] >> 4) & 0x0f00u) | ushort(sc[3] & 0xf000u);\n"
"        const float d = float(as_type<half>(dbits));\n"
"        const uint g = gi;\n"
"        const uint nibble = (uint(qh[g >> 1]) >> (4u * (g & 1u))) & 15u;\n"
"        const uint grid_index = uint(qs[g]) | ((nibble & 7u) << 8);\n"
"        const float delta = (nibble & 8u) ? -0.125f : 0.125f;\n"
"        const uint scale_index = g >> 1;\n"
"        const uint scale = (uint(sc[scale_index >> 2]) >> (3u * (scale_index & 3u))) & 7u;\n"
"        const float dl = d * float(2u * scale + 1u);\n"
"        device const char4 *gv = (device const char4 *)(grid + grid_index * 8u);\n"
"        v0 = dl * (float4(gv[0]) + delta); v1 = dl * (float4(gv[1]) + delta);\n"
"    }\n"
"}\n"
"inline float rm_dot8(float4 x0, float4 x1, float4 v0, float4 v1) { const float4 s = fma(x0, v0, x1 * v1); return (s.x + s.y) + (s.z + s.w); }\n"
/* act[p][row] = SiLU(gate_e[row].x_t) * up_e[row].x_t for every pair p of expert e.
 * Thread = (expert, tile of 4 rows, lane): the lane's blocks are decoded for the 4 rows in lockstep so each
 * activation group is loaded once per pair and applied to 4 rows (gate and up). */
"kernel void redmetal_topk_gateup_b(\n"
"    device const redmetal_topk_slots3 &slots [[buffer(0)]],\n"
"    constant uint &ncols [[buffer(1)]],\n"
"    constant uint &rows [[buffer(2)]],\n"
"    constant uint &unused3 [[buffer(3)]],\n"
"    device const float *x [[buffer(4)]],\n"
"    device float *act [[buffer(5)]],\n"
"    device const char *grid [[buffer(6)]],\n"
"    constant uint &n_expert [[buffer(7)]],\n"
"    constant uint &type [[buffer(8)]],\n"
"    constant uint &lanes_per_row [[buffer(9)]],\n"
"    device const uint *pair_token [[buffer(10)]],\n"
"    device const uint *pair_expert [[buffer(11)]],\n"
"    constant uint &n_pairs [[buffer(12)]],\n"
"    device const uint *slice_first [[buffer(13)]],\n"
"    device const uint *slice_count [[buffer(14)]],\n"
"    constant uint &n_slices [[buffer(15)]],\n"
"    uint2 tid [[thread_position_in_grid]],\n"
"    ushort simd_lane [[thread_index_in_simdgroup]]) {\n"
"    const uint sl = tid.y; const uint p = sl < n_slices ? slice_first[sl] : 0u; const uint cnt = sl < n_slices ? slice_count[sl] : 0u;\n"
"    const uint expert = p < n_pairs ? pair_expert[p] : 0u;\n"
"    const uint tile = tid.x / lanes_per_row; const uint row0 = tile * 4u;\n"
"    const uint lane = uint(simd_lane) % lanes_per_row;\n"
"    const bool active = sl < n_slices && cnt > 0u && expert < n_expert && row0 < rows;\n"
"    const uint blocks = ncols / 256u;\n"
"    const ulong block_bytes = rm_block_bytes(rm_gate_type(type));\n"
"    const ulong row_bytes = ulong(blocks) * block_bytes;\n"
"    const uint nparts = max(1u, lanes_per_row / blocks); const uint part = lane % nparts; const uint lane_block = lane / nparts;\n"
"    const uint block_lanes = max(1u, lanes_per_row / nparts);\n"
"    const uint gper = 32u / nparts; const uint g0 = part * gper, g1 = g0 + gper;\n"
"    const uint ex = expert < n_expert ? expert : 0u;\n"
"    const uint rbase = active ? row0 : 0u;\n"
"    device const uchar *grow = slots.gate[ex] + ulong(rbase) * row_bytes;\n"
"    device const uchar *urow = slots.up[ex] + ulong(rbase) * row_bytes;\n"
"    {\n"
"        device const float *xs[RM_S];\n"
"        for (uint n = 0; n < RM_S; ++n) { const uint q = (n < cnt && p + n < n_pairs) ? p + n : (p < n_pairs ? p : 0u); xs[n] = x + ulong(pair_token[q]) * ncols; }\n"
"        float ag[4][RM_S], au[4][RM_S];\n"
"        for (uint r = 0; r < 4u; ++r) for (uint n = 0; n < RM_S; ++n) { ag[r][n] = 0.0f; au[r][n] = 0.0f; }\n"
"        if (active) {\n"
"            const uint nrows = min(4u, rows - row0);\n"
"            for (uint b = lane_block; b < blocks; b += block_lanes) {\n"
"                device const uchar *gb = grow + ulong(b) * block_bytes; device const uchar *ub = urow + ulong(b) * block_bytes;\n"
"                for (uint gi = g0; gi < g1; ++gi) {\n"
"                    const uint col = b * 256u + gi * 8u;\n"
"                    float4 x0[RM_S], x1[RM_S];\n"
"                    for (uint n = 0; n < RM_S; ++n) { x0[n] = *(device const float4 *)(xs[n] + col); x1[n] = *(device const float4 *)(xs[n] + col + 4u); }\n"
"                    for (uint r = 0; r < nrows; ++r) {\n"
"                        float4 v0, v1;\n"
"                        rm_group8(rm_gate_type(type), gb + ulong(r) * row_bytes, gi, grid, v0, v1); for (uint n = 0; n < RM_S; ++n) ag[r][n] += rm_dot8(x0[n], x1[n], v0, v1);\n"
"                        rm_group8(rm_gate_type(type), ub + ulong(r) * row_bytes, gi, grid, v0, v1); for (uint n = 0; n < RM_S; ++n) au[r][n] += rm_dot8(x0[n], x1[n], v0, v1);\n"
"                    }\n"
"                }\n"
"            }\n"
"        }\n"
"        for (uint off = lanes_per_row >> 1; off > 0u; off >>= 1) {\n"
"            for (uint r = 0; r < 4u; ++r) for (uint n = 0; n < RM_S; ++n) { ag[r][n] += simd_shuffle_xor(ag[r][n], ushort(off)); au[r][n] += simd_shuffle_xor(au[r][n], ushort(off)); }\n"
"        }\n"
"        if (active && lane == 0u) {\n"
"            for (uint n = 0; n < cnt && n < RM_S; ++n) {\n"
"                device float *out = act + (p + n) * rows + row0;\n"
"                for (uint r = 0; r < 4u && row0 + r < rows; ++r) { const float g = ag[r][n]; out[r] = (g / (1.0f + exp(-g))) * au[r][n]; }\n"
"            }\n"
"        }\n"
"    }\n"
"}\n"
"\n"
/* tmp[p][r] = down_e[r] . act[p] for every pair p of expert e; same 4-row tile */
"kernel void redmetal_topk_down_b(\n"
"    device const redmetal_topk_slots3 &slots [[buffer(0)]],\n"
"    constant uint &ncols [[buffer(1)]],\n"
"    constant uint &row_count [[buffer(2)]],\n"
"    constant uint &unused3 [[buffer(3)]],\n"
"    device const float *act [[buffer(4)]],\n"
"    device float *tmp [[buffer(5)]],\n"
"    device const char *grid [[buffer(6)]],\n"
"    constant uint &n_expert [[buffer(7)]],\n"
"    constant uint &type [[buffer(8)]],\n"
"    constant uint &lanes_per_row [[buffer(9)]],\n"
"    device const uint *pair_expert [[buffer(10)]],\n"
"    constant uint &n_pairs [[buffer(11)]],\n"
"    device const uint *slice_first [[buffer(12)]],\n"
"    device const uint *slice_count [[buffer(13)]],\n"
"    constant uint &n_slices [[buffer(14)]],\n"
"    uint2 tid [[thread_position_in_grid]],\n"
"    ushort simd_lane [[thread_index_in_simdgroup]]) {\n"
"    const uint sl = tid.y; const uint p = sl < n_slices ? slice_first[sl] : 0u; const uint cnt = sl < n_slices ? slice_count[sl] : 0u;\n"
"    const uint expert = p < n_pairs ? pair_expert[p] : 0u;\n"
"    const uint tile = tid.x / lanes_per_row; const uint row0 = tile * 4u;\n"
"    const uint lane = uint(simd_lane) % lanes_per_row;\n"
"    const bool active = sl < n_slices && cnt > 0u && expert < n_expert && row0 < row_count;\n"
"    const uint blocks = ncols / 256u;\n"
"    const ulong block_bytes = rm_block_bytes(rm_down_type(type));\n"
"    const ulong row_bytes = ulong(blocks) * block_bytes;\n"
"    const uint nparts = max(1u, lanes_per_row / blocks); const uint part = lane % nparts; const uint lane_block = lane / nparts;\n"
"    const uint block_lanes = max(1u, lanes_per_row / nparts);\n"
"    const uint gper = 32u / nparts; const uint g0 = part * gper, g1 = g0 + gper;\n"
"    device const uchar *drow = slots.down[expert < n_expert ? expert : 0u] + ulong(active ? row0 : 0u) * row_bytes;\n"
"    {\n"
"        device const float *xs[RM_S];\n"
"        for (uint n = 0; n < RM_S; ++n) { const uint q = (n < cnt && p + n < n_pairs) ? p + n : (p < n_pairs ? p : 0u); xs[n] = act + ulong(q) * ncols; }\n"
"        float a[4][RM_S];\n"
"        for (uint r = 0; r < 4u; ++r) for (uint n = 0; n < RM_S; ++n) a[r][n] = 0.0f;\n"
"        if (active) {\n"
"            const uint nrows = min(4u, row_count - row0);\n"
"            for (uint b = lane_block; b < blocks; b += block_lanes) {\n"
"                device const uchar *db = drow + ulong(b) * block_bytes;\n"
"                for (uint gi = g0; gi < g1; ++gi) {\n"
"                    const uint col = b * 256u + gi * 8u;\n"
"                    float4 x0[RM_S], x1[RM_S];\n"
"                    for (uint n = 0; n < RM_S; ++n) { x0[n] = *(device const float4 *)(xs[n] + col); x1[n] = *(device const float4 *)(xs[n] + col + 4u); }\n"
"                    for (uint r = 0; r < nrows; ++r) {\n"
"                        float4 v0, v1;\n"
"                        rm_group8(rm_down_type(type), db + ulong(r) * row_bytes, gi, grid, v0, v1); for (uint n = 0; n < RM_S; ++n) a[r][n] += rm_dot8(x0[n], x1[n], v0, v1);\n"
"                    }\n"
"                }\n"
"            }\n"
"        }\n"
"        for (uint off = lanes_per_row >> 1; off > 0u; off >>= 1) {\n"
"            for (uint r = 0; r < 4u; ++r) for (uint n = 0; n < RM_S; ++n) a[r][n] += simd_shuffle_xor(a[r][n], ushort(off));\n"
"        }\n"
"        if (active && lane == 0u) {\n"
"            for (uint n = 0; n < cnt && n < RM_S; ++n) {\n"
"                device float *out = tmp + (p + n) * row_count + row0;\n"
"                for (uint r = 0; r < 4u && row0 + r < row_count; ++r) out[r] = a[r][n];\n"
"            }\n"
"        }\n"
"    }\n"
"}\n"
"\n"
/* ---- dev30 batched prefill experts on simdgroup matrices ----
 * Tile = up to 16 consecutive (expert, token) pairs of one expert (slice) x 32 weight rows.
 * Per K step of 32 columns the 128 threads decode the tile's weights (one rm_group8 each per matrix)
 * and stage the pairs' activations in threadgroup memory; each simdgroup owns 8 weight rows. */
"kernel void redmetal_topk_gateup_mm(\n"
"    device const redmetal_topk_slots3 &slots [[buffer(0)]], constant uint &ncols [[buffer(1)]], constant uint &rows [[buffer(2)]],\n"
"    device const float *x [[buffer(4)]], device float *act [[buffer(5)]], device const char *grid [[buffer(6)]],\n"
"    constant uint &type [[buffer(8)]], device const uint *pair_token [[buffer(10)]], device const uint *pair_expert [[buffer(11)]],\n"
"    device const uint *slice_first [[buffer(13)]], device const uint *slice_count [[buffer(14)]],\n"
"    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {\n"
"    threadgroup float xs[16 * 32]; threadgroup float wg[32 * 32]; threadgroup float wu[32 * 32]; threadgroup float og[32 * 16]; threadgroup float ou[32 * 16];\n"
"    const uint row0 = tg.x * 32u; const uint p0 = slice_first[tg.y]; const uint cnt = slice_count[tg.y];\n"
"    const uint expert = pair_expert[p0];\n"
"    const uint blocks = ncols / 256u; const ulong block_bytes = rm_block_bytes(rm_gate_type(type)); const ulong row_bytes = ulong(blocks) * block_bytes;\n"
"    const uint wr = tid >> 2; const uint wq = tid & 3u;          /* decoder: weight row wr, 8-column group wq of the K step */\n"
"    const uint xr = tid >> 3; const uint xq = (tid & 7u) * 4u;    /* loader: pair xr, 4 columns */\n"
"    device const uchar *grow = slots.gate[expert] + ulong(row0 + wr) * row_bytes;\n"
"    device const uchar *urow = slots.up[expert] + ulong(row0 + wr) * row_bytes;\n"
"    device const float *xrow = x + ulong(pair_token[p0 + min(xr, cnt - 1u)]) * ncols;\n"
"    simdgroup_float8x8 ag[2][4], au[2][4]; for (uint j = 0; j < 2u; ++j) for (uint q = 0; q < 4u; ++q) { ag[j][q] = simdgroup_float8x8(0.0f); au[j][q] = simdgroup_float8x8(0.0f); }\n"
"    float4 g0, g1, u0, u1, xv;   /* dev50: step k+1 is loaded and decoded into registers while step k multiplies */\n"
"    rm_group8(rm_gate_type(type), grow, wq, grid, g0, g1); rm_group8(rm_gate_type(type), urow, wq, grid, u0, u1);\n"
"    xv = xr < cnt ? *(device const float4 *)(xrow + xq) : float4(0.0f);\n"
"    for (uint k0 = 0; k0 < ncols; k0 += 32u) {\n"
"        *(threadgroup float4 *)(wg + wr * 32u + wq * 8u) = g0; *(threadgroup float4 *)(wg + wr * 32u + wq * 8u + 4u) = g1;\n"
"        *(threadgroup float4 *)(wu + wr * 32u + wq * 8u) = u0; *(threadgroup float4 *)(wu + wr * 32u + wq * 8u + 4u) = u1;\n"
"        *(threadgroup float4 *)(xs + xr * 32u + xq) = xv;\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"        const uint kn = k0 + 32u;\n"
"        if (kn < ncols) {\n"
"            const uint b = kn >> 8; const uint gi = ((kn & 255u) >> 3) + wq;\n"
"            rm_group8(rm_gate_type(type), grow + ulong(b) * block_bytes, gi, grid, g0, g1);\n"
"            rm_group8(rm_gate_type(type), urow + ulong(b) * block_bytes, gi, grid, u0, u1);\n"
"            xv = xr < cnt ? *(device const float4 *)(xrow + kn + xq) : float4(0.0f);\n"
"        }\n"
"        simdgroup_float8x8 a_g, a_u, bx;\n"
"        for (uint q = 0; q < 4u; ++q) {\n"
"            simdgroup_load(a_g, wg + uint(sg) * 8u * 32u + q * 8u, 32); simdgroup_load(a_u, wu + uint(sg) * 8u * 32u + q * 8u, 32);\n"
"            for (uint j = 0; j < 2u; ++j) {\n"
"                simdgroup_load(bx, xs + j * 8u * 32u + q * 8u, 32, ulong2(0, 0), true);\n"
"                simdgroup_multiply_accumulate(ag[j][q], a_g, bx, ag[j][q]); simdgroup_multiply_accumulate(au[j][q], a_u, bx, au[j][q]);\n"
"            }\n"
"        }\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    {   /* (a0 + a1) + (a2 + a3) in registers: x * I + y is exact */\n"
"        const simdgroup_float8x8 id = simdgroup_float8x8(1.0f); simdgroup_float8x8 t0, t1;\n"
"        for (uint j = 0; j < 2u; ++j) {\n"
"            simdgroup_multiply_accumulate(t0, ag[j][0], id, ag[j][1]); simdgroup_multiply_accumulate(t1, ag[j][2], id, ag[j][3]); simdgroup_multiply_accumulate(t0, t0, id, t1);\n"
"            simdgroup_store(t0, og + uint(sg) * 8u * 16u + j * 8u, 16);\n"
"            simdgroup_multiply_accumulate(t0, au[j][0], id, au[j][1]); simdgroup_multiply_accumulate(t1, au[j][2], id, au[j][3]); simdgroup_multiply_accumulate(t0, t0, id, t1);\n"
"            simdgroup_store(t0, ou + uint(sg) * 8u * 16u + j * 8u, 16);\n"
"        }\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint i = tid; i < 32u * 16u; i += 128u) { const uint r = i >> 4; const uint n = i & 15u; if (n >= cnt || row0 + r >= rows) continue;\n"
"        const float g = og[i]; act[ulong(p0 + n) * rows + row0 + r] = (g / (1.0f + exp(-g))) * ou[i]; }\n"
"}\n"
"kernel void redmetal_topk_down_mm(\n"
"    device const redmetal_topk_slots3 &slots [[buffer(0)]], constant uint &ncols [[buffer(1)]], constant uint &rows [[buffer(2)]],\n"
"    device const float *act [[buffer(4)]], device float *tmp [[buffer(5)]], device const char *grid [[buffer(6)]],\n"
"    constant uint &type [[buffer(8)]], device const uint *pair_expert [[buffer(10)]],\n"
"    device const uint *slice_first [[buffer(12)]], device const uint *slice_count [[buffer(13)]],\n"
"    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {\n"
"    threadgroup float xs[16 * 32]; threadgroup float wd[32 * 32]; threadgroup float od[32 * 16];\n"
"    const uint row0 = tg.x * 32u; const uint p0 = slice_first[tg.y]; const uint cnt = slice_count[tg.y];\n"
"    const uint expert = pair_expert[p0];\n"
"    const uint blocks = ncols / 256u; const ulong block_bytes = rm_block_bytes(rm_down_type(type)); const ulong row_bytes = ulong(blocks) * block_bytes;\n"
"    const uint wr = tid >> 2; const uint wq = tid & 3u; const uint xr = tid >> 3; const uint xq = (tid & 7u) * 4u;\n"
"    device const uchar *drow = slots.down[expert] + ulong(row0 + wr) * row_bytes;\n"
"    device const float *xrow = act + ulong(p0 + min(xr, cnt - 1u)) * ncols;\n"
"    simdgroup_float8x8 ad[2][4]; for (uint j = 0; j < 2u; ++j) for (uint q = 0; q < 4u; ++q) ad[j][q] = simdgroup_float8x8(0.0f);\n"
"    float4 d0, d1, xv;\n"
"    rm_group8(rm_down_type(type), drow, wq, grid, d0, d1);\n"
"    xv = xr < cnt ? *(device const float4 *)(xrow + xq) : float4(0.0f);\n"
"    for (uint k0 = 0; k0 < ncols; k0 += 32u) {\n"
"        *(threadgroup float4 *)(wd + wr * 32u + wq * 8u) = d0; *(threadgroup float4 *)(wd + wr * 32u + wq * 8u + 4u) = d1;\n"
"        *(threadgroup float4 *)(xs + xr * 32u + xq) = xv;\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"        const uint kn = k0 + 32u;\n"
"        if (kn < ncols) {\n"
"            const uint b = kn >> 8; const uint gi = ((kn & 255u) >> 3) + wq;\n"
"            rm_group8(rm_down_type(type), drow + ulong(b) * block_bytes, gi, grid, d0, d1);\n"
"            xv = xr < cnt ? *(device const float4 *)(xrow + kn + xq) : float4(0.0f);\n"
"        }\n"
"        simdgroup_float8x8 a, bx;\n"
"        for (uint q = 0; q < 4u; ++q) {\n"
"            simdgroup_load(a, wd + uint(sg) * 8u * 32u + q * 8u, 32);\n"
"            for (uint j = 0; j < 2u; ++j) { simdgroup_load(bx, xs + j * 8u * 32u + q * 8u, 32, ulong2(0, 0), true); simdgroup_multiply_accumulate(ad[j][q], a, bx, ad[j][q]); }\n"
"        }\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    {\n"
"        const simdgroup_float8x8 id = simdgroup_float8x8(1.0f); simdgroup_float8x8 t0, t1;\n"
"        for (uint j = 0; j < 2u; ++j) {\n"
"            simdgroup_multiply_accumulate(t0, ad[j][0], id, ad[j][1]); simdgroup_multiply_accumulate(t1, ad[j][2], id, ad[j][3]); simdgroup_multiply_accumulate(t0, t0, id, t1);\n"
"            simdgroup_store(t0, od + uint(sg) * 8u * 16u + j * 8u, 16);\n"
"        }\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint i = tid; i < 32u * 16u; i += 128u) { const uint r = i >> 4; const uint n = i & 15u; if (n >= cnt || row0 + r >= rows) continue;\n"
"        tmp[ulong(p0 + n) * rows + row0 + r] = od[i]; }\n"
"}\n"
/* out[t][r] = sum_k weight[pair_k] * tmp[pair_k][r] over the token's top-k pairs in selection order */
"kernel void redmetal_topk_sum_b(\n"
"    device const float *tmp [[buffer(0)]],\n"
"    device const float *pair_weight [[buffer(1)]],\n"
"    device const uint *tok_pair [[buffer(2)]],\n"
"    device float *out [[buffer(3)]],\n"
"    constant uint &row_count [[buffer(4)]],\n"
"    constant uint &top_k [[buffer(5)]],\n"
"    constant uint &tok_first [[buffer(6)]],\n"
"    constant uint &ntok [[buffer(7)]],\n"
"    uint2 gid [[thread_position_in_grid]]) {\n"
"    if (gid.x >= row_count || gid.y >= ntok) return;\n"
"    float acc = 0.0f;\n"
"    for (uint k = 0; k < top_k; ++k) { const uint p = tok_pair[gid.y * top_k + k]; acc += pair_weight[p] * tmp[p * row_count + gid.x]; }\n"
"    out[ulong(tok_first + gid.y) * row_count + gid.x] = acc;\n"
"}\n"
"\n"
/* out[r] = sum_e weight[e] * tmp[e][r], experts accumulated in selection order */
"kernel void redmetal_topk_weighted_sum(\n"
"    device const float *tmp [[buffer(0)]],\n"
"    device const float *weights [[buffer(1)]],\n"
"    device float *out [[buffer(2)]],\n"
"    constant uint &row_count [[buffer(3)]],\n"
"    constant uint &top_k [[buffer(4)]],\n"
"    device const uint *rl_abort [[buffer(30), function_constant(redmetal_guard)]],\n"
"    uint gid [[thread_position_in_grid]]) {\n"
"    if (redmetal_guard) { if (rl_abort[0] != 0u) return; }\n"
"    if (gid >= row_count) return;\n"
"    float acc = 0.0f;\n"
"    for (uint e = 0; e < top_k; ++e) acc += weights[e] * tmp[e * row_count + gid];\n"
"    out[gid] = acc;\n"
"}\n";

@interface RMTopKPool : NSObject {
@public
    id<MTLDevice> _device;
    id<MTLCommandQueue> _queue;
    id<MTLComputePipelineState> _gateupPipeline;
    id<MTLComputePipelineState> _downPipeline;
    id<MTLComputePipelineState> _sumPipeline;
    id<MTLComputePipelineState> _gateupGPipeline, _downGPipeline, _sumGPipeline;   /* dev23 early-out variants */
    id<MTLComputePipelineState> _gateupG2Pipeline;   /* dev45: guarded, two input vectors */
    id<MTLComputePipelineState> _gateupBPipeline;
    id<MTLComputePipelineState> _downBPipeline;
    id<MTLComputePipelineState> _sumBPipeline;
    id<MTLComputePipelineState> _gateupMMPipeline, _downMMPipeline;   /* dev30 simdgroup-matrix batched experts */
    int _useMM;                                                       /* RL_PREFILL_EXPERT_MM != 0 (default) */
    id<MTLBuffer> _slotTable3;
    id<MTLBuffer> _residency;
    uint32_t _residencyLayers, _residencyExperts;
    id<MTLResidencySet> _residencySet API_AVAILABLE(macos(15.0));   /* every slab, committed once; attached to the engine queue */
    id<MTLBuffer> _pairTokenBuf;
    id<MTLBuffer> _pairExpertBuf;
    id<MTLBuffer> _sliceFirstBuf;
    id<MTLBuffer> _sliceCountBuf;
    id<MTLBuffer> _pairWeightBuf;
    id<MTLBuffer> _expertStartBuf;
    id<MTLBuffer> _tokPairBuf;
    id<MTLBuffer> _actB;
    id<MTLBuffer> _tmpB;
    id<MTLBuffer> _iq2Grid;
    id<MTLBuffer> _iq1Grid;
    id<MTLBuffer> _slotTable;
    id<MTLBuffer> _weightBuffer;
    id<MTLBuffer> _inBuffer;
    id<MTLBuffer> _actBuffer;
    id<MTLBuffer> _tmpBuffer;
    id<MTLBuffer> _outBuffer;
    NSMutableArray<id<MTLBuffer>> *_slabs;
    int _fd;
    uint64_t _fileSize;
    uint64_t _budgetBytes;
    uint64_t _slotBytes;
    uint32_t _capacity;
    uint32_t _slotsPerSlab;
    /* dev37: slot size classes (one by default). Class c owns slots [_clsFirst[c], _clsFirst[c + 1]) of _clsBytes[c]
     * bytes; its slabs are global slab indices from _clsSlab[c]; _slabMap maps a global slab index to _slabs + 1. */
    uint32_t _nCls;
    uint32_t _clsFirst[RL_TOPK_MAX_CLASSES + 1];
    uint64_t _clsBytes[RL_TOPK_MAX_CLASSES];
    uint32_t _clsSlab[RL_TOPK_MAX_CLASSES + 1];
    uint32_t *_slabMap;
    uint32_t *_slotInflight;
    uint32_t *_slotGeneration;   /* dev23: bumped by every load into the slot (release checks in-flight slots were not rewritten) */
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
- (uint64_t)bytesForSlot:(uint32_t)slot;
- (int)setClasses:(uint32_t)n slotBytes:(const uint64_t *)bytes capacity:(const uint32_t *)cap;
- (void)markSlot:(uint32_t)slot delta:(int)delta;
- (int)ensureScratchHidden:(uint32_t)hidden ffn:(uint32_t)ffn;
@end

@implementation RMTopKPool

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
        topk_set_error("invalid Red Metal top-k pool arguments");
        return nil;
    }
    const uint64_t aligned = round_up_u64(slotBytes, 4096u);
    if (!aligned || aligned > budget) {
        topk_set_error("top-k cache budget cannot fit one aligned expert slot");
        return nil;
    }
    _slotBytes = aligned;
    _budgetBytes = budget;
    _capacity = (uint32_t)(budget / aligned);
    _slotsPerSlab = slotsPerSlab ? slotsPerSlab : 64u;
    if (_slotsPerSlab > _capacity) _slotsPerSlab = _capacity;
    _nCls = 1u;
    _clsFirst[0] = 0u; _clsFirst[1] = _capacity; _clsBytes[0] = aligned;
    _clsSlab[0] = 0u; _clsSlab[1] = (_capacity + _slotsPerSlab - 1u) / _slotsPerSlab;

    _device = MTLCreateSystemDefaultDevice();
    if (!_device || ![_device hasUnifiedMemory]) {
        topk_set_error("Red Metal top-k requires Apple unified memory");
        return nil;
    }
    _queue = [_device newCommandQueue];
    if (!_queue) {
        topk_set_error("failed to create top-k Metal command queue");
        return nil;
    }

    NSError *libraryError = nil;
    char *iq3 = rl_iq3_metal_source();   /* dev31: IQ3_XXS / IQ3_S codebooks and decoder */
    if (!iq3) { topk_set_error("IQ3 Metal source allocation failed"); return nil; }
    NSString *source = [NSString stringWithFormat:@"#include <metal_stdlib>\nusing namespace metal;\n%s\n%@", iq3, kTopKSource];
    free(iq3);
    id<MTLLibrary> library = [_device newLibraryWithSource:source options:nil error:&libraryError];
    if (!library) {
        topk_set_error("failed to compile top-k kernels: %s",
            libraryError.localizedDescription.UTF8String ?: "unknown Metal error");
        return nil;
    }
    MTLFunctionConstantValues *plain = [MTLFunctionConstantValues new], *guarded = [MTLFunctionConstantValues new];
    bool guard_off = false, guard_on = true;
    [plain setConstantValue:&guard_off type:MTLDataTypeBool atIndex:0];
    [guarded setConstantValue:&guard_on type:MTLDataTypeBool atIndex:0];
    NSError *fnError = nil;
    id<MTLFunction> gateup = [library newFunctionWithName:@"redmetal_topk_gateup_lanes" constantValues:plain error:&fnError];
    id<MTLFunction> down = [library newFunctionWithName:@"redmetal_topk_down_lanes" constantValues:plain error:&fnError];
    id<MTLFunction> sum = [library newFunctionWithName:@"redmetal_topk_weighted_sum" constantValues:plain error:&fnError];
    id<MTLFunction> gateupG = [library newFunctionWithName:@"redmetal_topk_gateup_lanes" constantValues:guarded error:&fnError];
    id<MTLFunction> downG = [library newFunctionWithName:@"redmetal_topk_down_lanes" constantValues:guarded error:&fnError];
    id<MTLFunction> sumG = [library newFunctionWithName:@"redmetal_topk_weighted_sum" constantValues:guarded error:&fnError];
    id<MTLFunction> gateupB = [library newFunctionWithName:@"redmetal_topk_gateup_b"];
    id<MTLFunction> downB = [library newFunctionWithName:@"redmetal_topk_down_b"];
    id<MTLFunction> sumB = [library newFunctionWithName:@"redmetal_topk_sum_b"];
    id<MTLFunction> gateupMM = [library newFunctionWithName:@"redmetal_topk_gateup_mm"];
    id<MTLFunction> downMM = [library newFunctionWithName:@"redmetal_topk_down_mm"];
    if (!gateup || !down || !sum || !gateupB || !downB || !sumB || !gateupG || !downG || !sumG || !gateupMM || !downMM) {
        topk_set_error("one or more top-k Metal functions were not found");
        return nil;
    }
    NSError *pipelineError = nil;
    _gateupPipeline = [_device newComputePipelineStateWithFunction:gateup error:&pipelineError];
    if (!_gateupPipeline) {
        topk_set_error("failed to create top-k gate/up pipeline: %s", pipelineError.localizedDescription.UTF8String ?: "unknown Metal error");
        return nil;
    }
    pipelineError = nil;
    _downPipeline = [_device newComputePipelineStateWithFunction:down error:&pipelineError];
    if (!_downPipeline) {
        topk_set_error("failed to create top-k down pipeline: %s", pipelineError.localizedDescription.UTF8String ?: "unknown Metal error");
        return nil;
    }
    pipelineError = nil;
    _sumPipeline = [_device newComputePipelineStateWithFunction:sum error:&pipelineError];
    if (!_sumPipeline) {
        topk_set_error("failed to create top-k weighted sum pipeline: %s", pipelineError.localizedDescription.UTF8String ?: "unknown Metal error");
        return nil;
    }
    _gateupGPipeline = [_device newComputePipelineStateWithFunction:gateupG error:&pipelineError];
    {
        MTLFunctionConstantValues *two = [MTLFunctionConstantValues new];
        bool on = true;
        [two setConstantValue:&on type:MTLDataTypeBool atIndex:0];
        [two setConstantValue:&on type:MTLDataTypeBool atIndex:1];
        NSError *e2 = nil;
        id<MTLFunction> g2 = [library newFunctionWithName:@"redmetal_topk_gateup_lanes" constantValues:two error:&e2];
        _gateupG2Pipeline = g2 ? [_device newComputePipelineStateWithFunction:g2 error:&e2] : nil;
    }
    _downGPipeline = [_device newComputePipelineStateWithFunction:downG error:&pipelineError];
    _sumGPipeline = [_device newComputePipelineStateWithFunction:sumG error:&pipelineError];
    if (!_gateupGPipeline || !_downGPipeline || !_sumGPipeline) {
        topk_set_error("failed to create early-out top-k pipelines: %s", pipelineError.localizedDescription.UTF8String ?: "unknown Metal error");
        return nil;
    }
    pipelineError = nil;
    _gateupBPipeline = [_device newComputePipelineStateWithFunction:gateupB error:&pipelineError];
    if (!_gateupBPipeline) { topk_set_error("failed to create batched top-k gate/up pipeline: %s", pipelineError.localizedDescription.UTF8String ?: "unknown Metal error"); return nil; }
    pipelineError = nil;
    _downBPipeline = [_device newComputePipelineStateWithFunction:downB error:&pipelineError];
    if (!_downBPipeline) { topk_set_error("failed to create batched top-k down pipeline: %s", pipelineError.localizedDescription.UTF8String ?: "unknown Metal error"); return nil; }
    pipelineError = nil;
    _sumBPipeline = [_device newComputePipelineStateWithFunction:sumB error:&pipelineError];
    if (!_sumBPipeline) { topk_set_error("failed to create batched top-k sum pipeline: %s", pipelineError.localizedDescription.UTF8String ?: "unknown Metal error"); return nil; }
    _gateupMMPipeline = [_device newComputePipelineStateWithFunction:gateupMM error:&pipelineError];
    _downMMPipeline = [_device newComputePipelineStateWithFunction:downMM error:&pipelineError];
    if (!_gateupMMPipeline || !_downMMPipeline) { topk_set_error("failed to create batched top-k matrix pipelines: %s", pipelineError.localizedDescription.UTF8String ?: "unknown Metal error"); return nil; }
    { const char *mm = getenv("RL_PREFILL_EXPERT_MM"); _useMM = !mm || atoi(mm) != 0; }
    _iq2Grid = [_device newBufferWithBytes:iq2Grid length:iq2GridCount options:MTLResourceStorageModeShared];
    _iq1Grid = [_device newBufferWithBytes:iq1Grid length:iq1GridCount options:MTLResourceStorageModeShared];
    _slotTable = [_device newBufferWithLength:REDMETAL_TOPK_MAX * sizeof(uint64_t) options:MTLResourceStorageModeShared];
    _slotTable3 = [_device newBufferWithLength:3u * REDMETAL_TOPK_MAX * sizeof(uint64_t) options:MTLResourceStorageModeShared];
    _weightBuffer = [_device newBufferWithLength:REDMETAL_TOPK_MAX * sizeof(float) options:MTLResourceStorageModeShared];
    if (!_iq2Grid || !_iq1Grid || !_slotTable || !_slotTable3 || !_weightBuffer) {
        topk_set_error("failed to allocate top-k quant grid / table buffers");
        return nil;
    }

    _fd = open(path, O_RDONLY);
    if (_fd < 0) {
        topk_set_error("open(%s) failed: %s", path, strerror(errno));
        return nil;
    }
    struct stat st;
    if (fstat(_fd, &st) != 0 || st.st_size <= 0) {
        topk_set_error("failed to stat model file: %s", strerror(errno));
        close(_fd);
        _fd = -1;
        return nil;
    }
    { const char *nc = getenv("RL_POOL_NOCACHE"); if (nc && atoi(nc) != 0) fcntl(_fd, F_NOCACHE, 1); }   /* dev46: expert reads bypass the page cache (A/B on 24 GiB Macs) */
    _fileSize = (uint64_t)st.st_size;
    _slotInflight = calloc(_capacity, sizeof(uint32_t));
    _slotGeneration = calloc(_capacity, sizeof(uint32_t));
    _slabMap = calloc(_clsSlab[1] ? _clsSlab[1] : 1u, sizeof(uint32_t));
    if (!_slotInflight || !_slotGeneration || !_slabMap) {
        topk_set_error("failed to allocate top-k in-flight table");
        close(_fd);
        _fd = -1;
        return nil;
    }
    _slabs = [NSMutableArray array];
    if (@available(macOS 15.0, *)) {
        MTLResidencySetDescriptor *rd = [[MTLResidencySetDescriptor alloc] init];
        rd.label = @"redlite_topk_slabs";
        rd.initialCapacity = 512;
        NSError *re = nil;
        _residencySet = [_device newResidencySetWithDescriptor:rd error:&re];
        if (_residencySet) [_residencySet commit];
    }
    return self;
}

- (void)dealloc {
    if (_fd >= 0) close(_fd);
    free(_slotInflight);
    _slotInflight = NULL;
    free(_slotGeneration);
    _slotGeneration = NULL;
    free(_slabMap);
    _slabMap = NULL;
}

/* dev37: replace the uniform layout by size classes; only before any slab exists, within the budget */
- (int)setClasses:(uint32_t)n slotBytes:(const uint64_t *)bytes capacity:(const uint32_t *)cap {
    if (!n || n > RL_TOPK_MAX_CLASSES || !bytes || !cap) { topk_set_error("invalid top-k slot classes"); return 0; }
    @synchronized (self) {
        if (_slabs.count) { topk_set_error("top-k slot classes must be set before any slab is allocated"); return 0; }
        uint64_t total = 0; uint32_t slots = 0, slabs = 0;
        uint32_t first[RL_TOPK_MAX_CLASSES + 1], slab[RL_TOPK_MAX_CLASSES + 1]; uint64_t sz[RL_TOPK_MAX_CLASSES];
        for (uint32_t c = 0; c < n; ++c) {
            sz[c] = round_up_u64(bytes[c], 4096u);
            if (!sz[c] || !cap[c] || (uint64_t)cap[c] > UINT32_MAX - slots || sz[c] * cap[c] > _budgetBytes - total) {
                topk_set_error("top-k slot classes exceed the cache budget"); return 0;
            }
            first[c] = slots; slab[c] = slabs;
            slots += cap[c]; total += sz[c] * cap[c]; slabs += (cap[c] + _slotsPerSlab - 1u) / _slotsPerSlab;
        }
        first[n] = slots; slab[n] = slabs;
        uint32_t *inflight = calloc(slots, sizeof(uint32_t)), *generation = calloc(slots, sizeof(uint32_t)), *map = calloc(slabs, sizeof(uint32_t));
        if (!inflight || !generation || !map) { free(inflight); free(generation); free(map); topk_set_error("top-k slot class tables allocation failed"); return 0; }
        free(_slotInflight); free(_slotGeneration); free(_slabMap);
        _slotInflight = inflight; _slotGeneration = generation; _slabMap = map;
        _nCls = n; _capacity = slots;
        for (uint32_t c = 0; c <= n; ++c) { _clsFirst[c] = first[c]; _clsSlab[c] = slab[c]; }
        for (uint32_t c = 0; c < n; ++c) _clsBytes[c] = sz[c];
        _slotBytes = sz[0];
        for (uint32_t c = 1; c < n; ++c) if (sz[c] > _slotBytes) _slotBytes = sz[c];
    }
    return 1;
}

static uint32_t pool_class_of(const RMTopKPool *p, uint32_t slot) {
    uint32_t c = 0;
    while (c + 1u < p->_nCls && slot >= p->_clsFirst[c + 1u]) ++c;
    return c;
}

- (uint64_t)bytesForSlot:(uint32_t)slot {
    return slot < _capacity ? _clsBytes[pool_class_of(self, slot)] : 0u;
}

- (int)ensureBatchPairs:(uint32_t)pairs experts:(uint32_t)experts tokens:(uint32_t)ntok topk:(uint32_t)top_k hidden:(uint32_t)hidden ffn:(uint32_t)ffn {
    const NSUInteger pt = (NSUInteger)pairs * sizeof(uint32_t), pw = (NSUInteger)pairs * sizeof(float);
    const NSUInteger es = (NSUInteger)(experts + 1u) * sizeof(uint32_t), tp = (NSUInteger)ntok * top_k * sizeof(uint32_t);
    const NSUInteger actBytes = (NSUInteger)pairs * ffn * sizeof(float), tmpBytes = (NSUInteger)pairs * hidden * sizeof(float);
    if (!_pairTokenBuf || _pairTokenBuf.length < pt) _pairTokenBuf = [_device newBufferWithLength:pt options:MTLResourceStorageModeShared];
    if (!_pairExpertBuf || _pairExpertBuf.length < pt) _pairExpertBuf = [_device newBufferWithLength:pt options:MTLResourceStorageModeShared];
    if (!_sliceFirstBuf || _sliceFirstBuf.length < pt) _sliceFirstBuf = [_device newBufferWithLength:pt options:MTLResourceStorageModeShared];
    if (!_sliceCountBuf || _sliceCountBuf.length < pt) _sliceCountBuf = [_device newBufferWithLength:pt options:MTLResourceStorageModeShared];
    if (!_pairWeightBuf || _pairWeightBuf.length < pw) _pairWeightBuf = [_device newBufferWithLength:pw options:MTLResourceStorageModeShared];
    if (!_expertStartBuf || _expertStartBuf.length < es) _expertStartBuf = [_device newBufferWithLength:es options:MTLResourceStorageModeShared];
    if (!_tokPairBuf || _tokPairBuf.length < tp) _tokPairBuf = [_device newBufferWithLength:tp options:MTLResourceStorageModeShared];
    if (!_actB || _actB.length < actBytes) _actB = [_device newBufferWithLength:actBytes options:MTLResourceStorageModeShared];
    if (!_tmpB || _tmpB.length < tmpBytes) _tmpB = [_device newBufferWithLength:tmpBytes options:MTLResourceStorageModeShared];
    if (!_pairTokenBuf || !_pairExpertBuf || !_sliceFirstBuf || !_sliceCountBuf || !_pairWeightBuf || !_expertStartBuf || !_tokPairBuf || !_actB || !_tmpB) {
        topk_set_error("failed to allocate batched top-k buffers");
        return 0;
    }
    return 1;
}

- (int)ensureScratchHidden:(uint32_t)hidden ffn:(uint32_t)ffn {
    const NSUInteger inBytes = (NSUInteger)hidden * sizeof(float);
    const NSUInteger actBytes = (NSUInteger)REDMETAL_TOPK_MAX * ffn * sizeof(float);
    const NSUInteger tmpBytes = (NSUInteger)REDMETAL_TOPK_MAX * hidden * sizeof(float);
    if (!_inBuffer || _inBuffer.length < inBytes) _inBuffer = [_device newBufferWithLength:inBytes options:MTLResourceStorageModeShared];
    if (!_actBuffer || _actBuffer.length < actBytes) _actBuffer = [_device newBufferWithLength:actBytes options:MTLResourceStorageModeShared];
    if (!_tmpBuffer || _tmpBuffer.length < tmpBytes) _tmpBuffer = [_device newBufferWithLength:tmpBytes options:MTLResourceStorageModeShared];
    if (!_outBuffer || _outBuffer.length < inBytes) _outBuffer = [_device newBufferWithLength:inBytes options:MTLResourceStorageModeShared];
    if (!_inBuffer || !_actBuffer || !_tmpBuffer || !_outBuffer) {
        topk_set_error("failed to allocate top-k activation buffers");
        return 0;
    }
    return 1;
}

- (id<MTLBuffer>)bufferForSlot:(uint32_t)slot offset:(NSUInteger *)offset {
    if (slot >= _capacity) {
        topk_set_error("top-k slot %u exceeds capacity %u", slot, _capacity);
        return nil;
    }
    const uint32_t cls = pool_class_of(self, slot);
    const uint32_t inClass = slot - _clsFirst[cls];
    const uint32_t slabIndex = _clsSlab[cls] + inClass / _slotsPerSlab;
    const uint32_t localSlot = inClass % _slotsPerSlab;
    const uint64_t clsBytes = _clsBytes[cls];
    @synchronized (self) {
        if (!_slabMap[slabIndex]) {
            const uint32_t newSlab = (uint32_t)_slabs.count;
            const uint32_t first = (inClass / _slotsPerSlab) * _slotsPerSlab;
            const uint32_t remaining = (_clsFirst[cls + 1u] - _clsFirst[cls]) - first;
            const uint32_t count = remaining < _slotsPerSlab ? remaining : _slotsPerSlab;
            const uint64_t bytes64 = (uint64_t)count * clsBytes;
            if (!bytes64 || bytes64 > (uint64_t)NSUIntegerMax ||
                _allocatedBytes > _budgetBytes - bytes64) {
                topk_set_error("top-k slab allocation would exceed hard budget");
                return nil;
            }
            id<MTLBuffer> slab = [_device newBufferWithLength:(NSUInteger)bytes64
                                                       options:MTLResourceStorageModeShared];
            if (!slab) {
                topk_set_error("failed to allocate top-k Metal slab %.2f MiB",
                    (double)bytes64 / (1024.0 * 1024.0));
                return nil;
            }
            slab.label = [NSString stringWithFormat:@"redlite_topk_slab_%u", newSlab];
            [_slabs addObject:slab];
            _slabMap[slabIndex] = (uint32_t)_slabs.count;
            _allocatedBytes += bytes64;
            if (@available(macOS 15.0, *)) {
                if (_residencySet) { [_residencySet addAllocation:slab]; [_residencySet commit]; }
            }
        }
        id<MTLBuffer> slab = _slabs[_slabMap[slabIndex] - 1u];
        const uint64_t inner = (uint64_t)localSlot * clsBytes;
        if (inner > (uint64_t)NSUIntegerMax) {
            topk_set_error("top-k slot offset exceeds NSUInteger");
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

static RMTopKPool *topk_obj(redmetal_topk_pool_t handle) {
    if (!handle) {
        topk_set_error("Red Metal top-k pool is null");
        return nil;
    }
    return (__bridge RMTopKPool *)handle;
}

static int topk_has_residency_set(RMTopKPool *p) {
    if (@available(macOS 15.0, *)) return p->_residencySet != nil;
    return 0;
}

static id<MTLBuffer> topk_grid_for_type(RMTopKPool *p, uint32_t type) {
    if (!tw_ok(type)) { topk_set_error("unsupported top-k GGML type word 0x%x", type); return nil; }
    const uint32_t g = tw_gate(type), d = tw_down(type);
    if (g == GGML_TYPE_IQ2_XS || d == GGML_TYPE_IQ2_XS) return p->_iq2Grid;
    if (g == GGML_TYPE_IQ1_M || d == GGML_TYPE_IQ1_M) return p->_iq1Grid;
    return p->_iq2Grid;   /* IQ3 types use constant codebooks; any buffer satisfies the grid argument */
}

static void mark_slots(RMTopKPool *p, const uint32_t *slot_ids, uint32_t top_k, int delta) {
    for (uint32_t i = 0; i < top_k; ++i) [p markSlot:slot_ids[i] delta:delta];
}

/* lanes per row: blocks x sub-block parts, capped at 32 (power of two); each lane decodes a
 * contiguous range of quant groups of one block (16 groups per IQ2_XS block, 32 per IQ1_M block) */
static uint32_t lanes_for_blocks(uint32_t blocks) {
    uint32_t lanes = 1u;
    while (lanes < blocks && lanes < 32u) lanes <<= 1;
    /* dev38: at most 4 lanes per 256-value block (was 16). The 512-column down rows went from 32 lanes of 16 values
     * each plus a 5-step shuffle reduction to 8 lanes of 64 values: decode 77.0 -> 84.1 tok/s (IQ2_XXS, full residency) */
    while (lanes * 2u <= 32u && (lanes * 2u) / blocks <= 4u && (lanes * 2u) % blocks == 0u) lanes <<= 1;
    return lanes;
}

uint32_t redmetal_topk_abi_version(void) { return REDMETAL_TOPK_ABI_VERSION; }
const char *redmetal_topk_last_error(void) {
    return g_topk_error[0] ? g_topk_error : "unknown Red Metal top-k error";
}

redmetal_topk_pool_t redmetal_topk_pool_create(
        const char *model_path,
        uint64_t budget_bytes,
        uint64_t slot_bytes,
        uint32_t slots_per_slab,
        const int8_t *iq2_xs_grid,
        uint32_t iq2_xs_grid_count,
        const int8_t *iq1_m_grid,
        uint32_t iq1_m_grid_count) {
    @autoreleasepool {
        g_topk_error[0] = '\0';
        RMTopKPool *pool = [[RMTopKPool alloc] initWithModelPath:model_path
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

void redmetal_topk_pool_destroy(redmetal_topk_pool_t handle) {
    if (!handle) return;
    @autoreleasepool {
        RMTopKPool *pool = (__bridge_transfer RMTopKPool *)handle;
        (void)pool;
    }
}

int redmetal_topk_pool_set_classes(redmetal_topk_pool_t handle, uint32_t n, const uint64_t *slot_bytes, const uint32_t *capacity) {
    RMTopKPool *p = topk_obj(handle);
    return p ? [p setClasses:n slotBytes:slot_bytes capacity:capacity] : 0;
}

void *redmetal_topk_pool_tmp_buffer(redmetal_topk_pool_t handle) {
    RMTopKPool *p = topk_obj(handle);
    return p ? (__bridge void *)p->_tmpBuffer : NULL;
}

uint32_t redmetal_topk_pool_capacity(redmetal_topk_pool_t handle) {
    RMTopKPool *p = topk_obj(handle); return p ? p->_capacity : 0;
}
uint32_t redmetal_topk_pool_slab_count(redmetal_topk_pool_t handle) {
    RMTopKPool *p = topk_obj(handle); return p ? (uint32_t)p->_slabs.count : 0;
}
uint64_t redmetal_topk_pool_allocated_bytes(redmetal_topk_pool_t handle) {
    RMTopKPool *p = topk_obj(handle); return p ? p->_allocatedBytes : 0;
}
uint64_t redmetal_topk_pool_bytes_read(redmetal_topk_pool_t handle) {
    RMTopKPool *p = topk_obj(handle); if (!p) return 0; @synchronized (p) { return p->_bytesRead; }
}
uint64_t redmetal_topk_pool_read_calls(redmetal_topk_pool_t handle) {
    RMTopKPool *p = topk_obj(handle); if (!p) return 0; @synchronized (p) { return p->_readCalls; }
}
double redmetal_topk_pool_read_ms(redmetal_topk_pool_t handle) {
    RMTopKPool *p = topk_obj(handle); if (!p) return 0.0; @synchronized (p) { return p->_readMs; }
}
uint32_t redmetal_topk_pool_slot_generation(redmetal_topk_pool_t handle, uint32_t slot_id) {
    RMTopKPool *p = topk_obj(handle);
    if (!p || slot_id >= p->_capacity) return 0;
    @synchronized (p) { return p->_slotGeneration[slot_id]; }
}

int redmetal_topk_pool_slot_inflight(redmetal_topk_pool_t handle, uint32_t slot_id) {
    RMTopKPool *p = topk_obj(handle);
    if (!p || slot_id >= p->_capacity) return 0;
    @synchronized (p) { return p->_slotInflight[slot_id] != 0; }
}

int redmetal_topk_pool_load_expert(
        redmetal_topk_pool_t handle,
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
        RMTopKPool *p = topk_obj(handle);
        if (!p || slot_id >= p->_capacity) return 0;
        @synchronized (p) {
            if (p->_slotInflight[slot_id] != 0) {
                topk_set_error("refusing to overwrite in-flight top-k slot %u", slot_id);
                return 0;
            }
            p->_slotGeneration[slot_id]++;
        }
        if (gate_bytes > UINT64_MAX - up_bytes || gate_bytes + up_bytes > UINT64_MAX - down_bytes) {
            topk_set_error("top-k expert payload size overflow");
            return 0;
        }
        const uint64_t payload = gate_bytes + up_bytes + down_bytes;
        if (payload > [p bytesForSlot:slot_id]) {
            topk_set_error("top-k expert payload exceeds slot size");
            return 0;
        }
        NSUInteger baseOffset = 0;
        id<MTLBuffer> slab = [p bufferForSlot:slot_id offset:&baseOffset];
        if (!slab) return 0;
        uint8_t *base = (uint8_t *)[slab contents];
        if (!base) {
            topk_set_error("top-k shared Metal slab is not CPU visible");
            return 0;
        }
        uint64_t localBytes = 0, localCalls = 0;
        const double t0 = topk_now_ms();
        const int ok =
            topk_pread_all(p->_fd, p->_fileSize, gate_file_offset, gate_bytes,
                           base + baseOffset, &localBytes, &localCalls) &&
            topk_pread_all(p->_fd, p->_fileSize, up_file_offset, up_bytes,
                           base + baseOffset + gate_bytes, &localBytes, &localCalls) &&
            topk_pread_all(p->_fd, p->_fileSize, down_file_offset, down_bytes,
                           base + baseOffset + gate_bytes + up_bytes, &localBytes, &localCalls);
        const double ms = topk_now_ms() - t0;
        if (!ok) return 0;
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

int redmetal_topk_pool_slot_addresses(
        redmetal_topk_pool_t handle,
        uint32_t slot_id,
        uint64_t gate_bytes,
        uint64_t up_bytes,
        uint64_t down_bytes,
        uint64_t *gate_address,
        uint64_t *up_address,
        uint64_t *down_address) {
    @autoreleasepool {
        RMTopKPool *p = topk_obj(handle);
        if (!p || slot_id >= p->_capacity) return 0;
        if (gate_bytes > UINT64_MAX - up_bytes || gate_bytes + up_bytes > UINT64_MAX - down_bytes ||
            gate_bytes + up_bytes + down_bytes > [p bytesForSlot:slot_id]) return 0;
        NSUInteger inner = 0;
        id<MTLBuffer> slab = [p bufferForSlot:slot_id offset:&inner];
        if (!slab) return 0;
        uint64_t base = 0;
        if (@available(macOS 13.0, *)) base = (uint64_t)[slab gpuAddress] + (uint64_t)inner;
        if (!base) {
            topk_set_error("top-k slab did not expose a GPU virtual address");
            return 0;
        }
        if (gate_address) *gate_address = base;
        if (up_address) *up_address = base + gate_bytes;
        if (down_address) *down_address = base + gate_bytes + up_bytes;
        return 1;
    }
}

static int topk_validate(RMTopKPool *p, const uint32_t *slot_ids, const uint64_t *gate_bytes, const uint64_t *up_bytes,
        const uint64_t *down_bytes, uint32_t top_k, uint32_t ggml_type, uint32_t hidden_size, uint32_t ffn_size,
        uint32_t output_row_start, uint32_t output_row_count) {
    if (!p || !slot_ids || !gate_bytes || !up_bytes || !down_bytes) return 0;
    if (!top_k || top_k > REDMETAL_TOPK_MAX) {
        topk_set_error("top_k=%u must be in [1,%u]", top_k, REDMETAL_TOPK_MAX);
        return 0;
    }
    if (!topk_grid_for_type(p, ggml_type)) return 0;
    if (!hidden_size || hidden_size % QK_IQ || !ffn_size || ffn_size % QK_IQ ||
        !output_row_count || output_row_start >= hidden_size || output_row_count > hidden_size - output_row_start) {
        topk_set_error("invalid top-k dimensions hidden=%u ffn=%u start=%u count=%u",
            hidden_size, ffn_size, output_row_start, output_row_count);
        return 0;
    }
    for (uint32_t i = 0; i < top_k; ++i) {
        if (slot_ids[i] >= p->_capacity) {
            topk_set_error("top-k slot %u exceeds capacity %u", slot_ids[i], p->_capacity);
            return 0;
        }
        for (uint32_t j = 0; j < i; ++j) {
            if (slot_ids[i] == slot_ids[j]) {
                topk_set_error("top-k slot list contains duplicate slot %u", slot_ids[i]);
                return 0;
            }
        }
    }
    const uint64_t gateRowBytes = (uint64_t)(hidden_size / QK_IQ) * tw_block_bytes(tw_gate(ggml_type));
    const uint64_t downRowBytes = (uint64_t)(ffn_size / QK_IQ) * tw_block_bytes(tw_down(ggml_type));
    const uint64_t expectedGate = (uint64_t)ffn_size * gateRowBytes;
    const uint64_t expectedDown = (uint64_t)hidden_size * downRowBytes;
    for (uint32_t i = 0; i < top_k; ++i) {
        if (gate_bytes[i] != expectedGate || up_bytes[i] != expectedGate || down_bytes[i] != expectedDown) {
            topk_set_error("top-k expert %u matrix byte mismatch", i);
            return 0;
        }
        if (gate_bytes[i] > UINT64_MAX - up_bytes[i] ||
            gate_bytes[i] + up_bytes[i] > UINT64_MAX - down_bytes[i] ||
            gate_bytes[i] + up_bytes[i] + down_bytes[i] > [p bytesForSlot:slot_ids[i]]) {
            topk_set_error("top-k expert %u exceeds slot bounds", i);
            return 0;
        }
    }
    return 1;
}

/* Fill the slot address table / weights and encode the three dispatches into cb. Marks slots in-flight. */
static int topk_encode_into(RMTopKPool *p, id<MTLCommandBuffer> cb, const uint32_t *slot_ids, const uint64_t *gate_bytes,
        const uint64_t *up_bytes, const float *router_weights, uint32_t top_k, uint32_t ggml_type, uint32_t hidden_size,
        uint32_t ffn_size, uint32_t output_row_start, uint32_t output_row_count,
        id<MTLBuffer> input, NSUInteger input_offset, id<MTLBuffer> output, NSUInteger output_offset) {
    id<MTLBuffer> grid = topk_grid_for_type(p, ggml_type);
    if (!grid || !cb || !input || !output) { topk_set_error("invalid top-k encode buffers"); return 0; }
    if (![p ensureScratchHidden:hidden_size ffn:ffn_size]) return 0;
    /* dev72: the slot addresses and weights go into the command buffer (setBytes), not into shared buffers, so two
     * encodes may be in flight at once (the bounded-cache 2-row verify) */
    if (top_k > REDMETAL_TOPK_MAX) { topk_set_error("top-k encode: too many experts"); return 0; }
    uint64_t table[REDMETAL_TOPK_MAX];
    float weights[REDMETAL_TOPK_MAX];
    NSMutableArray<id<MTLBuffer>> *resident = [NSMutableArray arrayWithCapacity:top_k];
    for (uint32_t i = 0; i < top_k; ++i) {
        NSUInteger slotBase = 0;
        id<MTLBuffer> slab = [p bufferForSlot:slot_ids[i] offset:&slotBase];
        if (!slab) return 0;
        uint64_t base = 0;
        if (@available(macOS 13.0, *)) base = (uint64_t)[slab gpuAddress];
        if (!base) { topk_set_error("top-k slab did not expose a GPU virtual address"); return 0; }
        table[i] = base + (uint64_t)slotBase;
        weights[i] = router_weights[i];
        if (![resident containsObject:slab]) [resident addObject:slab];
    }
    const uint32_t gate_lanes = lanes_for_blocks(hidden_size / QK_IQ);
    const uint32_t down_lanes = lanes_for_blocks(ffn_size / QK_IQ);
    const uint64_t up_offset = gate_bytes[0];
    const uint64_t down_offset = gate_bytes[0] + up_bytes[0];

    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    if (!enc) { topk_set_error("failed to create top-k compute encoder"); return 0; }
    for (id<MTLBuffer> slab in resident) [enc useResource:slab usage:MTLResourceUsageRead];
    [enc setComputePipelineState:p->_gateupPipeline];
    [enc setBytes:table length:(NSUInteger)top_k * sizeof(uint64_t) atIndex:0];
    [enc setBytes:&hidden_size length:sizeof(hidden_size) atIndex:1];
    [enc setBytes:&ffn_size length:sizeof(ffn_size) atIndex:2];
    [enc setBytes:&up_offset length:sizeof(up_offset) atIndex:3];
    [enc setBuffer:input offset:input_offset atIndex:4];
    [enc setBuffer:p->_actBuffer offset:0 atIndex:5];
    [enc setBuffer:grid offset:0 atIndex:6];
    [enc setBytes:&top_k length:sizeof(top_k) atIndex:7];
    [enc setBytes:&ggml_type length:sizeof(ggml_type) atIndex:8];
    [enc setBytes:&gate_lanes length:sizeof(gate_lanes) atIndex:9];
    [enc dispatchThreads:MTLSizeMake((NSUInteger)ffn_size * gate_lanes, top_k, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [enc endEncoding];

    enc = [cb computeCommandEncoder];
    for (id<MTLBuffer> slab in resident) [enc useResource:slab usage:MTLResourceUsageRead];
    [enc setComputePipelineState:p->_downPipeline];
    [enc setBytes:table length:(NSUInteger)top_k * sizeof(uint64_t) atIndex:0];
    [enc setBytes:&ffn_size length:sizeof(ffn_size) atIndex:1];
    [enc setBytes:&output_row_start length:sizeof(output_row_start) atIndex:2];
    [enc setBytes:&output_row_count length:sizeof(output_row_count) atIndex:3];
    [enc setBytes:&down_offset length:sizeof(down_offset) atIndex:4];
    [enc setBuffer:p->_actBuffer offset:0 atIndex:5];
    [enc setBuffer:p->_tmpBuffer offset:0 atIndex:6];
    [enc setBuffer:grid offset:0 atIndex:7];
    [enc setBytes:&top_k length:sizeof(top_k) atIndex:8];
    [enc setBytes:&ggml_type length:sizeof(ggml_type) atIndex:9];
    [enc setBytes:&down_lanes length:sizeof(down_lanes) atIndex:10];
    [enc dispatchThreads:MTLSizeMake((NSUInteger)output_row_count * down_lanes, top_k, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [enc endEncoding];

    enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:p->_sumPipeline];
    [enc setBuffer:p->_tmpBuffer offset:0 atIndex:0];
    [enc setBytes:weights length:(NSUInteger)top_k * sizeof(float) atIndex:1];
    [enc setBuffer:output offset:output_offset atIndex:2];
    [enc setBytes:&output_row_count length:sizeof(output_row_count) atIndex:3];
    [enc setBytes:&top_k length:sizeof(top_k) atIndex:4];
    {
        const NSUInteger tg = MIN((NSUInteger)output_row_count, (NSUInteger)64u);
        [enc dispatchThreads:MTLSizeMake(output_row_count, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    }
    [enc endEncoding];
    mark_slots(p, slot_ids, top_k, +1);
    return 1;
}

int redmetal_topk_pool_execute(
        redmetal_topk_pool_t handle,
        const uint32_t *slot_ids,
        const uint64_t *gate_bytes,
        const uint64_t *up_bytes,
        const uint64_t *down_bytes,
        const float *router_weights,
        uint32_t top_k,
        uint32_t ggml_type,
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
        RMTopKPool *p = topk_obj(handle);
        if (!p || !router_weights || !input || !output) return 0;
        if (!topk_validate(p, slot_ids, gate_bytes, up_bytes, down_bytes, top_k, ggml_type, hidden_size, ffn_size,
                output_row_start, output_row_count)) return 0;
        if (input_count != hidden_size || output_count < output_row_count) {
            topk_set_error("invalid top-k host buffer sizes");
            return 0;
        }
        if (![p ensureScratchHidden:hidden_size ffn:ffn_size]) return 0;
        memcpy([p->_inBuffer contents], input, (size_t)hidden_size * sizeof(float));
        id<MTLCommandBuffer> cb = [p->_queue commandBuffer];
        if (!cb) {
            topk_set_error("failed to create top-k command buffer");
            return 0;
        }
        if (!topk_encode_into(p, cb, slot_ids, gate_bytes, up_bytes, router_weights, top_k, ggml_type, hidden_size, ffn_size,
                output_row_start, output_row_count, p->_inBuffer, 0, p->_outBuffer, 0)) {
            topk_set_error("failed to encode top-k expert execution");
            return 0;
        }
        const double t0 = topk_now_ms();
        [cb commit];
        [cb waitUntilCompleted];
        const double ms = topk_now_ms() - t0;
        mark_slots(p, slot_ids, top_k, -1);
        if (cb.status == MTLCommandBufferStatusError) {
            topk_set_error("top-k Metal command failed: %s",
                cb.error.localizedDescription.UTF8String ?: "unknown Metal error");
            return 0;
        }
        memcpy(output, [p->_outBuffer contents], (size_t)output_row_count * sizeof(float));
        if (elapsed_ms) *elapsed_ms = ms;
        return 1;
    }
}

/* Batched prefill: n_expert experts addressed by three GPU addresses each (gate/up/down rows) serving
 * n_pairs (expert, token) pairs sorted by expert. `resident` lists the buffers the addresses live in. */
static int topk_encode_batched_dispatch(RMTopKPool *p, id<MTLCommandBuffer> cb, NSArray<id<MTLBuffer>> *resident,
        uint32_t n_expert, uint32_t ggml_type, uint32_t hidden_size, uint32_t ffn_size,
        uint32_t tok_first, uint32_t ntok, uint32_t top_k, uint32_t n_pairs,
        const uint32_t *pair_token, const float *pair_weight, const uint32_t *expert_start, const uint32_t *tok_pair,
        id<MTLBuffer> input, NSUInteger input_offset, id<MTLBuffer> output, NSUInteger output_offset) {
    id<MTLBuffer> grid = topk_grid_for_type(p, ggml_type);
    if (!grid || !cb || !input || !output) { topk_set_error("invalid batched top-k encode buffers"); return 0; }
    if (![p ensureBatchPairs:n_pairs experts:n_expert tokens:ntok topk:top_k hidden:hidden_size ffn:ffn_size]) return 0;
    memcpy(p->_pairTokenBuf.contents, pair_token, (size_t)n_pairs * sizeof(uint32_t));
    memcpy(p->_pairWeightBuf.contents, pair_weight, (size_t)n_pairs * sizeof(float));
    memcpy(p->_expertStartBuf.contents, expert_start, (size_t)(n_expert + 1u) * sizeof(uint32_t));
    memcpy(p->_tokPairBuf.contents, tok_pair, (size_t)ntok * top_k * sizeof(uint32_t));
    uint32_t n_slices = 0;
    /* dev30: the matrix kernels take slices of up to 16 pairs and need 32-row / 32-column tiles */
    const int mm = p->_useMM && hidden_size % 256u == 0u && ffn_size % 256u == 0u;
    const uint32_t slice = mm ? REDMETAL_TOPK_MM_SLICE : REDMETAL_TOPK_SLICE;
    {
        /* slices of at most REDMETAL_TOPK_SLICE consecutive pairs of the same expert: one decoded weight serves all of them */
        uint32_t *pe = (uint32_t *)p->_pairExpertBuf.contents;
        uint32_t *sf = (uint32_t *)p->_sliceFirstBuf.contents;
        uint32_t *sc = (uint32_t *)p->_sliceCountBuf.contents;
        for (uint32_t u = 0; u < n_expert; ++u) {
            for (uint32_t q = expert_start[u]; q < expert_start[u + 1u]; ++q) pe[q] = u;
            for (uint32_t q = expert_start[u]; q < expert_start[u + 1u]; q += slice) {
                const uint32_t left = expert_start[u + 1u] - q;
                sf[n_slices] = q; sc[n_slices] = left >= slice ? slice : left; n_slices++;
            }
        }
    }
    const uint32_t gate_lanes = lanes_for_blocks(hidden_size / QK_IQ);
    const uint32_t down_lanes = lanes_for_blocks(ffn_size / QK_IQ);
    const uint32_t zero = 0u;

    const int use_res = !topk_has_residency_set(p) || resident.count <= 3u;   /* mapped tensors are not slabs */
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    if (!enc) { topk_set_error("failed to create batched top-k compute encoder"); return 0; }
    if (use_res) for (id<MTLBuffer> b in resident) [enc useResource:b usage:MTLResourceUsageRead];
    [enc setComputePipelineState:p->_gateupBPipeline];
    [enc setBuffer:p->_slotTable3 offset:0 atIndex:0];
    [enc setBytes:&hidden_size length:sizeof(hidden_size) atIndex:1];
    [enc setBytes:&ffn_size length:sizeof(ffn_size) atIndex:2];
    [enc setBytes:&zero length:sizeof(zero) atIndex:3];
    [enc setBuffer:input offset:input_offset atIndex:4];
    [enc setBuffer:p->_actB offset:0 atIndex:5];
    [enc setBuffer:grid offset:0 atIndex:6];
    [enc setBytes:&n_expert length:sizeof(n_expert) atIndex:7];
    [enc setBytes:&ggml_type length:sizeof(ggml_type) atIndex:8];
    [enc setBytes:&gate_lanes length:sizeof(gate_lanes) atIndex:9];
    [enc setBuffer:p->_pairTokenBuf offset:0 atIndex:10];
    [enc setBuffer:p->_pairExpertBuf offset:0 atIndex:11];
    [enc setBytes:&n_pairs length:sizeof(n_pairs) atIndex:12];
    [enc setBuffer:p->_sliceFirstBuf offset:0 atIndex:13];
    [enc setBuffer:p->_sliceCountBuf offset:0 atIndex:14];
    [enc setBytes:&n_slices length:sizeof(n_slices) atIndex:15];
    if (mm) {
        [enc setComputePipelineState:p->_gateupMMPipeline];
        [enc dispatchThreadgroups:MTLSizeMake(ffn_size / 32u, n_slices, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    } else {
        [enc dispatchThreads:MTLSizeMake((NSUInteger)((ffn_size + 3u) / 4u) * gate_lanes, n_slices, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    }
    [enc endEncoding];

    enc = [cb computeCommandEncoder];
    if (use_res) for (id<MTLBuffer> b in resident) [enc useResource:b usage:MTLResourceUsageRead];
    [enc setComputePipelineState:p->_downBPipeline];
    [enc setBuffer:p->_slotTable3 offset:0 atIndex:0];
    [enc setBytes:&ffn_size length:sizeof(ffn_size) atIndex:1];
    [enc setBytes:&hidden_size length:sizeof(hidden_size) atIndex:2];
    [enc setBytes:&zero length:sizeof(zero) atIndex:3];
    [enc setBuffer:p->_actB offset:0 atIndex:4];
    [enc setBuffer:p->_tmpB offset:0 atIndex:5];
    [enc setBuffer:grid offset:0 atIndex:6];
    [enc setBytes:&n_expert length:sizeof(n_expert) atIndex:7];
    [enc setBytes:&ggml_type length:sizeof(ggml_type) atIndex:8];
    [enc setBytes:&down_lanes length:sizeof(down_lanes) atIndex:9];
    [enc setBuffer:p->_pairExpertBuf offset:0 atIndex:10];
    [enc setBytes:&n_pairs length:sizeof(n_pairs) atIndex:11];
    [enc setBuffer:p->_sliceFirstBuf offset:0 atIndex:12];
    [enc setBuffer:p->_sliceCountBuf offset:0 atIndex:13];
    [enc setBytes:&n_slices length:sizeof(n_slices) atIndex:14];
    if (mm) {
        [enc setComputePipelineState:p->_downMMPipeline];
        [enc dispatchThreadgroups:MTLSizeMake(hidden_size / 32u, n_slices, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    } else {
        [enc dispatchThreads:MTLSizeMake((NSUInteger)((hidden_size + 3u) / 4u) * down_lanes, n_slices, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    }
    [enc endEncoding];

    enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:p->_sumBPipeline];
    [enc setBuffer:p->_tmpB offset:0 atIndex:0];
    [enc setBuffer:p->_pairWeightBuf offset:0 atIndex:1];
    [enc setBuffer:p->_tokPairBuf offset:0 atIndex:2];
    [enc setBuffer:output offset:output_offset atIndex:3];
    [enc setBytes:&hidden_size length:sizeof(hidden_size) atIndex:4];
    [enc setBytes:&top_k length:sizeof(top_k) atIndex:5];
    [enc setBytes:&tok_first length:sizeof(tok_first) atIndex:6];
    [enc setBytes:&ntok length:sizeof(ntok) atIndex:7];
    [enc dispatchThreads:MTLSizeMake(hidden_size, ntok, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [enc endEncoding];
    return 1;
}

/* slot-pool variant: gate/up/down of expert i live in slot slot_ids[i] at offsets 0 / gate_bytes / gate_bytes+up_bytes */
static int topk_encode_batched_into(RMTopKPool *p, id<MTLCommandBuffer> cb, const uint32_t *slot_ids, const uint64_t *gate_bytes,
        const uint64_t *up_bytes, uint32_t n_expert, uint32_t ggml_type, uint32_t hidden_size, uint32_t ffn_size,
        uint32_t tok_first, uint32_t ntok, uint32_t top_k, uint32_t n_pairs,
        const uint32_t *pair_token, const float *pair_weight, const uint32_t *expert_start, const uint32_t *tok_pair,
        id<MTLBuffer> input, NSUInteger input_offset, id<MTLBuffer> output, NSUInteger output_offset) {
    uint64_t *table = (uint64_t *)[p->_slotTable3 contents];
    NSMutableArray<id<MTLBuffer>> *resident = [NSMutableArray arrayWithCapacity:n_expert];
    for (uint32_t i = 0; i < n_expert; ++i) {
        NSUInteger slotBase = 0;
        id<MTLBuffer> slab = [p bufferForSlot:slot_ids[i] offset:&slotBase];
        if (!slab) return 0;
        uint64_t base = 0;
        if (@available(macOS 13.0, *)) base = (uint64_t)[slab gpuAddress];
        if (!base) { topk_set_error("top-k slab did not expose a GPU virtual address"); return 0; }
        const uint64_t slot = base + (uint64_t)slotBase;
        table[i] = slot;
        table[REDMETAL_TOPK_MAX + i] = slot + gate_bytes[0];
        table[2u * REDMETAL_TOPK_MAX + i] = slot + gate_bytes[0] + up_bytes[0];
        if (![resident containsObject:slab]) [resident addObject:slab];
    }
    if (!topk_encode_batched_dispatch(p, cb, resident, n_expert, ggml_type, hidden_size, ffn_size, tok_first, ntok, top_k, n_pairs,
            pair_token, pair_weight, expert_start, tok_pair, input, input_offset, output, output_offset)) return 0;
    mark_slots(p, slot_ids, n_expert, +1);
    return 1;
}

int redmetal_topk_pool_encode_mapped(
        redmetal_topk_pool_t handle,
        void *mtl_command_buffer,
        void *gate_buffer, uint64_t gate_addr0,
        void *up_buffer, uint64_t up_addr0,
        void *down_buffer, uint64_t down_addr0,
        uint64_t gate_bytes, uint64_t up_bytes, uint64_t down_bytes,
        const uint32_t *expert_ids,
        uint32_t n_expert,
        uint32_t ggml_type,
        uint32_t hidden_size,
        uint32_t ffn_size,
        uint32_t tok_first,
        uint32_t ntok,
        uint32_t top_k,
        uint32_t n_pairs,
        const uint32_t *pair_token,
        const float *pair_weight,
        const uint32_t *expert_start,
        const uint32_t *tok_pair,
        void *mtl_input_buffer,
        uint64_t input_offset,
        void *mtl_output_buffer,
        uint64_t output_offset) {
    @autoreleasepool {
        RMTopKPool *p = topk_obj(handle);
        if (!p || !mtl_command_buffer || !gate_buffer || !up_buffer || !down_buffer || !expert_ids || !pair_token || !pair_weight ||
            !expert_start || !tok_pair || !mtl_input_buffer || !mtl_output_buffer) return 0;
        if (!n_expert || n_expert > REDMETAL_TOPK_MAX || !ntok || !top_k || n_pairs != ntok * top_k) { topk_set_error("invalid mapped top-k request"); return 0; }
        if ((ggml_type != 17u && ggml_type != 29u) || hidden_size % QK_IQ || ffn_size % QK_IQ) { topk_set_error("mapped top-k supports IQ2_XS/IQ1_M with 256-aligned sizes"); return 0; }
        if (expert_start[0] != 0u || expert_start[n_expert] != n_pairs) { topk_set_error("mapped top-k expert_start is inconsistent"); return 0; }
        for (uint32_t i = 0; i < n_pairs; ++i) if (pair_token[i] >= tok_first + ntok) { topk_set_error("mapped top-k pair token out of range"); return 0; }
        for (uint32_t i = 0; i < ntok * top_k; ++i) if (tok_pair[i] >= n_pairs) { topk_set_error("mapped top-k tok_pair out of range"); return 0; }
        id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)mtl_command_buffer;
        id<MTLBuffer> gb = (__bridge id<MTLBuffer>)gate_buffer, ub = (__bridge id<MTLBuffer>)up_buffer, db = (__bridge id<MTLBuffer>)down_buffer;
        id<MTLBuffer> input = (__bridge id<MTLBuffer>)mtl_input_buffer;
        id<MTLBuffer> output = (__bridge id<MTLBuffer>)mtl_output_buffer;
        if (input.length < input_offset + (uint64_t)(tok_first + ntok) * hidden_size * sizeof(float) ||
            output.length < output_offset + (uint64_t)(tok_first + ntok) * hidden_size * sizeof(float)) {
            topk_set_error("mapped top-k external buffers are too small");
            return 0;
        }
        uint64_t *table = (uint64_t *)[p->_slotTable3 contents];
        for (uint32_t i = 0; i < n_expert; ++i) {
            table[i] = gate_addr0 + (uint64_t)expert_ids[i] * gate_bytes;
            table[REDMETAL_TOPK_MAX + i] = up_addr0 + (uint64_t)expert_ids[i] * up_bytes;
            table[2u * REDMETAL_TOPK_MAX + i] = down_addr0 + (uint64_t)expert_ids[i] * down_bytes;
        }
        NSMutableArray<id<MTLBuffer>> *resident = [NSMutableArray arrayWithObjects:gb, nil];
        if (![resident containsObject:ub]) [resident addObject:ub];
        if (![resident containsObject:db]) [resident addObject:db];
        if (!topk_encode_batched_dispatch(p, cb, resident, n_expert, ggml_type, hidden_size, ffn_size, tok_first, ntok, top_k, n_pairs,
                pair_token, pair_weight, expert_start, tok_pair, input, (NSUInteger)input_offset, output, (NSUInteger)output_offset)) {
            topk_set_error("failed to encode mapped top-k expert execution");
            return 0;
        }
        return 1;
    }
}

int redmetal_topk_pool_encode_batched(
        redmetal_topk_pool_t handle,
        void *mtl_command_buffer,
        const uint32_t *slot_ids,
        const uint64_t *gate_bytes,
        const uint64_t *up_bytes,
        const uint64_t *down_bytes,
        uint32_t n_expert,
        uint32_t ggml_type,
        uint32_t hidden_size,
        uint32_t ffn_size,
        uint32_t tok_first,
        uint32_t ntok,
        uint32_t top_k,
        uint32_t n_pairs,
        const uint32_t *pair_token,
        const float *pair_weight,
        const uint32_t *expert_start,
        const uint32_t *tok_pair,
        void *mtl_input_buffer,
        uint64_t input_offset,
        void *mtl_output_buffer,
        uint64_t output_offset) {
    @autoreleasepool {
        RMTopKPool *p = topk_obj(handle);
        if (!p || !mtl_command_buffer || !mtl_input_buffer || !mtl_output_buffer || !pair_token || !pair_weight || !expert_start || !tok_pair) return 0;
        if (!ntok || !top_k || !n_pairs || n_pairs != ntok * top_k || n_pairs > REDMETAL_TOPK_MAX * 64u) { topk_set_error("invalid batched top-k pair layout"); return 0; }
        /* the per-expert validation of the plain encode covers slot ids, byte layouts and sizes */
        if (!topk_validate(p, slot_ids, gate_bytes, up_bytes, down_bytes, n_expert, ggml_type, hidden_size, ffn_size, 0u, hidden_size)) return 0;
        if (expert_start[0] != 0u || expert_start[n_expert] != n_pairs) { topk_set_error("batched top-k expert_start is inconsistent"); return 0; }
        for (uint32_t i = 0; i < n_pairs; ++i) if (pair_token[i] >= tok_first + ntok) { topk_set_error("batched top-k pair token out of range"); return 0; }
        for (uint32_t i = 0; i < ntok * top_k; ++i) if (tok_pair[i] >= n_pairs) { topk_set_error("batched top-k tok_pair out of range"); return 0; }
        id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)mtl_command_buffer;
        id<MTLBuffer> input = (__bridge id<MTLBuffer>)mtl_input_buffer;
        id<MTLBuffer> output = (__bridge id<MTLBuffer>)mtl_output_buffer;
        if (input.length < input_offset + (uint64_t)(tok_first + ntok) * hidden_size * sizeof(float) ||
            output.length < output_offset + (uint64_t)(tok_first + ntok) * hidden_size * sizeof(float)) {
            topk_set_error("batched top-k external buffers are too small");
            return 0;
        }
        if (!topk_encode_batched_into(p, cb, slot_ids, gate_bytes, up_bytes, n_expert, ggml_type, hidden_size, ffn_size,
                tok_first, ntok, top_k, n_pairs, pair_token, pair_weight, expert_start, tok_pair,
                input, (NSUInteger)input_offset, output, (NSUInteger)output_offset)) {
            topk_set_error("failed to encode batched top-k expert execution");
            return 0;
        }
        return 1;
    }
}

int redmetal_topk_pool_residency_table_init(redmetal_topk_pool_t handle, uint32_t layers, uint32_t experts) {
    @autoreleasepool {
        RMTopKPool *p = topk_obj(handle);
        if (!p || !layers || !experts) return 0;
        const NSUInteger bytes = (NSUInteger)layers * experts * sizeof(uint64_t);
        p->_residency = [p->_device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (!p->_residency) { topk_set_error("failed to allocate the expert residency table"); return 0; }
        memset(p->_residency.contents, 0, bytes);
        p->_residencyLayers = layers; p->_residencyExperts = experts;
        return 1;
    }
}

void *redmetal_topk_pool_residency_table(redmetal_topk_pool_t handle) {
    RMTopKPool *p = topk_obj(handle);
    return p ? (__bridge void *)p->_residency : NULL;
}

int redmetal_topk_pool_residency_set(redmetal_topk_pool_t handle, uint32_t layer, uint32_t expert, uint32_t slot_id) {
    @autoreleasepool {
        RMTopKPool *p = topk_obj(handle);
        if (!p || !p->_residency || layer >= p->_residencyLayers || expert >= p->_residencyExperts) return 0;
        NSUInteger inner = 0;
        id<MTLBuffer> slab = [p bufferForSlot:slot_id offset:&inner];
        if (!slab) return 0;
        uint64_t base = 0;
        if (@available(macOS 13.0, *)) base = (uint64_t)[slab gpuAddress] + (uint64_t)inner;
        if (!base) return 0;
        ((uint64_t *)p->_residency.contents)[(size_t)layer * p->_residencyExperts + expert] = base;
        return 1;
    }
}

void redmetal_topk_pool_residency_clear(redmetal_topk_pool_t handle, uint32_t layer, uint32_t expert) {
    RMTopKPool *p = topk_obj(handle);
    if (!p || !p->_residency || layer >= p->_residencyLayers || expert >= p->_residencyExperts) return;
    ((uint64_t *)p->_residency.contents)[(size_t)layer * p->_residencyExperts + expert] = 0;
}

void *redmetal_topk_pool_slab_residency_set(redmetal_topk_pool_t handle) {
    RMTopKPool *p = topk_obj(handle);
    if (!p) return NULL;
    if (@available(macOS 15.0, *)) return (__bridge void *)p->_residencySet;
    return NULL;
}

void redmetal_topk_pool_use_all_slabs(redmetal_topk_pool_t handle, void *mtl_compute_encoder) {
    RMTopKPool *p = topk_obj(handle);
    if (!p || !mtl_compute_encoder) return;
    if (topk_has_residency_set(p)) return;   /* the residency set attached to the queue covers every slab */
    id<MTLComputeCommandEncoder> enc = (__bridge id<MTLComputeCommandEncoder>)mtl_compute_encoder;
    @synchronized (p) {
        for (id<MTLBuffer> slab in p->_slabs) [enc useResource:slab usage:MTLResourceUsageRead];
    }
}

int redmetal_topk_pool_encode_device(
        redmetal_topk_pool_t handle,
        void *mtl_command_buffer,
        void *slot_table_buffer, uint64_t slot_table_offset,
        void *weight_buffer, uint64_t weight_offset,
        uint32_t top_k,
        uint32_t ggml_type,
        uint32_t hidden_size,
        uint32_t ffn_size,
        uint64_t gate_bytes,
        uint64_t up_bytes,
        void *mtl_input_buffer,
        uint64_t input_offset,
        void *mtl_output_buffer,
        uint64_t output_offset) {
    @autoreleasepool {
        RMTopKPool *p = topk_obj(handle);
        if (!p || !mtl_command_buffer || !slot_table_buffer || !weight_buffer || !mtl_input_buffer || !mtl_output_buffer) return 0;
        if (!top_k || top_k > REDMETAL_TOPK_MAX || !tw_ok(ggml_type) || hidden_size % QK_IQ || ffn_size % QK_IQ) {
            topk_set_error("invalid device-driven top-k request"); return 0;
        }
        if (![p ensureScratchHidden:hidden_size ffn:ffn_size]) return 0;
        id<MTLBuffer> grid = topk_grid_for_type(p, ggml_type);
        id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)mtl_command_buffer;
        id<MTLBuffer> table = (__bridge id<MTLBuffer>)slot_table_buffer;
        id<MTLBuffer> weights = (__bridge id<MTLBuffer>)weight_buffer;
        id<MTLBuffer> input = (__bridge id<MTLBuffer>)mtl_input_buffer;
        id<MTLBuffer> output = (__bridge id<MTLBuffer>)mtl_output_buffer;
        const uint32_t gate_lanes = lanes_for_blocks(hidden_size / QK_IQ);
        const uint32_t down_lanes = lanes_for_blocks(ffn_size / QK_IQ);
        const uint64_t up_offset = gate_bytes;
        const uint64_t down_offset = gate_bytes + up_bytes;
        const uint32_t row_start = 0u;

        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (!enc) { topk_set_error("failed to create device top-k compute encoder"); return 0; }
        redmetal_topk_pool_use_all_slabs(handle, (__bridge void *)enc);
        [enc setComputePipelineState:p->_gateupPipeline];
        [enc setBuffer:table offset:(NSUInteger)slot_table_offset atIndex:0];
        [enc setBytes:&hidden_size length:sizeof(hidden_size) atIndex:1];
        [enc setBytes:&ffn_size length:sizeof(ffn_size) atIndex:2];
        [enc setBytes:&up_offset length:sizeof(up_offset) atIndex:3];
        [enc setBuffer:input offset:(NSUInteger)input_offset atIndex:4];
        [enc setBuffer:p->_actBuffer offset:0 atIndex:5];
        [enc setBuffer:grid offset:0 atIndex:6];
        [enc setBytes:&top_k length:sizeof(top_k) atIndex:7];
        [enc setBytes:&ggml_type length:sizeof(ggml_type) atIndex:8];
        [enc setBytes:&gate_lanes length:sizeof(gate_lanes) atIndex:9];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)ffn_size * gate_lanes, top_k, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [enc endEncoding];

        enc = [cb computeCommandEncoder];
        redmetal_topk_pool_use_all_slabs(handle, (__bridge void *)enc);
        [enc setComputePipelineState:p->_downPipeline];
        [enc setBuffer:table offset:(NSUInteger)slot_table_offset atIndex:0];
        [enc setBytes:&ffn_size length:sizeof(ffn_size) atIndex:1];
        [enc setBytes:&row_start length:sizeof(row_start) atIndex:2];
        [enc setBytes:&hidden_size length:sizeof(hidden_size) atIndex:3];
        [enc setBytes:&down_offset length:sizeof(down_offset) atIndex:4];
        [enc setBuffer:p->_actBuffer offset:0 atIndex:5];
        [enc setBuffer:p->_tmpBuffer offset:0 atIndex:6];
        [enc setBuffer:grid offset:0 atIndex:7];
        [enc setBytes:&top_k length:sizeof(top_k) atIndex:8];
        [enc setBytes:&ggml_type length:sizeof(ggml_type) atIndex:9];
        [enc setBytes:&down_lanes length:sizeof(down_lanes) atIndex:10];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)hidden_size * down_lanes, top_k, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [enc endEncoding];

        enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:p->_sumPipeline];
        [enc setBuffer:p->_tmpBuffer offset:0 atIndex:0];
        [enc setBuffer:weights offset:(NSUInteger)weight_offset atIndex:1];
        [enc setBuffer:output offset:(NSUInteger)output_offset atIndex:2];
        [enc setBytes:&hidden_size length:sizeof(hidden_size) atIndex:3];
        [enc setBytes:&top_k length:sizeof(top_k) atIndex:4];
        [enc dispatchThreads:MTLSizeMake(hidden_size, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [enc endEncoding];
        return 1;
    }
}

static int encode_device_into_impl(
        redmetal_topk_pool_t handle,
        void *mtl_compute_encoder,
        void *slot_table_buffer, uint64_t slot_table_offset,
        void *weight_buffer, uint64_t weight_offset,
        uint32_t top_k,
        uint32_t ggml_type,
        uint32_t hidden_size,
        uint32_t ffn_size,
        uint64_t gate_bytes,
        uint64_t up_bytes,
        void *mtl_input_buffer,
        uint64_t input_offset,
        void *mtl_output_buffer,
        uint64_t output_offset,
        void *mtl_input1_buffer,
        uint64_t input1_offset,
        uint32_t x_split) {
    @autoreleasepool {
        RMTopKPool *p = topk_obj(handle);
        if (!p || !mtl_compute_encoder || !slot_table_buffer || !weight_buffer || !mtl_input_buffer) return 0;
        if (mtl_input1_buffer && !p->_gateupG2Pipeline) { topk_set_error("two-input top-k pipeline unavailable"); return 0; }
        if (!top_k || top_k > REDMETAL_TOPK_MAX || !tw_ok(ggml_type) || hidden_size % QK_IQ || ffn_size % QK_IQ) {
            topk_set_error("invalid device-driven top-k request"); return 0;
        }
        if (![p ensureScratchHidden:hidden_size ffn:ffn_size]) return 0;
        id<MTLBuffer> grid = topk_grid_for_type(p, ggml_type);
        id<MTLBuffer> table = (__bridge id<MTLBuffer>)slot_table_buffer;
        id<MTLBuffer> weights = (__bridge id<MTLBuffer>)weight_buffer;
        id<MTLBuffer> input = (__bridge id<MTLBuffer>)mtl_input_buffer;
        id<MTLBuffer> output = mtl_output_buffer ? (__bridge id<MTLBuffer>)mtl_output_buffer : nil;
        const uint32_t gate_lanes = lanes_for_blocks(hidden_size / QK_IQ);
        const uint32_t down_lanes = lanes_for_blocks(ffn_size / QK_IQ);
        const uint64_t up_offset = gate_bytes;
        const uint64_t down_offset = gate_bytes + up_bytes;
        const uint32_t row_start = 0u;

        id<MTLComputeCommandEncoder> enc = (__bridge id<MTLComputeCommandEncoder>)mtl_compute_encoder;
        redmetal_topk_pool_use_all_slabs(handle, (__bridge void *)enc);
        [enc setComputePipelineState:mtl_input1_buffer ? p->_gateupG2Pipeline : p->_gateupGPipeline];
        if (mtl_input1_buffer) {
            [enc setBuffer:(__bridge id<MTLBuffer>)mtl_input1_buffer offset:(NSUInteger)input1_offset atIndex:11];
            [enc setBytes:&x_split length:sizeof(x_split) atIndex:12];
        }
        [enc setBuffer:table offset:(NSUInteger)slot_table_offset atIndex:0];
        [enc setBytes:&hidden_size length:sizeof(hidden_size) atIndex:1];
        [enc setBytes:&ffn_size length:sizeof(ffn_size) atIndex:2];
        [enc setBytes:&up_offset length:sizeof(up_offset) atIndex:3];
        [enc setBuffer:input offset:(NSUInteger)input_offset atIndex:4];
        [enc setBuffer:p->_actBuffer offset:0 atIndex:5];
        [enc setBuffer:grid offset:0 atIndex:6];
        [enc setBytes:&top_k length:sizeof(top_k) atIndex:7];
        [enc setBytes:&ggml_type length:sizeof(ggml_type) atIndex:8];
        [enc setBytes:&gate_lanes length:sizeof(gate_lanes) atIndex:9];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)ffn_size * gate_lanes, top_k, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)]; if (enc.dispatchType == MTLDispatchTypeConcurrent) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];   /* dev38 */

        [enc setComputePipelineState:p->_downGPipeline];
        [enc setBuffer:table offset:(NSUInteger)slot_table_offset atIndex:0];
        [enc setBytes:&ffn_size length:sizeof(ffn_size) atIndex:1];
        [enc setBytes:&row_start length:sizeof(row_start) atIndex:2];
        [enc setBytes:&hidden_size length:sizeof(hidden_size) atIndex:3];
        [enc setBytes:&down_offset length:sizeof(down_offset) atIndex:4];
        [enc setBuffer:p->_actBuffer offset:0 atIndex:5];
        [enc setBuffer:p->_tmpBuffer offset:0 atIndex:6];
        [enc setBuffer:grid offset:0 atIndex:7];
        [enc setBytes:&top_k length:sizeof(top_k) atIndex:8];
        [enc setBytes:&ggml_type length:sizeof(ggml_type) atIndex:9];
        [enc setBytes:&down_lanes length:sizeof(down_lanes) atIndex:10];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)hidden_size * down_lanes, top_k, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)]; if (enc.dispatchType == MTLDispatchTypeConcurrent) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];   /* dev38 */

        if (!output) return 1;   /* dev39: the caller sums tmp[e * hidden + r] * weights[e] itself (redmetal_topk_pool_tmp_buffer) */
        [enc setComputePipelineState:p->_sumGPipeline];
        [enc setBuffer:p->_tmpBuffer offset:0 atIndex:0];
        [enc setBuffer:weights offset:(NSUInteger)weight_offset atIndex:1];
        [enc setBuffer:output offset:(NSUInteger)output_offset atIndex:2];
        [enc setBytes:&hidden_size length:sizeof(hidden_size) atIndex:3];
        [enc setBytes:&top_k length:sizeof(top_k) atIndex:4];
        [enc dispatchThreads:MTLSizeMake(hidden_size, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)]; if (enc.dispatchType == MTLDispatchTypeConcurrent) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];   /* dev38 */
        return 1;
    }
}

int redmetal_topk_pool_encode_device_into(
        redmetal_topk_pool_t handle,
        void *mtl_compute_encoder,
        void *slot_table_buffer, uint64_t slot_table_offset,
        void *weight_buffer, uint64_t weight_offset,
        uint32_t top_k,
        uint32_t ggml_type,
        uint32_t hidden_size,
        uint32_t ffn_size,
        uint64_t gate_bytes,
        uint64_t up_bytes,
        void *mtl_input_buffer,
        uint64_t input_offset,
        void *mtl_output_buffer,
        uint64_t output_offset) {
    return encode_device_into_impl(handle, mtl_compute_encoder, slot_table_buffer, slot_table_offset, weight_buffer, weight_offset, top_k, ggml_type, hidden_size, ffn_size, gate_bytes, up_bytes, mtl_input_buffer, input_offset, mtl_output_buffer, output_offset, NULL, 0u, 0u);
}

int redmetal_topk_pool_encode_device_into2(
        redmetal_topk_pool_t handle,
        void *mtl_compute_encoder,
        void *slot_table_buffer, uint64_t slot_table_offset,
        void *weight_buffer, uint64_t weight_offset,
        uint32_t top_k,
        uint32_t ggml_type,
        uint32_t hidden_size,
        uint32_t ffn_size,
        uint64_t gate_bytes,
        uint64_t up_bytes,
        void *mtl_input_buffer,
        uint64_t input_offset,
        void *mtl_output_buffer,
        uint64_t output_offset,
        void *mtl_input1_buffer,
        uint64_t input1_offset,
        uint32_t x_split) {
    return encode_device_into_impl(handle, mtl_compute_encoder, slot_table_buffer, slot_table_offset, weight_buffer, weight_offset, top_k, ggml_type, hidden_size, ffn_size, gate_bytes, up_bytes, mtl_input_buffer, input_offset, mtl_output_buffer, output_offset, mtl_input1_buffer, input1_offset, x_split);
}

int redmetal_topk_pool_encode(
        redmetal_topk_pool_t handle,
        void *mtl_command_buffer,
        const uint32_t *slot_ids,
        const uint64_t *gate_bytes,
        const uint64_t *up_bytes,
        const uint64_t *down_bytes,
        const float *router_weights,
        uint32_t top_k,
        uint32_t ggml_type,
        uint32_t hidden_size,
        uint32_t ffn_size,
        uint32_t output_row_start,
        uint32_t output_row_count,
        void *mtl_input_buffer,
        uint64_t input_offset,
        void *mtl_output_buffer,
        uint64_t output_offset) {
    @autoreleasepool {
        RMTopKPool *p = topk_obj(handle);
        if (!p || !router_weights || !mtl_command_buffer || !mtl_input_buffer || !mtl_output_buffer) return 0;
        if (!topk_validate(p, slot_ids, gate_bytes, up_bytes, down_bytes, top_k, ggml_type, hidden_size, ffn_size,
                output_row_start, output_row_count)) return 0;
        id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)mtl_command_buffer;
        id<MTLBuffer> input = (__bridge id<MTLBuffer>)mtl_input_buffer;
        id<MTLBuffer> output = (__bridge id<MTLBuffer>)mtl_output_buffer;
        if (input.length < input_offset + (uint64_t)hidden_size * sizeof(float) ||
            output.length < output_offset + (uint64_t)output_row_count * sizeof(float)) {
            topk_set_error("top-k external buffers are too small");
            return 0;
        }
        if (!topk_encode_into(p, cb, slot_ids, gate_bytes, up_bytes, router_weights, top_k, ggml_type, hidden_size, ffn_size,
                output_row_start, output_row_count, input, (NSUInteger)input_offset, output, (NSUInteger)output_offset)) {
            topk_set_error("failed to encode top-k expert execution");
            return 0;
        }
        return 1;
    }
}

void redmetal_topk_pool_release(redmetal_topk_pool_t handle, const uint32_t *slot_ids, uint32_t top_k) {
    RMTopKPool *p = topk_obj(handle);
    if (!p || !slot_ids) return;
    mark_slots(p, slot_ids, top_k, -1);
}
