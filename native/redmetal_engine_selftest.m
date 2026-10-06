#define _FILE_OFFSET_BITS 64
#define _POSIX_C_SOURCE 200809L
#define _DARWIN_C_SOURCE 1

/*
 * Model-free self-test of the dev21-dev23 decode kernels (`redlite-engine kernel-selftest`).
 * It compiles the engine's own kernel library and runs, on synthetic weights:
 *   - the dev22 sub-block GEMV (rl_rows2_{f32,q4k,q6k,iq2xxs}) and the dev18 block kernels
 *     (rl_rows_*) through emit_rows, against the double-precision CPU row dot of
 *     redlite_native_quant_cpu.c;
 *   - the dev23 early-out guard: with the flag at buffer 30 set, rl_rows2_*, rl_copy_f32 and
 *     rl_route leave their outputs untouched;
 *   - rl_copy_f32;
 *   - rl_route (dev21 GPU router) against rl_native_router_select_softmax_topk, including an exact
 *     logit tie, the residency lookup and the miss that raises the early-out flag;
 *   - the dev26 split-K decode attention (attn_gqa_split + attn_gqa_merge) and the single-threadgroup
 *     attn_gqa, against a double-precision CPU GQA with the sigmoid output gate, across block edges.
 * No GGUF is read.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "redmetal_engine_private.h"
#include "redlite_native_iq2_xxs.h"
#include "redlite_native_quant_cpu.h"
#include "redlite_native_iq3.h"
#include "redlite_native_router_exec.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t st_rng = 0x9E3779B97F4A7C15ull;
static uint32_t st_next(void) {
    st_rng ^= st_rng << 13; st_rng ^= st_rng >> 7; st_rng ^= st_rng << 17;
    return (uint32_t)(st_rng >> 16);
}
static float st_uniform(void) { return (float)(st_next() & 0xFFFFFFu) / 16777216.0f * 2.0f - 1.0f; }

/* float -> IEEE half for the normal range used by the block scales */
static uint16_t st_half(float v) {
    uint32_t b; memcpy(&b, &v, 4);
    const uint32_t sign = (b >> 16) & 0x8000u;
    const int32_t e = (int32_t)((b >> 23) & 255u) - 127 + 15;
    return (uint16_t)(sign | ((uint32_t)e << 10) | ((b >> 13) & 0x3FFu));
}
static void st_put_half(uint8_t *p, float v) { const uint16_t h = st_half(v); p[0] = (uint8_t)(h & 255u); p[1] = (uint8_t)(h >> 8); }

static size_t st_row_bytes(uint32_t type, uint32_t cols) {
    switch (type) {
        case 0:  return (size_t)cols * 4u;
        case 8:  return (size_t)cols / 32u * 34u;
        case 12: return (size_t)cols / 256u * 144u;
        case 13: return (size_t)cols / 256u * 176u;
        case 14: return (size_t)cols / 256u * 210u;
        case 16: return (size_t)cols / 256u * 66u;
        default: return rl_iq3_row_bytes(type, cols);   /* dev31: IQ3_XXS, IQ3_S, IQ2_S, IQ4_XS */
    }
}

/* random bytes everywhere, then realistic block scales (d, dmin) so values stay finite */
static void st_fill_row(uint32_t type, uint32_t cols, uint8_t *row) {
    const size_t bytes = st_row_bytes(type, cols);
    if (type == 0u) { float *f = (float *)(void *)row; for (uint32_t i = 0; i < cols; ++i) f[i] = st_uniform(); return; }
    for (size_t i = 0; i < bytes; ++i) row[i] = (uint8_t)st_next();
    if (type == 8u) { for (uint32_t b = 0; b < cols / 32u; ++b) st_put_half(row + (size_t)b * 34u, 0.001f + 0.01f * fabsf(st_uniform())); return; }
    for (uint32_t b = 0; b < cols / 256u; ++b) {
        if (type == 12u) { uint8_t *bp = row + (size_t)b * 144u; st_put_half(bp, 0.002f + 0.01f * fabsf(st_uniform())); st_put_half(bp + 2, 0.001f + 0.01f * fabsf(st_uniform())); }
        if (type == 14u) { uint8_t *bp = row + (size_t)b * 210u; st_put_half(bp + 208, 0.001f + 0.004f * fabsf(st_uniform())); }
        if (type == 16u) { uint8_t *bp = row + (size_t)b * 66u; st_put_half(bp, 0.01f + 0.05f * fabsf(st_uniform())); }
        if (type == 13u) { uint8_t *bp = row + (size_t)b * 176u; st_put_half(bp, 0.002f + 0.01f * fabsf(st_uniform())); st_put_half(bp + 2, 0.001f + 0.01f * fabsf(st_uniform())); }
        if (rl_iq3_supported(type)) { uint8_t *bp = row + (size_t)b * rl_iq3_block_bytes(type); st_put_half(bp, 0.002f + 0.01f * fabsf(st_uniform())); }
    }
}

static int st_run(rl_metal_engine *m, void (^body)(id<MTLComputeCommandEncoder> enc), char *error, size_t cap) {
    id<MTLCommandBuffer> cb = [m->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    body(enc);
    [enc endEncoding];
    return commit_wait(cb, "kernel self-test", NULL, error, cap);
}

/* one weight shape through the block kernel and the sub-block kernel, both against the CPU row dot */
static int st_rows_case(rl_metal_engine *m, uint32_t type, uint32_t rows, uint32_t cols, double *worst, char *error, size_t cap) {
    const size_t rb = st_row_bytes(type, cols);
    id<MTLBuffer> wbuf = [m->dev newBufferWithLength:rb * rows options:MTLResourceStorageModeShared];
    id<MTLBuffer> xbuf = [m->dev newBufferWithLength:(size_t)cols * 4u options:MTLResourceStorageModeShared];
    id<MTLBuffer> out = [m->dev newBufferWithLength:(size_t)rows * 4u options:MTLResourceStorageModeShared];
    float *deq = (float *)malloc((size_t)cols * sizeof(float));
    if (!wbuf || !xbuf || !out || !deq) { free(deq); snprintf(error, cap, "self-test allocation failed"); return 0; }
    for (uint32_t r = 0; r < rows; ++r) st_fill_row(type, cols, (uint8_t *)wbuf.contents + (size_t)r * rb);
    float *x = (float *)xbuf.contents;
    for (uint32_t i = 0; i < cols; ++i) x[i] = st_uniform();
    const uint8_t *grid = (const uint8_t *)m->grid.contents;
    mweight w = { wbuf, 0, type, rows, cols };
    int ok = 1;
    for (int variant = 0; variant < 2 && ok; ++variant) {
        m->rows2 = variant;
        memset(out.contents, 0, (size_t)rows * 4u);
        ok = st_run(m, ^(id<MTLComputeCommandEncoder> enc) { [enc setBuffer:m->abort_zero offset:0 atIndex:30]; emit_rows(m, enc, &w, xbuf, out); }, error, cap);
        for (uint32_t r = 0; r < rows && ok; ++r) {
            const uint8_t *row = (const uint8_t *)wbuf.contents + (size_t)r * rb;
            double ref = 0.0, mag = 0.0;
            if (!rl_quant_row_dot(type, row, x, cols, grid, &ref) || !rl_quant_dequant_row(type, row, cols, grid, deq)) {
                snprintf(error, cap, "CPU reference failed for type %u", type); ok = 0; break;
            }
            for (uint32_t i = 0; i < cols; ++i) mag += fabs((double)deq[i] * x[i]);
            const double err = fabs((double)((float *)out.contents)[r] - ref) / (mag + 1e-12);
            if (err > *worst) *worst = err;
            if (!(err <= 2e-5)) {
                snprintf(error, cap, "%s type %u %ux%u row %u: gpu %.9g cpu %.9g (relative to sum|w*x| %.3g)",
                    variant ? "rl_rows2" : "rl_rows", type, rows, cols, r, ((float *)out.contents)[r], ref, err);
                ok = 0;
            }
        }
    }
    free(deq);
    return ok;
}

static int st_guard_and_copy(rl_metal_engine *m, char *error, size_t cap) {
    const uint32_t n = 1000u, cols = 512u, rows = 9u;
    id<MTLBuffer> src = [m->dev newBufferWithLength:n * 4u options:MTLResourceStorageModeShared];
    id<MTLBuffer> dst = [m->dev newBufferWithLength:n * 4u options:MTLResourceStorageModeShared];
    id<MTLBuffer> wbuf = [m->dev newBufferWithLength:(size_t)rows * cols * 4u options:MTLResourceStorageModeShared];
    id<MTLBuffer> out = [m->dev newBufferWithLength:rows * 4u options:MTLResourceStorageModeShared];
    if (!src || !dst || !wbuf || !out) { snprintf(error, cap, "self-test allocation failed"); return 0; }
    for (uint32_t i = 0; i < n; ++i) { ((float *)src.contents)[i] = st_uniform(); ((float *)dst.contents)[i] = -7.0f; }
    for (uint32_t i = 0; i < rows * cols; ++i) ((float *)wbuf.contents)[i] = st_uniform();
    for (uint32_t r = 0; r < rows; ++r) ((float *)out.contents)[r] = -7.0f;
    mweight w = { wbuf, 0, 0u, rows, cols };
    m->rows2 = 1;
    /* flag raised: every guarded kernel returns before writing */
    ((uint32_t *)m->abort.contents)[0] = 1u;
    if (!st_run(m, ^(id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:m->abort offset:0 atIndex:30];
            emit_rows(m, enc, &w, src, out);
            [enc setComputePipelineState:m->p_copy];
            [enc setBuffer:src offset:0 atIndex:0]; [enc setBuffer:dst offset:0 atIndex:1]; [enc setBytes:&n length:4 atIndex:2];
            enc_1d(enc, m->p_copy, n, 256u);
        }, error, cap)) return 0;
    for (uint32_t i = 0; i < n; ++i) if (((float *)dst.contents)[i] != -7.0f) { snprintf(error, cap, "rl_copy_f32 wrote with the early-out flag set"); return 0; }
    for (uint32_t r = 0; r < rows; ++r) if (((float *)out.contents)[r] != -7.0f) { snprintf(error, cap, "rl_rows2_f32 wrote with the early-out flag set"); return 0; }
    /* flag clear: the copy is exact */
    ((uint32_t *)m->abort.contents)[0] = 0u;
    if (!st_run(m, ^(id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:m->abort offset:0 atIndex:30];
            [enc setComputePipelineState:m->p_copy];
            [enc setBuffer:src offset:0 atIndex:0]; [enc setBuffer:dst offset:0 atIndex:1]; [enc setBytes:&n length:4 atIndex:2];
            enc_1d(enc, m->p_copy, n, 256u);
        }, error, cap)) return 0;
    if (memcmp(src.contents, dst.contents, n * 4u) != 0) { snprintf(error, cap, "rl_copy_f32 output differs from its input"); return 0; }
    return 1;
}

static int st_route(rl_metal_engine *m, uint32_t *cases, char *error, size_t cap) {
    const uint32_t experts = 512u, topk = 10u;
    id<MTLBuffer> logits = [m->dev newBufferWithLength:experts * 4u options:MTLResourceStorageModeShared];
    id<MTLBuffer> resident = [m->dev newBufferWithLength:experts * 8u options:MTLResourceStorageModeShared];
    id<MTLBuffer> slots = [m->dev newBufferWithLength:512u * 8u options:MTLResourceStorageModeShared];
    id<MTLBuffer> weights = [m->dev newBufferWithLength:512u * 4u options:MTLResourceStorageModeShared];
    id<MTLBuffer> ids = [m->dev newBufferWithLength:64u * 4u options:MTLResourceStorageModeShared];
    id<MTLBuffer> miss = [m->dev newBufferWithLength:16u options:MTLResourceStorageModeShared];
    if (!logits || !resident || !slots || !weights || !ids || !miss) { snprintf(error, cap, "self-test allocation failed"); return 0; }
    uint64_t *res = (uint64_t *)resident.contents;
    void (^route)(id<MTLComputeCommandEncoder>) = ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:m->abort offset:0 atIndex:30];
        [enc setComputePipelineState:m->p_route];
        [enc setBuffer:logits offset:0 atIndex:0]; [enc setBuffer:resident offset:0 atIndex:1];
        [enc setBuffer:slots offset:0 atIndex:2]; [enc setBuffer:weights offset:0 atIndex:3];
        [enc setBuffer:ids offset:0 atIndex:4]; [enc setBuffer:miss offset:0 atIndex:5];
        const float no_bias = 0.0f;
        [enc setBytes:&experts length:4 atIndex:6]; [enc setBytes:&topk length:4 atIndex:7]; [enc setBytes:&no_bias length:4 atIndex:8];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    };
    float probs[512];
    uint32_t ref_ids[64];
    float ref_w[64];
    for (uint32_t c = 0; c < 64u; ++c) {
        float *lg = (float *)logits.contents;
        for (uint32_t i = 0; i < experts; ++i) lg[i] = 3.0f * st_uniform();
        if (c % 4u == 1u) { lg[300] = 5.0f; lg[7] = 5.0f; }             /* exact tie at the top: lower id first */
        if (c % 4u == 2u) { const float v = lg[123]; lg[400] = v; }     /* exact tie anywhere */
        for (uint32_t i = 0; i < experts; ++i) res[i] = 0x10000ull * (i + 1u);
        ((uint32_t *)m->abort.contents)[0] = 0u; ((uint32_t *)miss.contents)[0] = 77u;
        if (!st_run(m, route, error, cap)) return 0;
        if (!rl_native_router_select_softmax_topk(lg, experts, topk, ref_ids, ref_w, probs, error, cap)) return 0;
        const uint32_t *gid = (const uint32_t *)ids.contents;
        const float *gw = (const float *)weights.contents;
        for (uint32_t k = 0; k < topk; ++k) {
            if (gid[k] != ref_ids[k]) { snprintf(error, cap, "rl_route case %u rank %u: id %u, CPU %u", c, k, gid[k], ref_ids[k]); return 0; }
            if (fabsf(gw[k] - ref_w[k]) > 1e-6f + 1e-5f * ref_w[k]) { snprintf(error, cap, "rl_route case %u rank %u: weight %.9g, CPU %.9g", c, k, gw[k], ref_w[k]); return 0; }
            if (((const uint64_t *)slots.contents)[k] != res[gid[k]]) { snprintf(error, cap, "rl_route case %u: slot address of rank %u is wrong", c, k); return 0; }
        }
        if (((uint32_t *)miss.contents)[0] != 0u || ((uint32_t *)m->abort.contents)[0] != 0u) { snprintf(error, cap, "rl_route case %u: miss reported with every expert resident", c); return 0; }
        /* one selected expert not resident: counted, and the early-out flag is raised */
        res[ref_ids[c % topk]] = 0u;
        if (!st_run(m, route, error, cap)) return 0;
        if (((uint32_t *)miss.contents)[0] != 1u || ((uint32_t *)m->abort.contents)[0] != 1u) { snprintf(error, cap, "rl_route case %u: a non-resident expert did not raise the early-out flag", c); return 0; }
        /* flag already raised: the kernel does nothing */
        ((uint32_t *)miss.contents)[0] = 77u;
        if (!st_run(m, route, error, cap)) return 0;
        if (((uint32_t *)miss.contents)[0] != 77u) { snprintf(error, cap, "rl_route ran with the early-out flag set"); return 0; }
        (*cases)++;
    }
    return 1;
}

/* GQA decode over a synthetic K/V cache ([pos][kv_head][dim]) against a double-precision reference */
static int st_attention(rl_metal_engine *m, uint32_t *cases, double *worst, char *error, size_t cap) {
    const uint32_t head_dim = 256u, qheads = 16u, kvheads = 2u;
    static const uint32_t lens[6] = { 1u, 200u, 256u, 257u, 1024u, 1500u };
    const uint32_t max_len = 1500u, nblocks_max = (max_len + 127u) / 128u;
    id<MTLBuffer> q = [m->dev newBufferWithLength:qheads * head_dim * 4u options:MTLResourceStorageModeShared];
    id<MTLBuffer> gate = [m->dev newBufferWithLength:qheads * head_dim * 4u options:MTLResourceStorageModeShared];
    id<MTLBuffer> kc = [m->dev newBufferWithLength:(size_t)max_len * kvheads * head_dim * 4u options:MTLResourceStorageModeShared];
    id<MTLBuffer> vc = [m->dev newBufferWithLength:(size_t)max_len * kvheads * head_dim * 4u options:MTLResourceStorageModeShared];
    id<MTLBuffer> out = [m->dev newBufferWithLength:qheads * head_dim * 4u options:MTLResourceStorageModeShared];
    id<MTLBuffer> ml = [m->dev newBufferWithLength:(size_t)qheads * nblocks_max * 8u options:MTLResourceStorageModeShared];
    id<MTLBuffer> acc = [m->dev newBufferWithLength:(size_t)qheads * nblocks_max * head_dim * 4u options:MTLResourceStorageModeShared];
    double *ref = (double *)malloc((size_t)qheads * head_dim * sizeof(double));
    double *sc = (double *)malloc((size_t)max_len * sizeof(double));
    if (!q || !gate || !kc || !vc || !out || !ml || !acc || !ref || !sc) { free(ref); free(sc); snprintf(error, cap, "self-test allocation failed"); return 0; }
    float *qf = (float *)q.contents, *gf = (float *)gate.contents, *kf = (float *)kc.contents, *vf = (float *)vc.contents;
    for (uint32_t i = 0; i < qheads * head_dim; ++i) { qf[i] = 0.5f * st_uniform(); gf[i] = 2.0f * st_uniform(); }
    for (size_t i = 0; i < (size_t)max_len * kvheads * head_dim; ++i) { kf[i] = 0.5f * st_uniform(); vf[i] = st_uniform(); }
    int ok = 1;
    for (uint32_t c = 0; c < 6u && ok; ++c) {
        const uint32_t seq_len = lens[c];
        for (uint32_t h = 0; h < qheads; ++h) {
            const uint32_t kvh = h / (qheads / kvheads);
            double mx = -INFINITY, l = 0.0;
            for (uint32_t p = 0; p < seq_len; ++p) {
                double d = 0.0;
                for (uint32_t i = 0; i < head_dim; ++i) d += (double)qf[h * head_dim + i] * kf[((size_t)p * kvheads + kvh) * head_dim + i];
                sc[p] = d / sqrt((double)head_dim); if (sc[p] > mx) mx = sc[p];
            }
            for (uint32_t p = 0; p < seq_len; ++p) { sc[p] = exp(sc[p] - mx); l += sc[p]; }
            for (uint32_t i = 0; i < head_dim; ++i) {
                double a = 0.0;
                for (uint32_t p = 0; p < seq_len; ++p) a += sc[p] * vf[((size_t)p * kvheads + kvh) * head_dim + i];
                ref[h * head_dim + i] = (a / l) / (1.0 + exp(-(double)gf[h * head_dim + i]));
            }
        }
        for (int variant = 0; variant < 4 && ok; ++variant) {   /* 0 single group, 1 split-K per head, 2/3 split-K per KV head (dev35), blocks of 128/256 (dev66) */
            memset(out.contents, 0, qheads * head_dim * 4u);
            ok = st_run(m, ^(id<MTLComputeCommandEncoder> enc) {
                [enc setBuffer:m->abort_zero offset:0 atIndex:30];
                if (variant) {
                    [enc setComputePipelineState:variant >= 2 ? m->p_attn_split_g : m->p_attn_split];
                    [enc setBuffer:q offset:0 atIndex:0]; [enc setBuffer:kc offset:0 atIndex:1]; [enc setBuffer:vc offset:0 atIndex:2];
                    [enc setBuffer:ml offset:0 atIndex:3]; [enc setBuffer:acc offset:0 atIndex:4];
                    const uint32_t blk = variant == 2 ? 128u : 256u, nb = (seq_len + blk - 1u) / blk;
                    [enc setBytes:&head_dim length:4 atIndex:5]; [enc setBytes:&qheads length:4 atIndex:6]; [enc setBytes:&kvheads length:4 atIndex:7]; [enc setBytes:&seq_len length:4 atIndex:8];
                    [enc setBytes:&blk length:4 atIndex:9];
                    [enc dispatchThreadgroups:MTLSizeMake(nb, variant >= 2 ? kvheads : qheads, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    [enc setComputePipelineState:m->p_attn_merge];
                    [enc setBuffer:ml offset:0 atIndex:0]; [enc setBuffer:acc offset:0 atIndex:1]; [enc setBuffer:gate offset:0 atIndex:2]; [enc setBuffer:out offset:0 atIndex:3];
                    [enc setBytes:&head_dim length:4 atIndex:4]; [enc setBytes:&seq_len length:4 atIndex:5]; [enc setBytes:&blk length:4 atIndex:6];
                    [enc dispatchThreadgroups:MTLSizeMake(qheads, 1, 1) threadsPerThreadgroup:MTLSizeMake(head_dim, 1, 1)];
                } else {
                    [enc setComputePipelineState:m->p_attn_gqa];
                    [enc setBuffer:q offset:0 atIndex:0]; [enc setBuffer:kc offset:0 atIndex:1]; [enc setBuffer:vc offset:0 atIndex:2];
                    [enc setBuffer:gate offset:0 atIndex:3]; [enc setBuffer:out offset:0 atIndex:4];
                    [enc setBytes:&head_dim length:4 atIndex:5]; [enc setBytes:&qheads length:4 atIndex:6]; [enc setBytes:&kvheads length:4 atIndex:7]; [enc setBytes:&seq_len length:4 atIndex:8];
                    [enc dispatchThreadgroups:MTLSizeMake(qheads, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                }
            }, error, cap);
            for (uint32_t i = 0; i < qheads * head_dim && ok; ++i) {
                const double err = fabs((double)((float *)out.contents)[i] - ref[i]);
                if (err > *worst) *worst = err;
                if (!(err <= 2e-6)) {
                    snprintf(error, cap, "%s seq_len %u out[%u]: gpu %.9g cpu %.9g", variant >= 2 ? "attn_gqa_split_g/merge" : variant ? "attn_gqa_split/merge" : "attn_gqa", seq_len, i, ((float *)out.contents)[i], ref[i]);
                    ok = 0;
                }
            }
        }
        if (ok) (*cases)++;
    }
    free(ref); free(sc);
    return ok;
}

/* dev33 (development): effective bandwidth of the decode GEMV per weight type at the shapes of the two GGUFs.
 * 16 copies of each matrix (beyond the system cache), 256 dispatches per command buffer, best of 5 after a warm-up. */
int rl_metal_kernel_bench(char *report, size_t report_cap, char *error, size_t cap) {
    @autoreleasepool {
        rl_metal_engine *m = (rl_metal_engine *)calloc(1, sizeof(*m));
        if (!m) { snprintf(error, cap, "bench allocation failed"); return 0; }
        int ok = 0;
        uint8_t grid[RL_IQ2_XXS_GRID_COUNT];
        m->dev = MTLCreateSystemDefaultDevice();
        m->queue = m->dev ? [m->dev newCommandQueue] : nil;
        NSError *le = nil;
        m->lib = m->dev ? rl_metal_engine_library(m->dev, &le) : nil;
        if (!m->queue || !m->lib) { snprintf(error, cap, "Metal engine library compile failed"); goto bout; }
        {
            struct { __strong id<MTLComputePipelineState> *slot; NSString *name; } pipes[] = {
                {&m->p_rows_f32, @"rl_rows_f32"}, {&m->p_rows_q8, @"rl_rows_q8"}, {&m->p_rows_q4k, @"rl_rows_q4k"}, {&m->p_rows_q5k, @"rl_rows_q5k"},
                {&m->p_rows_q6k, @"rl_rows_q6k"}, {&m->p_rows_iq2xxs, @"rl_rows_iq2xxs"}, {&m->p_rows_iq3, @"rl_rows_iq3"},
                {&m->p_rows2_f32, @"rl_rows2_f32"}, {&m->p_rows2_q4k, @"rl_rows2_q4k"}, {&m->p_rows2_q6k, @"rl_rows2_q6k"}, {&m->p_rows2_iq2xxs, @"rl_rows2_iq2xxs"},
            };
            for (size_t i = 0; i < sizeof(pipes) / sizeof(pipes[0]); ++i) {
                *pipes[i].slot = make_pipe(m->dev, m->lib, pipes[i].name, error, cap);
                if (!*pipes[i].slot) goto bout;
            }
        }
        if (!rl_native_iq2_xxs_build_grid(grid, error, cap)) goto bout;
        m->grid = [m->dev newBufferWithBytes:grid length:sizeof(grid) options:MTLResourceStorageModeShared];
        m->abort_zero = [m->dev newBufferWithLength:16u options:MTLResourceStorageModeShared];
        memset(m->abort_zero.contents, 0, 16u);
        m->rows2 = 1;
        {
            static const struct { const char *what; uint32_t type, rows, cols; } cases[] = {
                {"attn_qkv IQ2_XXS", 16u, 8192u, 2048u}, {"attn_qkv IQ3_XXS", 18u, 8192u, 2048u},
                {"attn_q   IQ2_S  ", 22u, 8192u, 2048u}, {"ssm_out  Q4_K   ", 12u, 2048u, 4096u},
                {"ssm_out  Q8_0   ", 8u, 2048u, 4096u},  {"attn_out Q6_K   ", 14u, 2048u, 4096u},
                {"shexp up Q8_0   ", 8u, 512u, 2048u},   {"shexp up IQ4_XS ", 23u, 512u, 2048u},
                {"shexp dn IQ4_XS ", 23u, 2048u, 512u},  {"shexp dn Q8_0   ", 8u, 2048u, 512u},
                {"output   Q5_K   ", 13u, 151936u, 2048u},
            };
            size_t off = 0;
            for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); ++c) {
                const uint32_t type = cases[c].type, rows = cases[c].rows, cols = cases[c].cols;
                const size_t rb = st_row_bytes(type, cols), mb = rb * rows;
                const uint32_t copies = mb * 16u > (size_t)1u << 31 ? 2u : 16u;
                id<MTLBuffer> w = [m->dev newBufferWithLength:mb * copies options:MTLResourceStorageModeShared];
                id<MTLBuffer> x = [m->dev newBufferWithLength:(size_t)cols * 4u options:MTLResourceStorageModeShared];
                id<MTLBuffer> out = [m->dev newBufferWithLength:(size_t)rows * 4u options:MTLResourceStorageModeShared];
                if (!w || !x || !out) { snprintf(error, cap, "bench allocation failed"); goto bout; }
                for (uint32_t r = 0; r < rows * copies; ++r) st_fill_row(type, cols, (uint8_t *)w.contents + (size_t)r * rb);
                for (uint32_t i = 0; i < cols; ++i) ((float *)x.contents)[i] = st_uniform();
                double best = 1e30;
                for (int rep = 0; rep < 6; ++rep) {   /* the first pass only ramps the GPU clock */
                    id<MTLCommandBuffer> cb = [m->queue commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    [enc setBuffer:m->abort_zero offset:0 atIndex:30];
                    for (uint32_t k = 0; k < 256u; ++k) {
                        mweight mw = { w, (NSUInteger)(mb * (k % copies)), type, rows, cols };
                        emit_rows(m, enc, &mw, x, out);
                    }
                    [enc endEncoding];
                    [cb commit]; [cb waitUntilCompleted];
                    const double ms = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0 / 256.0;
                    if (rep > 0 && ms < best) best = ms;
                }
                off += (size_t)snprintf(report + off, off < report_cap ? report_cap - off : 0, "%s %6u x %5u  %7.1f us  %6.1f GB/s\n",
                    cases[c].what, rows, cols, best * 1000.0, (double)mb / (best * 1e6));
                if (type == 16u) {   /* dev45: the same matrix over two activation vectors (MTP verify rows) */
                    id<MTLComputePipelineState> p2 = make_pipe(m->dev, m->lib, @"rl_rows2r2_iq2xxs", error, cap);
                    id<MTLBuffer> x1 = [m->dev newBufferWithLength:(size_t)cols * 4u options:MTLResourceStorageModeShared];
                    id<MTLBuffer> out1 = [m->dev newBufferWithLength:(size_t)rows * 4u options:MTLResourceStorageModeShared];
                    if (!p2 || !x1 || !out1) goto bout;
                    for (uint32_t i = 0; i < cols; ++i) ((float *)x1.contents)[i] = st_uniform();
                    uint32_t lanes = 1u; while (lanes < (cols / 256u) * 8u && lanes < 32u) lanes <<= 1;
                    double b2 = 1e30;
                    for (int rep = 0; rep < 6; ++rep) {
                        id<MTLCommandBuffer> cb = [m->queue commandBuffer];
                        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                        [enc setBuffer:m->abort_zero offset:0 atIndex:30];
                        [enc setComputePipelineState:p2];
                        for (uint32_t k = 0; k < 256u; ++k) {
                            [enc setBuffer:w offset:(NSUInteger)(mb * (k % copies)) atIndex:0]; [enc setBytes:&cols length:4 atIndex:1];
                            [enc setBuffer:x offset:0 atIndex:2]; [enc setBuffer:out offset:0 atIndex:3]; [enc setBytes:&rows length:4 atIndex:4];
                            [enc setBytes:&lanes length:4 atIndex:5]; [enc setBuffer:m->grid offset:0 atIndex:6];
                            [enc setBuffer:x1 offset:0 atIndex:8]; [enc setBuffer:out1 offset:0 atIndex:9];
                            [enc dispatchThreads:MTLSizeMake((((NSUInteger)rows * lanes) + 31u) & ~(NSUInteger)31u, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                        }
                        [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
                        const double ms = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0 / 256.0;
                        if (rep > 0 && ms < b2) b2 = ms;
                    }
                    off += (size_t)snprintf(report + off, off < report_cap ? report_cap - off : 0, "%s two x vectors   %7.1f us (%.2fx one vector)\n",
                        cases[c].what, b2 * 1000.0, b2 / best);
                }
                if (off >= report_cap) break;
            }
            /* dev65: decode attention against context length, as the engine dispatches it for one token
             * (12 attention layers; Qwen3-Next: 16 query heads, 2 KV heads, head_dim 256; RL_ENGINE_ATTN_BLK sweeps the block) */
            {
                id<MTLComputePipelineState> pg = make_pipe(m->dev, m->lib, @"attn_gqa_split_g", error, cap);
                id<MTLComputePipelineState> pm = make_pipe(m->dev, m->lib, @"attn_gqa_merge", error, cap);
                if (!pg || !pm) goto bout;
                const char *ab = getenv("RL_ENGINE_ATTN_BLK");
                const uint32_t blk_env = ab && atoi(ab) >= 32 && atoi(ab) <= 256 ? (uint32_t)atoi(ab) : 0u;
                const uint32_t hd = 256u, qh = 16u, kvh = 2u, layers = 12u;
                static const uint32_t lens[] = { 1024u, 2048u, 4096u, 8192u, 16384u, 32768u, 65536u, 131072u, 262144u };
                const uint32_t maxlen = lens[sizeof(lens) / sizeof(lens[0]) - 1u];
                const size_t kvb = (size_t)maxlen * kvh * hd * 4u;
                /* RL_BENCH_ATTN_DISTINCT=1: one K and one V buffer per layer, as the engine allocates them (12 GiB at
                 * 256K positions), instead of one pair read by every layer */
                const char *dv = getenv("RL_BENCH_ATTN_DISTINCT");
                const uint32_t nbuf = dv && atoi(dv) != 0 ? layers : 1u;
                id<MTLBuffer> kcs[12], vcs[12];
                for (uint32_t bi = 0; bi < nbuf; ++bi) {
                    kcs[bi] = [m->dev newBufferWithLength:kvb options:MTLResourceStorageModeShared];
                    vcs[bi] = [m->dev newBufferWithLength:kvb options:MTLResourceStorageModeShared];
                    if (!kcs[bi] || !vcs[bi]) { snprintf(error, cap, "attention bench KV allocation failed"); goto bout; }
                    for (size_t i = 0; i < kvb / 4u; ++i) { ((float *)kcs[bi].contents)[i] = st_uniform(); ((float *)vcs[bi].contents)[i] = st_uniform(); }
                }
                id<MTLBuffer> kc = kcs[0], vc = vcs[0];
                id<MTLBuffer> q = [m->dev newBufferWithLength:(size_t)qh * hd * 4u options:MTLResourceStorageModeShared];
                id<MTLBuffer> g = [m->dev newBufferWithLength:(size_t)qh * hd * 4u options:MTLResourceStorageModeShared];
                id<MTLBuffer> o = [m->dev newBufferWithLength:(size_t)qh * hd * 4u options:MTLResourceStorageModeShared];
                const size_t nbmax = (maxlen + 31u) / 32u;
                id<MTLBuffer> ml = [m->dev newBufferWithLength:(size_t)qh * nbmax * 8u options:MTLResourceStorageModeShared];
                id<MTLBuffer> acc = [m->dev newBufferWithLength:(size_t)qh * nbmax * hd * 4u options:MTLResourceStorageModeShared];
                if (!kc || !vc || !q || !g || !o || !ml || !acc) { snprintf(error, cap, "attention bench allocation failed"); goto bout; }
                for (uint32_t i = 0; i < qh * hd; ++i) { ((float *)q.contents)[i] = st_uniform(); ((float *)g.contents)[i] = st_uniform(); }
                for (size_t li = 0; li < sizeof(lens) / sizeof(lens[0]); ++li) {
                    const uint32_t seq = lens[li], blk = blk_env ? blk_env : rl_attn_auto_blk(seq), nb = (seq + blk - 1u) / blk;
                    double best = 1e30;
                    for (int rep = 0; rep < 5; ++rep) {
                        id<MTLCommandBuffer> cb = [m->queue commandBuffer];
                        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                        [enc setBuffer:m->abort_zero offset:0 atIndex:30];
                        for (uint32_t l = 0; l < layers; ++l) {
                            [enc setComputePipelineState:pg];
                            [enc setBuffer:q offset:0 atIndex:0]; [enc setBuffer:kcs[l % nbuf] offset:0 atIndex:1]; [enc setBuffer:vcs[l % nbuf] offset:0 atIndex:2];
                            [enc setBuffer:ml offset:0 atIndex:3]; [enc setBuffer:acc offset:0 atIndex:4];
                            [enc setBytes:&hd length:4 atIndex:5]; [enc setBytes:&qh length:4 atIndex:6]; [enc setBytes:&kvh length:4 atIndex:7];
                            [enc setBytes:&seq length:4 atIndex:8]; [enc setBytes:&blk length:4 atIndex:9];
                            [enc dispatchThreadgroups:MTLSizeMake(nb, kvh, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                            [enc setComputePipelineState:pm];
                            [enc setBuffer:ml offset:0 atIndex:0]; [enc setBuffer:acc offset:0 atIndex:1];
                            [enc setBuffer:g offset:0 atIndex:2]; [enc setBuffer:o offset:0 atIndex:3];
                            [enc setBytes:&hd length:4 atIndex:4]; [enc setBytes:&seq length:4 atIndex:5]; [enc setBytes:&blk length:4 atIndex:6];
                            [enc dispatchThreadgroups:MTLSizeMake(qh, 1, 1) threadsPerThreadgroup:MTLSizeMake(hd, 1, 1)];
                        }
                        [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
                        const double ms = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
                        if (rep > 0 && ms < best) best = ms;
                    }
                    const double bytes = (double)seq * kvh * hd * 4.0 * 2.0 * layers;
                    off += (size_t)snprintf(report + off, off < report_cap ? report_cap - off : 0,
                        "attention x%u layers, %6u positions, blk %3u, %2u KV pairs  %8.2f ms/token  %6.1f GB/s\n", layers, seq, blk, nbuf, best, bytes / (best * 1e6));
                }
            }
        }
        ok = 1;
    bout:
        m->p_rows_f32 = m->p_rows_q8 = m->p_rows_q4k = m->p_rows_q5k = m->p_rows_q6k = m->p_rows_iq2xxs = m->p_rows_iq3 = nil;
        m->p_rows2_f32 = m->p_rows2_q4k = m->p_rows2_q6k = m->p_rows2_iq2xxs = nil;
        m->grid = nil; m->abort_zero = nil; m->lib = nil; m->queue = nil; m->dev = nil;
        free(m);
        return ok;
    }
}

int rl_metal_kernel_selftest(char *report, size_t report_cap, char *error, size_t cap) {
    @autoreleasepool {
        rl_metal_engine *m = (rl_metal_engine *)calloc(1, sizeof(*m));
        if (!m) { snprintf(error, cap, "self-test allocation failed"); return 0; }
        int ok = 0;
        uint8_t grid[RL_IQ2_XXS_GRID_COUNT];
        m->dev = MTLCreateSystemDefaultDevice();
        m->queue = m->dev ? [m->dev newCommandQueue] : nil;
        NSError *le = nil;
        m->lib = m->dev ? rl_metal_engine_library(m->dev, &le) : nil;
        if (!m->queue || !m->lib) { snprintf(error, cap, "Metal engine library compile failed: %s", le.localizedDescription.UTF8String ?: "no device"); goto out; }
        {
            struct { __strong id<MTLComputePipelineState> *slot; NSString *name; } pipes[] = {
                {&m->p_rows_f32, @"rl_rows_f32"}, {&m->p_rows_q4k, @"rl_rows_q4k"}, {&m->p_rows_q6k, @"rl_rows_q6k"}, {&m->p_rows_iq2xxs, @"rl_rows_iq2xxs"}, {&m->p_rows_iq3, @"rl_rows_iq3"},
                {&m->p_rows2_f32, @"rl_rows2_f32"}, {&m->p_rows2_q4k, @"rl_rows2_q4k"}, {&m->p_rows2_q6k, @"rl_rows2_q6k"}, {&m->p_rows2_iq2xxs, @"rl_rows2_iq2xxs"},
                {&m->p_route, @"rl_route"}, {&m->p_copy, @"rl_copy_f32"},
                {&m->p_attn_gqa, @"attn_gqa"}, {&m->p_attn_split, @"attn_gqa_split"}, {&m->p_attn_merge, @"attn_gqa_merge"}, {&m->p_attn_split_g, @"attn_gqa_split_g"},
            };
            for (size_t i = 0; i < sizeof(pipes) / sizeof(pipes[0]); ++i) {
                *pipes[i].slot = make_pipe(m->dev, m->lib, pipes[i].name, error, cap);
                if (!*pipes[i].slot) goto out;
            }
        }
        if (!rl_native_iq2_xxs_build_grid(grid, error, cap)) goto out;
        m->grid = [m->dev newBufferWithBytes:grid length:sizeof(grid) options:MTLResourceStorageModeShared];
        m->abort = [m->dev newBufferWithLength:16u options:MTLResourceStorageModeShared];
        m->abort_zero = [m->dev newBufferWithLength:16u options:MTLResourceStorageModeShared];
        if (!m->grid || !m->abort || !m->abort_zero) { snprintf(error, cap, "self-test allocation failed"); goto out; }
        memset(m->abort.contents, 0, 16u); memset(m->abort_zero.contents, 0, 16u);
        {
            static const uint32_t types[8] = { 0u, 12u, 14u, 16u, 18u, 21u, 22u, 23u };
            static const uint32_t cols_q[4] = { 256u, 768u, 2048u, 4096u };
            static const uint32_t cols_f[4] = { 64u, 516u, 2048u, 4096u };
            double worst[8] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
            uint32_t shapes = 0, route_cases = 0, attn_cases = 0;
            double attn_worst = 0.0;
            for (uint32_t t = 0; t < 8u; ++t)
                for (uint32_t c = 0; c < 4u; ++c) {
                    if (!st_rows_case(m, types[t], 37u, types[t] == 0u ? cols_f[c] : cols_q[c], &worst[t], error, cap)) goto out;
                    shapes++;
                }
            if (!st_guard_and_copy(m, error, cap)) goto out;
            if (!st_route(m, &route_cases, error, cap)) goto out;
            if (!st_attention(m, &attn_cases, &attn_worst, error, cap)) goto out;
            snprintf(report, report_cap,
                "rows/rows2 vs CPU row dot: %u shapes x 2 kernels OK (worst relative error F32 %.2e Q4_K %.2e Q6_K %.2e IQ2_XXS %.2e\n"
                "                           IQ3_XXS %.2e IQ3_S %.2e IQ2_S %.2e IQ4_XS %.2e)\n"
                "early-out guard       : OK (rl_rows2, rl_copy_f32, rl_route return with the flag set)\n"
                "rl_copy_f32           : OK\n"
                "rl_route vs CPU router: %u cases OK (ids, weights, slots, ties, miss -> early-out flag)\n"
                "decode attention      : %u lengths x 4 kernels OK vs double GQA (split-K + merge, single group; worst abs %.2e)",
                shapes, worst[0], worst[1], worst[2], worst[3], worst[4], worst[5], worst[6], worst[7], route_cases, attn_cases, attn_worst);
        }
        ok = 1;
    out:
        m->p_rows_f32 = m->p_rows_q4k = m->p_rows_q6k = m->p_rows_iq2xxs = nil;
        m->p_rows2_f32 = m->p_rows2_q4k = m->p_rows2_q6k = m->p_rows2_iq2xxs = m->p_route = m->p_copy = nil;
        m->p_attn_gqa = m->p_attn_split = m->p_attn_merge = m->p_attn_split_g = nil;
        m->grid = m->abort = m->abort_zero = nil; m->lib = nil; m->queue = nil; m->dev = nil;
        free(m);
        return ok;
    }
}
