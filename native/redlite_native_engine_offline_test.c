#define _POSIX_C_SOURCE 200809L

/*
 * Model-free and (optionally) real-GGUF checks for the engine foundations:
 *   - scalar quant arithmetic self-test (Q8_0/Q2_K/Q4_K/Q5_K/Q6_K/IQ2_XXS)
 *   - full GGUF directory/metadata parser on a synthetic v3 fixture
 *   - with MODEL: hyper-parameters, tokenizer counts and dequantized row digests
 *     that can be cross-checked against gguf-py (scripts/dev/quant_rows_reference.py)
 */

#include "redlite_native_gguf_dir.h"
#include "redlite_native_iq2_xxs.h"
#include "redlite_native_quant_cpu.h"
#include "redlite_native_sampler.h"

#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void put_u32(FILE *f, uint32_t v) { uint8_t b[4] = {v & 255u, (v >> 8) & 255u, (v >> 16) & 255u, (v >> 24) & 255u}; fwrite(b, 1, 4, f); }
static void put_u64(FILE *f, uint64_t v) { uint8_t b[8]; for (int i = 0; i < 8; ++i) b[i] = (uint8_t)(v >> (8 * i)); fwrite(b, 1, 8, f); }
static void put_str(FILE *f, const char *s) { put_u64(f, strlen(s)); fwrite(s, 1, strlen(s), f); }
static void put_kv_u32(FILE *f, const char *k, uint32_t v) { put_str(f, k); put_u32(f, 4u); put_u32(f, v); }
static void put_kv_f32(FILE *f, const char *k, float v) { uint32_t b; memcpy(&b, &v, 4); put_str(f, k); put_u32(f, 6u); put_u32(f, b); }
static void put_kv_str(FILE *f, const char *k, const char *v) { put_str(f, k); put_u32(f, 8u); put_str(f, v); }

static int write_fixture(const char *path, uint32_t token_type_count);

static int synthetic_gguf_test(char *error, size_t cap) {
    char path[512];
    snprintf(path, sizeof(path), "/tmp/redlite_engine_fixture_%ld.gguf", (long)getpid());
    if (!write_fixture(path, 3u)) { snprintf(error, cap, "fixture open failed"); return 0; }

    rl_gguf_model m;
    int ok = rl_gguf_model_open(path, &m, error, cap);
    if (ok) {
        const rl_gguf_tensor *k = rl_gguf_find_layer(&m, 0u, "attn_k.weight");
        const rl_gguf_tensor *e = rl_gguf_find(&m, "token_embd.weight");
        const rl_gguf_tensor *n = rl_gguf_find(&m, "output_norm.weight");
        if (!k || !e || !n) { snprintf(error, cap, "fixture tensor lookup failed"); ok = 0; }
        else if (m.n_layer != 2u || m.n_embd != 256u || fabsf(m.rms_eps - 1e-6f) > 1e-12f || m.vocab_count != 3u ||
                 m.merge_count != 1u || m.eos_id != 2 || !m.token_types || m.token_types[2] != 1 ||
                 strcmp(m.tokens[2], "ab") != 0 || strcmp(m.merges[0], "a b") != 0 ||
                 strcmp(m.architecture, "qwen3next") != 0) {
            snprintf(error, cap, "fixture metadata mismatch"); ok = 0;
        } else if (k->payload_bytes != 288u || k->span_bytes != 288u || e->payload_bytes != 84u || n->payload_bytes != 1024u ||
                   k->offset != m.data_base + 1024u || e->offset != m.data_base + 1312u) {
            snprintf(error, cap, "fixture directory geometry mismatch"); ok = 0;
        } else if (!rl_gguf_model_map(&m, error, cap)) {
            ok = 0;
        } else {
            const uint8_t *p = rl_gguf_tensor_data(&m, k);
            if (!p || p[0] != (uint8_t)(1024u & 255u) || p[1] != (uint8_t)(1025u & 255u)) { snprintf(error, cap, "mmap payload mismatch"); ok = 0; }
        }
        rl_gguf_model_close(&m);
    }
    remove(path);
    if (!ok) return 0;
    /* a token_type array whose length differs from the vocabulary must be rejected, as in llama.cpp */
    if (!write_fixture(path, 2u)) { snprintf(error, cap, "fixture open failed"); return 0; }
    char scratch[256];
    ok = !rl_gguf_model_open(path, &m, scratch, sizeof(scratch));
    if (!ok) { rl_gguf_model_close(&m); snprintf(error, cap, "short token_type array was accepted"); }
    remove(path);
    return ok;
}

static int write_fixture(const char *path, uint32_t token_type_count) {
    FILE *f = fopen(path, "wb");
    if (!f) return 0;
    fwrite("GGUF", 1, 4, f);
    put_u32(f, 3u);
    put_u64(f, 3u);   /* tensors */
    put_u64(f, 9u);   /* kv */
    put_kv_str(f, "general.architecture", "qwen3next");
    put_kv_u32(f, "general.alignment", 32u);
    put_kv_u32(f, "qwen3next.block_count", 2u);
    put_kv_u32(f, "qwen3next.embedding_length", 256u);
    put_kv_f32(f, "qwen3next.attention.layer_norm_rms_epsilon", 1e-6f);
    put_str(f, "tokenizer.ggml.tokens"); put_u32(f, 9u); put_u32(f, 8u); put_u64(f, 3u); put_str(f, "a"); put_str(f, "b"); put_str(f, "ab");
    put_str(f, "tokenizer.ggml.token_type"); put_u32(f, 9u); put_u32(f, 5u); put_u64(f, token_type_count);
    for (uint32_t i = 0; i < token_type_count; ++i) put_u32(f, 1u);
    put_str(f, "tokenizer.ggml.merges"); put_u32(f, 9u); put_u32(f, 8u); put_u64(f, 1u); put_str(f, "a b");
    put_str(f, "tokenizer.ggml.eos_token_id"); put_u32(f, 4u); put_u32(f, 2u);
    /* tensor directory: F32[256], Q4_K[256,2], Q2_K[256] */
    put_str(f, "output_norm.weight"); put_u32(f, 1u); put_u64(f, 256u); put_u32(f, 0u); put_u64(f, 0u);
    put_str(f, "blk.0.attn_k.weight"); put_u32(f, 2u); put_u64(f, 256u); put_u64(f, 2u); put_u32(f, 12u); put_u64(f, 1024u);
    put_str(f, "token_embd.weight"); put_u32(f, 1u); put_u64(f, 256u); put_u32(f, 10u); put_u64(f, 1024u + 288u);
    long pos = ftell(f);
    while (pos % 32) { fputc(0, f); ++pos; }
    for (uint32_t i = 0; i < 1024u + 288u + 84u; ++i) fputc((int)(i & 255u), f);
    fclose(f);
    return 1;
}

/* Model-free sampler checks: greedy is the argmax, top-k=1 is the argmax at any temperature,
 * every draw stays inside the top-k / top-p candidate set, and the partial top-k selection
 * matches a full sort. */
static int sampler_selftest(char *error, size_t cap) {
    enum { V = 4096 };
    float logits[V];
    uint64_t st = 12345u;
    for (uint32_t i = 0; i < V; ++i) {
        st = st * 6364136223846793005ull + 1442695040888963407ull;
        logits[i] = (float)((double)(st >> 11) / 9007199254740992.0) * 12.0f - 6.0f;
    }
    logits[777] = 20.0f;  /* unique argmax; with logits[4000] it carries >99.9% of the softmax mass */
    logits[4000] = 19.5f;
    rl_sampler_params p;
    rl_sampler_params_default(&p);
    rl_sampler s;
    if (!rl_sampler_init(&s, &p, V)) { snprintf(error, cap, "sampler init failed"); return 0; }
    int ok = rl_sampler_sample(&s, logits) == 777u;
    if (!ok) { snprintf(error, cap, "greedy sample is not the argmax"); rl_sampler_free(&s); return 0; }
    rl_sampler_free(&s);

    p.temperature = 1.5f; p.top_k = 1u; p.top_p = 1.0f; p.seed = 7u;
    if (!rl_sampler_init(&s, &p, V)) { snprintf(error, cap, "sampler init failed"); return 0; }
    for (int i = 0; i < 64 && ok; ++i) ok = rl_sampler_sample(&s, logits) == 777u;
    rl_sampler_free(&s);
    if (!ok) { snprintf(error, cap, "top-k=1 did not return the argmax"); return 0; }

    /* top-k=8: the partial selection must equal the 8 largest logits from a full sort */
    p.top_k = 8u; p.top_p = 1.0f; p.temperature = 0.7f;
    if (!rl_sampler_init(&s, &p, V)) { snprintf(error, cap, "sampler init failed"); return 0; }
    (void)rl_sampler_sample(&s, logits);
    uint32_t expect[8];
    for (uint32_t j = 0; j < 8u; ++j) {
        uint32_t best = UINT32_MAX;
        for (uint32_t i = 0; i < V; ++i) {
            int used = 0;
            for (uint32_t q = 0; q < j; ++q) if (expect[q] == i) used = 1;
            if (used) continue;
            if (best == UINT32_MAX || logits[i] > logits[best]) best = i;
        }
        expect[j] = best;
    }
    for (uint32_t j = 0; j < 8u && ok; ++j) ok = s.index[j] == expect[j];
    if (!ok) { snprintf(error, cap, "partial top-k selection differs from a full sort"); rl_sampler_free(&s); return 0; }
    for (int i = 0; i < 256 && ok; ++i) {
        const uint32_t id = rl_sampler_sample(&s, logits);
        int inside = 0;
        for (uint32_t j = 0; j < 8u; ++j) if (expect[j] == id) inside = 1;
        ok = inside;
    }
    rl_sampler_free(&s);
    if (!ok) { snprintf(error, cap, "top-k sample left the candidate set"); return 0; }

    /* top-p=0.9: the nucleus is exactly the two dominant ids (0.62 after 777, 0.9998 after 4000) */
    p.top_k = 0u; p.top_p = 0.9f; p.temperature = 1.0f;
    if (!rl_sampler_init(&s, &p, V)) { snprintf(error, cap, "sampler init failed"); return 0; }
    for (int i = 0; i < 256 && ok; ++i) { const uint32_t id = rl_sampler_sample(&s, logits); ok = id == 777u || id == 4000u; }
    rl_sampler_free(&s);
    if (!ok) { snprintf(error, cap, "top-p sample left the nucleus"); return 0; }

    /* min-p (llama.cpp rule: keep logit >= max + log(min_p)) and the exact distribution API */
    p.top_k = 0u; p.top_p = 1.0f; p.temperature = 1.0f; p.min_p = 0.5f;
    if (!rl_sampler_init(&s, &p, V)) { snprintf(error, cap, "sampler init failed"); return 0; }
    const uint32_t *ids = NULL;
    const float *probs = NULL;
    uint32_t n = rl_sampler_distribution(&s, logits, &ids, &probs);
    const double p777 = 1.0 / (1.0 + exp(-0.5));
    ok = n == 2u && ids[0] == 777u && ids[1] == 4000u && fabs(probs[0] - p777) < 1e-6 && fabs(probs[1] - (1.0 - p777)) < 1e-6;
    for (int i = 0; i < 256 && ok; ++i) { const uint32_t id = rl_sampler_sample(&s, logits); ok = id == 777u || id == 4000u; }
    rl_sampler_free(&s);
    if (!ok) { snprintf(error, cap, "min-p 0.5 must keep exactly the two dominant ids with softmax weights"); return 0; }
    p.min_p = 0.7f; /* log(0.7) = -0.357 < -0.5: only the argmax survives */
    if (!rl_sampler_init(&s, &p, V)) { snprintf(error, cap, "sampler init failed"); return 0; }
    n = rl_sampler_distribution(&s, logits, &ids, &probs);
    ok = n == 1u && ids[0] == 777u && probs[0] == 1.0f;
    rl_sampler_free(&s);
    if (!ok) { snprintf(error, cap, "min-p 0.7 must keep only the argmax"); return 0; }
    p.temperature = 0.0f; p.min_p = 0.0f;
    if (!rl_sampler_init(&s, &p, V)) { snprintf(error, cap, "sampler init failed"); return 0; }
    n = rl_sampler_distribution(&s, logits, &ids, &probs);
    ok = n == 1u && ids[0] == 777u && probs[0] == 1.0f;
    rl_sampler_free(&s);
    if (!ok) { snprintf(error, cap, "greedy distribution must be the argmax with probability 1"); return 0; }
    return 1;
}

static void digest_row(const char *label, uint32_t type, const uint8_t *row, uint32_t ncols, const uint8_t *grid) {
    float *y = (float *)malloc((size_t)ncols * sizeof(float));
    if (!y || !rl_quant_dequant_row(type, row, ncols, grid, y)) { printf("%s: dequant unsupported\n", label); free(y); return; }
    double sum = 0.0, abs_sum = 0.0;
    for (uint32_t i = 0; i < ncols; ++i) { sum += y[i]; abs_sum += fabs(y[i]); }
    printf("%s: type=%s sum=%.9g abs_sum=%.9g first=[%.9g %.9g %.9g %.9g] last=%.9g\n",
        label, rl_gguf_type_name(type), sum, abs_sum, y[0], y[1], y[2], y[3], y[ncols - 1]);
    float *x = (float *)malloc((size_t)ncols * sizeof(float));
    for (uint32_t i = 0; i < ncols; ++i) x[i] = (float)(((int)((i * 7u + 3u) % 31u) - 15) / 16.0);
    double dot = 0.0, ref = 0.0;
    rl_quant_row_dot(type, row, x, ncols, grid, &dot);
    for (uint32_t i = 0; i < ncols; ++i) ref += (double)y[i] * x[i];
    printf("%s: dot=%.9g dequant_dot=%.9g delta=%.3e\n", label, dot, ref, dot - ref);
    free(x); free(y);
}

int main(int argc, char **argv) {
    char error[512] = {0};
    if (!rl_quant_selftest(error, sizeof(error))) { fprintf(stderr, "quant selftest failed: %s\n", error); return 1; }
    printf("quant selftest        : OK (Q8_0 Q2_K Q4_K Q5_K Q6_K IQ2_XXS)\n");
    if (!synthetic_gguf_test(error, sizeof(error))) { fprintf(stderr, "synthetic GGUF test failed: %s\n", error); return 1; }
    printf("synthetic GGUF parse  : OK (token_type length mismatch rejected)\n");
    if (!sampler_selftest(error, sizeof(error))) { fprintf(stderr, "sampler selftest failed: %s\n", error); return 1; }
    printf("sampler selftest      : OK (greedy, top-k selection, top-p nucleus, min-p, distribution)\n");
    if (argc < 2) return 0;

    rl_gguf_model m;
    if (!rl_gguf_model_open(argv[1], &m, error, sizeof(error)) || !rl_gguf_model_map(&m, error, sizeof(error))) {
        fprintf(stderr, "open %s failed: %s\n", argv[1], error); return 1;
    }
    printf("arch=%s layers=%u embd=%u heads=%u/%u key=%u rope=%u experts=%u/%u ff_exp=%u ff_shexp=%u conv=%u state=%u group=%u rank=%u inner=%u eps=%.3g base=%.9g ctx=%u\n",
        m.architecture, m.n_layer, m.n_embd, m.n_head, m.n_head_kv, m.key_length, m.rope_dims, m.n_expert, m.n_expert_used,
        m.n_ff_exp, m.n_ff_shexp, m.ssm_conv, m.ssm_state, m.ssm_group, m.ssm_dt_rank, m.ssm_inner, m.rms_eps, m.rope_freq_base, m.context_length);
    printf("tokenizer=%s pre=%s vocab=%u merges=%u eos=%d bos=%d pad=%d add_bos=%d types=%s template=%zu bytes\n",
        m.tokenizer_model, m.tokenizer_pre, m.vocab_count, m.merge_count, m.eos_id, m.bos_id, m.pad_id, m.add_bos,
        m.token_types ? "yes" : "no", m.chat_template ? strlen(m.chat_template) : 0u);
    printf("tensors=%" PRIu64 " data_base=%" PRIu64 " file=%" PRIu64 "\n", m.tensor_count, m.data_base, m.file_size);
    uint8_t grid[RL_IQ2_XXS_GRID_COUNT];
    rl_native_iq2_xxs_build_grid(grid, error, sizeof(error));
    const char *names[] = {"token_embd.weight", "output.weight", "output_norm.weight", "blk.0.ssm_out.weight", "blk.3.attn_k.weight", "blk.0.ffn_gate_shexp.weight", "blk.0.ssm_ba.weight", "blk.0.attn_qkv.weight"};
    const uint32_t rows[] = {0u, 151645u, 0u, 5u, 7u, 3u, 1u, 9u};
    for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); ++i) {
        const rl_gguf_tensor *t = rl_gguf_find(&m, names[i]);
        if (!t) { printf("%s: missing\n", names[i]); continue; }
        const uint32_t ncols = (uint32_t)t->shape[0];
        const size_t rb = rl_gguf_row_bytes(t->ggml_type, ncols);
        char label[160];
        snprintf(label, sizeof(label), "%s[row %u]", names[i], rows[i]);
        digest_row(label, t->ggml_type, rl_gguf_tensor_data(&m, t) + (size_t)rows[i] * rb, ncols, grid);
    }
    rl_gguf_model_close(&m);
    return 0;
}
