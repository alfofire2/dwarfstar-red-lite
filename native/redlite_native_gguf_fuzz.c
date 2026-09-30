/*
 * Model-free robustness test for both GGUF readers:
 *   - rl_gguf_model_open / rl_gguf_model_map (full directory + metadata reader)
 *   - rl_native_build_expert_map / rl_native_expert_layout (routed-expert map)
 *
 * Two valid synthetic GGUF v3 seeds are built in memory, then every strict
 * truncation, a systematic overwrite of each header offset with boundary values
 * and N seeded random multi-byte mutations are written to a temp file and fed to
 * both readers. Pass criteria: the seeds are accepted, every truncation of the
 * directory seed is rejected, and no input crashes, hangs or trips a sanitizer.
 * Accepted mutants are exercised further (mmap, tensor lookup, first/last payload
 * byte, per-expert layouts) so an accepted-but-inconsistent directory still faults.
 *
 * Deterministic: the PRNG seed is fixed unless --seed is given.
 */
#include "redlite_native_gguf.h"
#include "redlite_native_gguf_dir.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct {
    uint8_t *data;
    size_t len;
    size_t cap;
} buf;

static void need(buf *b, size_t extra) {
    if (b->len + extra <= b->cap) return;
    size_t cap = b->cap ? b->cap : 256u;
    while (cap < b->len + extra) cap *= 2u;
    uint8_t *p = realloc(b->data, cap);
    if (!p) { fprintf(stderr, "out of memory\n"); exit(2); }
    b->data = p;
    b->cap = cap;
}
static void put_bytes(buf *b, const void *p, size_t n) { need(b, n); memcpy(b->data + b->len, p, n); b->len += n; }
static void put_u8(buf *b, uint8_t v) { put_bytes(b, &v, 1u); }
static void put_u32(buf *b, uint32_t v) { for (int i = 0; i < 4; ++i) put_u8(b, (uint8_t)(v >> (8 * i))); }
static void put_u64(buf *b, uint64_t v) { for (int i = 0; i < 8; ++i) put_u8(b, (uint8_t)(v >> (8 * i))); }
static void put_str(buf *b, const char *s) { put_u64(b, strlen(s)); put_bytes(b, s, strlen(s)); }
static void put_kv_u32(buf *b, const char *k, uint32_t v) { put_str(b, k); put_u32(b, 4u); put_u32(b, v); }
static void put_kv_f32(buf *b, const char *k, float v) { uint32_t u; memcpy(&u, &v, 4); put_str(b, k); put_u32(b, 6u); put_u32(b, u); }
static void put_kv_str(buf *b, const char *k, const char *v) { put_str(b, k); put_u32(b, 8u); put_str(b, v); }
static void pad_to(buf *b, size_t align) { while (b->len % align) put_u8(b, 0u); }

/* Same shape as the engine offline fixture, plus a chat template and a BOS id. */
static size_t build_dir_seed(buf *b) {
    put_bytes(b, "GGUF", 4u);
    put_u32(b, 3u);
    put_u64(b, 3u);  /* tensors */
    put_u64(b, 11u); /* kv */
    put_kv_str(b, "general.architecture", "qwen3next");
    put_kv_u32(b, "general.alignment", 32u);
    put_kv_u32(b, "qwen3next.block_count", 2u);
    put_kv_u32(b, "qwen3next.embedding_length", 256u);
    put_kv_f32(b, "qwen3next.attention.layer_norm_rms_epsilon", 1e-6f);
    put_str(b, "tokenizer.ggml.tokens"); put_u32(b, 9u); put_u32(b, 8u); put_u64(b, 3u);
    put_str(b, "a"); put_str(b, "b"); put_str(b, "ab");
    put_str(b, "tokenizer.ggml.token_type"); put_u32(b, 9u); put_u32(b, 5u); put_u64(b, 3u);
    for (int i = 0; i < 3; ++i) put_u32(b, 1u);
    put_str(b, "tokenizer.ggml.merges"); put_u32(b, 9u); put_u32(b, 8u); put_u64(b, 1u); put_str(b, "a b");
    put_kv_u32(b, "tokenizer.ggml.eos_token_id", 2u);
    put_kv_u32(b, "tokenizer.ggml.bos_token_id", 0u);
    put_kv_str(b, "tokenizer.chat_template", "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}");
    put_str(b, "output_norm.weight"); put_u32(b, 1u); put_u64(b, 256u); put_u32(b, 0u); put_u64(b, 0u);
    put_str(b, "blk.0.attn_k.weight"); put_u32(b, 2u); put_u64(b, 256u); put_u64(b, 2u); put_u32(b, 12u); put_u64(b, 1024u);
    put_str(b, "token_embd.weight"); put_u32(b, 1u); put_u64(b, 256u); put_u32(b, 10u); put_u64(b, 1024u + 288u);
    pad_to(b, 32u);
    const size_t header = b->len;
    for (uint32_t i = 0; i < 1024u + 288u + 84u; ++i) put_u8(b, (uint8_t)(i & 255u));
    return header;
}

/* Same shape as the routed-map offline fixture: one layer, four IQ2_XS experts, F16 router. */
static size_t build_routed_seed(buf *b) {
    put_bytes(b, "GGUF", 4u);
    put_u32(b, 3u);
    put_u64(b, 4u);
    put_u64(b, 1u);
    put_kv_u32(b, "general.alignment", 32u);
    const char *names[3] = {"blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight", "blk.0.ffn_down_exps.weight"};
    const uint64_t d0[3] = {256u, 256u, 8u}, d1[3] = {8u, 8u, 256u};
    for (int i = 0; i < 3; ++i) {
        put_str(b, names[i]); put_u32(b, 3u);
        put_u64(b, d0[i]); put_u64(b, d1[i]); put_u64(b, 4u);
        put_u32(b, 17u); put_u64(b, 128u * (uint64_t)i);
    }
    put_str(b, "blk.0.ffn_gate_inp.weight"); put_u32(b, 2u); put_u64(b, 256u); put_u64(b, 4u); put_u32(b, 1u); put_u64(b, 384u);
    pad_to(b, 32u);
    const size_t header = b->len;
    for (uint32_t i = 0; i < 2432u; ++i) put_u8(b, 0u);
    return header;
}

static uint64_t rng_state = 0x9E3779B97F4A7C15ull;
static uint64_t rng(void) {
    uint64_t x = rng_state;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    return rng_state = x;
}

static const uint64_t interesting[] = {
    0u, 1u, 2u, 7u, 8u, 31u, 32u, 255u, 256u, 0x7Fu, 0x80u, 0xFFu, 0xFFFFu, 0x7FFFFFFFu, 0x80000000u,
    0xFFFFFFFFu, 0x100000000ull, 0x7FFFFFFFFFFFFFFFull, 0x8000000000000000ull, 0xFFFFFFFFFFFFFFFFull,
};
#define N_INTERESTING (sizeof(interesting) / sizeof(interesting[0]))

static char tmp_path[512];
static unsigned long accepted_dir = 0, accepted_map = 0, total_inputs = 0;

static int write_file(const uint8_t *p, size_t n) {
    FILE *f = fopen(tmp_path, "wb");
    if (!f) return 0;
    const int ok = (n == 0 || fwrite(p, 1, n, f) == n) && fclose(f) == 0;
    return ok;
}

static volatile uint32_t sink;

/* Returns bit 0 = directory reader accepted, bit 1 = expert map accepted. */
static int run_readers(const uint8_t *p, size_t n) {
    if (!write_file(p, n)) { fprintf(stderr, "cannot write %s\n", tmp_path); exit(2); }
    ++total_inputs;
    int result = 0;
    char error[256];

    rl_gguf_model m;
    if (rl_gguf_model_open(tmp_path, &m, error, sizeof(error))) {
        result |= 1;
        ++accepted_dir;
        if (rl_gguf_model_map(&m, error, sizeof(error))) {
            for (uint64_t i = 0; i < m.tensor_count; ++i) {
                const rl_gguf_tensor *t = &m.tensors[i];
                if (rl_gguf_find(&m, t->name) == NULL) { fprintf(stderr, "accepted tensor %s is not findable\n", t->name); exit(1); }
                const uint8_t *d = rl_gguf_tensor_data(&m, t);
                if (d && t->payload_bytes) sink += d[0] + d[t->payload_bytes - 1u];
            }
            for (uint32_t i = 0; i < m.vocab_count; ++i) sink += (uint32_t)strlen(m.tokens[i]);
            for (uint32_t i = 0; i < m.merge_count; ++i) sink += (uint32_t)strlen(m.merges[i]);
            if (m.chat_template) sink += (uint32_t)strlen(m.chat_template);
        }
        rl_gguf_model_close(&m);
    }

    rl_expert_map map;
    memset(&map, 0, sizeof(map));
    if (rl_native_build_expert_map(tmp_path, 4u, &map, error, sizeof(error))) {
        result |= 2;
        ++accepted_map;
        for (uint32_t layer = 0; layer < map.layer_count && layer < 4u; ++layer) {
            for (uint32_t e = 0; e < map.expert_count; ++e) {
                rl_expert_layout lay;
                if (rl_native_expert_layout(&map, layer, e, &lay, error, sizeof(error))) {
                    if (lay.down_offset + lay.down_bytes < lay.down_offset) { fprintf(stderr, "layout overflow accepted\n"); exit(1); }
                }
            }
        }
        rl_native_free_expert_map(&map);
    }
    return result;
}

static void mutate(uint8_t *p, size_t *n, size_t header) {
    const int count = 1 + (int)(rng() % 4u);
    for (int k = 0; k < count && *n > 0; ++k) {
        /* Bias three quarters of the mutations into the header/directory, where the parsers live. */
        const size_t span = (rng() % 4u) ? (header < *n ? header : *n) : *n;
        const size_t at = (size_t)(rng() % span);
        switch (rng() % 6u) {
        case 0: p[at] ^= (uint8_t)(1u << (rng() % 8u)); break;
        case 1: p[at] = (uint8_t)rng(); break;
        case 2: {
            const uint64_t v = interesting[rng() % N_INTERESTING];
            for (size_t i = 0; i < 4u && at + i < *n; ++i) p[at + i] = (uint8_t)(v >> (8u * i));
            break;
        }
        case 3: {
            const uint64_t v = interesting[rng() % N_INTERESTING];
            for (size_t i = 0; i < 8u && at + i < *n; ++i) p[at + i] = (uint8_t)(v >> (8u * i));
            break;
        }
        case 4: *n = at; break; /* truncate */
        default: {                /* delete a small chunk */
            const size_t len = 1u + (size_t)(rng() % 16u);
            if (at + len <= *n) { memmove(p + at, p + at + len, *n - at - len); *n -= len; }
            break;
        }
        }
    }
}

int main(int argc, char **argv) {
    unsigned long iterations = 2000u;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--iterations") && i + 1 < argc) iterations = strtoul(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) rng_state = strtoull(argv[++i], NULL, 0) | 1u;
        else {
            printf("Usage: redlite-gguf-fuzz [--iterations N] [--seed S]\n");
            return !strcmp(argv[i], "--help") ? 0 : 2;
        }
    }
    const char *tmpdir = getenv("TMPDIR");
    snprintf(tmp_path, sizeof(tmp_path), "%s/redlite_gguf_fuzz_%ld.gguf", tmpdir && *tmpdir ? tmpdir : "/tmp", (long)getpid());

    buf seeds[2] = {{0}, {0}};
    size_t headers[2];
    headers[0] = build_dir_seed(&seeds[0]);
    headers[1] = build_routed_seed(&seeds[1]);

    int ok = 1;
    if (!(run_readers(seeds[0].data, seeds[0].len) & 1)) { fprintf(stderr, "directory seed rejected by rl_gguf_model_open\n"); ok = 0; }
    if (!(run_readers(seeds[1].data, seeds[1].len) & 2)) { fprintf(stderr, "routed seed rejected by rl_native_build_expert_map\n"); ok = 0; }

    /* Every strict truncation. The directory seed's last payload ends exactly at EOF, so all must be rejected. */
    unsigned long truncation_accepts = 0;
    for (int s = 0; s < 2; ++s) {
        for (size_t n = 0; n < seeds[s].len; ++n) {
            const int r = run_readers(seeds[s].data, n);
            if (s == 0 && (r & 1)) {
                if (!truncation_accepts) fprintf(stderr, "directory seed truncated to %zu/%zu bytes was accepted\n", n, seeds[s].len);
                ++truncation_accepts;
            }
        }
    }
    if (truncation_accepts) ok = 0;

    /* Systematic boundary values at every header offset, both widths. */
    uint8_t *work = malloc(seeds[0].len > seeds[1].len ? seeds[0].len : seeds[1].len);
    if (!work) return 2;
    for (int s = 0; s < 2; ++s) {
        for (size_t at = 0; at < headers[s]; ++at) {
            for (size_t v = 0; v < N_INTERESTING; ++v) {
                for (size_t width = 4u; width <= 8u; width += 4u) {
                    memcpy(work, seeds[s].data, seeds[s].len);
                    for (size_t i = 0; i < width && at + i < seeds[s].len; ++i) work[at + i] = (uint8_t)(interesting[v] >> (8u * i));
                    (void)run_readers(work, seeds[s].len);
                }
            }
        }
    }

    /* Seeded random multi-mutations. */
    for (unsigned long it = 0; it < iterations; ++it) {
        const int s = (int)(it & 1u);
        size_t n = seeds[s].len;
        memcpy(work, seeds[s].data, n);
        mutate(work, &n, headers[s]);
        (void)run_readers(work, n);
    }

    unlink(tmp_path);
    free(work);
    free(seeds[0].data);
    free(seeds[1].data);
    printf("gguf fuzz inputs    : %lu (directory accepted %lu, expert map accepted %lu)\n", total_inputs, accepted_dir, accepted_map);
    printf("truncations accepted: %lu (directory seed; must be 0)\n", truncation_accepts);
    printf("gguf fuzz           : %s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
