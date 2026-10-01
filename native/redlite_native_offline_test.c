#define _FILE_OFFSET_BITS 64
#define _POSIX_C_SOURCE 200809L

#include "redlite_native_cache.h"
#include "redlite_native_gguf.h"
#include "redlite_native_model.h"
#include "redlite_native_router.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int write_u32(FILE *f, uint32_t v) {
    unsigned char b[4] = {
        (unsigned char)(v & 0xffu),
        (unsigned char)((v >> 8) & 0xffu),
        (unsigned char)((v >> 16) & 0xffu),
        (unsigned char)((v >> 24) & 0xffu),
    };
    return fwrite(b, 1, sizeof(b), f) == sizeof(b);
}

static int write_u64(FILE *f, uint64_t v) {
    unsigned char b[8];
    for (unsigned i = 0; i < 8; ++i) b[i] = (unsigned char)((v >> (8u * i)) & 0xffu);
    return fwrite(b, 1, sizeof(b), f) == sizeof(b);
}

static int write_string(FILE *f, const char *s) {
    const size_t n = strlen(s);
    return write_u64(f, (uint64_t)n) && fwrite(s, 1, n, f) == n;
}

static int write_tensor3(
        FILE *f,
        const char *name,
        uint64_t d0,
        uint64_t d1,
        uint64_t d2,
        uint32_t type,
        uint64_t offset) {
    return write_string(f, name) && write_u32(f, 3u) &&
           write_u64(f, d0) && write_u64(f, d1) && write_u64(f, d2) &&
           write_u32(f, type) && write_u64(f, offset);
}

static int write_tensor2(
        FILE *f,
        const char *name,
        uint64_t d0,
        uint64_t d1,
        uint32_t type,
        uint64_t offset) {
    return write_string(f, name) && write_u32(f, 2u) &&
           write_u64(f, d0) && write_u64(f, d1) &&
           write_u32(f, type) && write_u64(f, offset);
}

static int make_fixture(char path[64]) {
    strcpy(path, "/tmp/redlite-native-fixture-XXXXXX");
    int fd = mkstemp(path);
    if (fd < 0) return 0;
    FILE *f = fdopen(fd, "wb");
    if (!f) {
        close(fd);
        unlink(path);
        return 0;
    }

    int ok = fwrite("GGUF", 1, 4, f) == 4 &&
             write_u32(f, 3u) && write_u64(f, 4u) && write_u64(f, 1u) &&
             write_string(f, "general.alignment") && write_u32(f, 4u) && write_u32(f, 32u) &&
             write_tensor3(f, "blk.0.ffn_gate_exps.weight", 256u, 8u, 4u, 17u, 0u) &&
             write_tensor3(f, "blk.0.ffn_up_exps.weight",   256u, 8u, 4u, 17u, 128u) &&
             write_tensor3(f, "blk.0.ffn_down_exps.weight",   8u, 256u, 4u, 17u, 256u) &&
             write_tensor2(f, "blk.0.ffn_gate_inp.weight", 256u, 4u, 1u, 384u);

    long pos = ftell(f);
    if (pos < 0) ok = 0;
    while (ok && ((unsigned long)pos % 32u)) {
        if (fputc(0, f) == EOF) ok = 0;
        pos++;
    }
    unsigned char zero[2432] = {0};
    if (ok && fwrite(zero, 1, sizeof(zero), f) != sizeof(zero)) ok = 0;
    if (fclose(f) != 0) ok = 0;
    if (!ok) unlink(path);
    return ok;
}

static int test_synthetic_gguf(void) {
    char path[64];
    if (!make_fixture(path)) {
        fprintf(stderr, "failed to create synthetic GGUF fixture\n");
        return 0;
    }

    char error[256];
    rl_expert_map map;
    const int parsed = rl_native_build_expert_map(path, 4u, &map, error, sizeof(error));
    if (!parsed) {
        fprintf(stderr, "synthetic GGUF parse failed: %s\n", error);
        unlink(path);
        return 0;
    }

    int ok = map.version == 3u && map.alignment == 32u && map.tensor_count == 4u &&
             map.routed_tensor_count == 3u && map.layer_count == 1u && map.expert_count == 4u &&
             map.all_slice_safe && map.total_routed_payload_bytes == 384u &&
             map.max_expert_triplet_bytes == 96u;
    if (!ok) {
        fprintf(stderr, "synthetic GGUF map invariants failed\n");
        rl_native_free_expert_map(&map);
        unlink(path);
        return 0;
    }

    rl_native_layer_info info;
    if (!rl_native_get_layer_info(&map, 0u, &info, error, sizeof(error)) ||
        info.ggml_type != 17u || info.hidden_size != 256u || info.ffn_size != 8u ||
        info.expert_count != 4u) {
        fprintf(stderr, "synthetic GGUF layer-info invariants failed: %s\n", error);
        rl_native_free_expert_map(&map);
        unlink(path);
        return 0;
    }

    rl_expert_layout layout;
    if (!rl_native_expert_layout(&map, 0u, 2u, &layout, error, sizeof(error)) ||
        layout.ggml_type != 17u || layout.gate_bytes != 32u || layout.up_bytes != 32u ||
        layout.down_bytes != 32u || layout.total_bytes != 96u ||
        layout.up_offset - layout.gate_offset != 128u ||
        layout.down_offset - layout.up_offset != 128u) {
        fprintf(stderr, "synthetic GGUF expert-layout invariants failed: %s\n", error);
        rl_native_free_expert_map(&map);
        unlink(path);
        return 0;
    }

    rl_router_tensor_info routers[4];
    uint32_t router_count = 0;
    if (!rl_native_router_audit(path, routers, 4u, &router_count, error, sizeof(error)) ||
        router_count != 1u || routers[0].layer != 0u || routers[0].ggml_type != 1u ||
        routers[0].n_dims != 2u || routers[0].shape[0] != 256u || routers[0].shape[1] != 4u ||
        routers[0].tensor_span_bytes != 2048u) {
        fprintf(stderr, "synthetic router audit invariants failed: %s\n", error);
        rl_native_free_expert_map(&map);
        unlink(path);
        return 0;
    }

    rl_native_free_expert_map(&map);
    unlink(path);
    return 1;
}

/* dev37: size classes. Layer 0 -> class 0 (2 entries), layer 1 -> class 1 (3 entries): a class only evicts its own
 * entries, slot ids stay inside the class range, and a selection larger than its class is refused. */
static int test_lru_classes(void) {
    rl_native_lru cache;
    if (!rl_native_lru_init(&cache, 5u)) return 0;
    const uint32_t cap[2] = {2u, 3u};
    const uint8_t layer_class[2] = {0u, 1u};
    char error[256];
    int ok = rl_native_lru_set_classes(&cache, 2u, cap, layer_class, 2u);
    rl_cache_key a[2] = {{0u, 1u}, {0u, 2u}}, b[3] = {{1u, 1u}, {1u, 2u}, {1u, 3u}}, c[1] = {{0u, 3u}}, big[3] = {{0u, 4u}, {0u, 5u}, {0u, 6u}};
    uint32_t slots[3];
    ok = ok && rl_native_lru_acquire_many(&cache, a, 2u, slots, error, sizeof(error)) && slots[0] < 2u && slots[1] < 2u;
    ok = ok && rl_native_lru_acquire_many(&cache, b, 3u, slots, error, sizeof(error)) && slots[0] >= 2u && slots[1] >= 2u && slots[2] >= 2u;
    /* a new layer-0 expert must evict the oldest layer-0 entry (expert 1), never the newer layer-1 entries */
    ok = ok && rl_native_lru_acquire_many(&cache, c, 1u, slots, error, sizeof(error)) && slots[0] < 2u;
    ok = ok && !rl_native_lru_lookup(&cache, a[0], NULL) && rl_native_lru_lookup(&cache, a[1], NULL);
    for (uint32_t i = 0; i < 3u; ++i) ok = ok && rl_native_lru_lookup(&cache, b[i], NULL);
    ok = ok && !rl_native_lru_acquire_many(&cache, big, 3u, slots, error, sizeof(error));
    ok = ok && !rl_native_lru_set_classes(&cache, 2u, cap, layer_class, 2u);   /* refused once entries are resident */
    rl_native_lru_free(&cache);
    return ok;
}

static int test_lru_transaction(void) {
    rl_native_lru cache;
    if (!rl_native_lru_init(&cache, 2u)) return 0;
    char error[256];
    rl_cache_key initial[2] = {{0u, 1u}, {0u, 2u}};
    uint32_t slots[2];
    if (!rl_native_lru_acquire_many(&cache, initial, 2u, slots, error, sizeof(error))) {
        rl_native_lru_free(&cache);
        return 0;
    }
    const uint64_t hits_before = cache.hits;
    const uint64_t misses_before = cache.misses;
    const uint64_t evictions_before = cache.evictions;

    rl_cache_key selected[2] = {{0u, 1u}, {0u, 3u}};
    rl_cache_transaction tx = {0};
    if (!rl_native_lru_prepare_many(&cache, selected, 2u, &tx, error, sizeof(error))) {
        fprintf(stderr, "transaction prepare failed: %s\n", error);
        rl_native_lru_free(&cache);
        return 0;
    }
    if (!tx.items[0].hit || tx.items[1].hit ||
        cache.hits != hits_before || cache.misses != misses_before || cache.evictions != evictions_before) {
        fprintf(stderr, "transaction prepare mutated published LRU state\n");
        rl_native_lru_transaction_free(&tx);
        rl_native_lru_free(&cache);
        return 0;
    }
    uint32_t ignored = 0;
    if (rl_native_lru_lookup(&cache, selected[1], &ignored) ||
        !rl_native_lru_lookup(&cache, initial[1], &ignored)) {
        fprintf(stderr, "uncommitted transaction became visible\n");
        rl_native_lru_transaction_free(&tx);
        rl_native_lru_free(&cache);
        return 0;
    }

    rl_native_lru_abort(&cache, &tx);
    rl_native_lru_transaction_free(&tx);
    if (!rl_native_lru_lookup(&cache, initial[0], &ignored) ||
        rl_native_lru_lookup(&cache, initial[1], &ignored) ||
        rl_native_lru_lookup(&cache, selected[1], &ignored) || cache.resident != 1u) {
        fprintf(stderr, "transaction abort did not invalidate touched victim safely\n");
        rl_native_lru_free(&cache);
        return 0;
    }

    rl_cache_transaction retry = {0};
    if (!rl_native_lru_prepare_many(&cache, selected, 2u, &retry, error, sizeof(error)) ||
        !rl_native_lru_commit(&cache, &retry, slots, error, sizeof(error))) {
        fprintf(stderr, "transaction retry/commit failed: %s\n", error);
        rl_native_lru_transaction_free(&retry);
        rl_native_lru_free(&cache);
        return 0;
    }
    rl_native_lru_transaction_free(&retry);
    if (!rl_native_lru_lookup(&cache, selected[0], &ignored) ||
        !rl_native_lru_lookup(&cache, selected[1], &ignored) || cache.resident != 2u) {
        fprintf(stderr, "committed transaction was not published correctly\n");
        rl_native_lru_free(&cache);
        return 0;
    }

    rl_native_lru_free(&cache);
    return 1;
}

int main(void) {
    if (!test_synthetic_gguf()) return 1;
    if (!test_lru_transaction()) return 1;
    if (!test_lru_classes()) { fprintf(stderr, "LRU size class test failed\n"); return 1; }
    printf("synthetic GGUF      : OK\n");
    printf("native layer info  : OK\n");
    printf("expert slicing     : OK\n");
    printf("router discovery   : OK\n");
    printf("LRU transaction    : OK\n");
    return 0;
}
