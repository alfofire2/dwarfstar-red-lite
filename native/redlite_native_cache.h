#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t layer;
    uint32_t expert;
} rl_cache_key;

typedef struct {
    rl_cache_key key;
    uint32_t slot_id;
    uint64_t stamp;
    uint32_t inflight;
    int valid;
} rl_cache_entry;

typedef struct {
    uint32_t capacity;
    uint32_t resident;
    uint64_t clock;
    uint64_t hits;
    uint64_t misses;
    uint64_t evictions;
    uint64_t blocked_victims;
    rl_cache_entry *entries;
} rl_native_lru;

int rl_native_lru_init(rl_native_lru *cache, uint32_t capacity);
void rl_native_lru_free(rl_native_lru *cache);

int rl_native_lru_acquire_many(
    rl_native_lru *cache,
    const rl_cache_key *keys,
    uint32_t count,
    uint32_t *slot_ids,
    char *error,
    size_t error_cap);

int rl_native_lru_set_inflight(
    rl_native_lru *cache,
    uint32_t slot_id,
    uint32_t inflight);

int rl_native_lru_lookup(
    const rl_native_lru *cache,
    rl_cache_key key,
    uint32_t *slot_id);

#ifdef __cplusplus
}
#endif
