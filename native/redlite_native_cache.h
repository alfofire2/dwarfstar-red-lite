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

typedef struct {
    rl_cache_key key;
    uint32_t entry_index;
    uint32_t slot_id;
    int hit;
} rl_cache_reservation;

typedef struct {
    rl_cache_reservation *items;
    uint32_t count;
    uint64_t blocked_victims;
    int active;
} rl_cache_transaction;

int rl_native_lru_init(rl_native_lru *cache, uint32_t capacity);
void rl_native_lru_free(rl_native_lru *cache);

/*
 * Prepare a complete top-k selection without mutating resident metadata.
 * Miss slots may then be filled by I/O. Commit publishes all mappings only
 * after every load succeeds; abort invalidates every miss slot because its
 * previous bytes may already have been overwritten.
 */
int rl_native_lru_prepare_many(
    rl_native_lru *cache,
    const rl_cache_key *keys,
    uint32_t count,
    rl_cache_transaction *transaction,
    char *error,
    size_t error_cap);

int rl_native_lru_commit(
    rl_native_lru *cache,
    rl_cache_transaction *transaction,
    uint32_t *slot_ids,
    char *error,
    size_t error_cap);

void rl_native_lru_abort(
    rl_native_lru *cache,
    rl_cache_transaction *transaction);

void rl_native_lru_transaction_free(rl_cache_transaction *transaction);

/* Convenience helper for metadata-only users/tests: prepare + immediate commit. */
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
