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

#define RL_LRU_MAX_CLASSES 8u
#define RL_LRU_MAX_LAYERS 256u

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
    uint32_t *index;        /* open-addressing hash of resident keys -> entry index + 1 (0 = empty) */
    uint32_t index_mask;
    /* dev37: size classes. Class c owns entries [class_first[c], class_first[c + 1]); a key of layer l lives in class
     * layer_class[l] (layers >= RL_LRU_MAX_LAYERS: class 0). One class covering every entry after init. */
    uint32_t n_class;
    uint32_t class_first[RL_LRU_MAX_CLASSES + 1];
    uint8_t layer_class[RL_LRU_MAX_LAYERS];
    /* dev79: entries of this layer are never chosen as victims (UINT32_MAX: none); the event decode prefetches the
     * next layer's experts while the GPU may be reading that layer's residency entries */
    uint32_t protect_layer;
} rl_native_lru;

typedef struct {
    rl_cache_key key;
    uint32_t entry_index;
    uint32_t slot_id;
    int hit;
    int evicts;               /* miss that replaces a resident key (evicted_key) */
    rl_cache_key evicted_key;
} rl_cache_reservation;

typedef struct {
    rl_cache_reservation *items;
    uint32_t count;
    uint64_t blocked_victims;
    int active;
} rl_cache_transaction;

int rl_native_lru_init(rl_native_lru *cache, uint32_t capacity);
void rl_native_lru_free(rl_native_lru *cache);

/* dev37: split the entries into n_class consecutive classes of class_capacity[c] entries (sum == capacity) and map
 * each of n_layer layers to one; only while nothing is resident. Victims are chosen within the key's class. */
int rl_native_lru_set_classes(rl_native_lru *cache, uint32_t n_class, const uint32_t *class_capacity,
                              const uint8_t *layer_class, uint32_t n_layer);

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
