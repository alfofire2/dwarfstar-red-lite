#include "redlite_native_cache.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int key_equal(rl_cache_key a, rl_cache_key b) {
    return a.layer == b.layer && a.expert == b.expert;
}

static void set_error(char *dst, size_t cap, const char *msg) {
    if (!dst || !cap) return;
    snprintf(dst, cap, "%s", msg ? msg : "unknown cache error");
}

int rl_native_lru_init(rl_native_lru *cache, uint32_t capacity) {
    if (!cache || !capacity) return 0;
    memset(cache, 0, sizeof(*cache));
    cache->entries = (rl_cache_entry *)calloc(capacity, sizeof(*cache->entries));
    if (!cache->entries) return 0;
    uint32_t size = 16u;
    while (size < 2u * capacity) size <<= 1;
    cache->index = (uint32_t *)calloc(size, sizeof(uint32_t));
    if (!cache->index) { free(cache->entries); cache->entries = NULL; return 0; }
    cache->index_mask = size - 1u;
    cache->capacity = capacity;
    cache->n_class = 1u;
    cache->class_first[1] = capacity;
    return 1;
}

int rl_native_lru_set_classes(rl_native_lru *cache, uint32_t n_class, const uint32_t *class_capacity,
                              const uint8_t *layer_class, uint32_t n_layer) {
    if (!cache || !cache->entries || cache->resident || !n_class || n_class > RL_LRU_MAX_CLASSES || !class_capacity ||
        !layer_class || n_layer > RL_LRU_MAX_LAYERS) return 0;
    uint32_t sum = 0;
    for (uint32_t c = 0; c < n_class; ++c) {
        if (!class_capacity[c] || class_capacity[c] > cache->capacity - sum) return 0;
        sum += class_capacity[c];
    }
    if (sum != cache->capacity) return 0;
    for (uint32_t l = 0; l < n_layer; ++l) if (layer_class[l] >= n_class) return 0;
    cache->n_class = n_class;
    cache->class_first[0] = 0;
    for (uint32_t c = 0; c < n_class; ++c) cache->class_first[c + 1u] = cache->class_first[c] + class_capacity[c];
    memset(cache->layer_class, 0, sizeof(cache->layer_class));
    memcpy(cache->layer_class, layer_class, n_layer);
    return 1;
}

static uint32_t class_of(const rl_native_lru *cache, rl_cache_key key) {
    return key.layer < RL_LRU_MAX_LAYERS ? cache->layer_class[key.layer] : 0u;
}

void rl_native_lru_free(rl_native_lru *cache) {
    if (!cache) return;
    free(cache->entries);
    free(cache->index);
    memset(cache, 0, sizeof(*cache));
}

/* ---- key index: open addressing with linear probing and backward-shift deletion ---- */
static uint32_t key_hash(rl_cache_key key) {
    uint32_t h = key.layer * 0x9E3779B1u;
    h ^= key.expert * 0x85EBCA77u;
    h ^= h >> 15;
    h *= 0x2C1B3C6Du;
    h ^= h >> 12;
    return h;
}

static int index_for_key(const rl_native_lru *cache, rl_cache_key key) {
    if (!cache->index) return -1;
    uint32_t pos = key_hash(key) & cache->index_mask;
    for (;;) {
        const uint32_t v = cache->index[pos];
        if (!v) return -1;
        const rl_cache_entry *entry = &cache->entries[v - 1u];
        if (entry->valid && key_equal(entry->key, key)) return (int)(v - 1u);
        pos = (pos + 1u) & cache->index_mask;
    }
}

static void index_insert(rl_native_lru *cache, rl_cache_key key, uint32_t entry_index) {
    uint32_t pos = key_hash(key) & cache->index_mask;
    while (cache->index[pos]) pos = (pos + 1u) & cache->index_mask;
    cache->index[pos] = entry_index + 1u;
}

static void index_remove(rl_native_lru *cache, rl_cache_key key, uint32_t entry_index) {
    uint32_t pos = key_hash(key) & cache->index_mask;
    while (cache->index[pos] && cache->index[pos] != entry_index + 1u) pos = (pos + 1u) & cache->index_mask;
    if (!cache->index[pos]) return;
    /* backward shift: pull later probe-chain members into the hole */
    uint32_t hole = pos;
    uint32_t j = (pos + 1u) & cache->index_mask;
    while (cache->index[j]) {
        const uint32_t home = key_hash(cache->entries[cache->index[j] - 1u].key) & cache->index_mask;
        /* the element at j may move to the hole when its home is not in (hole, j] cyclically */
        const int in_range = hole <= j ? (home > hole && home <= j) : (home > hole || home <= j);
        if (!in_range) { cache->index[hole] = cache->index[j]; hole = j; }
        j = (j + 1u) & cache->index_mask;
    }
    cache->index[hole] = 0;
}

int rl_native_lru_lookup(const rl_native_lru *cache, rl_cache_key key, uint32_t *slot_id) {
    if (!cache || !cache->entries) return 0;
    const int index = index_for_key(cache, key);
    if (index < 0) return 0;
    if (slot_id) *slot_id = cache->entries[index].slot_id;
    return 1;
}

int rl_native_lru_set_inflight(rl_native_lru *cache, uint32_t slot_id, uint32_t inflight) {
    if (!cache || !cache->entries || slot_id >= cache->capacity) return 0;
    for (uint32_t i = 0; i < cache->capacity; ++i) {
        if (cache->entries[i].valid && cache->entries[i].slot_id == slot_id) {
            cache->entries[i].inflight = inflight;
            return 1;
        }
    }
    return 0;
}



/* Oldest valid entry that is neither reserved by this transaction nor in flight. Every selected key
 * that is resident has already been reserved (hits are processed before misses), so the selection
 * itself never needs to be scanned here. */
static int planned_victim_entry(
        const rl_native_lru *cache,
        const uint8_t *reserved,
        uint32_t lo,
        uint32_t hi,
        uint64_t *blocked) {
    int victim = -1;
    uint64_t oldest = UINT64_MAX;
    for (uint32_t i = lo; i < hi; ++i) {
        const rl_cache_entry *entry = &cache->entries[i];
        if (!entry->valid || reserved[i]) continue;
        if (entry->inflight) {
            if (blocked) (*blocked)++;
            continue;
        }
        if (entry->stamp < oldest) {
            oldest = entry->stamp;
            victim = (int)i;
        }
    }
    return victim;
}

void rl_native_lru_transaction_free(rl_cache_transaction *transaction) {
    if (!transaction) return;
    free(transaction->items);
    memset(transaction, 0, sizeof(*transaction));
}

int rl_native_lru_prepare_many(
        rl_native_lru *cache,
        const rl_cache_key *keys,
        uint32_t count,
        rl_cache_transaction *transaction,
        char *error,
        size_t error_cap) {
    if (!cache || !cache->entries || !keys || !count || !transaction) {
        set_error(error, error_cap, "invalid native LRU prepare_many arguments");
        return 0;
    }
    if (transaction->active || transaction->items) {
        set_error(error, error_cap, "native LRU transaction is already active");
        return 0;
    }
    uint32_t per_class[RL_LRU_MAX_CLASSES] = {0};
    for (uint32_t i = 0; i < count; ++i) per_class[class_of(cache, keys[i])]++;
    for (uint32_t c = 0; c < cache->n_class; ++c) {
        if (per_class[c] > cache->class_first[c + 1u] - cache->class_first[c]) {
            set_error(error, error_cap, "native cache capacity is smaller than current top-k selection");
            return 0;
        }
    }
    for (uint32_t i = 0; i < count; ++i) {
        for (uint32_t j = i + 1; j < count; ++j) {
            if (key_equal(keys[i], keys[j])) {
                set_error(error, error_cap, "top-k selection contains duplicate experts");
                return 0;
            }
        }
    }

    rl_cache_reservation *items = (rl_cache_reservation *)calloc(count, sizeof(*items));
    uint8_t *reserved = (uint8_t *)calloc(cache->capacity, 1);
    if (!items || !reserved) {
        free(items);
        free(reserved);
        set_error(error, error_cap, "out of memory preparing native LRU transaction");
        return 0;
    }

    uint64_t blocked = 0;
    /* pass 1: every resident key of the selection is reserved before any victim is chosen */
    for (uint32_t i = 0; i < count; ++i) {
        items[i].key = keys[i];
        const int existing = index_for_key(cache, keys[i]);
        items[i].hit = existing >= 0;
        if (existing >= 0) {
            items[i].entry_index = (uint32_t)existing;
            items[i].slot_id = cache->entries[existing].slot_id;
            reserved[existing] = 1;
        }
    }
    /* pass 2: misses take free entries of their class first, then its oldest unreserved, not-in-flight entries */
    uint32_t free_scan[RL_LRU_MAX_CLASSES];
    for (uint32_t c = 0; c < cache->n_class; ++c) free_scan[c] = cache->class_first[c];
    for (uint32_t i = 0; i < count; ++i) {
        if (items[i].hit) continue;
        const uint32_t c = class_of(cache, keys[i]);
        const uint32_t hi = cache->class_first[c + 1u];
        int index = -1;
        for (; free_scan[c] < hi; ++free_scan[c]) {
            if (!cache->entries[free_scan[c]].valid && !reserved[free_scan[c]]) { index = (int)free_scan[c]; break; }
        }
        if (index < 0) {
            index = planned_victim_entry(cache, reserved, cache->class_first[c], hi, &blocked);
        }
        if (index < 0) {
            free(items);
            free(reserved);
            set_error(error, error_cap, "all native LRU slots are protected, reserved or in-flight");
            return 0;
        }
        items[i].entry_index = (uint32_t)index;
        items[i].slot_id = cache->entries[index].valid ? cache->entries[index].slot_id : (uint32_t)index;
        items[i].hit = 0;
        items[i].evicts = cache->entries[index].valid;
        if (items[i].evicts) items[i].evicted_key = cache->entries[index].key;
        reserved[index] = 1;
    }
    free(reserved);

    transaction->items = items;
    transaction->count = count;
    transaction->blocked_victims = blocked;
    transaction->active = 1;
    if (error && error_cap) error[0] = '\0';
    return 1;
}

int rl_native_lru_commit(
        rl_native_lru *cache,
        rl_cache_transaction *transaction,
        uint32_t *slot_ids,
        char *error,
        size_t error_cap) {
    if (!cache || !cache->entries || !transaction || !transaction->active || !transaction->items) {
        set_error(error, error_cap, "invalid native LRU transaction commit");
        return 0;
    }

    /* Validate every planned hit before publishing any mutation. */
    for (uint32_t i = 0; i < transaction->count; ++i) {
        const rl_cache_reservation *r = &transaction->items[i];
        if (r->entry_index >= cache->capacity) {
            set_error(error, error_cap, "native LRU reservation index is out of range");
            return 0;
        }
        if (r->hit) {
            const rl_cache_entry *entry = &cache->entries[r->entry_index];
            if (!entry->valid || entry->slot_id != r->slot_id || !key_equal(entry->key, r->key)) {
                set_error(error, error_cap, "native LRU changed while transaction was prepared");
                return 0;
            }
        }
    }

    for (uint32_t i = 0; i < transaction->count; ++i) {
        const rl_cache_reservation *r = &transaction->items[i];
        rl_cache_entry *entry = &cache->entries[r->entry_index];
        if (r->hit) {
            cache->hits++;
            entry->stamp = ++cache->clock;
        } else {
            cache->misses++;
            if (entry->valid) { cache->evictions++; index_remove(cache, entry->key, r->entry_index); }
            else cache->resident++;
            index_insert(cache, r->key, r->entry_index);
            entry->key = r->key;
            entry->slot_id = r->slot_id;
            entry->stamp = ++cache->clock;
            entry->inflight = 0;
            entry->valid = 1;
        }
        if (slot_ids) slot_ids[i] = r->slot_id;
    }
    cache->blocked_victims += transaction->blocked_victims;
    transaction->active = 0;
    if (error && error_cap) error[0] = '\0';
    return 1;
}

void rl_native_lru_abort(rl_native_lru *cache, rl_cache_transaction *transaction) {
    if (!cache || !cache->entries || !transaction || !transaction->active || !transaction->items) return;

    /*
     * A miss slot may already contain a full or partial replacement payload.
     * Never restore metadata for the previous key: its bytes are no longer
     * trustworthy. Invalidate every touched miss slot instead. Hits were never
     * overwritten and remain resident.
     */
    for (uint32_t i = 0; i < transaction->count; ++i) {
        const rl_cache_reservation *r = &transaction->items[i];
        if (r->hit || r->entry_index >= cache->capacity) continue;
        rl_cache_entry *entry = &cache->entries[r->entry_index];
        if (entry->valid) {
            index_remove(cache, entry->key, r->entry_index);
            memset(entry, 0, sizeof(*entry));
            if (cache->resident) cache->resident--;
            cache->evictions++;
        }
    }
    transaction->active = 0;
}

int rl_native_lru_acquire_many(
        rl_native_lru *cache,
        const rl_cache_key *keys,
        uint32_t count,
        uint32_t *slot_ids,
        char *error,
        size_t error_cap) {
    if (!slot_ids) {
        set_error(error, error_cap, "native LRU acquire_many requires slot output");
        return 0;
    }
    rl_cache_transaction transaction = {0};
    if (!rl_native_lru_prepare_many(cache, keys, count, &transaction, error, error_cap)) return 0;
    const int ok = rl_native_lru_commit(cache, &transaction, slot_ids, error, error_cap);
    rl_native_lru_transaction_free(&transaction);
    return ok;
}
