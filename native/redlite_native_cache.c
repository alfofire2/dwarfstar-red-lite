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
    cache->capacity = capacity;
    return 1;
}

void rl_native_lru_free(rl_native_lru *cache) {
    if (!cache) return;
    free(cache->entries);
    memset(cache, 0, sizeof(*cache));
}

static int index_for_key(const rl_native_lru *cache, rl_cache_key key) {
    for (uint32_t i = 0; i < cache->capacity; ++i) {
        if (cache->entries[i].valid && key_equal(cache->entries[i].key, key)) return (int)i;
    }
    return -1;
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

static int selected_contains(const rl_cache_key *keys, uint32_t count, rl_cache_key key) {
    for (uint32_t i = 0; i < count; ++i) {
        if (key_equal(keys[i], key)) return 1;
    }
    return 0;
}

static int first_unreserved_free_entry(const rl_native_lru *cache, const uint8_t *reserved) {
    for (uint32_t i = 0; i < cache->capacity; ++i) {
        if (!cache->entries[i].valid && !reserved[i]) return (int)i;
    }
    return -1;
}

static int planned_victim_entry(
        const rl_native_lru *cache,
        const rl_cache_key *protected_keys,
        uint32_t protected_count,
        const uint8_t *reserved,
        uint64_t *blocked) {
    int victim = -1;
    uint64_t oldest = UINT64_MAX;
    for (uint32_t i = 0; i < cache->capacity; ++i) {
        const rl_cache_entry *entry = &cache->entries[i];
        if (!entry->valid || reserved[i]) continue;
        if (entry->inflight || selected_contains(protected_keys, protected_count, entry->key)) {
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
    if (count > cache->capacity) {
        set_error(error, error_cap, "native cache capacity is smaller than current top-k selection");
        return 0;
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
    for (uint32_t i = 0; i < count; ++i) {
        items[i].key = keys[i];
        const int existing = index_for_key(cache, keys[i]);
        if (existing >= 0) {
            items[i].entry_index = (uint32_t)existing;
            items[i].slot_id = cache->entries[existing].slot_id;
            items[i].hit = 1;
            reserved[existing] = 1;
            continue;
        }

        int index = first_unreserved_free_entry(cache, reserved);
        if (index < 0) {
            index = planned_victim_entry(cache, keys, count, reserved, &blocked);
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
            if (entry->valid) cache->evictions++;
            else cache->resident++;
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
