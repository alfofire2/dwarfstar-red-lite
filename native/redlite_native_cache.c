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

static int first_free_entry(const rl_native_lru *cache) {
    for (uint32_t i = 0; i < cache->capacity; ++i) {
        if (!cache->entries[i].valid) return (int)i;
    }
    return -1;
}

static int victim_entry(rl_native_lru *cache, const rl_cache_key *protected_keys, uint32_t protected_count) {
    int victim = -1;
    uint64_t oldest = UINT64_MAX;
    for (uint32_t i = 0; i < cache->capacity; ++i) {
        rl_cache_entry *entry = &cache->entries[i];
        if (!entry->valid) continue;
        if (entry->inflight || selected_contains(protected_keys, protected_count, entry->key)) {
            cache->blocked_victims++;
            continue;
        }
        if (entry->stamp < oldest) {
            oldest = entry->stamp;
            victim = (int)i;
        }
    }
    return victim;
}

int rl_native_lru_acquire_many(
        rl_native_lru *cache,
        const rl_cache_key *keys,
        uint32_t count,
        uint32_t *slot_ids,
        char *error,
        size_t error_cap) {
    if (!cache || !cache->entries || !keys || !slot_ids || !count) {
        set_error(error, error_cap, "invalid native LRU acquire_many arguments");
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

    /* Resolve hits first so current selected residents are protected before any miss can evict. */
    uint8_t *resolved = (uint8_t *)calloc(count, 1);
    if (!resolved) {
        set_error(error, error_cap, "out of memory resolving top-k LRU selection");
        return 0;
    }
    for (uint32_t i = 0; i < count; ++i) {
        const int index = index_for_key(cache, keys[i]);
        if (index < 0) continue;
        rl_cache_entry *entry = &cache->entries[index];
        entry->stamp = ++cache->clock;
        slot_ids[i] = entry->slot_id;
        cache->hits++;
        resolved[i] = 1;
    }

    for (uint32_t i = 0; i < count; ++i) {
        if (resolved[i]) continue;
        cache->misses++;
        int index = first_free_entry(cache);
        uint32_t slot = 0;
        if (index >= 0) {
            slot = (uint32_t)index;
            cache->resident++;
        } else {
            index = victim_entry(cache, keys, count);
            if (index < 0) {
                free(resolved);
                set_error(error, error_cap, "all native LRU slots are protected or in-flight");
                return 0;
            }
            slot = cache->entries[index].slot_id;
            cache->evictions++;
        }
        rl_cache_entry *entry = &cache->entries[index];
        entry->key = keys[i];
        entry->slot_id = slot;
        entry->stamp = ++cache->clock;
        entry->inflight = 0;
        entry->valid = 1;
        slot_ids[i] = slot;
        resolved[i] = 1;
    }
    free(resolved);
    if (error && error_cap) error[0] = '\0';
    return 1;
}
