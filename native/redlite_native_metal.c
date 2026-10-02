#include "redlite_native_metal.h"

#include "redlite_native_cache.h"
#include "redlite_native_tables.h"
#include "redmetal_topk.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef __APPLE__
#include <dispatch/dispatch.h>
#endif

#define RL_NATIVE_TOPK_MAX 512u

struct rl_native_metal_runtime {
    redmetal_topk_pool_t pool;
    rl_native_lru lru;
    uint64_t expert_loads;
    uint64_t slot_bytes;
    double prep_lru_ms, prep_load_ms, prep_commit_ms;   /* wall-clock profile of prepare_topk */
    int residency;                                       /* GPU residency table enabled */
};

static double now_ms_local(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

void rl_native_metal_prepare_profile(const rl_native_metal_runtime *runtime, double *lru_ms, double *load_ms, double *commit_ms) {
    if (lru_ms) *lru_ms = runtime ? runtime->prep_lru_ms : 0.0;
    if (load_ms) *load_ms = runtime ? runtime->prep_load_ms : 0.0;
    if (commit_ms) *commit_ms = runtime ? runtime->prep_commit_ms : 0.0;
}

/* dev31: routed quant types with Metal expert kernels */
static int routed_type_ok(uint32_t t) { return t == 17u || t == 29u || t == 18u || t == 21u || t == 12u; }   /* dev36: Q4_K (IQ3_M down) */

static void set_error(char *error, size_t cap, const char *message) {
    if (error && cap) snprintf(error, cap, "%s", message ? message : "unknown native Metal error");
}

static void set_metal_error(char *error, size_t cap, const char *prefix) {
    if (!error || !cap) return;
    const char *native = redmetal_topk_last_error();
    snprintf(error, cap, "%s: %s", prefix ? prefix : "Red Metal error",
             native ? native : "unknown Red Metal error");
}

rl_native_metal_runtime *rl_native_metal_create(
        const char *model_path,
        const rl_expert_map *map,
        uint64_t budget_bytes,
        uint32_t slots_per_slab,
        char *error,
        size_t error_cap) {
    if (!model_path || !map || !map->all_slice_safe || !map->max_expert_triplet_bytes || !budget_bytes) {
        set_error(error, error_cap, "invalid native Metal runtime arguments");
        return NULL;
    }
    if (redmetal_topk_abi_version() != 1u) {
        set_error(error, error_cap, "unsupported Red Metal top-k ABI");
        return NULL;
    }

    rl_native_quant_tables tables;
    if (!rl_native_quant_tables_init(&tables, error, error_cap)) return NULL;

    redmetal_topk_pool_t pool = redmetal_topk_pool_create(
        model_path,
        budget_bytes,
        map->max_expert_triplet_bytes,
        slots_per_slab ? slots_per_slab : 64u,
        tables.iq2_xs,
        RL_IQ2_XS_GRID_COUNT,
        tables.iq1_m,
        RL_IQ1_M_GRID_COUNT);
    rl_native_quant_tables_free(&tables);
    if (!pool) {
        set_metal_error(error, error_cap, "failed to create native top-k Metal pool");
        return NULL;
    }

    /* dev37: one slot size class per distinct 4 KiB-aligned expert triplet size, the same number of slots for every
     * layer (budget / sum of the layers' slot sizes, at most expert_count). Uniform slots of the largest size wasted
     * 24% of an IQ2_XXS cache on padding (37 of 48 layers hold smaller IQ1_M experts). Kept uniform when there is
     * one size, when a layer would get fewer than 16 slots (small stage-tool caches), or with RL_POOL_CLASSES=0. */
    uint8_t layer_class[RL_LRU_MAX_LAYERS];
    uint32_t class_cap[RL_LRU_MAX_CLASSES] = {0};
    uint32_t n_class = 0;
    {
        uint64_t size_of[RL_LRU_MAX_CLASSES] = {0}, layer_bytes[RL_LRU_MAX_LAYERS] = {0}, sum = 0;
        const char *env = getenv("RL_POOL_CLASSES");
        int ok = (!env || atoi(env) != 0) && map->layer_count && map->layer_count <= RL_LRU_MAX_LAYERS && map->expert_count;
        for (uint32_t i = 0; ok && i < map->routed_tensor_count; ++i) {
            if (map->routed[i].layer >= RL_LRU_MAX_LAYERS) ok = 0;
            else layer_bytes[map->routed[i].layer] += map->routed[i].expert_stride_bytes;
        }
        for (uint32_t l = 0; ok && l < map->layer_count; ++l) {
            const uint64_t b = (layer_bytes[l] + 4095u) & ~(uint64_t)4095u;
            if (!layer_bytes[l]) { ok = 0; break; }
            uint32_t c = 0;
            while (c < n_class && size_of[c] != b) ++c;
            if (c == n_class) { if (n_class == RL_LRU_MAX_CLASSES) { ok = 0; break; } size_of[n_class++] = b; }
            layer_class[l] = (uint8_t)c;
            sum += b;
        }
        uint64_t per_layer = ok && sum ? budget_bytes / sum : 0;
        if (per_layer > map->expert_count) per_layer = map->expert_count;
        /* a class must hold one layer's whole chunk (up to expert_count) while the next layer's is prefetched */
        uint32_t layers_in_class[RL_LRU_MAX_CLASSES] = {0};
        for (uint32_t l = 0; ok && l < map->layer_count; ++l) layers_in_class[layer_class[l]]++;
        for (uint32_t c = 0; ok && c < n_class; ++c)
            if ((uint64_t)per_layer * layers_in_class[c] < 2ull * map->expert_count + 64u) ok = 0;
        if (!ok || n_class < 2u || per_layer < 16u) {
            n_class = 0;
        } else {
            for (uint32_t l = 0; l < map->layer_count; ++l) class_cap[layer_class[l]] += (uint32_t)per_layer;
            if (!redmetal_topk_pool_set_classes(pool, n_class, size_of, class_cap)) {
                redmetal_topk_pool_destroy(pool);
                set_metal_error(error, error_cap, "failed to set native top-k slot classes");
                return NULL;
            }
        }
    }

    const uint32_t capacity = redmetal_topk_pool_capacity(pool);
    if (!capacity) {
        redmetal_topk_pool_destroy(pool);
        set_error(error, error_cap, "native top-k Metal pool has zero capacity");
        return NULL;
    }

    rl_native_metal_runtime *runtime = (rl_native_metal_runtime *)calloc(1, sizeof(*runtime));
    if (!runtime) {
        redmetal_topk_pool_destroy(pool);
        set_error(error, error_cap, "out of memory creating native Metal runtime");
        return NULL;
    }
    runtime->pool = pool;
    runtime->slot_bytes = map->max_expert_triplet_bytes;
    if (!rl_native_lru_init(&runtime->lru, capacity)) {
        redmetal_topk_pool_destroy(pool);
        free(runtime);
        set_error(error, error_cap, "failed to create native Metal LRU");
        return NULL;
    }
    if (n_class && !rl_native_lru_set_classes(&runtime->lru, n_class, class_cap, layer_class, map->layer_count)) {
        rl_native_lru_free(&runtime->lru);
        redmetal_topk_pool_destroy(pool);
        free(runtime);
        set_error(error, error_cap, "failed to set native Metal LRU classes");
        return NULL;
    }
    if (error && error_cap) error[0] = '\0';
    return runtime;
}

void rl_native_metal_destroy(rl_native_metal_runtime *runtime) {
    if (!runtime) return;
    rl_native_lru_free(&runtime->lru);
    if (runtime->pool) redmetal_topk_pool_destroy(runtime->pool);
    memset(runtime, 0, sizeof(*runtime));
    free(runtime);
}

/* mirror one committed transaction into the GPU residency table */
static void residency_apply(rl_native_metal_runtime *runtime, const rl_cache_transaction *tx, uint32_t layer, const uint32_t *slots) {
    if (!runtime->residency) return;
    for (uint32_t i = 0; i < tx->count; ++i) {
        const rl_cache_reservation *r = &tx->items[i];
        if (r->hit) continue;
        if (r->evicts) redmetal_topk_pool_residency_clear(runtime->pool, r->evicted_key.layer, r->evicted_key.expert);
        redmetal_topk_pool_residency_set(runtime->pool, layer, r->key.expert, slots[i]);
    }
}

static void residency_abort(rl_native_metal_runtime *runtime, const rl_cache_transaction *tx) {
    if (!runtime->residency) return;
    for (uint32_t i = 0; i < tx->count; ++i) {
        const rl_cache_reservation *r = &tx->items[i];
        if (!r->hit && r->evicts) redmetal_topk_pool_residency_clear(runtime->pool, r->evicted_key.layer, r->evicted_key.expert);
    }
}

int rl_native_metal_residency_enable(rl_native_metal_runtime *runtime, uint32_t layers, uint32_t experts, char *error, size_t error_cap) {
    if (!runtime || !runtime->pool) { set_error(error, error_cap, "invalid runtime"); return 0; }
    if (runtime->lru.resident) { set_error(error, error_cap, "residency table must be enabled before any expert is loaded"); return 0; }
    if (!redmetal_topk_pool_residency_table_init(runtime->pool, layers, experts)) { set_metal_error(error, error_cap, "residency table"); return 0; }
    runtime->residency = 1;
    return 1;
}

void *rl_native_metal_residency_table(rl_native_metal_runtime *runtime) {
    return runtime && runtime->residency ? redmetal_topk_pool_residency_table(runtime->pool) : NULL;
}

int rl_native_metal_touch_resident(rl_native_metal_runtime *runtime, uint32_t layer, const uint32_t *expert_ids, uint32_t count, char *error, size_t error_cap) {
    if (!runtime || !expert_ids || !count || count > RL_NATIVE_TOPK_MAX) { set_error(error, error_cap, "invalid touch request"); return 0; }
    rl_cache_key keys[RL_NATIVE_TOPK_MAX];
    for (uint32_t i = 0; i < count; ++i) { keys[i].layer = layer; keys[i].expert = expert_ids[i]; }
    rl_cache_transaction tx = {0};
    if (!rl_native_lru_prepare_many(&runtime->lru, keys, count, &tx, error, error_cap)) return 0;
    for (uint32_t i = 0; i < count; ++i) {
        if (!tx.items[i].hit) {
            rl_native_lru_abort(&runtime->lru, &tx);
            residency_abort(runtime, &tx);
            rl_native_lru_transaction_free(&tx);
            set_error(error, error_cap, "GPU-routed expert is not resident");
            return 0;
        }
    }
    uint32_t slots[RL_NATIVE_TOPK_MAX];
    const int ok = rl_native_lru_commit(&runtime->lru, &tx, slots, error, error_cap);
    rl_native_lru_transaction_free(&tx);
    return ok;
}

int rl_native_metal_execute_topk(
        rl_native_metal_runtime *runtime,
        const rl_expert_map *map,
        uint32_t layer,
        const uint32_t *expert_ids,
        const float *router_weights,
        uint32_t top_k,
        uint32_t output_row_start,
        uint32_t output_row_count,
        const float *input,
        uint32_t input_count,
        float *output,
        uint32_t output_count,
        rl_native_metal_telemetry *telemetry,
        char *error,
        size_t error_cap) {
    if (!runtime || !runtime->pool || !map || !expert_ids || !router_weights || !input || !output ||
        !top_k || top_k > RL_NATIVE_TOPK_MAX) {
        set_error(error, error_cap, "invalid native top-k execution request");
        return 0;
    }

    rl_native_layer_info info;
    if (!rl_native_get_layer_info(map, layer, &info, error, error_cap)) return 0;
    if (!routed_type_ok(info.ggml_type) || !routed_type_ok(info.down_type)) {
        set_error(error, error_cap, "native Metal supports routed IQ2_XS, IQ1_M, IQ3_XXS, IQ3_S and Q4_K only");
        return 0;
    }
    if (input_count != info.hidden_size || !output_row_count ||
        output_row_start >= info.hidden_size || output_row_count > info.hidden_size - output_row_start ||
        output_count < output_row_count) {
        set_error(error, error_cap, "invalid native top-k input/output dimensions");
        return 0;
    }
    if (top_k > runtime->lru.capacity) {
        set_error(error, error_cap, "native Metal cache cannot hold the complete top-k selection");
        return 0;
    }

    rl_cache_key keys[RL_NATIVE_TOPK_MAX];
    uint32_t slots[RL_NATIVE_TOPK_MAX];
    rl_expert_layout layouts[RL_NATIVE_TOPK_MAX];
    uint64_t gate_bytes[RL_NATIVE_TOPK_MAX];
    uint64_t up_bytes[RL_NATIVE_TOPK_MAX];
    uint64_t down_bytes[RL_NATIVE_TOPK_MAX];

    for (uint32_t i = 0; i < top_k; ++i) {
        if (expert_ids[i] >= map->expert_count) {
            set_error(error, error_cap, "native top-k expert id is out of range");
            return 0;
        }
        for (uint32_t j = 0; j < i; ++j) {
            if (expert_ids[i] == expert_ids[j]) {
                set_error(error, error_cap, "native top-k expert ids must be unique");
                return 0;
            }
        }
        keys[i].layer = layer;
        keys[i].expert = expert_ids[i];
        if (!rl_native_expert_layout(map, layer, expert_ids[i], &layouts[i], error, error_cap)) return 0;
        if (layouts[i].ggml_type != info.ggml_type || layouts[i].down_type != info.down_type || layouts[i].total_bytes > runtime->slot_bytes) {
            set_error(error, error_cap, "native expert layout is incompatible with the layer/pool");
            return 0;
        }
        gate_bytes[i] = layouts[i].gate_bytes;
        up_bytes[i] = layouts[i].up_bytes;
        down_bytes[i] = layouts[i].down_bytes;
    }

    rl_cache_transaction transaction = {0};
    if (!rl_native_lru_prepare_many(&runtime->lru, keys, top_k, &transaction, error, error_cap)) return 0;

    uint32_t miss_index[RL_NATIVE_TOPK_MAX];
    uint32_t miss_count = 0;
    for (uint32_t i = 0; i < top_k; ++i) {
        const rl_cache_reservation *reservation = &transaction.items[i];
        slots[i] = reservation->slot_id;
        if (!reservation->hit) miss_index[miss_count++] = i;
    }
    /* Load every miss into its reserved slot. Slots are distinct, so the
     * positional reads are independent and run concurrently (GCD); the
     * transaction is committed only if all of them succeed. */
    int load_ok[RL_NATIVE_TOPK_MAX];
    for (uint32_t k = 0; k < miss_count; ++k) load_ok[k] = 0;
    if (miss_count) {
        rl_expert_layout *lay = layouts;
        uint32_t *slot_ids = slots;
        uint32_t *mi = miss_index;
        int *lo = load_ok;
        redmetal_topk_pool_t pool = runtime->pool;
#ifdef __APPLE__
        dispatch_apply(miss_count, DISPATCH_APPLY_AUTO, ^(size_t k) {
            const uint32_t i = mi[k];
            uint64_t bytes_read = 0;
            double load_ms = 0.0;
            lo[k] = redmetal_topk_pool_load_expert(pool, slot_ids[i],
                lay[i].gate_offset, lay[i].gate_bytes, lay[i].up_offset, lay[i].up_bytes,
                lay[i].down_offset, lay[i].down_bytes, &bytes_read, &load_ms);
        });
#else
        for (uint32_t k = 0; k < miss_count; ++k) {
            const uint32_t i = mi[k];
            uint64_t bytes_read = 0;
            double load_ms = 0.0;
            lo[k] = redmetal_topk_pool_load_expert(pool, slot_ids[i],
                lay[i].gate_offset, lay[i].gate_bytes, lay[i].up_offset, lay[i].up_bytes,
                lay[i].down_offset, lay[i].down_bytes, &bytes_read, &load_ms);
        }
#endif
    }
    for (uint32_t k = 0; k < miss_count; ++k) {
        if (!load_ok[k]) {
            rl_native_lru_abort(&runtime->lru, &transaction);
            residency_abort(runtime, &transaction);
            rl_native_lru_transaction_free(&transaction);
            set_metal_error(error, error_cap, "native expert load failed; LRU transaction aborted");
            return 0;
        }
        runtime->expert_loads++;
    }

    if (!rl_native_lru_commit(&runtime->lru, &transaction, slots, error, error_cap)) {
        rl_native_lru_abort(&runtime->lru, &transaction);
        residency_abort(runtime, &transaction);
        rl_native_lru_transaction_free(&transaction);
        return 0;
    }
    residency_apply(runtime, &transaction, layer, slots);
    rl_native_lru_transaction_free(&transaction);

    for (uint32_t i = 0; i < top_k; ++i) {
        if (redmetal_topk_pool_slot_inflight(runtime->pool, slots[i])) {
            set_error(error, error_cap, "native top-k slot unexpectedly in-flight before dispatch");
            return 0;
        }
    }

    const uint64_t bytes_before = redmetal_topk_pool_bytes_read(runtime->pool);
    const uint64_t calls_before = redmetal_topk_pool_read_calls(runtime->pool);
    double gpu_ms = 0.0;
    if (!redmetal_topk_pool_execute(
            runtime->pool,
            slots,
            gate_bytes,
            up_bytes,
            down_bytes,
            router_weights,
            top_k,
            rl_native_expert_type_word(info.ggml_type, info.down_type),
            info.hidden_size,
            info.ffn_size,
            output_row_start,
            output_row_count,
            input,
            input_count,
            output,
            output_count,
            &gpu_ms)) {
        set_metal_error(error, error_cap, "native top-k Metal execution failed");
        return 0;
    }
    const uint64_t bytes_after = redmetal_topk_pool_bytes_read(runtime->pool);
    const uint64_t calls_after = redmetal_topk_pool_read_calls(runtime->pool);
    if (bytes_after != bytes_before || calls_after != calls_before) {
        set_error(error, error_cap, "native top-k execution performed unexpected SSD reads");
        return 0;
    }

    for (uint32_t i = 0; i < top_k; ++i) {
        if (redmetal_topk_pool_slot_inflight(runtime->pool, slots[i])) {
            set_error(error, error_cap, "native top-k slot remained in-flight after synchronous wait");
            return 0;
        }
    }

    if (telemetry) {
        memset(telemetry, 0, sizeof(*telemetry));
        telemetry->cache_hits = runtime->lru.hits;
        telemetry->cache_misses = runtime->lru.misses;
        telemetry->cache_evictions = runtime->lru.evictions;
        telemetry->expert_loads = runtime->expert_loads;
        telemetry->bytes_read_total = bytes_after;
        telemetry->read_calls_total = calls_after;
        telemetry->bytes_read_during_execute = bytes_after - bytes_before;
        telemetry->read_calls_during_execute = calls_after - calls_before;
        telemetry->resident_slots = runtime->lru.resident;
        telemetry->slot_capacity = runtime->lru.capacity;
        telemetry->slab_count = redmetal_topk_pool_slab_count(runtime->pool);
        telemetry->allocated_bytes = redmetal_topk_pool_allocated_bytes(runtime->pool);
        telemetry->gpu_ms = gpu_ms;
        telemetry->read_ms_total = redmetal_topk_pool_read_ms(runtime->pool);
    }
    if (error && error_cap) error[0] = '\0';
    return 1;
}

/* ---- deferred plan API ---- */

double rl_native_metal_read_ms(const rl_native_metal_runtime *runtime) {
    return runtime && runtime->pool ? redmetal_topk_pool_read_ms(runtime->pool) : 0.0;
}


int rl_native_metal_prepare_topk(
        rl_native_metal_runtime *runtime,
        const rl_expert_map *map,
        uint32_t layer,
        const uint32_t *expert_ids,
        const float *router_weights,
        uint32_t top_k,
        rl_native_topk_plan *plan,
        char *error,
        size_t error_cap) {
    if (!runtime || !runtime->pool || !map || !expert_ids || !router_weights || !plan || !top_k || top_k > RL_NATIVE_TOPK_MAX) {
        set_error(error, error_cap, "invalid native top-k prepare request");
        return 0;
    }
    memset(plan, 0, sizeof(*plan));
    rl_native_layer_info info;
    if (!rl_native_get_layer_info(map, layer, &info, error, error_cap)) return 0;
    if (!routed_type_ok(info.ggml_type) || !routed_type_ok(info.down_type)) {
        set_error(error, error_cap, "native Metal supports routed IQ2_XS, IQ1_M, IQ3_XXS, IQ3_S and Q4_K only");
        return 0;
    }
    if (top_k > runtime->lru.capacity) {
        set_error(error, error_cap, "native Metal cache cannot hold the complete top-k selection");
        return 0;
    }
    rl_cache_key keys[RL_NATIVE_TOPK_MAX];
    rl_expert_layout layouts[RL_NATIVE_TOPK_MAX];
    for (uint32_t i = 0; i < top_k; ++i) {
        if (expert_ids[i] >= map->expert_count) { set_error(error, error_cap, "native top-k expert id is out of range"); return 0; }
        for (uint32_t j = 0; j < i; ++j) if (expert_ids[i] == expert_ids[j]) { set_error(error, error_cap, "native top-k expert ids must be unique"); return 0; }
        keys[i].layer = layer;
        keys[i].expert = expert_ids[i];
        if (!rl_native_expert_layout(map, layer, expert_ids[i], &layouts[i], error, error_cap)) return 0;
        if (layouts[i].ggml_type != info.ggml_type || layouts[i].down_type != info.down_type || layouts[i].total_bytes > runtime->slot_bytes) {
            set_error(error, error_cap, "native expert layout is incompatible with the layer/pool");
            return 0;
        }
        plan->gate_bytes[i] = layouts[i].gate_bytes;
        plan->up_bytes[i] = layouts[i].up_bytes;
        plan->down_bytes[i] = layouts[i].down_bytes;
        plan->weights[i] = router_weights[i];
    }
    rl_cache_transaction transaction = {0};
    const double t_lru = now_ms_local();
    if (!rl_native_lru_prepare_many(&runtime->lru, keys, top_k, &transaction, error, error_cap)) return 0;
    runtime->prep_lru_ms += now_ms_local() - t_lru;
    const double t_load = now_ms_local();
    uint32_t miss_index[RL_NATIVE_TOPK_MAX];
    uint32_t miss_count = 0;
    for (uint32_t i = 0; i < top_k; ++i) {
        plan->slots[i] = transaction.items[i].slot_id;
        if (!transaction.items[i].hit) miss_index[miss_count++] = i;
    }
    int load_ok[RL_NATIVE_TOPK_MAX];
    for (uint32_t k = 0; k < miss_count; ++k) load_ok[k] = 0;
    if (miss_count) {
        rl_expert_layout *lay = layouts;
        uint32_t *slot_ids = plan->slots;
        uint32_t *mi = miss_index;
        int *lo = load_ok;
        redmetal_topk_pool_t pool = runtime->pool;
#ifdef __APPLE__
        dispatch_apply(miss_count, DISPATCH_APPLY_AUTO, ^(size_t k) {
            const uint32_t i = mi[k];
            uint64_t bytes_read = 0;
            double load_ms = 0.0;
            lo[k] = redmetal_topk_pool_load_expert(pool, slot_ids[i],
                lay[i].gate_offset, lay[i].gate_bytes, lay[i].up_offset, lay[i].up_bytes,
                lay[i].down_offset, lay[i].down_bytes, &bytes_read, &load_ms);
        });
#else
        for (uint32_t k = 0; k < miss_count; ++k) {
            const uint32_t i = mi[k];
            uint64_t bytes_read = 0;
            double load_ms = 0.0;
            lo[k] = redmetal_topk_pool_load_expert(pool, slot_ids[i],
                lay[i].gate_offset, lay[i].gate_bytes, lay[i].up_offset, lay[i].up_bytes,
                lay[i].down_offset, lay[i].down_bytes, &bytes_read, &load_ms);
        }
#endif
    }
    runtime->prep_load_ms += now_ms_local() - t_load;
    const double t_commit = now_ms_local();
    for (uint32_t k = 0; k < miss_count; ++k) {
        if (!load_ok[k]) {
            rl_native_lru_abort(&runtime->lru, &transaction);
            residency_abort(runtime, &transaction);
            rl_native_lru_transaction_free(&transaction);
            set_metal_error(error, error_cap, "native expert load failed; LRU transaction aborted");
            return 0;
        }
        runtime->expert_loads++;
    }
    if (!rl_native_lru_commit(&runtime->lru, &transaction, plan->slots, error, error_cap)) {
        rl_native_lru_abort(&runtime->lru, &transaction);
        residency_abort(runtime, &transaction);
        rl_native_lru_transaction_free(&transaction);
        return 0;
    }
    residency_apply(runtime, &transaction, layer, plan->slots);
    rl_native_lru_transaction_free(&transaction);
    runtime->prep_commit_ms += now_ms_local() - t_commit;
    for (uint32_t i = 0; i < top_k; ++i) {
        if (redmetal_topk_pool_slot_inflight(runtime->pool, plan->slots[i])) {
            set_error(error, error_cap, "native top-k slot unexpectedly in-flight before encode");
            return 0;
        }
    }
    plan->layer = layer;
    plan->top_k = top_k;
    plan->ggml_type = rl_native_expert_type_word(info.ggml_type, info.down_type);
    plan->hidden = info.hidden_size;
    plan->ffn = info.ffn_size;
    if (error && error_cap) error[0] = '\0';
    return 1;
}

int rl_native_metal_encode_topk(
        rl_native_metal_runtime *runtime,
        rl_native_topk_plan *plan,
        void *mtl_command_buffer,
        void *mtl_input_buffer,
        uint64_t input_offset,
        void *mtl_output_buffer,
        uint64_t output_offset,
        char *error,
        size_t error_cap) {
    if (!runtime || !plan || !plan->top_k || plan->active) { set_error(error, error_cap, "invalid native top-k encode request"); return 0; }
    plan->bytes_read_at_encode = redmetal_topk_pool_bytes_read(runtime->pool);
    plan->calls_at_encode = redmetal_topk_pool_read_calls(runtime->pool);
    for (uint32_t i = 0; i < plan->top_k; ++i) plan->slot_generation[i] = redmetal_topk_pool_slot_generation(runtime->pool, plan->slots[i]);
    if (!redmetal_topk_pool_encode(runtime->pool, mtl_command_buffer, plan->slots, plan->gate_bytes, plan->up_bytes,
            plan->down_bytes, plan->weights, plan->top_k, plan->ggml_type, plan->hidden, plan->ffn, 0u, plan->hidden,
            mtl_input_buffer, input_offset, mtl_output_buffer, output_offset)) {
        set_metal_error(error, error_cap, "native top-k Metal encode failed");
        return 0;
    }
    plan->active = 1;
    if (error && error_cap) error[0] = '\0';
    return 1;
}

int rl_native_metal_encode_topk_batched(
        rl_native_metal_runtime *runtime,
        rl_native_topk_plan *plan,
        void *mtl_command_buffer,
        uint32_t tok_first,
        uint32_t ntok,
        uint32_t top_k,
        const uint32_t *pair_token,
        const float *pair_weight,
        const uint32_t *expert_start,
        const uint32_t *tok_pair,
        void *mtl_input_buffer,
        uint64_t input_offset,
        void *mtl_output_buffer,
        uint64_t output_offset,
        char *error,
        size_t error_cap) {
    if (!runtime || !plan || !plan->top_k || plan->active) { set_error(error, error_cap, "invalid native batched top-k encode request"); return 0; }
    plan->bytes_read_at_encode = redmetal_topk_pool_bytes_read(runtime->pool);
    plan->calls_at_encode = redmetal_topk_pool_read_calls(runtime->pool);
    for (uint32_t i = 0; i < plan->top_k; ++i) plan->slot_generation[i] = redmetal_topk_pool_slot_generation(runtime->pool, plan->slots[i]);
    if (!redmetal_topk_pool_encode_batched(runtime->pool, mtl_command_buffer, plan->slots, plan->gate_bytes, plan->up_bytes,
            plan->down_bytes, plan->top_k, plan->ggml_type, plan->hidden, plan->ffn, tok_first, ntok, top_k, ntok * top_k,
            pair_token, pair_weight, expert_start, tok_pair, mtl_input_buffer, input_offset, mtl_output_buffer, output_offset)) {
        set_metal_error(error, error_cap, "native batched top-k Metal encode failed");
        return 0;
    }
    plan->active = 1;
    if (error && error_cap) error[0] = '\0';
    return 1;
}

void *rl_native_metal_pool_handle(rl_native_metal_runtime *runtime) { return runtime ? (void *)runtime->pool : NULL; }

uint32_t rl_native_metal_slot_capacity(const rl_native_metal_runtime *runtime) { return runtime ? runtime->lru.capacity : 0u; }

int rl_native_metal_is_cached(const rl_native_metal_runtime *runtime, uint32_t layer, uint32_t expert) {
    const rl_cache_key key = {layer, expert};
    return runtime ? rl_native_lru_lookup(&runtime->lru, key, NULL) : 0;
}

uint32_t rl_native_metal_layer_class(const rl_native_metal_runtime *runtime, uint32_t layer) {
    return runtime && layer < RL_LRU_MAX_LAYERS ? runtime->lru.layer_class[layer] : 0u;
}

uint32_t rl_native_metal_layer_capacity(const rl_native_metal_runtime *runtime, uint32_t layer) {
    if (!runtime) return 0u;
    const uint32_t c = rl_native_metal_layer_class(runtime, layer);
    return runtime->lru.class_first[c + 1u] - runtime->lru.class_first[c];
}

int rl_native_metal_release_topk(
        rl_native_metal_runtime *runtime,
        rl_native_topk_plan *plan,
        rl_native_metal_telemetry *telemetry,
        char *error,
        size_t error_cap) {
    if (!runtime || !plan) { set_error(error, error_cap, "invalid native top-k release request"); return 0; }
    const uint64_t bytes_after = redmetal_topk_pool_bytes_read(runtime->pool);
    const uint64_t calls_after = redmetal_topk_pool_read_calls(runtime->pool);
    if (plan->active) {
        int rewritten = 0;
        for (uint32_t i = 0; i < plan->top_k; ++i)
            if (redmetal_topk_pool_slot_generation(runtime->pool, plan->slots[i]) != plan->slot_generation[i]) rewritten = 1;
        redmetal_topk_pool_release(runtime->pool, plan->slots, plan->top_k);
        plan->active = 0;
        if (rewritten) {
            set_error(error, error_cap, "native top-k experts were overwritten by SSD reads while in flight");
            return 0;
        }
    }
    /* an inactive plan (GPU-routed token, preload) still reports the counters */
    if (telemetry) {
        memset(telemetry, 0, sizeof(*telemetry));
        telemetry->cache_hits = runtime->lru.hits;
        telemetry->cache_misses = runtime->lru.misses;
        telemetry->cache_evictions = runtime->lru.evictions;
        telemetry->expert_loads = runtime->expert_loads;
        telemetry->bytes_read_total = bytes_after;
        telemetry->read_calls_total = calls_after;
        telemetry->resident_slots = runtime->lru.resident;
        telemetry->slot_capacity = runtime->lru.capacity;
        telemetry->slab_count = redmetal_topk_pool_slab_count(runtime->pool);
        telemetry->allocated_bytes = redmetal_topk_pool_allocated_bytes(runtime->pool);
        telemetry->read_ms_total = redmetal_topk_pool_read_ms(runtime->pool);
    }
    if (error && error_cap) error[0] = '\0';
    return 1;
}
