#include "redlite_native_metal.h"

#include "redlite_native_cache.h"
#include "redlite_native_tables.h"
#include "redmetal_topk.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define RL_NATIVE_TOPK_MAX 64u

struct rl_native_metal_runtime {
    redmetal_topk_pool_t pool;
    rl_native_lru lru;
    uint64_t expert_loads;
    uint64_t slot_bytes;
};

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
    if (info.ggml_type != 17u && info.ggml_type != 29u) {
        set_error(error, error_cap, "native Metal supports routed IQ2_XS and IQ1_M only");
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
        if (layouts[i].ggml_type != info.ggml_type || layouts[i].total_bytes > runtime->slot_bytes) {
            set_error(error, error_cap, "native expert layout is incompatible with the layer/pool");
            return 0;
        }
        gate_bytes[i] = layouts[i].gate_bytes;
        up_bytes[i] = layouts[i].up_bytes;
        down_bytes[i] = layouts[i].down_bytes;
    }

    rl_cache_transaction transaction = {0};
    if (!rl_native_lru_prepare_many(&runtime->lru, keys, top_k, &transaction, error, error_cap)) return 0;

    for (uint32_t i = 0; i < top_k; ++i) {
        const rl_cache_reservation *reservation = &transaction.items[i];
        slots[i] = reservation->slot_id;
        if (reservation->hit) continue;
        uint64_t bytes_read = 0;
        double load_ms = 0.0;
        if (!redmetal_topk_pool_load_expert(
                runtime->pool,
                slots[i],
                layouts[i].gate_offset, layouts[i].gate_bytes,
                layouts[i].up_offset, layouts[i].up_bytes,
                layouts[i].down_offset, layouts[i].down_bytes,
                &bytes_read, &load_ms)) {
            (void)bytes_read;
            (void)load_ms;
            rl_native_lru_abort(&runtime->lru, &transaction);
            rl_native_lru_transaction_free(&transaction);
            set_metal_error(error, error_cap, "native expert load failed; LRU transaction aborted");
            return 0;
        }
        runtime->expert_loads++;
    }

    if (!rl_native_lru_commit(&runtime->lru, &transaction, slots, error, error_cap)) {
        rl_native_lru_abort(&runtime->lru, &transaction);
        rl_native_lru_transaction_free(&transaction);
        return 0;
    }
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
            info.ggml_type,
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
    }
    if (error && error_cap) error[0] = '\0';
    return 1;
}
