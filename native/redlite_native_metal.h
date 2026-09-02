#pragma once

#include "redlite_native_gguf.h"
#include "redlite_native_model.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct rl_native_metal_runtime rl_native_metal_runtime;

typedef struct {
    uint64_t cache_hits;
    uint64_t cache_misses;
    uint64_t cache_evictions;
    uint64_t expert_loads;
    uint64_t bytes_read_total;
    uint64_t read_calls_total;
    uint64_t bytes_read_during_execute;
    uint64_t read_calls_during_execute;
    uint32_t resident_slots;
    uint32_t slot_capacity;
    uint32_t slab_count;
    uint64_t allocated_bytes;
    double gpu_ms;
} rl_native_metal_telemetry;

rl_native_metal_runtime *rl_native_metal_create(
    const char *model_path,
    const rl_expert_map *map,
    uint64_t budget_bytes,
    uint32_t slots_per_slab,
    char *error,
    size_t error_cap);

void rl_native_metal_destroy(rl_native_metal_runtime *runtime);

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
    size_t error_cap);

#ifdef __cplusplus
}
#endif
