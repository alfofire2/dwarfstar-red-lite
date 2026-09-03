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
    double read_ms_total;
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

/*
 * Deferred execution for the persistent engine:
 *   prepare  - LRU reservation + concurrent miss loads + commit (CPU side)
 *   encode   - encode the expert dispatches into an external command buffer
 *              reading/writing external MTLBuffers (opaque pointers)
 *   release  - after the command buffer completed: release in-flight slots and
 *              verify no SSD reads happened while the experts were in flight
 */
typedef struct {
    uint32_t layer;
    uint32_t top_k;
    uint32_t ggml_type;
    uint32_t hidden;
    uint32_t ffn;
    uint32_t slots[64];
    uint64_t gate_bytes[64];
    uint64_t up_bytes[64];
    uint64_t down_bytes[64];
    float weights[64];
    uint64_t bytes_read_at_encode;
    uint64_t calls_at_encode;
    int active;
} rl_native_topk_plan;

double rl_native_metal_read_ms(const rl_native_metal_runtime *runtime);

int rl_native_metal_prepare_topk(
    rl_native_metal_runtime *runtime,
    const rl_expert_map *map,
    uint32_t layer,
    const uint32_t *expert_ids,
    const float *router_weights,
    uint32_t top_k,
    rl_native_topk_plan *plan,
    char *error,
    size_t error_cap);

int rl_native_metal_encode_topk(
    rl_native_metal_runtime *runtime,
    rl_native_topk_plan *plan,
    void *mtl_command_buffer,
    void *mtl_input_buffer,
    uint64_t input_offset,
    void *mtl_output_buffer,
    uint64_t output_offset,
    char *error,
    size_t error_cap);

int rl_native_metal_release_topk(
    rl_native_metal_runtime *runtime,
    rl_native_topk_plan *plan,
    rl_native_metal_telemetry *telemetry,
    char *error,
    size_t error_cap);

#ifdef __cplusplus
}
#endif
