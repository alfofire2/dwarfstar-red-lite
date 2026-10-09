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
 *              verify that none of the plan's slots was rewritten while in flight
 *              (dev23: per-slot load generations; loads into other slots, e.g. a
 *              prefetch, are allowed while the plan is in flight)
 */
/* dev31: routed-expert "type word" passed to the Metal expert kernels: gate/up type in bits 0-7 and, when the down
 * projection has another type (IQ3_XXS GGUF: IQ3_S in some layers), the down type in bits 8-15 */
static inline uint32_t rl_native_expert_type_word(uint32_t gate_up_type, uint32_t down_type) {
    return gate_up_type | (down_type != gate_up_type ? (down_type << 8) : 0u);
}

typedef struct {
    uint32_t layer;
    uint32_t top_k;
    uint32_t ggml_type;   /* type word (rl_native_expert_type_word) */
    uint32_t hidden;
    uint32_t ffn;
    uint32_t slots[512];
    uint64_t gate_bytes[512];
    uint64_t up_bytes[512];
    uint64_t down_bytes[512];
    float weights[512];
    uint64_t bytes_read_at_encode;
    uint64_t calls_at_encode;
    uint32_t slot_generation[512];   /* dev23: load generation of each plan slot at encode */
    int active;
} rl_native_topk_plan;

/* dev72: a plan for some of a prepared plan's experts: ids[0..k) (each one of union_ids[0..union->top_k), the ids the
 * union plan was prepared with) in this order with these weights, on the union plan's slots. Both rows of a 2-row verify
 * with a bounded cache are encoded from one prepare of their union; release each encoded subset, not the union. */
int rl_native_metal_plan_subset(const rl_native_topk_plan *union_plan, const uint32_t *union_ids, const uint32_t *ids,
                                const float *weights, uint32_t k, rl_native_topk_plan *out);

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

/* Batched prefill: encode the plan's experts (a union over several tokens) for the given pair layout (see redmetal_topk.h). */
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
    size_t error_cap);
/* Opaque pool handle for the mapped-expert prefill path (redmetal_topk_pool_encode_mapped). */
void *rl_native_metal_pool_handle(rl_native_metal_runtime *runtime);
/* Cumulative wall-clock profile of rl_native_metal_prepare_topk: LRU reservation, miss loads, commit. */
void rl_native_metal_prepare_profile(const rl_native_metal_runtime *runtime, double *lru_ms, double *load_ms, double *commit_ms);
/*
 * GPU-driven decode (dev21): a layers x experts table of slot GPU addresses
 * (0 = not resident) maintained as the LRU commits and evicts; the routing
 * kernel reads it. Enable once per runtime; returns the id<MTLBuffer>.
 */
int rl_native_metal_residency_enable(rl_native_metal_runtime *runtime, uint32_t layers, uint32_t experts, char *error, size_t error_cap);
void *rl_native_metal_residency_table(rl_native_metal_runtime *runtime);
/* Refresh the LRU stamps of experts a GPU-routed token used (all must be resident; returns 0 if any is not). */
int rl_native_metal_touch_resident(rl_native_metal_runtime *runtime, uint32_t layer, const uint32_t *expert_ids, uint32_t count, char *error, size_t error_cap);
/* Slot capacity of the bounded expert cache. */
uint32_t rl_native_metal_slot_capacity(const rl_native_metal_runtime *runtime);
/* dev43: slots a layer's experts can use (its size class; the whole pool without classes) and the class index */
uint32_t rl_native_metal_layer_capacity(const rl_native_metal_runtime *runtime, uint32_t layer);
uint32_t rl_native_metal_layer_class(const rl_native_metal_runtime *runtime, uint32_t layer);
/* dev46: 1 when (layer, expert) holds a slot of the cache */
int rl_native_metal_is_cached(const rl_native_metal_runtime *runtime, uint32_t layer, uint32_t expert);
int rl_native_metal_release_topk(
    rl_native_metal_runtime *runtime,
    rl_native_topk_plan *plan,
    rl_native_metal_telemetry *telemetry,
    char *error,
    size_t error_cap);

#ifdef __cplusplus
}
#endif
