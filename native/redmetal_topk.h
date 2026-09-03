#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void * redmetal_topk_pool_t;

uint32_t redmetal_topk_abi_version(void);
const char * redmetal_topk_last_error(void);

redmetal_topk_pool_t redmetal_topk_pool_create(
    const char *model_path,
    uint64_t budget_bytes,
    uint64_t slot_bytes,
    uint32_t slots_per_slab,
    const int8_t *iq2_xs_grid,
    uint32_t iq2_xs_grid_count,
    const int8_t *iq1_m_grid,
    uint32_t iq1_m_grid_count);

void redmetal_topk_pool_destroy(redmetal_topk_pool_t pool);
uint32_t redmetal_topk_pool_capacity(redmetal_topk_pool_t pool);
uint32_t redmetal_topk_pool_slab_count(redmetal_topk_pool_t pool);
uint64_t redmetal_topk_pool_allocated_bytes(redmetal_topk_pool_t pool);
uint64_t redmetal_topk_pool_bytes_read(redmetal_topk_pool_t pool);
uint64_t redmetal_topk_pool_read_calls(redmetal_topk_pool_t pool);
double redmetal_topk_pool_read_ms(redmetal_topk_pool_t pool);
int redmetal_topk_pool_slot_inflight(redmetal_topk_pool_t pool, uint32_t slot_id);

int redmetal_topk_pool_load_expert(
    redmetal_topk_pool_t pool,
    uint32_t slot_id,
    uint64_t gate_file_offset,
    uint64_t gate_bytes,
    uint64_t up_file_offset,
    uint64_t up_bytes,
    uint64_t down_file_offset,
    uint64_t down_bytes,
    uint64_t *bytes_read,
    double *elapsed_ms);

int redmetal_topk_pool_slot_addresses(
    redmetal_topk_pool_t pool,
    uint32_t slot_id,
    uint64_t gate_bytes,
    uint64_t up_bytes,
    uint64_t down_bytes,
    uint64_t *gate_address,
    uint64_t *up_address,
    uint64_t *down_address);

int redmetal_topk_pool_execute(
    redmetal_topk_pool_t pool,
    const uint32_t *slot_ids,
    const uint64_t *gate_bytes,
    const uint64_t *up_bytes,
    const uint64_t *down_bytes,
    const float *router_weights,
    uint32_t top_k,
    uint32_t ggml_type,
    uint32_t hidden_size,
    uint32_t ffn_size,
    uint32_t output_row_start,
    uint32_t output_row_count,
    const float *input,
    uint32_t input_count,
    float *output,
    uint32_t output_count,
    double *elapsed_ms);

/*
 * Encode the same three-dispatch execution into an external Metal command
 * buffer (opaque id<MTLCommandBuffer>) reading the input vector from an
 * external MTLBuffer and writing the routed output into another one. Marks the
 * slots in-flight; the caller must call redmetal_topk_pool_release after the
 * command buffer completed.
 */
int redmetal_topk_pool_encode(
    redmetal_topk_pool_t pool,
    void *mtl_command_buffer,
    const uint32_t *slot_ids,
    const uint64_t *gate_bytes,
    const uint64_t *up_bytes,
    const uint64_t *down_bytes,
    const float *router_weights,
    uint32_t top_k,
    uint32_t ggml_type,
    uint32_t hidden_size,
    uint32_t ffn_size,
    uint32_t output_row_start,
    uint32_t output_row_count,
    void *mtl_input_buffer,
    uint64_t input_offset,
    void *mtl_output_buffer,
    uint64_t output_offset);

void redmetal_topk_pool_release(redmetal_topk_pool_t pool, const uint32_t *slot_ids, uint32_t top_k);

#ifdef __cplusplus
}
#endif
