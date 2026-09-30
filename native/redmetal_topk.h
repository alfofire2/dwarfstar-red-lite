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

/*
 * GPU-driven decode (dev21). The pool keeps a residency table of
 * layers x experts GPU slot addresses (0 = not resident) that a routing
 * kernel can read; the owner updates it as the LRU commits and evicts.
 */
int redmetal_topk_pool_residency_table_init(redmetal_topk_pool_t pool, uint32_t layers, uint32_t experts);
void *redmetal_topk_pool_residency_table(redmetal_topk_pool_t pool);   /* id<MTLBuffer>, uint64_t[layers][experts] */
int redmetal_topk_pool_residency_set(redmetal_topk_pool_t pool, uint32_t layer, uint32_t expert, uint32_t slot_id);
void redmetal_topk_pool_residency_clear(redmetal_topk_pool_t pool, uint32_t layer, uint32_t expert);
/* The MTLResidencySet holding every slab (macOS 15+; NULL otherwise). Attach it to the command queue that
 * executes GPU-driven or batched expert work; per-encoder useResource is then skipped for slabs. */
void *redmetal_topk_pool_slab_residency_set(redmetal_topk_pool_t pool);
/* Mark every slab resident for the given compute encoder (id<MTLComputeCommandEncoder>); no-op with a residency set. */
void redmetal_topk_pool_use_all_slabs(redmetal_topk_pool_t pool, void *mtl_compute_encoder);
/*
 * Encode the single-token three-dispatch execution reading the slot table
 * (uint64 slot base addresses, REDMETAL_TOPK_MAX entries) and the router
 * weights (float[top_k]) from external MTLBuffers written on the GPU, so no
 * CPU round trip is needed. Every slab is marked resident; no in-flight
 * marking (the caller guarantees the LRU does not load during the command buffer).
 */
int redmetal_topk_pool_encode_device(
    redmetal_topk_pool_t pool,
    void *mtl_command_buffer,
    void *slot_table_buffer, uint64_t slot_table_offset,
    void *weight_buffer, uint64_t weight_offset,
    uint32_t top_k,
    uint32_t ggml_type,
    uint32_t hidden_size,
    uint32_t ffn_size,
    uint64_t gate_bytes,
    uint64_t up_bytes,
    void *mtl_input_buffer,
    uint64_t input_offset,
    void *mtl_output_buffer,
    uint64_t output_offset);

/* dev22: same three dispatches as redmetal_topk_pool_encode_device, appended to an open compute encoder
 * (serial dispatch) instead of three encoders of a command buffer. */
int redmetal_topk_pool_encode_device_into(
    redmetal_topk_pool_t pool,
    void *mtl_compute_encoder,
    void *slot_table_buffer, uint64_t slot_table_offset,
    void *weight_buffer, uint64_t weight_offset,
    uint32_t top_k,
    uint32_t ggml_type,
    uint32_t hidden_size,
    uint32_t ffn_size,
    uint64_t gate_bytes,
    uint64_t up_bytes,
    void *mtl_input_buffer,
    uint64_t input_offset,
    void *mtl_output_buffer,
    uint64_t output_offset);

/*
 * Batched prefill (dev20b): n_expert unique experts (slot table) serve n_pairs
 * (expert, token) pairs sorted by expert (expert_start[n_expert+1] prefix
 * sums; pair_token / pair_weight per pair). tok_pair[ntok*top_k] lists each
 * token's pairs in router selection order. Input rows are hidden floats per
 * token at input_offset (token index = pair_token), output rows likewise for
 * tokens tok_first..tok_first+ntok-1. Marks the slots in flight; release with
 * redmetal_topk_pool_release(slot_ids, n_expert).
 */
/*
 * Mapped experts (dev20b prefill): the experts are read in place from three
 * external MTLBuffers (the mmap'd gate/up/down tensors of one layer);
 * *_addr0 is the GPU address of expert 0 inside each buffer and *_bytes the
 * per-expert stride. Same pair layout as the batched encode; no slots, no
 * in-flight marking. Uses the pool's pipelines and scratch only.
 */
int redmetal_topk_pool_encode_mapped(
    redmetal_topk_pool_t pool,
    void *mtl_command_buffer,
    void *gate_buffer, uint64_t gate_addr0,
    void *up_buffer, uint64_t up_addr0,
    void *down_buffer, uint64_t down_addr0,
    uint64_t gate_bytes, uint64_t up_bytes, uint64_t down_bytes,
    const uint32_t *expert_ids,
    uint32_t n_expert,
    uint32_t ggml_type,
    uint32_t hidden_size,
    uint32_t ffn_size,
    uint32_t tok_first,
    uint32_t ntok,
    uint32_t top_k,
    uint32_t n_pairs,
    const uint32_t *pair_token,
    const float *pair_weight,
    const uint32_t *expert_start,
    const uint32_t *tok_pair,
    void *mtl_input_buffer,
    uint64_t input_offset,
    void *mtl_output_buffer,
    uint64_t output_offset);

int redmetal_topk_pool_encode_batched(
    redmetal_topk_pool_t pool,
    void *mtl_command_buffer,
    const uint32_t *slot_ids,
    const uint64_t *gate_bytes,
    const uint64_t *up_bytes,
    const uint64_t *down_bytes,
    uint32_t n_expert,
    uint32_t ggml_type,
    uint32_t hidden_size,
    uint32_t ffn_size,
    uint32_t tok_first,
    uint32_t ntok,
    uint32_t top_k,
    uint32_t n_pairs,
    const uint32_t *pair_token,
    const float *pair_weight,
    const uint32_t *expert_start,
    const uint32_t *tok_pair,
    void *mtl_input_buffer,
    uint64_t input_offset,
    void *mtl_output_buffer,
    uint64_t output_offset);

#ifdef __cplusplus
}
#endif
