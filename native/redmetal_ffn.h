#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void * redmetal_ffn_pool_t;

uint32_t redmetal_ffn_abi_version(void);
const char * redmetal_ffn_last_error(void);

redmetal_ffn_pool_t redmetal_ffn_pool_create(
    const char *model_path,
    uint64_t budget_bytes,
    uint64_t slot_bytes,
    uint32_t slots_per_slab,
    const int8_t *iq2_xs_grid,
    uint32_t iq2_xs_grid_count,
    const int8_t *iq1_m_grid,
    uint32_t iq1_m_grid_count);

void redmetal_ffn_pool_destroy(redmetal_ffn_pool_t pool);
uint32_t redmetal_ffn_pool_capacity(redmetal_ffn_pool_t pool);
uint32_t redmetal_ffn_pool_slab_count(redmetal_ffn_pool_t pool);
uint64_t redmetal_ffn_pool_allocated_bytes(redmetal_ffn_pool_t pool);
uint64_t redmetal_ffn_pool_bytes_read(redmetal_ffn_pool_t pool);
uint64_t redmetal_ffn_pool_read_calls(redmetal_ffn_pool_t pool);
double redmetal_ffn_pool_read_ms(redmetal_ffn_pool_t pool);
int redmetal_ffn_pool_slot_inflight(redmetal_ffn_pool_t pool, uint32_t slot_id);

int redmetal_ffn_pool_load_expert(
    redmetal_ffn_pool_t pool,
    uint32_t slot_id,
    uint64_t gate_file_offset,
    uint64_t gate_bytes,
    uint64_t up_file_offset,
    uint64_t up_bytes,
    uint64_t down_file_offset,
    uint64_t down_bytes,
    uint64_t *bytes_read,
    double *elapsed_ms);

int redmetal_ffn_pool_slot_addresses(
    redmetal_ffn_pool_t pool,
    uint32_t slot_id,
    uint64_t gate_bytes,
    uint64_t up_bytes,
    uint64_t down_bytes,
    uint64_t *gate_address,
    uint64_t *up_address,
    uint64_t *down_address);

int redmetal_ffn_pool_execute_expert(
    redmetal_ffn_pool_t pool,
    uint32_t slot_id,
    uint32_t ggml_type,
    uint64_t gate_bytes,
    uint64_t up_bytes,
    uint64_t down_bytes,
    uint32_t hidden_size,
    uint32_t ffn_size,
    uint32_t output_row_start,
    uint32_t output_row_count,
    const float *input,
    uint32_t input_count,
    float *output,
    uint32_t output_count,
    double *elapsed_ms);

#ifdef __cplusplus
}
#endif
