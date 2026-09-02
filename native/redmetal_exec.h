#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void * redmetal_exec_pool_t;

uint32_t redmetal_exec_abi_version(void);
const char * redmetal_exec_last_error(void);

redmetal_exec_pool_t redmetal_exec_pool_create(
    const char *model_path,
    uint64_t budget_bytes,
    uint64_t slot_bytes,
    uint32_t slots_per_slab);

void redmetal_exec_pool_destroy(redmetal_exec_pool_t pool);

uint32_t redmetal_exec_pool_capacity(redmetal_exec_pool_t pool);
uint32_t redmetal_exec_pool_slab_count(redmetal_exec_pool_t pool);
uint64_t redmetal_exec_pool_allocated_bytes(redmetal_exec_pool_t pool);
uint64_t redmetal_exec_pool_bytes_read(redmetal_exec_pool_t pool);
uint64_t redmetal_exec_pool_read_calls(redmetal_exec_pool_t pool);
double redmetal_exec_pool_read_ms(redmetal_exec_pool_t pool);

int redmetal_exec_pool_load_expert(
    redmetal_exec_pool_t pool,
    uint32_t slot_id,
    uint64_t gate_file_offset,
    uint64_t gate_bytes,
    uint64_t up_file_offset,
    uint64_t up_bytes,
    uint64_t down_file_offset,
    uint64_t down_bytes,
    uint64_t *bytes_read,
    double *elapsed_ms);

int redmetal_exec_pool_bind_expert(
    redmetal_exec_pool_t pool,
    uint32_t layer,
    uint32_t expert,
    uint32_t slot_id,
    uint64_t gate_bytes,
    uint64_t up_bytes,
    uint64_t down_bytes);

int redmetal_exec_pool_unbind_expert(
    redmetal_exec_pool_t pool,
    uint32_t layer,
    uint32_t expert);

int redmetal_exec_pool_bound_addresses(
    redmetal_exec_pool_t pool,
    uint32_t layer,
    uint32_t expert,
    uint64_t *gate_address,
    uint64_t *up_address,
    uint64_t *down_address);

int redmetal_exec_pool_slot_inflight(
    redmetal_exec_pool_t pool,
    uint32_t slot_id);

int redmetal_exec_pool_iq2_xxs_rows(
    redmetal_exec_pool_t pool,
    uint32_t slot_id,
    uint64_t matrix_slot_offset,
    uint32_t ncols,
    uint32_t nrows,
    uint32_t row_start,
    uint32_t row_count,
    const float *input,
    uint32_t input_count,
    float *output,
    uint32_t output_count,
    double *elapsed_ms);

#ifdef __cplusplus
}
#endif
