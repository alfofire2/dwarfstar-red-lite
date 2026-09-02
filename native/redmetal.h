#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void * redmetal_pool_t;

uint32_t redmetal_abi_version(void);
const char * redmetal_last_error(void);

int redmetal_system_device_name(char *dst, size_t capacity);

redmetal_pool_t redmetal_pool_create(
    const char *model_path,
    uint64_t budget_bytes,
    uint64_t slot_bytes,
    uint32_t slots_per_slab);

void redmetal_pool_destroy(redmetal_pool_t pool);

uint32_t redmetal_pool_capacity(redmetal_pool_t pool);
uint32_t redmetal_pool_slab_count(redmetal_pool_t pool);
uint64_t redmetal_pool_allocated_bytes(redmetal_pool_t pool);
uint64_t redmetal_pool_bytes_read(redmetal_pool_t pool);
uint64_t redmetal_pool_read_calls(redmetal_pool_t pool);
double redmetal_pool_read_ms(redmetal_pool_t pool);

int redmetal_pool_load_expert(
    redmetal_pool_t pool,
    uint32_t slot_id,
    uint64_t gate_file_offset,
    uint64_t gate_bytes,
    uint64_t up_file_offset,
    uint64_t up_bytes,
    uint64_t down_file_offset,
    uint64_t down_bytes,
    uint64_t *bytes_read,
    double *elapsed_ms);

int redmetal_pool_gpu_probe(
    redmetal_pool_t pool,
    uint32_t slot_id,
    uint64_t payload_bytes,
    uint32_t *checksum,
    double *elapsed_ms);

#ifdef __cplusplus
}
#endif
