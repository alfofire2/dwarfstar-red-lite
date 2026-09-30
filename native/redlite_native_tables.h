#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RL_IQ2_XS_GRID_COUNT (512u * 8u)
#define RL_IQ1_M_GRID_COUNT  (2048u * 8u)

typedef struct {
    int8_t *iq2_xs;
    int8_t *iq1_m;
} rl_native_quant_tables;

int rl_native_quant_tables_init(
    rl_native_quant_tables *tables,
    char *error,
    size_t error_cap);

void rl_native_quant_tables_free(rl_native_quant_tables *tables);

uint64_t rl_native_quant_table_fnv1a(const int8_t *data, size_t count);

#ifdef __cplusplus
}
#endif
