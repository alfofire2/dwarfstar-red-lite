#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RL_IQ2_XXS_GRID_COUNT (256u * 8u)

int rl_native_iq2_xxs_build_grid(uint8_t out[RL_IQ2_XXS_GRID_COUNT], char *error, size_t error_cap);

#ifdef __cplusplus
}
#endif
