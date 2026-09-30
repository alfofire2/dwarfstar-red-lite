#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RL_LAYER_MAP_MAX_LAYERS 256u

typedef enum {
    RL_LAYER_MAP_UNKNOWN = 0,
    RL_LAYER_MAP_RECURRENT = 1,
    RL_LAYER_MAP_FULL_ATTENTION = 2,
    RL_LAYER_MAP_MIXED = 3,
} rl_layer_map_kind;

typedef struct {
    uint32_t layer_count;
    uint32_t recurrent_count;
    uint32_t full_attention_count;
    rl_layer_map_kind kind[RL_LAYER_MAP_MAX_LAYERS];
} rl_native_layer_map;

int rl_native_layer_map_audit(
    const char *model_path,
    rl_native_layer_map *out,
    char *error,
    size_t error_cap);

const char *rl_native_layer_map_kind_name(rl_layer_map_kind kind);

#ifdef __cplusplus
}
#endif
