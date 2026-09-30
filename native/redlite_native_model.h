#pragma once

#include "redlite_native_gguf.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t layer;
    uint32_t ggml_type;
    uint32_t hidden_size;
    uint32_t ffn_size;
    uint32_t expert_count;
} rl_native_layer_info;

int rl_native_get_layer_info(
    const rl_expert_map *map,
    uint32_t layer,
    rl_native_layer_info *out,
    char *error,
    size_t error_cap);

#ifdef __cplusplus
}
#endif
