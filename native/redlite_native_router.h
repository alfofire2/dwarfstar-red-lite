#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RL_ROUTER_MAX_DIMS 8u
#define RL_ROUTER_MAX_LAYERS 256u

typedef struct {
    uint32_t layer;
    uint32_t ggml_type;
    uint32_t n_dims;
    uint64_t shape[RL_ROUTER_MAX_DIMS];
    uint64_t tensor_offset;
    uint64_t tensor_span_bytes;
} rl_router_tensor_info;

int rl_native_router_audit(
    const char *model_path,
    rl_router_tensor_info *out,
    uint32_t out_capacity,
    uint32_t *out_count,
    char *error,
    size_t error_cap);

const char *rl_native_ggml_type_name(uint32_t ggml_type);

#ifdef __cplusplus
}
#endif
