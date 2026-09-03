#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RL_SHARED_MAX_DIMS 8u
#define RL_SHARED_MAX_LAYERS 256u

typedef enum {
    RL_SHARED_GATE_INPUT = 0,
    RL_SHARED_GATE = 1,
    RL_SHARED_UP = 2,
    RL_SHARED_DOWN = 3,
} rl_shared_kind;

typedef struct {
    uint32_t layer;
    rl_shared_kind kind;
    uint32_t ggml_type;
    uint32_t n_dims;
    uint64_t shape[RL_SHARED_MAX_DIMS];
    uint64_t tensor_offset;
    uint64_t tensor_span_bytes;
} rl_shared_tensor_info;

int rl_native_shared_audit(
    const char *model_path,
    rl_shared_tensor_info *out,
    uint32_t out_capacity,
    uint32_t *out_count,
    char *error,
    size_t error_cap);

const char *rl_native_shared_kind_name(rl_shared_kind kind);

#ifdef __cplusplus
}
#endif
