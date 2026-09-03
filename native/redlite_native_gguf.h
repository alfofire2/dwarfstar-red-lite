#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RL_NATIVE_MAX_DIMS 8u
#define RL_NATIVE_MAX_LAYERS 256u

typedef enum {
    RL_EXPERT_GATE = 0,
    RL_EXPERT_UP = 1,
    RL_EXPERT_DOWN = 2,
} rl_expert_kind;

typedef struct {
    uint32_t layer;
    rl_expert_kind kind;
    uint32_t ggml_type;
    uint32_t n_dims;
    uint64_t shape[RL_NATIVE_MAX_DIMS];
    uint64_t tensor_offset;
    uint64_t tensor_span_bytes;
    uint64_t payload_bytes;
    uint64_t tail_padding_bytes;
    uint64_t expert_stride_bytes;
    int slice_safe;
} rl_expert_tensor;

typedef struct {
    uint32_t layer;
    uint32_t ggml_type;
    uint32_t n_dims;
    uint64_t shape[RL_NATIVE_MAX_DIMS];
    uint64_t tensor_offset;
    uint64_t tensor_span_bytes;
} rl_router_tensor;

typedef struct {
    uint32_t version;
    uint32_t alignment;
    uint64_t tensor_count;
    uint64_t kv_count;
    uint32_t expert_count;
    uint32_t layer_count;
    uint32_t routed_tensor_count;
    uint32_t router_tensor_count;
    uint64_t total_routed_payload_bytes;
    uint64_t max_expert_triplet_bytes;
    int all_slice_safe;
    rl_expert_tensor *routed;
    rl_router_tensor *routers;
} rl_expert_map;

typedef struct {
    uint32_t layer;
    uint32_t expert;
    uint32_t ggml_type;
    uint64_t gate_offset;
    uint64_t gate_bytes;
    uint64_t up_offset;
    uint64_t up_bytes;
    uint64_t down_offset;
    uint64_t down_bytes;
    uint64_t total_bytes;
} rl_expert_layout;

int rl_native_build_expert_map(
    const char *path,
    uint32_t expected_experts,
    rl_expert_map *out,
    char *error,
    size_t error_cap);

void rl_native_free_expert_map(rl_expert_map *map);

int rl_native_expert_layout(
    const rl_expert_map *map,
    uint32_t layer,
    uint32_t expert,
    rl_expert_layout *out,
    char *error,
    size_t error_cap);

int rl_native_router_tensor(
    const rl_expert_map *map,
    uint32_t layer,
    rl_router_tensor *out,
    char *error,
    size_t error_cap);

const char *rl_native_quant_name(uint32_t ggml_type);
const char *rl_native_type_name(uint32_t ggml_type);
const char *rl_native_kind_name(rl_expert_kind kind);

#ifdef __cplusplus
}
#endif
