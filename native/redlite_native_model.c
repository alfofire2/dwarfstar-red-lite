#include "redlite_native_model.h"

#include <limits.h>
#include <stdio.h>
#include <string.h>

static void set_error(char *error, size_t cap, const char *message) {
    if (error && cap) snprintf(error, cap, "%s", message ? message : "unknown layer-info error");
}

int rl_native_get_layer_info(
        const rl_expert_map *map,
        uint32_t layer,
        rl_native_layer_info *out,
        char *error,
        size_t error_cap) {
    if (!map || !out || !map->all_slice_safe) {
        set_error(error, error_cap, "invalid native routed layer request");
        return 0;
    }

    const rl_expert_tensor *gate = NULL;
    const rl_expert_tensor *up = NULL;
    const rl_expert_tensor *down = NULL;
    for (uint32_t i = 0; i < map->routed_tensor_count; ++i) {
        const rl_expert_tensor *t = &map->routed[i];
        if (t->layer != layer) continue;
        if (t->kind == RL_EXPERT_GATE) gate = t;
        else if (t->kind == RL_EXPERT_UP) up = t;
        else if (t->kind == RL_EXPERT_DOWN) down = t;
    }
    if (!gate || !up || !down) {
        set_error(error, error_cap, "routed layer is missing gate/up/down tensors");
        return 0;
    }
    if (gate->ggml_type != up->ggml_type || gate->ggml_type != down->ggml_type) {
        set_error(error, error_cap, "mixed gate/up/down quant types are unsupported");
        return 0;
    }
    if (gate->n_dims != 3u || up->n_dims != 3u || down->n_dims != 3u) {
        set_error(error, error_cap, "routed layer tensors must be rank 3");
        return 0;
    }
    if (gate->shape[2] != map->expert_count || up->shape[2] != map->expert_count ||
        down->shape[2] != map->expert_count) {
        set_error(error, error_cap, "routed layer expert-axis size mismatch");
        return 0;
    }
    if (gate->shape[0] != up->shape[0] || gate->shape[1] != up->shape[1] ||
        down->shape[0] != gate->shape[1] || down->shape[1] != gate->shape[0]) {
        set_error(error, error_cap, "unexpected routed gate/up/down matrix shapes");
        return 0;
    }
    if (!gate->shape[0] || !gate->shape[1] || gate->shape[0] > UINT32_MAX || gate->shape[1] > UINT32_MAX) {
        set_error(error, error_cap, "routed layer dimensions exceed native limits");
        return 0;
    }

    memset(out, 0, sizeof(*out));
    out->layer = layer;
    out->ggml_type = gate->ggml_type;
    out->hidden_size = (uint32_t)gate->shape[0];
    out->ffn_size = (uint32_t)gate->shape[1];
    out->expert_count = map->expert_count;
    if (error && error_cap) error[0] = '\0';
    return 1;
}
