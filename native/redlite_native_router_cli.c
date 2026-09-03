#include "redlite_native_router.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MIB (1024.0 * 1024.0)

typedef struct {
    uint32_t type;
    uint32_t count;
} type_count;

static void usage(FILE *out) {
    fprintf(out,
        "redlite-router-audit 0.3.0.dev13\n"
        "Audit Qwen routed-expert router tensors without Python.\n\n"
        "Usage:\n"
        "  redlite-router-audit MODEL [--layers]\n");
}

int main(int argc, char **argv) {
    if (argc < 2 || strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0) {
        usage(stdout);
        return argc < 2 ? 2 : 0;
    }
    const char *model = argv[1];
    int show_layers = 0;
    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--layers") == 0) show_layers = 1;
        else {
            fprintf(stderr, "unknown router-audit option: %s\n", argv[i]);
            return 2;
        }
    }

    rl_router_tensor_info routers[RL_ROUTER_MAX_LAYERS];
    uint32_t count = 0;
    char error[512];
    if (!rl_native_router_audit(model, routers, RL_ROUTER_MAX_LAYERS, &count, error, sizeof(error))) {
        fprintf(stderr, "native router audit failed: %s\n", error);
        return 1;
    }

    type_count types[32] = {{0}};
    uint32_t n_types = 0;
    uint64_t total_bytes = 0;
    int consistent_shape = count != 0;
    uint64_t d0 = count ? routers[0].shape[0] : 0;
    uint64_t d1 = count && routers[0].n_dims > 1 ? routers[0].shape[1] : 0;
    uint8_t seen[RL_ROUTER_MAX_LAYERS] = {0};
    int unique_layers = 1;

    for (uint32_t i = 0; i < count; ++i) {
        const rl_router_tensor_info *r = &routers[i];
        total_bytes += r->tensor_span_bytes;
        if (r->layer >= RL_ROUTER_MAX_LAYERS || seen[r->layer]) unique_layers = 0;
        else seen[r->layer] = 1;
        if (r->n_dims != 2u || r->shape[0] != d0 || r->shape[1] != d1) consistent_shape = 0;

        uint32_t t = 0;
        while (t < n_types && types[t].type != r->ggml_type) t++;
        if (t == n_types && n_types < 32u) {
            types[n_types].type = r->ggml_type;
            types[n_types].count = 0;
            n_types++;
        }
        if (t < n_types) types[t].count++;
    }

    printf("runtime            : native C router audit (no Python)\n");
    printf("router tensors     : %u\n", count);
    printf("unique layers      : %s\n", unique_layers ? "YES" : "NO");
    printf("consistent shape   : %s\n", consistent_shape ? "YES" : "NO");
    if (count) printf("router shape       : (%" PRIu64 ", %" PRIu64 ")\n", d0, d1);
    printf("router payload     : %.3f MiB\n", (double)total_bytes / MIB);
    printf("router types       : ");
    if (!n_types) printf("none");
    for (uint32_t i = 0; i < n_types; ++i) {
        printf("%s%s(%u)=%u", i ? " " : "", rl_native_ggml_type_name(types[i].type), types[i].type, types[i].count);
    }
    printf("\n");

    if (show_layers) {
        printf("per layer          :\n");
        for (uint32_t layer = 0; layer < RL_ROUTER_MAX_LAYERS; ++layer) {
            for (uint32_t i = 0; i < count; ++i) {
                const rl_router_tensor_info *r = &routers[i];
                if (r->layer != layer) continue;
                printf("  %3u: type=%s(%u) shape=(", layer,
                       rl_native_ggml_type_name(r->ggml_type), r->ggml_type);
                for (uint32_t d = 0; d < r->n_dims; ++d)
                    printf("%s%" PRIu64, d ? "," : "", r->shape[d]);
                printf(") span=%.3f MiB offset=%" PRIu64 "\n",
                       (double)r->tensor_span_bytes / MIB, r->tensor_offset);
            }
        }
    }

    if (count == 0) {
        fprintf(stderr, "no blk.N.ffn_gate_inp.weight router tensors found\n");
        return 1;
    }
    return 0;
}
