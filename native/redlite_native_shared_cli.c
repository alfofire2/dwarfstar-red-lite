#include "redlite_native_shared.h"
#include "redlite_native_router.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_SHARED_TENSORS (RL_SHARED_MAX_LAYERS * 4u)

static void usage(const char *argv0) {
    fprintf(stderr,
        "usage: %s MODEL [--layers]\n"
        "\n"
        "Audit exact Qwen3-Next shared-expert tensors in a GGUF.\n",
        argv0);
}

static void print_shape(const rl_shared_tensor_info *t) {
    putchar('(');
    for (uint32_t d = 0; d < t->n_dims; ++d) {
        if (d) putchar(',');
        printf("%" PRIu64, t->shape[d]);
    }
    putchar(')');
}

static const rl_shared_tensor_info *find_kind(
        const rl_shared_tensor_info *items,
        uint32_t count,
        uint32_t layer,
        rl_shared_kind kind) {
    for (uint32_t i = 0; i < count; ++i) {
        if (items[i].layer == layer && items[i].kind == kind) return &items[i];
    }
    return NULL;
}

static void print_type_summary(
        const rl_shared_tensor_info *items,
        uint32_t count,
        rl_shared_kind kind) {
    uint32_t counts[64] = {0};
    uint32_t other = 0;
    for (uint32_t i = 0; i < count; ++i) {
        if (items[i].kind != kind) continue;
        if (items[i].ggml_type < 64u) counts[items[i].ggml_type]++;
        else other++;
    }
    printf("%-18s: ", rl_native_shared_kind_name(kind));
    int first = 1;
    for (uint32_t type = 0; type < 64u; ++type) {
        if (!counts[type]) continue;
        if (!first) printf(" ");
        printf("%s(%u)=%u", rl_native_ggml_type_name(type), type, counts[type]);
        first = 0;
    }
    if (other) {
        if (!first) printf(" ");
        printf("OTHER=%u", other);
    }
    if (first) printf("NONE");
    putchar('\n');
}

int main(int argc, char **argv) {
    if (argc < 2 || strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0) {
        usage(argv[0]);
        return argc < 2 ? 2 : 0;
    }
    const char *model = argv[1];
    int show_layers = 0;
    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--layers") == 0) show_layers = 1;
        else {
            fprintf(stderr, "unknown option: %s\n", argv[i]);
            usage(argv[0]);
            return 2;
        }
    }

    rl_shared_tensor_info *items = calloc(MAX_SHARED_TENSORS, sizeof(*items));
    if (!items) {
        fprintf(stderr, "out of memory\n");
        return 1;
    }
    char error[256] = {0};
    uint32_t count = 0;
    if (!rl_native_shared_audit(model, items, MAX_SHARED_TENSORS, &count, error, sizeof(error))) {
        fprintf(stderr, "shared audit failed: %s\n", error);
        free(items);
        return 1;
    }

    uint8_t layer_seen[RL_SHARED_MAX_LAYERS] = {0};
    uint32_t layer_count = 0;
    uint32_t complete_layers = 0;
    uint64_t total_payload = 0;
    for (uint32_t i = 0; i < count; ++i) {
        total_payload += items[i].tensor_span_bytes;
        if (!layer_seen[items[i].layer]) {
            layer_seen[items[i].layer] = 1;
            layer_count++;
        }
    }

    int shapes_consistent = 1;
    uint64_t hidden = 0;
    uint64_t shared_ffn = 0;
    for (uint32_t layer = 0; layer < RL_SHARED_MAX_LAYERS; ++layer) {
        if (!layer_seen[layer]) continue;
        const rl_shared_tensor_info *gi = find_kind(items, count, layer, RL_SHARED_GATE_INPUT);
        const rl_shared_tensor_info *g  = find_kind(items, count, layer, RL_SHARED_GATE);
        const rl_shared_tensor_info *u  = find_kind(items, count, layer, RL_SHARED_UP);
        const rl_shared_tensor_info *d  = find_kind(items, count, layer, RL_SHARED_DOWN);
        if (gi && g && u && d) complete_layers++;
        else { shapes_consistent = 0; continue; }

        if (gi->n_dims != 1u || g->n_dims != 2u || u->n_dims != 2u || d->n_dims != 2u) {
            shapes_consistent = 0;
            continue;
        }
        const uint64_t h = gi->shape[0];
        const uint64_t f = g->shape[1];
        if (!h || !f || g->shape[0] != h || u->shape[0] != h || u->shape[1] != f ||
            d->shape[0] != f || d->shape[1] != h) {
            shapes_consistent = 0;
            continue;
        }
        if (!hidden) { hidden = h; shared_ffn = f; }
        else if (hidden != h || shared_ffn != f) shapes_consistent = 0;
    }

    printf("runtime            : native C shared-expert audit (no Python)\n");
    printf("shared tensors     : %u\n", count);
    printf("shared layers      : %u\n", layer_count);
    printf("complete layers    : %u/%u\n", complete_layers, layer_count);
    printf("shape consistent   : %s\n", shapes_consistent ? "YES" : "NO");
    if (hidden && shared_ffn) {
        printf("hidden / shared ffn: %" PRIu64 " / %" PRIu64 "\n", hidden, shared_ffn);
    }
    printf("shared payload     : %.3f MiB\n", (double)total_payload / (1024.0 * 1024.0));
    printf("types per kind     :\n");
    print_type_summary(items, count, RL_SHARED_GATE_INPUT);
    print_type_summary(items, count, RL_SHARED_GATE);
    print_type_summary(items, count, RL_SHARED_UP);
    print_type_summary(items, count, RL_SHARED_DOWN);

    if (show_layers) {
        printf("per layer          :\n");
        for (uint32_t layer = 0; layer < RL_SHARED_MAX_LAYERS; ++layer) {
            if (!layer_seen[layer]) continue;
            printf("%5u:\n", layer);
            for (rl_shared_kind kind = RL_SHARED_GATE_INPUT; kind <= RL_SHARED_DOWN; kind++) {
                const rl_shared_tensor_info *t = find_kind(items, count, layer, kind);
                printf("        %-10s ", rl_native_shared_kind_name(kind));
                if (!t) {
                    printf("MISSING\n");
                    continue;
                }
                printf("type=%s(%u) shape=", rl_native_ggml_type_name(t->ggml_type), t->ggml_type);
                print_shape(t);
                printf(" span=%.3f MiB offset=%" PRIu64 "\n",
                    (double)t->tensor_span_bytes / (1024.0 * 1024.0), t->tensor_offset);
            }
        }
    }

    free(items);
    return (count && complete_layers == layer_count && shapes_consistent) ? 0 : 1;
}
