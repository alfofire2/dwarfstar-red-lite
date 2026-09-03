#include "redlite_native_router_exec.h"

#include <math.h>
#include <stdio.h>

int main(void) {
    char error[256] = {0};
    const float logits[4] = {0.0f, 1.0f, 2.0f, 3.0f};
    float probs[4] = {0};
    uint32_t ids[2] = {0};
    float weights[2] = {0};
    if (!rl_native_router_select_softmax_topk(logits, 4u, 2u, ids, weights, probs, error, sizeof(error))) {
        fprintf(stderr, "router selection test failed: %s\n", error);
        return 1;
    }
    const float expected0 = 0.7310585786f;
    const float expected1 = 0.2689414214f;
    if (ids[0] != 3u || ids[1] != 2u ||
        fabsf(weights[0] - expected0) > 1e-6f || fabsf(weights[1] - expected1) > 1e-6f ||
        fabsf(weights[0] + weights[1] - 1.0f) > 1e-6f) {
        fprintf(stderr, "softmax/top-k/renorm semantics mismatch\n");
        return 1;
    }

    const float ties[4] = {1.0f, 1.0f, 0.0f, 0.0f};
    if (!rl_native_router_select_softmax_topk(ties, 4u, 2u, ids, weights, NULL, error, sizeof(error)) ||
        ids[0] != 0u || ids[1] != 1u) {
        fprintf(stderr, "router deterministic tie-break mismatch\n");
        return 1;
    }

    printf("router softmax      : OK\n");
    printf("router top-k        : OK\n");
    printf("router renorm       : OK\n");
    printf("router tie-break    : OK\n");
    return 0;
}
