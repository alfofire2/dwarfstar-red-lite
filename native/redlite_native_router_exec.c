#define _FILE_OFFSET_BITS 64
#define _POSIX_C_SOURCE 200809L

#include "redlite_native_router_exec.h"

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static void set_error(char *dst, size_t cap, const char *msg) {
    if (dst && cap) snprintf(dst, cap, "%s", msg ? msg : "unknown router execution error");
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

static int pread_full(int fd, void *dst, size_t bytes, uint64_t offset) {
    unsigned char *p = (unsigned char *)dst;
    size_t done = 0;
    while (done < bytes) {
        const ssize_t n = pread(fd, p + done, bytes - done, (off_t)(offset + done));
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return 0;
        done += (size_t)n;
    }
    return 1;
}

int rl_native_router_cpu_f32(
        const char *model_path,
        const rl_router_tensor_info *router,
        const float *input,
        uint32_t input_count,
        float *logits,
        uint32_t logits_count,
        rl_native_router_telemetry *telemetry,
        char *error,
        size_t error_cap) {
    if (!model_path || !router || !input || !logits || router->ggml_type != 0u ||
        router->n_dims != 2u || !router->shape[0] || !router->shape[1] ||
        router->shape[0] > UINT32_MAX || router->shape[1] > UINT32_MAX) {
        set_error(error, error_cap, "invalid F32 router CPU arguments");
        return 0;
    }
    const uint32_t hidden = (uint32_t)router->shape[0];
    const uint32_t experts = (uint32_t)router->shape[1];
    const uint64_t expected_bytes = (uint64_t)hidden * experts * sizeof(float);
    if (input_count != hidden || logits_count < experts || router->tensor_span_bytes < expected_bytes ||
        expected_bytes > SIZE_MAX) {
        set_error(error, error_cap, "router F32 shape/span mismatch");
        return 0;
    }

    float *weights = (float *)malloc((size_t)expected_bytes);
    if (!weights) {
        set_error(error, error_cap, "out of memory for F32 router weights");
        return 0;
    }
    const int fd = open(model_path, O_RDONLY);
    if (fd < 0) {
        free(weights);
        set_error(error, error_cap, "open GGUF for router CPU failed");
        return 0;
    }

    const double read_start = now_ms();
    const int read_ok = pread_full(fd, weights, (size_t)expected_bytes, router->tensor_offset);
    const double read_end = now_ms();
    close(fd);
    if (!read_ok) {
        free(weights);
        set_error(error, error_cap, "pread F32 router weights failed");
        return 0;
    }

    const double compute_start = now_ms();
    for (uint32_t e = 0; e < experts; ++e) {
        const float *row = weights + (size_t)e * hidden;
        double sum = 0.0;
        for (uint32_t i = 0; i < hidden; ++i) sum += (double)row[i] * (double)input[i];
        logits[e] = (float)sum;
    }
    const double compute_end = now_ms();
    free(weights);

    if (telemetry) {
        telemetry->read_ms = read_end - read_start;
        telemetry->compute_ms = compute_end - compute_start;
    }
    if (error && error_cap) error[0] = '\0';
    return 1;
}

int rl_native_router_select_softmax_topk(
        const float *logits,
        uint32_t expert_count,
        uint32_t top_k,
        uint32_t *expert_ids,
        float *router_weights,
        float *probabilities,
        char *error,
        size_t error_cap) {
    if (!logits || !expert_ids || !router_weights || !expert_count || !top_k || top_k > expert_count) {
        set_error(error, error_cap, "invalid router top-k selection arguments");
        return 0;
    }

    float *owned_probs = probabilities;
    if (!owned_probs) {
        owned_probs = (float *)malloc((size_t)expert_count * sizeof(float));
        if (!owned_probs) {
            set_error(error, error_cap, "out of memory for router probabilities");
            return 0;
        }
    }

    float max_logit = logits[0];
    for (uint32_t i = 1; i < expert_count; ++i) if (logits[i] > max_logit) max_logit = logits[i];
    double denom = 0.0;
    for (uint32_t i = 0; i < expert_count; ++i) {
        const float p = expf(logits[i] - max_logit);
        owned_probs[i] = p;
        denom += (double)p;
    }
    if (!(denom > 0.0) || !isfinite(denom)) {
        if (!probabilities) free(owned_probs);
        set_error(error, error_cap, "invalid router softmax denominator");
        return 0;
    }
    const float inv_denom = (float)(1.0 / denom);
    for (uint32_t i = 0; i < expert_count; ++i) owned_probs[i] *= inv_denom;

    unsigned char *used = (unsigned char *)calloc(expert_count, 1);
    if (!used) {
        if (!probabilities) free(owned_probs);
        set_error(error, error_cap, "out of memory for router top-k mask");
        return 0;
    }

    double selected_sum = 0.0;
    for (uint32_t k = 0; k < top_k; ++k) {
        uint32_t best = UINT32_MAX;
        float best_p = -INFINITY;
        for (uint32_t i = 0; i < expert_count; ++i) {
            if (used[i]) continue;
            const float p = owned_probs[i];
            if (best == UINT32_MAX || p > best_p || (p == best_p && i < best)) {
                best = i;
                best_p = p;
            }
        }
        if (best == UINT32_MAX) {
            free(used);
            if (!probabilities) free(owned_probs);
            set_error(error, error_cap, "router top-k selection failed");
            return 0;
        }
        used[best] = 1;
        expert_ids[k] = best;
        router_weights[k] = best_p;
        selected_sum += (double)best_p;
    }
    free(used);

    if (!(selected_sum > 0.0) || !isfinite(selected_sum)) {
        if (!probabilities) free(owned_probs);
        set_error(error, error_cap, "invalid selected router weight sum");
        return 0;
    }
    const float inv_selected = (float)(1.0 / selected_sum);
    for (uint32_t k = 0; k < top_k; ++k) router_weights[k] *= inv_selected;

    if (!probabilities) free(owned_probs);
    if (error && error_cap) error[0] = '\0';
    return 1;
}
