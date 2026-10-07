#pragma once

/*
 * Native token sampling: greedy, or top-k -> top-p -> min-p -> temperature -> categorical draw,
 * the order and arithmetic of the pinned llama.cpp sampler chain (docs/REDLITE_DEV26_ROBUSTNESS.md).
 * The PRNG (splitmix64) is not llama.cpp's mt19937: distributions match, individual seeded draws do not.
 */

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float temperature;   /* <= 0 -> greedy */
    uint32_t top_k;      /* 0 -> disabled */
    float top_p;         /* >= 1 -> disabled */
    float min_p;         /* <= 0 -> disabled (llama.cpp default 0.05) */
    uint64_t seed;
    /* dev63: OpenAI penalties over the tokens generated so far (rl_sampler_accept): every such token's logit is
     * lowered by presence_penalty + frequency_penalty x its count, before the chain above. 0 = off. */
    float presence_penalty;
    float frequency_penalty;
} rl_sampler_params;

typedef struct {
    rl_sampler_params params;
    uint64_t rng_state;
    float *scratch;      /* vocab floats: unnormalized weights of the candidates */
    float *probs;        /* vocab floats: normalized final distribution (rl_sampler_distribution) */
    uint32_t *index;     /* vocab ids: candidates in descending logit order */
    uint32_t vocab;
    float *penalized;    /* dev63: vocab floats, the logits after the penalties (penalties on only) */
    uint32_t *counts;    /* per token id, times generated */
    uint32_t *seen;      /* the distinct ids generated, seen_count of them */
    uint32_t seen_count;
} rl_sampler;

void rl_sampler_params_default(rl_sampler_params *p);
int rl_sampler_init(rl_sampler *s, const rl_sampler_params *p, uint32_t vocab);
void rl_sampler_free(rl_sampler *s);

/* dev63: records a generated token for the penalties (a no-op when they are off). */
void rl_sampler_accept(rl_sampler *s, uint32_t token);

/* Returns the selected token id. */
uint32_t rl_sampler_sample(rl_sampler *s, const float *logits);

/*
 * The exact distribution rl_sampler_sample draws from for these logits, without drawing:
 * *ids / *probs point at count candidates (descending logit) and their probabilities.
 * Greedy (temperature <= 0) returns the argmax with probability 1.
 */
uint32_t rl_sampler_distribution(rl_sampler *s, const float *logits, const uint32_t **ids, const float **probs);

uint64_t rl_sampler_random_u64(rl_sampler *s);

/* dev70 prompt lookup: a draft for the token after seq[0..n) from the context itself. Finds the latest earlier
 * occurrence of the last g tokens (g = 3, then 2) and returns 1 with *draft = the token that followed it; 0 when
 * there is none. Exact speculation verifies the draft, so a wrong one costs time, never output. */
static inline int rl_lookup_draft(const uint32_t *seq, uint32_t n, uint32_t *draft) {
    /* ponytail: linear scan back from the end, ~n compares per token (60 us at 64K); an n-gram index if it shows */
    if (!seq || !draft) return 0;
    for (uint32_t g = 3u; g >= 2u; --g) {
        if (n < g + 1u) continue;
        const uint32_t *key = seq + n - g;
        for (uint32_t end = n - 1u; end-- > g - 1u;) {   /* candidate occurrence seq[end-g+1 .. end], end < n - 1 */
            if (seq[end] != key[g - 1u] || memcmp(seq + end + 1u - g, key, (size_t)g * sizeof(uint32_t)) != 0) continue;
            *draft = seq[end + 1u];
            return 1;
        }
    }
    return 0;
}

#ifdef __cplusplus
}

#endif
