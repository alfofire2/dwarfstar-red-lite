#pragma once

/*
 * Native token sampling: greedy, or top-k -> top-p -> min-p -> temperature -> categorical draw,
 * the order and arithmetic of the pinned llama.cpp sampler chain (docs/REDLITE_DEV26_ROBUSTNESS.md).
 * The PRNG (splitmix64) is not llama.cpp's mt19937: distributions match, individual seeded draws do not.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float temperature;   /* <= 0 -> greedy */
    uint32_t top_k;      /* 0 -> disabled */
    float top_p;         /* >= 1 -> disabled */
    float min_p;         /* <= 0 -> disabled (llama.cpp default 0.05) */
    uint64_t seed;
} rl_sampler_params;

typedef struct {
    rl_sampler_params params;
    uint64_t rng_state;
    float *scratch;      /* vocab floats: unnormalized weights of the candidates */
    float *probs;        /* vocab floats: normalized final distribution (rl_sampler_distribution) */
    uint32_t *index;     /* vocab ids: candidates in descending logit order */
    uint32_t vocab;
} rl_sampler;

void rl_sampler_params_default(rl_sampler_params *p);
int rl_sampler_init(rl_sampler *s, const rl_sampler_params *p, uint32_t vocab);
void rl_sampler_free(rl_sampler *s);

/* Returns the selected token id. */
uint32_t rl_sampler_sample(rl_sampler *s, const float *logits);

/*
 * The exact distribution rl_sampler_sample draws from for these logits, without drawing:
 * *ids / *probs point at count candidates (descending logit) and their probabilities.
 * Greedy (temperature <= 0) returns the argmax with probability 1.
 */
uint32_t rl_sampler_distribution(rl_sampler *s, const float *logits, const uint32_t **ids, const float **probs);

uint64_t rl_sampler_random_u64(rl_sampler *s);

#ifdef __cplusplus
}
#endif
