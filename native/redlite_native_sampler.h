#pragma once

/* Native token sampling: greedy, temperature, top-k, top-p with a seeded PRNG. */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float temperature;   /* <= 0 -> greedy */
    uint32_t top_k;      /* 0 -> disabled */
    float top_p;         /* >= 1 -> disabled */
    uint64_t seed;
} rl_sampler_params;

typedef struct {
    rl_sampler_params params;
    uint64_t rng_state;
    float *scratch;      /* vocab floats */
    uint32_t *index;     /* vocab ids */
    uint32_t vocab;
} rl_sampler;

void rl_sampler_params_default(rl_sampler_params *p);
int rl_sampler_init(rl_sampler *s, const rl_sampler_params *p, uint32_t vocab);
void rl_sampler_free(rl_sampler *s);

/* Returns the selected token id. */
uint32_t rl_sampler_sample(rl_sampler *s, const float *logits);

uint64_t rl_sampler_random_u64(rl_sampler *s);

#ifdef __cplusplus
}
#endif
