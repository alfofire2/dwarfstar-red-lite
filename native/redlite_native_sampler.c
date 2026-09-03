#include "redlite_native_sampler.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

void rl_sampler_params_default(rl_sampler_params *p) {
    if (!p) return;
    p->temperature = 0.0f;
    p->top_k = 40u;
    p->top_p = 0.95f;
    p->seed = 0u;
}

static uint64_t splitmix64(uint64_t *state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

int rl_sampler_init(rl_sampler *s, const rl_sampler_params *p, uint32_t vocab) {
    if (!s || !vocab) return 0;
    memset(s, 0, sizeof(*s));
    if (p) s->params = *p; else rl_sampler_params_default(&s->params);
    s->rng_state = s->params.seed ? s->params.seed : 0x243F6A8885A308D3ull;
    s->vocab = vocab;
    s->scratch = (float *)malloc((size_t)vocab * sizeof(float));
    s->index = (uint32_t *)malloc((size_t)vocab * sizeof(uint32_t));
    if (!s->scratch || !s->index) { rl_sampler_free(s); return 0; }
    return 1;
}

void rl_sampler_free(rl_sampler *s) {
    if (!s) return;
    free(s->scratch); free(s->index);
    memset(s, 0, sizeof(*s));
}

uint64_t rl_sampler_random_u64(rl_sampler *s) { return splitmix64(&s->rng_state); }

static const float *g_sort_logits;

static int cmp_desc(const void *a, const void *b) {
    const float *v = g_sort_logits;
    const float x = v[*(const uint32_t *)a], y = v[*(const uint32_t *)b];
    if (x > y) return -1;
    if (x < y) return 1;
    return *(const uint32_t *)a < *(const uint32_t *)b ? -1 : (*(const uint32_t *)a > *(const uint32_t *)b);
}

uint32_t rl_sampler_sample(rl_sampler *s, const float *logits) {
    const uint32_t n = s->vocab;
    uint32_t best = 0;
    for (uint32_t i = 1; i < n; ++i) if (logits[i] > logits[best]) best = i;
    if (s->params.temperature <= 0.0f) return best;

    /* candidate set: top-k by logit (or all) */
    uint32_t count = n;
    for (uint32_t i = 0; i < n; ++i) s->index[i] = i;
    g_sort_logits = logits;
    qsort(s->index, n, sizeof(uint32_t), cmp_desc);
    if (s->params.top_k && s->params.top_k < count) count = s->params.top_k;

    /* softmax with temperature over candidates */
    const double inv_t = 1.0 / (double)s->params.temperature;
    const double max_logit = logits[s->index[0]];
    double sum = 0.0;
    for (uint32_t i = 0; i < count; ++i) {
        const double p = exp(((double)logits[s->index[i]] - max_logit) * inv_t);
        s->scratch[i] = (float)p;
        sum += p;
    }
    for (uint32_t i = 0; i < count; ++i) s->scratch[i] = (float)((double)s->scratch[i] / sum);

    /* top-p nucleus */
    if (s->params.top_p > 0.0f && s->params.top_p < 1.0f) {
        double cum = 0.0;
        uint32_t keep = count;
        for (uint32_t i = 0; i < count; ++i) {
            cum += s->scratch[i];
            if (cum >= (double)s->params.top_p) { keep = i + 1u; break; }
        }
        count = keep;
        double renorm = 0.0;
        for (uint32_t i = 0; i < count; ++i) renorm += s->scratch[i];
        for (uint32_t i = 0; i < count; ++i) s->scratch[i] = (float)((double)s->scratch[i] / renorm);
    }

    const double r = (double)(splitmix64(&s->rng_state) >> 11) * (1.0 / 9007199254740992.0);
    double acc = 0.0;
    for (uint32_t i = 0; i < count; ++i) {
        acc += s->scratch[i];
        if (r < acc) return s->index[i];
    }
    return s->index[count - 1u];
}
