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

/* Partial selection: leaves the k largest logits (descending, ties by lower id) in index[0..k).
 * Single pass over the vocabulary with an insertion-sorted window of k entries; a full sort is
 * only needed when k is a large fraction of the vocabulary. */
static uint32_t select_top_k(rl_sampler *s, const float *logits, uint32_t k) {
    const uint32_t n = s->vocab;
    uint32_t *idx = s->index;
    if (k >= n || k > 1024u) {
        for (uint32_t i = 0; i < n; ++i) idx[i] = i;
        g_sort_logits = logits;
        qsort(idx, n, sizeof(uint32_t), cmp_desc);
        return k < n ? k : n;
    }
    uint32_t filled = 0;
    for (uint32_t i = 0; i < n; ++i) {
        const float v = logits[i];
        if (filled == k && v <= logits[idx[k - 1u]]) continue; /* not better than the current k-th (ties keep the lower id) */
        uint32_t pos = filled < k ? filled : k - 1u;
        while (pos > 0 && logits[idx[pos - 1u]] < v) { idx[pos] = idx[pos - 1u]; --pos; }
        idx[pos] = i;
        if (filled < k) ++filled;
    }
    return filled;
}

/* Order follows the default llama.cpp sampler chain: top-k, top-p (nucleus over the softmax of the
 * raw logits), then temperature and a categorical draw. Greedy (temperature <= 0) is the argmax. */
uint32_t rl_sampler_sample(rl_sampler *s, const float *logits) {
    const uint32_t n = s->vocab;
    uint32_t best = 0;
    for (uint32_t i = 1; i < n; ++i) if (logits[i] > logits[best]) best = i;
    if (s->params.temperature <= 0.0f) return best;

    /* candidate set: top-k by logit (or all when disabled) */
    const uint32_t k = s->params.top_k && s->params.top_k < n ? s->params.top_k : n;
    uint32_t count = select_top_k(s, logits, k);
    if (!count) return best;

    /* top-p nucleus on the untempered distribution over the candidates */
    if (s->params.top_p > 0.0f && s->params.top_p < 1.0f && count > 1u) {
        const double max_logit = logits[s->index[0]];
        double sum = 0.0;
        for (uint32_t i = 0; i < count; ++i) {
            const double p = exp((double)logits[s->index[i]] - max_logit);
            s->scratch[i] = (float)p;
            sum += p;
        }
        double cum = 0.0;
        uint32_t keep = count;
        for (uint32_t i = 0; i < count; ++i) {
            cum += (double)s->scratch[i] / sum;
            if (cum >= (double)s->params.top_p) { keep = i + 1u; break; }
        }
        count = keep;
    }

    /* temperature, then softmax over what is left */
    const double inv_t = 1.0 / (double)s->params.temperature;
    const double max_logit = logits[s->index[0]];
    double sum = 0.0;
    for (uint32_t i = 0; i < count; ++i) {
        const double p = exp(((double)logits[s->index[i]] - max_logit) * inv_t);
        s->scratch[i] = (float)p;
        sum += p;
    }
    const double r = (double)(splitmix64(&s->rng_state) >> 11) * (1.0 / 9007199254740992.0) * sum;
    double acc = 0.0;
    for (uint32_t i = 0; i < count; ++i) {
        acc += s->scratch[i];
        if (r < acc) return s->index[i];
    }
    return s->index[count - 1u];
}
