#include "redlite_native_sampler.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

void rl_sampler_params_default(rl_sampler_params *p) {
    if (!p) return;
    p->temperature = 0.0f;
    p->top_k = 40u;
    p->top_p = 0.95f;
    p->min_p = 0.0f;
    p->seed = 0u;
    p->presence_penalty = 0.0f;
    p->frequency_penalty = 0.0f;
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
    s->probs = (float *)malloc((size_t)vocab * sizeof(float));
    s->index = (uint32_t *)malloc((size_t)vocab * sizeof(uint32_t));
    if (!s->scratch || !s->probs || !s->index) { rl_sampler_free(s); return 0; }
    if (s->params.presence_penalty != 0.0f || s->params.frequency_penalty != 0.0f) {
        s->penalized = (float *)malloc((size_t)vocab * sizeof(float));
        s->counts = (uint32_t *)calloc(vocab, sizeof(uint32_t));
        s->seen = (uint32_t *)malloc((size_t)vocab * sizeof(uint32_t));
        if (!s->penalized || !s->counts || !s->seen) { rl_sampler_free(s); return 0; }
    }
    return 1;
}

void rl_sampler_accept(rl_sampler *s, uint32_t token) {
    if (!s || !s->counts || token >= s->vocab) return;
    if (s->counts[token]++ == 0u) s->seen[s->seen_count++] = token;
}

/* The logits the chain works on: the input, or a penalized copy (only the generated ids change). */
static const float *penalize(rl_sampler *s, const float *logits) {
    if (!s->counts || !s->seen_count) return logits;
    memcpy(s->penalized, logits, (size_t)s->vocab * sizeof(float));
    for (uint32_t i = 0; i < s->seen_count; ++i) {
        const uint32_t t = s->seen[i];
        s->penalized[t] -= s->params.presence_penalty + s->params.frequency_penalty * (float)s->counts[t];
    }
    return s->penalized;
}

void rl_sampler_free(rl_sampler *s) {
    if (!s) return;
    free(s->scratch); free(s->probs); free(s->index);
    free(s->penalized); free(s->counts); free(s->seen);
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

/*
 * Candidate filtering in the order and float arithmetic of the pinned llama.cpp chain
 * (top_k -> top_p -> min_p -> temp -> dist, src/llama-sampler.cpp):
 *   top-p: float softmax of the raw logits over the top-k candidates, float running sum, keep up
 *          to the first index where the sum reaches p;
 *   min-p: keep candidates with logit >= max_logit + logf(min_p);
 *   temp : logit / temperature, then float weights expf(l - max) summed in double.
 * Leaves the candidates in index[0..count) and their unnormalized weights in scratch; *sum_out
 * receives the weight sum. Returns 0 for greedy.
 */
static uint32_t candidates(rl_sampler *s, const float *logits, double *sum_out) {
    const uint32_t n = s->vocab;
    if (s->params.temperature <= 0.0f) return 0;

    const uint32_t k = s->params.top_k && s->params.top_k < n ? s->params.top_k : n;
    uint32_t count = select_top_k(s, logits, k);
    if (!count) return 0;

    if (s->params.top_p < 1.0f && count > 1u) {
        const float max_logit = logits[s->index[0]];
        float sum = 0.0f;
        for (uint32_t i = 0; i < count; ++i) {
            s->scratch[i] = expf(logits[s->index[i]] - max_logit);
            sum += s->scratch[i];
        }
        float cum = 0.0f;
        for (uint32_t i = 0; i < count; ++i) {
            cum += s->scratch[i] / sum;
            if (cum >= s->params.top_p) { count = i + 1u; break; }
        }
    }

    if (s->params.min_p > 0.0f && count > 1u) {
        const float min_logit = logits[s->index[0]] + logf(s->params.min_p);
        uint32_t keep = 1u; /* the first candidate always stays */
        while (keep < count && logits[s->index[keep]] >= min_logit) ++keep;
        count = keep;
    }

    const float t = s->params.temperature;
    const float max_scaled = logits[s->index[0]] / t;
    double sum = 0.0;
    for (uint32_t i = 0; i < count; ++i) {
        s->scratch[i] = expf(logits[s->index[i]] / t - max_scaled);
        sum += s->scratch[i];
    }
    *sum_out = sum;
    return count;
}

static uint32_t argmax(const float *logits, uint32_t n) {
    uint32_t best = 0;
    for (uint32_t i = 1; i < n; ++i) if (logits[i] > logits[best]) best = i;
    return best;
}

uint32_t rl_sampler_distribution(rl_sampler *s, const float *logits, const uint32_t **ids, const float **probs) {
    logits = penalize(s, logits);
    double sum = 0.0;
    uint32_t count = candidates(s, logits, &sum);
    if (!count) {
        s->index[0] = argmax(logits, s->vocab);
        s->probs[0] = 1.0f;
        count = 1u;
    } else {
        for (uint32_t i = 0; i < count; ++i) s->probs[i] = (float)(s->scratch[i] / sum);
    }
    if (ids) *ids = s->index;
    if (probs) *probs = s->probs;
    return count;
}

uint32_t rl_sampler_sample(rl_sampler *s, const float *logits) {
    logits = penalize(s, logits);
    double sum = 0.0;
    const uint32_t count = candidates(s, logits, &sum);
    if (!count) return argmax(logits, s->vocab);
    /* like llama.cpp's dist: the first candidate whose running weight reaches u * sum, u in [0, 1) */
    const double target = (double)(splitmix64(&s->rng_state) >> 11) * (1.0 / 9007199254740992.0) * sum;
    double acc = 0.0;
    for (uint32_t i = 0; i < count; ++i) {
        acc += s->scratch[i];
        if (acc >= target) return s->index[i];
    }
    return s->index[count - 1u];
}
