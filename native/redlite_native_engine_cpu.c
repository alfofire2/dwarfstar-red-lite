#define _FILE_OFFSET_BITS 64
#define _POSIX_C_SOURCE 200809L
#define _DARWIN_C_SOURCE 1

/*
 * CPU oracle for the persistent engine.
 *
 * Every stage reproduces the arithmetic of the field-validated stage oracles
 * (dev13 router, dev14 shared/FFN, dev15 DeltaNet, dev16 full attention) but
 * operates on real persistent state instead of fixtures: DeltaNet conv/recurrent
 * states and the KV caches are read and updated in place for every token.
 *
 * Semantic reference: pinned llama.cpp src/models/qwen3next.cpp and
 * src/models/delta-net-base.cpp (build_delta_net_autoregressive).
 */

#include "redlite_native_engine_internal.h"
#include "redlite_native_quant_cpu.h"
#include "redlite_native_reference.h"
#include "redlite_native_router_exec.h"

#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- parallel rows ---- */

typedef struct {
    uint32_t ggml_type;
    const uint8_t *weights;
    size_t row_bytes;
    const float *x;
    uint32_t ncols;
    const uint8_t *grid;
    float *out;
    uint32_t begin, end;
    int ok;
} rows_job;

static void *rows_worker(void *arg) {
    rows_job *j = (rows_job *)arg;
    j->ok = 1;
    for (uint32_t r = j->begin; r < j->end; ++r) {
        double d = 0.0;
        if (!rl_quant_row_dot(j->ggml_type, j->weights + (size_t)r * j->row_bytes, j->x, j->ncols, j->grid, &d)) { j->ok = 0; return NULL; }
        j->out[r] = (float)d;
    }
    return NULL;
}

/* out[r] = dot(W[r], x) for r in [0, rows), float result of a double accumulation */
static int matvec(const rl_engine *e, const rl_gguf_tensor *t, const float *x, float *out, char *error, size_t cap) {
    const uint32_t ncols = (uint32_t)t->shape[0];
    const uint32_t rows = (uint32_t)t->shape[1];
    const size_t rb = rl_gguf_row_bytes(t->ggml_type, ncols);
    const uint8_t *w = rl_gguf_tensor_data(&e->gguf, t);
    if (!rb || !w) { snprintf(error, cap, "matvec on %s: unsupported layout", t->name); return 0; }
    int threads = e->cpu_threads;
    if ((uint32_t)threads > rows) threads = (int)rows;
    if (threads < 1) threads = 1;
    rows_job jobs[64];
    pthread_t tids[64];
    int spawned[64] = {0};
    const uint32_t chunk = (rows + (uint32_t)threads - 1u) / (uint32_t)threads;
    for (int i = 0; i < threads; ++i) {
        jobs[i].ggml_type = t->ggml_type; jobs[i].weights = w; jobs[i].row_bytes = rb; jobs[i].x = x;
        jobs[i].ncols = ncols; jobs[i].grid = e->iq2_grid; jobs[i].out = out;
        jobs[i].begin = (uint32_t)i * chunk; jobs[i].end = jobs[i].begin + chunk > rows ? rows : jobs[i].begin + chunk;
        jobs[i].ok = 1;
        if (i > 0) spawned[i] = pthread_create(&tids[i], NULL, rows_worker, &jobs[i]) == 0;
    }
    rows_worker(&jobs[0]);
    for (int i = 1; i < threads; ++i) if (!spawned[i]) rows_worker(&jobs[i]); /* thread creation failed: run inline */
    int ok = jobs[0].ok;
    for (int i = 1; i < threads; ++i) { if (spawned[i]) pthread_join(tids[i], NULL); ok = ok && jobs[i].ok; }
    if (!ok) { snprintf(error, cap, "matvec on %s failed", t->name); return 0; }
    return 1;
}

/* ---- scalar helpers (validated oracle arithmetic) ---- */

static void rmsnorm_d(const float *x, const float *w, uint32_t n, float eps, float *out) {
    double ss = 0.0;
    for (uint32_t i = 0; i < n; ++i) ss += (double)x[i] * x[i];
    const double inv = 1.0 / sqrt(ss / n + (double)eps);
    for (uint32_t i = 0; i < n; ++i) out[i] = (float)((double)x[i] * inv * w[i]);
}

static double sigmoid_stable(double x) {
    if (x >= 0.0) { const double z = exp(-x); return 1.0 / (1.0 + z); }
    const double z = exp(x); return z / (1.0 + z);
}

static double softplus_stable(double x) {
    if (x > 20.0) return x;
    if (x < -20.0) return exp(x);
    return log1p(exp(x));
}

static const float *f32_data(const rl_engine *e, const rl_gguf_tensor *t) {
    return (const float *)rl_gguf_tensor_data(&e->gguf, t);
}

/* ---- Gated DeltaNet layer (stateful) ---- */

static int deltanet_branch(rl_engine *e, rl_backend_state *s, const rl_layer_tensors *t, const float *normed,
        float *out, char *error, size_t cap) {
    const rl_engine_info *in = &e->info;
    const uint32_t hidden = in->hidden, channels = in->channels, rank = in->dt_rank, groups = in->n_group;
    const uint32_t S = in->d_state, head_v = in->head_v, d_inner = in->d_inner, d_conv = in->d_conv;
    const uint32_t qk_each = S * groups;
    const size_t conv_count = rl_engine_conv_count(e), rec_count = rl_engine_rec_count(e);
    float *conv_state = s->conv + (size_t)t->recurrent_index * conv_count;
    float *rec_state = s->rec + (size_t)t->recurrent_index * rec_count;

    float *qkv = (float *)calloc(channels, sizeof(float));
    float *z = (float *)calloc(d_inner, sizeof(float));
    float *ba = (float *)calloc(2u * rank, sizeof(float));
    float *beta = (float *)calloc(rank, sizeof(float));
    float *gate = (float *)calloc(rank, sizeof(float));
    float *conv_silu = (float *)calloc(channels, sizeof(float));
    float *q = (float *)calloc(qk_each, sizeof(float));
    float *k = (float *)calloc(qk_each, sizeof(float));
    float *core = (float *)calloc(d_inner, sizeof(float));
    float *ng = (float *)calloc(d_inner, sizeof(float));
    float *next_conv = (float *)calloc(conv_count, sizeof(float));
    int ok = 0;
    if (!qkv || !z || !ba || !beta || !gate || !conv_silu || !q || !k || !core || !ng || !next_conv) {
        snprintf(error, cap, "DeltaNet scratch allocation failed"); goto done;
    }
    if (!matvec(e, t->qkv, normed, qkv, error, cap) || !matvec(e, t->z, normed, z, error, cap) ||
        !matvec(e, t->ba, normed, ba, error, cap)) goto done;

    /* beta / gate (dev15 semantics: softplus(alpha + dt_bias) * A) */
    {
        const float *dt = f32_data(e, t->dt), *avec = f32_data(e, t->a);
        const uint32_t width = rank / groups, stride = 2u * width;
        for (uint32_t h = 0; h < rank; ++h) {
            const uint32_t g = h / width, l = h - g * width;
            const double b = ba[g * stride + l];
            const double a = (double)ba[g * stride + width + l] + dt[h];
            beta[h] = (float)sigmoid_stable(b);
            gate[h] = (float)(softplus_stable(a) * (double)avec[h]);
        }
    }
    /* causal conv over [state, new] then SiLU */
    {
        const float *kernel = f32_data(e, t->conv);
        const uint32_t ns = d_conv - 1u;
        for (uint32_t c = 0; c < channels; ++c) {
            double acc = 0.0;
            for (uint32_t j = 0; j < ns; ++j) acc += (double)conv_state[(size_t)c * ns + j] * kernel[(size_t)c * d_conv + j];
            acc += (double)qkv[c] * kernel[(size_t)c * d_conv + ns];
            conv_silu[c] = (float)(acc / (1.0 + exp(-acc)));
        }
        for (uint32_t c = 0; c < channels; ++c) {
            const size_t b = (size_t)c * ns;
            for (uint32_t j = 0; j + 1u < ns; ++j) next_conv[b + j] = conv_state[b + j + 1u];
            next_conv[b + ns - 1u] = qkv[c];
        }
    }
    /* L2 normalized q/k per key head; v = conv tail */
    {
        for (uint32_t which = 0; which < 2u; ++which) {
            float *dst = which ? k : q;
            const uint32_t offset = which ? qk_each : 0u;
            for (uint32_t h = 0; h < groups; ++h) {
                const uint32_t b = offset + h * S;
                double ss = 0.0;
                for (uint32_t j = 0; j < S; ++j) ss += (double)conv_silu[b + j] * conv_silu[b + j];
                const double inv = 1.0 / fmax(sqrt(ss), (double)in->rms_eps);
                for (uint32_t j = 0; j < S; ++j) dst[(size_t)h * S + j] = (float)((double)conv_silu[b + j] * inv);
            }
        }
    }
    const float *v = conv_silu + 2u * qk_each;
    /* recurrent gated delta rule (pinned build_delta_net_autoregressive), state updated in place */
    {
        const size_t matrix = (size_t)S * S;
        const float scale = 1.0f / sqrtf((float)S);
        const uint32_t ratio = rank / groups; /* value heads per key head: repeat-interleave */
        float *delta = (float *)calloc(S, sizeof(float));
        if (!delta) { snprintf(error, cap, "delta allocation failed"); goto done; }
        for (uint32_t h = 0; h < rank; ++h) {
            const uint32_t kh = h / ratio;
            const float *qh = q + (size_t)kh * S, *khv = k + (size_t)kh * S, *vh = v + (size_t)h * S;
            float *mh = rec_state + (size_t)h * matrix;
            float *oh = core + (size_t)h * S;
            const float decay = expf(gate[h]);
            for (uint32_t j = 0; j < S; ++j) { float *row = mh + (size_t)j * S; for (uint32_t i = 0; i < S; ++i) row[i] *= decay; }
            for (uint32_t j = 0; j < S; ++j) {
                float sum = 0.0f; const float *row = mh + (size_t)j * S;
                for (uint32_t i = 0; i < S; ++i) sum += row[i] * khv[i];
                delta[j] = (vh[j] - sum) * beta[h];
            }
            for (uint32_t j = 0; j < S; ++j) { float *row = mh + (size_t)j * S; const float d = delta[j]; for (uint32_t i = 0; i < S; ++i) row[i] += khv[i] * d; }
            for (uint32_t j = 0; j < S; ++j) {
                float sum = 0.0f; const float *row = mh + (size_t)j * S;
                for (uint32_t i = 0; i < S; ++i) sum += row[i] * qh[i];
                oh[j] = sum * scale;
            }
        }
        free(delta);
    }
    /* gated RMSNorm per value head with SiLU(z) then Q4_K output projection */
    {
        const float *w = f32_data(e, t->ssm_norm);
        for (uint32_t h = 0; h < rank; ++h) {
            const float *x = core + (size_t)h * head_v, *g = z + (size_t)h * head_v;
            float *y = ng + (size_t)h * head_v;
            double ss = 0.0;
            for (uint32_t i = 0; i < head_v; ++i) ss += (double)x[i] * x[i];
            const float inv = 1.0f / sqrtf((float)(ss / head_v) + in->rms_eps);
            for (uint32_t i = 0; i < head_v; ++i) { const float silu = g[i] / (1.0f + expf(-g[i])); y[i] = x[i] * inv * w[i] * silu; }
        }
    }
    if (!matvec(e, t->ssm_out, ng, out, error, cap)) goto done;
    memcpy(conv_state, next_conv, conv_count * sizeof(float));
    (void)hidden;
    ok = 1;
done:
    free(qkv); free(z); free(ba); free(beta); free(gate); free(conv_silu); free(q); free(k); free(core); free(ng); free(next_conv);
    return ok;
}

/* ---- full attention layer (stateful KV cache) ---- */

static void rope_neox(float *x, uint32_t rope_dims, uint32_t position, float freq_base) {
    const uint32_t half = rope_dims / 2u;
    for (uint32_t i = 0; i < half; ++i) {
        const float theta = (float)position * powf(freq_base, -2.0f * (float)i / (float)rope_dims);
        const float c = cosf(theta), s = sinf(theta);
        const float x0 = x[i], x1 = x[half + i];
        x[i] = x0 * c - x1 * s;
        x[half + i] = x0 * s + x1 * c;
    }
}

static int attention_branch(rl_engine *e, rl_backend_state *s, const rl_layer_tensors *t, const float *normed,
        float *out, char *error, size_t cap) {
    const rl_engine_info *in = &e->info;
    const uint32_t head_dim = in->head_dim, qh_count = in->n_head, kvh_count = in->n_head_kv;
    const uint32_t qcount = qh_count * head_dim, kvcount = kvh_count * head_dim;
    const uint32_t position = s->position, seq_len = position + 1u;
    const size_t kv_layer = (size_t)in->context * kvcount;
    float *kcache = s->kcache + (size_t)t->attention_index * kv_layer;
    float *vcache = s->vcache + (size_t)t->attention_index * kv_layer;

    float *qgate_raw = (float *)calloc(2u * qcount, sizeof(float));
    float *key_raw = (float *)calloc(kvcount, sizeof(float));
    float *value = (float *)calloc(kvcount, sizeof(float));
    float *query = (float *)calloc(qcount, sizeof(float));
    float *gate = (float *)calloc(qcount, sizeof(float));
    float *key = (float *)calloc(kvcount, sizeof(float));
    float *gated = (float *)calloc(qcount, sizeof(float));
    double *scores = (double *)calloc(seq_len, sizeof(double));
    int ok = 0;
    if (!qgate_raw || !key_raw || !value || !query || !gate || !key || !gated || !scores) {
        snprintf(error, cap, "attention scratch allocation failed"); goto done;
    }
    if (!matvec(e, t->q, normed, qgate_raw, error, cap) || !matvec(e, t->k, normed, key_raw, error, cap) ||
        !matvec(e, t->v, normed, value, error, cap)) goto done;
    {
        const float *qn = f32_data(e, t->q_norm), *kn = f32_data(e, t->k_norm);
        for (uint32_t h = 0; h < qh_count; ++h) {
            const float *src = qgate_raw + (size_t)h * head_dim * 2u;
            rmsnorm_d(src, qn, head_dim, in->rms_eps, query + (size_t)h * head_dim);
            memcpy(gate + (size_t)h * head_dim, src + head_dim, (size_t)head_dim * sizeof(float));
            rope_neox(query + (size_t)h * head_dim, in->rope_dims, position, in->rope_freq_base);
        }
        for (uint32_t h = 0; h < kvh_count; ++h) {
            rmsnorm_d(key_raw + (size_t)h * head_dim, kn, head_dim, in->rms_eps, key + (size_t)h * head_dim);
            rope_neox(key + (size_t)h * head_dim, in->rope_dims, position, in->rope_freq_base);
        }
    }
    memcpy(kcache + (size_t)position * kvcount, key, (size_t)kvcount * sizeof(float));
    memcpy(vcache + (size_t)position * kvcount, value, (size_t)kvcount * sizeof(float));
    {
        const double scale = 1.0 / sqrt((double)head_dim);
        for (uint32_t h = 0; h < qh_count; ++h) {
            const uint32_t kvh = h / (qh_count / kvh_count);
            const float *qv = query + (size_t)h * head_dim;
            double max_score = -INFINITY;
            for (uint32_t pos = 0; pos < seq_len; ++pos) {
                const float *kv = kcache + ((size_t)pos * kvh_count + kvh) * head_dim;
                double dot = 0.0;
                for (uint32_t i = 0; i < head_dim; ++i) dot += (double)qv[i] * kv[i];
                scores[pos] = dot * scale;
                if (scores[pos] > max_score) max_score = scores[pos];
            }
            double denom = 0.0;
            for (uint32_t pos = 0; pos < seq_len; ++pos) { scores[pos] = exp(scores[pos] - max_score); denom += scores[pos]; }
            for (uint32_t i = 0; i < head_dim; ++i) {
                double acc = 0.0;
                for (uint32_t pos = 0; pos < seq_len; ++pos) acc += (scores[pos] / denom) * vcache[((size_t)pos * kvh_count + kvh) * head_dim + i];
                const size_t index = (size_t)h * head_dim + i;
                gated[index] = (float)(acc / (1.0 + exp(-(double)gate[index])));
            }
        }
    }
    if (!matvec(e, t->o, gated, out, error, cap)) goto done;
    ok = 1;
done:
    free(qgate_raw); free(key_raw); free(value); free(query); free(gate); free(key); free(gated); free(scores);
    return ok;
}

/* ---- MoE FFN: router + routed experts + gated shared expert ---- */

static int ffn(rl_engine *e, rl_backend_state *s, uint32_t layer, const rl_layer_tensors *t, const float *normed,
        float *out, rl_engine_step_stats *stats, char *error, size_t cap) {
    const rl_engine_info *in = &e->info;
    const uint32_t hidden = in->hidden, experts = in->n_expert, topk = in->top_k;
    const uint32_t ffn_size = (uint32_t)t->sh_gate->shape[1];
    float *logits = (float *)calloc(experts, sizeof(float));
    float *probs = (float *)calloc(experts, sizeof(float));
    float *sh_gate = (float *)calloc(ffn_size, sizeof(float));
    float *sh_up = (float *)calloc(ffn_size, sizeof(float));
    float *act = (float *)calloc(ffn_size, sizeof(float));
    float *sh_out = (float *)calloc(hidden, sizeof(float));
    double *routed = (double *)calloc(hidden, sizeof(double));
    uint32_t ids[RL_ENGINE_MAX_TOPK];
    float weights[RL_ENGINE_MAX_TOPK];
    int ok = 0;
    if (!logits || !probs || !sh_gate || !sh_up || !act || !sh_out || !routed) { snprintf(error, cap, "FFN scratch allocation failed"); goto done; }

    double t0 = rl_engine_now_ms();
    {
        const float *w = f32_data(e, t->router);
        for (uint32_t x = 0; x < experts; ++x) {
            double acc = 0.0;
            const float *row = w + (size_t)x * hidden;
            for (uint32_t i = 0; i < hidden; ++i) acc += (double)row[i] * normed[i];
            logits[x] = (float)acc;
        }
    }
    if (!rl_native_router_select_softmax_topk(logits, experts, topk, ids, weights, probs, error, cap)) goto done;
    memcpy(s->router_ids + (size_t)layer * RL_ENGINE_MAX_TOPK, ids, (size_t)topk * sizeof(uint32_t));
    double t1 = rl_engine_now_ms();
    stats->router_ms += t1 - t0;

    double routed_ms = 0.0;
    if (!rl_native_reference_topk(e->gguf.path, &e->expert_map, layer, ids, weights, topk, 0u, hidden,
            normed, hidden, routed, hidden, &routed_ms, error, cap)) goto done;
    double t2 = rl_engine_now_ms();
    stats->routed_ms += t2 - t1;

    /* shared expert: down(SiLU(gate x) * up x) * sigmoid(gate_inp . x) */
    if (!matvec(e, t->sh_gate, normed, sh_gate, error, cap) || !matvec(e, t->sh_up, normed, sh_up, error, cap)) goto done;
    for (uint32_t i = 0; i < ffn_size; ++i) {
        const double g = sh_gate[i];
        act[i] = (float)((g / (1.0 + exp(-g))) * (double)sh_up[i]);
    }
    if (!matvec(e, t->sh_down, act, sh_out, error, cap)) goto done;
    {
        const float *gw = f32_data(e, t->sh_gate_inp);
        double acc = 0.0;
        for (uint32_t i = 0; i < hidden; ++i) acc += (double)gw[i] * normed[i];
        const double scalar = 1.0 / (1.0 + exp(-acc));
        for (uint32_t i = 0; i < hidden; ++i) out[i] = (float)(routed[i] + (double)sh_out[i] * scalar);
    }
    stats->shared_ms += rl_engine_now_ms() - t2;
    ok = 1;
done:
    free(logits); free(probs); free(sh_gate); free(sh_up); free(act); free(sh_out); free(routed);
    return ok;
}

/* ---- full step ---- */

int rl_engine_cpu_step(rl_engine *e, uint32_t token, float *logits, rl_engine_step_stats *stats, char *error, size_t cap) {
    rl_backend_state *s = &e->cpu;
    const rl_engine_info *in = &e->info;
    const uint32_t hidden = in->hidden;
    const double start = rl_engine_now_ms();
    float *x = (float *)calloc(hidden, sizeof(float));
    float *normed = (float *)calloc(hidden, sizeof(float));
    float *branch = (float *)calloc(hidden, sizeof(float));
    float *resid = (float *)calloc(hidden, sizeof(float));
    float *ffn_out = (float *)calloc(hidden, sizeof(float));
    int ok = 0;
    if (!x || !normed || !branch || !resid || !ffn_out) { snprintf(error, cap, "step scratch allocation failed"); goto done; }
    if (!rl_engine_embed_token(e, token, x, error, cap)) goto done;
    memcpy(s->embed, x, (size_t)hidden * sizeof(float));
    stats->embed_ms = rl_engine_now_ms() - start;

    for (uint32_t l = 0; l < in->n_layer; ++l) {
        const rl_layer_tensors *t = &e->layers[l];
        const double l0 = rl_engine_now_ms();
        rmsnorm_d(x, f32_data(e, t->attn_norm), hidden, in->rms_eps, normed);
        if (t->kind == RL_LAYER_MAP_RECURRENT) {
            if (!deltanet_branch(e, s, t, normed, branch, error, cap)) goto done;
            stats->recurrent_ms += rl_engine_now_ms() - l0;
        } else {
            if (!attention_branch(e, s, t, normed, branch, error, cap)) goto done;
            stats->attention_ms += rl_engine_now_ms() - l0;
        }
        {
            double ss = 0.0;
            const float *pw = f32_data(e, t->post_norm);
            for (uint32_t i = 0; i < hidden; ++i) { resid[i] = x[i] + branch[i]; ss += (double)resid[i] * resid[i]; }
            const double inv = 1.0 / sqrt(ss / (double)hidden + (double)in->rms_eps);
            for (uint32_t i = 0; i < hidden; ++i) normed[i] = (float)((double)resid[i] * inv * pw[i]);
        }
        if (!ffn(e, s, l, t, normed, ffn_out, stats, error, cap)) goto done;
        for (uint32_t i = 0; i < hidden; ++i) x[i] = resid[i] + ffn_out[i];
        memcpy(s->layer_out + (size_t)l * hidden, x, (size_t)hidden * sizeof(float));
    }
    stats->layers_ms = rl_engine_now_ms() - start - stats->embed_ms;

    const double o0 = rl_engine_now_ms();
    rmsnorm_d(x, f32_data(e, e->output_norm), hidden, in->rms_eps, s->final_norm);
    if (logits && !matvec(e, e->output, s->final_norm, logits, error, cap)) goto done;
    stats->output_ms = rl_engine_now_ms() - o0;
    s->position++;
    ok = 1;
done:
    stats->total_ms = rl_engine_now_ms() - start;
    free(x); free(normed); free(branch); free(resid); free(ffn_out);
    return ok;
}
