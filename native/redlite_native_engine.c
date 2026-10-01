#define _FILE_OFFSET_BITS 64
#define _POSIX_C_SOURCE 200809L
#define _DARWIN_C_SOURCE 1

#include "redlite_native_engine_internal.h"
#include "redlite_native_quant_cpu.h"
#include "redlite_native_iq3.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static void set_error(char *error, size_t cap, const char *msg) {
    if (error && cap) snprintf(error, cap, "%s", msg ? msg : "unknown engine error");
}

double rl_engine_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

void rl_engine_config_default(rl_engine_config *cfg) {
    if (!cfg) return;
    memset(cfg, 0, sizeof(*cfg));
    cfg->context = 4096u;
    cfg->cache_mib = 4096u;
    cfg->top_k = 0u;
    cfg->enable_cpu = 0;
    cfg->enable_gpu = 1;
    cfg->cpu_threads = 0;
}

size_t rl_engine_conv_count(const rl_engine *e) { return (size_t)(e->info.d_conv - 1u) * e->info.channels; }
size_t rl_engine_rec_count(const rl_engine *e) { return (size_t)e->info.dt_rank * e->info.head_v * e->info.head_v; }
size_t rl_engine_kv_row_count(const rl_engine *e) { return (size_t)e->info.n_head_kv * e->info.head_dim; }

int rl_backend_state_alloc(rl_engine *e, rl_backend_state *s, int host_state, char *error, size_t cap) {
    memset(s, 0, sizeof(*s));
    s->host_state = host_state ? 1 : 0;
    const size_t conv = host_state ? rl_engine_conv_count(e) * e->info.n_recurrent : 0u;
    const size_t rec = host_state ? rl_engine_rec_count(e) * e->info.n_recurrent : 0u;
    const size_t kv = host_state ? rl_engine_kv_row_count(e) * (size_t)e->info.context * e->info.n_attention : 0u;
    s->conv = (float *)calloc(conv ? conv : 1u, sizeof(float));
    s->rec = (float *)calloc(rec ? rec : 1u, sizeof(float));
    s->kcache = (float *)calloc(kv ? kv : 1u, sizeof(float));
    s->vcache = (float *)calloc(kv ? kv : 1u, sizeof(float));
    s->embed = (float *)calloc(e->info.hidden, sizeof(float));
    s->layer_out = (float *)calloc((size_t)e->info.n_layer * e->info.hidden, sizeof(float));
    s->final_norm = (float *)calloc(e->info.hidden, sizeof(float));
    s->router_ids = (uint32_t *)calloc((size_t)e->info.n_layer * RL_ENGINE_MAX_TOPK, sizeof(uint32_t));
    if (!s->conv || !s->rec || !s->kcache || !s->vcache || !s->embed || !s->layer_out || !s->final_norm || !s->router_ids) {
        rl_backend_state_free(s);
        set_error(error, cap, "backend state allocation failed");
        return 0;
    }
    return 1;
}

void rl_backend_state_free(rl_backend_state *s) {
    if (!s) return;
    free(s->conv); free(s->rec); free(s->kcache); free(s->vcache);
    free(s->embed); free(s->layer_out); free(s->final_norm); free(s->router_ids);
    memset(s, 0, sizeof(*s));
}

void rl_backend_state_reset(rl_engine *e, rl_backend_state *s) {
    s->position = 0;
    if (!s->host_state) return;
    memset(s->conv, 0, rl_engine_conv_count(e) * e->info.n_recurrent * sizeof(float));
    memset(s->rec, 0, rl_engine_rec_count(e) * e->info.n_recurrent * sizeof(float));
    const size_t kv = rl_engine_kv_row_count(e) * (size_t)e->info.context * e->info.n_attention;
    memset(s->kcache, 0, kv * sizeof(float));
    memset(s->vcache, 0, kv * sizeof(float));
}

/* ---- tensor audit ---- */

static int expect(const rl_gguf_tensor *t, const char *what, uint32_t type, uint32_t dims,
        uint64_t d0, uint64_t d1, char *error, size_t cap) {
    if (!t) { snprintf(error, cap, "missing tensor %s", what); return 0; }
    if (t->ggml_type != type || t->n_dims != dims || t->shape[0] != d0 || (dims > 1 && t->shape[1] != d1)) {
        snprintf(error, cap, "tensor %s has type %s dims %u shape (%llu,%llu); expected %s (%llu,%llu)",
            t->name, rl_gguf_type_name(t->ggml_type), t->n_dims,
            (unsigned long long)t->shape[0], (unsigned long long)(t->n_dims > 1 ? t->shape[1] : 1),
            rl_gguf_type_name(type), (unsigned long long)d0, (unsigned long long)d1);
        return 0;
    }
    if (!t->payload_bytes || t->payload_bytes > t->span_bytes) {
        snprintf(error, cap, "tensor %s payload/span mismatch", t->name);
        return 0;
    }
    return 1;
}

static int expect_dense(const rl_gguf_tensor *t, const char *what, uint32_t dims, uint64_t d0, uint64_t d1,
        char *error, size_t cap) {
    if (!t) { snprintf(error, cap, "missing tensor %s", what); return 0; }
    const uint32_t ty = t->ggml_type;
    const int supported = ty == 0u || ty == 1u || ty == 8u || ty == 10u || ty == 12u || ty == 13u || ty == 14u || ty == 16u ||
                          rl_iq3_supported(ty);   /* dev31: IQ3_XXS, IQ3_S, IQ2_S, IQ4_XS */
    if (!supported) { snprintf(error, cap, "tensor %s has unsupported dense type %u", t->name, ty); return 0; }
    return expect(t, what, ty, dims, d0, d1, error, cap);
}

static int audit_layers(rl_engine *e, char *error, size_t cap) {
    const rl_gguf_model *g = &e->gguf;
    const rl_engine_info *in = &e->info;
    uint32_t r = 0, a = 0;
    e->layers = (rl_layer_tensors *)calloc(in->n_layer, sizeof(*e->layers));
    if (!e->layers) { set_error(error, cap, "layer table allocation failed"); return 0; }
    for (uint32_t l = 0; l < in->n_layer; ++l) {
        rl_layer_tensors *t = &e->layers[l];
        t->kind = e->layer_map.kind[l];
        t->recurrent_index = UINT32_MAX;
        t->attention_index = UINT32_MAX;
        t->attn_norm = rl_gguf_find_layer(g, l, "attn_norm.weight");
        t->post_norm = rl_gguf_find_layer(g, l, "post_attention_norm.weight");
        if (!expect(t->attn_norm, "attn_norm", 0u, 1u, in->hidden, 0, error, cap) ||
            !expect(t->post_norm, "post_attention_norm", 0u, 1u, in->hidden, 0, error, cap)) return 0;
        if (t->kind == RL_LAYER_MAP_RECURRENT) {
            t->recurrent_index = r++;
            t->qkv = rl_gguf_find_layer(g, l, "attn_qkv.weight");
            t->z = rl_gguf_find_layer(g, l, "attn_gate.weight");
            t->ba = rl_gguf_find_layer(g, l, "ssm_ba.weight");
            t->conv = rl_gguf_find_layer(g, l, "ssm_conv1d.weight");
            t->dt = rl_gguf_find_layer(g, l, "ssm_dt.bias");
            t->a = rl_gguf_find_layer(g, l, "ssm_a");
            t->ssm_norm = rl_gguf_find_layer(g, l, "ssm_norm.weight");
            t->ssm_out = rl_gguf_find_layer(g, l, "ssm_out.weight");
            if (!expect_dense(t->qkv, "attn_qkv", 2u, in->hidden, in->channels, error, cap) ||
                !expect_dense(t->z, "attn_gate", 2u, in->hidden, in->d_inner, error, cap) ||
                !expect_dense(t->ba, "ssm_ba", 2u, in->hidden, 2u * in->dt_rank, error, cap) ||
                !expect(t->conv, "ssm_conv1d", 0u, 2u, in->d_conv, in->channels, error, cap) ||
                !expect(t->dt, "ssm_dt", 0u, 1u, in->dt_rank, 0, error, cap) ||
                !expect(t->a, "ssm_a", 0u, 1u, in->dt_rank, 0, error, cap) ||
                !expect(t->ssm_norm, "ssm_norm", 0u, 1u, in->head_v, 0, error, cap) ||
                !expect_dense(t->ssm_out, "ssm_out", 2u, in->d_inner, in->hidden, error, cap)) return 0;
        } else if (t->kind == RL_LAYER_MAP_FULL_ATTENTION) {
            t->attention_index = a++;
            t->q = rl_gguf_find_layer(g, l, "attn_q.weight");
            t->k = rl_gguf_find_layer(g, l, "attn_k.weight");
            t->v = rl_gguf_find_layer(g, l, "attn_v.weight");
            t->q_norm = rl_gguf_find_layer(g, l, "attn_q_norm.weight");
            t->k_norm = rl_gguf_find_layer(g, l, "attn_k_norm.weight");
            t->o = rl_gguf_find_layer(g, l, "attn_output.weight");
            const uint64_t qcount = (uint64_t)in->n_head * in->head_dim;
            const uint64_t kvcount = (uint64_t)in->n_head_kv * in->head_dim;
            if (!expect_dense(t->q, "attn_q", 2u, in->hidden, 2u * qcount, error, cap) ||
                !expect_dense(t->k, "attn_k", 2u, in->hidden, kvcount, error, cap) ||
                !expect_dense(t->v, "attn_v", 2u, in->hidden, kvcount, error, cap) ||
                !expect(t->q_norm, "attn_q_norm", 0u, 1u, in->head_dim, 0, error, cap) ||
                !expect(t->k_norm, "attn_k_norm", 0u, 1u, in->head_dim, 0, error, cap) ||
                !expect_dense(t->o, "attn_output", 2u, qcount, in->hidden, error, cap)) return 0;
        } else {
            snprintf(error, cap, "layer %u has an unknown/mixed structure", l);
            return 0;
        }
        t->router = rl_gguf_find_layer(g, l, "ffn_gate_inp.weight");
        t->sh_gate_inp = rl_gguf_find_layer(g, l, "ffn_gate_inp_shexp.weight");
        t->sh_gate = rl_gguf_find_layer(g, l, "ffn_gate_shexp.weight");
        t->sh_up = rl_gguf_find_layer(g, l, "ffn_up_shexp.weight");
        t->sh_down = rl_gguf_find_layer(g, l, "ffn_down_shexp.weight");
        if (!expect(t->router, "ffn_gate_inp", 0u, 2u, in->hidden, in->n_expert, error, cap) ||
            !expect(t->sh_gate_inp, "ffn_gate_inp_shexp", 0u, 1u, in->hidden, 0, error, cap) ||
            !expect_dense(t->sh_gate, "ffn_gate_shexp", 2u, in->hidden, e->gguf.n_ff_shexp, error, cap) ||
            !expect_dense(t->sh_up, "ffn_up_shexp", 2u, in->hidden, e->gguf.n_ff_shexp, error, cap) ||
            !expect_dense(t->sh_down, "ffn_down_shexp", 2u, e->gguf.n_ff_shexp, in->hidden, error, cap)) return 0;
    }
    if (r != in->n_recurrent || a != in->n_attention) { set_error(error, cap, "layer kind counts diverged from the audit"); return 0; }
    return 1;
}

static uint64_t dense_bytes(const rl_engine *e) {
    uint64_t total = e->tok_embd->payload_bytes + e->output_norm->payload_bytes + e->output->payload_bytes;
    for (uint32_t l = 0; l < e->info.n_layer; ++l) {
        const rl_layer_tensors *t = &e->layers[l];
        const rl_gguf_tensor *all[] = {t->attn_norm, t->post_norm, t->qkv, t->z, t->ba, t->conv, t->dt, t->a, t->ssm_norm, t->ssm_out,
            t->q, t->k, t->v, t->q_norm, t->k_norm, t->o, t->router, t->sh_gate_inp, t->sh_gate, t->sh_up, t->sh_down};
        for (size_t i = 0; i < sizeof(all) / sizeof(all[0]); ++i) if (all[i]) total += all[i]->payload_bytes;
    }
    return total;
}

rl_engine *rl_engine_open(const char *model_path, const rl_engine_config *cfg_in, char *error, size_t cap) {
    if (!model_path) { set_error(error, cap, "model path required"); return NULL; }
    rl_engine *e = (rl_engine *)calloc(1, sizeof(*e));
    if (!e) { set_error(error, cap, "engine allocation failed"); return NULL; }
    if (cfg_in) e->cfg = *cfg_in; else rl_engine_config_default(&e->cfg);
    if (!e->cfg.context) e->cfg.context = 4096u;
    if (!e->cfg.cache_mib) e->cfg.cache_mib = 4096u;

    if (!rl_gguf_model_open(model_path, &e->gguf, error, cap) || !rl_gguf_model_map(&e->gguf, error, cap)) {
        rl_engine_close(e); return NULL;
    }
    const rl_gguf_model *g = &e->gguf;
    if (!g->architecture || strcmp(g->architecture, "qwen3next") != 0) {
        set_error(error, cap, "GGUF architecture is not qwen3next"); rl_engine_close(e); return NULL;
    }
    if (!g->n_layer || !g->n_embd || !g->n_head || !g->n_head_kv || !g->key_length || !g->rope_dims || !g->n_expert ||
        !g->n_expert_used || g->ssm_conv < 2u || !g->ssm_state || !g->ssm_group || !g->ssm_dt_rank || !g->ssm_inner ||
        !g->n_ff_shexp || g->rms_eps <= 0.0f || g->rope_freq_base <= 0.0f || g->key_length != g->value_length ||
        g->n_head % g->n_head_kv || g->ssm_dt_rank % g->ssm_group || g->ssm_inner % g->ssm_dt_rank || g->vocab_count == 0) {
        set_error(error, cap, "incomplete or inconsistent qwen3next metadata"); rl_engine_close(e); return NULL;
    }
    rl_engine_info *in = &e->info;
    in->n_layer = g->n_layer;
    in->hidden = g->n_embd;
    in->vocab = g->vocab_count;
    in->n_expert = g->n_expert;
    in->top_k = e->cfg.top_k ? e->cfg.top_k : g->n_expert_used;
    in->context = e->cfg.context;
    in->d_conv = g->ssm_conv;
    in->d_inner = g->ssm_inner;
    in->d_state = g->ssm_state;
    in->n_group = g->ssm_group;
    in->dt_rank = g->ssm_dt_rank;
    in->head_v = g->ssm_inner / g->ssm_dt_rank;
    in->channels = 2u * g->ssm_state * g->ssm_group + g->ssm_inner;
    in->n_head = g->n_head;
    in->n_head_kv = g->n_head_kv;
    in->head_dim = g->key_length;
    in->rope_dims = g->rope_dims;
    in->rope_freq_base = g->rope_freq_base;
    in->rms_eps = g->rms_eps;
    if (in->head_v != in->d_state) { set_error(error, cap, "DeltaNet head_v != state size"); rl_engine_close(e); return NULL; }
    if (in->top_k == 0 || in->top_k > RL_ENGINE_MAX_TOPK || in->top_k > in->n_expert) {
        set_error(error, cap, "top-k out of range"); rl_engine_close(e); return NULL;
    }
    if (in->rope_dims > in->head_dim || in->rope_dims % 2u) { set_error(error, cap, "unsupported RoPE dims"); rl_engine_close(e); return NULL; }

    if (!rl_native_layer_map_audit(model_path, &e->layer_map, error, cap)) { rl_engine_close(e); return NULL; }
    if (e->layer_map.layer_count != in->n_layer) { set_error(error, cap, "layer map count != block_count"); rl_engine_close(e); return NULL; }
    in->n_recurrent = e->layer_map.recurrent_count;
    in->n_attention = e->layer_map.full_attention_count;

    e->tok_embd = rl_gguf_find(g, "token_embd.weight");
    e->output_norm = rl_gguf_find(g, "output_norm.weight");
    e->output = rl_gguf_find(g, "output.weight");
    if (!e->output) e->output = e->tok_embd; /* tied embeddings fallback (pinned llama.cpp semantics) */
    if (!expect_dense(e->tok_embd, "token_embd", 2u, in->hidden, in->vocab, error, cap) ||
        !expect(e->output_norm, "output_norm", 0u, 1u, in->hidden, 0, error, cap) ||
        !expect_dense(e->output, "output", 2u, in->hidden, in->vocab, error, cap)) { rl_engine_close(e); return NULL; }

    if (!audit_layers(e, error, cap)) { rl_engine_close(e); return NULL; }
    if (!rl_native_build_expert_map(model_path, in->n_expert, &e->expert_map, error, cap)) { rl_engine_close(e); return NULL; }
    if (e->cfg.cache_mib == RL_ENGINE_CACHE_FULL) e->cfg.cache_mib = rl_engine_full_residency_mib(e);   /* dev31 */
    if (!e->expert_map.all_slice_safe || e->expert_map.layer_count != in->n_layer) {
        set_error(error, cap, "routed expert map is not slice-safe for every layer"); rl_engine_close(e); return NULL;
    }
    if (!rl_native_iq2_xxs_build_grid(e->iq2_grid, error, cap)) { rl_engine_close(e); return NULL; }

    in->dense_bytes = dense_bytes(e);
    in->state_bytes = (uint64_t)(rl_engine_conv_count(e) + rl_engine_rec_count(e)) * in->n_recurrent * sizeof(float) +
        (uint64_t)rl_engine_kv_row_count(e) * in->context * in->n_attention * 2u * sizeof(float);

    e->cpu_threads = e->cfg.cpu_threads > 0 ? e->cfg.cpu_threads : (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (e->cpu_threads < 1) e->cpu_threads = 1;
    if (e->cpu_threads > 64) e->cpu_threads = 64;

    if (e->cfg.enable_cpu) {
        if (!rl_backend_state_alloc(e, &e->cpu, 1, error, cap)) { rl_engine_close(e); return NULL; }
        e->cpu_enabled = 1;
    }
    if (e->cfg.enable_gpu) {
#ifdef __APPLE__
        if (!rl_backend_state_alloc(e, &e->gpu, 0, error, cap)) { rl_engine_close(e); return NULL; }
        e->metal = rl_metal_engine_create(e, error, cap);
        if (!e->metal) { rl_engine_close(e); return NULL; }
        e->gpu_enabled = 1;
#else
        set_error(error, cap, "the Metal backend requires macOS"); rl_engine_close(e); return NULL;
#endif
    }
    if (error && cap) error[0] = '\0';
    return e;
}

void rl_engine_close(rl_engine *e) {
    if (!e) return;
#ifdef __APPLE__
    if (e->metal) rl_metal_engine_destroy(e->metal);
#endif
    rl_backend_state_free(&e->cpu);
    rl_backend_state_free(&e->gpu);
    rl_native_free_expert_map(&e->expert_map);
    free(e->layers);
    rl_gguf_model_close(&e->gguf);
    free(e);
}

const rl_engine_info *rl_engine_info_get(const rl_engine *e) { return e ? &e->info : NULL; }

static rl_backend_state *state_for(rl_engine *e, rl_engine_backend b) {
    if (b == RL_BACKEND_CPU) return e->cpu_enabled ? &e->cpu : NULL;
    if (b == RL_BACKEND_GPU) return e->gpu_enabled ? &e->gpu : NULL;
    return NULL;
}

int rl_engine_reset(rl_engine *e, rl_engine_backend b, char *error, size_t cap) {
    rl_backend_state *s = e ? state_for(e, b) : NULL;
    if (!s) { set_error(error, cap, "backend not enabled"); return 0; }
    rl_backend_state_reset(e, s);
#ifdef __APPLE__
    if (b == RL_BACKEND_GPU && e->metal && !rl_metal_engine_reset(e->metal, error, cap)) return 0;
#endif
    return 1;
}

uint32_t rl_engine_position(const rl_engine *e, rl_engine_backend b) {
    const rl_backend_state *s = e ? state_for((rl_engine *)e, b) : NULL;
    return s ? s->position : 0u;
}

static void stats_accumulate(rl_engine_step_stats *acc, const rl_engine_step_stats *one) {
    acc->embed_ms += one->embed_ms; acc->layers_ms += one->layers_ms; acc->recurrent_ms += one->recurrent_ms;
    acc->attention_ms += one->attention_ms; acc->router_ms += one->router_ms; acc->routed_ms += one->routed_ms;
    acc->shared_ms += one->shared_ms; acc->output_ms += one->output_ms; acc->total_ms += one->total_ms;
    acc->routed_load_ms += one->routed_load_ms; acc->routed_gpu_ms += one->routed_gpu_ms; acc->gpu_ms += one->gpu_ms;
    acc->expert_loads = one->expert_loads; acc->cache_hits = one->cache_hits; acc->cache_misses = one->cache_misses;
    acc->ssd_bytes = one->ssd_bytes; acc->ssd_reads = one->ssd_reads; acc->resident_slots = one->resident_slots;
    acc->slot_capacity = one->slot_capacity; acc->expert_plans += one->expert_plans;
    acc->prep_lru_ms += one->prep_lru_ms; acc->prep_load_ms += one->prep_load_ms; acc->prep_commit_ms += one->prep_commit_ms; acc->expert_wait_ms += one->expert_wait_ms; acc->prefetch_ms += one->prefetch_ms;
    acc->speculative += one->speculative; acc->speculative_fallback += one->speculative_fallback;
}

int rl_engine_prefill(rl_engine *e, rl_engine_backend b, const uint32_t *tokens, uint32_t count,
                      float *logits, rl_engine_step_stats *stats, char *error, size_t cap) {
    if (!e) { set_error(error, cap, "engine required"); return 0; }
    rl_backend_state *s = state_for(e, b);
    if (!s) { set_error(error, cap, "backend not enabled"); return 0; }
    if (!tokens || !count) { set_error(error, cap, "prefill needs at least one token"); return 0; }
    for (uint32_t i = 0; i < count; ++i) if (tokens[i] >= e->info.vocab) { set_error(error, cap, "token id out of range"); return 0; }
    if (count > e->info.context - s->position) { set_error(error, cap, "context capacity exhausted"); return 0; }
    rl_engine_step_stats local;
    if (!stats) stats = &local;
    memset(stats, 0, sizeof(*stats));
    const uint32_t batch = rl_engine_prefill_batch(e);
    if (b == RL_BACKEND_CPU || batch == 1u) {
        for (uint32_t i = 0; i < count; ++i) {
            rl_engine_step_stats one;
            memset(&one, 0, sizeof(one));
            const int ok = b == RL_BACKEND_CPU
                ? rl_engine_cpu_step(e, tokens[i], i + 1u == count ? logits : NULL, &one, error, cap)
#ifdef __APPLE__
                : rl_metal_engine_step(e, e->metal, tokens[i], i + 1u == count ? logits : NULL, &one, error, cap);
#else
                : 0;
#endif
            if (!ok) return 0;
            stats_accumulate(stats, &one);
        }
        return 1;
    }
#ifdef __APPLE__
    return rl_metal_engine_prefill(e, e->metal, tokens, count, logits, stats, error, cap);
#else
    set_error(error, cap, "Metal backend unavailable"); return 0;
#endif
}

int rl_engine_step(rl_engine *e, rl_engine_backend b, uint32_t token, float *logits,
                   rl_engine_step_stats *stats, char *error, size_t cap) {
    if (!e) { set_error(error, cap, "engine required"); return 0; }
    rl_backend_state *s = state_for(e, b);
    if (!s) { set_error(error, cap, "backend not enabled"); return 0; }
    if (token >= e->info.vocab) { set_error(error, cap, "token id out of range"); return 0; }
    if (s->position >= e->info.context) { set_error(error, cap, "context capacity exhausted"); return 0; }
    rl_engine_step_stats local;
    if (!stats) stats = &local;
    memset(stats, 0, sizeof(*stats));
    if (b == RL_BACKEND_CPU) return rl_engine_cpu_step(e, token, logits, stats, error, cap);
#ifdef __APPLE__
    return rl_metal_engine_step(e, e->metal, token, logits, stats, error, cap);
#else
    set_error(error, cap, "Metal backend unavailable"); return 0;
#endif
}

const float *rl_engine_last_embedding(const rl_engine *e, rl_engine_backend b) {
    const rl_backend_state *s = e ? state_for((rl_engine *)e, b) : NULL;
    return s ? s->embed : NULL;
}

const float *rl_engine_last_layer_output(const rl_engine *e, rl_engine_backend b, uint32_t layer) {
    const rl_backend_state *s = e ? state_for((rl_engine *)e, b) : NULL;
    if (!s || layer >= e->info.n_layer) return NULL;
    return s->layer_out + (size_t)layer * e->info.hidden;
}

const float *rl_engine_last_final_norm(const rl_engine *e, rl_engine_backend b) {
    const rl_backend_state *s = e ? state_for((rl_engine *)e, b) : NULL;
    return s ? s->final_norm : NULL;
}

const uint32_t *rl_engine_last_router_ids(const rl_engine *e, rl_engine_backend b, uint32_t layer) {
    const rl_backend_state *s = e ? state_for((rl_engine *)e, b) : NULL;
    if (!s || layer >= e->info.n_layer) return NULL;
    return s->router_ids + (size_t)layer * RL_ENGINE_MAX_TOPK;
}

double rl_engine_now_ms_public(void) { return rl_engine_now_ms(); }

uint64_t rl_engine_full_residency_mib(const rl_engine *e) {
    if (!e || !e->expert_map.max_expert_triplet_bytes) return 0;
    const uint64_t slot = (e->expert_map.max_expert_triplet_bytes + 4095u) & ~(uint64_t)4095u;   /* redmetal_topk slot alignment */
    uint64_t bytes = slot * e->expert_map.layer_count * e->expert_map.expert_count;
    /* dev37: slots of each layer's own size (rl_native_metal_create size classes), unless RL_POOL_CLASSES=0 */
    const char *env = getenv("RL_POOL_CLASSES");
    if (!env || atoi(env) != 0) {
        uint64_t per[256] = {0}, sum = 0;
        int ok = e->expert_map.layer_count <= 256u;
        for (uint32_t i = 0; ok && i < e->expert_map.routed_tensor_count; ++i) {
            if (e->expert_map.routed[i].layer >= 256u) ok = 0;
            else per[e->expert_map.routed[i].layer] += e->expert_map.routed[i].expert_stride_bytes;
        }
        for (uint32_t l = 0; ok && l < e->expert_map.layer_count; ++l) sum += (per[l] + 4095u) & ~(uint64_t)4095u;
        if (ok && sum) bytes = sum * e->expert_map.expert_count;
    }
    return (bytes + (1024u * 1024u) - 1u) / (1024u * 1024u);
}

int rl_engine_parse_cache_mib(const char *text, uint64_t *out) {
    if (!text || !out || !*text) return 0;
    if (strcmp(text, "full") == 0) { *out = RL_ENGINE_CACHE_FULL; return 1; }
    char *end = NULL;
    const unsigned long long v = strtoull(text, &end, 10);
    if (!end || *end || !v || v > 1048576ull) return 0;
    *out = (uint64_t)v;
    return 1;
}

uint32_t rl_engine_prefill_batch(const rl_engine *e) {
    if (!e) return 512u;
    if (e->cfg.prefill_batch) return e->cfg.prefill_batch;
    /* dev30: with every expert resident, 2048-token chunks give each expert tile more pairs (1100-token prompt, M4 Max:
     * ~663 tok/s in chunks of 512, ~811 in one chunk). dev34: also with a bounded cache, where every chunk reloads nearly
     * every expert of every layer: 8192 tokens at 4 GiB load 161 GB of experts in chunks of 512, 49 GB in chunks of 2048
     * (page-cache copies on a 48 GiB Mac, SSD reads on a 24 GiB one). Cost: ~0.26 MiB of scratch per chunk token
     * (+571 MiB footprint at 4 GiB with an 8387-token prompt). */
    (void)e;
    return 2048u;
}

int rl_engine_experts_preloaded(const rl_engine *e, double *preload_ms) {
    if (preload_ms) *preload_ms = 0.0;
#ifdef __APPLE__
    if (e && e->metal) return rl_metal_engine_preloaded(e->metal, preload_ms);
#endif
    (void)e;
    return 0;
}

int rl_engine_embed_token(const rl_engine *e, uint32_t token, float *out, char *error, size_t cap) {
    if (!e || !out) { set_error(error, cap, "invalid embed arguments"); return 0; }
    if (token >= e->info.vocab) { set_error(error, cap, "token id out of range"); return 0; }
    const size_t rb = rl_gguf_row_bytes(e->tok_embd->ggml_type, e->info.hidden);
    const uint8_t *row = rl_gguf_tensor_data(&e->gguf, e->tok_embd) + (size_t)token * rb;
    if (!rb || !rl_quant_dequant_row(e->tok_embd->ggml_type, row, e->info.hidden, e->iq2_grid, out)) {
        set_error(error, cap, "token embedding dequantization failed"); return 0;
    }
    return 1;
}
