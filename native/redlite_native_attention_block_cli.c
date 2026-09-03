#define main rl_dev16_attention_cli_unused
#include "redlite_native_attention_cli.c"
#undef main

#include "redlite_native_block_ops.h"
#include "redlite_native_gguf.h"
#include "redlite_native_metal.h"
#include "redlite_native_reference.h"
#include "redlite_native_router.h"
#include "redlite_native_router_exec.h"
#include "redlite_native_shared.h"

#define AB_MIB (1024ull * 1024ull)
#define AB_MAX_TOPK 64u
#define AB_MAX_SHARED (RL_SHARED_MAX_LAYERS * 4u)

typedef struct {
    uint64_t offset;
    uint64_t span;
    uint32_t type;
    uint32_t dims;
    uint64_t shape[RL_ATTN_PROJ_MAX_DIMS];
    int found;
} ab_tensor;

static int ab_find_tensor(const char *model, const char *wanted, ab_tensor *out,
        char *error, size_t cap) {
    memset(out, 0, sizeof(*out));
    FILE *f = fopen(model, "rb");
    if (!f) { snprintf(error, cap, "open failed: %s", strerror(errno)); return 0; }
    if (fseeko(f, 0, SEEK_END) != 0) { fclose(f); return 0; }
    const off_t end = ftello(f);
    if (end <= 0 || fseeko(f, 0, SEEK_SET) != 0) { fclose(f); return 0; }
    uint8_t magic[4];
    uint32_t version = 0;
    uint64_t tensor_count = 0, kv_count = 0;
    model_meta meta;
    if (!read_exact(f, magic, sizeof(magic)) || memcmp(magic, "GGUF", 4) != 0 ||
        !read_u32(f, &version) || (version != 2u && version != 3u) ||
        !read_u64(f, &tensor_count) || !read_u64(f, &kv_count) ||
        !read_metadata(f, kv_count, &meta) || tensor_count > SIZE_MAX / sizeof(raw_tensor)) {
        fclose(f); snprintf(error, cap, "GGUF parse failed"); return 0;
    }
    raw_tensor *raw = (raw_tensor *)calloc((size_t)tensor_count, sizeof(*raw));
    if (!raw) { fclose(f); snprintf(error, cap, "tensor directory allocation failed"); return 0; }
    for (uint64_t i = 0; i < tensor_count; ++i) {
        uint32_t dims = 0;
        if (!read_string(f, &raw[i].name) || !read_u32(f, &dims) || dims > RL_ATTN_PROJ_MAX_DIMS) {
            free_raw(raw, tensor_count); fclose(f); return 0;
        }
        raw[i].n_dims = dims;
        for (uint32_t d = 0; d < dims; ++d) if (!read_u64(f, &raw[i].shape[d])) {
            free_raw(raw, tensor_count); fclose(f); return 0;
        }
        if (!read_u32(f, &raw[i].ggml_type) || !read_u64(f, &raw[i].relative_offset)) {
            free_raw(raw, tensor_count); fclose(f); return 0;
        }
    }
    const off_t directory_end = ftello(f);
    fclose(f);
    const uint64_t data_base = directory_end < 0 ? 0 : round_up_u64((uint64_t)directory_end, meta.alignment);
    if (!data_base || data_base > (uint64_t)end) {
        free_raw(raw, tensor_count); snprintf(error, cap, "invalid GGUF data base"); return 0;
    }
    qsort(raw, (size_t)tensor_count, sizeof(*raw), tensor_cmp);
    for (uint64_t i = 0; i < tensor_count; ++i) {
        if (strcmp(raw[i].name, wanted) != 0) continue;
        const uint64_t next = i + 1u < tensor_count ? raw[i + 1u].relative_offset : (uint64_t)end - data_base;
        if (next < raw[i].relative_offset) break;
        out->offset = data_base + raw[i].relative_offset;
        out->span = next - raw[i].relative_offset;
        out->type = raw[i].ggml_type;
        out->dims = raw[i].n_dims;
        memcpy(out->shape, raw[i].shape, sizeof(out->shape));
        out->found = 1;
        break;
    }
    free_raw(raw, tensor_count);
    if (!out->found) { snprintf(error, cap, "tensor %s not found", wanted); return 0; }
    if (error && cap) error[0] = '\0';
    return 1;
}

static int ab_attention(const char *model, uint32_t layer, uint32_t position,
        const float *input, uint32_t hidden, float *cpu_output, float *gpu_output,
        error_stats *cache_error, double *cpu_ms, rl_attn_proj_telemetry *gpu_telemetry,
        char *error, size_t cap) {
    model_meta meta;
    rl_attn_proj_tensor_info tensors[RL_ATTN_PROJ_TENSOR_COUNT];
    size_t weight_bytes[RL_ATTN_PROJ_TENSOR_COUNT] = {0};
    if (!audit_layer(model, layer, &meta, tensors, error, cap) ||
        !validate_layout(&meta, tensors, weight_bytes, error, cap) || meta.hidden != hidden) return 0;

    const uint32_t head_dim = meta.key_length;
    const uint32_t query_count = meta.query_heads * head_dim;
    const uint32_t kv_count = meta.kv_heads * head_dim;
    const uint32_t qgate_count = 2u * query_count;
    const uint32_t seq_len = position + 1u;
    const size_t cache_count = (size_t)seq_len * kv_count;
    const size_t iq2_rb = iq2_row_bytes(hidden);
    const size_t q4_in_rb = q4_row_bytes(hidden);
    const size_t q4_out_rb = q4_row_bytes(query_count);
    uint8_t *weights[RL_ATTN_PROJ_TENSOR_COUNT] = {0};
    float *cpu_norm = NULL, *gpu_norm = NULL, *cpu_query = NULL, *gpu_query = NULL;
    float *cpu_gate = NULL, *gpu_gate = NULL, *cpu_key = NULL, *gpu_key = NULL;
    float *cpu_value = NULL, *gpu_value = NULL, *cpu_attn = NULL, *gpu_attn = NULL;
    float *cpu_gated = NULL, *gpu_gated = NULL, *cpu_kcache = NULL, *gpu_kcache = NULL;
    float *cpu_vcache = NULL, *gpu_vcache = NULL, *qgate_raw = NULL, *key_raw = NULL;
    int ok = 0;

    for (uint32_t i = 0; i < RL_ATTN_PROJ_TENSOR_COUNT; ++i) {
        weights[i] = (uint8_t *)malloc(weight_bytes[i]);
        if (!weights[i]) { snprintf(error, cap, "attention weight allocation failed"); goto done; }
    }
#define AB_ALLOC(name, count) do { name = (float *)calloc((count), sizeof(float)); if (!(name)) { snprintf(error, cap, "attention scratch allocation failed"); goto done; } } while (0)
    AB_ALLOC(cpu_norm, hidden); AB_ALLOC(gpu_norm, hidden);
    AB_ALLOC(cpu_query, query_count); AB_ALLOC(gpu_query, query_count);
    AB_ALLOC(cpu_gate, query_count); AB_ALLOC(gpu_gate, query_count);
    AB_ALLOC(cpu_key, kv_count); AB_ALLOC(gpu_key, kv_count);
    AB_ALLOC(cpu_value, kv_count); AB_ALLOC(gpu_value, kv_count);
    AB_ALLOC(cpu_attn, query_count); AB_ALLOC(gpu_attn, query_count);
    AB_ALLOC(cpu_gated, query_count); AB_ALLOC(gpu_gated, query_count);
    AB_ALLOC(cpu_kcache, cache_count); AB_ALLOC(gpu_kcache, cache_count);
    AB_ALLOC(cpu_vcache, cache_count); AB_ALLOC(gpu_vcache, cache_count);
    AB_ALLOC(qgate_raw, qgate_count); AB_ALLOC(key_raw, kv_count);
#undef AB_ALLOC

    int fd = open(model, O_RDONLY);
    if (fd < 0) { snprintf(error, cap, "open attention weights failed: %s", strerror(errno)); goto done; }
    uint64_t calls = 0;
    int read_ok = 1;
    for (uint32_t i = 0; i < RL_ATTN_PROJ_TENSOR_COUNT; ++i) {
        if (!pread_full(fd, weights[i], weight_bytes[i], tensors[i].tensor_offset, &calls)) { read_ok = 0; break; }
    }
    close(fd);
    if (!read_ok) { snprintf(error, cap, "attention weight read failed"); goto done; }

    uint8_t grid[RL_IQ2_XXS_GRID_COUNT];
    if (!rl_native_iq2_xxs_build_grid(grid, error, cap)) goto done;
    make_prior_cache(cpu_kcache, cpu_vcache, position, meta.kv_heads, head_dim,
        meta.rope_dims, meta.rope_freq_base);
    memcpy(gpu_kcache, cpu_kcache, cache_count * sizeof(float));
    memcpy(gpu_vcache, cpu_vcache, cache_count * sizeof(float));

    const double start = now_ms();
    rmsnorm_f32(input, (const float *)weights[0], hidden, meta.rms_eps, cpu_norm);
    for (uint32_t row = 0; row < qgate_count; ++row) {
        double dot = 0.0;
        if (!rl_native_shared_quant_row_dot(weights[1] + (size_t)row * iq2_rb, iq2_rb, 16u,
                cpu_norm, hidden, grid, sizeof(grid), &dot, error, cap)) goto done;
        qgate_raw[row] = (float)dot;
    }
    for (uint32_t row = 0; row < kv_count; ++row) {
        double kd = 0.0, vd = 0.0;
        if (!q4_k_row_dot(weights[2] + (size_t)row * q4_in_rb, cpu_norm, hidden, &kd) ||
            !q4_k_row_dot(weights[3] + (size_t)row * q4_in_rb, cpu_norm, hidden, &vd)) {
            snprintf(error, cap, "CPU attention K/V projection failed"); goto done;
        }
        key_raw[row] = (float)kd;
        cpu_value[row] = (float)vd;
    }
    for (uint32_t h = 0; h < meta.query_heads; ++h) {
        const float *src = qgate_raw + (size_t)h * head_dim * 2u;
        rmsnorm_f32(src, (const float *)weights[4], head_dim, meta.rms_eps,
            cpu_query + (size_t)h * head_dim);
        memcpy(cpu_gate + (size_t)h * head_dim, src + head_dim, (size_t)head_dim * sizeof(float));
        rope_neox(cpu_query + (size_t)h * head_dim, head_dim, meta.rope_dims, position, meta.rope_freq_base);
    }
    for (uint32_t h = 0; h < meta.kv_heads; ++h) {
        rmsnorm_f32(key_raw + (size_t)h * head_dim, (const float *)weights[5], head_dim,
            meta.rms_eps, cpu_key + (size_t)h * head_dim);
        rope_neox(cpu_key + (size_t)h * head_dim, head_dim, meta.rope_dims, position, meta.rope_freq_base);
    }
    memcpy(cpu_kcache + (size_t)position * kv_count, cpu_key, (size_t)kv_count * sizeof(float));
    memcpy(cpu_vcache + (size_t)position * kv_count, cpu_value, (size_t)kv_count * sizeof(float));
    cpu_gqa(cpu_query, cpu_kcache, cpu_vcache, cpu_gate, seq_len, meta.query_heads,
        meta.kv_heads, head_dim, cpu_attn, cpu_gated);
    for (uint32_t row = 0; row < hidden; ++row) {
        double dot = 0.0;
        if (!q4_k_row_dot(weights[6] + (size_t)row * q4_out_rb, cpu_gated, query_count, &dot)) {
            snprintf(error, cap, "CPU attention output projection failed"); goto done;
        }
        cpu_output[row] = (float)dot;
    }
    if (cpu_ms) *cpu_ms = now_ms() - start;

#ifdef __APPLE__
    if (!rl_attention_gpu_execute(model, tensors, input, hidden, meta.rms_eps,
            position, meta.rope_dims, meta.rope_freq_base,
            gpu_kcache, gpu_vcache, (uint32_t)cache_count,
            gpu_norm, hidden, gpu_query, query_count, gpu_gate, query_count,
            gpu_key, kv_count, gpu_value, kv_count, gpu_attn, query_count,
            gpu_gated, query_count, gpu_output, hidden, gpu_telemetry, error, cap)) goto done;
#else
    snprintf(error, cap, "full-attention block requires macOS Metal"); goto done;
#endif

    if (cache_error) *cache_error = compare_arrays(gpu_kcache, cpu_kcache, cache_count);
    const int stages_ok =
        within(gpu_norm, cpu_norm, hidden, 3.0e-6f, 3.0e-5f) &&
        within(gpu_query, cpu_query, query_count, 2.0e-4f, 2.0e-4f) &&
        within(gpu_gate, cpu_gate, query_count, 2.0e-4f, 2.0e-4f) &&
        within(gpu_key, cpu_key, kv_count, 2.0e-4f, 2.0e-4f) &&
        within(gpu_value, cpu_value, kv_count, 2.0e-4f, 2.0e-4f) &&
        within(gpu_kcache, cpu_kcache, cache_count, 2.0e-4f, 2.0e-4f) &&
        within(gpu_vcache, cpu_vcache, cache_count, 2.0e-4f, 2.0e-4f) &&
        within(gpu_attn, cpu_attn, query_count, 4.0e-4f, 4.0e-4f) &&
        within(gpu_gated, cpu_gated, query_count, 4.0e-4f, 4.0e-4f) &&
        within(gpu_output, cpu_output, hidden, 2.0e-3f, 4.0e-4f);
    if (!stages_ok) { snprintf(error, cap, "full-attention substage parity failed"); goto done; }
    ok = 1;

done:
    for (uint32_t i = 0; i < RL_ATTN_PROJ_TENSOR_COUNT; ++i) free(weights[i]);
    free(cpu_norm); free(gpu_norm); free(cpu_query); free(gpu_query); free(cpu_gate); free(gpu_gate);
    free(cpu_key); free(gpu_key); free(cpu_value); free(gpu_value); free(cpu_attn); free(gpu_attn);
    free(cpu_gated); free(gpu_gated); free(cpu_kcache); free(gpu_kcache); free(cpu_vcache); free(gpu_vcache);
    free(qgate_raw); free(key_raw);
    return ok;
}

static const rl_router_tensor_info *ab_router(const rl_router_tensor_info *routers,
        uint32_t count, uint32_t layer) {
    for (uint32_t i = 0; i < count; ++i) if (routers[i].layer == layer) return &routers[i];
    return NULL;
}

static int ab_shared(const rl_shared_tensor_info *all, uint32_t count, uint32_t layer,
        rl_shared_tensor_info out[4]) {
    uint8_t seen[4] = {0};
    memset(out, 0, 4u * sizeof(*out));
    for (uint32_t i = 0; i < count; ++i) {
        if (all[i].layer != layer || all[i].kind > RL_SHARED_DOWN) continue;
        const uint32_t kind = (uint32_t)all[i].kind;
        if (seen[kind]) return 0;
        out[kind] = all[i];
        seen[kind] = 1;
    }
    return seen[0] && seen[1] && seen[2] && seen[3];
}

static int ab_ffn(const char *model, uint32_t layer, uint32_t topk, uint64_t cache_mib,
        const float *cpu_input, const float *gpu_input, uint32_t hidden,
        float *cpu_output, float *gpu_output, char *error, size_t cap) {
    rl_router_tensor_info routers[RL_ROUTER_MAX_LAYERS];
    uint32_t router_count = 0;
    if (!rl_native_router_audit(model, routers, RL_ROUTER_MAX_LAYERS, &router_count, error, cap)) return 0;
    const rl_router_tensor_info *router = ab_router(routers, router_count, layer);
    if (!router || router->shape[0] != hidden || router->shape[1] > UINT32_MAX) {
        snprintf(error, cap, "full-attention block router mismatch"); return 0;
    }
    const uint32_t experts = (uint32_t)router->shape[1];
    if (topk > experts) { snprintf(error, cap, "top-k exceeds expert count"); return 0; }
    rl_shared_tensor_info *all = (rl_shared_tensor_info *)calloc(AB_MAX_SHARED, sizeof(*all));
    rl_shared_tensor_info shared[4];
    uint32_t shared_count = 0;
    if (!all || !rl_native_shared_audit(model, all, AB_MAX_SHARED, &shared_count, error, cap) ||
        !ab_shared(all, shared_count, layer, shared)) { free(all); return 0; }
    free(all);

    float *cpu_logits = (float *)malloc((size_t)experts * sizeof(float));
    float *gpu_logits = (float *)malloc((size_t)experts * sizeof(float));
    float *cpu_probs = (float *)malloc((size_t)experts * sizeof(float));
    float *gpu_probs = (float *)malloc((size_t)experts * sizeof(float));
    float *gpu_routed = (float *)calloc(hidden, sizeof(float));
    float *gpu_shared = (float *)calloc(hidden, sizeof(float));
    double *cpu_routed = (double *)calloc(hidden, sizeof(double));
    double *cpu_shared = (double *)calloc(hidden, sizeof(double));
    int ok = 0;
    if (!cpu_logits || !gpu_logits || !cpu_probs || !gpu_probs || !gpu_routed || !gpu_shared ||
        !cpu_routed || !cpu_shared) { snprintf(error, cap, "FFN allocation failed"); goto done; }
    rl_native_router_telemetry cpu_router_tel = {0}, gpu_router_tel = {0};
    if (!rl_native_router_cpu_f32(model, router, cpu_input, hidden, cpu_logits, experts,
            &cpu_router_tel, error, cap) ||
        !rl_native_router_gpu_f32(model, router, gpu_input, hidden, gpu_logits, experts,
            &gpu_router_tel, error, cap)) goto done;
    uint32_t cpu_ids[AB_MAX_TOPK], gpu_ids[AB_MAX_TOPK];
    float cpu_weights[AB_MAX_TOPK], gpu_weights[AB_MAX_TOPK];
    if (!rl_native_router_select_softmax_topk(cpu_logits, experts, topk, cpu_ids, cpu_weights,
            cpu_probs, error, cap) ||
        !rl_native_router_select_softmax_topk(gpu_logits, experts, topk, gpu_ids, gpu_weights,
            gpu_probs, error, cap)) goto done;
    for (uint32_t k = 0; k < topk; ++k) if (cpu_ids[k] != gpu_ids[k]) {
        snprintf(error, cap, "router IDs diverged in full-attention block"); goto done;
    }
    rl_expert_map map;
    if (!rl_native_build_expert_map(model, experts, &map, error, cap)) goto done;
    rl_native_metal_runtime *runtime = rl_native_metal_create(model, &map, cache_mib * AB_MIB,
        64u, error, cap);
    if (!runtime) { rl_native_free_expert_map(&map); goto done; }
    rl_native_metal_telemetry metal_tel = {0};
    double cpu_ms = 0.0;
    const int routed_ok = rl_native_metal_execute_topk(runtime, &map, layer, gpu_ids, gpu_weights,
            topk, 0, hidden, gpu_input, hidden, gpu_routed, hidden, &metal_tel, error, cap) &&
        rl_native_reference_topk(model, &map, layer, cpu_ids, cpu_weights, topk, 0, hidden,
            cpu_input, hidden, cpu_routed, hidden, &cpu_ms, error, cap);
    rl_native_metal_destroy(runtime);
    rl_native_free_expert_map(&map);
    if (!routed_ok) goto done;
    rl_native_shared_telemetry cpu_shared_tel = {0}, gpu_shared_tel = {0};
    if (!rl_native_shared_cpu_execute(model, shared, cpu_input, hidden, 0, hidden,
            cpu_shared, hidden, &cpu_shared_tel, error, cap) ||
        !rl_native_shared_gpu_execute(model, shared, gpu_input, hidden, 0, hidden,
            gpu_shared, hidden, &gpu_shared_tel, error, cap)) goto done;
    for (uint32_t i = 0; i < hidden; ++i) {
        cpu_output[i] = (float)(cpu_routed[i] + cpu_shared[i]);
        gpu_output[i] = gpu_routed[i] + gpu_shared[i];
    }
    ok = 1;

done:
    free(cpu_logits); free(gpu_logits); free(cpu_probs); free(gpu_probs);
    free(gpu_routed); free(gpu_shared); free(cpu_routed); free(cpu_shared);
    return ok;
}

static void ab_cpu_resnorm(const float *residual, const float *branch, const float *weight,
        uint32_t n, float eps, float *sum, float *norm) {
    double ss = 0.0;
    for (uint32_t i = 0; i < n; ++i) { sum[i] = residual[i] + branch[i]; ss += (double)sum[i] * sum[i]; }
    const double inv = 1.0 / sqrt(ss / (double)n + eps);
    for (uint32_t i = 0; i < n; ++i) norm[i] = (float)((double)sum[i] * inv * weight[i]);
}

static void ab_usage(FILE *out) {
    fprintf(out,
        "redlite-attention-block 0.3.0.dev16g\n\n"
        "Usage:\n"
        "  redlite-attention-block parity MODEL --layer N [--position N] [--top-k N] [--cache-mib N]\n");
}

int main(int argc, char **argv) {
    if (argc < 5 || strcmp(argv[1], "parity") != 0) { ab_usage(argc > 1 ? stderr : stdout); return 2; }
    const char *model = argv[2];
    uint32_t layer = UINT32_MAX, position = 7u, topk = 10u;
    uint64_t cache_mib = 256u;
    for (int i = 3; i < argc; ++i) {
        if (i + 1 >= argc) return 2;
        if (strcmp(argv[i], "--layer") == 0) {
            if (!parse_u32(argv[++i], &layer)) return 2;
        } else if (strcmp(argv[i], "--position") == 0) {
            if (!parse_u32(argv[++i], &position)) return 2;
        } else if (strcmp(argv[i], "--top-k") == 0) {
            if (!parse_u32(argv[++i], &topk)) return 2;
        } else if (strcmp(argv[i], "--cache-mib") == 0) {
            char *end = NULL;
            cache_mib = strtoull(argv[++i], &end, 10);
            if (!end || *end) return 2;
        } else return 2;
    }
    if (layer == UINT32_MAX || position >= RL_ATTN_MAX_CONTEXT || !topk || topk > AB_MAX_TOPK || !cache_mib) {
        fprintf(stderr, "full-attention block arguments out of range\n"); return 2;
    }

    char error[512] = {0};
    model_meta meta;
    rl_attn_proj_tensor_info attention_tensors[RL_ATTN_PROJ_TENSOR_COUNT];
    size_t attention_bytes[RL_ATTN_PROJ_TENSOR_COUNT] = {0};
    if (!audit_layer(model, layer, &meta, attention_tensors, error, sizeof(error)) ||
        !validate_layout(&meta, attention_tensors, attention_bytes, error, sizeof(error))) {
        fprintf(stderr, "full-attention block audit failed: %s\n", error); return 1;
    }
    const uint32_t hidden = meta.hidden;
    char post_name[96];
    snprintf(post_name, sizeof(post_name), "blk.%u.post_attention_norm.weight", layer);
    ab_tensor post;
    if (!ab_find_tensor(model, post_name, &post, error, sizeof(error)) || post.type != 0u ||
        post.dims != 1u || post.shape[0] != hidden || post.span < (uint64_t)hidden * sizeof(float)) {
        fprintf(stderr, "post-attention norm audit failed: %s\n", error); return 1;
    }

    float *post_weight = (float *)malloc((size_t)hidden * sizeof(float));
    float *input = (float *)calloc(hidden, sizeof(float));
    float *cpu_attention = (float *)calloc(hidden, sizeof(float));
    float *gpu_attention = (float *)calloc(hidden, sizeof(float));
    float *cpu_residual = (float *)calloc(hidden, sizeof(float));
    float *gpu_residual = (float *)calloc(hidden, sizeof(float));
    float *cpu_norm = (float *)calloc(hidden, sizeof(float));
    float *gpu_norm = (float *)calloc(hidden, sizeof(float));
    float *cpu_ffn = (float *)calloc(hidden, sizeof(float));
    float *gpu_ffn = (float *)calloc(hidden, sizeof(float));
    float *cpu_output = (float *)calloc(hidden, sizeof(float));
    float *gpu_output = (float *)calloc(hidden, sizeof(float));
    if (!post_weight || !input || !cpu_attention || !gpu_attention || !cpu_residual || !gpu_residual ||
        !cpu_norm || !gpu_norm || !cpu_ffn || !gpu_ffn || !cpu_output || !gpu_output) {
        fprintf(stderr, "full-attention block allocation failed\n"); return 1;
    }
    make_input(input, hidden);
    int fd = open(model, O_RDONLY);
    uint64_t calls = 0;
    if (fd < 0 || !pread_full(fd, post_weight, (size_t)hidden * sizeof(float), post.offset, &calls)) {
        if (fd >= 0) close(fd);
        fprintf(stderr, "post-attention norm read failed\n"); return 1;
    }
    close(fd);

    error_stats cache_e = {0};
    double attention_cpu_ms = 0.0;
    rl_attn_proj_telemetry attention_gpu_tel = {0};
    if (!ab_attention(model, layer, position, input, hidden, cpu_attention, gpu_attention,
            &cache_e, &attention_cpu_ms, &attention_gpu_tel, error, sizeof(error))) {
        fprintf(stderr, "composed full attention failed: %s\n", error); return 1;
    }
    ab_cpu_resnorm(input, cpu_attention, post_weight, hidden, meta.rms_eps, cpu_residual, cpu_norm);
    rl_block_ops_telemetry resnorm_tel = {0}, final_tel = {0};
#ifdef __APPLE__
    if (!rl_block_residual_rmsnorm_gpu(input, gpu_attention, post_weight, hidden, meta.rms_eps,
            gpu_residual, gpu_norm, &resnorm_tel, error, sizeof(error))) {
        fprintf(stderr, "Metal post-attention residual/norm failed: %s\n", error); return 1;
    }
#else
    fprintf(stderr, "full-attention block requires macOS Metal\n"); return 1;
#endif
    if (!ab_ffn(model, layer, topk, cache_mib, cpu_norm, gpu_norm, hidden,
            cpu_ffn, gpu_ffn, error, sizeof(error))) {
        fprintf(stderr, "composed FFN failed: %s\n", error); return 1;
    }
    for (uint32_t i = 0; i < hidden; ++i) cpu_output[i] = cpu_residual[i] + cpu_ffn[i];
#ifdef __APPLE__
    if (!rl_block_residual_gpu(gpu_residual, gpu_ffn, hidden, gpu_output,
            &final_tel, error, sizeof(error))) {
        fprintf(stderr, "Metal final residual failed: %s\n", error); return 1;
    }
#endif

    const error_stats attention_e = compare_arrays(gpu_attention, cpu_attention, hidden);
    const error_stats residual_e = compare_arrays(gpu_residual, cpu_residual, hidden);
    const error_stats norm_e = compare_arrays(gpu_norm, cpu_norm, hidden);
    const error_stats ffn_e = compare_arrays(gpu_ffn, cpu_ffn, hidden);
    const error_stats output_e = compare_arrays(gpu_output, cpu_output, hidden);
    const int attention_ok = within(gpu_attention, cpu_attention, hidden, 2.0e-3f, 4.0e-4f);
    const int residual_ok = within(gpu_residual, cpu_residual, hidden, 2.1e-3f, 5.0e-4f);
    const int norm_ok = within(gpu_norm, cpu_norm, hidden, 2.1e-3f, 1.5e-3f);
    const int ffn_ok = within(gpu_ffn, cpu_ffn, hidden, 3.0e-3f, 2.0e-3f);
    const int output_ok = within(gpu_output, cpu_output, hidden, 4.0e-3f, 2.0e-3f);
    const int all_ok = attention_ok && residual_ok && norm_ok && ffn_ok && output_ok;

    printf("runtime              : complete native Qwen3-Next full-attention transformer block parity\n");
    printf("layer / position     : %u / %u (context=%u)\n", layer, position, position + 1u);
    printf("hidden / top-k       : %u / %u\n", hidden, topk);
    printf("attention abs/rel    : %.6g / %.6g parity=%s\n", attention_e.max_abs, attention_e.max_rel, attention_ok ? "YES" : "NO");
    printf("KV cache abs/rel     : %.6g / %.6g parity=YES\n", cache_e.max_abs, cache_e.max_rel);
    printf("attention residual   : %.6g / %.6g parity=%s\n", residual_e.max_abs, residual_e.max_rel, residual_ok ? "YES" : "NO");
    printf("post-attn RMSNorm    : %.6g / %.6g parity=%s\n", norm_e.max_abs, norm_e.max_rel, norm_ok ? "YES" : "NO");
    printf("full FFN abs/rel     : %.6g / %.6g parity=%s\n", ffn_e.max_abs, ffn_e.max_rel, ffn_ok ? "YES" : "NO");
    printf("FINAL block abs/rel  : %.6g / %.6g parity=%s\n", output_e.max_abs, output_e.max_rel, output_ok ? "YES" : "NO");
    printf("attention CPU/Metal  : %.3f / %.3f ms\n", attention_cpu_ms, attention_gpu_tel.compute_ms);
    printf("resnorm/final Metal  : %.3f / %.3f ms\n", resnorm_tel.compute_ms, final_tel.compute_ms);
    printf("COMPLETE FULL ATTENTION BLOCK: %s\n", all_ok ? "YES" : "NO");
    for (uint32_t i = 0; i < 4u; ++i)
        printf("row %-3u             : gpu=%+.7f cpu=%+.7f delta=%+.3e\n", i, gpu_output[i], cpu_output[i],
            (double)gpu_output[i] - cpu_output[i]);

    free(post_weight); free(input); free(cpu_attention); free(gpu_attention);
    free(cpu_residual); free(gpu_residual); free(cpu_norm); free(gpu_norm);
    free(cpu_ffn); free(gpu_ffn); free(cpu_output); free(gpu_output);
    return all_ok ? 0 : 3;
}
