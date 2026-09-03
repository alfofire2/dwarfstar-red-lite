// Development-only reference oracle built against the pinned llama.cpp.
//
// It never takes part in Red Lite inference: it exists to dump what the pinned
// upstream implementation computes for the same token sequence so the native
// engine can be compared layer by layer (embedding, l_out-N, result_norm,
// logits) and so the native tokenizer can be checked against llama_tokenize.
//
//   redlite-ref-llama MODEL logits --tokens a,b,c --out FILE [--ctx N]
//   redlite-ref-llama MODEL tokenize --text "..." [--no-special]
//   redlite-ref-llama MODEL greedy --tokens a,b,c --max N [--ctx N]
//
// Dump layout per token (little-endian f32):
//   [hidden] embedding, [n_layer][hidden] layer outputs, [hidden] result_norm, [vocab] logits

#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct capture {
    int n_layer = 0;
    int hidden = 0;
    std::vector<float> embed;
    std::vector<std::vector<float>> layers;
    std::vector<float> result_norm;
    bool want(const char *name) const {
        return strncmp(name, "l_out-", 6) == 0 || strcmp(name, "result_norm") == 0 || strcmp(name, "model.embed_tokens") == 0;
    }
};

static bool eval_cb(struct ggml_tensor *t, bool ask, void *ud) {
    capture *c = (capture *)ud;
    const char *name = ggml_get_name(t);
    if (ask) return c->want(name);
    if (!c->want(name)) return true;
    const int64_t n = ggml_nelements(t);
    std::vector<float> buf((size_t)n);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, buf.data(), 0, ggml_nbytes(t));
    } else {
        fprintf(stderr, "unexpected tensor type for %s\n", name);
        return true;
    }
    if (strncmp(name, "l_out-", 6) == 0) {
        const int il = atoi(name + 6);
        if (il >= 0 && il < c->n_layer) c->layers[(size_t)il].assign(buf.begin(), buf.begin() + c->hidden);
    } else if (strcmp(name, "result_norm") == 0) {
        c->result_norm.assign(buf.begin(), buf.begin() + c->hidden);
    } else if (strcmp(name, "model.embed_tokens") == 0) {
        c->embed.assign(buf.begin(), buf.begin() + c->hidden);
    }
    return true;
}

static std::vector<llama_token> parse_tokens(const char *list) {
    std::vector<llama_token> out;
    const char *p = list;
    while (p && *p) {
        char *end = nullptr;
        const long v = strtol(p, &end, 10);
        if (end == p) break;
        out.push_back((llama_token)v);
        p = end;
        if (*p == ',') ++p;
    }
    return out;
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: redlite-ref-llama MODEL logits|tokenize|greedy ...\n"); return 2; }
    const char *model_path = argv[1];
    const std::string cmd = argv[2];
    const char *tokens_arg = nullptr, *out_path = nullptr, *text = nullptr;
    int n_ctx = 64, max_new = 16;
    bool special = true;
    for (int i = 3; i < argc; ++i) {
        if (strcmp(argv[i], "--no-special") == 0) { special = false; continue; }
        if (i + 1 >= argc) return 2;
        if (strcmp(argv[i], "--tokens") == 0) tokens_arg = argv[++i];
        else if (strcmp(argv[i], "--out") == 0) out_path = argv[++i];
        else if (strcmp(argv[i], "--text") == 0) text = argv[++i];
        else if (strcmp(argv[i], "--ctx") == 0) n_ctx = atoi(argv[++i]);
        else if (strcmp(argv[i], "--max") == 0) max_new = atoi(argv[++i]);
        else { fprintf(stderr, "unknown option %s\n", argv[i]); return 2; }
    }

    llama_log_set([](ggml_log_level level, const char *msg, void *) { if (level >= GGML_LOG_LEVEL_ERROR) fputs(msg, stderr); }, nullptr);
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    llama_model *model = llama_model_load_from_file(model_path, mp);
    if (!model) { fprintf(stderr, "model load failed\n"); return 1; }
    const llama_vocab *vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    if (cmd == "tokenize") {
        if (!text) { fprintf(stderr, "--text required\n"); return 2; }
        std::vector<llama_token> ids((size_t)strlen(text) + 16);
        int n = llama_tokenize(vocab, text, (int32_t)strlen(text), ids.data(), (int32_t)ids.size(), false, special);
        if (n < 0) { ids.resize((size_t)-n); n = llama_tokenize(vocab, text, (int32_t)strlen(text), ids.data(), (int32_t)ids.size(), false, special); }
        for (int i = 0; i < n; ++i) printf("%s%d", i ? "," : "", ids[(size_t)i]);
        printf("\n");
        for (int i = 0; i < n; ++i) {
            char piece[256];
            const int len = llama_token_to_piece(vocab, ids[(size_t)i], piece, sizeof(piece), 0, true);
            printf("%d\t%.*s\n", ids[(size_t)i], len > 0 ? len : 0, piece);
        }
        llama_model_free(model);
        return 0;
    }

    if (!tokens_arg) { fprintf(stderr, "--tokens required\n"); return 2; }
    std::vector<llama_token> tokens = parse_tokens(tokens_arg);
    capture cap;
    cap.n_layer = llama_model_n_layer(model);
    cap.hidden = llama_model_n_embd(model);
    cap.layers.resize((size_t)cap.n_layer);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t)n_ctx;
    cp.n_batch = 1;
    cp.n_ubatch = 1;
    cp.n_seq_max = 1;
    cp.type_k = GGML_TYPE_F32;
    cp.type_v = GGML_TYPE_F32;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    if (cmd == "logits") { cp.cb_eval = eval_cb; cp.cb_eval_user_data = &cap; }
    llama_context *ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "context init failed\n"); return 1; }

    FILE *out = out_path ? fopen(out_path, "wb") : nullptr;
    std::vector<llama_token> seq = tokens;
    const size_t prompt_len = tokens.size();
    const size_t total = cmd == "greedy" ? prompt_len + (size_t)max_new : prompt_len;
    for (size_t i = 0; i < total; ++i) {
        llama_token tok = seq[i];
        llama_batch batch = llama_batch_get_one(&tok, 1);
        if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed at %zu\n", i); return 1; }
        const float *logits = llama_get_logits_ith(ctx, -1);
        int best = 0;
        for (int v = 1; v < n_vocab; ++v) if (logits[v] > logits[best]) best = v;
        printf("token[%zu]=%d -> argmax=%d logit=%.6f\n", i, tok, best, logits[best]);
        if (out && cmd == "logits") {
            std::vector<float> zeros((size_t)cap.hidden, 0.0f);
            fwrite(cap.embed.empty() ? zeros.data() : cap.embed.data(), sizeof(float), (size_t)cap.hidden, out);
            for (int l = 0; l < cap.n_layer; ++l)
                fwrite(cap.layers[(size_t)l].empty() ? zeros.data() : cap.layers[(size_t)l].data(), sizeof(float), (size_t)cap.hidden, out);
            fwrite(cap.result_norm.empty() ? zeros.data() : cap.result_norm.data(), sizeof(float), (size_t)cap.hidden, out);
            fwrite(logits, sizeof(float), (size_t)n_vocab, out);
        }
        if (cmd == "greedy" && i + 1 >= prompt_len && i + 1 < total) {
            if (i + 1 >= seq.size()) seq.push_back(best);
        }
    }
    if (cmd == "greedy") {
        printf("generated:");
        for (size_t i = prompt_len; i < seq.size(); ++i) printf(" %d", seq[i]);
        printf("\n");
        std::string text_out;
        for (size_t i = prompt_len; i < seq.size(); ++i) {
            char piece[256];
            const int len = llama_token_to_piece(vocab, seq[i], piece, sizeof(piece), 0, true);
            if (len > 0) text_out.append(piece, (size_t)len);
        }
        printf("text: %s\n", text_out.c_str());
    }
    if (out) fclose(out);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
