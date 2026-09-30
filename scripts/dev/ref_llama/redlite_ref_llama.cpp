// Development-only reference oracle built against the pinned llama.cpp.
//
// It never takes part in Red Lite inference: it exists to dump what the pinned
// upstream implementation computes for the same token sequence so the native
// engine can be compared layer by layer (embedding, l_out-N, result_norm,
// logits) and so the native tokenizer can be checked against llama_tokenize.
//
//   redlite-ref-llama MODEL logits --tokens a,b,c --out FILE [--ctx N] [--dump-last | --dump-from N]
//   redlite-ref-llama MODEL tokenize --text "..." [--no-special]
//   redlite-ref-llama MODEL tokenize --file CORPUS [--no-special]   (one input per line, \n \t \\ escapes)
//   redlite-ref-llama MODEL greedy --tokens a,b,c --max N [--ctx N]
//   redlite-ref-llama MODEL dequant --tensor NAME [--row-first N] [--rows N] --out FILE
//       rows of a tensor dequantized by ggml's own to_float (dev31: bitwise check of the native CPU reference)
//   redlite-ref-llama MODEL bench --tokens a,b,c [--max N] [--ctx N]
//       throughput of the pinned llama.cpp on the same ids, with its default context parameters
//       (n_batch 2048, n_ubatch 512, flash attention auto, f16 KV): one prefill of the ids, then N
//       greedy tokens one at a time. Development measurement only.
//
// Dump layout per token (little-endian f32):
//   [hidden] embedding, [n_layer][hidden] layer outputs, [hidden] result_norm, [vocab] logits

#include "ggml-backend.h"
#include "gguf.h"
#include "ggml.h"
#include "llama.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <string>
#include <vector>

struct capture {
    int n_layer = 0;
    int hidden = 0;
    std::vector<float> embed;
    std::vector<std::vector<float>> layers;
    std::vector<float> result_norm;
    int router_layer = -1;              /* --router-layer L: capture the MoE routing of layer L */
    std::vector<float> router_probs;    /* ffn_moe_probs-L, [n_expert] */
    std::vector<int32_t> router_topk;   /* ffn_moe_topk-L, [n_expert_used] */
    bool want(const char *name) const {
        if (router_layer >= 0) {
            char pn[64], tn[64];
            snprintf(pn, sizeof(pn), "ffn_moe_probs-%d", router_layer);
            snprintf(tn, sizeof(tn), "ffn_moe_topk-%d", router_layer);
            if (strcmp(name, pn) == 0 || strcmp(name, tn) == 0) return true;
        }
        return strncmp(name, "l_out-", 6) == 0 || strcmp(name, "result_norm") == 0 || strcmp(name, "model.embed_tokens") == 0;
    }
};

static bool eval_cb(struct ggml_tensor *t, bool ask, void *ud) {
    capture *c = (capture *)ud;
    const char *name = ggml_get_name(t);
    if (ask) return c->want(name);
    if (!c->want(name)) return true;
    const int64_t n = ggml_nelements(t);
    if (t->type == GGML_TYPE_I32 && strncmp(name, "ffn_moe_topk-", 13) == 0) {
        c->router_topk.resize((size_t)n);
        ggml_backend_tensor_get(t, c->router_topk.data(), 0, ggml_nbytes(t));
        return true;
    }
    std::vector<float> buf((size_t)n);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, buf.data(), 0, ggml_nbytes(t));
    } else {
        fprintf(stderr, "unexpected tensor type for %s\n", name);
        return true;
    }
    if (strncmp(name, "ffn_moe_probs-", 14) == 0) {
        c->router_probs = buf;
    } else if (strncmp(name, "l_out-", 6) == 0) {
        const int il = atoi(name + 6);
        if (il >= 0 && il < c->n_layer) c->layers[(size_t)il].assign(buf.begin(), buf.begin() + c->hidden);
    } else if (strcmp(name, "result_norm") == 0) {
        c->result_norm.assign(buf.begin(), buf.begin() + c->hidden);
    } else if (strcmp(name, "model.embed_tokens") == 0) {
        c->embed.assign(buf.begin(), buf.begin() + c->hidden);
    }
    return true;
}

static std::string unescape_line(const std::string &in) {
    std::string out;
    for (size_t i = 0; i < in.size(); ++i) {
        if (in[i] == '\\' && i + 1 < in.size()) {
            const char c = in[i + 1];
            if (c == 'n') { out.push_back('\n'); ++i; continue; }
            if (c == 't') { out.push_back('\t'); ++i; continue; }
            if (c == 'r') { out.push_back('\r'); ++i; continue; }
            if (c == '\\') { out.push_back('\\'); ++i; continue; }
        }
        out.push_back(in[i]);
    }
    return out;
}

static std::vector<llama_token> tokenize_text(const llama_vocab *vocab, const std::string &text, bool special) {
    std::vector<llama_token> ids(text.size() + 16);
    int n = llama_tokenize(vocab, text.c_str(), (int32_t)text.size(), ids.data(), (int32_t)ids.size(), false, special);
    if (n < 0) { ids.resize((size_t)-n); n = llama_tokenize(vocab, text.c_str(), (int32_t)text.size(), ids.data(), (int32_t)ids.size(), false, special); }
    ids.resize(n > 0 ? (size_t)n : 0u);
    return ids;
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
    const char *tokens_arg = nullptr, *out_path = nullptr, *text = nullptr, *corpus = nullptr, *tensor_name = nullptr;
    long row_first = 0, row_count = 16;
    int n_ctx = 64, max_new = 16;
    size_t dump_from = 0;
    int router_layer = -1;
    bool special = true, dump_last = false;
    for (int i = 3; i < argc; ++i) {
        if (strcmp(argv[i], "--no-special") == 0) { special = false; continue; }
        if (strcmp(argv[i], "--dump-last") == 0) { dump_last = true; continue; }
        if (i + 1 >= argc) return 2;
        if (strcmp(argv[i], "--tokens") == 0) tokens_arg = argv[++i];
        else if (strcmp(argv[i], "--out") == 0) out_path = argv[++i];
        else if (strcmp(argv[i], "--text") == 0) text = argv[++i];
        else if (strcmp(argv[i], "--file") == 0) corpus = argv[++i];
        else if (strcmp(argv[i], "--dump-from") == 0) dump_from = (size_t)atol(argv[++i]);
        else if (strcmp(argv[i], "--router-layer") == 0) router_layer = atoi(argv[++i]);
        else if (strcmp(argv[i], "--ctx") == 0) n_ctx = atoi(argv[++i]);
        else if (strcmp(argv[i], "--max") == 0) max_new = atoi(argv[++i]);
        else if (strcmp(argv[i], "--tensor") == 0) tensor_name = argv[++i];
        else if (strcmp(argv[i], "--row-first") == 0) row_first = atol(argv[++i]);
        else if (strcmp(argv[i], "--rows") == 0) row_count = atol(argv[++i]);
        else { fprintf(stderr, "unknown option %s\n", argv[i]); return 2; }
    }

    if (cmd == "dequant") {
        if (!tensor_name || !out_path) { fprintf(stderr, "--tensor and --out required\n"); return 2; }
        gguf_init_params gp = { /*no_alloc*/ true, /*ctx*/ nullptr };
        gguf_context *gc = gguf_init_from_file(model_path, gp);
        if (!gc) { fprintf(stderr, "gguf open failed\n"); return 1; }
        const int64_t ti = gguf_find_tensor(gc, tensor_name);
        if (ti < 0) { fprintf(stderr, "tensor not found\n"); return 1; }
        const ggml_type type = gguf_get_tensor_type(gc, ti);
        const size_t off = gguf_get_data_offset(gc) + gguf_get_tensor_offset(gc, ti);
        // ne0 from the tensor size: the rows are ne0 values; the caller passes the row length through the dump size
        const size_t tbytes = gguf_get_tensor_size(gc, ti);
        const ggml_type_traits *tr = ggml_get_type_traits(type);
        if (!tr->to_float) { fprintf(stderr, "no to_float for type %d\n", (int)type); return 1; }
        // row length: read the shape through a no_alloc ggml context
        ggml_context *meta = nullptr;
        gguf_init_params gp2 = { true, &meta };
        gguf_context *gc2 = gguf_init_from_file(model_path, gp2);
        ggml_tensor *t = ggml_get_tensor(meta, tensor_name);
        const int64_t ne0 = t->ne[0];
        const size_t rb = ggml_row_size(type, ne0);
        FILE *mf = fopen(model_path, "rb");
        std::vector<uint8_t> raw(rb);
        std::vector<float> row((size_t)ne0);
        FILE *of = fopen(out_path, "wb");
        for (long r = row_first; r < row_first + row_count; ++r) {
            if ((size_t)(r + 1) * rb > tbytes) { fprintf(stderr, "row out of range\n"); return 1; }
            fseeko(mf, (off_t)(off + (size_t)r * rb), SEEK_SET);
            if (fread(raw.data(), 1, rb, mf) != rb) { fprintf(stderr, "read failed\n"); return 1; }
            tr->to_float(raw.data(), row.data(), ne0);
            fwrite(row.data(), sizeof(float), (size_t)ne0, of);
        }
        fclose(of); fclose(mf);
        printf("dequantized %s (%s) rows %ld..%ld, %lld columns\n", tensor_name, ggml_type_name(type), row_first, row_first + row_count - 1, (long long)ne0);
        gguf_free(gc2); ggml_free(meta); gguf_free(gc);
        return 0;
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
        if (!text && !corpus) { fprintf(stderr, "--text or --file required\n"); return 2; }
        if (corpus) {
            FILE *cf = fopen(corpus, "rb");
            if (!cf) { fprintf(stderr, "cannot open %s\n", corpus); return 1; }
            std::string line;
            int ch;
            bool pending = false;
            while ((ch = fgetc(cf)) != EOF) {
                if (ch == '\n') {
                    if (!line.empty() && line.back() == '\r') line.pop_back();
                    const std::vector<llama_token> ids = tokenize_text(vocab, unescape_line(line), special);
                    for (size_t i = 0; i < ids.size(); ++i) printf("%s%d", i ? "," : "", ids[i]);
                    printf("\n");
                    line.clear(); pending = false;
                } else { line.push_back((char)ch); pending = true; }
            }
            if (pending) {
                const std::vector<llama_token> ids = tokenize_text(vocab, unescape_line(line), special);
                for (size_t i = 0; i < ids.size(); ++i) printf("%s%d", i ? "," : "", ids[i]);
                printf("\n");
            }
            fclose(cf);
            llama_model_free(model);
            return 0;
        }
        const std::vector<llama_token> ids = tokenize_text(vocab, text, special);
        const int n = (int)ids.size();
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
    cap.router_layer = router_layer;

    if (cmd == "bench") {
        llama_context_params bp = llama_context_default_params();
        bp.n_ctx = (uint32_t)std::max<size_t>((size_t)n_ctx, tokens.size() + (size_t)max_new + 16);
        bp.n_seq_max = 1;
        llama_context *bctx = llama_init_from_model(model, bp);
        if (!bctx) { fprintf(stderr, "context init failed\n"); return 1; }
        const size_t nb = (size_t)llama_n_batch(bctx);
        const auto t0 = std::chrono::steady_clock::now();
        for (size_t off = 0; off < tokens.size(); off += nb) {
            const size_t n = std::min(nb, tokens.size() - off);
            if (llama_decode(bctx, llama_batch_get_one(tokens.data() + off, (int32_t)n)) != 0) { fprintf(stderr, "prefill failed\n"); return 1; }
        }
        llama_synchronize(bctx);
        const auto t1 = std::chrono::steady_clock::now();
        const double pms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        printf("llama prefill %zu tokens: %.1f ms (%.2f tok/s) n_batch=%u n_ubatch=%u\n", tokens.size(), pms,
               1000.0 * (double)tokens.size() / pms, llama_n_batch(bctx), llama_n_ubatch(bctx));
        llama_token tok = 0;
        const auto t2 = std::chrono::steady_clock::now();
        for (int i = 0; i < max_new; ++i) {
            const float *lg = llama_get_logits_ith(bctx, -1);
            int best = 0;
            for (int v = 1; v < n_vocab; ++v) if (lg[v] > lg[best]) best = v;
            tok = best;
            if (llama_decode(bctx, llama_batch_get_one(&tok, 1)) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
        }
        llama_synchronize(bctx);
        const double dms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t2).count();
        if (max_new > 0) printf("llama decode %d tokens: %.1f ms (%.2f tok/s)\n", max_new, dms, 1000.0 * max_new / dms);
        llama_free(bctx);
        llama_model_free(model);
        return 0;
    }

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
        if (router_layer >= 0 && i >= dump_from && !cap.router_probs.empty()) {
            printf("  router[%d] topk:", router_layer);
            for (size_t k = 0; k < cap.router_topk.size(); ++k) printf(" %d", cap.router_topk[k]);
            std::vector<int> order((size_t)cap.router_probs.size());
            for (size_t k = 0; k < order.size(); ++k) order[k] = (int)k;
            std::partial_sort(order.begin(), order.begin() + 12, order.end(), [&](int a, int b) { return cap.router_probs[(size_t)a] > cap.router_probs[(size_t)b]; });
            printf(" | probs:");
            for (int k = 0; k < 12; ++k) printf(" %d:%.7f", order[(size_t)k], cap.router_probs[(size_t)order[(size_t)k]]);
            printf("\n");
        }
        if (out && cmd == "logits" && (!dump_last || i + 1 == total) && i >= dump_from) {
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
