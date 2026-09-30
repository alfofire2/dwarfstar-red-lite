// Development-only sampler oracle built against the pinned llama.cpp (never used at runtime).
//
// Applies the pinned llama.cpp sampler chain in the order common/sampling.cpp builds it for
// these parameters (top_k -> top_p -> min_p -> temp -> dist) to a logits vector and prints
// either the final candidate distribution ("id probability", as dist normalizes it) or the
// ids drawn N times with the given seed ("id count"). Same CLI as redlite-sampler-dist.
//
//   redlite-ref-sampler LOGITS.f32 [--temperature T] [--top-k K] [--top-p P] [--min-p M] [--draws N --seed S]

#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: redlite-ref-sampler LOGITS.f32 [--temperature T] [--top-k K] [--top-p P] [--min-p M] [--draws N --seed S]\n");
        return 2;
    }
    float temp = 1.0f, top_p = 0.95f, min_p = 0.0f;
    int top_k = 40;
    unsigned long draws = 0;
    uint32_t seed = 0;
    for (int i = 2; i < argc; ++i) {
        if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", argv[i]); return 2; }
        const char *v = argv[++i];
        if (strcmp(argv[i - 1], "--temperature") == 0) temp = strtof(v, nullptr);
        else if (strcmp(argv[i - 1], "--top-k") == 0) top_k = atoi(v);
        else if (strcmp(argv[i - 1], "--top-p") == 0) top_p = strtof(v, nullptr);
        else if (strcmp(argv[i - 1], "--min-p") == 0) min_p = strtof(v, nullptr);
        else if (strcmp(argv[i - 1], "--draws") == 0) draws = strtoul(v, nullptr, 10);
        else if (strcmp(argv[i - 1], "--seed") == 0) seed = (uint32_t)strtoul(v, nullptr, 10);
        else { fprintf(stderr, "unknown option %s\n", argv[i - 1]); return 2; }
    }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 1; }
    fseek(f, 0, SEEK_END);
    const long bytes = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (bytes <= 0 || bytes % 4) { fprintf(stderr, "bad logits file\n"); fclose(f); return 1; }
    std::vector<float> logits((size_t)bytes / 4);
    if (fread(logits.data(), 4, logits.size(), f) != logits.size()) { fprintf(stderr, "read failed\n"); fclose(f); return 1; }
    fclose(f);

    llama_sampler *chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(chain, llama_sampler_init_top_k(top_k));
    llama_sampler_chain_add(chain, llama_sampler_init_top_p(top_p, 0));
    llama_sampler_chain_add(chain, llama_sampler_init_min_p(min_p, 0));
    llama_sampler_chain_add(chain, llama_sampler_init_temp_ext(temp, 0.0f, 1.0f));
    llama_sampler_chain_add(chain, llama_sampler_init_dist(seed));

    std::vector<llama_token_data> data(logits.size());
    auto fill = [&]() {
        for (size_t i = 0; i < logits.size(); ++i) data[i] = llama_token_data{(llama_token)i, logits[i], 0.0f};
        return llama_token_data_array{data.data(), data.size(), -1, false};
    };
    if (draws) {
        std::map<llama_token, unsigned long> counts;
        for (unsigned long d = 0; d < draws; ++d) {
            llama_token_data_array cur = fill();
            llama_sampler_apply(chain, &cur);
            counts[cur.data[cur.selected].id]++;
        }
        for (const auto &kv : counts) printf("%d %lu\n", kv.first, kv.second);
    } else {
        llama_token_data_array cur = fill();
        llama_sampler_apply(chain, &cur);
        for (size_t i = 0; i < cur.size; ++i) printf("%d %.9g\n", cur.data[i].id, (double)cur.data[i].p);
    }
    llama_sampler_free(chain);
    return 0;
}
