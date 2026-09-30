#define _POSIX_C_SOURCE 200809L

/*
 * redlite-sampler-dist: the native sampler's exact candidate distribution (or N draws) for a
 * logits vector. Model-free; compared against the pinned llama.cpp chain by
 * scripts/dev/compare_sampler.py (oracle: scripts/dev/ref_llama/redlite_ref_sampler.cpp).
 *
 *   redlite-sampler-dist LOGITS.f32 [--temperature T] [--top-k K] [--top-p P] [--min-p M]
 *                        [--draws N --seed S]
 *
 * LOGITS.f32: little-endian float32 vector (the vocabulary size is the file size / 4).
 * Output: "id probability" per candidate (descending logit), or "id count" per drawn id.
 */

#include "redlite_native_sampler.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc < 2 || strcmp(argv[1], "--help") == 0) {
        printf("Usage: redlite-sampler-dist LOGITS.f32 [--temperature T] [--top-k K] [--top-p P] [--min-p M] [--draws N --seed S]\n");
        return argc < 2 ? 2 : 0;
    }
    rl_sampler_params p;
    rl_sampler_params_default(&p);
    p.temperature = 1.0f;
    unsigned long draws = 0;
    for (int i = 2; i < argc; ++i) {
        if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", argv[i]); return 2; }
        const char *v = argv[++i];
        if (strcmp(argv[i - 1], "--temperature") == 0) p.temperature = strtof(v, NULL);
        else if (strcmp(argv[i - 1], "--top-k") == 0) p.top_k = (uint32_t)strtoul(v, NULL, 10);
        else if (strcmp(argv[i - 1], "--top-p") == 0) p.top_p = strtof(v, NULL);
        else if (strcmp(argv[i - 1], "--min-p") == 0) p.min_p = strtof(v, NULL);
        else if (strcmp(argv[i - 1], "--draws") == 0) draws = strtoul(v, NULL, 10);
        else if (strcmp(argv[i - 1], "--seed") == 0) p.seed = strtoull(v, NULL, 10);
        else { fprintf(stderr, "unknown option %s\n", argv[i - 1]); return 2; }
    }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 1; }
    fseek(f, 0, SEEK_END);
    const long bytes = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (bytes <= 0 || bytes % 4) { fprintf(stderr, "logits file size must be a positive multiple of 4\n"); fclose(f); return 1; }
    const uint32_t vocab = (uint32_t)(bytes / 4);
    float *logits = (float *)malloc((size_t)bytes);
    uint32_t *counts = draws ? (uint32_t *)calloc(vocab, sizeof(uint32_t)) : NULL;
    if (!logits || (draws && !counts) || fread(logits, 4, vocab, f) != vocab) { fprintf(stderr, "read failed\n"); fclose(f); return 1; }
    fclose(f);

    rl_sampler s;
    if (!rl_sampler_init(&s, &p, vocab)) { fprintf(stderr, "sampler init failed\n"); return 1; }
    if (draws) {
        for (unsigned long d = 0; d < draws; ++d) counts[rl_sampler_sample(&s, logits)]++;
        for (uint32_t i = 0; i < vocab; ++i) if (counts[i]) printf("%u %u\n", i, counts[i]);
    } else {
        const uint32_t *ids = NULL;
        const float *probs = NULL;
        const uint32_t n = rl_sampler_distribution(&s, logits, &ids, &probs);
        for (uint32_t i = 0; i < n; ++i) printf("%u %.9g\n", ids[i], (double)probs[i]);
    }
    rl_sampler_free(&s);
    free(counts);
    free(logits);
    return 0;
}
