#define _POSIX_C_SOURCE 200809L
#define _DARWIN_C_SOURCE 1

/*
 * redlite-generate: native end-to-end Qwen3-Next text generation.
 *
 *   redlite-generate MODEL --prompt "..." [--system "..."] [--raw] [--max-tokens N]
 *                    [--temperature T] [--top-k K] [--top-p P] [--seed S]
 *                    [--context N] [--cache-mib N] [--no-stream] [--stats] [--tokens-out FILE]
 *   redlite-generate MODEL --interactive [--system "..."] [generation options]
 *
 * Ctrl-C stops the current answer (the engine state stays consistent: the answer is
 * closed as if it had reached --max-tokens); at the interactive prompt it quits.
 * --json writes one machine-readable statistics object per answer to stderr.
 *
 * prompt -> chat template -> native BPE -> prefill (per token) -> logits ->
 * sampler -> next token -> decode -> repeat until EOS / max-tokens.
 */

#include "redlite_native_engine.h"
#include "redlite_native_gguf_dir.h"
#include "redlite_native_sampler.h"
#include "redlite_native_tokenizer.h"

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>
#ifdef __APPLE__
#include <libproc.h>
#endif

static volatile sig_atomic_t g_interrupt = 0;
static void on_sigint(int sig) { (void)sig; g_interrupt = 1; }

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

static uint64_t peak_rss_bytes(void) {
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) != 0) return 0;
#ifdef __APPLE__
    return (uint64_t)ru.ru_maxrss;          /* bytes on macOS */
#else
    return (uint64_t)ru.ru_maxrss * 1024u;  /* KiB elsewhere */
#endif
}

static uint64_t phys_footprint_bytes(void) {
#ifdef __APPLE__
    struct rusage_info_v4 ri;
    if (proc_pid_rusage(getpid(), RUSAGE_INFO_V4, (rusage_info_t *)&ri) == 0) return (uint64_t)ri.ri_phys_footprint;
#endif
    return 0;
}

static void usage(FILE *out) {
    fprintf(out,
        "redlite-generate 0.4.0 - native Qwen3-Next generation (no llama.cpp, no Python)\n\n"
        "Usage:\n"
        "  redlite-generate MODEL --prompt \"...\" [options]\n"
        "  redlite-generate MODEL --interactive [--prompt \"first message\"] [options]\n\n"
        "Options:\n"
        "  -i, --interactive   keep the model loaded and open a terminal chat\n"
        "  --system TEXT       optional system prompt (chat template)\n"
        "  --raw               tokenize the prompt verbatim (no chat template)\n"
        "  --max-tokens N      maximum generated tokens (default 256)\n"
        "  --temperature T     0 = greedy (default 0)\n"
        "  --top-k K           top-k candidates when sampling (default 40, 0 = off)\n"
        "  --top-p P           nucleus probability when sampling (default 0.95)\n"
        "  --min-p M           drop candidates below M x the top probability (default 0 = off; llama.cpp uses 0.05)\n"
        "  --seed S            PRNG seed for sampling (default 0 -> fixed constant)\n"
        "  --context N         KV cache positions (default 4096)\n"
        "  --cache-mib N|full  routed-expert cache budget in MiB (default 4096); full = every expert of the file,\n"
        "                      computed from its expert payload (redlite-engine info prints it)\n"
        "  --batch N           prompt tokens per batched Metal prefill chunk (default 2048 when every expert is\n"
        "                      preloaded, else 512; 1 = token by token)\n"
        "  --no-stream         print the completion only when finished\n"
        "  --stats             print timing, memory and cache statistics\n"
        "  --json              write the statistics as one JSON object per answer on stderr\n"
        "  --tokens-out FILE   write prompt+generated token ids (one per line)\n");
}

static int parse_u32(const char *s, uint32_t *out) {
    if (!s || !*s) return 0;
    errno = 0;
    char *end = NULL;
    const unsigned long v = strtoul(s, &end, 10);
    if (errno || !end || *end || v > UINT32_MAX) return 0;
    *out = (uint32_t)v;
    return 1;
}

static int parse_f32(const char *s, float *out) {
    if (!s || !*s) return 0;
    char *end = NULL;
    const double v = strtod(s, &end);
    if (!end || *end || !isfinite(v)) return 0;
    *out = (float)v;
    return 1;
}

static int interactive_chat(
        rl_engine *engine,
        const rl_engine_info *info,
        rl_tokenizer *tokenizer,
        rl_sampler *sampler,
        const rl_sampler_params *sampler_params,
        const char *initial_prompt,
        const char *system_prompt,
        uint32_t max_tokens,
        int stream,
        int show_stats,
        int json_stats,
        char *error,
        size_t error_cap) {
    char *line = NULL;
    size_t line_cap = 0;
    const char *message = initial_prompt;
    uint32_t close_token = UINT32_MAX;
    int first_turn = 1;
    const int32_t im_end = rl_tokenizer_lookup(tokenizer, "<|im_end|>");
    if (im_end < 0) {
        snprintf(error, error_cap, "tokenizer is missing <|im_end|>");
        return 0;
    }

    printf("Red Lite chat ready. Commands: /reset, /help, /quit\n");
    printf("The model stays loaded and the conversation is kept. Ctrl-C stops an answer; at the prompt it quits.\n\n");
    uint32_t turn = 0;
    for (;;) {
        g_interrupt = 0;
        if (!message) {
            printf("you> ");
            fflush(stdout);
            const ssize_t got = getline(&line, &line_cap, stdin);
            if (got < 0) { printf("\n"); break; } /* EOF, or Ctrl-C at the prompt (EINTR) */
            size_t len = (size_t)got;
            while (len && (line[len - 1u] == '\n' || line[len - 1u] == '\r')) line[--len] = '\0';
            if (!len) continue;
            if (strcmp(line, "/quit") == 0 || strcmp(line, "/exit") == 0) break;
            if (strcmp(line, "/help") == 0) {
                printf("/reset  clear the conversation and the model state\n/quit   leave the chat\n\n");
                continue;
            }
            if (strcmp(line, "/reset") == 0) {
                if (!rl_engine_reset(engine, RL_BACKEND_GPU, error, error_cap)) { free(line); return 0; }
                rl_sampler_free(sampler);
                if (!rl_sampler_init(sampler, sampler_params, info->vocab)) {
                    snprintf(error, error_cap, "sampler reset failed"); free(line); return 0;
                }
                first_turn = 1;
                close_token = UINT32_MAX;
                printf("Conversation cleared.\n\n");
                continue;
            }
            message = line;
        } else {
            printf("you> %s\n", message);
        }

        const size_t template_cap = strlen(message) + (system_prompt ? strlen(system_prompt) : 0u) + 256u;
        char *templated = (char *)malloc(template_cap);
        if (!templated) { snprintf(error, error_cap, "chat template allocation failed"); free(line); return 0; }
        const int template_ok = first_turn
            ? rl_tokenizer_chat_prompt(system_prompt, message, templated, template_cap)
            : rl_tokenizer_chat_continuation(message, templated, template_cap);
        if (!template_ok) {
            snprintf(error, error_cap, "chat prompt is too long"); free(templated); free(line); return 0;
        }
        const int32_t encoded = rl_tokenizer_encode(tokenizer, templated, strlen(templated), 1,
            NULL, 0, error, error_cap);
        if (encoded < 0) { free(templated); free(line); return 0; }
        const uint32_t prefix = first_turn ? 0u : 1u;
        const uint32_t prompt_tokens = (uint32_t)encoded + prefix;
        const uint32_t position = rl_engine_position(engine, RL_BACKEND_GPU);
        if (position > info->context || prompt_tokens > info->context - position ||
            max_tokens > info->context - position - prompt_tokens) {
            fprintf(stderr, "Context exhausted (%u + %u + %u > %u). Use /reset.\n",
                position, prompt_tokens, max_tokens, info->context);
            free(templated);
            message = NULL;
            continue;
        }
        uint32_t *ids = (uint32_t *)malloc((size_t)prompt_tokens * sizeof(uint32_t));
        float *logits = (float *)malloc((size_t)info->vocab * sizeof(float));
        char *answer = (char *)malloc((size_t)max_tokens * 512u + 1u);
        if (!ids || !logits || !answer) {
            snprintf(error, error_cap, "interactive turn allocation failed");
            free(ids); free(logits); free(answer); free(templated); free(line); return 0;
        }
        if (prefix) ids[0] = close_token;
        if (rl_tokenizer_encode(tokenizer, templated, strlen(templated), 1,
                ids + prefix, (size_t)encoded, error, error_cap) != encoded) {
            snprintf(error, error_cap, "interactive tokenization changed between passes");
            free(ids); free(logits); free(answer); free(templated); free(line); return 0;
        }
        free(templated);

        rl_engine_step_stats step = {0};
        const double prefill_start = now_ms();
        if (!rl_engine_prefill(engine, RL_BACKEND_GPU, ids, prompt_tokens, logits, &step, error, error_cap)) {
            free(ids); free(logits); free(answer); free(line); return 0;
        }
        const double prefill_ms = now_ms() - prefill_start;
        printf("redlite> ");
        fflush(stdout);
        size_t answer_len = 0;
        uint32_t generated = 0, spec_turn = 0;
        int stopped_on_eog = 0, interrupted = 0;
        const double generation_start = now_ms();
        while (generated < max_tokens) {
            if (g_interrupt) {
                /* every emitted token is already in the state: close the answer like a length stop */
                close_token = (uint32_t)im_end;
                interrupted = 1;
                break;
            }
            const uint32_t next = rl_sampler_sample(sampler, logits);
            if (rl_tokenizer_is_eog(tokenizer, next)) {
                close_token = next;
                stopped_on_eog = 1;
                break;
            }
            char piece[512];
            const int32_t bytes = rl_tokenizer_decode(tokenizer, next, piece, sizeof(piece));
            if (bytes < 0 || answer_len + (size_t)bytes > (size_t)max_tokens * 512u) {
                snprintf(error, error_cap, "token decode failed");
                free(ids); free(logits); free(answer); free(line); return 0;
            }
            memcpy(answer + answer_len, piece, (size_t)bytes);
            answer_len += (size_t)bytes;
            generated++;
            if (stream && bytes) { fwrite(piece, 1, (size_t)bytes, stdout); fflush(stdout); }
            if (generated == max_tokens) {
                if (!rl_engine_step(engine, RL_BACKEND_GPU, next, NULL, &step, error, error_cap)) {
                    free(ids); free(logits); free(answer); free(line); return 0;
                }
                close_token = (uint32_t)im_end;
                break;
            }
            if (!rl_engine_step(engine, RL_BACKEND_GPU, next, logits, &step, error, error_cap)) {
                free(ids); free(logits); free(answer); free(line); return 0;
            }
            spec_turn += step.speculative;
        }
        const double generation_ms = now_ms() - generation_start;
        answer[answer_len] = '\0';
        if (!stream) fwrite(answer, 1, answer_len, stdout);
        printf(interrupted ? " [interrupted]\n\n" : "\n\n");
        const char *finish = interrupted ? "interrupted" : stopped_on_eog ? "stop" : "length";
        const double decode_tps = generated > 1u ? (generated - 1u) * 1000.0 / generation_ms : 0.0;
        turn++;
        if (show_stats) {
            fprintf(stderr, "[turn: prompt=%u tokens %.2fs, answer=%u tokens %.2fs (%.1f tok/s, %u GPU-routed), position=%u/%u, %s]\n\n",
                prompt_tokens, prefill_ms / 1000.0, generated, generation_ms / 1000.0, decode_tps, spec_turn,
                rl_engine_position(engine, RL_BACKEND_GPU), info->context, finish);
        }
        if (json_stats) {
            fprintf(stderr, "{\"turn\":%u,\"prompt_tokens\":%u,\"prefill_ms\":%.1f,\"prefill_tok_s\":%.2f,"
                "\"generated_tokens\":%u,\"decode_ms\":%.1f,\"decode_tok_s\":%.2f,\"gpu_routed\":%u,"
                "\"position\":%u,\"context\":%u,\"finish\":\"%s\"}\n",
                turn, prompt_tokens, prefill_ms, prefill_ms > 0 ? prompt_tokens * 1000.0 / prefill_ms : 0.0,
                generated, generation_ms, decode_tps, spec_turn,
                rl_engine_position(engine, RL_BACKEND_GPU), info->context, finish);
        }
        free(ids); free(logits); free(answer);
        first_turn = 0;
        message = NULL;
    }
    free(line);
    return 1;
}

int main(int argc, char **argv) {
    if (argc < 2 || strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0) { usage(argc < 2 ? stderr : stdout); return argc < 2 ? 2 : 0; }
    const char *model = argv[1];
    const char *prompt = NULL, *system_prompt = NULL, *tokens_out = NULL;
    int raw = 0, stream = 1, stats = 0, interactive = 0, json = 0;
    uint32_t max_tokens = 256u;
    rl_engine_config cfg;
    rl_engine_config_default(&cfg);
    rl_sampler_params sp;
    rl_sampler_params_default(&sp);
    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--raw") == 0) { raw = 1; continue; }
        if (strcmp(argv[i], "--interactive") == 0 || strcmp(argv[i], "-i") == 0) { interactive = 1; continue; }
        if (strcmp(argv[i], "--no-stream") == 0) { stream = 0; continue; }
        if (strcmp(argv[i], "--stats") == 0) { stats = 1; continue; }
        if (strcmp(argv[i], "--json") == 0) { json = 1; continue; }
        if (i + 1 >= argc) { usage(stderr); return 2; }
        if (strcmp(argv[i], "--prompt") == 0) prompt = argv[++i];
        else if (strcmp(argv[i], "--system") == 0) system_prompt = argv[++i];
        else if (strcmp(argv[i], "--tokens-out") == 0) tokens_out = argv[++i];
        else if (strcmp(argv[i], "--max-tokens") == 0) { if (!parse_u32(argv[++i], &max_tokens)) return 2; }
        else if (strcmp(argv[i], "--temperature") == 0) { if (!parse_f32(argv[++i], &sp.temperature)) return 2; }
        else if (strcmp(argv[i], "--top-k") == 0) { if (!parse_u32(argv[++i], &sp.top_k)) return 2; }
        else if (strcmp(argv[i], "--top-p") == 0) { if (!parse_f32(argv[++i], &sp.top_p)) return 2; }
        else if (strcmp(argv[i], "--min-p") == 0) { if (!parse_f32(argv[++i], &sp.min_p) || sp.min_p < 0.0f || sp.min_p > 1.0f) return 2; }
        else if (strcmp(argv[i], "--seed") == 0) { char *end = NULL; sp.seed = strtoull(argv[++i], &end, 10); if (!end || *end) return 2; }
        else if (strcmp(argv[i], "--context") == 0) { if (!parse_u32(argv[++i], &cfg.context)) return 2; }
        else if (strcmp(argv[i], "--cache-mib") == 0) { if (!rl_engine_parse_cache_mib(argv[++i], &cfg.cache_mib)) return 2; }
        else if (strcmp(argv[i], "--batch") == 0) { if (!parse_u32(argv[++i], &cfg.prefill_batch)) return 2; }
        else { fprintf(stderr, "unknown option %s\n", argv[i]); usage(stderr); return 2; }
    }
    if (!prompt && !interactive) { fprintf(stderr, "--prompt is required (or use --interactive)\n"); return 2; }
    if (interactive && raw) { fprintf(stderr, "--raw cannot be combined with --interactive\n"); return 2; }
    if (interactive && tokens_out) { fprintf(stderr, "--tokens-out is not supported in interactive mode\n"); return 2; }
    if (!max_tokens) { fprintf(stderr, "--max-tokens must be greater than zero\n"); return 2; }

    /* no SA_RESTART: Ctrl-C at the interactive prompt interrupts getline() */
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = on_sigint;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, NULL);

    char error[512] = {0};
    const double t_open = now_ms();
    cfg.enable_cpu = 0; cfg.enable_gpu = 1;
    rl_engine *e = rl_engine_open(model, &cfg, error, sizeof(error));
    if (!e) { fprintf(stderr, "engine open failed: %s\n", error); return 1; }
    const rl_engine_info *in = rl_engine_info_get(e);
    rl_gguf_model g;
    if (!rl_gguf_model_open(model, &g, error, sizeof(error))) { fprintf(stderr, "tokenizer model open failed: %s\n", error); return 1; }
    rl_tokenizer *tk = rl_tokenizer_create(&g, error, sizeof(error));
    if (!tk) { fprintf(stderr, "tokenizer failed: %s\n", error); return 1; }
    const double open_ms = now_ms() - t_open;
    int template_from_gguf = 0;
    const char *template_source = rl_tokenizer_chat_template_source(g.chat_template, &template_from_gguf);
    if (stats && !raw) fprintf(stderr, "chat template        : %s\n", template_source);

    if (interactive) {
        rl_sampler sampler;
        if (!rl_sampler_init(&sampler, &sp, in->vocab)) { fprintf(stderr, "sampler allocation failed\n"); return 1; }
        const int ok = interactive_chat(e, in, tk, &sampler, &sp, prompt, system_prompt,
            max_tokens, stream, stats, json, error, sizeof(error));
        if (!ok) fprintf(stderr, "chat failed: %s\n", error);
        rl_sampler_free(&sampler);
        rl_tokenizer_destroy(tk);
        rl_gguf_model_close(&g);
        rl_engine_close(e);
        return ok ? 0 : 1;
    }

    char *templated = (char *)malloc(strlen(prompt) + (system_prompt ? strlen(system_prompt) : 0u) + 256u);
    if (!templated) return 1;
    if (raw) strcpy(templated, prompt);
    else if (!rl_tokenizer_chat_prompt(system_prompt, prompt, templated, strlen(prompt) + (system_prompt ? strlen(system_prompt) : 0u) + 256u)) { fprintf(stderr, "prompt too long\n"); return 1; }
    const int32_t needed = rl_tokenizer_encode(tk, templated, strlen(templated), 1, NULL, 0, error, sizeof(error));
    if (needed < 0) { fprintf(stderr, "tokenize failed: %s\n", error); return 1; }
    if (needed == 0) { fprintf(stderr, "prompt produced no tokens; nothing to prefill\n"); return 1; }
    if ((uint32_t)needed + max_tokens > in->context) { fprintf(stderr, "prompt (%d) + max-tokens (%u) exceed --context %u\n", needed, max_tokens, in->context); return 1; }
    uint32_t *ids = (uint32_t *)malloc(((size_t)needed + max_tokens + 1u) * sizeof(uint32_t));
    if (!ids) { fprintf(stderr, "token buffer allocation failed\n"); return 1; }
    if (rl_tokenizer_encode(tk, templated, strlen(templated), 1, ids, (size_t)needed, error, sizeof(error)) != needed) {
        fprintf(stderr, "tokenization changed between passes\n"); return 1;
    }
    const uint32_t prompt_len = (uint32_t)needed;

    float *logits = (float *)malloc((size_t)in->vocab * sizeof(float));
    rl_sampler sampler;
    if (!logits || !rl_sampler_init(&sampler, &sp, in->vocab)) { fprintf(stderr, "sampler allocation failed\n"); return 1; }

    /* prefill: batched Metal ingestion of the prompt (chunks of --batch), LM head on the last token */
    rl_engine_step_stats st;
    const double t_prefill = now_ms();
    if (!rl_engine_prefill(e, RL_BACKEND_GPU, ids, prompt_len, logits, &st, error, sizeof(error))) {
        fprintf(stderr, "prefill failed: %s\n", error); return 1;
    }
    const double prefill_ms = now_ms() - t_prefill;

    /* generation */
    /* every decoded piece is at most 512 bytes (see rl_tokenizer_decode), so this bound is exact */
    char *text = (char *)malloc((size_t)max_tokens * 512u + 1u);
    if (!text) { fprintf(stderr, "text buffer allocation failed\n"); return 1; }
    size_t text_len = 0;
    uint32_t generated = 0;
    const double t_gen = now_ms();
    double last_routed_ms = 0.0;
    uint64_t ssd_start = st.ssd_bytes, reads_start = st.ssd_reads, loads_start = st.expert_loads;
    uint64_t hits_start = st.cache_hits, misses_start = st.cache_misses;
    uint32_t spec_tokens = 0, spec_fallbacks = 0;
    int interrupted = 0, stopped_on_eog = 0;
    while (generated < max_tokens) {
        if (g_interrupt) { interrupted = 1; break; }
        const uint32_t next = rl_sampler_sample(&sampler, logits);
        ids[prompt_len + generated] = next;
        generated++;
        if (rl_tokenizer_is_eog(tk, next)) { stopped_on_eog = 1; break; }
        char piece[512];
        const int32_t n = rl_tokenizer_decode(tk, next, piece, sizeof(piece));
        if (n > 0 && text_len + (size_t)n <= (size_t)max_tokens * 512u) {
            memcpy(text + text_len, piece, (size_t)n);
            text_len += (size_t)n;
            if (stream) { fwrite(piece, 1, (size_t)n, stdout); fflush(stdout); }
        }
        if (generated >= max_tokens) break;
        if (!rl_engine_step(e, RL_BACKEND_GPU, next, logits, &st, error, sizeof(error))) {
            fprintf(stderr, "\ndecode failed: %s\n", error); return 1;
        }
        spec_tokens += st.speculative; spec_fallbacks += st.speculative_fallback;
        last_routed_ms = st.routed_ms;
    }
    const double gen_ms = now_ms() - t_gen;
    text[text_len] = '\0';
    if (!stream) printf("%s", text);
    printf(interrupted ? " [interrupted]\n" : "\n");

    if (tokens_out) {
        FILE *f = fopen(tokens_out, "w");
        if (f) { for (uint32_t i = 0; i < prompt_len + generated; ++i) fprintf(f, "%u\n", ids[i]); fclose(f); }
    }
    if (stats) {
        const uint32_t decoded = generated > 0 ? generated - 1u : 0u; /* forward passes after prefill */
        fprintf(stderr, "\n--- redlite-generate stats ---\n");
        {
            double pre_ms = 0.0;
            fprintf(stderr, "model open           : %.1f ms%s\n", open_ms,
                rl_engine_experts_preloaded(e, &pre_ms) ? " (every routed expert preloaded: full residency)" : "");
            if (pre_ms > 0.0) fprintf(stderr, "expert preload       : %.1f ms\n", pre_ms);
        }
        fprintf(stderr, "prompt tokens        : %u (%.1f ms, %.2f tok/s)\n", prompt_len, prefill_ms, prompt_len * 1000.0 / prefill_ms);
        fprintf(stderr, "generated tokens     : %u (%.1f ms, %.2f tok/s over %u decode passes)\n", generated, gen_ms,
            decoded ? decoded * 1000.0 / gen_ms : 0.0, decoded);
        fprintf(stderr, "last step            : %.1f ms (rec %.1f attn %.1f router %.1f routed %.1f [load %.1f gpu %.1f] shared %.1f out %.1f) dense GPU %.1f ms\n",
            st.total_ms, st.recurrent_ms, st.attention_ms, st.router_ms, st.routed_ms, st.routed_load_ms, st.routed_gpu_ms, st.shared_ms, st.output_ms, st.gpu_ms);
        fprintf(stderr, "decode path          : %u GPU-routed tokens (%u per-layer early-outs), %u synchronous\n", spec_tokens, spec_fallbacks, decoded - spec_tokens);
        fprintf(stderr, "expert cache         : hits=%" PRIu64 " misses=%" PRIu64 " loads=%" PRIu64 " resident=%u/%u slots (hit rate %.1f%%)\n",
            st.cache_hits - hits_start, st.cache_misses - misses_start, st.expert_loads - loads_start, st.resident_slots, st.slot_capacity,
            (st.cache_hits + st.cache_misses - hits_start - misses_start) ? 100.0 * (double)(st.cache_hits - hits_start) / (double)(st.cache_hits + st.cache_misses - hits_start - misses_start) : 0.0);
        fprintf(stderr, "SSD expert traffic   : %.1f MiB / %" PRIu64 " reads during generation (%.2f MiB/token)\n",
            (double)(st.ssd_bytes - ssd_start) / (1024.0 * 1024.0), st.ssd_reads - reads_start,
            decoded ? (double)(st.ssd_bytes - ssd_start) / (1024.0 * 1024.0) / decoded : 0.0);
        fprintf(stderr, "SSD expert total     : %.1f MiB / %" PRIu64 " reads since open\n", (double)st.ssd_bytes / (1024.0 * 1024.0), st.ssd_reads);
        fprintf(stderr, "peak RSS             : %.1f MiB\n", (double)peak_rss_bytes() / (1024.0 * 1024.0));
        fprintf(stderr, "physical footprint   : %.1f MiB (process, incl. Metal buffers; mmap'd weights are file-backed)\n", (double)phys_footprint_bytes() / (1024.0 * 1024.0));
        (void)last_routed_ms;
    }
    if (json) {
        const uint32_t decoded = generated > 0 ? generated - 1u : 0u;
        const uint64_t hits = st.cache_hits - hits_start, misses = st.cache_misses - misses_start;
        double pre_ms = 0.0;
        const int preloaded = rl_engine_experts_preloaded(e, &pre_ms);
        fprintf(stderr, "{\"open_ms\":%.1f,\"experts_preloaded\":%s,\"preload_ms\":%.1f,\"chat_template_from_gguf\":%s,"
            "\"prompt_tokens\":%u,\"prefill_ms\":%.1f,\"prefill_tok_s\":%.2f,"
            "\"generated_tokens\":%u,\"decode_passes\":%u,\"decode_ms\":%.1f,\"decode_tok_s\":%.2f,"
            "\"gpu_routed\":%u,\"early_outs\":%u,\"cache_hits\":%llu,\"cache_misses\":%llu,"
            "\"expert_loads\":%llu,\"ssd_mib\":%.1f,\"peak_rss_mib\":%.1f,\"phys_footprint_mib\":%.1f,"
            "\"cache_mib\":%llu,\"context\":%u,\"batch\":%u,\"finish\":\"%s\"}\n",
            open_ms, preloaded ? "true" : "false", pre_ms, template_from_gguf ? "true" : "false",
            prompt_len, prefill_ms, prefill_ms > 0 ? prompt_len * 1000.0 / prefill_ms : 0.0,
            generated, decoded, gen_ms, decoded ? decoded * 1000.0 / gen_ms : 0.0,
            spec_tokens, spec_fallbacks, (unsigned long long)hits, (unsigned long long)misses,
            (unsigned long long)(st.expert_loads - loads_start), (double)(st.ssd_bytes - ssd_start) / (1024.0 * 1024.0),
            (double)peak_rss_bytes() / (1024.0 * 1024.0), (double)phys_footprint_bytes() / (1024.0 * 1024.0),
            (unsigned long long)(cfg.cache_mib == RL_ENGINE_CACHE_FULL ? rl_engine_full_residency_mib(e) : cfg.cache_mib), in->context, rl_engine_prefill_batch(e),
            interrupted ? "interrupted" : stopped_on_eog ? "stop" : "length");
    }
    rl_sampler_free(&sampler);
    free(logits); free(ids); free(text); free(templated);
    rl_tokenizer_destroy(tk);
    rl_gguf_model_close(&g);
    rl_engine_close(e);
    return interrupted ? 130 : 0;
}
