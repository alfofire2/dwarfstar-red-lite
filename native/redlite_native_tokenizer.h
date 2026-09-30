#pragma once

/*
 * Native byte-level BPE tokenizer for the Qwen3-Next GGUF (tokenizer.ggml.model
 * = gpt2, tokenizer.ggml.pre = qwen2). Mirrors the pinned llama.cpp behaviour:
 * special-token partitioning, the qwen2 pre-tokenizer split, GPT-2 byte
 * encoding, rank-ordered merges and byte fallback.
 */

#include "redlite_native_gguf_dir.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct rl_tokenizer rl_tokenizer;

rl_tokenizer *rl_tokenizer_create(const rl_gguf_model *model, char *error, size_t error_cap);
void rl_tokenizer_destroy(rl_tokenizer *t);

uint32_t rl_tokenizer_vocab_size(const rl_tokenizer *t);
int32_t rl_tokenizer_eos(const rl_tokenizer *t);
int32_t rl_tokenizer_bos(const rl_tokenizer *t);
int rl_tokenizer_is_control(const rl_tokenizer *t, uint32_t id);
int rl_tokenizer_is_eog(const rl_tokenizer *t, uint32_t id); /* end-of-generation: eos or <|endoftext|>/<|im_end|> */

/*
 * Encode UTF-8 text. parse_special=1 recognizes special-token strings such as
 * <|im_start|>. Returns the token count or -1 (error). If out is NULL or
 * out_cap is too small, the needed count is still returned.
 */
int32_t rl_tokenizer_encode(const rl_tokenizer *t, const char *text, size_t text_len, int parse_special,
                            uint32_t *out, size_t out_cap, char *error, size_t error_cap);

/* Decode one token to UTF-8 bytes (control tokens decode to their literal text). Returns bytes written or -1. */
int32_t rl_tokenizer_decode(const rl_tokenizer *t, uint32_t id, char *out, size_t out_cap);

/* Look up a token id by its literal vocabulary string (e.g. "<|im_end|>"); -1 if absent. */
int32_t rl_tokenizer_lookup(const rl_tokenizer *t, const char *piece);

/*
 * Apply the Instruct chat template for a single user turn (optional system
 * prompt) with add_generation_prompt=true. Writes the templated text.
 */
int rl_tokenizer_chat_prompt(const char *system_prompt, const char *user_prompt, char *out, size_t out_cap);

/* Continue an existing ChatML conversation after the previous assistant EOG token. */
int rl_tokenizer_chat_continuation(const char *user_prompt, char *out, size_t out_cap);

/*
 * Which chat format to use given the GGUF tokenizer.chat_template (may be NULL). The template is
 * not interpreted (no Jinja): a ChatML template is confirmed, anything else falls back to the
 * built-in ChatML. Returns a short human-readable description; *from_gguf is 1 when confirmed.
 */
const char *rl_tokenizer_chat_template_source(const char *gguf_template, int *from_gguf);

#ifdef __cplusplus
}
#endif
