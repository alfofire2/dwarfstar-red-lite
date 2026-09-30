#pragma once

/*
 * Complete GGUF v2/v3 directory + metadata reader for the native Red Lite engine.
 *
 * Unlike the routed-expert map (redlite_native_gguf.h), this reader keeps the
 * entire tensor directory, the Qwen3-Next hyper-parameters, the tokenizer
 * vocabulary/merges and an optional read-only mmap of the whole file so every
 * dense tensor can be addressed in place without positional reads.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RL_GGUF_MAX_DIMS 8u

typedef struct {
    char *name;
    uint32_t n_dims;
    uint64_t shape[RL_GGUF_MAX_DIMS];
    uint32_t ggml_type;
    uint64_t offset;        /* absolute file offset of the payload */
    uint64_t span_bytes;    /* bytes until the next tensor (or EOF) */
    uint64_t payload_bytes; /* exact row-major payload size derived from type/shape (0 if unknown type) */
} rl_gguf_tensor;

typedef struct {
    /* file */
    char *path;
    uint32_t version;
    uint32_t alignment;
    uint64_t tensor_count;
    uint64_t kv_count;
    uint64_t data_base;
    uint64_t file_size;
    rl_gguf_tensor *tensors; /* sorted by ascending offset */

    /* architecture hyper-parameters (qwen3next.*) */
    char *architecture;
    uint32_t n_layer;
    uint32_t n_embd;
    uint32_t n_ff;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t key_length;
    uint32_t value_length;
    uint32_t rope_dims;
    uint32_t n_expert;
    uint32_t n_expert_used;
    uint32_t n_ff_exp;
    uint32_t n_ff_shexp;
    uint32_t ssm_conv;
    uint32_t ssm_state;
    uint32_t ssm_group;
    uint32_t ssm_dt_rank;
    uint32_t ssm_inner;
    uint32_t context_length;
    uint32_t full_attention_interval;
    float rms_eps;
    float rope_freq_base;

    /* tokenizer */
    char *tokenizer_model;
    char *tokenizer_pre;
    uint32_t vocab_count;
    char **tokens;
    int32_t *token_types;   /* llama token type enum; NULL if absent */
    uint32_t merge_count;
    char **merges;
    int32_t eos_id;
    int32_t bos_id;
    int32_t pad_id;
    int add_bos;
    int add_eos;
    char *chat_template;

    /* mmap (optional) */
    int fd;
    uint8_t *map;
    size_t map_len;
} rl_gguf_model;

/* Parse header/metadata/directory. Does not mmap. */
int rl_gguf_model_open(const char *path, rl_gguf_model *model, char *error, size_t error_cap);

/* Map the entire file read-only (PROT_READ, MAP_PRIVATE). Idempotent. */
int rl_gguf_model_map(rl_gguf_model *model, char *error, size_t error_cap);

void rl_gguf_model_close(rl_gguf_model *model);

const rl_gguf_tensor *rl_gguf_find(const rl_gguf_model *model, const char *name);
const rl_gguf_tensor *rl_gguf_find_layer(const rl_gguf_model *model, uint32_t layer, const char *suffix);

/* Pointer into the mmap for a tensor payload; NULL if the model is not mapped. */
const uint8_t *rl_gguf_tensor_data(const rl_gguf_model *model, const rl_gguf_tensor *tensor);

/* Expected row-major payload bytes for a ggml type/shape; 0 when the type is unknown. */
uint64_t rl_gguf_payload_bytes(uint32_t ggml_type, uint32_t n_dims, const uint64_t *shape);

/* Row bytes for the dense ggml types the engine understands (0 when unsupported). */
size_t rl_gguf_row_bytes(uint32_t ggml_type, uint64_t ncols);

const char *rl_gguf_type_name(uint32_t ggml_type);

#ifdef __cplusplus
}
#endif
