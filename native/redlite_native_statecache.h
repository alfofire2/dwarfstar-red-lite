#pragma once

/*
 * dev43: session state checkpoints on disk (the ds4 "KV cache as a disk citizen" idea, adapted to Qwen3-Next).
 *
 * A file holds the prompt ids of a prefix, the logits of its last token and the engine's raw state at that
 * position (rl_engine_state_write: DeltaNet conv/recurrent states, the prefix's K/V rows). Prefixes are only
 * stored at multiples of the prefill chunk, so a restored session ingests the rest of a prompt in the same
 * chunks as a cold run and reaches a bit-identical state.
 *
 * Files are valid only for the same model (rl_engine_model_tag), the same build and the same chunk size;
 * anything else is ignored. The directory is trimmed to max_bytes, least recently used first.
 */

#include "redlite_native_engine.h"

#include <stddef.h>
#include <stdint.h>

typedef struct {
    const char *dir;     /* NULL or "": disabled */
    uint64_t max_bytes;  /* 0: no limit */
} rl_statecache;

/*
 * Bring a backend from any state to the end of ids[0..count): restore the longest stored prefix (or reset),
 * prefill the rest, and store a checkpoint at the largest chunk-aligned position that has none. `logits`
 * receives the last token's logits; `stats` covers the last prefill call only. loaded/saved: positions
 * restored from / written to disk (0 when none).
 */
int rl_statecache_prefill(rl_engine *engine, rl_engine_backend backend, const rl_statecache *cache,
                          const uint32_t *ids, uint32_t count, float *logits, rl_engine_step_stats *stats,
                          uint32_t *loaded, uint32_t *saved, char *error, size_t error_cap);
