# Red Lite dev35 — long-context decode: grouped split-K attention

Status: **done** on branch `dev/long-decode` (from `dev/bounded-prefill` at 40e4fc0). Apple M4 Max
48 GiB only.

## Finding

Decode slows with position because of attention: at ~8400 positions the stage took 252.8 ms per
64 tokens (~4 ms/token, `RL_ENGINE_PROFILE=1`), against a floor of ~1 ms/token for reading the
12 layers' KV cache once (~413 MB at ~400 GB/s). The dev26 split-K kernel ran one threadgroup per
(256-position block, **query** head): the 8 query heads sharing a KV head each read the same K and
V rows.

## Change

`attn_gqa_split_g`: one threadgroup per (block, **KV** head) computes all its query heads
(scores: one thread per position, `float4` K loads, 8 accumulators; values: one thread per
dimension, 8 accumulators); the merge kernel is shared and now takes the block length.
`RL_ENGINE_ATTN_GROUP=0` restores the dev26 kernel; `RL_ENGINE_ATTN_BLK` sets the block.

Attention stage at ~8400 positions (IQ2_XXS, full residency, 64 tokens, single profiled runs):

| kernel | ms |
|---|---:|
| dev26 per query head, blocks of 256 | 252.8 |
| grouped, blocks of 256 | 164.3 |
| **grouped, blocks of 128 (default)** | **112.8** |
| grouped, blocks of 64 / 32 | 146.5 / 180.2 |
| grouped, 4 threads per position (blocks of 256) | 175.1 (reverted) |

Decode, `redlite-generate --raw` on the fixture text repeated to P tokens, 128 tokens, full
residency, cooled alternating A/B (3 runs each):

| position | dev26 kernel | grouped | runs |
|---|---:|---:|---|
| ~4096 | 66.53 tok/s | 67.79 tok/s | 68.25/65.96/66.53 vs 67.32/67.79/69.21 |
| ~8192 | 57.80 tok/s | **65.95 tok/s** | 59.06/57.68/57.80 vs 65.91/66.04/65.95 |

## Parity

Kernel self-test: the grouped kernel vs a double-precision GQA at 6 lengths (worst abs 7.6e-8).
1100-position context vs llama.cpp unchanged (IQ2_XXS 0.99 / KL 3.5e-3, IQ3_XXS 0.57 / KL 3.9e-4,
full and 4 GiB); `long_positions.sh`: 4096 PASS (2.30 / KL 7.2e-3), 8192 PASS (4.48 / KL 2.2e-6).

## Validation

`rm -rf .deps/redmetal`, `make redmetal`, `make native`, `make sanitize`: 0 warnings; `make test`
and ruff green; `scripts/regress_m4.sh`: IQ2_XXS **49/49**, IQ3_XXS **36 passed, 0 failed, 13
skipped** (the IQ2_XXS-layout dense stage tools).

## Scope boundary

- M4 Max 48 GiB only; the same kernel serves the 4 GiB path (not re-benchmarked there).
- Requires head_dim 256 and at most 8 query heads per KV head (Qwen3-Next: 16 / 2); otherwise the
  dev26 kernel runs.
