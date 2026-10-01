# Red Lite dev37 — expert slots of each layer's own size

Status: **done** on branch `dev/expert-slots` (from `dev/iq3m` at 93789ce). Measured on the Apple M4 Max
48 GiB only; the 4 GiB-cache path it targets is the 24 GiB M4 Pro default and was **not** run there.

## Why (and how the change was chosen)

The bounded expert cache (4 GiB, the default on 24 GiB) decides decode speed on small Macs: every miss
is an SSD or page-cache read. Recent work on MoE offloading (Mira, arXiv 2609.38090; SpecPrefetch,
2607.24787; OLED-MoE, 2609.33385) proposes smarter replacement and staging. Before writing any C, the
decision was made on data:

1. `RL_ROUTE_TRACE=file redlite-generate ...` (new) records each decoded token's router ids
   (48 × 10 uint16). Six prompts (explanation, code, arithmetic, Italian prose, a story, a list),
   511 tokens each, IQ2_XXS.
2. `scripts/dev/cache_policy_sim.py` (new) replays them through policies at the real slot count. Its
   LRU equals the engine's (84.4 % cold vs 85.2 % measured with the post-prompt warm cache).

Misses per decoded token, 4 GiB (4723 slots), all six traces:

| policy | misses/token |
|---|---:|
| LRU (the engine) | 64.1 |
| decayed frequency, half-life 250 / 1000 transactions (Mira-style score) | 63.5 / 60.9 |
| segmented LRU, protected 50 / 80 / 95 % | 62.1 / 61.1 / 64.7 |
| Belady (offline optimum, unreachable) | 34.7 |
| **LRU, slots of each layer's own size (same 4 GiB)** | **50.5** |

Replacement policies gain at most 5 %. The real waste was the slot size: every slot had the size of the
largest expert triplet (an IQ2_XS layer, 888 KiB), while 37 of the 48 layers hold IQ1_M experts of
672 KiB. A quarter of the cache was padding.

## Change

- `redmetal_topk.m`: optional slot size classes (`redmetal_topk_pool_set_classes`); each class has
  its own lazily allocated slabs; payload checks use the slot's class size. The Python bridge keeps
  one uniform class.
- `redlite_native_cache.[ch]`: optional LRU classes (`rl_native_lru_set_classes`): a class is a
  contiguous range of entries and a key only evicts entries of its layer's class.
- `redlite_native_metal.c`: one class per distinct 4 KiB-aligned layer triplet size, the same number
  of slots for every layer (budget / sum of layer slot sizes, at most 512). Uniform as before with
  one size, below 16 slots per layer (small stage-tool caches) or with `RL_POOL_CLASSES=0`.
- Full residency is 512 × the sum of the layers' slot sizes: `--cache-mib full` is 17316 MiB for
  IQ2_XXS (was 21312) and 28800 MiB for IQ3_XXS (was 29376); `redlite/planner.py` uses the same
  formula.

## Measurements (M4 Max 48 GiB, IQ2_XXS)

Engine counters, same six prompts, 512 tokens, 4 GiB cache (output identical, so the comparison is exact):

| prefetch | misses/token uniform | misses/token classes |
|---|---:|---:|
| off | 60.0 | **45.4** (−24 %) |
| on (default) | 39.2 | **29.8** (−24 %) |

Decode tok/s, alternated runs with 45–60 s cooling (`RL_POOL_CLASSES=0` vs default):

| case | uniform | classes |
|---|---|---|
| 4 GiB, prefetch off, Italian prose | 46.89 / 49.52 / 49.85 (median 49.5) | 51.06 / 51.29 / 49.75 (median 51.1) |
| 4 GiB, prefetch on, arithmetic | 47.48 / 46.85 / 46.26 (median 46.9) | 47.10 / 48.31 / 49.12 (median 48.3) |
| full residency | 70.58 / 72.33 / 72.31 (median 72.3) | 72.63 / 71.61 / 72.21 (median 72.2) |

Full residency footprint: 22445 → **18447 MiB** (−3.9 GiB) at the same speed.

On this Mac a miss is a copy from a warm page cache, so fewer misses gain only ~3 %. On a 24 GiB Mac
misses are SSD reads; the gain there is expected to be larger but is **not measured**.

To repeat on another Mac: the prompts are `tests/fixtures/cache_trace_prompts.txt`; run each with
`redlite-generate MODEL --prompt "..." --max-tokens 512 --cache-mib 4096 --no-stream --json` under
`RL_POOL_CLASSES=0` and the default, and compare `cache_misses` and `decode_tok_s`.

## Validation

Parity with the classes active: `quick_parity.sh --long --batch 2048` (engine CPU vs Metal, GPU-routed,
logits and greedy vs llama.cpp, 1200-token long context): PASS. Offline test `test_lru_classes`
(class-local eviction, slot ranges, oversize selection refused). `rm -rf .deps/redmetal`, `make redmetal`, `make native`, `make sanitize`: 0 warnings; `make test` and ruff green; `scripts/regress_m4.sh`: IQ2_XXS **49/49**, IQ3_XXS **36 passed, 0 failed, 13 skipped**.

## Also measured, not done (see `docs/WHAT_DID_NOT_WORK.md`)

- Speculative decoding: the GGUF has no multi-token-prediction head, and the batched prefill path is
  slower than token-by-token decode at small batches (8-token chunks 137 ms vs 8 decode steps 98 ms),
  so verification needs new small-batch kernels.
- LensVLM (apple/lensvlm-9b): a separate 9B vision-language model (Qwen3.5-9B + Qwen3-VL encoder); our
  text-only Qwen3-Next cannot read image tokens. See the scope boundary.

## Scope boundary

- M4 Max 48 GiB only. No number here is from the M4 Pro 24 GiB; the 4 GiB-cache gain there is the
  open question this milestone prepares (`RL_POOL_CLASSES=0` exists for that A/B).
- Hit rates come from six prompts of 512 tokens; other workloads will differ.
- Replacement policy is still LRU within a class; the simulator shows ≤ 5 % more from smarter
  policies and 46 % from an unreachable optimum.
- LensVLM is not implemented: it is another model family and would need a vision encoder and a second
  language model in the runtime.
