# Red Lite dev46 — expert-cache options for 24 GiB Macs, and engine perplexity

Status: **implemented and measured on the M4 Max 48 GiB** on branch `dev/small-mac-options`. The speed effect
these options are for must be measured on the M4 Pro 24 GiB, with `scripts/dev/small_mac_ab.sh`.

## Why

With a 4 GiB expert cache, every miss is a load. On the 48 GiB M4 Max it is a copy from a warm page cache, so
decode barely depends on it. On a 24 GiB Mac the GGUF does not fit in the page cache and a miss is an SSD read.
The October survey (`docs/RESEARCH_2026_10.md`) points at two training-free levers:

- cache-aware routing (arXiv 2412.00099);
- the expert I/O path (Flash-MoE: page cache vs. own pool, prefetch side effects).

## Changes (all opt-in, defaults unchanged)

- **`RL_ROUTE_CACHE_BIAS=λ`.** Experts already in the cache rank λ higher in the top-10 selection (GPU
  `rl_route` and the CPU selection of the synchronous path). The mixture weights still come from the unbiased
  logits, renormalized over the selected experts. This changes outputs, so it is off by default; parity
  checks run without it.
- **`RL_POOL_NOCACHE=1`.** Expert reads use `F_NOCACHE` (they bypass the page cache).
- **`redlite-engine perplexity MODEL --tokens IDS [--context 512] [--cache-mib N]`.** Perplexity of the
  engine itself, token by token on the Metal decode path, so runtime options apply. It scores the second half
  of each `--context` chunk, each chunk starting from a reset state. The llama.cpp oracle cannot measure
  options that only exist in Red Lite.
- **`scripts/dev/small_mac_ab.sh MODEL [configs]`.** Runs the six prompts of
  `tests/fixtures/cache_trace_prompts.txt` at 4 GiB in each configuration (`default`, `uniform`, `bias05`,
  `nocache`, `noprefetch`). It reports misses per token, MiB read per token, and decode tok/s per prompt.

## Measurements (M4 Max 48 GiB, IQ2_XXS, 4 GiB cache)

Cache bias: misses per token over the six prompts (256 tokens each), and engine perplexity over the first
4096 ids of `tests/fixtures/perplexity_corpus.txt` (8 chunks × 512, second halves scored):

| λ | misses / token | MiB read / token | perplexity |
|---:|---:|---:|---:|
| 0 (default) | 29.5 | 43.1 | 17.586 |
| **0.5** | **22.9 (−22 %)** | **33.1** | **17.583** |
| 1.0 | 22.3 | — | 17.674 (+0.5 %) |
| 2.0 | 22.0 | — | 17.676 (+0.5 %) |

Decode on this Mac is unchanged by the bias (median 52.6 vs 52.0 tok/s, single uncooled runs): its misses are
page-cache copies. Whether 22 % fewer SSD reads speed up a 24 GiB Mac is the open question.

## Validation

`rm -rf .deps/redmetal`, `make redmetal`, `make native`, `make sanitize`: 0 warnings; `make test` and ruff green; `scripts/regress_m4.sh` (defaults, no bias): IQ2_XXS **50/50**, IQ3_XXS **37 passed, 0 failed, 13 skipped**; `quick_parity.sh --long --batch 2048` on both models: PASS. The kernel self-test passes the zero bias to `rl_route`.

## Scope boundary

- No speed claim. The options exist for the M4 Pro 24 GiB A/B; nothing was run there yet.
- The bias changes which experts run. The perplexity sample is 2048 scored tokens on one corpus. That is
  enough to reject a large loss, not to prove there is none.
- `F_NOCACHE` was not measured here (on 48 GiB the page cache is the point of comparison).
