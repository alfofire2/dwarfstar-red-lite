# Red Lite dev21 — GPU-routed decode (full residency)

Target machine for this milestone: Apple M4 Max, 48 GiB. Model: Bartowski
`Qwen_Qwen3-Next-80B-A3B-Instruct-IQ2_XXS.gguf`.

## Why

Decode was ~40 tok/s against 68 tok/s for the pinned llama.cpp fully resident,
and the profile said the gap was not expert loading but the 48 CPU↔GPU round
trips per token: the GPU computes a layer's router logits, the CPU selects the
top-10, resolves their cache slots (loading misses) and encodes the expert
dispatches, the GPU runs them. With a warm cache each round trip is ~0.3 ms of
latency in which the GPU idles (~15 ms per token against ~12 ms of GPU work).

## What changed

- **Residency table** (`redmetal_topk_pool_residency_*`, maintained by
  `redlite_native_metal.c` on every LRU commit and abort): `layers × experts`
  GPU slot addresses, 0 when an expert is not resident. The LRU reservation
  now reports the key a miss evicts (`rl_cache_reservation.evicted_key`).
- **`rl_route` kernel**: softmax top-k selection on the GPU with the same rule
  as `rl_native_router_select_softmax_topk` (candidates ordered by logit, ties
  to the lower index; softmax is monotone so the order is identical), weights
  renormalized over the selected experts, expert lookups in the residency
  table; writes the slot table and weights the single-token expert kernels
  read (`redmetal_topk_pool_encode_device`) and a per-layer miss count.
- **Speculative step** (`step_speculative` in `redmetal_engine.m`): the whole
  token — 48 layers, router, expert lookup, experts, residual, LM head — in one
  command buffer with one wait. Before it, the DeltaNet conv/recurrent states
  are blitted to backups; if any layer selected a non-resident expert the
  states are restored and the token is decoded by the unchanged synchronous
  path (`rl_metal_engine_step_sync`). The LRU stamps of the experts the GPU
  used are refreshed afterwards (`rl_native_metal_touch_resident`).
- **Policy**: a token is attempted speculatively only when the previous token
  needed no expert load; `RL_ENGINE_SPECULATIVE=0` disables the path. With a
  bounded cache (79–93 % hit rate) practically every token misses somewhere
  in its 480 selections, so the path never triggers there and nothing changes
  for 24 GiB machines.
- **Full residency**: when the cache can hold every routed expert
  (`--cache-mib ≥ 21300` for this GGUF: 24 576 slots of 0.87 MiB), every
  expert is loaded at open (`RL_ENGINE_PRELOAD=0` disables; 3.7 s from the
  page cache on the M4 Max) and every token is GPU-routed from the first one.
- **Residency set**: the pool's slabs are attached to the engine queue through
  an `MTLResidencySet` (macOS 15+), so no per-encoder `useResource` is needed.
  Without it the first attempt ran at 11 tok/s: 384 slabs × 2 encoders × 48
  layers of `useResource` made Metal redo the residency of 22 GiB per command
  buffer. The first command buffer after the preload still pays ~2–3 s once;
  the engine takes it at open with a warm-up blit.

## Validation (M4 Max)

`redlite-engine parity --repeat 2`: the second pass replays the sequence on a
warm cache, so it exercises the GPU-routed path against the CPU oracle.

| cache | GPU-routed tokens | router ids | worst layer abs | logits abs | argmax |
|---|---|---|---:|---:|---|
| 4 GiB, 6 tokens × 2 passes | 5 of 12 (pass 2 after the first token) | identical | 3.1e-05 | 1.5e-05 | identical |
| 22 GiB (preloaded), 6 tokens × 2 passes | 12 of 12 | identical | 5.3e-05 | 1.7e-05 | identical |

Greedy generation output is unchanged; `regress_m4.sh` 36/36 plus the new
`engine.parity.gpu_routed` check (runs only on machines with ≥ 40 GiB).

## Throughput (M4 Max, greedy, chat turns of 19–21 prompt tokens)

| configuration | decode tok/s |
|---|---:|
| 4 GiB cache, synchronous (dev20d) | ~40 |
| 22 GiB cache preloaded, `RL_ENGINE_SPECULATIVE=0` | 18 → 36 (cache warming) |
| 22 GiB cache preloaded, GPU-routed | **54–57** (every token speculative, 0 fallbacks) |
| 200-token answer, 22 GiB | 56.6 |
| reference: pinned llama.cpp fully resident (`llama-bench` tg64) | 68.0 |

The GPU-routed token is now GPU-bound: the single command buffer measures
16.1–16.9 ms of GPU time per token, so the remaining gap to llama.cpp is kernel
efficiency (dense row kernels ~9.6 ms, experts ~4.4 ms, DeltaNet/attention
glue), not synchronization. Physical footprint at 22 GiB cache: 22.1 GiB.

## Scope boundary

- Validated on the M4 Max 48 GiB only. On 24 GiB machines the cache cannot
  hold the routed payload, the path never triggers and decode is unchanged.
- Speculation restarts a whole token on any miss; a per-layer restart with
  early-out would make speculation neutral on bounded caches but not faster,
  so it was not built.
- The preload is page-cache bound (3.7 s warm); a cold SSD read of 22 GiB
  takes correspondingly longer.
- Prefill is unchanged (its expert plans keep the residency table in sync).
