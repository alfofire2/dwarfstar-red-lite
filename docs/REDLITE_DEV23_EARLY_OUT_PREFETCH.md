# Red Lite dev23 — per-layer early-out and pre-gated expert prefetch

Status: **target reached** on the M4 Max 48 GiB (macOS 27). Decode with a 4 GiB expert
cache rose from 44.7 (dev22) to **50.1 tok/s**, against a target of ≥ 45. Decode with full
residency rose from 66.2 to **72.7 tok/s**. Greedy output is identical to llama.cpp and
parity with the CPU oracle is unchanged.

**Implemented and validated in parity on the M4 Max with a limited (4 GiB) cache; not
measured on the M4 Pro 24 GiB.** The 4 GiB configuration is the one intended for
24 GiB machines, but every number here comes from the 48 GiB machine, whose page cache
holds the whole GGUF.

## Why

- **4 GiB cache.** The synchronous decode made one CPU/GPU round trip per layer: 48 per
  token.
  - Per token: 8.3 ms dense GPU, 6.2 ms expert GPU, 1.9 ms of miss loads, and about
    6 ms of pure bubble (command-buffer completion and submission latency ×48).
  - dev21's speculative token path never helps here. At a 85% per-expert hit rate
    almost every token needs at least one load, so dev21 threw the token away and redid
    it synchronously.
- **Full residency.** dev21 backed up every DeltaNet state (~72 MiB of blits) at the
  start of every token, only to restore it after a miss that cannot happen when every
  expert is resident.

## What changed

1. **Per-layer early-out (`step_routed`).**
   - Every decode kernel takes an early-out flag at buffer index 30 and returns at once
     when it is set. That covers the 19 engine kernels, plus function-constant variants
     of the three device-driven expert kernels, so the Python bridge's pipelines are
     unchanged.
   - The GPU router kernel `rl_route` raises the flag when layer *f* selects a
     non-resident expert.
   - Layers before *f* are then complete, and layer *f* ran up to its router exactly
     once, so nothing is restored. The CPU loads only layer *f*'s missing experts, using
     the ids and renormalized weights the GPU selected, runs them, and resumes the
     GPU-routed chain at *f+1* in a new command buffer.
   - The state backups and the whole-token redo of dev21 are gone.
2. **Policy.**
   - The early-out path is tried after a token that needed no expert load (as in dev21).
   - A token with more than 4 early-outs sends the next tokens to the synchronous path.
   - Reason: a resume re-encodes the rest of the token, and a skipped dispatch still
     costs about 3 µs at the command processor. At 4 GiB, with ~32 missing layers per
     token, always taking the early-out path ran at 8.3 tok/s.
3. **Pre-gated prefetch** in the synchronous path.
   - Layer *l*'s command buffer also applies layer *l+1*'s router to layer *l*'s FFN
     input: an F32 GEMV of ~17 µs.
   - While the GPU runs `experts_{l-1}` and `dense_l`, the CPU loads the predicted
     non-resident experts of layer *l* through the normal LRU (prepare, then immediate
     release).
   - When layer *l*'s real selection arrives, most of its misses are already resident.
   - `RL_ENGINE_PREFETCH=0` disables it.
4. **Per-slot load generations** in the expert pool.
   - The release check "no SSD reads while the experts were in flight" was global, and
     rejected the prefetch's loads into *other* slots.
   - Each plan now records its slots' load generations at encode and checks them at
     release.
   - The pool still refuses any load into an in-flight slot.

## Validation (M4 Max 48 GiB)

Gate for each change: `scripts/dev/quick_parity.sh`.

| check | result |
|---|---|
| `engine.parity` (1 GiB cache: almost every layer misses) | YES. 287 early-outs over 6 tokens, 0 router mismatches, worst layer abs ≤ 1.5e-05 |
| `engine.parity.gpu_routed` (22 GiB, `--repeat 2`) | YES. 12 GPU-routed tokens, 0 early-outs |
| `logits.vs_llama` | YES |
| 24 greedy tokens at 22 GiB, and at 4 GiB with prefetch | identical to the pinned llama.cpp |

Full suite after `rm -rf .deps/redmetal && make native` (0 warnings):
`scripts/regress_m4.sh` **44/44 PASS**. The label change made after it ("fallbacks" → "per-layer early-outs") was rechecked with `quick_parity.sh` and `generate.json`.

## Throughput

Command: `scripts/dev/bench_m4.sh models/Qwen_Qwen3-Next-80B-A3B-Instruct-IQ2_XXS.gguf --reps 3`
(medians, 256 greedy tokens).

| build | decode 22 GiB | decode 4 GiB | prefill 1100 tok (4 GiB) |
|---|---:|---:|---:|
| dev22 (`fdd3c69`) | 66.23 | 44.70 | 248.5 |
| + early-out, always taken | 69.87 | 8.32 | — |
| + policy (early-out only after a load-free token) | 69.37 | 45.05 | — |
| + prefetch, backups removed (dev23) | **72.72** | **50.11** | 252.3 |

- **Prefetch A/B**, same build and command, runs alternated (`RL_ENGINE_PREFETCH`): off
  45.31 / 44.93, on 47.62 / 47.07 tok/s.
- **4 GiB with `--stats`:** misses loaded on the critical path went from 1.9 to 0.6 ms per
  token. The reported hit rate rises from 85.4% to 91.0% because prefetch lookups are
  counted too. SSD/page-cache traffic rises from 50.3 to 61.2 MiB per token, because
  wrong predictions are loaded as well.
- **Physical footprint:** 4 GiB cache 4806 MiB; 22 GiB cache 22 032 MiB (no backups).

## What did not work (measured, reverted or superseded)

- **Spin-waiting on `MTLCommandBuffer.status`** instead of `waitUntilCompleted`. 4 GiB
  decode fell from 44.6 to 27.5 tok/s (A/B, alternated runs): polling the status contends
  with the driver's completion path. Reverted.
- **Early-out as the only 4 GiB path.** 8.3 tok/s, see *Policy* above.
- **Periodic probing** (32 synchronous tokens, then one GPU-routed attempt). 39.5 tok/s:
  the probes cost more than they gain. Replaced by the load-free-token rule.
- **Per-layer miss statistics at 4 GiB** (215 tokens). Every layer misses on 45–87% of
  tokens, on average 31.7 of 48 layers per token. A layer's miss state matches the
  previous token's only 60% of the time. There are no "always-hit" layers the early-out
  could chain, so round trips can only shrink once misses shrink: hence the prefetch.

## Scope boundary

- **Machine.** Measured on the M4 Max 48 GiB only, with the GGUF warm in the page cache.
  Not measured on the M4 Pro 24 GiB. There, a 4 GiB cache sits next to a page cache
  that cannot hold the whole model, so prefetch loads compete with real SSD reads. The
  gain may be smaller or reversed; `RL_ENGINE_PREFETCH=0` exists for that reason.
- **Prediction.** The prefetch predictor is the next layer's router on the current
  layer's FFN input. Its accuracy is not measured separately; only the end-to-end effect
  is.
- **Where each technique helps.** At 4 GiB the early-out path is almost never taken
  (≈ every token loads something). Its gain is at full residency, from removing the
  backups. A chunked early-out with a bounded lookahead window is not built.
- **Unchanged.** The prefill and the kernel arithmetic are unchanged.
