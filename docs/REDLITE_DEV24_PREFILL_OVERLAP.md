# Red Lite dev24 — prefill: pre-gated expert prefetch and parallel routing

Status: **target reached** on the M4 Max 48 GiB (macOS 27). Prefill of the first 1100
tokens of the long-context fixture with a 4 GiB expert cache, in the default 512-token
chunks, went from 261.7 to **314.3 tok/s**, against a target of ≥ 300. Parity with
llama.cpp on the 1100-token prompt holds at chunks of 96, 256, 512 and 1100. Nothing here
is measured on the M4 Pro.

## Why

The dev23 profile of this prefill (chunk 512, 144 expert plans, wall ms) was:

| part | ms |
|---|---:|
| dense GPU (norms, DeltaNet/attention, router and shared-expert GEMMs) | 1452 |
| router selection on the CPU (softmax top-10 of 1100 × 48 tokens) | 167 |
| expert phase wall | 2749 |
| … of which expert loads into the pool (critical path) | 841 |
| … of which expert GPU | 1720 |

The GPU sat idle while the CPU routed and loaded experts, layer after layer. With a 4 GiB
cache a layer's experts do not survive until the next chunk (the whole union of a
512-token chunk is ~350 MB per layer, the cache holds about 11 layers), so every plan
loads again.

## What changed

1. **Pre-gated prefetch in the batched prefill** (`prefill_prefetch`).
   - Layer *l*'s dense command buffer also applies layer *l+1*'s router to layer *l*'s FFN
     input (one extra F32 GEMM, the dev23 predictor applied to the whole chunk).
   - While layer *l*'s experts run on the GPU, a background thread selects the predicted
     top-10 of every token, takes their union and loads it through the normal LRU
     (prepare, then immediate release).
   - The thread keeps going through layer *l+1*'s dense pass and is joined just before
     layer *l+1* routes. Layer *l*'s plan stays pinned until the join, so the prefetch
     can never evict experts still in use.
   - It is skipped when the cache cannot hold the in-flight plan and the predicted union
     together. `RL_ENGINE_PREFETCH=0` disables it (it also controls the dev23 decode
     prefetch).
2. **Parallel router selection** (`route_tokens`). The chunk's softmax top-10 selections
   are spread over the CPU cores with `dispatch_apply`, in blocks of 32 tokens. Each
   selection is the unchanged single-token routine, so ids and weights are bit-identical.
   Router selection went from ~180 to ~39 ms.
3. **Stats.** `redlite-engine logits` prints a `prefill split:` line with the wall split
   (dense, router, expert LRU/load/commit/GPU/wait and prefetch). The pool counters are
   read once per chunk, after every prefetch has joined; they include the prefetch's
   own loads.

The expert and dense kernels are unchanged.

## Validation (M4 Max 48 GiB)

Gate for each change: `scripts/dev/quick_parity.sh MODEL --long --batch N`.

| check | result |
|---|---|
| `engine.parity`, `engine.parity.gpu_routed` (12 GPU-routed tokens), `logits.vs_llama` | YES |
| 24 greedy tokens at 22 GiB | identical to the pinned llama.cpp |
| `logits.long_context_vs_llama` (1100-token prefill + 100 steps), chunks 96 / 256 / 512 / 1100 | YES at every size |

Full suite after `rm -rf .deps/redmetal && make native` (0 warnings):
`scripts/regress_m4.sh` **44/44 PASS**, including `logits.long_context_vs_llama` and `generate.vs_llama`.

Two defects were found and fixed on the way.

- **Stale-dump PASS.** The first prefetch build aborted (a command
buffer committed twice), and `quick_parity.sh` still printed PASS, because it compared
the dump left by the previous run. The script now deletes its outputs before running.
- **Race on the predicted logits.** The prefetch thread read the predicted logits while
  layer *l+1*'s dense pass wrote layer *l+2*'s prediction into the same buffer. Outputs
  were not affected, only which experts got prefetched. There are now two buffers,
  selected by layer parity.

## Throughput

Command (medians of three runs, 90 s cool-down before each run):
`scripts/dev/bench_m4.sh models/Qwen_Qwen3-Next-80B-A3B-Instruct-IQ2_XXS.gguf --only prefill --reps 3 --cool 90`.
The alternated A/B used the same prefill command directly.

| build | prefill 1100 tok, 4 GiB, chunk 512 |
|---|---:|
| dev23 (`65a0624`), cooled | 261.7 (258.4 / 262.6 / 261.7) |
| dev24, `RL_ENGINE_PREFETCH=0` (parallel routing only) | 271.6 (271.6 / 271.9 / 271.0) |
| **dev24** | **314.3** (314.3 / 314.6 / 313.8) |
| dev24 chunk 1100 (single run during the gate, not cooled) | 326.6 |
| pinned llama.cpp fully resident (`llama-bench` pp48, dev19 record) | 338.6 |

An earlier alternated A/B (before the race fix below; same method) gave the same
picture: prefetch on 313.9 / 313.7 / 312.1, off 271.6 / 271.5 / 272.5.

Expert phase wall at chunk 512, prefetch on: ~1850 ms against 1711 ms of expert GPU. The
loads (~600 ms, now counting the prefetch) are almost entirely hidden.

### Throttling under sustained load

Back-to-back prefill runs on this machine slow down run after run, even though `pmset`
reports no thermal or performance warning. Five consecutive runs gave 311 → 187 → 168 →
147 → 133 tok/s. The GPU time itself grows: dense GPU went from 1480 to 4250 ms.
Therefore:

- The prefill numbers above are taken with a cool-down (`bench_m4.sh --cool`).
- The dev22/dev23 prefill figures (248.5 / 252.3) were measured right after the decode
  runs without a pause, and are not comparable to them. The dev23 build (a worktree of
  `dev23-green`) was re-measured with the cool-down: 261.7.

## Scope boundary

- **Machine.** Measured on the M4 Max 48 GiB only, with the GGUF warm in the page cache.
  Not measured on the M4 Pro 24 GiB. There, prefetch loads come partly from the SSD and
  compete with the rest of the page cache, so the overlap may be smaller;
  `RL_ENGINE_PREFETCH=0` exists for that.
- **Memory.** The predicted-logit buffers add `2 × 512 experts × 4 B` per chunk token
  (two buffers by layer parity: 2 MiB at chunk 512). The prefetch uses cache slots the in-flight plan does not need,
  and never grows the cache.
- **Not done.** Fused gate/up/down expert kernels, a GPU-side weighted sum folded into
  the down projection, and GPU router selection for the batch were not needed for the
  target and are not built. The expert GPU time (1.7 s at chunk 512) and the dense GPU
  time (1.5 s) are now the whole critical path. The next gains are in those kernels.
- **Numbers.** Throughput from a warm page cache and a cooled GPU. Elapsed times printed
  by the parity tools are not benchmarks.
