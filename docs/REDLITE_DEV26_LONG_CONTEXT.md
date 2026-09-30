# Red Lite dev26 (completion) — long positions, sanitized chat turn, kernel self-test, split-K attention

Status: **done** on the M4 Max 48 GiB (macOS 27). This completes the dev26 items left open
by `docs/REDLITE_DEV26_ROBUSTNESS.md`:

- parity and benchmarks at 4096 and 8192 positions;
- a sanitizer run of a real chat turn;
- model-free tests for the dev21–dev24 kernels.

The long-position benchmark showed decode collapsing with context, so this milestone
also adds a split-K decode attention. Nothing here is measured on the M4 Pro.

## Long positions: parity with llama.cpp

`scripts/dev/long_positions.sh MODEL`

- **Prompt.** The 1397 token ids of the frozen `tests/fixtures/long_context_prompt.txt`,
  repeated and cut to P+100 ids (no new fixture text).
- **Native.** Batched prefill of the first P ids (chunks of 512, 4 GiB cache), then 100
  decode steps.
- **Reference.** `redlite-ref-llama` on the same ids: pinned llama.cpp, F32 K/V, no flash
  attention.
- **Compared.** The logits of positions P..P+99.

| P | argmax agreement | worst KL (limit 2e-2) | worst logit abs (limit 4.0) | result |
|---:|---|---:|---:|---|
| 4096 | 100/100 | 6.4e-3 | 3.03 | YES |
| 8192 | 100/100 | 9.8e-7 | 2.27 | YES |

**Threshold.** The 1100-position check (`logits.long_context_vs_llama`) uses a max-logit
bound of 2.0. At 4096 that bound fails on 7 of 100 positions, while every argmax agrees
and the KL stays at most 6.4e-3. So before accepting a looser bound for long positions,
this was measured:

- Early and attention layers match as well as at 1100: layer 0 1.2e-7, layer 3 3.3e-6,
  layer 7 5.6e-5 (median max-abs). The drift starts in the late layers (layer 47 median
  0.44, against 0.12 at 1100).
- **Native against native, without llama.cpp.** Token-by-token decode vs the batched
  prefill over the same 4196 ids (the same kernels, a different summation order and
  chunking) differ by up to **2.62** in the logits, KL 8.0e-3. That is as much as either
  differs from llama.cpp (3.03 and 2.38).

So at this length, rounding-order differences alone move a few low-probability logits by
more than 2. The long-position check keeps argmax agreement and the KL limit, and uses a
max-logit bound of **4.0** (about 1.5× the measured native self-consistency floor). This
bound was chosen after the measurement above; it is not a pre-registered threshold.
`logits.long_context_vs_llama` in `regress_m4.sh` keeps 2.0.

## Long positions: throughput

`scripts/dev/long_positions.sh MODEL --bench-only`

- `redlite-generate --raw` on the fixture text repeated to at least P tokens, 128 greedy
  tokens.
- One run per cell, 90 s cool-down before each (see dev24 on throttling). These are
  single runs, not medians.

| position (prompt tokens) | cache | before split-K: prefill / decode tok/s | after split-K: prefill / decode tok/s |
|---|---|---:|---:|
| ~4096 (4193) | 4 GiB | 266.7 / 24.8 | 266.7 / **43.6** |
| ~4096 (4193) | 22 GiB | 267.0 / 31.5 | 265.8 / **65.5** |
| ~8192 (8387) | 4 GiB | 203.3 / 17.2 | 201.9 / **38.8** |
| ~8192 (8387) | 22 GiB | 204.1 / 20.1 | 204.3 / **57.4** |

After the final threshold change (split only above 256 positions), a repeat gave 42.3 / 63.8 /
39.2 / 56.3 tok/s decode in the same cells. In that repeat the first cell's prefill read
196 tok/s, an outlier of the single-run method.

**Short context** (`bench_m4.sh --only decode22|decode4 --reps 3 --cool 60`, medians; 256
tokens after a ~30-token prompt, so the first ~256 positions use the old kernel):

| build | decode 22 GiB | decode 4 GiB |
|---|---:|---:|
| split-K for every length (first version) | 68.47 | 45.84 |
| `RL_ENGINE_ATTN_SPLIT=0` (old kernel) | 70.20 | 46.83 |
| **dev26** (split-K above 256 positions) | 69.83 / 69.81 | 46.51 |
| dev23 build (`65a0624`), re-measured the same hour | 70.12 | 47.10 |

- At up to 256 positions the split has the same 16 threadgroups as the old kernel, plus the
  merge dispatch, and measured slower. So it is used only above 256 positions.
- Crossover, single cooled runs at 22 GiB:

  | positions | split-K | old kernel |
  |---:|---:|---:|
  | ~970 | 65.1 | 54.7 |
  | ~1880 | 65.1 | 45.3 |

- The dev22/dev23 records (72.7 / 50.1) were taken earlier the same night. Re-measured
  here, the dev23 build gives 70.1 / 47.1, so the lower short-context numbers are the
  machine's state, not a regression.

## Split-K decode attention (`attn_gqa_split` + `attn_gqa_merge`)

**Before.** The dev18 decode attention ran one threadgroup per query head: 16
threadgroups on a 40-core GPU. Each thread dotted a whole 256-float key row serially with
uncoalesced loads, then walked every position serially for the values. At ~8400
positions, `RL_ENGINE_PROFILE` gave attention 2269 ms of GPU time over 63 steps, 36 ms per
token, against ~12 ms for everything else in the token.

**Now:**

- `attn_gqa_split` runs one threadgroup per (query head, block of 256 positions).
  - Scores are simdgroup-cooperative dot products: lanes stride the head dimension, so key
    reads are coalesced.
  - The block writes its max, exp-sum and unnormalized value sum as a partial.
- `attn_gqa_merge` combines a head's partials with the same online-softmax algebra as the
  old kernel, then applies the sigmoid output gate.
- Same profile: **254 ms** (−89%).
- Used in both decode paths (synchronous and GPU-routed); the batched prefill keeps its
  own kernel.
- `RL_ENGINE_ATTN_SPLIT=0` restores the old kernel for A/B runs.

**Gate.** `quick_parity.sh` (engine parity, GPU-routed parity, logits vs llama.cpp, greedy
identical to llama.cpp, 1100-token long context) and `long_positions.sh` at 4096 and 8192:
all YES.

## Sanitized chat turn

`scripts/dev/sanitize_chat.sh MODEL`, new `regress_m4.sh` check `generate.sanitize`.

- **Build.** `redlite-generate` with `-fsanitize=address,undefined,float-cast-overflow
  -fno-sanitize-recover=all`.
- **Turn.** A 359-token chat turn in batched-prefill chunks of 64, which runs the dev24
  prefetch thread, then 32 tokens of decode through the synchronous path with the dev23
  prefetch. It is run twice: greedy, then a seeded top-k/top-p/min-p draw.
- **Result.** No sanitizer report, and the greedy text is identical to the normal build's.
- **Limits.** The Metal framework is not instrumented, and LeakSanitizer is unavailable on
  Apple platforms.

## Model-free kernel self-test

`redlite-engine kernel-selftest` runs from `build_engine.sh`, `make native` and
`make sanitize` (under ASan/UBSan on macOS), and is the new `regress_m4.sh` check
`selftest.engine_kernels`. It compiles the engine's own kernel library, reads no GGUF, and
checks:

| kernels | check |
|---|---|
| `rl_rows2_{f32,q4k,q6k,iq2xxs}` (dev22) and `rl_rows_*` (dev18) | 16 shapes (37 rows, 64–4096 columns, both lane regimes) against the double-precision CPU row dot; relative error ≤ 2e-5 of Σ\|w·x\| (worst 1.1e-7) |
| early-out guard (dev23) | with the flag set, `rl_rows2`, `rl_copy_f32` and `rl_route` leave their outputs untouched |
| `rl_copy_f32` | exact copy |
| `rl_route` (dev21) | 64 cases against `rl_native_router_select_softmax_topk`: ids, weights, slot addresses, exact ties, and a non-resident expert raising the flag |
| `attn_gqa_split` + `attn_gqa_merge` (dev26), `attn_gqa` (dev18) | 6 lengths (1, 200, 256, 257, 1024, 1500) against a double-precision GQA with the output gate; abs error ≤ 2e-6 |

**Mutation check.** A deliberately wrong Q6_K sub-block scale index in `rl_rows2_q6k` makes
the test fail (relative error 0.026), and restoring it passes again.

## Regression

After `rm -rf .deps/redmetal && make native` (0 warnings): `scripts/regress_m4.sh`
**46/46 PASS**. That includes the new `selftest.engine_kernels` and `generate.sanitize`
checks, `selftest.sanitize` (which now includes the sanitized kernel self-test) and every
llama.cpp oracle check. The self-test is also a step of the M4 field-validation workflow.

## Defects found on the way (fixed)

- **`compare_dumps.py` reported `ORACLE LOGITS PARITY: YES` on an empty native dump.** A
  crashed native run therefore passed every logits check that uses it. It now fails
  unless both dumps hold the same non-zero number of tokens. The existing regression
  dumps pass under the stricter check.
- **`redlite-engine --tokens` capped a list at 4096 ids and silently truncated longer
  lists.** The capacity is now 65 536, and a longer list is rejected.

## Scope boundary

- **Machine.** Measured on the M4 Max 48 GiB only. Not measured on the M4 Pro 24 GiB.
- **Positions.** Parity covers one synthetic long prompt (a repeated frozen fixture) at two
  positions, with 100 decode steps each. The 4.0 bound is justified by a native
  self-consistency measurement on the same prompt, not by a model of the error.
- **Not changed.** The batched-prefill attention (`attn_gqa_b`) is unchanged. It dominates
  prefill time at 8192: prefill falls from ~314 tok/s at 1100 tokens to ~205 at 8192.
- **Self-test.** It is synthetic. It checks each kernel's arithmetic against the CPU
  reference, not the model.
- **Not unit-tested.** The pool's per-slot load generations and the prefill prefetch thread
  have no model-free test. They run in every real-model check, and the sanitized chat turn
  exercises both.
