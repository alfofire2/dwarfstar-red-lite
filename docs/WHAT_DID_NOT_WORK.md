# What did not work

Everything that was tried on the native runtime and not kept, every trap that cost time, and
every target that was not reached, in one place, with the measurement that decided it. All on
the Apple M4 Max 48 GiB unless stated. Each entry links the milestone document with the details.

Method behind every "no gain": the change passed the parity gates, then an A/B against the
previous build (cooled runs, alternated in the same session) did not beat run-to-run noise or
was slower. Reverted changes leave no code behind; their environment switches were removed with
them.

## Performance attempts that were reverted

| milestone | attempt | measurement | why it failed (best understanding) |
|---|---|---|---|
| dev20 | experts read in place from the mmap (`RL_PREFILL_MAPPED_EXPERTS=1`, still opt-in) | numerically identical, 4–10× slower prefill | Metal re-established residency of each layer's ~350 MB expert window for every command buffer (~3.3 s per 48-layer chunk). Not re-tried since dev30's residency set ([DEV20](REDLITE_DEV20_BATCHED_PREFILL.md)) |
| dev20 | batched expert kernel layouts: 8-pair register tile / adaptive 4-2-1 tiles / 4-row tile / grid over pairs | 981 / 370 / 295 / 256 ms vs 312 baseline (96-token chunk) | padding waste and load imbalance between experts; kept: slices of 4 pairs ([DEV20](REDLITE_DEV20_BATCHED_PREFILL.md)) |
| dev22 | fewer Metal encoders alone (≈1 200 → 1 per token) | +1.9 % (57.6 → 58.7 tok/s) | encoder boundaries are cheap on this GPU; the time is inside kernels ([DEV22](REDLITE_DEV22_DECODE_KERNELS.md)) |
| dev23 | spin-waiting on `MTLCommandBuffer.status` instead of `waitUntilCompleted` | 4 GiB decode 44.6 → 27.5 tok/s | polling contends with the driver's completion path ([DEV23](REDLITE_DEV23_EARLY_OUT_PREFETCH.md)) |
| dev23 | per-layer early-out as the only 4 GiB decode path | 8.3 tok/s | 45–87 % of tokens miss in every layer; no always-hit layers to chain ([DEV23](REDLITE_DEV23_EARLY_OUT_PREFETCH.md)) |
| dev23 | periodic probing (32 synchronous tokens, then one GPU-routed try) | 39.5 tok/s vs 44+ | probes cost more than they gain; replaced by the load-free-token rule |
| dev26 | split-K decode attention at every length | short context 68.47 vs 70.20 tok/s | one block has the single-threadgroup kernel's parallelism plus a merge dispatch; split only above 256 positions ([DEV26](REDLITE_DEV26_LONG_CONTEXT.md)) |
| dev30 | router selection on the GPU (batched `rl_route` arithmetic) | prefill 1100: median 571 vs 567 tok/s (runs 571/482/614 vs 577/567/513) | the CPU step fell 70–115 → 45 ms, but the gain is inside run-to-run spread ([DEV30](REDLITE_DEV30_PREFILL.md)) |
| dev30 | matrix expert kernels: skip the empty 8-pair block of small slices | variable loop bound: experts 1 729 ms (accumulators spilled); constant bound + `break`: 901 ms vs 750–850 | the extra branch and register pressure cost more than the saved MACs ([DEV30](REDLITE_DEV30_PREFILL.md)) |
| dev30 | matrix expert kernels: split accumulators summed through threadgroup memory | experts 966 vs 777 ms | 26 KB of threadgroup memory per group cut occupancy; kept: summed in registers with an exact `x·I + y` ([DEV30](REDLITE_DEV30_PREFILL.md)) |
| dev33 | IQ codebooks staged in threadgroup memory, 256-thread GEMV groups (what llama.cpp does) | IQ2_S 35.7 vs 33.2 µs, IQ3_XXS 38.2 vs 37.9 µs | divergent `constant`-memory reads were not the bound once the GPU was warm ([DEV33](REDLITE_DEV33_IQ_KERNELS.md)) |
| dev33 | `uchar4` codebook loads in the IQ2_XXS dense GEMV | 25.6 / 27.5 vs 27.3 / 25.1 µs | not the bound; the IQ2_XXS GEMV stays at ~169 GB/s ([DEV33](REDLITE_DEV33_IQ_KERNELS.md)) |
| dev35 | grouped decode attention, 4 threads per position for the scores | attention 175 vs 165 ms per 64 tokens at ~8400 positions | more threads per row did not add memory parallelism that mattered |
| dev35 | grouped decode attention with blocks of 256 / 64 / 32 positions | 164 / 146 / 180 ms vs 113 ms with 128 | 256: too few threadgroups (66); 64 and 32: more merge work and shorter loops ([DEV35](REDLITE_DEV35_LONG_DECODE.md)) |

## Correctness traps (found, fixed or guarded)

- **Single accumulator in the matrix expert kernel (dev30).** Rounding along the 2048-column
  reduction flipped an argmax at position 1135 of the long-context check (two logits 0.012
  apart, after the documented router near-tie at 1035). Fixed with four partial accumulators;
  the gate caught it before commit.
- **A stale dump passed a parity check (dev24).** `quick_parity.sh` compared an old file after
  a crash; the script now deletes its outputs first. In dev26 `compare_dumps.py` was found to
  accept an empty native dump; it now needs equal, non-zero token counts.
- **Silent truncation (dev26).** `redlite-engine --tokens` cut lists at 4 096 ids; now 65 536 and
  longer lists are rejected.
- **The 8192-position logit bound (dev30).** The 4.0 max-logit bound set in dev26 from a
  4096-position measurement is below the native self-consistency floor at 8192: the 0.3.0
  binary's own token-by-token path differs by 4.29 from llama.cpp and 4.45 from its batched
  prefill there. The bound is 5.0 at ≥ 8192 positions; argmax and KL gates did not change.
- **The `logits` CLI default chunk (dev30).** Without `--batch` it ran token by token, so a
  "default chunk" parity run silently tested another path. It now uses the engine's chunk.
- **Benchmark artifacts.** GPU clock ramp: the first kernel-bench numbers were 2–3× off until a
  warm-up pass was added (dev33). Thermal state: the same binary measured 64.7–71 tok/s at
  different hours of one night, and back-to-back prefill runs throttled to a third of the speed
  (dev24): only cooled, alternated, same-session A/B numbers are compared.
- **Page cache state.** 4 GiB-cache numbers depend on whether the GGUF is still cached: after
  runs that allocate 22–29 GiB of expert slots, the first 4 GiB run read from SSD (IQ3_XXS 8192
  tokens: 169 vs ~600 tok/s). `bench_m4.sh` warms up before 4 GiB prefill runs.
- **zsh word splitting** broke two benchmark loops (`set -- $var` does not split in zsh); the
  measurement scripts run under `bash`.

## Not reached, blocked or not attempted

- **GitHub CI green (dev28): not reached.** GitHub refuses to start hosted jobs on this account
  ("recent account payments have failed or your spending limit needs to be increased"); the
  same steps pass in a Linux container. Needs the account's billing fixed.
- **IQ3_XXS prefill vs llama.cpp:** 837 vs 861 tok/s at 1100 tokens with full residency (dev33).
- **4 GiB-cache prefill on a 24 GiB Mac: not measured.** The dev34 gain (3.3× fewer expert bytes
  loaded) is inferred from load counts, not measured on an M4 Pro.
- **Dense stage tools on the IQ3_XXS GGUF.** The dev11–dev17 per-stage parity tools only accept
  the IQ2_XXS dense layout; `regress_m4.sh` reports 13 SKIPs on the IQ3_XXS file.
- **Larger GGUFs (IQ3_XS, IQ3_M, IQ4_XS): not run.** They do not fit with every expert resident
  in the 37.44 GiB Metal working set of this Mac (dev31).
- **4096-token prefill chunks: not attempted.** They exceed the 32 768 pairs one expert plan
  accepts and add ~0.5 GB of scratch (dev34).
- **llama-perplexity in the oracle build tree.** Rebuilding it there failed (OpenSSL target)
  after relinking one oracle library from the same pinned source; it is built in its own tree
  (`.deps/llama.cpp/build-ppl`) by `scripts/dev/perplexity.sh` (dev31).
