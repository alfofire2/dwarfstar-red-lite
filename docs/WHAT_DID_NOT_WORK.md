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
| dev36 | IQ2_XXS dense GEMV with two rows per lane (one activation load serves both rows) | 25.4 vs 25.5 µs (8192 × 2048, warm kernel-bench) | third attempt on this kernel without gain; activation loads are not the bound. The IQ2_XXS dense GEMV stays at ~169 GB/s |
| dev37 | smarter expert replacement than LRU (simulated on routing traces, 4 GiB): decayed frequency (the Mira-style score), segmented LRU | best 60.9 vs 64.1 misses/token (−5 %); long half-lives and a 95 % protected segment are worse than LRU | decode routing is recency-dominated; the real waste was slot padding, fixed instead (dev37) |
| dev38 | two more concurrent-encoder groups ({ssm_out, conv-state copy}, {layer-output copy, next RMSNorm}) | 83.72 vs 83.77 tok/s | the removed barriers were not on the critical path |
| dev38 | (bound, not a change) concurrent decode encoder with no barriers at all | 74 → 118 tok/s, wrong output | most dispatches depend on the previous one; the kept version gains 2–6 % |
| dev39 | expert tail + next RMSNorm in one single-threadgroup kernel (−3 barriers per layer) | 79.1 vs 84.4 tok/s (5 pairs) | the 10-expert weighted sum on one GPU core costs more than the barriers; the parallel tail without the RMSNorm is kept (+1.6 %) |
| dev40 | prefill expert tiles of 32 (expert, token) pairs instead of 16 (each 32-row weight tile decoded once for twice the pairs), 128 threads with doubled per-thread accumulators | IQ3_XXS 1100 / 8192 tokens 400 / 404 vs 872 / 914 tok/s, bit-identical logits | register spills |
| dev40 | same with 256 threads (simdgroups 0-3 pairs 0-15, 4-7 pairs 16-31, per-thread work unchanged) | 1100: 886.5 / 866.2 / 878.0 vs 875.4 / 874.8 / 726.1; 8192: 921.6 / 906.8 / 909.2 vs 914.0 / 914.8 / 901.7 (medians 878.0 vs 874.8, 909.2 vs 914.0), bit-identical | re-decoding weights per 16-pair slice is not the prefill bound at these sizes |
| dev41 | decode expert kernels (gate/up, down) in threadgroups of 64 / 128 / 256 threads instead of 32 | 80.97 / 80.80-81.30 / 80.74-80.93 vs 81.30 / 81.51 tok/s (IQ2_XXS, full residency, two passes) | scheduling of the 5120-20480 one-row groups is not the bound |
| dev50 | 32-pair prefill expert tiles, 256 threads, simdgroups 4-7 skip empty pairs (bit-identical) | experts 4.21 vs 3.97 s (M4 Max, 8192 tokens) | the halved decode per pair did not pay for the larger threadgroup |
| dev50 | 32-pair tiles, 128 threads, two partial accumulators (register count of the 16-pair kernel); with outputs aliased on the weight tiles; with the branch unrolled; without the empty-pair skip | experts 6.87 / 8.01 / 8.00 / 9.57 vs 3.86 s | on this GPU 32-pair tiles lose with every accumulator layout tried (dev40 too); cause not understood |
| dev45 | relying on the compiler to share the weight decode between two dot calls (2-row verify) | 1.74x one vector instead of ~1.06x | not shared; explicit two-vector functions are used |
| dev45b | MTP draft through the trunk's Q5_K head instead of the MTP file's Q8_0 head | draft 1.56 → 1.51 ms, decode +0.3-0.8 % | inside noise; the head is not the draft's bound |
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
- **Ad-hoc parity token lists (dev39).** `redlite-engine parity --tokens 9707,11,1879,0,785,12884,13,151645,198
  --cache-mib 2048` fails at position 7 with a 2-id router mismatch and ~1e-5 logit errors, identically with the
  dev37 engine: a router near-tie on that sequence. Use the regression token lists.
- **`redlite-engine prefill` on long IQ3_XXS sequences (dev39)** reports `BATCHED PREFILL PARITY: NO` at 1100
  tokens (worst 0.027, 0 router mismatches, same argmax; dev37: 0.040). Its bound is sized for short sequences;
  the long-context check against llama.cpp is the gate.
- **Shared GPU (dev39).** With other applications using the GPU, the same binary measured 30–45 tok/s decode and
  335 tok/s prefill (instead of ~85 and ~800). Pairs run in such a window are excluded.
- **A stale binary in a release benchmark (0.4.1).** After reverting the dev40 kernels in the source, `.deps/redmetal`
  was not rebuilt, and the first 0.4.1 table was measured on the experimental build (IQ2_XXS prefill 854.3 / 858.3
  tok/s instead of 0.4.0's 891.9 / 926.9). The table was re-measured after `rm -rf .deps/redmetal` and a clean build.
  Rule: rebuild from a clean tree before any benchmark that goes into a record.
- **Battery power (dev47).** On battery at 6 % the same binary decoded at 32 tok/s instead of ~81 and a
  33.5K-token prompt took 17 minutes; every measurement of that window was discarded. `pmset -g batt` before
  benchmarks.
- **The token-by-token llama.cpp oracle at long positions (dev47).** `redlite-ref-llama logits` dumps every
  position one decode at a time: 16K positions took about 2 hours, 32K would take about 5 and 60K 9. The
  32K / 60K comparisons were not run.
- **The llama.cpp oracle on 24 GiB (dev47).** `redlite-ref-llama` puts the whole 18 GiB model on Metal and fails
  with `kIOGPUCommandBufferCallbackErrorOutOfMemory` on the M4 Pro. Correctness there is shown by bit-identity
  with the M4 Max's native outputs.
- **Command Line Tools with an SDK newer than their linker (dev47).** On the M4 Pro (CLT 26.6, ld-1267) the
  MacOSX27 SDK's `.tbd` files read as "unknown architecture arm64e.x1"; `REDLITE_SDK` points the builds at
  MacOSX26.5.
- **Chunk-parallel DeltaNet prefill (dev44): not built.** Measured ceiling: the recurrence is at most 6.7 % of an
  8387-token prefill, so the published 1.3–1.45× would give about 2 %.
- **Float16 KV cache (dev53): built, measured, not kept.** Every Metal attention kernel was switched to float16 storage
  (`attn_fa_b` with `simdgroup_half8x8` K/V tiles, mixed-precision multiply-accumulate), and the CPU oracle rounded K/V
  to float16 (identical to the hardware conversion on 4.3 M bit patterns). Model-free self-test, `quick_parity.sh`
  IQ2_XXS and the 4K/8K/16K comparisons with llama.cpp passed. The median KL vs llama.cpp was slightly better
  (16K: 6.9e-9 vs 9.0e-9); single-logit outliers moved (16K: 9.66 vs 6.21, at positions with KL ~1e-7).
  - **Gain:** half the KV memory (64K context: footprint 21,567 → 20,030 MiB on the M4 Max). On the M4 Pro 24 GiB
    with every expert resident, MTP + a 32K context fit only with float16 (20.4 GiB; float32 ran out of GPU memory).
    But MTP there decoded 32.6 tok/s against 37.2 without it, and float32 without MTP already fit (19.9 GiB).
  - **No speed gain:** M4 Max 33.5K-token prompt, decode 56.1 / 56.8 vs 54.5 / 57.4 tok/s; prefill 465* / 620 vs
    566 / 587 tok/s (*disturbed run).
  - **Gates failed:**
    - `engine.parity.steered`: 2 router id mismatches CPU vs Metal; logits 1.5e-4 vs 5.7e-6. The unsteered parity
      passed but its logits diff rose from ~1.5e-5 to 1.0e-4.
    - IQ3_XXS: `prefill.parity.96` (6 router mismatches), and `logits.long_context_vs_llama` /
      `long_context_full_residency` (argmax disagreement; KL 2.9e-3 within its bound).
  - **Why:** CPU (double) and Metal (float) compute nearly equal K/V values. Float16 storage turns a tiny difference
    that straddles a rounding boundary into one float16 step (~1e-3 relative). Next to the router near-ties of these
    fixtures, that flips expert choices. A CPU/Metal router divergence is a hard failure, and the gain is memory
    alone, so it was reverted.
- **The Qwen API as a greedy reference (dev48).** `temperature: 0` alone gave a different text in one run of
  three; `top_k: 1` is needed. Its logprobs are misaligned: the chosen token was missing from its own top 5 at
  23 of 30 positions. Only text is compared.
- **A first single run overstated a gain (dev49).** The vectorized expert decode measured −10 % expert time in one
  run against an earlier baseline run (4.40 → 3.94 s); alternated pairs gave −3.4 % (4.105 → 3.968 s). The
  baseline had been measured in a hotter state. Only alternated pairs are reported.
- **Removing the multiplications to bound a kernel (dev49).** With the simdgroup MACs deleted, the expert time
  fell to 266 ms because the compiler also dropped the now-unused weight decode. Only the "decode replaced by
  constants" variant (2.45 s of 4.40 s) is a valid bound.
- **Steering doses (dev52).** A difference-of-means vector at strength 1 over 16 layers, or 0.6+ over 8 layers,
  breaks the text into repeated tokens: the vector (|v| 4.35) is added at every steered layer, against a residual
  norm of about 10. Usable doses were 0.2–0.5 over 8–12 layers.
- **zsh word splitting** broke two benchmark loops (`set -- $var` does not split in zsh); the
  measurement scripts run under `bash`.

## Not reached, blocked or not attempted

- **GitHub CI green (dev28): not reached, then removed (2026-10-03).** GitHub refused to start hosted jobs on
  this account ("recent account payments have failed or your spending limit needs to be increased"). The
  project does not pay for hosted runners. The same steps run locally with `scripts/dev/local_ci.sh`.
- **IQ3_XXS prefill vs llama.cpp:** 837 vs 861 tok/s at 1100 tokens with full residency (dev33). Re-measured
  2026-10-01 at 547539f on an idle machine: native 859.4 (3 runs), 871.1 / 876.7 / 872.5 / 875.4 / 874.8 / 872.4
  (A/B baselines), llama.cpp 878.8 (3 runs) on the same ids: about 0.6 % behind, inside run-to-run spread.
- **4 GiB-cache prefill on a 24 GiB Mac: measured in dev47.** 280.8 tok/s (1100 tokens) and 349.8 tok/s
  (8192 tokens) on the M4 Pro 24 GiB. The dev34 gain alone (3.3× fewer expert bytes loaded) was not isolated
  there.
- **Dense stage tools on the IQ3_XXS GGUF.** The dev11–dev17 per-stage parity tools only accept
  the IQ2_XXS dense layout; `regress_m4.sh` reports 13 SKIPs on the IQ3_XXS file.
- **IQ3_M on this Mac (dev36): runs, but is not worth it.** The file is supported since dev36
  (its Q4_K expert down projections got a decoder). Measured on the 48 GiB M4 Max:
  - Full residency (34944 MiB) does not hold: GPU-routed decode ran at 46.8 tok/s, then 6.0 tok/s
    on the next run, then failed with `kIOGPUCommandBufferCallbackErrorOutOfMemory` (footprint
    35.2 GiB). The planner's 75 % rule admitted it; the rule is now 70 %.
  - A 28 GiB cache gives 28.7 tok/s (97 % hits), against ~70 tok/s for IQ3_XXS with full residency.
  - Perplexity on the frozen corpus is 14.05 ± 0.26 against 14.29 for IQ3_XXS: inside the error bar.
  - Correctness is not the problem: with bounded caches it matches llama.cpp (logits, 24 greedy tokens,
    1200-token long context). It is left out of the automatic model choice because on this Mac it is
    slower than IQ3_XXS for no measurable quality gain.
- **IQ3_XS, IQ4_XS: not run.** Header reads (no download) show IQ3_XS needs no new type and would fit
  at full residency only under the old 75 % rule; IQ4_XS (39168 MiB of experts) does not fit this Mac.
- **Speculative decoding (dev37): not started then, done in dev45** with the MTP block from a separate file
  ([DEV45](REDLITE_DEV45_MTP.md)). The 2026-09 reasoning, still valid for 24 GiB Macs: the IQ2_XXS / IQ3 GGUFs carry no Qwen3-Next
  multi-token-prediction layers (48 blocks, no `nextn` keys), so the MTP-based methods do not apply.
  Training-free drafts (prompt lookup, a reduced-expert self-draft) need a cheap multi-token
  verification, and the batched prefill path is slower than decode at small batches: 8-token chunks
  137 ms against 8 decode steps 98 ms (full residency, `redlite-engine prefill --batch 8`). Papers on
  MoE speculation (EcoSpec 2607.12696, MoE-Spec 2602.16052, DraftExpert 2607.24434) also report that
  verifying several tokens loads the union of their experts, which is the 24 GiB bottleneck.
- **LensVLM (dev37): not implemented.** `apple/lensvlm-9b` reads text rendered as compressed images
  (5–15× fewer tokens) and expands relevant pages on demand. It is a different model (Qwen3.5-9B +
  Qwen3-VL vision encoder, BF16 safetensors); Qwen3-Next-80B is text-only and cannot consume vision
  tokens, so the idea cannot be applied to this model without a second model family in the runtime.
- **OLED-MoE (2609.33385): not applicable.** Its inter-iteration expert retention targets diffusion LLMs.
- **Overlapping CPU encoding with GPU execution (dev38): not attempted.** At full residency the CPU
  gap between GPU-routed tokens is 0.44 ms of 12.1 ms.
- **4096-token prefill chunks: not attempted.** They exceed the 32 768 pairs one expert plan
  accepts and add ~0.5 GB of scratch (dev34).
- **llama-perplexity in the oracle build tree.** Rebuilding it there failed (OpenSSL target)
  after relinking one oracle library from the same pinned source; it is built in its own tree
  (`.deps/llama.cpp/build-ppl`) by `scripts/dev/perplexity.sh` (dev31).
