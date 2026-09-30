# Red Lite dev30 — batched prefill: tiled attention, matrix experts, faster dense pass

Status: **done; both targets reached** on the Apple M4 Max 48 GiB in the configuration this
machine runs by default (full residency, 22 GiB cache). Every number below was measured
there; nothing was run on an M4 Pro or an M4 Air. The 4 GiB-cache path (the 24 GiB
configuration) is faster too and did not regress, but on this machine it is still below
llama.cpp at 1100 tokens.

## Targets

| target | result | reached |
|---|---|---|
| prefill 1100 tokens ≥ pinned llama.cpp on the same ids (858.34 tok/s here) | **891.9 tok/s** at 22 GiB, default chunk (545.8 at 4 GiB, 512-token chunks) | yes (full residency) |
| prefill 8192 tokens ≥ 280 tok/s | **926.9 tok/s** at 22 GiB; 645.6 at 4 GiB | yes |
| decode at 22 GiB not below 0.3.0 (71.50 tok/s) | 71.15 tok/s (5 runs); alternating A/B with the 0.3.0 binary: 71.09 vs 70.47 | yes |

Method: `scripts/dev/bench_m4.sh MODEL --reps 3 --cool 90 --only prefill|prefill8192|prefill22`,
median of three runs, 90 s pause before each. `prefill` / `prefill8192`: 4 GiB cache,
512-token chunks (from dev30 on with one untimed warm-up run first). `prefill22`: 22 GiB cache
and the engine's default chunk. llama.cpp: `--only llama` (dev28), fully resident.

## Changes, one at a time (each kept only after the parity gates)

| # | change | prefill 1100 / 8192 tok/s (4 GiB, chunks of 512) | commit |
|---|---|---|---|
| — | 0.3.0 baseline | 307.3 / 212.6 | 48c96cd |
| 1 | tiled causal attention `attn_fa_b` | 310.1 / 309.1 | d5d7fd8 |
| 2 | experts on simdgroup matrices (`redmetal_topk_gateup_mm`, `_down_mm`) | 428.3 / 439.0 | f058a0f |
| 3 | DeltaNet recurrence, one simdgroup per state row (`dn_state_seq_sg`) | 490.2 / 534.0 | 65bce55 |
| 4 | threadgroup-staged dense GEMM `rl_gemm_tg` (bit-identical output) | 518.0 / 631.1 | 55e6cbe |
| 5 | one Metal residency set for every engine buffer | 565.6 / 605.4 | 7e96a3e |
| 6 | 2048-token chunks by default under full residency | 22 GiB: 881.9 / 911.9 | aed1ec5 |
| 7 | no expert prefetch under full residency (bit-identical) | 22 GiB: **891.9 / 926.9** | fca7d78 |

1. **Tiled attention.** One threadgroup per (query head, 8 tokens), key blocks of 32
   positions: S = QKᵀ and O = diag(α)·O + P·V on f32 simdgroup matrices, online softmax in the
   flash-attention order. The KV caches get 32 padding positions for the block overhang.
   Attention was 17.3 s of the 40.7 s of an 8192-token prefill in the baseline profile.
2. **Matrix experts.** A threadgroup computes 32 weight rows × up to 16 (expert, token) pairs
   of one expert: the 128 threads decode the rows with the validated `rm_group8` arithmetic
   and stage the activations in threadgroup memory; f32 multiply-accumulate. The first version
   used one accumulator per output along the 2048-column reduction and **flipped an argmax**
   of the long-context check (position 1135, logits 0.012 apart after the documented router
   near-tie at 1035): not kept. Four partial accumulators per output (one per 8-column
   sub-step) fixed it; summing them through threadgroup memory cost occupancy (experts
   966 ms), summing them in registers with an exact `x·I + y` did not (777 ms).
3. **DeltaNet recurrence.** `dn_state_seq` ran a 128-thread group per state row with three
   threadgroup barriers per token; `dn_state_seq_sg` gives each row to one simdgroup (four
   columns per lane, `simd_sum` only). DeltaNet conv/state/tail GPU time for 1100 tokens:
   389 → 96 ms.
4. **Staged GEMM.** 64 rows × 32 tokens per threadgroup, K steps of 32 staged in threadgroup
   memory, the same k order as `rl_gemm_wt`: the logits dump is byte-identical (`cmp`).
5. **Residency set.** The dense weights are no-copy windows of the GGUF mmap. Without a
   residency set every command buffer re-established their residency: the prefill's dense
   wall time was 1236 ms against 712 ms of GPU time for 1100 tokens (~3.8 ms per layer
   command buffer). All engine buffers (and the prefill scratch) now live in one
   `MTLResidencySet` attached to the queue (macOS 15+; `RL_ENGINE_RESIDENCY=0` for A/B).
   Output byte-identical.
6. **Chunk size.** With every expert resident there are no loads to bound, and larger chunks
   give each expert tile more pairs: 1100 tokens at 22 GiB, single runs, 663 tok/s in chunks
   of 512, 722 in chunks of 1100, 811 with a 2048 cap. `rl_engine_prefill_batch()` returns
   `--batch`, else 2048 when every expert was preloaded, else 512 (the bounded-cache path is
   unchanged). Decode is unaffected (alternating A/B at 22 GiB: 70.9 vs 70.8 tok/s).
7. **No prefetch under full residency.** The dev24 pre-gating (next-layer router GEMM +
   loader thread) has nothing to load there.

`RL_PREFILL_PROFILE=1` (development) commits at every stage boundary and prints the GPU time
per prefill stage. Every change has an environment switch back to the previous kernel:
`RL_PREFILL_ATTN_TILE=0`, `RL_PREFILL_EXPERT_MM=0`, `RL_PREFILL_STATE_SG=0`,
`RL_PREFILL_GEMM_TG=0`, `RL_ENGINE_RESIDENCY=0`, `--batch 512`.

### Tried and reverted

- **Router selection on the GPU** (batched form of the dev21 `rl_route` arithmetic). Parity
  green, zero router-id mismatches. Alternating cooled A/B at 4 GiB: 571.3 / 481.7 / 614.0
  tok/s with it, 577.4 / 566.5 / 513.3 without (medians 571 vs 567). The CPU selection step
  fell from 70–115 to ~45 ms, but the gain is inside the run-to-run spread: reverted.
- **Skipping the empty 8-pair block** of the matrix expert kernels for slices of ≤ 8 pairs.
  With a variable loop bound the accumulators spilled (experts 1729 ms); with a constant
  bound and a uniform `break`, 901 ms against 750–850 without: reverted.

## Profiles (single runs, `RL_PREFILL_PROFILE=1` / `prefill split`)

1100 tokens, 4 GiB, chunks of 512: baseline dense GPU 1629 ms, experts GPU 1925 ms. After
change 4: recurrent input GEMMs 304 ms, DeltaNet core 104, `ssm_out` 117, attention q/k/v 75,
attention core 25, attention output 38, router 55, shared expert 77; experts GPU ≈ 780 ms.
At 22 GiB with one 2048 chunk the whole prefill is ≈ 600 ms dense + ≈ 600 ms experts.

## Parity

For every kept change: `scripts/dev/quick_parity.sh MODEL --long --batch N` (engine parity
CPU vs Metal, GPU-routed parity, logits vs llama.cpp, greedy vs llama.cpp, 1100-position long
context vs llama.cpp) at N = 512, 1100, 256, 96 (and 2048 for change 6), the batched-prefill
parity checks of `regress_m4.sh` (19 tokens in chunks of 8, 96 tokens in chunks of 32, zero
router-id mismatches) and `scripts/dev/long_positions.sh` at 4096 / 8192. The long-context
check stays at worst max-abs 0.99 / KL 3.5e-3 (argmax identical), as in 0.3.0.

**Bound at 8192 positions.** With 2048-token chunks at 22 GiB the 8192-position check gave
argmax 100/100 and KL 2.2e-6, but a worst max-abs logit of 4.48 against the 4.0 bound that
dev26 set from a 4096-position measurement. Measured on the **0.3.0** binary at 22 GiB: its
token-by-token path differs by 4.29 from llama.cpp and by 4.45 from its own batched prefill at
8192 positions (KL ~1e-6). The dev30 binary is byte-identical to 0.3.0 token by token, and its
512- and 2048-token chunks are byte-identical to each other at 22 GiB. The native
self-consistency floor at 8192 is therefore above 4.0, and `long_positions.sh` uses 5.0 at
P ≥ 8192 (argmax and KL gates unchanged). At 4 GiB with 512-token chunks the dev30 build gives
3.81 at 8192 (0.3.0: 2.27; 0.3.0 vs dev30: 3.87).

`regress_m4.sh` adds `logits.long_context_full_residency`: the 1100-position check at
22 GiB with the default chunk (≥ 40 GiB machines).

## Validation

- `rm -rf .deps/redmetal && scripts/regress_m4.sh MODEL`: **48/48** (47 + the new
  `logits.long_context_full_residency`), 0 compiler warnings.
- Decode, 5 cooled runs: 71.15 tok/s at 22 GiB (0.3.0 baseline 71.50, runs 68.6–72.4),
  47.77 at 4 GiB (0.3.0: 47.93). Alternating A/B at 22 GiB, 4 runs each, 90 s pauses: this
  build 71.09, the 0.3.0 binary 70.47 (medians). The decode kernels did not change (the
  token-by-token path is byte-identical to 0.3.0); the residency set applies to it too.

## Scope boundary

- Measured on the M4 Max 48 GiB only. The 4 GiB path uses the same kernels (changes 1–5);
  it was not measured on the M4 Pro 24 GiB or the M4 Air 16 GiB, where the page cache,
  not the GPU, may bound expert loads.
- At 4 GiB with 512-token chunks the 1100-token prefill (545.8 tok/s) is below llama.cpp's
  fully resident 858; the target is reached with full residency only.
- Kernels are f32 throughout (no half-precision staging), so the parity margins of 0.3.0
  are kept; the matrix kernels change the summation order, which the long-context and
  long-position checks above bound.
- The matrix expert kernels cover the IQ2_XS / IQ1_M experts of the reference GGUF; other
  expert types use the dev20d kernels until dev31 adds them.
