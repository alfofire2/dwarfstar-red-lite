# Red Lite dev49 — prompt ingestion on the 24 GiB Mac

Status: on branch `dev49/prefill-experts`. Measured on the Apple M4 Pro 24 GiB (16 GPU cores) and the Apple M4 Max
48 GiB (40 GPU cores) on 2026-10-03.

## The question

The M4 Pro ingests prompts at 280–350 tok/s with the 4 GiB expert cache (dev47). antirez calls 500–600 tok/s the
point where reading becomes "interesting" (`docs/RESEARCH_2026_10.md` section 6), and coding agents mostly read.
Where does the time go?

## The answer: GPU compute, not the disk

8192 prompt tokens on the M4 Pro, 4 GiB cache, 2048-token chunks: 23.5 s (348 tok/s).

| part | time |
|---|---:|
| dense layers, GPU | 10.9 s |
| routed experts, GPU | 10.3 s |
| expert loading (49 GB read, 69,071 loads) | 12.4 s, on the prefetch thread, overlapped with the GPU |

Dense stages (`RL_PREFILL_PROFILE=1`):

| stage | time |
|---|---:|
| DeltaNet projections | 3.6 s |
| attention | 2.1 s |
| DeltaNet conv/state/tail | 1.5 s |
| DeltaNet output | 1.25 s |
| attention q/k/v | 0.9 s |
| shared expert | 0.63 s |
| attention out | 0.42 s |
| router | 0.40 s |

- **Loading is hidden.** Each 2048-token chunk touches almost every expert of every layer, so the 4 GiB cache
  reloads about 17 GiB per chunk. The prefetch thread loads the next layer's experts while the GPU computes, so
  the loads are hidden behind the 21 s of GPU work.
- **The M4 Pro runs at the M4 Max's efficiency.** The M4 Pro has 16 GPU cores, the M4 Max 40. The M4 Max reads
  ~910 tok/s at full residency; 910 / 2.5 = 364 tok/s, and the M4 Pro does 348.
- **What 500 tok/s would take.** The kernels must do the same work about 1.45× faster. That is not a 24 GiB
  setting: the same kernels would also make the M4 Max ~1.45× faster, well ahead of the pinned llama.cpp it now
  matches.

## Where the expert kernel's time goes (M4 Max)

The prefill expert kernels (`redmetal_topk_gateup_mm` / `down_mm`, dev30) work on tiles of 32 weight rows × 16
(expert, token) pairs:
1. decode the 2-bit weights of each 32-column step into threadgroup memory;
2. multiply them with float simdgroup matrices.

With the weight decode replaced by constants (wrong output, measurement only), the expert GPU time fell from
4.40 s to 2.45 s. **About 44 % of the expert time is decoding weights.**

Each tile decodes its weights again for every group of 16 pairs. In a 2048-token chunk an expert serves about
40 pairs on average, so its weights are decoded about three times.

## Kept: a cheaper decode (bit-identical)

`rm_group8` read the codebook entry one byte at a time and chose each sign with a branch. Now it does two
`char4` loads and builds the sign vectors with `select`. The arithmetic is unchanged, so logits are bit-identical
(1000-token prefill + 100 positions).

| | base | dev49 |
|---|---:|---:|
| M4 Max 48 GiB, full residency, experts GPU | 4.105 s | 3.968 s (−3.4 %) |
| M4 Max 48 GiB, 8192 tokens | 912 tok/s | 925 tok/s (+1.4 %) |
| M4 Pro 24 GiB, 4 GiB cache, experts GPU | 10.30 s | 9.94 s (−3.4 %) |
| M4 Pro 24 GiB, 8192 tokens | 348 tok/s | 353 tok/s (+1.4 %) |

Three alternated pairs on the M4 Max, two on the M4 Pro, 45–60 s cooling. `quick_parity.sh` passes on IQ2_XXS and
IQ3_XXS.

## What would still help, with ceilings

- **Decode each expert's weights once per chunk instead of about three times.** One threadgroup would serve all
  of an expert's pairs from one decoded tile. The accumulators of three 16-pair tiles do not fit in registers:
  dev40's 32-pair tile spilled (400 vs 872 tok/s). So the partial sums would live in threadgroup memory, over
  K-blocks of 128 columns with half-precision weights (as llama.cpp does). This changes rounding, so it is not
  bit-identical.
  - Ceiling: about two thirds of the decode time, i.e. ~1 s of 9 s on the M4 Max (≈ +12 %) and ~2.7 s on the
    M4 Pro (≈ 400 tok/s).
- **The dense projections already run at about 60 % of the GPU's float32 peak** on the M4 Max (DeltaNet
  projections: 15.3 TFLOP in 1.55 s, about 9.9 TFLOPS). Little is left there.

## Scope boundary

- 500 tok/s on the M4 Pro was not reached, and the measurements say it cannot be reached by one kernel change.
- The decode-once rewrite is not built.
- Only IQ2_XXS was measured. The IQ3 decode (`rl_iq3_group8f`) is unchanged.
