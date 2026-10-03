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

## dev50: decoding each weight once per 32 pairs — tried four ways, kept a pipelined decode

**What the decode costs.** Two variants of the 16-pair kernel, each removing one part of the decode (M4 Max, IQ2_XXS,
8192 tokens, full residency):

| variant | expert GPU time |
|---|---:|
| kernel as shipped | 3.86 s |
| decode arithmetic kept, every group read from the same block (no weight traffic) | 3.50 s |
| real weight bytes read, decode arithmetic replaced by a byte-to-float conversion | 2.77 s |

The decode **arithmetic** costs ~1.1 s; reading the weights costs ~0.36 s.

**32-pair tiles, which halve the decodes per pair: all slower.** Each was compared with the 16-pair kernel in the
same session (3.86–3.97 s):

| variant | expert GPU time |
|---|---:|
| 256 threads, pairs 16-31 on simdgroups 4-7, which skip their multiplications when empty (bit-identical) | 4.21 s |
| 128 threads, two partial accumulators instead of four (same register count as the 16-pair kernel) | 6.87 s |
| same, output tiles aliased onto the weight tiles (12 KB of threadgroup memory) | 8.01 s |
| same, pairs-16-31 branch unrolled by hand | 8.00 s |
| same, without skipping empty pairs 16-31 | 9.57 s |

On this GPU, 32-pair tiles lose whatever the accumulator layout, as dev40 also found (400 vs 872 tok/s). Doubling
the multiplications per decoded step makes each multiplication much slower. Spilling and occupancy were both
ruled out as the only cause (aliasing cut the threadgroup memory below the 16-pair kernel's and made it worse).
The bound is not understood.

**Kept: software pipelining.** Each thread loads and decodes step k+1's weights and activations into registers
while the simdgroups multiply step k. Same arithmetic, logits bit-identical.

| | dev49 | dev50 |
|---|---:|---:|
| M4 Max 48 GiB, full residency, experts GPU | 3.967 s | 3.86 s (−2.7 %) |
| M4 Max 48 GiB, 8192 tokens | 922 tok/s | 936 tok/s (+1.5 %) |
| M4 Pro 24 GiB, 4 GiB cache, experts GPU | 9.93 s | 9.49 s (−4.4 %) |
| M4 Pro 24 GiB, 8192 tokens | 353 tok/s | 360 tok/s (+1.9 %) |

Three alternated pairs on the M4 Max, two on the M4 Pro. `quick_parity.sh` passes on both GGUFs.

**dev49 + dev50 on the M4 Pro: 348 → 360 tok/s (+3.4 %)**, with output identical to 0.4.1.

## Scope boundary

- 500 tok/s on the M4 Pro was not reached, and the measurements say it cannot be reached by one kernel change.
- Decoding once per 32 pairs was built four ways and reverted; see the dev50 section.
- Only IQ2_XXS was measured. The IQ3 decode (`rl_iq3_group8f`) is unchanged.
