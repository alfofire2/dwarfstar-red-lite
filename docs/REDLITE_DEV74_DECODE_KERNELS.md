# Red Lite dev74 — faster decode kernels, and a half KV cache that stays opt-in

Status: done. Measured on the Apple M4 Max 48 GiB and the Apple M4 Pro 24 GiB (AC), 2026-10-08.

## 1. Where a decode token's time went

dev72/73 profiled the prefill kernels; this milestone profiled a real decode token. `RL_GPU_CAPTURE=FILE
RL_GPU_CAPTURE_STEP=N` (on `redlite-generate`, with `MTL_CAPTURE_ENABLED=1`) records the N-th decode step. The trace is
19 GB because it holds the loaded model, but one token is short enough that Xcode profiles it in full mode.

GPU time of one CF2 token (every expert resident, M4 Max, 1,046 dispatches), before dev74:

| Kernel | Share | What it is |
|---|---:|---|
| `rl_rows2_q4k` | 34.5 % | dense Q4_K projections (DeltaNet qkv/z, ssm_out, attention, shared expert) |
| `redmetal_topk_gateup_lanes` | 17.5 % | routed experts, gate and up |
| `redmetal_topk_down_lanes` | 9.3 % | routed experts, down |
| `rl_route` | 7.1 % | router selection (the profiler inflates it: a rewrite gained nothing) |
| `dn_state_fused` | 5.6 % | DeltaNet state update |
| `rl_rows_q5k` | 5.4 % | output head (151,936 × 2048, Q5_K) |
| `attn_gqa` | 4.9 % | decode attention, short context |

CF2 and the Instruct F2 file read 864 MiB of Q4_K dense weights per token, against about 0.36 GiB of routed experts.
The dense projections, not the experts, were the largest single cost.

## 2. Four kernel changes and two scheduling changes

Each was measured alone, alternated against the build before it (decode-bench, CF2, every expert resident, M4 Max).

| Change | Kernel | Decode |
|---|---|---:|
| **Q4_K GEMV: one item per pair of sub-blocks.** The two sub-blocks that share 32 bytes of quants (low and high nibbles) are one item; quants are read as 32-bit words (four per load) instead of bytes. | `kernel-bench`: qkv 35 → 29.5 µs, z 18 → 14.5, ssm_out 19 → 15 | 78.9 → 82.2 tok/s, 5 of 5 pairs |
| **IQ1_M expert blocks decoded in pairs of groups.** Groups 2i and 2i+1 share the `qh` byte and the 3-bit scale: read once, constant shifts, same arithmetic and order. | decode expert bench −3 % | 82.6 → 84.5, 5 of 6 pairs |
| **DeltaNet state: one SIMD group per state row** (4 values per lane, 4 rows per threadgroup, no threadgroup barrier), the arithmetic of the prefill's `dn_state_seq_sg` (dev30). Also for the 2-row verify. `RL_ENGINE_STATE_SG=0` restores `dn_state_fused`. | profile share 5.6 → 2.2 % | 84.5 → 87.3, 5 of 5 pairs |
| **Q5_K output head as sub-block pairs**, like Q4_K (and its 2-row verify kernel). | 622 → 535 µs (−14 %) | about 1 % |
| **Shared expert beside the routed experts.** SiLU runs with the router selection and the shared down projection with the routed gate/up, instead of two dependent stages before them. `RL_ENGINE_SH_OVERLAP=0` restores the old order. | scheduling only | 94.7 → 96.9, 5 of 5 pairs |
| **DeltaNet convolution window shifted in place**, without a copy back at the end of every DeltaNet layer. `RL_ENGINE_SHIFT_INPLACE=0` restores the copy. | one dispatch and barrier fewer per layer | 97.0 → 97.6, 4 of 5 pairs |

The IQ1_M change keeps every sum in its order, and the two scheduling changes run the same kernels: greedy output is
identical token for token with and without them. The Q4_K, Q5_K and DeltaNet kernels sum in a different order, so the
results differ at float-rounding level. The model-free self-test now covers Q5_K rows too (`kernel-selftest`), and the
real-model gates pass (section 7).

## 3. Results

**M4 Max 48 GiB, CF2, every expert resident** (`decode-bench`, medians of three alternated pairs on a machine with
nothing else running):

| Context | 0.9.1 | 0.9.2 | Gain |
|---:|---:|---:|---:|
| 64 | 85.3 tok/s | 97.8 tok/s | +15 % |
| 4K | 83.5 tok/s | 96.4 tok/s | +15 % |
| 32K | 66.7 tok/s | 74.1 tok/s | +11 % |

At short context this is now slightly faster than MLX's 3-bit Coder on the same Mac (94–96 tok/s, dev72), with a file
14.5 GiB smaller.

**M4 Max, Bartowski's IQ2_XXS reference file** (`bench_m4.sh`, 256 greedy tokens, median of three cooled runs, both
releases measured the same day): every expert resident 83.5 → 90.9 tok/s (+9 %), 4 GiB expert cache 53.0 → 56.0
(+6 %). This file has only 230 MiB of Q4_K dense weights, so it gains less. Prompt ingestion of 1100 tokens: 587.6
against 584.1 tok/s (unchanged).

**M4 Pro 24 GiB, CF2** (`decode-bench`, four alternated pairs; dev74 before the two scheduling changes):

| Setting | 0.9.1 | dev74 | Gain |
|---|---:|---:|---:|
| 4 GiB expert cache | 33.7 tok/s | 37.9 tok/s | +12 % |
| Every expert resident (GPU limit 21,741 MiB) | 48.2 tok/s | 57.1 tok/s | +18 % |

## 4. The half KV cache: built, faster, not the default

`RL_KV_F16=1` stores K and V as half, which is llama.cpp's default KV type. The default stays float.

- **How:** every attention kernel reads and writes `RL_KV`, a `#define` set when the Metal library is compiled. The
  prefill flash-attention loads half tiles into float simdgroup matrices element by element (exact). The CPU oracle
  rounds its K/V rows to half the same way (ties to even; identical to `_Float16` on 20 M random floats). The model
  tag of state files includes the KV element size, so a state file of the other type is not read.
- **Measured** (M4 Max, CF2, same binary, `RL_KV_F16` 0 against 1):

  | Context | Float KV | Half KV | Gain |
  |---:|---:|---:|---:|
  | 32K | 72.7 tok/s | 79.5 tok/s | +9 % |
  | 64K | 60.0 tok/s | 68.1 tok/s | +13 % |

  Prefill attention of 512 tokens at 32K: 33.1 → 30.4 ms. The KV cache takes half the memory: on the M4 Pro 24 GiB at
  the 21,741 MiB GPU limit, every expert resident plus MTP plus a 32K context ran out of GPU memory with float (as in
  dev55) and ran with half (52.9 tok/s).
- **Why it is not the default:** with half K/V, the 1200-token comparison with llama.cpp on CF2 failed its bounds
  (worst KL 5.8e-2 against a limit of 2e-2, one argmax different at positions 1100–1199). With float K/V and the same
  dev74 kernels the same check gives KL 2.8e-7, as 0.9.1 does. The reference IQ2_XXS file passed all 59 regression
  checks with half K/V; CF2 sits nearer to router ties. dev53 reached the same conclusion on another file.
- **Where it pays:** long contexts and memory on 24 GiB Macs, for users who accept the drift. It is a documented
  opt-in; the planner keeps sizing for float (48 KiB per position), which overestimates when half is on.

## 5. What limits decode now

- **Dense GEMV:** `kernel-bench` gives 453 GB/s for a 65,536 × 2,048 Q4_K matrix and 463 GB/s for the output head. The
  kernels are near the memory limit; a 9.4 MB projection takes ~25 µs, with ~5 µs of start-up.
- **Routed experts:** about 160 GB/s of 1.75-bit weights, bound by the integer work of decoding them, plus about 15 µs
  of fill and drain per call (gate/up, down, sum). `decode-expert-bench` with `RL_BENCH_K=40`: 12 calls of 40 experts
  take 2.05 ms where 48 calls of 10 take 2.6.
- **Dependent dispatches:** about 1.7 µs each (8 extra trivial stages per DeltaNet layer cost 0.48 ms per token).
- Thirteen attempts that did not pay are in [WHAT_DID_NOT_WORK](WHAT_DID_NOT_WORK.md) with their measurements: a
  barrier-free router, XOR signs and a half or `constant` grid for the expert decoders, four rows per SIMD group for
  Q4_K, larger threadgroups, three prefill changes, wider half-KV loads in the decode attention.

## 6. Tools added

- `RL_GPU_CAPTURE_STEP=N`: capture one decode step (see section 1).
- `redlite-engine decode-expert-bench MODEL --repeat L+1`: the decode expert kernels of layer L on 480 distinct experts
  copied from the file, no model on the GPU; `RL_BENCH_K` sets the experts per call.
- `redlite-engine prefill-gemm-bench [ROWS [K [TOKENS [KERNEL]]]]`: the dense prefill GEMM on random data
  (`rl_gemm_tg`: 10.4 TFLOPS at 8192 × 2048 × 512).
- `kernel-bench`: Q4_K qkv and z shapes, a 65,536-row Q4_K matrix.
- `agent_eval.py --repo-ref TAG`: the repository tasks copy a fixed tag, so later commits do not change them.
- A profiling trap: every Xcode replay leaves a `GPUToolsReplayService` process holding about 20 GB, even after
  Xcode quits. Five of them filled 61 GB of swap and made benchmarks erratic. Kill them after profiling.

## 7. Validation

On the M4 Max, with the final build (float KV):
- `regress_m4.sh` on Bartowski's IQ2_XXS: 59 pass, 0 fail, 0 skip;
- `regress_m4.sh` on CF2: all checks pass except the 13 stage tools that only read the reference file's layout,
  which skip by design;
- `quick_parity.sh --long` on both files: 5 / 5, greedy identical to llama.cpp;
- `verify-check` with a bounded cache and at full residency;
- `scripts/dev/local_ci.sh`.

The regression caught one bug on the way: the in-place convolution shift, first applied to every caller, broke the
paired-slot path (WHAT_DID_NOT_WORK).

## Scope boundary

- Measured on one M4 Max and one M4 Pro, mostly with CF2. F2 has CF2's dense layout and Bartowski's IQ2_XXS (measured
  in section 3) its IQ1_M experts; the other files were not measured.
- Prompt ingestion is unchanged: three prefill attempts did not pay (WHAT_DID_NOT_WORK).
- The half KV cache is opt-in and validated only as far as section 4 says.
