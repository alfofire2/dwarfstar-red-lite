# Red Lite dev75 — `--kv f16`, faster prompt ingestion, two more decode steps

Status: done. Measured on the Apple M4 Max 48 GiB only (the M4 Pro 24 GiB was not available), 2026-10-08, with the
display off. With the display on (and a remote-desktop viewer connected) the same binaries decoded about 9 % slower
at short context, so every comparison below is between runs in the same state.

## 1. `--kv f16` for `redlite chat` and `redlite serve --native`

dev74 built the half-precision KV cache behind `RL_KV_F16=1`. It is now an option of the launcher:

- `--kv f16` sets the variable for the native binary. The planner reads the same variable and counts 24 KiB of KV per
  position instead of 48 (`native_kv_mib_per_pos`); a second server slot's DeltaNet state is 3,072 positions instead
  of 1,536.
- The dev74 M4 Pro outcomes at the 21,741 MiB GPU limit are now tests: with half KV, MTP with a 32K context and
  512-token chunks fits, MTP with 16K and 2048-token chunks fits, and 32K with 2048-token chunks does not.
- The default stays float. The README explains the trade.

## 2. Prompt ingestion

| Change | Measure (CF2, M4 Max) |
|---|---|
| **IQ1_M experts decoded two groups at a time in the batched expert matrix kernels.** Each decoder thread decodes two consecutive groups of one row (gate, up or down), which share the block scale, the `qh` byte and the 3-bit scale, read once. A function-constant pipeline variant is used for IQ1_M layers only, so other types keep their code. Bit-identical logits. | `prefill-expert-bench`, IQ1_M layer: 20.53 → 18.88 ms (−8 %); IQ2_XS layer unchanged |
| **The dense GEMM through MetalPerformanceShaders** (`MPSMatrixMultiplication`, float32, one object per shape). Different summation order (float rounding). `RL_PREFILL_MPS=0` restores `rl_gemm_tg`. | 12.6–12.9 against 10.1–10.8 TFLOPS on the 8192 / 4096 × 2048 and 2048 × 4096 projections at 2048 tokens |

Prompt ingestion of CF2, every expert resident, alternated:

| Prompt | Before (0.9.2) | dev75 | Gain |
|---:|---:|---:|---:|
| 1.4K tokens (`redlite-generate`) | 945 tok/s | 1,026 tok/s | +8.6 % |
| 8,192 tokens (`redlite-engine logits`, 45 s cooling) | 940 tok/s | 1,026 tok/s | +9 % |

With the 4 GiB cache and 512-token chunks, 1,100 tokens: 567 → 571–602 tok/s (MPS alone).

**Where the prefill time is now** (1,090 tokens, `RL_PREFILL_PROFILE`): routed experts 540 ms, dense 506 ms, of which
DeltaNet projections 189, DeltaNet recurrence and tail 91, `ssm_out` 64, attention 83, shared expert 36. The experts
are bound by decoding their weights for every 16-pair slice; dequantizing whole experts once per chunk would write
about 6 GB per layer.

## 3. Decode

| Change | Decode, CF2, every expert resident |
|---|---|
| **Down projection and MoE tail in one dispatch** (`redmetal_topk_down_tail`): one SIMD group per output row, four experts per round on 8-lane groups with the down kernel's partition and reduction, then the weighted sum in selection order. Bit-identical greedy output. `RL_ENGINE_DOWN_TAIL=0` restores two dispatches. | 106.8 → 107.7 tok/s, 5 of 5 pairs |
| **Gate/up rows with at most 2 lanes per 256-value block** (16 lanes for 2048 columns instead of 32). Different summation order. | expert bench 2.38 → 2.27 ms; 107.4 → 108.1 tok/s, 5 of 5 pairs |

Decode against 0.9.2 (three alternated pairs): 107.1 → 108.1 tok/s at short context, 105.9 → 107.0 at 4K, 79.7 →
80.6 at 32K (about +1 %).

## 4. Not kept

- **IQ2_XS decoded two groups at a time** in the same pair pipeline: IQ2_XS layers −1 %, IQ1_M layers +6 % (the second
  decoder in the variant cost registers). A third, IQ2_XS-specific pipeline was not worth 1 %.
- **The pair remap applied to every type** (before the function constant): IQ2_XS layers +2–5 %.
- **8 gate/up lanes per row**: expert bench 2.50 ms against 2.38 (32) and 2.27 (16).

## 5. Validation

On the M4 Max with the final build: `regress_m4.sh` 59 / 0 / 0 on Bartowski's IQ2_XXS and 46 / 0 / 13 on CF2 (the 13
skips are the reference-layout stage tools); `quick_parity.sh --long` 5 / 5 on both files after each change; batched
prefill parity; `verify-check` with a bounded cache and at full residency; `scripts/dev/local_ci.sh` (the sanitize
builds link MetalPerformanceShaders too).

## Scope boundary

- M4 Max only. The 24 GiB M4 Pro was not used; the planner's half-KV numbers rest on the dev74 M4 Pro outcomes.
- MPS is Apple's library; its kernels are not inspected. It is deterministic run to run on this Mac.
