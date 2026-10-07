# Red Lite dev72 — prompt lookup on 24 GiB Macs, MLX, the llama.cpp drift, GPU profiling

Status: done. Measured on the Apple M4 Max 48 GiB and the Apple M4 Pro 24 GiB (AC), 2026-10-07/08.

## 1. The 2-row verify with a bounded expert cache

Until now the 2-row verify behind MTP and prompt lookup needed every expert resident: a 24 GiB Mac at the default GPU
limit, or any Mac with a 64K window, did not get either.

- **`verify2_bounded`** runs, per layer:
  - the dense part of both rows (`emit_verify2_dense`, split out of `emit_verify2` unchanged);
  - router selection of both rows on the CPU;
  - one `prepare` of the **union** of both rows' experts, so a row cannot evict the other's experts while they are in
    flight. `rl_native_metal_plan_subset` then gives each row a plan of its own experts in its own selection order,
    so each sum runs in the plain step's order;
  - each row's experts in its own command buffer, with the pre-gated prefetch of both rows' predicted experts as in
    `rl_metal_engine_step_sync`.
- **Exact routing only.** Cache-aware routing (`RL_ROUTE_CACHE_BIAS`, dev46) picks experts from the cache state, which a
  2-row pass changes in a different order than two steps do. `rl_engine_verify_available` says when the verify can run.
- **A bug found on the way.** The first version produced wrong text while every check of the plain path passed: the
  expert pool wrote its slot address and weight tables into shared buffers when it *encoded*, so encoding row 1 before
  the GPU ran row 0 gave row 0 the tables of row 1.
  - The fix passes the tables into the command buffer (`setBytes`, 240 bytes) instead.
  - Plain decoding is unchanged (same data, same kernels).
  - `redlite-engine verify-check` now compares a verify against plain steps (logits within 1e-3, same argmax). It is in
    `regress_m4.sh` with a bounded cache (`engine.verify_bounded`) and at full residency (`engine.verify_full`).

**Measured, CF2, greedy, identical answers:**

| | Plain, exact routing | Plain, cache-aware routing (the old default) | Prompt lookup |
|---|---:|---:|---:|
| File rewrite, M4 Max, 4 GiB cache | 49.5 tok/s | — | 65.2 tok/s |
| File rewrite, M4 Pro 24 GiB, 4 GiB cache | 28.7 tok/s | 31.9 tok/s | 37.8 tok/s |

**Coding agent, M4 Pro 24 GiB, 4 GiB cache, 64K window** (the hard suite, three runs each):

| Context of the request | Cache-aware routing (old default) | Prompt lookup | Drafts accepted |
|---|---:|---:|---:|
| 0–8K | 25.9 tok/s | 27.6 tok/s (+7 %) | 79 % |
| 8–16K | 23.9 tok/s | 25.7 tok/s (+8 %) | 84 % |
| 16–32K | 21.0 tok/s | 22.5 tok/s (+7 %) | 75 % |
| 32K and more | 19.5 tok/s | 21.4 tok/s (+10 %) | 82 % |

Tasks: 13 / 18 each, 130 minutes with lookup against 140. **`redlite serve` and `redlite chat` now use prompt lookup
with a bounded cache too**, with exact routing instead of cache-aware routing; a user's `RL_ROUTE_CACHE_BIAS` keeps the
old behaviour. MTP with a bounded cache uses the same verify but was not measured, so it stays off by default there.

## 2. Red Lite and MLX on the M4 Max

`mlx-lm` 0.32.0 with community MLX quantizations of Qwen3-Coder-Next. Same two prompts, greedy, 300 tokens.

| | Red Lite CF2 (18 GiB) | MLX 3-bit (32.5 GiB) | MLX 2-bit (23.2 GiB) |
|---|---:|---:|---:|
| Decode | 80 tok/s (109 rewriting a file, prompt lookup) | 94–96 tok/s | 107–110 tok/s |
| Prompt ingestion, 1.7K tokens, warm | 929 tok/s | 1,427 tok/s | 1,439 tok/s |
| Peak memory | ~21 GB | 36.8 GB | 26.9 GB |
| File rewrite faithful (first ~1,050 characters) | yes, 0 differences | yes, 0 differences | **no**: 4 places changed, the rename not done |
| Runs on a 24 GiB Mac | yes | no | no |

- **MLX decodes faster** than Red Lite plain decoding while reading more bytes per token. Its prompt ingestion is
  about 50 % faster. Red Lite's kernels have room, in decode and in prefill.
- **MLX's 2-bit file is fast but not usable for code:** it could not copy a file faithfully.
- **Red Lite's place:** the only option of the three on a 24 GiB Mac, with smaller files of the same fidelity, and
  prompt lookup.

## 3. The drift from llama.cpp past 64K is numerics, not a bug

Per-layer dumps of the last 4 positions at 32K and 64K, native against the pinned llama.cpp (CF2):
- **Error growth:** layers 0–5 differ at float-noise level (1e-7–1e-6). The difference grows slowly to about 3e-4 by
  layer 20, then jumps at a few layers (×10 at layer 21, a recurrent layer, and again at 36).
- **Origin:** the first attention layer differs by about 1e-5 at 32K. That is the size expected from summing tens of
  thousands of positions in float in a different order than llama.cpp (√N · ε).
- **Amplification:** the MoE router turns that into occasional discrete jumps when two experts are close to a tie.
- **No length dependence:** the early layers were further apart at 32K than at 64K, and the KL at these 8 positions
  was 1e-8–1e-6. The 0.025 at 128K (dev65) was the worst of 50 positions, one such flip.
- **RoPE:** ruled out in dev67.

## 4. GPU profiling with Xcode

`xctrace` from the command line records only one GPU counter on macOS 27 / Xcode 27, whatever template it is given. A
Metal GPU capture opened in Xcode gives everything: limiters, occupancy, registers per kernel. But a capture holds
every live buffer of the device.
- **Captures with the model loaded:** 19 GB (one prefill layer), 35 GB and 101 GB (bounded cache: every expert load
  is recorded). Xcode then profiles in "lite" mode, timeline only.
- **The tools added here run the kernels with no model buffer on the GPU:**
  - `redlite-engine prefill-attn-bench [TOKENS [POS]]`: `attn_fa_b2` on random data, a 56 MB trace;
  - `redlite-engine prefill-expert-bench MODEL --batch B --repeat L+1`: the routed-expert kernels of layer L on a copy
    of that layer's experts, from a CPU-only engine, a 569 MB trace;
  - `MTL_CAPTURE_ENABLED=1 RL_GPU_CAPTURE=file.gputrace` captures one run;
  - also `RL_GPU_CAPTURE_LAYER` / `RL_GPU_CAPTURE_SKIP` for one layer of a real prefill, `RL_PIPELINE_STATS`, and
    `prefill-kernel-bench` (with the engine).

**First profile: `attn_fa_b2`**, 512 tokens at position 7,680, M4 Max, 8.6 ms (median of 10):

| Counter | Value |
|---|---:|
| F32 limiter / F32 utilization | 75 % / 62 % |
| Instruction throughput limiter | 66 % |
| Kernel occupancy | 14 % |
| Registers per thread | 166, no spills |
| ALU instructions: float / integer | 74 % / 25 % |
| Device memory bandwidth | negligible |

- **What this means:** the kernel is bound by the float32 units, as dev66 inferred from FLOP counts.
- **Where the room is:** a quarter of the instructions are integer (addressing), and 166 registers keep occupancy at
  14 %.
- **One attempt:** limiting the unrolling of the score loop changed nothing (8.6 ms). Not kept.

**Second profile: the routed experts** (`prefill-expert-bench`, layer 0 of CF2 (IQ1_M), 2,048 tokens, 512 experts,
20,480 pairs, 22.4 ms):

| Kernel | Share of the time | Registers |
|---|---:|---:|
| `redmetal_topk_gateup_mm` | 43 % | 178 |
| `redmetal_topk_down_mm` | 30 % | 118 |
| `redmetal_topk_sum_b` | 2 % | 18 |

For `gateup_mm`:
- instruction throughput limiter 72 %, F32 limiter 70 % (utilization 55 %);
- occupancy 21 % (target 28 %);
- instructions: 65.5 % float, **34.5 % integer**, which is the IQ1_M / IQ2_XS bit decoding.

Two changes guided by this, alternated against the base (median of 10 runs each, three pairs). Neither was kept (see
WHAT_DID_NOT_WORK):
- **One accumulator per product** instead of four partial sums combined by identity products (fewer registers,
  three fewer matrix products per tile): 22.25 ms against 22.42 ms, about 1 %, within run-to-run spread.
- **The same plus 32-pair tiles**, which spilled in dev69 with four partial sums and fit now: 23.3 ms against
  22.35 ms, 4 % slower. Experts get about 40 pairs per 2,048-token chunk, and the partial second tile wastes its
  matrix work.

What the two profiles say: both hot prefill kernels are bound by instruction throughput on the float32 units, not by
memory, registers or tile shape. A third of the expert kernel's instructions decode the 1–2-bit weights. A real gain
needs fewer instructions per weight, for example a cheaper decode. That is a redesign, not a tuning pass.

## Scope boundary

- The bounded verify is measured with prompt lookup and Qwen3-Coder-Next only. MTP with a bounded cache is not
  measured and stays off by default.
- The MLX comparison is two prompts on one Mac with community quantizations. MLX's 4-bit and larger files do not fit a
  48 GiB Mac with the default GPU limit.
- Profiling covers one kernel so far. No kernel was changed by it yet.
