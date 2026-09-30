# Red Lite dev22 — decode kernels: one encoder per token, sub-block GEMV

Status: **target reached** on the M4 Max 48 GiB (macOS 27). Decode with full expert
residency went from 57.6 to **66.2 tok/s**, against a target of ≥ 62 and 68 tok/s for the
pinned llama.cpp fully resident. Parity with the CPU oracle and with llama.cpp is
unchanged or better. Nothing here is measured on the M4 Pro.

## Why

dev21 put a whole GPU-routed token into one command buffer. That path spent 16.6 ms of GPU
time per token (17.2 ms wall), while about 1.4 GB of weights are read per token. That is
~86 GB/s against ~546 GB/s available, so the decode was kernel-bound, not bandwidth-bound.

The LM head (Q5_K, 151 936 rows) already ran its row kernel at ~410 GB/s. The small
matrices ran the same kernel at 95–150 GB/s: 512–12 288 rows in the recurrent
projections, `ssm_out`, the router and the shared expert. In the dev18 row kernel one
lane dequantizes a whole 256-value block serially and uses at most one lane per block.
A 2048-column matrix therefore gets only 8 lanes per row, each doing 256 dependent
multiply-adds with scalar loads.

## What changed

1. **One compute encoder per token** (GPU-routed path) and **one per layer** (synchronous
   path).
   - The layer bodies are written once as emitters (`emit_recurrent`, `emit_attention`,
     `emit_ffn_pre`, `emit_scale_add`, `emit_rows`, `emit_rms`) that append dispatches
     to an open serial encoder.
   - The in-token blits (conv-state update, per-layer output copy) became an
     `rl_copy_f32` dispatch.
   - The routed experts are appended to the same encoder by
     `redmetal_topk_pool_encode_device_into`.
   - Under `RL_ENGINE_PROFILE` the emitter still closes the encoder and commits at every
     stage boundary, so the profile keeps its meaning.
   - The GPU-routed path went from about 25 encoders per layer (≈1 200 per token) to one
     per token.
2. **Sub-block decode GEMV** (`rl_rows2_q4k`, `rl_rows2_iq2xxs`, `rl_rows2_q6k`,
   `rl_rows2_f32`).
   - One lane handles one 32-value sub-block (Q4_K, IQ2_XXS) or 16-value sub-block
     (Q6_K), with up to 32 lanes per row, so a 2048-column row gets 32 lanes instead
     of 8.
   - Activations are read as `float4`. The F32 router reads `float4` weights.
   - The dequantization is the block kernels' own arithmetic, split per sub-block; only
     the summation order changes.
   - The old block kernels remain for the batched prefill, the Q5_K head (already
     bandwidth-bound) and Q8_0. `RL_ENGINE_ROWS2=0` switches decode back to them for
     A/B runs.

## Validation (M4 Max 48 GiB, macOS 27)

Gate for every change (`scripts/dev/quick_parity.sh`, the same commands as `regress_m4.sh`):

- `engine.parity` (sync path);
- `engine.parity.gpu_routed` (12 GPU-routed tokens, `router_mismatch=0`);
- `logits.vs_llama`;
- 24 greedy tokens with full residency identical to llama.cpp.

| change | gpu_routed worst layer abs | router mismatches | greedy vs llama.cpp |
|---|---:|---:|---|
| dev21 baseline | 5.34e-05 | 0 | identical |
| + one encoder | 5.34e-05 (bit-identical arithmetic) | 0 | identical |
| + sub-block GEMV | 1.53e-05 | 0 | identical |

Full suite after `rm -rf .deps/redmetal && make native` (0 warnings):
`scripts/regress_m4.sh` **44/44 PASS**. That includes `generate.vs_llama` (24 tokens identical), `logits.long_context_vs_llama` and `server.stream_greedy`.

## Throughput

Command: `scripts/dev/bench_m4.sh models/Qwen_Qwen3-Next-80B-A3B-Instruct-IQ2_XXS.gguf --reps 3`.
It reports median `decode_tok_s` for 256 greedy tokens (255 decode passes) of a fixed
prompt, and prefill of the first 1100 tokens of the long-context fixture in chunks of 512.

| commit | decode, 22 GiB cache | decode, 4 GiB cache | prefill 1100 tok, 4 GiB |
|---|---:|---:|---:|
| `3e2257e` (dev21 + PR #2) | 57.57 | 39.85 | 240.1 |
| one encoder per token | 58.65 | 40.17 | — |
| + sub-block GEMV (dev22) | **66.23** | **44.70** | 248.5 (unchanged path) |
| same build, `RL_ENGINE_ROWS2=0` | 59.87 | — | — |
| pinned llama.cpp fully resident (`llama-bench` tg64, dev19 record) | 68.0 | — | 338.6 (pp48) |

The 4 GiB decode is still the synchronous path, so it gains from the same kernels.
Its target (≥ 45) belongs to dev23.

### `RL_ENGINE_PROFILE=1` before / after

The profile disables the GPU-routed path and commits every stage separately. It
therefore measures the synchronous path with extra per-stage overhead, not the
end-to-end token. Setup: cumulative GPU ms, 256 greedy tokens, 22 GiB cache, `--batch 1`,
285 steps.

| stage | before (`3e2257e`) | after (dev22) |
|---|---:|---:|
| rms | 136.8 | 128.7 |
| dn_proj (IQ2_XXS qkv/gate, Q8_0 ba) | 682.6 | 580.6 |
| dn_prestate | 200.5 | 190.1 |
| dn_state | 237.6 | 216.2 |
| dn_tail (+ Q4_K ssm_out) | 359.5 | 341.9 |
| attn_proj | 207.7 | 172.2 |
| attn_norm_rope | 25.7 | 25.3 |
| attn_gqa | 242.3 | 244.4 |
| attn_out | 115.2 | 108.7 |
| resid_rms | 161.9 | 135.2 |
| router (F32) | 383.9 | 227.9 |
| shared (Q6_K) | 631.0 | 496.4 |
| experts | 1514.9 | 1533.8 |
| head (Q5_K) | 147.2 | 151.8 |

## What did not matter

- **Encoder count alone.** Going from ~1 200 encoders to one per token was worth only
  +1.9% (57.6 → 58.7 tok/s). Encoder boundaries are cheap on this GPU; the time was inside
  the kernels.

## Scope boundary

- **Machine.** Measured on the M4 Max 48 GiB only. The 22 GiB configuration needs
  ≥ 40 GiB of RAM and does not exist on the M4 Pro 24 GiB. Nothing is claimed there.
- **Scope of the change.** It covers only the decode GEMV of Q4_K, IQ2_XXS, Q6_K and F32
  matrices, plus encoder structure.
  - The routed-expert kernels (IQ2_XS/IQ1_M, ~5 ms of the token) are unchanged.
  - The DeltaNet state and attention kernels are unchanged.
  - The batched prefill is unchanged.
- **Numbers.** Throughput numbers are medians of three runs on an otherwise idle
  machine; single runs vary by ±1 tok/s. The profile numbers are not end-to-end timings.
