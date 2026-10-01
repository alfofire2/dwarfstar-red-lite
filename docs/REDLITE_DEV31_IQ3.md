# Red Lite dev31 — a higher-quality GGUF: IQ3_XXS with full residency on 48 GiB

Status: **done**. Implemented, tested model-free (Linux container and macOS, also under
ASan/UBSan) and validated with both real GGUF files on the Apple M4 Max 48 GiB. Nothing
here was run on an M4 Pro or an M4 Air, and the IQ3_XXS file does not fit their memory
with full residency.

## The file

Bartowski `Qwen_Qwen3-Next-80B-A3B-Instruct-GGUF`, checked through the Hugging Face API
before downloading (the only download of the 0.4.0 cycle):

| file | bytes | fits full residency on 48 GiB |
|---|---:|---|
| `…-IQ2_XXS.gguf` (reference, unchanged, kept) | 19,298,972,128 | yes (21,312 MiB of experts) |
| **`…-IQ3_XXS.gguf`** (downloaded) | **31,726,709,216** | **yes (29,376 MiB of experts + 1.42 GiB dense)** |
| `…-IQ3_XS.gguf` / `…-IQ3_M.gguf` | 33,031,045,600 / 36,643,914,208 | not attempted: above the preferred IQ3_XXS and too close to Metal's 37.44 GiB working set |

`models/Qwen_Qwen3-Next-80B-A3B-Instruct-IQ3_XXS.gguf`, SHA-256
`14f4d0c84ada768f503efd0bb54be065340eb74328c7f61df2916adaee4f4c4b` = the file's LFS object
id on Hugging Face. 363 GiB were free before the download.

## Quant types the file actually needs

From `redlite-layer-audit MODEL --tensors` and a census of all 843 tensors:

| tensors | IQ2_XXS file | IQ3_XXS file |
|---|---|---|
| routed experts gate / up | IQ2_XS (11 layers), IQ1_M (37) | **IQ3_XXS** (48) |
| routed experts down | as gate/up | **IQ3_S** (24 layers) / IQ3_XXS (24) — *differs from gate/up* |
| `attn_qkv`, `attn_gate` (DeltaNet) | IQ2_XXS | **IQ3_XXS** |
| `attn_q` | IQ2_XXS | **IQ2_S** |
| `attn_k`, `attn_v` | Q4_K | Q8_0 |
| `attn_output` | Q4_K | Q6_K |
| `ssm_out` | Q4_K | Q8_0 |
| shared expert gate/up/down | Q6_K / IQ2_XXS | Q8_0 (24) / **IQ4_XS** (24) |
| `token_embd` / `output` | Q2_K / Q5_K | **IQ3_S** / Q5_K |

Routed-expert payload: 30,198,988,800 bytes (IQ2_XXS file: 18,157,142,016).

## What changed

- **CPU reference** (`redlite_native_iq3.[ch]`): IQ3_XXS, IQ3_S, IQ2_S and IQ4_XS decoded
  8 values at a time, following the pinned llama.cpp `dequantize_row_*` value for value;
  the codebooks (`iq3xxs_grid`, `iq3s_grid`, `iq2s_grid`, `kvalues_iq4nl`) are copied from
  its `ggml-common.h` by `scripts/dev/gen_iq3_tables.py` (see `NOTICE.md`). Hooked into the
  dense CPU reference, the token embedding and the routed-expert CPU oracle.
- **Per-matrix expert types.** The expert layout and layer info carry the down type; the
  Metal expert kernels receive a type word (gate/up in bits 0–7, down in bits 8–15) so the
  `uint32` ABI of `redmetal_topk` (shared with the Python bridge) did not change.
- **Metal kernels.** The same group decoder is generated into each Metal library with the
  codebooks as `constant` arrays: `rl_rows_iq3` (decode GEMV), `rl_dq_iq3` and
  `rl_rowsb_iq3` (prefill), and every routed-expert kernel (decode lanes, GPU-routed,
  dev20 batched, dev30 matrix kernels).
- **`--cache-mib full`** for `redlite-generate`, `redlite-server` and `redlite-engine`:
  every expert of the file, from its real expert payload
  (`rl_engine_full_residency_mib`: 48 × 512 slots of the largest gate+up+down triplet
  rounded to 4 KiB). `redlite-engine info` prints it: IQ2_XXS 21,312 MiB, IQ3_XXS
  29,376 MiB. The Python planner computes the same number from the file
  (`native_residency`) instead of the old 22,528 constant.
- **`redlite chat` without a model path** takes the best file in `models/`: IQ3_XXS when
  RAM ≥ 40 GiB and its full residency plus dense weights stay below 75 % of RAM (Metal's
  working set is 37.44 GiB of 48 on this Mac), else the IQ2_XXS reference.
  `redlite serve --native` sizes the cache the same way for the file it is given.
- **Tools.** `redlite-engine dequant` / `redlite-ref-llama dequant` and
  `scripts/dev/dequant_check.sh` (bitwise check of the CPU dequantization against ggml's
  `to_float`), `scripts/dev/perplexity.sh` (pinned `llama-perplexity`, built in its own
  tree so the oracle libraries are not touched), the frozen corpus
  `tests/fixtures/perplexity_corpus.txt`.

## Validation on the IQ3_XXS file (M4 Max 48 GiB)

| check | result |
|---|---|
| CPU dequantization vs ggml `to_float` (IQ3_XXS, IQ3_S, IQ2_S, IQ4_XS, Q8_0, Q6_K, Q5_K; experts included) | bit-identical |
| engine parity, CPU oracle vs Metal, 6 tokens × 48 layers | YES, 0 router-id mismatches, worst logit 1.0e-5 |
| GPU-routed decode parity at full residency (`--repeat 2`) | YES, 12/12 tokens GPU-routed |
| logits vs llama.cpp (3 tokens) | max 1.0e-5, **KL 1.4e-12** (limit 1e-5) |
| greedy vs llama.cpp, 24 tokens, 2 GiB and full residency | identical |
| 1100 positions vs llama.cpp (4 GiB chunks 1100 / 512, full) | PASS, worst 0.57, KL 3.9e-4 |
| batched prefill parity (19 tokens / chunks of 8; 96 / 32) | YES, 0 router-id mismatches |
| kernel self-test (`rl_rows_iq3`, 4 types × 4 shapes × 2 variants) | worst relative error 6.5e-8 |

`rm -rf .deps/redmetal && scripts/regress_m4.sh MODEL`, 0 compiler warnings:

- IQ2_XXS: **49/49** (48 + `dequant.vs_llama`).
- IQ3_XXS: **36 passed, 0 failed, 13 skipped** — the 13 dense stage tools of dev11–dev17
  (see Scope boundary). Every engine-level check passes: model-free tests, the routed-expert
  stage tools, CPU-vs-Metal and GPU-routed parity, batched prefill parity, sanitized chat
  turn, server checks, tokenizer vs llama.cpp, dequant / logits / greedy / 1100-position and
  full-residency long-context parity vs llama.cpp.

## Quality and speed

`scripts/dev/perplexity.sh MODEL` — pinned `llama-perplexity` (7798007a2), frozen corpus
`tests/fixtures/perplexity_corpus.txt` (SHA-256 6948b2c3…), context 512, 111 chunks, the
same text and build for both files:

| file | perplexity |
|---|---:|
| IQ2_XXS | 16.4675 ± 0.30152 |
| **IQ3_XXS** | **14.2917 ± 0.26089** (−13.2 %) |

`scripts/dev/bench_m4.sh MODEL --reps 3 --cool 90 --cache-mib 29376 [--only prefill22|prefill8192|llama]`,
medians of three cooled runs (commit 28b4970 + the dev31 Python/regress commits; the native
kernels are those of dev30 plus the IQ3 decoders):

| IQ3_XXS | native | pinned llama.cpp, same ids | IQ2_XXS native (dev30) |
|---|---:|---:|---:|
| decode, full residency (29,376 MiB) | 63.39 tok/s | 68.54 tok/s | 71.15 |
| decode, 4 GiB cache | 43.77 (runs 36.9 / 43.8 / 46.9) | — | 47.77 |
| prefill 1100, full residency, 2048 chunk | 786.0 tok/s | 861.3 tok/s | 891.9 |
| prefill 8192, full residency, 2048 chunk | 860.3 tok/s | 900.7 tok/s | 926.9 |
| prefill 1100 / 8192, 4 GiB, 512 chunk | 428.4 / 588.3 | — | 545.8 / 645.6 |

IQ3_XXS weights are ~1.6× the IQ2_XXS bytes per expert, so decode (bandwidth-bound) is
~11 % slower than with IQ2_XXS. The native runtime is 7.5 % below llama.cpp on IQ3_XXS
decode and 9 % below at 1100-token prefill; dev31 added the quant types with parity, not
kernels tuned for them. The 4 GiB rows vary a lot run to run: with this 31.7 GB file the
page cache cannot keep the whole GGUF next to the other runs' 29 GiB of expert slots (the
first 8192 run read from SSD: 169 tok/s), so they are indicative only.

## Scope boundary

- Validated on the M4 Max 48 GiB only. The IQ3_XXS file with full residency needs about
  30 GiB of GPU-visible memory; it is not a configuration for the M4 Pro 24 GiB or the
  M4 Air 16 GiB (with a 4 GiB cache it runs, but was not measured there).
- The dense stage tools of dev11–dev17 (`redlite-ffn`, `redlite-deltanet-*`,
  `redlite-attention*`, `redlite-recurrent-block`, `redlite-decoder-stack`) still accept
  only the IQ2_XXS dense layout; `regress_m4.sh` reports them as SKIP on the IQ3_XXS file.
  The routed-expert stage tools (`topk-parity`, `router-parity`, `routed-parity`) pass on
  it, and the engine's CPU oracle checks every layer.
- The perplexity corpus is the project's own documentation (Markdown, technical English),
  frozen at a commit; it compares the two quantizations on the same text, it is not a
  standard benchmark.
- Only IQ3_XXS, IQ3_S, IQ2_S and IQ4_XS were added (the types these two files use);
  other Bartowski files use more types (Q3_K, IQ4_NL, …) and remain unsupported natively.
