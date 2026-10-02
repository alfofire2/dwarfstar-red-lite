# Red Lite dev42 — Qwen3-Coder-Next

Status: **done** on branch `dev/coder-next` (from `main` at a9193b9). Apple M4 Max 48 GiB only.

## Why

`Qwen/Qwen3-Coder-Next` (Apache-2.0, 2026-02-03) is a coding model with the Qwen3-Next-80B-A3B architecture:
`Qwen3NextForCausalLM`, 48 layers, hidden 2048, 512 experts top-10, the same attention and DeltaNet shapes, and
the same vocabulary. The only configuration difference is `rope_theta` (5e6 instead of 1e7), and it has no MTP
tensors. SWE-bench Verified is 70.6 according to Qwen. Bartowski publishes it at exactly our file sizes. See
`docs/RESEARCH_2026_10.md`.

## Change

None in the engine: the rope base is read from `qwen3next.rope.freq_base`. `redlite-engine info` prints
`base=5000000` for both files. `redlite download` gains the variants `iq3_xxs` (alias `48gb`, the 0.4.0 file
that was missing from the catalog), `coder_iq2_xxs` (alias `coder`) and `coder_iq3_xxs` (alias `coder-48gb`).
Coder-Next is not picked automatically: it is a specialised model and must be passed by path.

## Validation (M4 Max 48 GiB)

| file | bytes | SHA-256 (= Hugging Face LFS id) | dequantization vs ggml | `regress_m4.sh` vs pinned llama.cpp |
|---|---:|---|---|---|
| `Qwen_Qwen3-Coder-Next-IQ2_XXS.gguf` | 19,298,975,424 | `992dcf62…483446` | bit-identical | **50 / 50** |
| `Qwen_Qwen3-Coder-Next-IQ3_XXS.gguf` | 31,726,712,512 | `0b0a3356…aaaf96` | bit-identical | **37 passed, 0 failed, 13 skipped** |

The suite compares every native result with the pinned llama.cpp on the same file. That includes logits,
greedy decode, 1200-token long context, full residency, server stream, restart from a stored state, and the
sanitizer chat turn. The 13 IQ3_XXS skips are the IQ2_XXS-layout stage tools, as for the Instruct IQ3_XXS
file. Full residency is 17,316 / 28,800 MiB, the same as the Instruct files.

## Scope boundary

- Correctness only. Speed was not measured separately: the shapes and quant types are those of the Instruct
  files, but no benchmark was run on these files.
- No quality evaluation of the model itself (SWE-bench or similar) was run here.
- Not measured on the M4 Pro 24 GiB.
