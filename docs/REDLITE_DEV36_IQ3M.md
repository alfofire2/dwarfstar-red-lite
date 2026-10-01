# Red Lite dev36 — the IQ3_M GGUF: supported, measured, not chosen

Status: **done**, IQ3_M validated against llama.cpp with bounded caches; on branch `dev/iq3m` (from `dev/long-decode` at 6eddd40). Apple M4 Max 48 GiB only.

## Why

The question after dev35 was whether a higher-quality file than IQ3_XXS is worth running. The GGUF
headers of four Bartowski files were read with HTTP range requests (24 MiB each, no download) and
parsed with the repository's GGUF reader:

| file | size | experts (gate / down) | full-residency cache | new types needed |
|---|---:|---|---:|---|
| IQ3_XS | 30.76 GiB | IQ3_XXS+IQ3_S / IQ3_S | 31680 MiB | none |
| IQ3_M | 34.13 GiB | IQ3_S / Q4_K+IQ3_S | 34944 MiB | Q4_K experts |
| Q3_K_M | 34.14 GiB | Q3_K / Q4_K+Q3_K | 34944 MiB | Q3_K |
| IQ4_XS | 39.91 GiB | IQ4_XS / IQ4_XS | 39168 MiB | none, but does not fit |

IQ3_M was downloaded (36 643 914 208 bytes, SHA-256 `77eee87b…eaf2cd`, equal to the Hugging Face
LFS id).

## Change

- **Q4_K routed experts.** IQ3_M stores half of its expert down projections as Q4_K. The IQ3
  module's 8-value decoder (`rl_iq3_group8`, the Metal `rl_iq3_group8f` / `rl_iq3_dot32`) gained a
  Q4_K branch; `rl_iq3_group8_supported()` admits it for experts only, so dense Q4_K keeps its
  dedicated kernels. The offline test checks the expert decoder bit for bit against the dense Q4_K
  dequantizer.
- **Planner.** `NATIVE_WORKING_SET_FRACTION` 0.75 → 0.70 (see below); IQ3_M is not added to the
  automatic model choice.

## Measurements (IQ3_M, M4 Max 48 GiB)

- `dequant_check.sh` (one tensor of every type, including the Q4_K experts): bit-identical to ggml.
- `redlite-engine parity` CPU vs Metal, 1 GiB cache: YES. GPU-routed parity at full residency
  (`--repeat 2`): YES, 12 / 12 GPU-routed tokens, worst logit 1.1e-5.
- **Full residency does not hold on this Mac.** `redlite-generate --cache-mib full`: 46.8 tok/s,
  then 6.0 tok/s on the next run, then `GPU-routed token command buffer failed: Insufficient Memory
  (kIOGPUCommandBufferCallbackErrorOutOfMemory)`. Footprint 35.2 GiB (cache + 1.6 GiB dense = 74 %
  of RAM). The planner's 0.75 rule admitted it, so the rule is now 0.70; IQ3_XXS (63 %) is unaffected.
- 28 GiB cache: 28.7 tok/s decode, 97 % expert hits. Synchronous decode at full residency
  (`RL_ENGINE_SPECULATIVE=0`): 24–26 tok/s. Single runs, not benchmarks.
- Perplexity (`scripts/dev/perplexity.sh`, frozen corpus, ctx 512, pinned llama.cpp):
  **14.05 ± 0.26**, against 14.29 for IQ3_XXS and 16.47 for IQ2_XXS. The gain over IQ3_XXS is inside
  the error bar.

## Comparison with the pinned llama.cpp (IQ3_M, bounded caches)

Run on the M4 Max with the `regress_m4.sh` checks and thresholds, except the full-residency ones:

- tokenizer corpus: identical;
- logits on `9707,11,1879` (1 GiB cache): `ORACLE LOGITS PARITY: YES`;
- 24 greedy chat tokens (4 GiB cache): identical to llama.cpp;
- 1200-token long context, positions 1100–1199 (4 GiB cache, `--max-logit-abs 2.0 --max-kl 2e-2`):
  `ORACLE LOGITS PARITY: YES`.

## Validation

`rm -rf .deps/redmetal`, `make redmetal`, `make native`, `make sanitize`: 0 warnings; `make test` and
ruff green; `scripts/regress_m4.sh` on the two existing files: IQ2_XXS **49/49**, IQ3_XXS **36 passed, 0 failed, 13 skipped**.

## Scope boundary

- **Validated with bounded caches only.** The llama.cpp comparison below ran with 1–4 GiB caches. Full
  residency of IQ3_M was checked only by CPU vs Metal parity, and does not hold on this Mac in real use.
- M4 Max 48 GiB only. Full residency of IQ3_M needs a larger Mac; nothing was run on one.
- IQ3_XS, Q3_K_M and IQ4_XS were not downloaded or run.
