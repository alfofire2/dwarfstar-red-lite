# Red Lite dev33 — faster IQ kernels (after 0.4.0)

Status: **done** on branch `dev/iq3-kernels` (from `dev/0.4` at 74002cc, the 0.4.0 release).
Apple M4 Max 48 GiB only; nothing measured on an M4 Pro or an M4 Air.

## Why

0.4.0 added the IQ3_XXS GGUF with exact parity but generic kernels: decode 63.4 tok/s vs
68.5 for the pinned llama.cpp on the same file, prefill 786 vs 861 tok/s. A new development
tool, `redlite-engine kernel-bench`, times the decode GEMV of every weight type at the real
shapes (16 copies of each matrix, 256 dispatches, best of 5 **after a warm-up pass**: without
it the first figures were GPU clock-ramp artifacts, e.g. Q4_K at 61 µs instead of 17 µs).
On a warm GPU the IQ GEMVs ran at ~168 GB/s against 350–410 GB/s for Q8_0 / Q5_K:
decode-bound, not bandwidth-bound.

## Changes (each kept only after the parity gates and an A/B)

Every A/B below alternates the 0.4.0 binary and the new one, 4 runs each, 90 s pauses,
same session (the machine's thermal state moves the absolute numbers during a night: the
0.4.0 binary measured 64.7–71 tok/s at different hours, so only same-session pairs are
compared).

| # | change | effect | commit |
|---|---|---|---|
| 1 | `rl_iq3_dot32`: one 32-value sub-block of IQ3_XXS / IQ3_S / IQ2_S / IQ4_XS per call, scale and sign words decoded once, codebook entries unpacked as `uchar4` into `float4` dots; dense decode GEMV and decode expert kernels | kernel bench IQ3_XXS 38 → 28–31 µs, IQ2_S 33 → 25 µs, IQ4_XS −15..20 %. **IQ3_XXS decode (full residency) 64.7 → 70.1 tok/s**, above llama.cpp's 68.5 | `7ce8141` |
| 2 | `rl_iq3_group8f`: 8-value IQ3_XXS / IQ3_S decode with 16-bit loads and `uchar4` unpacking (same per-value product) for the prefill expert kernels and the dequant pass | **IQ3_XXS prefill 1100 tokens (full residency) 787.4 → 837.3 tok/s** (llama.cpp 861.3), every pair faster | `5f80901` |
| 3 | IQ2_XS / IQ1_M decode expert dots with `char4` codebook and `float4` activation loads, group scale applied to the group sum | **IQ2_XXS decode (full residency) 65.9 → 68.3 tok/s**, every pair faster | `3ec604f` |

A/B runs (tok/s):
- IQ3_XXS decode: 0.4.0 65.01 / 64.60 / 64.75 / 59.83, new 70.20 / 70.62 / 65.77 / 69.92.
- IQ3_XXS prefill 1100: 0.4.0 752.7 / 770.1 / 804.6 / 820.8, new 802.6 / 819.3 / 856.0 / 855.4.
- IQ2_XXS decode: 0.4.0 66.25 / 66.79 / 65.40 / 65.59, new 68.55 / 68.96 / 68.10 / 66.85.

### Tried and reverted

- **Codebooks staged in threadgroup memory** (as llama.cpp does), 256-thread GEMV groups:
  no gain once the benchmark was warm (IQ2_S 35.7 vs 33.2 µs, IQ3_XXS 38.2 vs 37.9 µs).
- **`uchar4` codebook loads in the IQ2_XXS dense GEMV**: 25.6 / 27.5 µs before vs
  27.3 / 25.1 µs after, no gain.

## Parity

For the IQ3_XXS file after each kept change: engine parity CPU vs Metal YES (0 router-id
mismatches), GPU-routed parity at full residency YES (12/12), logits vs llama.cpp YES, 24 greedy
tokens identical to llama.cpp, batched prefill parity YES, 1100-position context vs llama.cpp
unchanged (worst 0.566, KL 3.9e-4) at full residency and at 4 GiB. For the IQ2_XXS file:
`quick_parity.sh --long` (engine, GPU-routed, logits, greedy, long context), `topk-parity`,
`routed-parity`: YES. The kernel self-test's IQ errors fell (IQ3_XXS 3.1e-8 → 2.2e-8).

## Validation

`rm -rf .deps/redmetal`, `make redmetal`, `make native`, `make sanitize`: 0 compiler warnings;
`make test` and ruff green; `scripts/regress_m4.sh`: IQ2_XXS **49/49**, IQ3_XXS **36 passed,
0 failed, 13 skipped** (the IQ2_XXS-layout dense stage tools, as in dev31).

## Scope boundary

- M4 Max 48 GiB only. The kernels are shared with the 4 GiB path; that path's numbers were not
  re-measured here.
- IQ3_XXS prefill remains ~3 % below llama.cpp at 1100 tokens with full residency.
- The IQ2_XXS dense GEMV (~169 GB/s) and the IQ2_S / IQ4_XS GEMVs are still decode-bound;
  two attempts on them are reverted above.
