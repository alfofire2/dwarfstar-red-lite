# Red Lite dev58 — more precision for the dense weights, at the same size

Status: done; F2 published. Measured on the Apple M4 Max 48 GiB (quality, speed) and the Apple M4 Pro
24 GiB (speed, fit), 2026-10-04.

## The idea

- **Where the bytes go in E3** (dev54):
  - 94 % of the file is routed experts, of which a token reads 10 of 512 per layer;
  - the dense projections that **every** token reads are 2-bit IQ2_XXS (DeltaNet `attn_qkv` and `attn_gate`,
    attention `attn_q`, part of the shared expert), only about 290 MiB.
- **The trade:** give those projections more bits, and pay for it by moving a few expert layers from IQ2_XS back
  to IQ1_M, so the file stays at E3's size.

## Variants

`scripts/dev/quant_mix.py --dense-type T` turns every tensor that is IQ2_XXS in the template file into T. Same
Q8_0 source, importance matrix and quantizer as dev54.

| variant | dense IQ2_XXS tensors become | experts IQ2_XS on layers | size |
|---|---|---|---:|
| E3 (published, dev54) | IQ2_XXS (unchanged) | 37–47 | 19.299 GB |
| F1 | IQ3_XXS | 38–47 | 19.334 GB |
| F2 | Q4_K | 40–47 | 19.320 GB |

## Quality (M4 Max)

Perplexity on the frozen corpus (`perplexity.sh`, ctx 512, 111 chunks), a paired per-chunk test against E3, and
agreement with Qwen's own API (`api_compare.py`, 235 prompts, every expert resident):

| variant | perplexity | paired vs E3 (nats/token, t, chunks better) | API: identical / median / mean words matching / first word differs |
|---|---:|---|---|
| E3 | 16.216 ± 0.30 | — | 4 / — / 13.6 % / 36 |
| F1 | 15.419 ± 0.28 | −0.050, t = −16.0, 105/111 | 3 / 8.6 % / 13.1 % / 36 |
| **F2** | **15.379 ± 0.29** | **−0.053, t = −14.9, 103/111** | **5 / 9.2 % / 15.2 % / 35** |

- **Both beat E3 by about 5 % in perplexity**, five times dev54's gain over Bartowski's scheme (−0.9 %), and on
  almost every chunk.
- **F2 against F1:** −0.0026 nats/token, t = −1.29 (63/111): not distinguishable. F2 is ahead on the API agreement.
- **Where the bits matter:** the few hundred MiB of dense weights every token reads are worth more than three expert
  layers at the higher precision.

## Speed and fit

Decode, every expert resident (`bench_m4.sh --only decode22 --cache-mib full`, 3 runs, median):

| | M4 Max | M4 Pro 24 GiB | M4 Pro + MTP (3 prompts × 256 tokens) |
|---|---:|---:|---|
| E3 | 85.21 tok/s | 45.75 tok/s | 50.8 / 58.0 / 47.0 |
| F1 | 84.42 (−0.9 %) | 45.31 (−1.0 %) | 53.6 / 57.1 / 50.1 |
| F2 | 82.81 (−2.8 %) | 44.52 (−2.7 %) | 54.8 / 58.2 / 48.9 |

- **Plain decode:** the extra dense bytes cost 1–3 %.
- **With MTP** (the default on a 24 GiB Mac with the raised GPU limit): F1 and F2 are as fast as E3 or faster, two
  prompts of three. A more accurate model makes drafts that are accepted more often.
- **GPU need at 4K context with MTP** (dev55 model): E3 21,536 MiB, F1 21,569, F2 21,556. All three fit the 21,741
  MiB limit `redlite doctor` recommends; the M4 Pro ran every test above without running out of memory.
- **Same output on both Macs:** greedy token ids on the M4 Max and the M4 Pro are identical (64 tokens, F1 and F2).

## Scope boundary

- **Quality:** perplexity on one corpus and agreement with one API, at context 512. No task benchmarks.
- **Speed:** single prompts, as in dev54. The bounded 4 GiB cache (24 GiB Macs without the raised limit) was not
  measured with F1 or F2.
- **Validation:** F2 passes `regress_m4.sh` 41/0/13 on the M4 Max. The 13 skips are the stage tools that only
  read the reference IQ2_XXS dense layout, as for the IQ3_XXS file. It also passes `quick_parity.sh` (CPU vs Metal,
  GPU-routed, logits and greedy vs the pinned llama.cpp). F1 was not regression-tested.

## Published

F2 is on Hugging Face as `Qwen3-Next-80B-A3B-Instruct-RedLite-F2.gguf`, SHA-256
`d22dbbc4e96ede01a028789d281992b758b817742d2842b82f2e3ce5a9a948c5`. `redlite download 24gb` fetches it,
`redlite download e3` the dev54 file, and `redlite chat` prefers F2 when it is in the models folder.
