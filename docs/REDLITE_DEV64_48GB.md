# Red Lite dev64 — a better file for 48 GiB Macs

Status: in progress. Quantization and perplexity measured on the Apple M4 Max 48 GiB, 2026-10-05. Speed and memory
are not measured yet.

## Goal

On a 48 GiB Mac, Bartowski's IQ3_XXS (29.55 GiB) runs with every expert resident and MTP inside the default GPU
limit of the M4 Max (`recommendedMaxWorkingSetSize`, 38,338 MiB): 31,601 MiB without MTP, 33,388 MiB with MTP at a
4K context. The goal is a better file in the remaining room, without raising the limit.

## Variants

Built with `scripts/dev/quant_mix.py` (dev63 options) from Bartowski's Q8_0 and importance matrix, using
Bartowski's IQ3_XXS as the template.

| File | Changes from IQ3_XXS | Bytes | GiB |
|---|---|---:|---:|
| IQ3_XXS (Bartowski) | none | 31,726,709,216 | 29.55 |
| G1 | dense tensors at IQ3_XXS, IQ2_S and IQ4_XS go to Q8_0 | 32,532,015,552 | 30.30 |
| G3 | as G1, plus experts at IQ3_S: `ffn_down` on every layer, `ffn_gate` and `ffn_up` on layers 24–47 | 34,343,954,880 | 31.99 |

## Perplexity

`scripts/dev/perplexity.sh` with the pinned llama.cpp (7798007a2), context 512, on the text corpus (111 chunks) and
the dev63 code corpus (90 chunks).

| File | Text | Code |
|---|---:|---:|
| IQ3_XXS | 14.292 ± 0.261 | 1.9751 ± 0.0228 |
| G1 | 14.232 ± 0.263 | 1.9686 ± 0.0231 |
| G3 | **14.143 ± 0.262** | **1.9421 ± 0.0226** |

Paired per chunk: mean difference in nats/token, t, chunks where the first file is better.

| Pair | Text | Code |
|---|---|---|
| G1 vs IQ3_XXS | −0.0042, t = −2.1, 62/111 | −0.0033, t = −1.8, 53/90 |
| G3 vs IQ3_XXS | −0.0105, t = −3.9, 67/111 | −0.0168, t = −5.4, 67/90 |
| G3 vs G1 | −0.0062, t = −3.7, 69/111 | −0.0136, t = −5.3, 61/90 |

- **The dense weights at Q8_0 (G1)** gain little: −0.4 % on text, −0.3 % on code, at the edge of noise. At 3 bits
  the dense part was not the bottleneck, unlike the 2-bit files of dev58.
- **The experts at IQ3_S (G3)** carry the gain: −1.0 % on text, −1.7 % on code, both clearly beyond noise, for
  2.44 GiB more than IQ3_XXS.

## Fit, estimated (to be measured)

The GPU need should grow by the file's growth (G3: +2,496 MiB): about 35,900 MiB with MTP at 4K and about
37,230 MiB at 32K, from IQ3_XXS's measured 33,388 and 34,732 MiB. That is under
the default limit of 38,338 MiB, but only about 1 GiB below it at 32K. To be measured with `redlite-generate --stats`
and the server at 4K and 32K, with and without MTP, together with decode and prompt speed against IQ3_XXS.

## Scope boundary

- **Quality** is perplexity on two corpora of about 190 KB each. The API comparison (dev48) and the agent suite have
  not been run on G1 or G3.
- **Speed and memory** are estimates until measured; nothing about G3 is validated on a Mac yet.
- **Coverage:** only the Instruct model. Qwen3-Coder-Next at 3 bits is a later step.
