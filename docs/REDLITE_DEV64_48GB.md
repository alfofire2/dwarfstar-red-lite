# Red Lite dev64 — a better file for 48 GiB Macs

Status: in progress. Quantization and perplexity measured on the Apple M4 Max 48 GiB, 2026-10-05. Speed and memory
are not measured yet.

## Goal

On a 48 GiB Mac at the default GPU limit, `redlite chat` and `serve` keep every expert resident when the expert
cache plus the dense weights (plus the MTP block, 1,787 MiB) stay under 70 % of RAM, 34,406 MiB
(`native_full_residency_fits`; 70 % since dev36, when IQ3_M at 74 % ran out of GPU memory on the M4 Max).
Bartowski's IQ3_XXS (29.55 GiB) is at 61.6 %, 65.2 % with MTP. The goal is a better file in the remaining room, still
with MTP.

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

## Fit on a 48 GiB Mac (planner rule, computed)

| File | Expert cache | Dense | Share of 48 GiB | With MTP |
|---|---:|---:|---:|---:|
| IQ3_XXS | 28,800 MiB | 1,457 MiB | 61.6 % | 65.2 % |
| G3 | 30,528 MiB | 2,225 MiB | 66.6 % | **70.3 %** |

G3 with MTP is above the 70 % rule, so the planner would run it without MTP, and MTP is worth +8–30 % of decode
(dev45). G3 was not run with MTP forced: the rule comes from a real out-of-memory failure, and on macOS 27.0.1 the
M4 Max has had GPU driver panics under Red Lite.

The gain of G3 comes from the experts: the Q8_0 dense weights cost 768 MiB and gain 0.3–0.4 % (G1). Next variant,
**G2**: G3's IQ3_S experts with IQ3_XXS's dense weights, 68.7 % with MTP (computed).

## Scope boundary

- **Quality** is perplexity on two corpora of about 190 KB each. The API comparison (dev48) and the agent suite have
  not been run on G1 or G3.
- **Speed and memory** are estimates until measured; nothing about G3 is validated on a Mac yet.
- **Coverage:** only the Instruct model. Qwen3-Coder-Next at 3 bits is a later step.
