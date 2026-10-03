# Red Lite dev54 — a better expert mix at the same size

Status: on branch `dev54/quant-mix`. Measured on the Apple M4 Max 48 GiB, 2026-10-03/04. Roadmap Phase 4, "a Red Lite
quantization".

## The question

Bartowski's IQ2_XXS file (the 24 GiB reference) keeps 11 of the 48 expert layers at IQ2_XS (2.31 bits per weight):
layers 0–5 and 43–47. The other 37 are at IQ1_M (1.75 bits). Is there a better choice of those 11 layers at the
same size?

Only the expert layers change. The native engine runs IQ2_XS and IQ1_M experts at this size; IQ2_XXS experts and
mixed IQ1_M/IQ2_XS layers are not supported (see E1).

## What the importance matrix says

Bartowski's `imatrix.gguf` records each expert's input energy (sum of squared inputs per token). Weighted by how
often each expert is used, the energy **grows steadily with depth**:
- down-projection input: 1.3 at layer 0, 9–13 in the middle, then 78, 109, 157, 246, 337, 490 and 565 from layer
  41 to 47;
- gate input: 156 to 2001.

Bartowski's scheme spends half of its precise layers on the six with the lowest energy (0–5).

## Method

1. **Source.** Bartowski's Q8_0 (three splits, 84.8 GB) and his imatrix.
2. **Quantizer.** The pinned llama.cpp's `llama-quantize --allow-requantize --imatrix --tensor-type-file`. Every
   non-expert tensor gets exactly the type it has in Bartowski's IQ2_XXS (482 overrides). The pinned llama.cpp's own
   IQ2_XXS recipe differs (IQ2_XXS experts, Q4_K attention) and was not used. Each variant takes 37–43 minutes.
3. **Variants:**
   - R: Bartowski's scheme, reproduced;
   - E3: IQ2_XS on layers 37–47, the highest energy;
   - E3b: IQ2_XS on 0–2 and 40–47;
   - E1: IQ2_XS on every down projection, gate/up IQ1_M (+3 % size).
4. **Measures:**
   - perplexity on the frozen corpus (`perplexity.sh`, ctx 512, 111 chunks), with a paired per-chunk test against R:
     the per-chunk loss comes from the running estimates in llama-perplexity's log;
   - agreement with Qwen's own API (`api_compare.py`, 235 prompts, every expert resident).

## Results (M4 Max 48 GiB)

| variant | size | perplexity | paired vs R (nats/token, t, chunks better) | API: identical / mean words matching / first word differs |
|---|---:|---:|---|---|
| R (Bartowski's scheme) | 19.30 GB | 16.370 ± 0.30 | — | 2 / 12.6 % / 38 |
| **E3** (37–47) | 19.30 GB | **16.216 ± 0.30** | **−0.0095, t = −3.91, 72/111** | **4 / 13.6 % / 36** |
| E3b (0–2, 40–47) | 19.30 GB | 16.326 ± 0.30 | −0.0027, t = −1.28, 60/111 | 2 / 13.2 % / 41 |
| E1 (down projections) | 19.87 GB | 16.423 ± 0.30 | +0.0032, t = +1.13, 50/111 | not runnable in the engine |

- **E3 is better than Bartowski's scheme at the same size, on both measures.** Perplexity is −0.94 %, and the paired
  test is clear (t = −3.9; it wins on 72 of 111 chunks). It also gets more identical answers and more matching words
  against the API.
- **E3b** (keeping three early layers precise) is not distinguishable from R.
- **E1** is worse and 3 % larger.
- **Correctness.** E3 passes `regress_m4.sh` 52/0/0 and `quick_parity.sh` (CPU vs Metal, GPU-routed, logits and
  greedy vs the pinned llama.cpp on its own references).
- **Speed.** Decode is unchanged: same types, same footprint (17,316 MiB at full residency; 87.5 tok/s on the
  sky-is-blue prompt).
- **R vs Bartowski's own file.** R measures 16.370 against 16.4675 for Bartowski's file (dev31) with the same
  corpus and perplexity build. The quantizer version differs; the variants are compared with R, built the same way.

## Reproducing

```bash
python3 scripts/dev/quant_mix.py --like models/Qwen_Qwen3-Next-80B-A3B-Instruct-IQ2_XXS.gguf \
    --q8 models/q8/Qwen_Qwen3-Next-80B-A3B-Instruct-Q8_0-00001-of-00003.gguf \
    --imatrix models/q8/Qwen_Qwen3-Next-80B-A3B-Instruct-imatrix.gguf --iq2xs-layers 37-47 --out E3.gguf
```

It needs the Q8_0 (84.8 GB download), the imatrix, the pinned llama.cpp (`make bootstrap`) and about 40 minutes.

## Scope boundary

- Only one size (≈ 19.3 GB) and one model (Qwen3-Next-80B-A3B-Instruct) were studied. Four variants were compared;
  this is not a search over all layer sets.
- Quality is measured by perplexity on one corpus and agreement with Qwen's API, not by task benchmarks.
- E3 is not distributed: a 19.3 GB file needs hosting (e.g. Hugging Face), which is the project owner's decision.
