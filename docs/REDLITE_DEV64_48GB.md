# Red Lite dev64 — a better file for 48 GiB Macs

Status: in progress. Quantization, perplexity and decode speed measured on the Apple M4 Max 48 GiB, 2026-10-05.
G2 is the candidate. Validation and a long-context test on the M4 Max are done; repeated long runs slow G2's
prompt ingestion (below).

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
| G2 | experts at IQ3_S: `ffn_down` on every layer, `ffn_gate` and `ffn_up` on layers 24–47 | 33,538,648,512 | 31.24 |
| G3 | G2 plus G1's Q8_0 dense tensors | 34,343,954,880 | 31.99 |

## Perplexity

`scripts/dev/perplexity.sh` with the pinned llama.cpp (7798007a2), context 512, on the text corpus (111 chunks) and
the dev63 code corpus (90 chunks).

| File | Text | Code |
|---|---:|---:|
| IQ3_XXS | 14.292 ± 0.261 | 1.9751 ± 0.0228 |
| G1 | 14.232 ± 0.263 | 1.9686 ± 0.0231 |
| G2 | 14.199 ± 0.259 | 1.9483 ± 0.0223 |
| G3 | **14.143 ± 0.262** | **1.9421 ± 0.0226** |

Paired per chunk: mean difference in nats/token, t, chunks where the first file is better.

| Pair | Text | Code |
|---|---|---|
| G1 vs IQ3_XXS | −0.0042, t = −2.1, 62/111 | −0.0033, t = −1.8, 53/90 |
| G2 vs IQ3_XXS | −0.0065, t = −3.5, 62/111 | −0.0137, t = −5.5, 68/90 |
| G3 vs IQ3_XXS | −0.0105, t = −3.9, 67/111 | −0.0168, t = −5.4, 67/90 |
| G3 vs G2 | −0.0039, t = −2.1, 64/111 | −0.0032, t = −1.9, 52/90 |
| G3 vs G1 | −0.0062, t = −3.7, 69/111 | −0.0136, t = −5.3, 61/90 |

- **The dense weights at Q8_0 (G1)** gain little: −0.4 % on text, −0.3 % on code, at the edge of noise. At 3 bits
  the dense part was not the bottleneck, unlike the 2-bit files of dev58.
- **The experts at IQ3_S** carry the gain. G2 has them alone: −0.65 % on text, −1.4 % on code, both clearly beyond
  noise, for 1.69 GiB more than IQ3_XXS. G3 adds the Q8_0 dense weights for another −0.4 % on text and an
  indistinguishable difference on code (t = −1.9).

## Fit on a 48 GiB Mac (planner rule, computed)

| File | Expert cache | Dense | Share of 48 GiB | With MTP |
|---|---:|---:|---:|---:|
| IQ3_XXS | 28,800 MiB | 1,457 MiB | 61.6 % | 65.2 % |
| G2 | 30,528 MiB | 1,457 MiB | 65.1 % | 68.7 % |
| G3 | 30,528 MiB | 2,225 MiB | 66.6 % | **70.3 %** |

G3 with MTP is above the 70 % rule, so the planner would run it without MTP, and MTP is worth +8–30 % of decode
(dev45). G3 was not run with MTP forced: the rule comes from a real out-of-memory failure, and on macOS 27.0.1 the
M4 Max has had GPU driver panics under Red Lite.

The gain of G3 comes from the experts: the Q8_0 dense weights cost 768 MiB and gain 0.3–0.4 % (G1). G2 keeps G3's
experts with IQ3_XXS's dense weights and fits with MTP, at 68.7 %.

## Decode speed (M4 Max 48 GiB, every expert resident)

`redlite-generate MODEL --cache-mib full --max-tokens 256 --temperature 0 --json`, the six prompts of
`tests/fixtures/cache_trace_prompts.txt`, one process per prompt, 30 s apart, nothing else running. MTP with
`models/Qwen3-Next-80B-A3B-Instruct-MTP-ONLY-Q8_0.gguf`. Footprint: `phys_footprint_mib`, the largest of the six.

| File, mode | tok/s per prompt | median | footprint |
|---|---|---:|---:|
| IQ3_XXS, plain | 69.3, 75.4, 73.9, 74.2, 74.7, 74.7 | 74.4 | 29,945 MiB |
| IQ3_XXS, MTP | 88.5, 93.9, 97.0, 83.5, 81.3, 90.1 | 89.3 | 31,685 MiB |
| **G2, MTP** | 88.7, 92.7, 95.3, 84.9, 83.2, 96.4 | **90.7** | 33,415 MiB |
| G3, plain (how the planner runs it) | 72.8, 72.2, 72.8, 71.9, 72.3, 73.0 | 72.5 | 31,674 MiB |

- **G2 with MTP decodes as fast as IQ3_XXS with MTP.** The per-prompt differences follow the drafts accepted, since
  the two files' answers differ.
- **G3 without MTP is 25 % slower than G2.** Its 0.4 % better text perplexity does not pay for that.
- **IQ3_XXS plain** is 74.4 tok/s here against 79.9 in the 0.5.0 record. That record used `bench_m4.sh` with one
  fixed prompt, so the two numbers are not comparable. Every row of this table was measured in one session under
  the same conditions.

## Validation (M4 Max)

- **`regress_m4.sh models/quant/G2.gguf`: 39 PASS, 3 FAIL, 13 SKIP.** The three failures are not regressions:
  - `logits.long_context_vs_llama` and `logits.long_context_full_residency` fail argmax agreement at 2 of 100
    positions. Both are near-ties: llama.cpp's top-1 and top-2 logits differ by 0.0003 at position 18 and by
    0.012 at position 39, the smallest and fourth-smallest gaps of the 100 (median 1.8). The logit and KL limits
    pass (worst |Δlogit| 0.57 against 2.0, worst KL 6e-4 against 2e-2), and the swap does not propagate.
  - `generate.greedy` looks for the words "Rayleigh scattering". G2's correct answer says "molecules … scatter
    shorter wavelengths" instead; `generate.vs_llama` (24 greedy tokens identical to llama.cpp) passes.
- **`quick_parity.sh`: 4/4 PASS**, including the GPU-routed greedy identical to llama.cpp.
- **`server_check.py --cache-mib full --mtp`: SERVER CHECK: YES** (stream and blocking answers identical to
  `redlite-generate`, a warm second turn identical to a cold server).

## Long context, run twice (M4 Max, MTP, `--context 32768`)

A 25,274-token prompt (the first 95,000 bytes of the code corpus and a question), 128 tokens of answer, the same
command run twice. Same output on every run, no errors.

| File, pause before run 2 | Prompt tok/s, run 1 → 2 | Decode tok/s, run 1 → 2 | Footprint |
|---|---|---|---:|
| G2, 60 s | 553 → 314 (−43 %) | 57.6 → 36.6 (−36 %) | 35,084 MiB |
| IQ3_XXS, 180 s | 638 → 624 (−2 %) | 52.6 → 52.0 (−1 %) | 33,354 MiB |
| G2, 180 s | 639 → 492 (−23 %) | 57.9 → 55.2 (−5 %) | 35,084 MiB |

- **G2 is stable** at this length (no failure, unlike IQ3_M in dev36), and its first run is as fast as IQ3_XXS's.
- **A second long run is slower for G2 and not for IQ3_XXS.** The likely cause is memory pressure: the system had
  4.2–5.2 GiB of swap in use during these runs, and G2's footprint is 1.7 GiB larger. This is not proven; the runs
  did not record swap activity during the prefill.

**In the server** (one process, `redlite-server --cache-mib full --mtp --context 32768`, two different long prompts
180 s apart, so the second cannot reuse the first's state). Swap-outs are `vm_stat` pages of 16 KiB written to swap
during the request.

| File, request | Prompt tokens | Prompt tok/s | Decode tok/s | Swapped out during the request |
|---|---:|---:|---:|---:|
| IQ3_XXS, 1 | 25,378 | 619.9 | 57.2 | 0 |
| IQ3_XXS, 2 | 22,518 | 595.5 | 61.0 | 0 |
| G2, 1 | 25,378 | **400.7** | 58.5 | about 9.5 GiB |
| G2, 2 | 22,518 | 572.5 | 56.8 | about 1.2 GiB |

- **The cause is memory pressure.** During G2's first long request, macOS swapped out about 9.5 GiB of other
  processes' memory and prompt ingestion fell by a third. Once the room was made, the second request ran close to
  IQ3_XXS. IQ3_XXS caused no swap-out.
- **Decode is unaffected** in every run.
- **It depends on the rest of the Mac.** This M4 Max had other applications open, with 4–5 GiB of swap in use
  before the tests. On a Mac with less else running the cost should be smaller, but that was not measured.

## Choice

**G2** for 48 GiB Macs: perplexity −0.65 % on text and −1.4 % on code against IQ3_XXS, the same decode speed with
MTP, and inside the planner's 70 % rule. It passes the parity checks. Before it replaces IQ3_XXS in
`redlite download 48gb`:
- a decision on long contexts: G2's 1.7 GiB more pushes a 48 GiB Mac into swap above about 20K tokens, which
  costs a third of prompt speed once;
- the Hugging Face upload.

## Scope boundary

- **Quality** is perplexity on two corpora of about 190 KB each. The API comparison (dev48) and the agent suite have
  not been run on G1, G2 or G3.
- **Speed** is decode at short context on the M4 Max only. Prompt ingestion, long contexts and the M4 Pro are not
  measured, and G2 is not validated against llama.cpp yet.
- **Coverage:** only the Instruct model. Qwen3-Coder-Next at 3 bits is a later step.
