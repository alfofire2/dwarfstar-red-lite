# What we learned

What building a single-model runtime for Qwen3-Next-80B-A3B on Apple Silicon taught us: what made it fast,
what did not, and where the limits are. Every number names its machine and comes from a record in
`benchmarks/` (the milestone documents have the commands). The charts are drawn from
`benchmarks/charts.json` by `scripts/dev/make_charts.py` and are redrawn whenever a milestone measures
something new.

Machines: an **Apple M4 Max with 48 GiB** (the development machine) and an **Apple M4 Pro with 24 GiB** (the
target class). No result is carried over from one to the other.

## 1. The idea: an 80B model that touches 3 % of itself per token

Qwen3-Next-80B-A3B has 48 layers. Each layer ends in a mixture of 512 experts, of which a token uses 10.
Of the 18 GiB IQ2_XXS file, only about 1 GiB is dense weights that every token needs. The other ~17 GiB are
experts, and one token reads about 3 % of them.

So Red Lite keeps the dense part resident and treats the experts as a cache:

- **24 GiB Macs:** a 4 GiB expert cache. The process uses about 5 GiB in total and leaves the rest of the
  memory to macOS.
- **48 GiB Macs:** every expert resident (17.3 GiB). Then a whole token can be routed on the GPU in one command
  buffer, with no CPU round trip between layers.

## 2. Where a token's time goes

<p align="center"><img src="img/token_time.svg" alt="GPU time per decode token by stage"></p>

With every expert resident, a token takes 12.1 ms on the M4 Max. A third of that is the 10 routed experts.
DeltaNet is the next largest block: its projections, state update and tail, 36 of the 48 layers. Attention is
small at short context, because only 12 layers have it. This profile decided what to optimize next. It is also
why speeding up encoders and barriers alone gained little (see section 7).

## 3. Speed by release, against llama.cpp

<p align="center">
  <img src="img/decode_m4max.svg" alt="Decode speed by release">
  <img src="img/prefill_m4max.svg" alt="Prompt ingestion speed by release">
</p>

The reference is the same llama.cpp commit that Red Lite uses as its correctness oracle, measured on the same
token ids.

- **Decode:** level with llama.cpp in 0.3.0, 12 % ahead in 0.4.1 and 19 % in 0.5.0 (86.2 vs 72.5 tok/s), mostly from GEMV kernels
  written for the exact quant types of this file, and from concurrent encoders with barriers only between
  dependent dispatches.
- **Prompt ingestion:** 4× slower than llama.cpp in 0.3.0. Now on par (899 vs 894 tok/s at 8192 tokens) even
  with only a 4 GiB expert cache. What got there:
  - batched matrix kernels for experts and DeltaNet;
  - tiled attention;
  - a background thread that loads the next layer's experts during compute;
  - 2048-token chunks, so a bounded cache reloads each expert fewer times.

## 4. Speculative decoding that never changes the answer (MTP)

<p align="center"><img src="img/mtp.svg" alt="Decode speed with and without MTP on six prompts"></p>

Qwen3-Next was trained with a **multi-token prediction** (MTP) block: a small extra layer that guesses the
token after next. The GGUF files leave it out, so Red Lite loads it from a separate 2.26 GiB file. Each step,
it decodes the pending token and the guess together, in one pass over the weights.

The guess is accepted only if it is exactly the token that decoding would have produced, with whatever sampler
is in use. So **the output is identical**, not merely similar. The gain depends on how predictable the text is:

| text | gain |
|---|---|
| code and arithmetic | +26–30 % (guess accepted 84 % of the time) |
| creative prose | +8–10 % (accepted 58–62 %) |
| interactive chat on code | 88.7 → 108.6 tok/s |

Three details made the paired pass cheap:

- kernels that decode each weight block once for both tokens: computing two dot products separately cost
  1.74× one, not ~1.06×;
- DeltaNet state snapshots written inside the kernel, so a rejected guess is undone by swapping two pointers;
- the two tokens' experts merged into one dispatch.

MTP is on by default in `redlite chat` and `redlite serve --native` when the head file is present and every
expert is resident. On a 24 GiB Mac that needs a raised GPU limit (section 5): there it gains +15 % median
(46 → 52.7 tok/s, up to 55.7 on code). With the bounded cache it does not apply: checking two tokens loads the
union of their experts.

**MTP and long contexts** (dev55). The 2-row verify reads the attention cache twice, so the gain shrinks as the
context grows.
- **M4 Pro** (half the M4 Max's memory bandwidth): +24 % at a 20-token prompt, +4 % at 5.6K, 0 % at 11.2K,
  −12 % at 16.8K.
- **M4 Max:** still +2 % at 16.8K.
- **What the product does:** below 40 GiB, `redlite chat` and `serve` skip MTP for answers that start past 8,192
  positions (`--mtp-max-context`).

<p align="center"><img src="img/mtp_context.svg" alt="MTP gain against prompt length on the M4 Pro"></p>

**Two requests in one pass** (dev56). The same 2-row pass can carry two different conversations instead of one
conversation's next two positions. `redlite serve --native --parallel 2` does that when two requests decode at the
same time:
- answers identical to serving them in turn;
- total throughput on the M4 Pro +28 % without MTP (43.5 → 55.9 tok/s) and +10 % against MTP (51.0 → 56.0);
- each answer is slower, about 29 tok/s instead of 46.

<p align="center"><img src="img/parallel_server.svg" alt="Two server requests in turn and at the same time, M4 Pro"></p>

## 5. 24 GiB Macs: what the expert cache needs

<p align="center">
  <img src="img/decode_m4pro.svg" alt="Decode speed on the M4 Pro 24 GiB">
  <img src="img/cache_ab_m4pro.svg" alt="Expert cache options on the M4 Pro">
</p>

Measured on the M4 Pro 24 GiB, 4 GiB expert cache, six prompts:

| change | misses per token | decode | effect |
|---|---:|---:|---|
| prefetch off vs on | 45.5 vs 29.5 | 29.2 vs 32.7 tok/s | **prefetch is worth 12 %** |
| uniform slots vs slot size classes (dev37) | 39.3 vs 29.5 | +1.6 % | layers with smaller experts fit more of them |
| cache-aware routing, λ = 0.5 (default with a bounded cache since dev51c) | 22.9 | +5 % | output changes; perplexity (17.586 → 17.583) and agreement with Qwen's API (8.1 vs 7.8 % median) did not |
| uncached reads (`F_NOCACHE`) | 29.5 | no change | — |

- **Prefetch:** applying the next layer's router to the current layer's input predicts most of the experts
  that layer will choose. The CPU loads them while the GPU is still computing.
- **Slot size classes:** with per-layer slot sizes the cache misses 25 % less often.
- **Cache-aware routing:** prefers experts that are already loaded when the router's scores are close. It
  changes the output but no quality measure moved, so `redlite chat` uses it with a bounded cache
  (`--exact-routing` turns it off).

**Smarter replacement than LRU does not pay.** On recorded routing traces, decayed-frequency and segmented LRU
save at most 5 % of misses. Even a clairvoyant policy, which knows the future, would only halve them (34.7 vs
64.1 misses per token in simulation).

**A bigger cache does not help either** (dev51). From 4 to 14 GiB, misses fell from 29.5 to 12.6 per token, but
decode stayed at 31–32 tok/s. With a bounded cache the engine waits for the GPU once per layer, 48 times per
token, and that round trip is the bound, not the loading.

**What does help: every expert resident.** macOS lets the GPU use 17.76 GiB on a 24 GiB Mac, just under the
18.4 GiB of IQ2_XXS's experts plus dense weights. With the limit raised (`sudo sysctl iogpu.wired_limit_mb`,
until reboot), the whole token runs on the GPU in one command buffer:

| M4 Pro 24 GiB, IQ2_XXS | decode | memory in use |
|---|---:|---:|
| 4 GiB expert cache | 32.5 tok/s | ~5 GiB |
| llama.cpp launcher (whole model resident) | 36–38 tok/s | ~18 GiB |
| every expert resident | **46.0 tok/s** | ~18 GiB |
| every expert resident + MTP | **52.7 tok/s** | ~20 GiB |

The trade-off is memory: about 1.3 GB of other apps went to swap and stayed there. `redlite doctor` prints the
limit to set; `redlite chat` picks full residency (and MTP) on its own once the limit allows it.

**Long contexts within the raised limit** (dev55). The GPU memory full residency needs was measured as a function
of the context, the prefill chunk and MTP:

    slots + dense + 580 MiB + 48 KiB per position + 572 MiB for 2048-token chunks + 1,787 MiB for MTP

`redlite chat` and `serve` use it to choose what fits. At a 21,741 MiB limit:

| context | prefill chunks | MTP |
|---|---|---|
| 4K | 2048 | yes |
| 16K | 512 | yes |
| 32K | 2048 | no |

A 16.8K-token prompt in a 32K context now reads at 358 tok/s and decodes at 36.4 tok/s; before, it ran out of GPU
memory with MTP on.

**Setting the limit for good** (dev55b). `sysctl` only lasts until a restart. A small LaunchDaemon sets the limit at
every boot (recipe in `docs/REDLITE_DEV51_24GB_DECODE.md`), and `redlite doctor` recognizes it. Verified on the M4
Pro: the limit is applied, and Metal then reports 21.23 GiB.

Since the dev18 build of September (27.8 tok/s decode; prompts ingested one token at a time at 16 tok/s), the
24 GiB numbers rose to 46–53 tok/s decode and about 360 tok/s ingestion.

## 6. Long context

Each position costs 48 KiB of attention cache. Only 12 of the 48 layers keep one; the DeltaNet layers have a
fixed 72 MiB state instead. So 64K positions need about 3 GiB, and 32K about 1.5 GiB.

On the M4 Max, with every expert resident:

- **Speed at a 33,551-token prompt** (dev53, current build, every expert resident):
  - ingestion 566–587 tok/s;
  - decode 54.5–57.4 tok/s, against 81–86 at short context.
  - dev47 had measured 417 and 25.7 tok/s on the same prompt; the gap is not explained (that session came right
    after a battery-powered one).
- **Float16 KV cache** (dev53): it halved the context memory without changing the speed. It moved a few expert
  choices between the CPU reference and Metal, which the parity rules do not allow, so it was not kept.
- **Agreement with llama.cpp at 16K positions:** the top token agrees at all 100 positions checked, KL
  divergence 7.9e-5. 32K and 60K were not compared, because the oracle's token-by-token dump would take 5–9
  hours.

## 7. How we know it is correct

- **Model-free tests.** Every native stage has a CPU reference in double precision, and Metal is compared with
  it.
- **Real-model checks against llama.cpp.** The pinned llama.cpp is used only as an oracle, never linked into a
  runtime binary. Each change is checked against it on the real model: tokenizer, logits, 24 greedy tokens and a
  1200-token long context. `scripts/regress_m4.sh` runs 51 such checks.
- **Router mismatches are hard failures.** If the selected experts differ between the CPU and Metal, the change
  is reverted. Small floating-point drift is compared across the whole stack, not stage by stage.
- **Against the model maker's own API (dev48).** 235 prompts, compared word for word with Alibaba Cloud's Qwen3-Next.
  IQ3_XXS is about twice as close as IQ2_XXS: 10 vs 2 identical answers; the median share of words matching from
  the start is 15.6 % vs 7.8 %. Greedy answers diverge early once one word differs, so the comparison between
  files says more than the absolute numbers. If you have the memory, IQ3_XXS is the more faithful file.

  <p align="center"><img src="img/api_agreement.svg" alt="Agreement with the Qwen API"></p>

- **The 24 GiB Mac.** llama.cpp itself runs out of GPU memory there with the whole model resident. Instead, the
  M4 Pro's own outputs were compared with the M4 Max's (which match llama.cpp), and they are bit-identical:
  logits, the long-context dump, greedy tokens and tokenizer.

## 8. A better file at the same size

Bartowski's IQ2_XXS keeps 11 of the 48 expert layers at the higher precision (IQ2_XS), half of them the first six.
Bartowski's importance matrix shows that the experts' input energy grows steadily with depth, 400× from layer 0 to
47. Moving the 11 precise layers to 37–47, with every other tensor unchanged:

| | Bartowski's scheme (rebuilt) | Red Lite E3 |
|---|---:|---:|
| size | 19.30 GB | 19.30 GB |
| perplexity | 16.370 | **16.216** (paired t = −3.9) |
| answers identical to Qwen's API | 2 / 235 | **4 / 235** |

E3 is on Hugging Face ([alfodaniello/Qwen3-Next-80B-A3B-Instruct-RedLite-GGUF](https://huggingface.co/alfodaniello/Qwen3-Next-80B-A3B-Instruct-RedLite-GGUF)),
and `redlite download 24gb` fetches it.

**The dense weights matter more** (dev58). In E3 the projections that every token reads (DeltaNet and attention
inputs, part of the shared expert) are still 2-bit: about 290 MiB of a 19.3 GB file. Raising them to IQ3_XXS (F1) or
Q4_K (F2), and paying with one or three expert layers back at IQ1_M, keeps the size:
- **perplexity −5 % against E3:** 15.42 (F1) and 15.38 (F2) against 16.22, better on 103–105 of 111 chunks;
- **API agreement:** F2 reaches 5 identical answers and 15.2 % mean matching words (E3: 4, 13.6 %);
- **speed:** plain decode 1–3 % slower; with MTP on the M4 Pro as fast as E3 or faster.

<p align="center"><img src="img/quant_ppl.svg" alt="Perplexity of R, E3, F1 and F2"></p>

## 9. What did not work

[WHAT_DID_NOT_WORK.md](WHAT_DID_NOT_WORK.md) lists every reverted attempt with its measurement. Highlights:

| attempt | result | lesson |
|---|---|---|
| reading experts in place from the memory-mapped file | 4–10× slower prompt ingestion | Metal re-establishes residency of each layer's expert window for every command buffer; copy into owned slots |
| spin-waiting on command buffer status | 44.6 → 27.5 tok/s | polling fights the driver's completion path |
| removing *all* GPU barriers (as a bound) | 118 tok/s, wrong output | most dispatches depend on the previous one; the correct version gains 2–6 % |
| one fewer encoder per token (1,200 → 1) | +1.9 % | encoder boundaries are cheap on Apple GPUs; the time is inside the kernels |
| staging codebooks in threadgroup memory (what llama.cpp does) | no gain | lookups from `constant` memory were not the bound once the GPU was warm |
| 32-pair prefill expert tiles instead of 16 | 400 vs 872 tok/s | register pressure; per-thread work, not decode reuse, sets the speed |
| chunk-parallel DeltaNet prefill (as in MLX) | not built: at most ~2 % | measured first: the recurrence is 6.7 % of an 8K-token ingestion |
| IQ3_M (a larger quant) on 48 GiB | slower than IQ3_XXS, perplexity within the error bar | full residency ran out of GPU memory; the planner rule went from 75 % to 70 % of RAM |

Traps that cost time and are now guarded:

- **Benchmark conditions:** the M4 Max decoded at less than half speed on battery, and back-to-back prompt
  runs throttled to a third of the speed. Only cooled, alternated, same-session runs on AC power are compared.
- **A stale binary:** a release table was once measured on an outdated build.
- **Weak checks:** a parity check passed on an empty dump. The checks now refuse empty or stale inputs.

## Updating these pages

When a milestone measures something that belongs in a chart:

1. Record the measurement as JSON in `benchmarks/`.
2. Add the row to `benchmarks/charts.json`, naming that record in the chart's `sources`.
3. Run `python3 scripts/dev/make_charts.py`.

`tests/test_charts.py` fails if a committed chart is out of date or cites a record that does not exist.
