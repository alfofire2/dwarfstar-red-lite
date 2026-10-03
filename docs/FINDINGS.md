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
expert is resident. On a 24 GiB Mac it does not apply: checking two tokens loads the union of their experts,
and expert loading is already that machine's bottleneck.

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
| cache-aware routing, λ = 0.5 (opt-in) | 22.9 | +5 % | output changes; perplexity did not (17.586 → 17.583, M4 Max) |
| uncached reads (`F_NOCACHE`) | 29.5 | no change | — |

- **Prefetch:** applying the next layer's router to the current layer's input predicts most of the experts
  that layer will choose. The CPU loads them while the GPU is still computing.
- **Slot size classes:** with per-layer slot sizes the cache misses 25 % less often.
- **Cache-aware routing:** prefers experts that are already loaded when the router's scores are close. It
  changes the output, so it is opt-in (`RL_ROUTE_CACHE_BIAS`).

**Smarter replacement than LRU does not pay.** On recorded routing traces, decayed-frequency and segmented LRU
save at most 5 % of misses. Even a clairvoyant policy, which knows the future, would only halve them (34.7 vs
64.1 misses per token in simulation).

**The honest comparison.** When the whole 18 GiB model fits in memory, the llama.cpp launcher decodes faster
on the same Mac: 36–38 tok/s against 33.0. The native runtime's advantage is memory: it runs in about 5 GiB,
so a 24 GiB Mac stays usable for other work.

Since the dev18 build of September (27.8 tok/s decode; prompts ingested one token at a time at 16 tok/s), the
24 GiB numbers rose to 33.0 tok/s decode and 280–350 tok/s ingestion.

## 6. Long context

Each position costs 48 KiB of attention cache. Only 12 of the 48 layers keep one; the DeltaNet layers have a
fixed 72 MiB state instead. So 64K positions need about 3 GiB, and 32K about 1.5 GiB.

On the M4 Max, with every expert resident:

- **Speed at a 33,551-token prompt:** ingestion 417 tok/s, then decode at 25.7 tok/s (81 tok/s at short
  context). At long context the decode time is attention reading the float32 cache; float16 would halve it and
  is not implemented yet.
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

## 8. What did not work

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
