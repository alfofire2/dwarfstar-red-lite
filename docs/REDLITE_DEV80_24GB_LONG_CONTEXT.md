# Red Lite dev80: long contexts on the 24 GiB Mac

Status: done. Apple M4 Pro 24 GiB (AC, GPU limit 21,741 MiB) and Apple M4 Max 48 GiB, 2026-10-09, Red Lite 0.9.5.

## 1. Coding agents with every expert resident

dev77 ran the hard agent suite on the M4 Pro with the 4 GiB expert cache (64K window), where 64K and every expert do
not fit together. After `redlite gpu-limit`, `redlite serve --native --context 65536 --kv f16` keeps every expert
resident: the half KV cache takes 1.5 GiB at 64K instead of 3. The float cache would need 22,077 MiB against the
21,741 MiB limit.

Same suite, limits (900 s per task) and temperature (0.3), pi on the M4 Max, CF2:

| Setup | Runs | Decode | Prefill | Tasks |
|---|---:|---:|---:|---:|
| 4 GiB cache, float KV (dev77, 0.9.3 / 0.9.4) | 4 | 16.9–24.8 tok/s | 139–233 tok/s | 6, 5, 5, 5 of 6 |
| Every expert resident, `--kv f16` (0.9.5) | 2 | 35.4 / 50.7 tok/s | 218 / 307 tok/s | 5, 5 of 6 |

- **Run 1** took 15 minutes for the six tasks (its `two_bugs` failed its check). The 4 GiB runs took 37–56 minutes.
- **Run 2** was slower: `two_bugs` passed after 795 s and 64 requests, and `debug_session` hit the 900 s limit after
  79 requests. The second run of a session was also the slower one in dev77.
- **The server now says so.** When the float cache is what keeps full residency out at the requested context and
  the half one would fit, `redlite serve --native` and `redlite chat` print a one-line tip for `--kv f16`. It is not
  made the default: dev74 measured logit drift for the half cache on CF2 (KL up to 5.8e-2 at 1.1K positions).

## 2. Decode as the context grows

`decode-bench`, CF2, M4 Pro, ms per token:

| Position | 4 GiB cache, float KV | Every expert resident, half KV | Every expert resident, float KV |
|---:|---:|---:|---:|
| 64 | 45.1 | 16.5 | — |
| 16K | 44.5 | 18.9 | 19.6 |
| 32K | 50.2 | 21.7 | 23.5 |
| 48K | 53.2 | 23.8 | — |

- **Every expert resident:** decode falls from 60.6 to 42.0 tok/s by 48K. The extra time is attention:
  - with the float cache it reads ~230 GB/s at 32K, close to the M4 Pro's bandwidth;
  - with the half cache it reads half the bytes but only reaches ~150 GB/s, so it is 8 % faster, not twice.
  - A faster half-precision decode attention kernel could give about 10 % at 32K, for `--kv f16` only. Not attempted.
- **4 GiB cache:** expert reads dominate. These runs came right after the full-residency agent runs had emptied the
  macOS file cache, so most reads went to the SSD: 45 ms per token at position 64, against 27 ms the day before with
  the file cache warm. Attention adds about the same 8 ms at 48K as with full residency.
- **The 15.6 tok/s of `two_bugs` in dev77** therefore came from reads more than from attention:
  - expert misses on a cold file cache;
  - attention at 20–45K;
  - the slowdown of a long session.

## 3. Prefill attention at long contexts (M4 Max)

At 32K, attention is about 44 % of the prefill (12 attention layers at 48.7 ms against 48 expert layers at 15.4 ms,
per 512 tokens). Three kernel rewrites gave nothing; they are in `WHAT_DID_NOT_WORK.md`. The next step there is the
per-kernel GPU counters, not another rewrite.

## 4. How much the half KV cache changes the answers (dev81)

`redlite-engine logits` on the M4 Pro, CF2, every expert resident. The run reads 32,100 tokens with the batched prefill
(32,000 tokens of Red Lite's Python and C sources, then prose from the dev docs that does not appear earlier), then
decodes 500 more one at a time. The dumped logits of those 500 positions are compared:

| Positions 32,100–32,599 | Mean KL | Median KL | Same top token | Perplexity |
|---|---:|---:|---:|---:|
| Half vs float KV cache | 0.0041 | 0.0013 | 487 / 500 (97.4 %) | 6.177 → 6.193 (+0.26 %) |
| CF2 vs Bartowski's Coder IQ2_XXS, both float (the yardstick) | 0.205 | 0.088 | 407 / 500 (81 %) | 6.177 → 6.238 |

The half cache moves the logits about 50 times less than choosing between two 2-bit files of the same model. At 1,100
positions of code it gave a mean KL of 0.0029, 494 / 500 top tokens, and perplexity 3.475 → 3.443. dev74's "KL up to
5.8e-2" was a maximum: here the maxima are 0.12 (32K) and 0.21 (1.1K), on single positions. Recommending
`--kv f16` for 24 GiB agents (section 1) is therefore safe on this evidence. It stays opt-in.

- **Two yardsticks that gave zero.** Prefill chunks of 512 against 2048 and `RL_PREFILL_MPS=0` both produced exactly
  the same logits here, so neither measures numerical noise.
- **A trap on the way.** `redlite-engine tokenize --text` prints at most 16,384 ids on its first line, then one line
  per token with its id and its text. Collecting every number from that output built a text whose second half
  repeated the first: perplexity 1.06, where both caches agree trivially. The corpus is now tokenized in pieces
  of under 16K tokens.

## 5. Eight value loads in flight in the decode attention (dev81)

In the grouped split-K decode attention (`attn_gqa_split_g`), each thread reads one value per position and keeps four
positions in flight (dev66). A half value is 2 bytes, so the half cache had half the bytes in flight of the float
one. With eight positions in flight, the summation order is the same, and the logits are **bit-identical** to before
with either cache: the 50 dumped positions after 32,550 tokens matched byte for byte on the M4 Pro. The kernel
self-test passes.

| `kernel-bench`, decode attention over 12 layers | Before | After |
|---|---:|---:|
| M4 Pro, half cache, 32K / 64K | 5.51 / 10.0 ms | 4.75 / 8.8 ms (−14 % / −12 %) |
| M4 Pro, float cache, 32K / 64K | 7.03 / 13.0 ms | 6.72 / 12.8 ms (−4 % / −1.5 %) |
| M4 Max, half cache, 32K / 128K | 2.85 / 9.1 ms | 2.48 / 8.1 ms (−13 % / −11 %) |
| M4 Max, float cache, 32K / 128K | 3.75 / 13.6 ms | 3.53 / 13.4 ms (−6 % / −1.5 %) |

`decode-bench` on the M4 Pro, every expert resident, two alternated pairs:

| Decode | Before | After |
|---|---:|---:|
| Half cache, 32K / 48K | 45.6–46.1 / 42.0–42.1 tok/s | 47.6–47.8 / 43.5 tok/s (+4 % / +3.5 %) |
| Float cache, 32K / 48K | 42.9–43.0 / 38.7–38.8 tok/s | 43.0–43.6 / 38.8 tok/s (≈ 0–1 %) |

Not kept: 16-byte key loads for the half cache (eight halves per load, 5.95 against 5.52 ms at 32K), and sixteen value
loads in flight (4.94 against 4.74 ms with half, the best being eight for both caches).

## Scope boundary

- **Agent runs:** two runs per setup on one suite, one Mac; sessions at temperature 0.3 differ from run to run.
- **Positions:** `decode-bench` jumps to the position without a prefill (random state), so it times the kernels, not
  the quality.
- **The `--kv f16` tip:** it changes no default.
- **The KV comparison:** one 500-position window of prose at 32K and one of code at 1.1K, on CF2 only.
