# Red Lite dev66 — faster decode at long contexts

Status: done. Measured on the Apple M4 Max 48 GiB (90 W charger) and the Apple M4 Pro 24 GiB (AC), 2026-10-06.

At long contexts decode time goes to attention: every token reads the whole KV cache, 48 KiB per position (3 GiB
at 64K positions, 12 GiB at 256K). dev65 measured that read at 255–300 GB/s on the M4 Max, about half of the chip's
memory bandwidth. dev66 raises it to about 415 GB/s.

## 1. The change

`attn_gqa_split_g` (dev35) gives one threadgroup to each pair of (KV head, block of positions). It runs in three
passes: query·key scores for the 8 query heads, the softmax statistics, then the weighted sum of the values.

- **Four value loads in flight.** In the value pass each thread (one per dimension) loaded the value of one
  position, waited for it, and only then loaded the next. Now it loads four positions, then accumulates them in
  the same order as before. Results are bit-identical.
- **Blocks of 256 positions from 12K positions on.** With the faster value pass, 256-position blocks beat
  128-position ones from 16K positions (`kernel-bench`: 16K −16 %, 256K −12 %), while 128 stays faster up to 8K.
  `rl_attn_auto_blk()` picks 256 from 12,288 positions. `RL_ENGINE_ATTN_BLK` still forces a fixed block. Different
  blocks merge partial sums in a different order, so the results are not bit-identical above 12K positions; the
  kernel self-test now also checks the 256-position blocks against the double-precision reference
  (worst absolute error 7.6e-8, the same as before).

## 2. Measurements

**`kernel-bench`, M4 Max** (12 attention layers, 12 distinct KV buffer pairs as in the engine, one token; 4K is
left out because it was the first length of the old benchmark and carried its warm-up):

| Positions | Before | After | Change | Bandwidth after |
|---:|---:|---:|---:|---:|
| 32,768 | 5.52 ms | 4.08 ms | −26 % | 395 GB/s |
| 65,536 | 11.52 ms | 7.55 ms | −34 % | 427 GB/s |
| 131,072 | 22.05 ms | 15.71 ms | −29 % | 410 GB/s |
| 262,144 | 42.70 ms | 31.05 ms | −27 % | 415 GB/s |

**`decode-bench`, M4 Max** (CF2, every expert resident, state filled with random values, median of 16 tokens,
the two engines alternated: before, after, after, before):

| Position | Before | After | Change |
|---:|---:|---:|---:|
| 4,096 | 81.1 / 86.4 tok/s | 84.0 / 85.1 tok/s | within noise |
| 65,536 | 46.0 / 45.8 tok/s | 53.9 / 53.9 tok/s | +17 % |
| 131,072 | 31.4 / 30.3 tok/s | 39.0 / 37.6 tok/s | +24 % |
| 262,000 | 18.6 / 18.6 tok/s | 23.8 / 24.6 tok/s | +30 % |

**Through `redlite-server`, M4 Max** (`needle_check.py`, CF2, the same prompts as dev65; dev65 numbers before):

| Prompt tokens | Passphrases | Decode before | Decode after | Ingestion |
|---:|---|---:|---:|---:|
| 62,197 | 3 / 3 | 41.2 tok/s | 47.6 tok/s | 513 tok/s |
| 127,183 | 3 / 3 | 26.3 tok/s | 33.3 tok/s | 318 tok/s |
| 256,155 | 3 / 3 | 16.0 tok/s | 21.6 tok/s | 174 tok/s |

The answers are the same text as in dev65, character for character. Ingestion did not change (the prefill is not
touched); 174 against 182 tok/s at 256K is run-to-run noise, with the battery at 75 % and charging.

**M4 Pro 24 GiB.**
- **`kernel-bench`** (one shared KV buffer pair: twelve 512 MiB pairs at 256K do not fit next to the model):
  256K 87.9 → 55.7 ms, 64K 22.5 → 14.4 ms, 32K 11.6 → 7.6 ms (−34 to −37 %).
- **`decode-bench`** (CF2, the 4 GiB expert cache, experts from the SSD): noisy, because the first run of each
  pair reads experts that the previous run did not leave in the page cache. The second pair, both runs warm:
  32K 21.9 → 24.5 tok/s, 64K 17.4 → 20.5, 128K 12.6 → 16.1. One run after the change measured 10.9 tok/s at 128K,
  a cold-cache outlier of the same kind.

## 3. Validation

- `kernel-selftest` / the build's self-test: 6 lengths × 4 kernels against double-precision GQA (the new case is
  the grouped kernel with 256-position blocks; lengths 1–1,500 include partial blocks and the 4-load remainder).
- `scripts/regress_m4.sh` on the reference IQ2_XXS: 55 / 0 / 0. `quick_parity.sh --long`: 5 / 5. M4 Max.
- Needles at 62K, 127K and 256K tokens through the server, answers identical to dev65.
- `scripts/dev/local_ci.sh`: PASS (ruff, compileall, native, sanitize, test).

## 4. What did not help (details in WHAT_DID_NOT_WORK)

- **Prefill attention** (the first ingestion of a long prompt): a kernel sharing key/value blocks across the 8
  query heads of a KV head, and one reading Q from device memory to fit more threadgroups per core. Both were
  byte-identical and slower (32K: 22.4 and 19 s against 17 s). `attn_fa_b2` is limited by the float32 matrix
  units, at about 6 TFLOPS.
- **Decode, other variants:** two threads per key row in the score pass (−2 to −4 %, gone once the value pass was
  fixed), eight value loads in flight instead of four (no further gain).

## Scope boundary

- Decode attention only. Prefill, experts, DeltaNet layers and the KV cache format (float32) are unchanged.
- The block threshold (12,288 positions) was chosen on the M4 Max; on the M4 Pro the crossover was not measured
  separately.
- No CPU-vs-Metal engine parity run reaches 12K positions (the CPU oracle is too slow there). The 256-position
  blocks are checked by the kernel self-test and by the long needle runs; below 12K the change is bit-identical.
