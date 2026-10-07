# Red Lite dev67 — decode attention: the score pass and the merge

Status: done. Measured on the Apple M4 Max 48 GiB (90 W charger) and the Apple M4 Pro 24 GiB (AC), 2026-10-07.

dev66 brought the decode attention to about 415 GB/s on the M4 Max. Two parts of it still left time on the table.

## 1. The changes

- **The merge.** `attn_gqa_merge` combines the per-block partial results of each query head. With 16 query heads it
  ran 16 threadgroups, each walking every block in turn and waiting on each load: 1,024 blocks at 256K positions.
  Without the merge the split pass alone took 25.4 ms of 29.6 at 256K (`kernel-bench`), so the merge cost 4.2 ms.
  - All threads of a head now compute the maximum over the blocks together (a maximum is exact in any order).
  - The sums then run in the same order as before, with four blocks' loads in flight.
  - The merge stays bit-identical and now costs about 1.3 ms at 256K.
- **The score pass.** Each thread read one key row of 1 KiB alone, so the 32 threads of a simdgroup touched 32
  different cache lines at a time.
  - Now eight threads share a row: each takes every eighth `float4`, so together they read 128-byte lines whole.
  - Three shuffles per query head add the eight parts.
  - The summation order changes. The self-test's worst absolute error against double precision is unchanged
    (7.6e-8).

## 2. Measurements

**`kernel-bench`, M4 Max** (12 layers, 12 distinct KV buffer pairs, 0.8.1 → dev67):

| Positions | 0.8.1 | dev67 | Change | Bandwidth |
|---:|---:|---:|---:|---:|
| 65,536 | 7.55 ms | 6.50 ms | −14 % | 495 GB/s |
| 131,072 | 15.71 ms | 13.66 ms | −13 % | 472 GB/s |
| 262,144 | 31.05 ms | 26.67 ms | −14 % | 483 GB/s |

**`decode-bench`, M4 Max** (CF2, every expert resident, random state, median of 16 tokens, alternated):

| Position | 0.8.1 | dev67 | Change |
|---:|---:|---:|---:|
| 32,768 | 64.2–70.9 tok/s | 65.7–72.5 tok/s | within noise (two series disagree) |
| 65,536 | 53.3 / 53.5 tok/s | 55.5 / 56.1 tok/s | +4 % |
| 131,072 | 37.4 / 37.0 tok/s | 42.1 / 42.4 tok/s | +13 % |
| 262,000 | 24.6 / 24.0 tok/s | 26.5 / 27.4 tok/s | +11 % |

At 32K a first series measured dev67 4 % slower and a second, longer series 3 % faster; the spread of a single
setting is about 10 % there. The decode is mostly experts and dense weights at that length, and attention is 3–4 ms
of 15.

**Through `redlite-server`** (needles, CF2): 3 / 3 at 62K and 127K, the same answers. Decode 44.2 and 36.2 tok/s
(dev66: 47.6 and 33.3). Single runs: ingestion, whose code did not change, also moved by 5 % (490 against 513 tok/s
at 62K), so these are not a comparison.

**M4 Pro 24 GiB** (`kernel-bench`, one KV pair): 256K 55.8 → 50.6 ms, 64K 14.4 → 12.9, 8K 2.13 → 1.78 (−8 to −16 % across 4K–256K).
`decode-bench` with the 4 GiB expert cache is dominated by expert reads; the warm runs moved within noise (32K 24.4
→ 24.3–24.6 tok/s, 64K 20.6 → 21.2–21.4, 128K 16.0 → 16.6).

**The 128/256 block threshold** was rechecked: 256 wins at 16K and from 64K, 128 wins at 32K in `kernel-bench`
(3.38 against 3.76 ms), but in `decode-bench` at 32K the difference is inside the noise (68–71 tok/s against
66–72). The rule stays at 256 from 12,288 positions.

**Coding agents** (the dev65 hard suite, CF2, temperature 0.3, three runs, 0.8.2):

| Machine, window | 0.8.0 | 0.8.2 | Time for 18 tasks |
|---|---:|---:|---:|
| M4 Max, 64K window (0.8.0) / 128K window (0.8.2) | 16 / 18 | 16 / 18 | 62 → 52 min |
| M4 Pro 24 GiB, 64K window, 4 GiB cache | 12 / 18 | 11 / 18 | 141 → 169 min |

- **A 128K window** passes the same tasks as 64K on the M4 Max (the same 16, per task): these sessions stay under
  64K, so the larger window neither helps nor hurts.
- **On the 24 GiB Mac** the change is within the run-to-run spread of the agent (6 runs at the time limit instead
  of 4). Agent requests sit at 8–30K tokens, where the decode there is bound by expert reads from the SSD, not by
  attention.

## 3. Validation

- Kernel self-test: 6 lengths × 4 kernels against double-precision GQA, worst abs 7.56e-8.
- `scripts/regress_m4.sh` on the reference IQ2_XXS: 55 / 0 / 0. `quick_parity.sh --long`: 5 / 5. M4 Max.
- Needles at 62K and 127K through the server, answers identical to dev65 and dev66.
- `scripts/dev/local_ci.sh`: PASS.

## 4. Smaller findings

- **RoPE precision is not the cause of the drift from llama.cpp past 64K.** A model-free probe computed the RoPE
  angles of Qwen3-Next (64 rotary dims, base 1e7) on the GPU:
  - Fast-math and `precise::` sin/cos give identical errors.
  - The GPU's `pow` adds a little: against the exact angle, the error at 128K is 2.1e-3 where float can do 1.5e-3
    (a frequency table computed in double on the CPU reaches 1.5e-3). At 256K both are 5.6e-3.
  - llama.cpp computes the angle with the same `pow` formula, so this cannot explain a KL that grows from 1.2e-3 at
    64K to 0.025 at 128K. Not changed.
- **Four key loads in flight** in the old one-thread-per-row score pass: −2 to −3 %, replaced by the eight-lane read.

## Scope boundary

- Decode attention only (`attn_gqa_split_g`, `attn_gqa_merge`). No other kernel, no format change.
- The block threshold is the dev66 one, chosen on the M4 Max.
- No CPU-vs-Metal engine parity reaches 12K positions; the long path is covered by the kernel self-test and the
  needle runs.
