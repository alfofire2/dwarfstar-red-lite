# Red Lite dev85: programming on a 48 GiB Mac

Status: done. Apple M4 Max 48 GiB (macOS 27.0.1), 2026-10-20/21.

Goal: Red Lite with a coding agent for everyday easy and medium programming tasks on the M4 Max, with long
contexts and no crashes, at the best quality that fits 48 GiB.

## 1. Stability: no GPU memory work off the engine thread

macOS 27.0.1 kernel-panicked twice on this Mac in Apple's GPU driver (`IOGPUFamily`), and a user's M5 once (issue
#80; Apple report FB25130929). The same driver bug shows up with MLX (ml-explore/mlx#3186: M1 Ultra, M4, M5, macOS 26
and 27), where it is linked to GPU memory accounting while another thread of the process has work in flight. In both
of our panics, two threads of the process were in the kernel.

**The audit** found one place where that happened: the expert pool created its slabs (`MTLBuffer`s) on first use.
- The slabs were created from the threads that load experts: GCD workers, and the prefill prefetch thread.
- With each one, the pool also called `addAllocation` + `commit` on a residency set attached to the engine's queue.
- This happened while the engine thread had command buffers on the GPU, with the bounded cache and on every first
  request with a cache that was not yet full.

**The fix** (`redmetal_topk_pool_allocate_all`, default; `RL_POOL_EAGER=0` restores the old behaviour): every slab is
created when the engine opens, on the opening thread, with one residency-set commit. After that, loads only write into
buffers that already exist.
- Opening takes about 11 ms longer with the 4 GiB cache.
- Greedy output is identical with the 4 GiB cache and with every expert resident.
- The other Metal allocations during serving (prefill scratch, verify buffers, MPS kernels) run on the engine thread
  while it waits for nothing on the GPU.

No panic occurred during the GPU work of this milestone: three benchmark runs, a regression run and four agent
runs. That is evidence, not proof: the bug is in the driver and was never reproducible on demand.

## 2. A benchmark that tells coding models apart: Aider's polyglot

The six-task agent suite of dev65 could not separate files: CF2 (2-bit) and Bartowski's 3-bit Coder both passed 16
of 18. `scripts/dev/polyglot_eval.py` runs Aider's polyglot benchmark against `redlite serve`.
- **The exercises:** Exercism exercises in Python, JavaScript, Rust and C++. The model gets the instructions and the
  files to complete, and answers with every file it changes in full.
- **The tests:** the exercise's tests run, including the ones Exercism skips by default.
- **The second try:** on a failure the model sees the end of the test output and tries once more.
- **The settings:** greedy decoding, at most 6,144 tokens per answer.
- **The harness check:** `--reference` puts the official solutions through the same tests. 127 of 139 pass; the 12
  others are left out:
  - 8 Rust exercises use crates that their `Cargo.toml` does not declare;
  - 4 C++ exercises need Catch2 or Boost installed.
- **A parser trap on the way:** the model often answers with a bare code block and no file name. The first parser
  dropped those, and CF2 passed almost nothing on the first try. `tests/test_polyglot_eval.py` now covers the formats.

## 3. CG2: G2's recipe on Qwen3-Coder-Next

`scripts/dev/quant_mix.py`, from Bartowski's Q8_0 and importance matrix, with his IQ3_XXS as the template:
- the expert `ffn_down` at IQ3_S on every layer;
- `ffn_gate` and `ffn_up` at IQ3_S on layers 24–47.

That is dev64's G2 recipe. The file is 33,538,651,840 bytes (31.24 GiB): a 30,528 MiB expert cache plus 1.4 GiB of
dense weights.

| File | Size | Polyglot, two tries | First try | Answers at the 6,144-token limit | Code perplexity |
|---|---:|---:|---:|---:|---:|
| Red Lite CF2 (2-bit) | 18.0 GiB | 25.2 % | 9.4 % | 21 | 2.537 ± 0.039 |
| Bartowski IQ3_XXS | 29.5 GiB | 49.6 % | 33.9 % | 6 | 1.917 ± 0.024 |
| **Red Lite CG2** | 31.2 GiB | **59.8 %** | **37.0 %** | 6 | **1.870 ± 0.023** |

By language (CG2, two tries): JavaScript 69.4 %, Python 55.9 %, C++ 54.5 %, Rust 50.0 %. The median time per exercise
fell from 21.9 s (CF2) to 15.7 s: fewer second tries and fewer answers that loop to the token limit. Perplexity is
`scripts/dev/perplexity.sh` on the dev63 code corpus, context 512.

**Validation:** `regress_m4.sh` on CG2 gave 45 PASS, 2 FAIL, 13 SKIP.
- **Against llama.cpp:** 24 greedy tokens are identical, and the logits pass.
- **Every server check passes:** stream, reuse, rewind, lookup, tool call, state restart, two slots.
- **The two failures** are the long-context checks. In 2 of 100 positions the top two tokens swap places (each
  file's first choice is the other's second), while the logit and KL limits pass (worst KL 1.7e-2 against 2e-2). G2
  failed the same two checks the same way (dev64).
- **The 13 skips** are the stage tools for the reference file's dense layout.

## 4. Using it

- `redlite download coder-48gb` now fetches CG2 (it fetched Bartowski's IQ3_XXS; that is `bartowski-coder-48gb`).
- `redlite serve --native --coder` (and `redlite chat --coder`) takes the best Qwen3-Coder-Next file in the models
  folder that this Mac can hold: CG2 on a 48 GiB Mac, else CF2. `redlite setup-pi` now prints that command.
- **Context on the M4 Max.** With every expert resident, CG2 fits up to about 96K positions under the planner's 70 %
  rule, with the half KV cache that dev83 picks by itself; 64K is `setup-pi`'s window. At 128K the planner falls back
  to the 4 GiB cache.
  - macOS lets the GPU use 38,339 MiB here (Metal's `recommendedMaxWorkingSetSize`), and CG2 at 128K with the half
    cache needs 36,209 MiB.
  - The 70 % rule stays anyway: in dev36 IQ3_M with every expert resident (footprint 35.2 GiB) ran out of GPU memory
    on its third run, below macOS's limit.

## 5. Coding-agent suite on CG2

dev65's hard suite (six tasks with pi on a copy of this repository, 900 s each, temperature 0.3), with:
- the 64K window, every expert resident and prompt lookup;
- CG2 with the half KV cache that `serve --coder` picks, CF2 with the float cache;
- CG2 and CF2 alternated, two runs each.

| Run | Passed | Total time | Prompt ingestion | Decode | Full recomputes |
|---|---:|---:|---:|---:|---:|
| CG2, run 1 | **6 / 6** | 710 s | 535 tok/s | 86.0 tok/s | 1 |
| CF2, run 1 | 5 / 6 | 412 s | 693 tok/s | 88.6 tok/s | 0 |
| CG2, run 2 | **6 / 6** | 984 s | 551 tok/s | 84.4 tok/s | 1 |
| CF2, run 2 | 4 / 6 | 2,449 s | 497 tok/s | 83.5 tok/s | 12 |

- **CG2 passed 12 of 12, CF2 9 of 12.**
  - CF2's failures: one answer cut at the output limit (`two_bugs`), and two loops that ran to the 900 s limit (377
    and 541 requests, `c_limit` and `debug_session`).
  - The loops also cost CF2's second run its prompt reuse: 12 full recomputes.
- **Decode by context** (CG2 / CF2, both runs): 98 / 111 tok/s below 8K, 86 / 86 at 16–32K, 79 / 75 above 32K.
  - CG2 reads more expert bytes per token, which costs about 12 % at short contexts.
  - At long contexts attention takes more of each token, and CG2's half KV cache probably makes up for its larger
    experts.
- **Prompt ingestion:** CG2 is about 20 % slower than CF2's first run (535–551 against 693 tok/s). With 95–98 % of
  each prompt reused, this is mostly the few thousand new tokens of each turn.
- **Stability:** the server logs show no error, and macOS did not panic.

For an agent on the M4 Max, CG2 is the file: its tasks took 15 s to 7.4 minutes, and it did not loop.

## Scope boundary

- **Machine:** the M4 Max 48 GiB only. CG2 does not fit a 24 GiB Mac.
- **The benchmark:** one run per file, greedy. Some exercises are near the edge, so a second run can move a few points.
  The gap between CG2 and CF2 (35 points) is far beyond that.
- **The panic fix:** it removes the pattern found in Red Lite. The driver bug itself is Apple's.
