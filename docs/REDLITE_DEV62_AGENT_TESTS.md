# Red Lite dev62 — five follow-up tests: routing quality, a full-precision reference, harder tasks, IQ3_XXS with an agent, two agents at once

Status: done. Measured on the Apple M4 Max 48 GiB and the Apple M4 Pro 24 GiB, 2026-10-05.

## 1. What cache-aware routing costs in quality (M4 Pro, 4 GiB cache)

With a bounded expert cache the product routes with a bias toward experts already in the cache (λ 0.5, dev51).
dev46 measured it on IQ2_XXS over 2,048 scored tokens. Here: `redlite-engine perplexity`, the first 16,384 ids
of the perplexity corpus, 32 chunks of 512 with the second halves scored (8,192 tokens), the engine's own decode
path:

| file | exact routing | λ 0.5 | change | expert misses |
|---|---:|---:|---:|---:|
| IQ3_XXS | 12.000 | 12.023 | +0.19 % | −18 % |
| F2 | 12.860 | 12.867 | +0.06 % | −19 % |

The bias costs at most 0.2 % in perplexity. The engine's perplexity is on its own scale (token-by-token decode,
half-chunk scoring), not comparable with llama-perplexity's 14.29 / 15.38; compare the columns.

## 2. A full-precision reference

The same pi tasks against Alibaba Cloud's `qwen3-next-80b-a3b-instruct`: the full-precision model this project
quantizes, through its OpenAI-compatible API. The key is read from `~/.config/redlite/dashscope_key` and passed
as an environment variable only.

- **The five dev60 tasks:** 15 of 15, as the 2-bit models do (14–15 of 15). They are too easy to separate models,
  so the test needed harder tasks.

## 3. Harder tasks on a real repository

`agent_eval.py --suite repo` copies this repository (`git archive HEAD`) for each task:

| task | what the agent must do | graded by |
|---|---|---|
| `models_json` | add `--json` to `redlite models` in the 576-line `cli.py`, with a test | the JSON lists every variant; `test_cli.py` passes; the test uses `--json` |
| `explain_stop` | find how the 1,900-line C server handles stop sequences | names `rl_stop_scan` and the held-back text; no file changed |
| `fix_planner` | a planted bug in `planner.py` fails a test; fix it without touching the tests | tests pass, test file unchanged |
| `long_session` | one pi session, four prompts: read two files, name a test class, add a test to it | the class named, a new test in it, tests pass |

Every expert resident, 32K context, M4 Max:

| model | passed | models_json | explain_stop | fix_planner | long_session | median time per task |
|---|---:|---:|---:|---:|---:|---|
| Qwen API, full precision (3 runs) | **10/12** | 3/3 | 3/3 | 2/3 | 2/3 | 10–22 s |
| Qwen3-Coder-Next IQ2_XXS (5 runs) | **14/20** | 4/5 | 5/5 | 1/5 | 4/5 | 52–270 s |
| Red Lite F2 (3 runs) | **6/12** | 2/3 | 0/3 | 1/3 | 3/3 | 3–272 s |

- **The coding model matters.** Qwen3-Coder-Next at 2 bits passes 70 % of the harder tasks against 83 % for
  the full-precision general model. F2 passes 50 %.
- **F2 does not read when it should:** in `explain_stop` it answers from nowhere, in one request, never opening
  the file.
- **Loops.** Both 2-bit models sometimes repeat the same tool calls until the 900 s limit (Qwen3-Coder-Next
  2 of 20 tasks, F2 1 of 12).
- **Three of the four Qwen3-Coder-Next `fix_planner` failures were the server's** (see below), all fixed.

### Server fixes found by these runs

- **256 messages were not enough.** An agent session adds two messages per tool call, and a long one was
  rejected with "too many messages". Messages now live on the heap, up to 16,384. The fixed-size array was also
  copied onto a worker thread's stack, so it could not simply grow.
- **`<command>` for `<parameter=command>`.** Qwen3-Coder-Next at 2 bits sometimes opens a parameter with its
  bare name and closes it with `</parameter>`. The call did not parse and went back as text, which ends the
  agent's turn. The XML parser now accepts `<KEY>`, closed by `</parameter>` or `</KEY>`, for a key of letters,
  digits, `_` and `-`.
- **Unparsed calls are logged** (first 2,000 bytes), so the next such case can be diagnosed from the server log.

**`fix_planner` again, with the fixes** (Qwen3-Coder-Next, four runs): 2/4, against 1/5 before.
- One failure is a garbled call (`<function=bfunction=bash>…`) that nothing could repair; it went back as text,
  as it should.
- The other is a wrong fix.

## 4. IQ3_XXS with the agent, from the SSD (M4 Pro, default GPU limit)

The five dev60 tasks, three runs, `redlite serve --native` with IQ3_XXS: 4 GiB cache, cache-aware routing.

| M4 Pro 24 GiB, 4 GiB cache | passed | sum of median task times | prompt reused |
|---|---:|---:|---:|
| F2 (dev61) | 15/15 | 119 s | 76 % |
| IQ3_XXS | 15/15 | 246 s | 80 % |

- **Same results, about twice the time.** Decode alone is only 14 % slower (dev61). An agent loop, though, is
  many short prompt additions (tool results), and IQ3_XXS ingests short inputs at 208 tok/s against 350.
- On these tasks the better file buys nothing. On the harder ones it was not run.

## 5. Two agents at once (`--parallel 2`, M4 Max, Qwen3-Coder-Next resident)

Two `agent_eval.py` runs one after the other, then two at the same time against the same server:

| round | one after the other | at the same time | tasks passed |
|---|---:|---:|---|
| 2 | 76 s | 74 s | 10/10, 10/10 |
| 3 | 97 s | 92 s | 9/10, 10/10 |

- **2–5 % for agents, against +28 % for plain chat (dev56).** Pairing only helps while both requests decode at
  once. An agent turn is mostly prompt ingestion, which holds the engine alone and makes the other wait, and
  time in the agent's own tools.
- Round 1 is not in the table: one agent looped for 207 s on `add_flag`, so the round measured that loop rather
  than the server.

## Scope boundary

- **Sample size:** three to five runs per configuration on four or five tasks. Pass rates of 6/12 vs 10/12 show a
  difference; one run of difference is noise.
- **API reference:** a remote model with its own sampling defaults; its times include the network.
- **Routing quality** (test 1) was measured with perplexity only, on one corpus.
- **Coverage:** IQ3_XXS was not run on the harder tasks, and the 900 s timeout was not tuned.
