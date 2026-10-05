# Red Lite dev63 — repetition penalties, agent loops versus sampling, a Red Lite quantization of Qwen3-Coder-Next

Status: in progress. Measured on the Apple M4 Max 48 GiB (quantization, perplexity) and the Apple M4 Pro 24 GiB
(agent runs), 2026-10-05.

## 1. `presence_penalty` and `frequency_penalty`

The server accepted these OpenAI request fields and ignored them. They now work as in OpenAI's API, over the tokens
generated so far in the answer:

    logit[t] -= presence_penalty * (count[t] > 0) + frequency_penalty * count[t]

- **Request fields** `presence_penalty` and `frequency_penalty`, each from −2 to 2. Out of range is an error, as
  OpenAI does.
- **Server defaults:** `--presence-penalty` and `--frequency-penalty`, both 0.
- **Sampler:** `rl_sampler_accept()` counts each emitted token. The penalties are applied to a copy of the logits
  before temperature, top-k, top-p and min-p. Nothing is allocated when both are 0, so the default path is
  unchanged.
- **Tests:**
  - an offline sampler test: a token's probability falls with presence 1.0 and with frequency 0.3;
  - the fake server echoes the parameters it received (`tests/test_native_server.py`).

  Validation so far: model-free tests only (`make native`, `make test`).

Qwen's model card for Qwen3-Next suggests `presence_penalty` between 0 and 2 against endless repetitions. Section 2
measures whether it helps an agent.

## 2. Agent loops versus sampling (M4 Pro 24 GiB)

Some failed agent tasks in dev62 were loops: the model repeated the same tool call until the timeout. Here,
Qwen3-Coder-Next IQ2_XXS with every expert resident, a 32K context, the repository suite of
`scripts/dev/agent_eval.py` (four tasks, 600 s timeout), three runs per setting:
- A, the server defaults (temperature 0.7, top-p 0.95, top-k 40);
- B, temperature 1.0 (Qwen3-Coder-Next's card);
- C, temperature 1.0 with `presence_penalty` 1.0;
- D, temperature 0.3.

**The first attempt measured a test bug, not the model.** A, B and C passed 4/12, 6/12 and 3/12, never more than
2 of 4 tasks in a run, while the same file passed 10/12 on the M4 Max.
- **Not pi's version:** pi 1.0.2, the M4 Pro's, passed on the M4 Max too.
- **The cause:** two tests of `tests/test_native_defaults.py` read the machine's real GPU limit. On the M4 Pro it is
  raised to 21,741 MiB, and in a clean copy of the repository the two tests fail before the agent does anything.
- **The effect:** the checks of `fix_planner` and `long_session` run these tests, so on the M4 Pro those two tasks
  could not pass. Part of the "loops" there (150 and 321 requests on `fix_planner`) was the model trying to fix
  failures it had not caused.
- **Fixed:** the tests now patch the GPU limit. The study was restarted on the fixed checkout; its results replace
  this paragraph's.

| Setting | Tasks passed per run | Loops |
|---|---|---|
| A | pending | pending |
| B | pending | pending |
| C | pending | pending |
| D | pending | pending |

## 3. Red Lite quantizations of Qwen3-Coder-Next (M4 Max)

The Instruct recipes of dev54 (E3) and dev58 (F2), applied to the Coder with `scripts/dev/quant_mix.py`, from
Bartowski's Q8_0 and his importance matrix. The other tensors keep the types of Bartowski's IQ2_XXS.

| File | Experts | Dense projections | Bytes |
|---|---|---|---:|
| Bartowski IQ2_XXS | IQ2_XS on layers 0–5 and 43–47, IQ1_M elsewhere | IQ2_XXS | 19,298,975,424 |
| CE3 | IQ2_XS on layers 37–47, IQ1_M elsewhere | IQ2_XXS | 19,298,975,424 |
| CF2 | IQ2_XS on layers 40–47, IQ1_M elsewhere | Q4_K | 19,319,619,264 |

### Perplexity

`scripts/dev/perplexity.sh` with the pinned llama.cpp (7798007a2), context 512.

- **Text corpus:** `tests/fixtures/perplexity_corpus.txt`, the corpus of dev54 and dev58.
- **Code corpus:** `tests/fixtures/perplexity_code_corpus.txt`, new in dev63 (`perplexity.sh --corpus`). It is
  `src/llama-vocab.cpp` of the pinned llama.cpp, frozen as a fixture (MIT).

| File | Text (111 chunks) | Code (90 chunks) |
|---|---:|---:|
| Bartowski IQ2_XXS | 18.53 ± 0.37 | 2.623 ± 0.039 |
| CE3 | 18.24 ± 0.36 | 2.584 ± 0.038 |
| CF2 | 18.72 ± 0.39 | **2.537 ± 0.039** |

Paired per chunk: mean difference in nats/token, t, chunks where the first file is better. The per-chunk losses are
recovered from llama.cpp's running perplexity.

| Pair | Text | Code |
|---|---|---|
| CE3 vs Bartowski | −0.016, t = −3.2, 68/111 | −0.015, t = −2.3, 55/90 |
| CF2 vs Bartowski | +0.010, t = +1.2, 56/111 | −0.034, t = −4.0, 67/90 |
| CF2 vs CE3 | +0.026, t = +3.2, 40/111 | −0.019, t = −2.4, 57/90 |

- **CE3** is better than Bartowski's file on both corpora, at exactly the same size.
- **CF2** is the best on code but no better than Bartowski's on text, and worse than CE3 there. This is unlike the
  Instruct model, where F2 won on text (dev58). For the Coder, the Q4_K dense projections help code and not prose.

### Agent suite (M4 Max 48 GiB)

dev62's settings: `agent_eval.py --suite repo --timeout 900`, every expert resident, 32K context, the server's
sampling defaults, pi 0.84.4, three runs per file, one server at a time.

| File | Passed | models_json | explain_stop | fix_planner | long_session | median / slowest task | most requests |
|---|---:|---:|---:|---:|---:|---|---:|
| Bartowski IQ2_XXS | 10/12 | 3/3 | 3/3 | 2/3 | 2/3 | 74 s / 557 s | 49 |
| CE3 | **12/12** | 3/3 | 3/3 | 3/3 | 3/3 | 52 s / 148 s | 20 |
| CF2 | **12/12** | 3/3 | 3/3 | 3/3 | 3/3 | 62 s / **87 s** | **14** |

- **Both Red Lite files pass every task.** On this Mac the suite no longer separates CE3 from CF2, and 12/12
  against 10/12 is a small sample.
- **Choice: CF2.** It has the best code perplexity (−3.3 % against Bartowski's file, t = −4.0) and the tightest
  agent runs: no task above 87 s, at most 14 requests.
- The M4 Pro runs of section 2 used the same Bartowski file and failed far more (4/12 and 6/12). The machines also
  differ in pi's version (1.0.2 there, 0.84.4 here); see section 2.

## 4. `quant_mix.py` for other sizes

New options, used by dev64 for 48 GiB files:
- `--keep-experts`: the experts keep the template's types;
- `--set NAME=TYPE[@LAYERS]`: one tensor type on some layers;
- `--dense-map FROM=TO`: every non-expert tensor of one type gets another;
- `--base-type`: the base type passed to llama-quantize;
- `--print-types`: print the per-tensor types and exit.

The old options give byte-identical type lists (checked for F2 40–47 Q4_K and E3 37–47).

## Scope boundary

- **Sampling:** the loop study uses one model (Qwen3-Coder-Next IQ2_XXS), one Mac and three runs per setting; one
  run of difference is noise.
- **Penalties:** synthetically tested. Their effect on agents is section 2, nothing else.
- **Quantization:** judged by perplexity on two corpora of about 190 KB each. The agent suite runs only on the file
  chosen.
