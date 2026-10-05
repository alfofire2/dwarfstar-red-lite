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

Some failed agent tasks in dev62 were loops: the model repeated the same tool call until the timeout (one run made
176 requests). Here, Qwen3-Coder-Next IQ2_XXS with every expert resident, a 32K context, the repository suite of
`scripts/dev/agent_eval.py` (four tasks, 600 s timeout), three runs per setting:

| Setting | Tasks passed per run | Loops |
|---|---|---|
| A: server defaults (temperature 0.7, top-p 0.95, top-k 40) | 1, 1, 2 of 4 | pending |
| B: temperature 1.0 (Qwen3-Coder-Next's card) | 0, 2, 2 of 4 | pending |
| C: temperature 1.0, `presence_penalty` 1.0 | 2 of 4, pending | pending |
| D: temperature 0.3 | pending | pending |

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

| File | Text | Code |
|---|---:|---:|
| Bartowski IQ2_XXS | 18.53 ± 0.37 | pending |
| CE3 | 18.24 ± 0.36 | pending |
| CF2 | 18.72 ± 0.39 | pending |

On text, CE3 is 1.6 % better than Bartowski's file. CF2 is 1.0 % worse, the opposite of the Instruct model, where F2
was the best. The error bars overlap; the paired comparison and the code corpus decide.

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
