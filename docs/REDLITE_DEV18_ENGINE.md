# Red Lite dev18 — native end-to-end Qwen3-Next inference

Target: Apple M4 Pro, 24 GiB unified memory, Bartowski
`Qwen_Qwen3-Next-80B-A3B-Instruct-IQ2_XXS.gguf` (17.97 GiB, 48 layers,
36 Gated DeltaNet + 12 full-attention blocks, 512 experts / top-10).

Dev18 closes the gap between the audited per-stage validators (dev9–dev17) and a
usable runtime: `redlite-generate` loads the GGUF, tokenizes a prompt, runs all
48 layers with persistent recurrent state and KV cache, produces logits, samples
and decodes text — natively, with no llama.cpp and no Python at inference time.
The pinned llama.cpp (`7798007a29a90e3053e799394da48cf53a2f8e0f`) is used only as
a development-time oracle.

## Validation classes used below

- **synthetic**: hand-computed fixtures, no model file;
- **real-GGUF stage**: one stage on real tensors, CPU oracle vs Metal;
- **engine parity**: the whole 48-layer engine replaying a stateful token
  sequence through the CPU oracle and the Metal backend;
- **oracle comparison**: the Metal engine vs the pinned llama.cpp activations,
  logits and greedy tokens on the same token ids;
- **end-to-end**: a prompt string generated to text.

## Components

| Piece | Files | Semantic reference | Validation |
|---|---|---|---|
| GGUF directory + metadata + mmap | `redlite_native_gguf_dir.[ch]` | GGUF v3 spec | synthetic fixture; real header (843 tensors, 49 KV) |
| Dense quant oracle Q8_0/Q2_K/Q4_K/Q5_K/Q6_K/IQ2_XXS | `redlite_native_quant_cpu.[ch]` | pinned `ggml-quants.c` | synthetic blocks; real rows identical to gguf-py dequantization (`scripts/dev/quant_rows_reference.py`) |
| Persistent engine (audit, states, CPU oracle) | `redlite_native_engine*.{h,c}` | `src/models/qwen3next.cpp`, `delta-net-base.cpp` | engine parity, oracle comparison |
| Metal backend | `redmetal_engine.m` | validated dev14–dev16 kernels + new Q5_K rows, threadgroup GQA, fused DeltaNet state | engine parity, oracle comparison |
| Batched expert pool | `redmetal_topk.m` (rewritten execution) | dev10/dev11 IQ2_XS/IQ1_M decode | `topk-parity`, `routed-parity`, full-width FFN parity, engine parity |
| Tokenizer | `redlite_native_tokenizer.[ch]`, `redlite_native_unicode_data.h` | pinned `llama-vocab.cpp`, `unicode.cpp` | identical ids to `llama_tokenize` on the 30-input corpus `tests/fixtures/tokenizer_corpus.txt` |
| Sampler | `redlite_native_sampler.[ch]` | llama.cpp default chain order (top-k → top-p → temperature) | greedy identical to llama.cpp; seeded runs reproducible; model-free selftest |
| Generation CLI | `redlite_native_generate_cli.c` | hard-coded Qwen3-Next Instruct chat template (matches this GGUF's `tokenizer.chat_template` for system/user/assistant turns; not interpreted at runtime) | end-to-end + oracle comparison |

## Model input / output path (real GGUF)

- `token_embd.weight`: Q2_K `(2048, 151936)`; one row is dequantized on the CPU
  per token (no scaling, `build_inp_embd` semantics).
- `output_norm.weight`: F32 `(2048)`.
- `output.weight`: Q5_K `(2048, 151936)`, **untied** (the tensor exists; the
  tied fallback is implemented but not used by this GGUF). 151 936 logits per
  token from a Q5_K row kernel.
- Tokenizer: `gpt2` model, `qwen2` pre-tokenizer, 151 936 tokens, 151 387 merges,
  EOS `151645` (`<|im_end|>`), BOS/PAD `151643` (`<|endoftext|>`),
  `add_bos_token=false`. Generation stops on `<|im_end|>` or `<|endoftext|>`.

## Semantic finding: DeltaNet key/value head pairing

The dev15 state kernels paired value head `h` with key head `h % 16`. The pinned
llama.cpp (`build_layer_attn_linear` → `ggml_repeat_4d` over a `[S,1,H_k]` view,
i.e. repeat-interleave) pairs value head `h` with key head `h / (H_v / H_k)` =
`h / 2`. Both variants are internally consistent, so CPU-vs-Metal parity could
not detect it; the layer-0 comparison against llama.cpp activations did
(max abs `1.4e-06` only with the interleave pairing). All CPU oracles and Metal
kernels now use `h / (value_heads / key_heads)`; the dev15 validators were
re-run and still report parity.

## Engine parity (M4 Pro, real GGUF)

```bash
.deps/redmetal/redlite-engine parity models/Qwen_Qwen3-Next-80B-A3B-Instruct-IQ2_XXS.gguf \
  --tokens 9707,11,1879,0,785,12884 --cache-mib 1024 --context 64
```

Three-token excerpt (stateful: positions 0, 1, 2; conv/recurrent states and KV
caches carried between tokens in both backends):

```text
token 0 id=9707 pos=0 : layer_max_abs=3.05e-05 final_norm_abs=2.67e-05 logits_abs=1.48e-05 argmax gpu=11 cpu=11 router_mismatch=0 -> OK
token 1 id=11   pos=1 : layer_max_abs=8.81e-06 final_norm_abs=1.50e-05 logits_abs=7.15e-06 argmax gpu=358 cpu=358 router_mismatch=0 -> OK
token 2 id=1879 pos=2 : layer_max_abs=1.14e-05 final_norm_abs=7.63e-06 logits_abs=6.68e-06 argmax gpu=0 cpu=0 router_mismatch=0 -> OK
MULTI-TOKEN ENGINE PARITY: YES
```

Router selections (top-10 ids, all 48 layers) are identical between the CPU
oracle and Metal for every token; a mismatch is a hard failure.

## Oracle comparison against pinned llama.cpp

`scripts/dev/ref_llama/redlite_ref_llama.cpp` (development-only, linked against
the bootstrapped llama.cpp build, Metal backend, single-token decode, F32 KV
cache) dumps `model.embed_tokens`, every `l_out-N`, `result_norm` and the logits;
`scripts/dev/compare_dumps.py` compares them with the engine's dump.

Tokens `9707,11,1879`, Metal engine vs llama.cpp Metal:

| quantity | max abs | cosine | note |
|---|---:|---:|---|
| embedding | 0 | 1 | identical Q2_K dequantization |
| layer outputs (48 layers × 3 tokens) | 5.1e-07 .. 6.3e-05 | 1.000000 | ref magnitudes up to 57 |
| final norm | ≤ 1.3e-04 | 1.000000 | |
| logits | ≤ 1.5e-05 | 1.000000 | KL(ref‖native) 6e-13 .. 3e-12 |

Argmax and top-5 identical for all three tokens.

Greedy generation of the chat-templated prompt *"Explain in one sentence why the
sky is blue."* (19 template tokens): the native run produces 28 tokens
(`785 12884 7952 … 71816 13 151645`) and stops at `<|im_end|>`; llama.cpp's
greedy decode of the same prompt ids produces the identical 28-token prefix.

Tolerance policy: float32 Metal accumulation vs float32 llama.cpp accumulation
differs at the 1e-5 level on logits of magnitude ~10–20; the criterion is
identical argmax/top-5 and identical greedy tokens, which holds.

## Tokenizer

`redlite-engine tokenize MODEL --text ... [--chat] [--no-special]` reproduces
`llama_tokenize` ids on the 30-input corpus in `tests/fixtures/tokenizer_corpus.txt`
(ASCII, contractions, digit runs, CJK/Cyrillic/Arabic/accents, emoji and flag
sequences, code, tabs, LF runs, leading and trailing whitespace, URLs, special
tokens inside text, the full chat template). `--file CORPUS` tokenizes one line
per input; the regression suite compares it against `redlite-ref-llama tokenize
--file` (check `tokenize.corpus_vs_llama`). Decoding round-trips UTF-8 byte-exactly.

## End-to-end

```bash
make native
.deps/redmetal/redlite-generate models/Qwen_Qwen3-Next-80B-A3B-Instruct-IQ2_XXS.gguf \
  --prompt "Explain in one sentence why the sky is blue." --max-tokens 40 --cache-mib 4096 --stats
```

Output on the M4 Pro (greedy):

```text
The sky appears blue because molecules in the Earth's atmosphere scatter shorter blue wavelengths of sunlight more than other colors due to Rayleigh scattering.
```

See "Final field run" below for the exact numbers of the clean-build run.

### Interactive terminal chat

`redlite-generate --interactive` keeps the same `rl_engine` alive across user
turns instead of reopening the GGUF for every prompt:

```bash
.deps/redmetal/redlite-generate models/Qwen_Qwen3-Next-80B-A3B-Instruct-IQ2_XXS.gguf \
  --interactive --context 4096 --cache-mib 4096 --max-tokens 256 --stats
```

The first message uses the complete Qwen ChatML generation template. Later
messages append the previous assistant EOG token followed by the next
`user`/`assistant` turn, and continue from the engine's current position. If a
response reaches `--max-tokens`, the CLI closes that assistant turn with
`<|im_end|>` before accepting the next user message. `/reset` clears both the
native recurrent/KV state and the seeded sampler; `/quit` exits.

A real-GGUF smoke test entered two successive turns in one process and observed
the native engine position advance from 0 to 11 and then 26 before a clean
`/quit`. This verifies the interactive control/state path; it is not a quality
or throughput benchmark because each response was deliberately limited to one
token.

A longer run (70 template tokens of prompt, 96 generated tokens, 166 positions)
about cooking pasta at altitude is also token-identical to llama.cpp's greedy
decode; with a warm 8 GiB expert cache it reached a 95 % hit rate, 17 MiB of
SSD expert traffic per token and 24.4 tok/s.

## Performance work (all measured on the M4 Pro, greedy, 27 decode passes)

| change | decode step | generation tok/s |
|---|---:|---:|
| first working engine (per-expert dispatches, 1-thread rows, 3 command buffers/layer) | 245 ms | 4.7 |
| batched expert dispatches with SIMD-lane rows | 173 ms | 6.9 |
| SIMD-lane dense row kernels, threadgroup norms | 106 ms | 11.4 |
| threadgroup GQA kernel, shared expert folded into first buffer | 66 ms | 15.9 |
| concurrent miss loads, CPU-side residual | 59 ms | 18.6 |
| deferred expert buffer (one GPU sync per layer) | 50 ms | 21.6 |
| fused DeltaNet state / tail / L2 / attention prep kernels | 43 ms | 25.9 |

The rows above were measured with `--cache-mib 8192` on a warm page cache; the
27.8 tok/s figure with the default 4 GiB cache is in the final field run below.

Per-token GPU profile after these changes (`RL_ENGINE_PROFILE=1`, cumulative
over a 46-pass run divided by 46): experts 9.0 ms, DeltaNet projections 3.8 ms,
DeltaNet tail 2.0 ms, DeltaNet state 1.9 ms, shared expert 1.7 ms, router 1.3 ms,
attention projections 1.0 ms, LM head 0.75 ms, attention output 0.6 ms,
everything else < 0.5 ms; ~24 ms of GPU work per token, the remainder of the
43 ms step is expert miss loading (~8 ms wall, concurrent `pread` into the LRU
slots) and CPU/GPU synchronization (one wait per layer).

Expert cache behaviour (this prompt): 1 GiB cache → 56 % hit rate, 150 MiB SSD
traffic per token; 2 GiB → 70 %, 105 MiB; 4 GiB (default) → 79 %, 71 MiB;
8 GiB → 80 %, 67 MiB. The routed experts are never resident as a whole: the
~16.9 GiB of routed expert payload (`redlite-native inspect`) streams through
the bounded cache.

For reference, the pinned llama.cpp with the whole model resident in Metal
memory generates at 36–38 tok/s on this machine (`docs/FIELD_VALIDATION_M4PRO_24GB.md`).
Red Lite keeps only the ~1.06 GiB of dense weights plus the bounded expert cache
resident.

## Memory (M4 Pro, `--cache-mib 8192`, 48-token run)

- dense weights wrapped in place from the mmap: 1083 MiB (file-backed, no copy);
- per-backend state at `--context 4096`: 267 MiB (DeltaNet conv+recurrent 75 MiB,
  KV caches 192 MiB);
- expert cache: hard budget, 9446 slots of 0.9 MiB; 6387 slots resident after
  46 passes;
- process physical footprint / peak RSS: see the final field run.

## Scope boundary

- Prompt ingestion is token-by-token (correct, validated against llama.cpp's
  single-token decode); batched prefill is a later optimization.
- The Metal path supports the dense quant types present in this GGUF (F32,
  Q8_0, Q4_K, Q5_K, Q6_K, IQ2_XXS) and routed IQ2_XS/IQ1_M; the CPU oracle also
  decodes F16 and Q2_K.
- Sampling parity with llama.cpp is claimed for greedy only; temperature /
  top-k / top-p sampling follows the order of llama.cpp's default sampler chain
  (top-k, then top-p over the untempered distribution, then temperature) and is
  reproducible per seed, but the PRNG differs, so sampled outputs are not
  compared token by token.
- `--context` bounds the KV cache; the GQA kernel handles any length by chunked
  online softmax. The multi-chunk path (> 1024 keys) is validated against the
  pinned llama.cpp on the frozen 1200-token fixture
  `tests/fixtures/long_context_prompt.txt` (`logits.long_context_vs_llama`,
  positions 1100–1199 compared, argmax identical at every position). On that
  fixture llama.cpp's layer-2 router hits an exact probability tie at position
  1035 (experts 403 and 101, both 0.0077861) which the two implementations
  break differently; before the tie, and on a second 1200-token prompt without
  one, the drift at position 1199 is 1.5e-05 max abs / 1e-11 KL, after it the
  logits differ by up to 1.1 (KL ≤ 6e-3) with identical argmax. Long-context
  throughput has not been benchmarked.

## Final field run (clean build)

```bash
rm -rf .deps/redmetal && make native
.deps/redmetal/redlite-generate models/Qwen_Qwen3-Next-80B-A3B-Instruct-IQ2_XXS.gguf \
  --prompt "Explain in one sentence why the sky is blue." --max-tokens 64 --cache-mib 8192 --stats
```

Commit of the run: `280f7881629a04faa15683f4db6140a6e5c4f2a3` (documentation and
this record were committed on top; no code changed after the run).

```text
The sky appears blue because molecules in the Earth's atmosphere scatter shorter blue wavelengths of sunlight more than other colors due to Rayleigh scattering.
--- redlite-generate stats ---
model open           : 174.2 ms
prompt tokens        : 19 (1341.3 ms, 14.17 tok/s)
generated tokens     : 28 (1518.6 ms, 17.78 tok/s over 27 decode passes)
last step            : 63.9 ms (rec 23.2 attn 8.5 router 0.2 routed 30.4 [load 76.5 gpu 8.8] shared 0.0 out 1.5) dense GPU 14.4 ms
expert cache         : hits=10433 misses=2527 loads=2527 resident=6387/9446 slots (hit rate 80.5%)
SSD expert traffic   : 1795.7 MiB / 7581 reads during generation (66.51 MiB/token)
SSD expert total     : 4556.6 MiB / 19161 reads since open
peak RSS             : 4653.5 MiB
physical footprint   : 5940.7 MiB (process, incl. Metal buffers; mmap'd weights are file-backed)
```

Generated ids: `785 12884 7952 6303 1576 34615 304 279 9237 594 16566 44477 23327
6303 92859 315 39020 803 1091 1008 7987 4152 311 13255 62969 71816 13 151645`
(28 tokens, stop at `<|im_end|>`), identical to the pinned llama.cpp greedy decode.

The same prompt immediately afterwards:

| setting | prompt tok/s | generation tok/s | decode step | hit rate | SSD MiB/token | physical footprint |
|---|---:|---:|---:|---:|---:|---:|
| `--cache-mib 8192`, second run | 12.0 | 17.9 | 63.1 ms | 80.5 % | 66.5 | 5.8 GiB (5940 MiB) |
| `--cache-mib 4096` (default) | 16.3 | **27.8** | 38.7 ms | 79.2 % | 71.2 | 4.4 GiB (4485 MiB) |

The 8 GiB cache runs were slower only because their expert miss reads came from
the SSD instead of the macOS page cache (the concurrent `pread` sum was 72–77 ms
per step versus 14.5 ms); on a 24 GiB machine a larger explicit expert cache
competes with the page cache for the 18 GiB file. The default 4 GiB cache is
therefore the recommended setting for this model on 24 GiB. Full regression
suite after the clean build: 31/31 pass (`scripts/regress_m4.sh`, 68 s).
