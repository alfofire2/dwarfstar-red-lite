# Red Lite dev65 — long contexts (64K–256K), a harder agent suite, Qwen3-Coder-Next at 3 bits

Status: in progress. Measured on the Apple M4 Max 48 GiB and the Apple M4 Pro 24 GiB, 2026-10-06.

**A caveat on the M4 Max numbers of this night.** The M4 Max ran on a 25 W charger: its battery went from 80 %
(21:59) to 47 % (02:16) to 7 % (about 07:00) under GPU load, and macOS slows a Mac with a nearly empty battery.
- **Valid:** the comparisons made by alternating two settings in the same window, and every pass/fail result.
- **To be measured again on a proper charger:** the absolute speeds of the later runs.
Each table below says which kind it is.

## 1. Long contexts

The engine has no context limit of its own. Each attention layer has one K and one V buffer of
`2 KV heads × 256 × context × 4 bytes`: 512 MiB at 262,144 positions, 12 GiB for the 24 buffers.

### Does the model read the whole context? (needles)

`scripts/dev/needle_check.py` sends a prompt made of this repository's own C, Objective-C and Python code through
`redlite-server`, with three passphrases in code comments at 10 %, 50 % and 90 % of it, and asks for all three.
Red Lite CF2 (Qwen3-Coder-Next, 2 bits), every expert resident, M4 Max.

| Prompt tokens | `--context` | Passphrases found | Peak footprint | Swap |
|---:|---:|---|---:|---|
| 62,197 | 65,536 | **3 / 3** | 21.0 GiB | none |
| 127,390 | 131,072 | **3 / 3** | 24.0 GiB | none |
| 256,189 | 262,144 | **3 / 3** | 30.0 GiB | none |

Qwen3-Coder-Next uses its whole native context (262,144) in Red Lite: the three needles are found at every length.

**On the 24 GiB M4 Pro** (CF2, the default 4 GiB expert cache, experts from the SSD; battery steady at 79 %):

| Prompt tokens | Passphrases found | Ingestion | Decode |
|---:|---|---:|---:|
| 62,197 | **3 / 3** | 168.1 tok/s (6 min) | 15.1 tok/s |
| 127,390 | **3 / 3** | 95.5 tok/s (22 min) | 10.5 tok/s |

A 24 GiB Mac reads a 128K-token context too. The first ingestion is slow; a later turn reuses the state.

### Decode speed against position (M4 Max, valid)

`redlite-engine decode-bench MODEL --start-position N --fill-state FIFO` (new in dev65) loads the state of a
position-N context from random data, then times 16 decoded tokens. No prefill, so the GPU is not hot. CF2, every
expert resident.

| Position | ms / token | tok/s |
|---:|---:|---:|
| 4,096 | 11.50 | 87.0 |
| 65,536 | 21.67 | 46.2 |
| 131,072 | 32.77 | 30.5 |
| 262,000 | 64.31 | 15.6 |

The cost grows by about 0.2 ms per 1,000 positions, all of it decode attention. The kernel benchmark
(`kernel-bench`, 12 layers) reads the KV cache at 255–300 GB/s at every length. With 12 distinct KV buffer pairs,
as the engine has them, 256K costs 50 ms against 45 ms with one shared pair.

### Prompt ingestion and decode through the server (M4 Max, valid)

The needle runs again on a 90 W charger (battery 67–68 %, charging), with `attn_fa_b2`, through `redlite-server`:

| Prompt tokens | Passphrases | Ingestion | Decode | Request time | Peak footprint |
|---:|---|---:|---:|---:|---:|
| 62,197 | 3 / 3 | 517.9 tok/s | 41.2 tok/s | 2.0 min | 21.0 GiB |
| 127,183 | 3 / 3 | 319.5 tok/s | 26.3 tok/s | 6.7 min | 24.0 GiB |
| 256,155 | 3 / 3 | 182.1 tok/s | 16.0 tok/s | 23.5 min | 30.0 GiB |

- The 256K decode equals `decode-bench`'s 15.6 tok/s, so the slow numbers of the first run came from the battery.
- **The first run of the night** (25 W charger, falling battery) measured 370 / 176 / 44.9 tok/s ingestion and
  26.7 / 23.5 / 5.0 tok/s decode, with the dev64 prefill kernel. They are kept only as the record of that trap.

**A faster prefill attention kernel** (`attn_fa_b2`, now the default; `RL_PREFILL_FA2=0` restores `attn_fa_b`):
- **Tiles of 16 tokens instead of 8:** every key/value block serves twice the tokens.
- **The online softmax spread over the four simdgroups** (four rows each) instead of simdgroup 0 alone.
- **Bit-identical results:** every element is computed in the same order. The last-token dumps are byte-identical
  to `attn_fa_b` at 2,048, 32,768 and 65,536 tokens.

| Prompt | Prefill attention, `attn_fa_b` | `attn_fa_b2` | Change |
|---:|---:|---:|---:|
| 32,767 | 91.7 s | 73.4 s | −20 % |
| 65,535 | 541.1 s | 384.7 s | −29 % |

Measured with `redlite-engine logits --dump-from N-1 --batch 2048` (alternated runs; this tool commits every
prefill stage, so its absolute times are slower than the server's).

### Agreement with llama.cpp at long contexts (M4 Max)

The last 50 positions of the needle haystack's ids, native (batched prefill, every expert resident, CF2) against the
pinned llama.cpp. The oracle decodes token by token with nothing captured before `--dump-from`
(`--prefix-batch 1`). That form is byte-identical to the full token-by-token oracle (checked at 1,200 ids) and much
faster: 13 minutes at 32K instead of hours.

| Positions | Top token | Worst KL (limit 0.02) | Worst |Δlogit| (limit 2.0) | Oracle time |
|---:|---|---:|---:|---:|
| 16,384 (dev47) | 100 / 100 | 0.000079 | | |
| 32,768 | 50 / 50 | 0.00015 | 2.27 | 13 min |
| 65,536 | 50 / 50 | 0.0012 | 2.58 | 33 min |
| 131,072 | 50 / 50 | **0.025** | 3.77 | 100 min |

- **The top token agrees at every position checked**, up to 128K.
- **The distributions drift apart with length:** about tenfold per doubling past 64K. At 128K the KL is just above
  the limit used at short contexts, and the largest logit difference is above it from 32K on. The two programs sum
  attention over hundreds of thousands of positions in different orders.
- **Not a reference: llama.cpp's own batched path.** With the prefix decoded in batches of 256, llama.cpp
  disagreed with its own token-by-token output already at 1,200 ids (KL 0.094, top token different). Its batched
  Metal matrix kernels round activations to half precision, so it cannot serve as the oracle.

## 2. A harder agent suite

`agent_eval.py --suite hard`: six tasks on a copy of this repository.

| Task | What the agent must do | Graded by |
|---|---|---|
| `two_bugs` | find and fix two planted bugs in two modules | the planner tests pass, tests untouched |
| `rename` | rename a function in every file that defines, imports or calls it | no old name left; tests pass |
| `write_tests` | write tests for two functions of `quant_mix.py` | the tests pass, and fail on two planted mutants |
| `c_limit` | cap `max_tokens` at 65,536 in the C server with a 400 error, and add a test | the check compiles the server and sends both requests |
| `explain_kv_pad` | explain `RL_ENGINE_KV_PAD` | the answer names 32 and the tiled prefill |
| `debug_session` | three prompts: find why an upper-case variant fails, fix it, add an alias | resolver, alias and tests checked |

Every check was validated: it fails on the planted copy and passes with a reference solution.

| File, Mac | Passed | Per run |
|---|---:|---|
| Red Lite CF2, M4 Pro 24 GiB (temperature 0.3, 32K) | 12 / 18 | 4, 3, 5 |
| Bartowski's Qwen3-Coder-Next IQ2_XXS, same settings | 6 / 18 | 3, 1, 2 |
| Red Lite CF2, temperature 0.7 | 11 / 18 | 3, 4, 4 |
| **M4 Max 48 GiB**, 64K context, temperature 0.3, 900 s: | | |
| Bartowski's Qwen3-Coder-Next IQ3_XXS (31.7 GB) | **16 / 18** | 6, 6, 4 |
| Red Lite CF2 (19.3 GB) | **16 / 18** | 6, 5, 5 |

- **CF2 passes twice as many hard tasks** as Bartowski's file of the same size. The repository suite (dev63) showed the
  same order, 12 against 7 of 12.
- **Bartowski's file** looped in `debug_session` (358 and 203 requests) and twice wrote a test file without tests.
- **Neither** fixed both bugs of `two_bugs` in any run.
- **Temperature, with CF2:** 12 / 18 at 0.3 and 11 / 18 at 0.7, two timeouts each, one loop (48 requests) at 0.3 and
  none at 0.7. dev63's loops at 0.7 and above were measured with Bartowski's file; CF2 rarely loops at either
  temperature. `setup-pi` keeps 0.3, which costs CF2 nothing.

## 3. Qwen3-Coder-Next at 3 bits on the M4 Max

- **Repository suite** (dev62's settings): Bartowski's `Qwen3-Coder-Next-IQ3_XXS` passed **12 / 12**, as CF2.
- **Hard suite** (section 2): both passed **16 / 18** on the M4 Max at a 64K context.
  - **IQ3_XXS:** median 126 s per task, one timeout (225 requests).
  - **CF2:** median 112 s, one timeout (456 requests).
- **On this Mac, the 3-bit file does not pass more hard tasks than CF2.** CF2 is 12.4 GB smaller, which leaves room
  for a 256K context with every expert resident.
- **The context window decides it.** The same CF2 on the same M4 Max with a 32K window passed **10 / 18** (4, 2, 4),
  against 16 / 18 with 64K.
  - The failures were `debug_session` (it forgot its own earlier fix), `two_bugs` and `c_limit`.
  - Prompts in these sessions reach 10–30K tokens per request, so a 32K window fills mid-task and pi drops or
    summarizes earlier turns.
  - The M4 Pro's 12 / 18 was at 32K too.
- **Default changed:** `redlite setup-pi` writes a 64K window (`--context 65536`), and the README's agent commands use
  64K. On a 24 GiB Mac a 64K window no longer fits next to every expert, so the server streams experts from the SSD.
  The hard suite in that setting is being measured on the M4 Pro.

## 4. Smaller findings

- **No MTP for the Coder:** `Qwen/Qwen3-Coder-Next` ships no `mtp.*` tensors (Qwen3-Next-80B-A3B-Instruct has 1,553).
- **The pi note** (`--append-system-prompt`: project files are relative to the working directory, not pi's install
  folder): `explain_stop` passed 11 / 12 with it and 10 / 12 without (M4 Pro, temperatures 0.7 and 0.3). It is not
  needed, so `setup-pi` does not write it.
- **`tokenize --text` printed at most 16,384 ids without saying so.** It now warns.
- **A decode attention variant** (one simdgroup per position for the scores) was 6–7 % faster at blocks of 128 with one
  shared buffer, but 34 % slower at 256K with distinct buffers. Not kept.

## Scope boundary

- **M4 Max speeds of this night:** see the caveat at the top. The needle results, the decode-bench table and the
  alternated A/B comparisons are valid.
- **Parity with llama.cpp beyond 16K positions:** the oracle now ingests the prefix in batches (`--prefix-batch`),
  but the 32K–256K comparison has not been run yet.
- **Coder IQ3_XXS at 128K/256K** with a partial expert cache, and on the hard suite: not run yet.
