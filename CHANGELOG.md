# Changelog

## 0.9.2 — 2026-10-08

**Faster decode: +15 % on the M4 Max and +12–18 % on the 24 GiB M4 Pro with Qwen3-Coder-Next CF2.**
- **Four decode kernels rewritten after profiling a real token** in Xcode (`RL_GPU_CAPTURE_STEP`):
  - the Q4_K projections read their quants as 32-bit words, one item per pair of sub-blocks (−16 to −20 % per
    projection). Q4_K is 864 MiB of the weights CF2 and F2 read per token;
  - the IQ1_M expert decoder reads each shared `qh` byte and scale once for two groups;
  - the DeltaNet state update runs one SIMD group per state row, with no threadgroup barrier;
  - the Q5_K output head uses the Q4_K layout (−14 %);
  - the shared expert's last two steps run beside the routed experts, and the DeltaNet convolution window is shifted
    in place (same kernels, identical output).
- **Decode, Qwen3-Coder-Next CF2:**
  - M4 Max, every expert resident: 85.3 → 97.8 tok/s at short context, 66.7 → 74.1 at 32K. That is slightly faster
    than MLX's 3-bit file (94–96 tok/s), which is 14.5 GiB larger;
  - M4 Pro 24 GiB: 33.7 → 37.9 tok/s with the 4 GiB expert cache, 48.2 → 57.1 with every expert resident;
  - Bartowski's IQ2_XXS (fewer Q4_K weights), M4 Max: 83.5 → 90.9 tok/s, and 53.0 → 56.0 with the 4 GiB cache.
  - a file rewrite with prompt lookup, M4 Max: 108.7 → 117.2 tok/s with every expert resident, with output identical
    to 0.9.1.
- **Prompt ingestion is unchanged.**
- **Half-precision KV cache, opt-in** (`RL_KV_F16=1`): +9 % decode at 32K, +13 % at 64K, and half the cache memory.
  With it, MTP plus a 32K context fits a 24 GiB Mac with every expert resident. It is not the default because on CF2
  it moved the 1200-token comparison with llama.cpp past its bounds (KL 5.8e-2).
- New dev tools: `decode-expert-bench`, `prefill-gemm-bench`, `agent_eval.py --repo-ref`, more `kernel-bench` shapes.
- Thirteen kernel attempts that did not pay are in WHAT_DID_NOT_WORK, with their measurements.
- Validation:
  - `regress_m4.sh` on the reference IQ2_XXS file and on CF2;
  - `quick_parity.sh --long`;
  - local CI.


- dev74: **decode kernels, half KV opt-in** (`docs/REDLITE_DEV74_DECODE_KERNELS.md`).

## 0.9.1 — 2026-10-08

**Prompt lookup on 24 GiB Macs too.**
- **The 2-row verify behind MTP and prompt lookup no longer needs every expert resident.** With a bounded expert cache
  it loads the union of both rows' experts in one step, with prefetch.
- **`redlite serve` and `redlite chat` use prompt lookup with a bounded cache too**, with exact routing instead of
  cache-aware routing. On the M4 Pro 24 GiB with the 4 GiB cache and a 64K window, coding-agent decode was 7–10 %
  faster than the old default, with exact answers. Rewriting a file there went from 31.9 to 37.8 tok/s.
- **Fixed on the way:** the expert pool wrote its slot and weight tables into shared buffers at encode time. Two
  encodes in flight would have read each other's tables; plain decoding never had two. The tables now go into the
  command buffer.
- **Measured against MLX** on the M4 Max (FINDINGS section 3).
  - MLX's 3-bit Coder decodes about 20 % faster and ingests prompts about 50 % faster, but needs 32.5 GiB.
  - MLX's 2-bit Coder corrupted a file it was asked to copy.
- **The drift from llama.cpp past 64K is numerics, not a bug:** per-layer dumps were checked at 32K and 64K.
- **GPU profiling of the kernels with Xcode** (`prefill-attn-bench`, `prefill-expert-bench`, `RL_GPU_CAPTURE`). First
  profile: the prefill attention is float32-bound, at 14 % occupancy with 166 registers per thread.
- Validation:
  - `regress_m4.sh` 59 / 0 / 0, with the new `engine.verify_bounded` and `engine.verify_full`;
  - `quick_parity.sh --long` 5 / 5;
  - local CI.


- dev72: **bounded-cache verify, MLX, the llama.cpp drift, GPU profiling**
  (`docs/REDLITE_DEV72_BOUNDED_VERIFY_PROFILING.md`).

## 0.9.0 — 2026-10-07

**Faster coding agents: prompt lookup speculative decoding.**
- **`redlite serve` drafts tokens from the conversation** and verifies each one exactly in the 2-row pass MTP uses. A
  draft is the token that followed the latest earlier occurrence of the context's last three tokens.
  - The output does not change, at any temperature.
  - On by default with every expert resident and no MTP head, which is Qwen3-Coder-Next's case. `--no-lookup`
    turns it off.
- **Coding-agent sessions on the M4 Max (CF2):** decode 15–25 % faster at every context length in two sets of
  runs, with 73–85 % of the drafts accepted.
- **24 GiB M4 Pro** with every expert resident (raised GPU limit, 32K): agent decode 28–29 % faster.
- **`redlite chat` does it too** (`redlite-generate --lookup`): a turn that rewrites a function went from 79 to
  110 tok/s.
- **Rewriting a file:** 80 → 109 tok/s on the M4 Max, 45 → 60 on the M4 Pro 24 GiB. Prose is unchanged, because few
  drafts are proposed there.
- Validation:
  - `regress_m4.sh` 57 / 0 / 0, with the new `server.lookup_greedy` and `generate.lookup_greedy`;
  - `quick_parity.sh --long` 5 / 5;
  - local CI.


- dev70: **prompt lookup** (`docs/REDLITE_DEV70_PROMPT_LOOKUP.md`).
  - **Code:**
    - `rl_lookup_draft`, with self-test cases;
    - `redlite-server --lookup`;
    - `redlite serve` default and `--no-lookup`;
    - `server_check.py --lookup`;
    - `scripts/dev/server_log_decode.py`.
  - **Finding:** a half-precision micro-benchmark shows half and float matrix products at the same rate on the M4 Max.
- dev71: **prompt lookup in `redlite-generate`/`redlite chat`, and three things not kept.**
  - **Added:** `RL_SERVER_TRACE` and `scripts/dev/lookup_sim.py`.
  - **Not built:** several drafts per verify, which simulates to at most +5–14 %.
  - **Not kept:**
    - an 8-simdgroup prefill attention kernel (slower);
    - the 3-bit Coder with lookup (15 / 18, decode 13 % slower than CF2).

## 0.8.2 — 2026-10-07

**Decode attention closer to the memory bandwidth.**
- **Faster decode at 64K–256K tokens:** +4 % at 64K, +13 % at 128K and +11 % at 256K on the M4 Max (`decode-bench`,
  CF2). The decode attention now reads the KV cache at about 480 GB/s.
- **How:**
  - the score pass reads each key row in whole 128-byte lines, eight threads per row;
  - the merge of the per-block results finds its maximum with every thread, and keeps four loads in flight.
- Validation:
  - `regress_m4.sh` 55 / 0 / 0 and `quick_parity.sh --long` 5 / 5 on the reference file (M4 Max);
  - needles at 62K and 127K;
  - local CI.


- dev67: **decode attention, score pass and merge** (`docs/REDLITE_DEV67_DECODE_ATTENTION_MERGE.md`).
  - **Kernel:** `attn_gqa_merge` 4.2 → 1.3 ms at 256K, bit-identical. `kernel-bench` −13 / −14 % at 64K–256K
    (M4 Max), −8 to −16 % on the M4 Pro.
  - **RoPE probe:** RoPE precision is not the cause of the KL drift from llama.cpp past 64K.

## 0.8.1 — 2026-10-07

**Faster decode at long contexts.**
- **Decode attention:** the value pass now keeps four loads in flight (same summation order, bit-identical), and
  from 12K positions the kernel uses 256-position blocks. The KV cache is read at about 415 GB/s instead of 290 on the
  M4 Max.
- **Decode through the server, M4 Max, CF2:** 41 → 48 tok/s at 62K tokens, 26 → 33 at 127K, 16 → 22 at 256K, with
  the same answers. On the 24 GiB M4 Pro, decode attention is about a third faster (`kernel-bench`).
- Validation:
  - `regress_m4.sh` 55 / 0 / 0 and `quick_parity.sh --long` 5 / 5 on the reference file (M4 Max);
  - needles at 62K, 127K and 256K;
  - local CI.


- dev66: **decode attention at long contexts** (`docs/REDLITE_DEV66_DECODE_ATTENTION.md`).
  - **Kept:**
    - four value loads in flight in `attn_gqa_split_g`;
    - `rl_attn_auto_blk()` (256-position blocks from 12,288 positions);
    - the kernel self-test now covers 256-position grouped blocks;
    - `kernel-bench` adds 1K–16K.
  - **Not kept** (WHAT_DID_NOT_WORK):
    - two prefill attention kernels (GQA-shared tiles, Q from device memory), byte-identical and slower;
    - two decode variants.

## 0.8.0 — 2026-10-06

**The model's full 262K context, and a 64K window for coding agents.**
- **Long contexts, measured:** a prompt of this repository's code with three hidden passphrases is searched
  correctly at 62K, 127K and 256K tokens on the M4 Max (ingestion 518 / 320 / 182 tok/s, decode 41 / 26 / 16), and at
  62K and 127K on a 24 GiB M4 Pro with experts streamed from the SSD.
- **Faster long prompts:** a prefill attention kernel with 16-token tiles and the softmax on every simdgroup,
  bit-identical to the old one, cuts prefill attention by 20 % at 32K and 29 % at 64K.
- **A 64K agent window by default:** on the M4 Max CF2 passed 16 of 18 harder agent tasks with 64K and 10 of 18
  with 32K. `redlite setup-pi` writes 64K. With 64K, CF2 matches Bartowski's 3-bit Coder (16 / 18) at 12 GB less.
- **Agreement with llama.cpp up to 128K:** the top token agrees at every position checked. The KL drifts to 0.025
  at 128K.
- Validation:
  - local CI;
  - `regress_m4.sh` 55 / 0 / 0 and `quick_parity.sh --long` 5 / 5 on the reference file (M4 Max);
  - agent runs and needle runs on both Macs.


- dev65: **long contexts, a harder agent suite, the context window.**
  - **Tools:**
    - `scripts/dev/needle_check.py`;
    - `redlite-engine decode-bench` (`--start-position`, `--fill-state`);
    - a decode-attention section in `kernel-bench`;
    - `agent_eval.py --suite hard` (six tasks, each validated), `--temperature`, `--append-system`;
    - oracle `--prefix-batch` and `--tokens @FILE`.
  - **Prefill:** `attn_fa_b2` by default (`RL_PREFILL_FA2=0` for A/B).
  - **Agents:** hard suite on both Macs, 16 / 18 for CF2 and IQ3_XXS at 64K, CF2 10 / 18 at 32K. On the M4 Pro: CF2
    12 / 18 at 32K and at 64K, Bartowski's IQ2_XXS 6 / 18.
  - **Fixes:** `tokenize --text` warns when it truncates.
  - **Traps** (WHAT_DID_NOT_WORK):
    - a 25 W charger that slowed the night's runs;
    - llama.cpp's batched path as an oracle;
    - a decode attention variant.
  - `docs/REDLITE_DEV65_LONG_CONTEXT_AGENTS.md`, FINDINGS sections 6 and 9 with two charts.

## 0.7.0 — 2026-10-06

**A better Qwen3-Coder file, an agent temperature that avoids loops, and a better option for 48 GiB Macs.**
- `redlite download coder` fetches **Red Lite CF2**, Qwen3-Coder-Next at the size of Bartowski's IQ2_XXS (19.32 GB):
  code perplexity −3.3 %, 12 of 12 harder agent tasks on the M4 Pro 24 GiB (Bartowski's file 7) and on the M4 Max
  (10). `bartowski-coder` keeps the previous file.
- `redlite setup-pi` sets **temperature 0.3** for the agent: the 2-bit Coder looped 4 times in 36 sessions at 0.7
  and above, never at 0.3.
- The server honours **`presence_penalty` and `frequency_penalty`** (OpenAI semantics), with server-wide defaults.
- `redlite download 48gb-g2` fetches **Red Lite G2** (31.24 GiB) for 48 GiB Macs: perplexity −0.65 % on text and
  −1.4 % on code against IQ3_XXS, the same decode speed with MTP. `redlite chat` prefers it when present. `48gb`
  stays IQ3_XXS, because G2 fills the Mac with 25K-token prompts.
- At the default GPU limit the planner's 70 %-of-RAM rule counts the context's KV cache; `chat` and `serve` turn
  MTP off when it does not fit at the chosen context.
- **The project site** has the documentation and a search: https://redlite.alfonsodaniello.it/.
- Validation:
  - local CI;
  - `regress_m4.sh` on G2 (39/3/13, the three failures explained), `quick_parity.sh` 4/4, `server_check.py` with
    MTP (M4 Max);
  - agent runs on both Macs.


- dev64: **a better file for 48 GiB Macs.**
  - Variants of Bartowski's IQ3_XXS from the same Q8_0 source (`quant_mix.py` dev63 options):
    - G1, Q8_0 dense, −0.4 %;
    - G2, IQ3_S experts, −0.65 % text / −1.4 % code;
    - G3, both, −1.0 % / −1.7 %, but above the 70 % rule with MTP.
  - **G2:** decode with MTP 90.7 tok/s against 89.3 for IQ3_XXS (M4 Max, six prompts). It passes the parity checks;
    its two long-context argmax swaps are near-ties.
  - **Long prompts:** with 25K tokens G2's prompt ingestion varied from 401 to 676 tok/s with the Mac swapping
    (IQ3_XXS 596–638).
  - **Planner:** `native_full_residency_fits` counts the KV cache, and `chat` / `serve` print "MTP off" when the head
    does not fit at the context.
  - `docs/REDLITE_DEV64_48GB.md`, FINDINGS section 10 with a chart.
- dev63: **penalties, agent loops against sampling, a Red Lite quantization of Qwen3-Coder-Next.**
  - **Penalties:** `presence_penalty` / `frequency_penalty` in the sampler (over the generated tokens) and the server
    (request fields, `--presence-penalty`, `--frequency-penalty`).
  - **Coder quantization:** CE3 and CF2, from Bartowski's Q8_0 with F2's and E3's recipes. They were judged by
    perplexity on text and on a new frozen code corpus (`perplexity.sh --corpus`), then by the agent suite on both
    Macs. **CF2** is published at `alfodaniello/Qwen3-Coder-Next-RedLite-GGUF`.
  - **Loops:** four sampling settings × 12 tasks on the M4 Pro. Pass rates 7–9 of 12 (noise); loops 4 in 36
    sessions at temperature ≥ 0.7, none at 0.3.
  - **A trap:** the first run of that study measured two tests that read the machine's GPU limit (raised on the
    M4 Pro), so two tasks could not pass there. The tests now patch it (`WHAT_DID_NOT_WORK.md`).
  - **Also:**
    - `quant_mix.py --keep-experts / --set / --dense-map / --base-type / --print-types`;
    - a load-proof two-worker server test;
    - `scripts/dev/build_site.py` for the site's docs and search.
  - `docs/REDLITE_DEV63_CODER_QUANT_SAMPLING.md`, FINDINGS section 9 with two charts.

## 0.6.1 — 2026-10-05

**Agents on a 24 GiB Mac without `sudo`, and sturdier tool calls.**
- `redlite setup-pi` configures the pi coding agent.
- At the default GPU limit (experts from the SSD) agents pass as with everything resident, 20–50 % slower.
- The 48 GiB IQ3_XXS file streams on 24 GiB at 29 tok/s.
- **The server:**
  - accepts long agent sessions (up to 16,384 messages; 256 before);
  - tolerates a frequent slip of the 2-bit Qwen3-Coder in its XML calls;
  - logs calls it cannot parse.
- Harder agent tasks on this repository: Qwen3-Coder-Next 70 %, F2 50 %, the full-precision model 83 %.
- Validation:
  - local and hosted CI;
  - protocol tests under ASan+UBSan;
  - real-model agent runs on the M4 Max and the M4 Pro.


- dev62: **five follow-up tests.**
  - **Cache-aware routing** costs +0.19 % (IQ3_XXS) and +0.06 % (F2) engine perplexity with the 4 GiB cache.
  - **Full-precision reference:** Qwen's API model passes the five dev60 tasks 15/15, so they are too easy.
  - **Harder tasks** on a copy of this repository (`agent_eval.py --suite repo`): API 10/12, Qwen3-Coder-Next
    14/20, F2 6/12.
  - **IQ3_XXS with the agent** on the 24 GiB Mac from the SSD: 15/15, but twice F2's time.
  - **`--parallel 2` with two agents:** 2–5 %.
  - **Server fixes found by the runs:**
    - chat messages on the heap, up to 16,384 (256 were hit by an agent session);
    - XML calls with `<KEY>` for `<parameter=KEY>` accepted (2-bit Qwen3-Coder);
    - unparsed calls logged.
  - `docs/REDLITE_DEV62_AGENT_TESTS.md`, FINDINGS section 9 with a chart.
- dev61: **the 24 GiB Mac at the default GPU limit** (bounded 4 GiB expert cache, experts streamed from the SSD).
  M4 Pro:
  - **Coding agent:** pi installed on the same Mac; F2 and Qwen3-Coder-Next pass 15/15 each, 20–50 % slower than
    with every expert resident.
  - **IQ3_XXS** (29.6 GiB, perplexity 14.29) streams at 29.3 tok/s decode against 34.0 for F2, and ingests
    208–334 tok/s.
  - **Cache size:** 8 and 12 GiB caches cut the reads but not the time; 4 GiB stays.
  - **`redlite setup-pi`:** writes pi's provider for `redlite serve --native`, keeps other providers and the
    file's permissions (new file 0600).
  - `docs/REDLITE_DEV61_STREAMING_24GB.md`, FINDINGS section 5 with a chart.

## 0.6.0 — 2026-10-04

**Coding agents on a local 80B model.**
- `redlite-server` speaks OpenAI tool calling, rendered exactly as the model's own chat template: Qwen3-Next JSON
  or Qwen3-Coder XML.
- Agent loops keep the engine state between turns.
- With the pi coding agent, 5 scripted tasks × 3 runs:
  - Qwen3-Coder-Next IQ2_XXS 15/15 (M4 Max) and 14/15 (M4 Pro 24 GiB);
  - Red Lite F2 15/15 on the M4 Pro 24 GiB.
- The README has a ready pi configuration. `redlite serve --native` picks the model like `redlite chat`.
- Hosted CI is back (Linux and macOS).
- Validation:
  - local CI and hosted CI PASS;
  - protocol tests under ASan+UBSan;
  - real-model tool round trips with F2, Bartowski's IQ2_XXS and Qwen3-Coder-Next (M4 Max);
  - agent runs on both Macs.


- dev59: **OpenAI tool calling in `redlite-server`.** It accepts `tools` (with `tool_choice`), assistant
  `tool_calls` and `tool` messages, and renders them as the model's own chat template does: Qwen3-Next JSON or
  Qwen3-Coder XML, chosen from the GGUF.
  - The calls the model writes come back as `tool_calls` with `finish_reason: "tool_calls"`, streaming included.
  - Prompts are byte-identical to the Jinja templates (tests against both).
  - An agent's next turn reuses the engine state: 203 of 230 prompt tokens with F2, 350 of 377 with
    Qwen3-Coder-Next.
  - New regress check `server.tool_call`. `docs/REDLITE_DEV59_TOOL_CALLS.md`.
- dev59b/dev60: **a coding agent on the native server.** Measured with the pi agent and
  `scripts/dev/agent_eval.py`, five scripted tasks, three runs each:
  - Qwen3-Coder-Next IQ2_XXS 15/15 on the M4 Max and 14/15 on the M4 Pro;
  - Red Lite F2 15/15 on the M4 Pro 24 GiB (6–29 s per task) and 14/15 on the M4 Max.
  - **Fixes found by the measurement:**
    - call arguments are rendered in `json.dumps` form, because agents send them back re-serialized;
    - the content before a call keeps all but the template's one separator newline;
    - with these two, agent turns reuse the engine state (F2 on the M4 Max: 0 % → 73 % median);
    - JSON calls with wrong closing brackets are repaired (F2 11 → 14 of 15).
  - `RL_SERVER_DEBUG_REUSE=1` logs where a prompt diverges from the held state.
  - `docs/REDLITE_DEV60_CODING_AGENT.md`, README *Use with a coding agent*.
- ci: hosted CI restored now that the repository is public (Linux as before, plus macOS running
  `scripts/dev/local_ci.sh`). It caught unused parameters in the non-Metal branches of `rl_engine_verify2` and
  `rl_engine_mtp_draft`, now fixed.

## 0.5.7 — 2026-10-04

**Fix:** in 0.5.6 `redlite chat` and `serve` did not see the F2 file on the full-residency path (a Mac with 40 GiB or
more, or a raised GPU limit). With only F2 in the models folder they answered "No native model found"; with E3 there
too, they picked E3. One preference list now serves both paths (`NATIVE_SMALL_MODELS` follows the 48 GiB file), with a
test of the raised-limit path. Checked on the M4 Pro: F2, every expert resident, MTP.

## 0.5.6 — 2026-10-04

**A better 24 GiB file: Red Lite F2.** Same size as Bartowski's IQ2_XXS (19.32 GB), perplexity 15.38 against 16.22
for E3 and 16.37 for Bartowski's scheme. `redlite download 24gb` fetches it and `redlite chat` prefers it. Plain
decode is 3 % slower than E3; with MTP on a 24 GiB Mac it is as fast or faster. Validation: `regress_m4.sh` 41/0/13
and `quick_parity.sh` on the M4 Max; token-identical greedy output on the M4 Pro.


- dev58: **dense weights with more precision, at the same size.** `quant_mix.py --dense-type` raises the 2-bit dense
  projections every token reads (about 290 MiB):
  - F1: IQ3_XXS, IQ2_XS experts on layers 38–47, 19.33 GB;
  - F2: Q4_K, IQ2_XS experts on layers 40–47, 19.32 GB.
  - **Perplexity** 15.42 / 15.38 against E3's 16.22 (paired t = −16 / −15).
  - **API agreement:** F2 5 identical answers, 15.2 % mean matching words; E3 4, 13.6 %.
  - **Speed:** plain decode −1 % / −3 %; with MTP on the M4 Pro as fast or faster.
  - F2 passes `regress_m4.sh` 41/0/13 and `quick_parity.sh` (M4 Max). It is on Hugging Face, and
    `redlite download 24gb` now fetches it (`e3` the previous file). `docs/REDLITE_DEV58_DENSE_PRECISION.md`.

## 0.5.5 — 2026-10-04

**Two requests at once, Homebrew, the GPU limit at boot.**
- `redlite serve --native --parallel 2` decodes two requests in one pass with every expert resident (M4 Pro: total
  throughput +28 %, +10 % against MTP; answers unchanged).
- Homebrew: `brew tap alfofire2/redlite https://github.com/alfofire2/dwarfstar-red-lite && brew install redlite`
  (binaries and CLI from this release's tarball, models in `~/.redlite/models`).
- A LaunchDaemon recipe sets the GPU limit at every boot; `redlite doctor` recognizes it.
- Known issue: on macOS 27.0.1 the GPU driver kernel-panicked twice on an M4 Max under Red Lite's Metal work (see
  `docs/WHAT_DID_NOT_WORK.md`).
- Validation: local CI PASS; `regress_m4.sh` IQ2_XXS 54/0/0 (M4 Max), E3 49/0/1 (M4 Pro).


- dev56: **two server requests at once.** `redlite-server --parallel 2` (and `redlite serve --native --parallel 2`)
  decodes two requests in one pass over the weights, with every expert resident.
  - **Engine:** slots (`rl_engine_slots_enable` / `rl_engine_select_slot`) and `rl_engine_step_pair`, the 2-row
    verify with row 1 on the other slot's state.
  - **Server core:** N worker threads.
  - **M4 Pro 24 GiB, E3 file:** total throughput 43.5 → 55.9 tok/s without MTP, 51.0 → 56.0 against MTP; answers
    identical to serving them in turn.
  - **Checks:** `redlite-engine pair` (max logit difference 1.2e-5, no argmax mismatch); `server_check.py --parallel`;
    both in `regress_m4.sh` (`engine.pair`, `server.parallel_greedy`).
  - `docs/REDLITE_DEV56_PARALLEL_SERVER.md`.

- dev57: **Homebrew.** The repository is its own tap: `brew tap alfofire2/redlite https://github.com/alfofire2/dwarfstar-red-lite`,
  then `brew install redlite`.
  - The formula (`Formula/redlite.rb`, written by `scripts/dev/brew_formula.py`) installs the three binaries and
    the `redlite` CLI from the release tarball, plus `hf` for downloads.
  - The tarball now carries `python/redlite` (stdlib only).
  - Outside a source checkout, the CLI finds the binaries on PATH and keeps models in `~/.redlite/models`
    (`REDLITE_MODELS` overrides).
  - Checked: install steps simulated in a scratch prefix on the M4 Max (no models → a message pointing to
    `redlite download`; with models → the right binary and file). A real `brew install` is not tested yet; it needs
    the repository and its releases to be public.

- dev55b: the GPU limit at every boot, with a LaunchDaemon (recipe in `docs/REDLITE_DEV51_24GB_DECODE.md`). Verified on the
  M4 Pro 24 GiB: it sets 21,741 MiB, and Metal reports 21.23 GiB. Persistence across a restart is not verified yet.
  `redlite doctor` reports a limit set at boot and points to the recipe.

## 0.5.4 — 2026-10-04

**Long contexts on a 24 GiB Mac with every expert resident.**
- `redlite chat` / `serve` size the prefill chunk and MTP to the raised GPU limit, so a 32K context no longer runs out
  of GPU memory.
- MTP is skipped for answers that start past 8,192 positions on Macs below 40 GiB, where it stops paying.
- Steering in the server.
- Validation: local CI PASS; `regress_m4.sh` IQ2_XXS 52/0/0 (M4 Max); end-to-end planner runs on the M4 Pro 24 GiB.


### dev55 — long contexts on a 24 GiB Mac with every expert resident (M4 Pro 24 GiB, M4 Max 48 GiB)

- **A measured GPU-need model** (context, prefill chunk, MTP). Under a raised GPU limit, `redlite chat` / `serve`
  pick the chunk and MTP that fit. M4 Pro at 21,741 MiB: 4K → 2048 + MTP, 16K → 512 + MTP, 32K → 2048 without MTP.
  A 32K context with a 16.8K-token prompt now runs (it ran out of GPU memory with MTP).
- **MTP and long contexts.** On the M4 Pro MTP stops paying at about 11K positions (+24 % at 20 tokens, 0 % at 11.2K,
  −12 % at 16.8K); the M4 Max still gains at 16.8K. `--mtp-max-context N` (generate, server); `redlite chat` /
  `serve` pass 8192 below 40 GiB.
- **Steering in `redlite-server`** and `redlite serve --native` (`--steer*`).
  See `docs/REDLITE_DEV55_LONG_CONTEXT_24GB.md`.

## 0.5.3 — 2026-10-04

**A better 24 GiB file.** The Red Lite E3 expert mix: the same 19.30 GB and tensor types as Bartowski's IQ2_XXS, with
perplexity 16.370 → 16.216 (paired t = −3.9) and more answers identical to Qwen's API (4 vs 2 of 235).
- Published on Hugging Face (`alfodaniello/Qwen3-Next-80B-A3B-Instruct-RedLite-GGUF`); `redlite download 24gb` fetches
  it, and `redlite chat` prefers it. Bartowski's file stays available as `bartowski-24gb`.
- Validation:
  - E3 on the M4 Max: `regress_m4.sh` 52/0/0 and `quick_parity.sh` PASS;
  - on the M4 Pro 24 GiB: download (SHA-256 checked), automatic choice by `redlite chat`, a correct answer;
  - local CI PASS.


### dev54 — a better expert mix at the same size (M4 Max 48 GiB)

- From Bartowski's Q8_0 and imatrix, with every non-expert tensor typed as in his IQ2_XXS, the 11 IQ2_XS expert
  layers were moved to 37–47 (the highest expert input energy in the imatrix) instead of 0–5 and 43–47.
- **Same size (19.30 GB):** perplexity 16.370 → 16.216 (paired t = −3.91, better on 72/111 chunks); against Qwen's API
  4 vs 2 identical answers, 13.6 vs 12.6 % mean words matching.
- Passes `regress_m4.sh` 52/0/0 and `quick_parity.sh`.
- `scripts/dev/quant_mix.py` reproduces it. Not distributed yet. See `docs/REDLITE_DEV54_QUANT_MIX.md`.

## 0.5.2 — 2026-10-03

Steering, prefilled conversations, and a faster bounded cache by default.

- **Activation steering** (dev52): `redlite chat --steer FILE --steer-layers A-B --steer-strength S`, `/steer S` in the
  chat, `--steer-tokens N` to steer only the start of an answer, and `scripts/dev/steer_extract.py` to build vectors
  from two prompt sets. Exact with MTP. No pre-made vectors are shipped.
- **`--history FILE`** (dev52b): a prefilled conversation of `user:` / `assistant:` turns, read before the first
  message.
- **Cache-aware routing by default with a bounded cache** (dev51c): +5 % decode on the M4 Pro 24 GiB; quality vs
  Qwen's API unchanged. `--exact-routing` turns it off.
- **Measured, not shipped:**
  - low-power mode costs 22 % of decode at full residency on the M4 Pro;
  - a float16 KV cache halves the context memory but broke CPU/Metal router parity, so it was reverted (dev53,
    `docs/WHAT_DID_NOT_WORK.md`).
- **Validation (M4 Max 48 GiB):** local CI PASS; `regress_m4.sh` IQ2_XXS 52/52 (new: `engine.parity.steered`), IQ3_XXS
  39/0/13; `quick_parity.sh` PASS on both GGUFs; 0 warnings.


### dev51c — cache-aware routing by default with a bounded cache (M4 Pro 24 GiB)

- **Measured against Qwen's API** (235 prompts, 4 GiB cache): exact routing 2 identical / 7.8 % median words
  matching; λ = 0.5 1 / 8.1 %. No measurable difference; perplexity was already unchanged (dev46).
- **Default:** `redlite chat` / `serve --native` set `RL_ROUTE_CACHE_BIAS=0.5` with a bounded cache, for +5 % decode on
  the M4 Pro. `--exact-routing` turns it off. Binaries stay exact by default.

### dev52 — activation steering (M4 Max 48 GiB)

- **Mechanism:** `strength × vector` is added to the residual stream at the input of chosen layers, on every decode
  path: bounded cache, GPU-routed, the 2-row MTP verify, and the CPU oracle.
  - The batched prefill is not steered; the last prompt token takes a steered step instead.
  - `--steer-tokens N` steers only the start of each answer.
- **Use:**
  - `redlite chat --steer FILE --steer-layers A-B --steer-strength S`, and `/steer S` during the chat;
  - `redlite-generate` takes the same options;
  - `scripts/dev/steer_extract.py` builds a vector from two prompt sets (difference of means).
- **Validation:**
  - CPU vs Metal parity with steering on: YES on the bounded and GPU-routed paths (new regress check
    `engine.parity.steered`);
  - steered answers identical with and without MTP;
  - unchanged output with steering off.
- No pre-made vectors are shipped. See `docs/REDLITE_DEV52_STEERING.md`.
- dev52b: `--history FILE`, a prefilled conversation (`user:` / `assistant:` turns) read before the first message,
  in `redlite chat` and `redlite-generate`.

## 0.5.1 — 2026-10-03

24 GiB Macs can now hold every expert. With the GPU limit raised, the M4 Pro 24 GiB decodes at 46.0 tok/s, and
52.7 with MTP (0.5.0: 32.5). `redlite doctor` prints the command, and `redlite chat` / `serve --native` switch on
their own. The native binaries print the right version. Validation: local CI PASS; `regress_m4.sh --quick` 49/49
(M4 Max); end-to-end `redlite doctor` / `chat --dry-run` on the M4 Pro with the default and with a raised limit.

### dev51 — every expert resident on a 24 GiB Mac (M4 Pro 24 GiB)

- **Cache size sweep, 4–14 GiB:** misses fall from 29.5 to 12.6 per token, but decode stays at 31–32 tok/s. The
  bounded path is limited by its per-layer GPU round trip; the 4 GiB default stays.
- **With `sudo sysctl iogpu.wired_limit_mb` raised** (until reboot), every IQ2_XXS expert is resident:
  - decode 32.5 → **46.0 tok/s** (six-prompt median), every token GPU-routed;
  - with MTP **52.7 tok/s** (up to 55.7), identical output;
  - footprint 18–20 GiB; ~1.3 GB of other processes go to swap and stay steady.
- `redlite chat` / `serve --native` use full residency (and MTP) on any Mac whose GPU limit fits experts + dense +
  1 GiB (+ the MTP head). `redlite doctor` prints the needed limit and the command.
- Native binaries print the version from `VERSION` (0.5.0 still printed 0.4.0).
- `regress_m4.sh`: llama.cpp oracle out of GPU memory → SKIP with the reason; `REDLITE_REF_NGL` for the oracle.
  See `docs/REDLITE_DEV51_24GB_DECODE.md`.

## 0.5.0 — 2026-10-03

Faster answers with identical output, sessions that survive restarts, a second model family, and the first native
numbers from a 24 GiB Mac since dev18. Work of dev42–dev51 below.

- **MTP speculative decoding** (dev45, dev45b).
  - `redlite chat` and `redlite serve --native` use it automatically when the 2.26 GiB head file
    (`redlite download mtp`) is in `models/` and every expert is resident.
  - The output is exactly the same as plain decoding, with any sampler.
  - M4 Max, IQ2_XXS, six prompts: +8 % to +30 % (80 → 87–104 tok/s); interactive chat on code 88.7 → 108.6 tok/s.
  - IQ3_XXS code prompt: 79.8 → 100.6 tok/s.
- **Session state on disk** (dev43): `--state-dir` saves a prompt prefix's state; a restart restores it
  bit-identically. M4 Pro: 4096 of 4212 tokens restored, first token in 1.4 s.
- **Qwen3-Coder-Next** (dev42): Bartowski IQ2_XXS / IQ3_XXS; `redlite download coder`.
- **24 GiB Mac** (M4 Pro, 4 GiB cache):
  - decode 33 tok/s (dev18: 27.8);
  - prompt ingestion 360 tok/s at 8192 tokens (dev18: 16 tok/s token by token);
  - native outputs bit-identical to the M4 Max's;
  - `regress_m4.sh` 45 passed / 0 failed / 4 skipped: the llama.cpp oracle does not fit that GPU, and MTP needs
    40 GiB (dev47, dev51).
  - A larger expert cache does not speed decode up there (dev51).
- **Answers compared with Qwen's own API** (dev48): IQ3_XXS is about twice as close as IQ2_XXS (10 vs 2 identical
  answers of 235).
- **M4 Max 48 GiB, every expert resident** (`bench_m4.sh`, median of 3):
  - decode IQ2_XXS 81.15 → **86.24** tok/s, IQ3_XXS 76.88 → **79.91**;
  - prompt ingestion at 8192 tokens 917.5–926.6 / 902.3–911.9 tok/s (0.4.1: 921.5 / 917.5).
  - Short-prompt and 4 GiB-cache numbers varied up to 2× between runs this session and are not compared. An
    alternated A/B against the 0.4.1 build showed no regression. Record: `benchmarks/m4max-48gb-0.5.0.json`.
- **For users:** `docs/FINDINGS.md` with charts; `docs/ROADMAP.md`.
- **Project:**
  - hosted GitHub CI removed; `scripts/dev/local_ci.sh` runs the same checks;
  - `REDLITE_SDK` for Command Line Tools whose SDK is newer than their linker.
- **Validation (M4 Max 48 GiB):** local CI PASS, `regress_m4.sh` IQ2_XXS 51/51, IQ3_XXS 38/0/13, `quick_parity.sh`
  PASS on both GGUFs, 0 warnings.

## Milestones of 0.5.0

### Project

- `docs/ROADMAP.md`: the agreed next steps.
- GitHub CI removed (hosted runners are paid); `scripts/dev/local_ci.sh` runs the same model-free checks locally.

### dev50 — pipelined prefill expert decode (M4 Pro 24 GiB, M4 Max 48 GiB)

- The prefill expert kernels decode step k+1's weights into registers while step k multiplies. Bit-identical.
  Experts −2.7 % / −4.4 %, prefill 922 → 936 tok/s (M4 Max) and 353 → 360 tok/s (M4 Pro).
- Measured: the decode **arithmetic** costs ~1.1 s of the experts' 3.86 s; reading the weights costs ~0.36 s.
- Four 32-pair-tile kernels, which halve the decodes per pair, were all slower (4.2–9.6 s vs 3.9 s) and were
  reverted. See `docs/REDLITE_DEV49_PREFILL_24GB.md`.

### dev49 — prompt ingestion on the 24 GiB Mac (M4 Pro 24 GiB, M4 Max 48 GiB)

- The M4 Pro's 348 tok/s on 8192 tokens is GPU compute: dense 10.9 s + experts 10.3 s. The 12.4 s of expert loading
  overlaps them on the prefetch thread. With 16 GPU cores against 40 it runs at the M4 Max's efficiency, so
  500 tok/s would need kernels about 1.45× faster on every Mac.
- About 44 % of the prefill expert time is decoding 2-bit weights (measured by removing the decode).
- Kept: vector codebook loads and branch-free signs in `rm_group8`. Bit-identical; experts −3.4 %, prefill +1.4 %
  on both Macs (M4 Pro 348 → 353, M4 Max 912 → 925 tok/s). See `docs/REDLITE_DEV49_PREFILL_24GB.md`.

### dev48 — answers compared with Qwen's own API (M4 Max 48 GiB)

- `scripts/dev/api_compare.py` compares greedy answers word for word with Alibaba Cloud's
  `qwen3-next-80b-a3b-instruct` (temperature 0, top_k 1). It covers 235 prompts in 8 categories
  (`tests/fixtures/api_prompts.txt`); the answers are stored once in `tests/fixtures/qwen_api_reference.json`.
- Identical answers: IQ2_XXS 2/235, IQ3_XXS 10/235. Median share of words matching from the start: 7.8 % vs
  15.6 %. IQ3_XXS is about twice as close.
- The API's `temperature: 0` alone is not deterministic, and its logprobs are misaligned. See
  `docs/REDLITE_DEV48_API_REFERENCE.md`.

### docs — findings page and charts

- `docs/FINDINGS.md`: what we learned, for users, with six SVG charts (decode and prefill by release vs llama.cpp,
  MTP per prompt, M4 Pro 24 GiB decode and cache options, a decode token's time by stage).
- Charts drawn by `scripts/dev/make_charts.py` (stdlib only) from `benchmarks/charts.json`; `tests/test_charts.py`
  fails on a stale chart or a missing source record. New record `benchmarks/m4max-48gb-dev45-mtp.json`.
- README: charts, MTP and `--state-dir` in the quick start, M4 Pro 24 GiB dev47 results, full-residency sizes
  since dev37 (17,316 / 28,800 MiB).

### dev47 — M4 Pro 24 GiB field session, long context, build fix (both machines)

- **M4 Pro 24 GiB** (first native measurement since dev18), IQ2_XXS, 4 GiB cache:
  - decode 33.0 tok/s (dev18: 27.8);
  - prefill 1100 / 8192 tokens 280.8 / 349.8 tok/s (dev18: 16.3 token by token);
  - native outputs bit-identical to the M4 Max's (the llama.cpp oracle itself runs out of GPU memory there).
  - A/B: dev37 slot classes +1.6 % (−25 % misses); cache bias 0.5 +5 %; prefetch +12 %; `F_NOCACHE` none.
- `REDLITE_SDK` selects the build SDK (the M4 Pro's Command Line Tools ship an SDK their linker cannot read);
  `scripts/dev/small_mac_session.sh` runs the whole 24 GiB session.
- **M4 Max long context:** parity with llama.cpp at 16K positions (argmax 100/100, KL 7.9e-5; max logit 6.2 above
  the 8K-derived 5.0 bound); 33.5K-token prompt prefill 416.7 tok/s, decode 25.7 tok/s.
- **dev44** (chunk-parallel DeltaNet prefill) not built: the recurrence is at most 6.7 % of an 8K prefill.
  See `docs/REDLITE_DEV47_LONG_CONTEXT.md`.

### dev45 — speculative decoding with the MTP block (M4 Max 48 GiB, full residency)

- `redlite-generate --mtp FILE` (`redlite download mtp`): the Qwen3-Next MTP block drafts one token, a 2-row
  verify checks it in one pass (two-vector GEMV kernels, 2-row DeltaNet kernels with in-kernel snapshots,
  merged 2-row experts). Output is exactly that of plain decode (greedy identical).
- IQ2_XXS, six prompts: +8.5 % to +30.0 % decode (80 → 87-104 tok/s), acceptance 0.58-0.84; IQ3_XXS code
  prompt 75.2 → 92.3 tok/s. See `docs/REDLITE_DEV45_MTP.md`.

### dev46 — expert-cache options for 24 GiB Macs (M4 Max 48 GiB)

- Opt-in: `RL_ROUTE_CACHE_BIAS=λ` (cache-aware routing: at λ = 0.5, 4 GiB cache misses 29.5 → 22.9 per token
  with engine perplexity 17.586 → 17.583), `RL_POOL_NOCACHE=1` (`F_NOCACHE` expert reads).
- `redlite-engine perplexity` (perplexity of the engine with its runtime options) and
  `scripts/dev/small_mac_ab.sh` for the 24 GiB A/B. No speed claim yet. See `docs/REDLITE_DEV46_SMALL_MAC_OPTIONS.md`.

### dev42 — Qwen3-Coder-Next (M4 Max 48 GiB)

- `Qwen/Qwen3-Coder-Next` runs unchanged (same architecture, rope base 5e6 read from the GGUF): Bartowski
  IQ2_XXS 50/50 and IQ3_XXS 37/0/13 in `regress_m4.sh` against the pinned llama.cpp, dequantization
  bit-identical. `redlite download coder` / `coder-48gb` (and the missing `48gb` = IQ3_XXS Instruct).
  See `docs/REDLITE_DEV42_CODER_NEXT.md`.

### dev43 — session state checkpoints on disk (M4 Max 48 GiB)

- `--state-dir DIR` for `redlite-generate` and `redlite-server` stores the session state (DeltaNet states,
  attention K/V, last logits) at multiples of the prefill chunk and restores the longest matching prefix:
  a 4209-token prompt 4602 → 387 ms prefill; server TTFT after a restart 4928 → 556 ms; greedy identical.
- Fixes a dev37 regression: with a 2 GiB cache, long prompts failed in the prefill prefetch (size classes
  are now used only when every class holds two layers' experts). See `docs/REDLITE_DEV43_STATE_CACHE.md`.

## 0.4.1 — 2026-10-01

Faster decode and a better use of a bounded expert cache, on the same two GGUFs. Work of dev33–dev39
below, all on the M4 Max 48 GiB (nothing measured on the M4 Pro 24 GiB or an M4 Air).

| GGUF, cache | Decode 0.4.0 → 0.4.1 | Prefill 1100 0.4.0 → 0.4.1 | Prefill 8192 0.4.0 → 0.4.1 |
|---|---|---|---|
| IQ2_XXS, full | 71.15 → **81.15** | 891.9 → 900.4 | 926.9 → 921.5 |
| IQ2_XXS, 4 GiB | 47.77 → **52.89** | 545.8 → **808.9** | 645.6 → **898.9** |
| IQ3_XXS, full | 63.39 → **76.88** | 786.0 → **876.7** | 860.3 → **917.5** |
| IQ3_XXS, 4 GiB | 43.8 → **49.59** | 428.4 → **763.0** | 588.3 → **889.2** |

`bench_m4.sh --reps 3 --cool 90` (median of three cooled runs; the new `--only prefill4` measures the
4 GiB cache with the default 2048-token chunk, 0.4.0's 4 GiB rows used 512). Record:
`benchmarks/m4max-48gb-0.4.1.json`.

- **Decode:** sub-block IQ dots (dev33), grouped split-K attention at long context (dev35), expert
  down-projection lanes and concurrent encoders (dev38), fused expert tail (dev39). IQ3_XXS decode is
  now above the pinned llama.cpp (76.88 vs 68.54).
- **Memory:** expert slots of each layer's own size (dev37): full residency 21,312 → 17,316 MiB
  (IQ2_XXS) and 29,376 → 28,800 MiB (IQ3_XXS); a 4 GiB cache holds 23 % more experts (−24 % misses
  per token).
- **Bounded-cache prefill:** 2048-token chunks at every cache size (dev34).
- **IQ3_M** (dev36): supported (Q4_K experts) and validated against llama.cpp with bounded caches; not
  chosen automatically on 48 GiB. The planner's full-residency limit is 70 % of RAM.
- **Records:** `docs/WHAT_DID_NOT_WORK.md` lists every reverted attempt, trap and unreached target.
- **Validation:** `scripts/regress_m4.sh` IQ2_XXS 49/49, IQ3_XXS 36 passed / 0 failed / 13 skipped
  (dev39 gates; 0.4.1 changes no native source after them), `make sanitize` 0 warnings.


### dev39 — fused expert tail (M4 Max 48 GiB)

- GPU-routed layers end with one `rl_moe_tail` dispatch (weighted expert sum + residual + layer-output
  copy) instead of three (`RL_ENGINE_FUSE_TAIL=0` restores them): IQ2_XXS full residency median
  84.2 → 85.5 tok/s (+1.6 %, three clean pairs). A single-threadgroup variant that also fused the next
  RMSNorm was 6 % slower and reverted. See `docs/REDLITE_DEV39_EXPERT_TAIL.md`.

### dev38 — decode: expert lanes and concurrent encoders (M4 Max 48 GiB)

- Expert down-projection rows use 8 lanes instead of 32 (at most 4 lanes per 256-value block), and
  decode encoders are concurrent with barriers only between dependent dispatches
  (`RL_ENGINE_CONCURRENT=0` restores serial encoders).
- Cooled A/B, full residency: IQ3_XXS 70.4 → 79.1 tok/s (llama.cpp 68.5), IQ2_XXS 75.3 → 82.6 tok/s;
  4 GiB cache IQ2_XXS 52.5 → 59.8 tok/s. Parity unchanged on both models.
- `quick_parity.sh` now compares non-reference models with their own llama.cpp dumps.
  See `docs/REDLITE_DEV38_DECODE_DISPATCH.md`.

### dev37 — expert slots of each layer's own size (M4 Max 48 GiB)

- The expert pool and LRU get slot size classes: one per distinct layer triplet size, the same
  slot count per layer (`RL_POOL_CLASSES=0` restores uniform slots). Uniform slots of the largest
  size wasted a quarter of an IQ2_XXS cache on padding.
- 4 GiB cache, six prompts: expert misses per token 39.2 → 29.8 with prefetch (−24 %), 60.0 → 45.4
  without; decode +3 % on this Mac (cooled A/B), output identical. Not measured on 24 GiB.
- Full residency 21312 → 17316 MiB for IQ2_XXS (footprint 22445 → 18447 MiB, same 72 tok/s),
  29376 → 28800 MiB for IQ3_XXS; the planner uses the same formula.
- Chosen from data: `RL_ROUTE_TRACE` routing traces replayed by `scripts/dev/cache_policy_sim.py`;
  smarter replacement policies gained ≤ 5 % in simulation and were not implemented.
  See `docs/REDLITE_DEV37_EXPERT_SLOTS.md`.

### dev36 — the IQ3_M GGUF (M4 Max 48 GiB)

- Q4_K routed experts (IQ3_M's down projections): decoder bit-identical to the dense Q4_K
  dequantizer; IQ3_M dequantizes bit-identically to ggml and passes CPU vs Metal parity.
- Not worth it on 48 GiB: full residency runs out of GPU memory (46.8 → 6.0 tok/s, then
  `kIOGPUCommandBufferCallbackErrorOutOfMemory`), a 28 GiB cache gives 28.7 tok/s, and perplexity
  14.05 ± 0.26 vs 14.29 for IQ3_XXS is inside the error bar. Matches llama.cpp with bounded caches
  (logits, 24 greedy tokens, 1200-token long context); not in the automatic model choice. The planner's full-residency limit is now 70 % of RAM (was 75 %).
- `docs/WHAT_DID_NOT_WORK.md` collects every reverted attempt, trap and unreached target.
  See `docs/REDLITE_DEV36_IQ3M.md`.

### dev35 — long-context decode: grouped split-K attention (M4 Max 48 GiB)

- One threadgroup per (128-position block, KV head) for the 8 query heads that share it: K/V read
  once instead of 8 times. Attention at ~8400 positions 253 → 113 ms per 64 tokens; decode at
  ~8192 positions 57.8 → 66.0 tok/s (IQ2_XXS, full residency, A/B). Parity unchanged.
  See `docs/REDLITE_DEV35_LONG_DECODE.md`.

### dev34 — bounded-cache prefill: 2048-token chunks (M4 Max 48 GiB)

- With a bounded cache each chunk reloads nearly every expert: 8192 tokens at 4 GiB loaded
  161 GB of experts in chunks of 512, 49 GB in chunks of 2048. The default chunk is now 2048
  for every cache (+571 MiB footprint at 4 GiB).
- IQ2_XXS, 4 GiB cache, cooled A/B: prefill 1100 tokens 540 → 795 tok/s, 8192 tokens
  640 → 886 tok/s. Not measured on a 24 GiB Mac. See `docs/REDLITE_DEV34_BOUNDED_PREFILL.md`.

### dev33 — faster IQ kernels (M4 Max 48 GiB)

- Sub-block IQ3_XXS / IQ3_S / IQ2_S / IQ4_XS dots for decode, a faster 8-value IQ3 decoder for
  prefill, vectorized IQ2_XS / IQ1_M decode expert dots; `redlite-engine kernel-bench`.
- Same-session A/B vs the 0.4.0 binary: IQ3_XXS decode 64.7 → 70.1 tok/s (llama.cpp 68.5),
  IQ3_XXS prefill 1100 tokens 787 → 837 tok/s (llama.cpp 861), IQ2_XXS decode 65.9 → 68.3.
- Two attempts reverted (codebooks in threadgroup memory; uchar4 loads in the IQ2_XXS GEMV).
  See `docs/REDLITE_DEV33_IQ_KERNELS.md`.

## 0.4.0 — 2026-09-30

The native runtime, faster and with a second, higher-quality GGUF. Work of dev28–dev32
below, all on the M4 Max 48 GiB (nothing re-measured on the M4 Pro 24 GiB or an M4 Air).

**What 0.4.0 adds**

- **Prefill** (dev30): tiled attention, experts on simdgroup matrices, a faster DeltaNet
  recurrence and dense GEMM, one residency set, 2048-token chunks with full residency.
  1100 tokens: 307.3 → 891.9 tok/s; 8192 tokens: 212.6 → 926.9 tok/s (IQ2_XXS, all experts
  resident; the pinned llama.cpp on the same ids: 858.3 / 893.7). Decode unchanged.
- **IQ3_XXS GGUF** (dev31): IQ3_XXS, IQ3_S, IQ2_S and IQ4_XS decoded bit-identically to
  ggml; the 31.7 GB file runs with every expert resident (29,376 MiB, sized from its
  payload). Perplexity on the same local text 14.29 vs 16.47 for IQ2_XXS; greedy identical
  to llama.cpp, KL 1.4e-12. `redlite chat` picks it when RAM ≥ 40 GiB.
- **Server** (dev29): conversation-state reuse (second-turn first token 3.8 s → 0.17 s),
  stop sequences, FIFO queue.
- **Distribution** (dev28): model-free Linux CI on every push/PR, `package_release.sh`
  arm64 tarball with SHA256.

**Milestone targets of the 0.4.0 cycle** (M4 Max 48 GiB):

| Milestone | Target | Result |
|---|---|---|
| dev28 | Linux CI on push/PR, green on GitHub; release tarball | CI and tarball done; **green on GitHub not reached** (GitHub does not start hosted jobs on this account: billing) |
| dev29 | state reuse (greedy identical), TTFT before/after, stop, FIFO, fake + real tests | reached: identical answer, 3833 → 173 ms |
| dev30 | prefill 1100 ≥ llama.cpp same prompt; 8192 ≥ 280 tok/s; decode 22 GiB not below 0.3.0 | reached with full residency: 891.9 (llama.cpp 858.34), 926.9, decode 71.09 vs 70.47 (A/B); at 4 GiB 1100 tokens 545.8, below llama.cpp |
| dev31 | better GGUF with full residency, new quant types, greedy = llama.cpp, KL ≤ 1e-5, perplexity both files, chat picks it | reached: IQ3_XXS, KL 1.4e-12, 24/24 greedy, PPL 14.29 vs 16.47 |
| dev32 | 0.4.0 release, README per model/cache, tarball, PR | done (PR open, not merged) |


### dev31 — IQ3_XXS GGUF: new quant types, full residency on 48 GiB (M4 Max 48 GiB)

- Bartowski `…-IQ3_XXS.gguf` (31,726,709,216 bytes, SHA-256 = HF LFS id) runs natively
  with every expert resident (29,376 MiB, computed from its expert payload).
- New quant types, CPU reference bit-identical to ggml's `to_float` and Metal kernels:
  IQ3_XXS, IQ3_S, IQ2_S, IQ4_XS; routed experts may have a down type different from
  gate/up (type word in the Metal kernels, ABI unchanged).
- Parity on the IQ3_XXS file: CPU oracle vs Metal YES, GPU-routed YES, logits vs llama.cpp
  KL 1.4e-12, 24 greedy tokens identical, 1100-position context PASS.
- Perplexity (pinned llama.cpp, frozen local corpus): IQ2_XXS 16.47, IQ3_XXS 14.29.
- `--cache-mib full`; `redlite chat` picks IQ3_XXS when RAM ≥ 40 GiB and it fits, else
  IQ2_XXS; the planner sizes the cache from the payload (no more 22,528 constant).
- `regress_m4.sh` runs on both files (dequant parity vs ggml added; legacy dense stage
  tools SKIP on the IQ3_XXS layout). See `docs/REDLITE_DEV31_IQ3.md`.

### dev30 — batched prefill: tiled attention, matrix experts, faster dense pass (M4 Max 48 GiB)

- Tiled causal attention (`attn_fa_b`, flash-attention order on f32 simdgroup matrices),
  batched experts on simdgroup matrices with split accumulators, DeltaNet recurrence with one
  simdgroup per state row, threadgroup-staged dense GEMM, one Metal residency set for every
  engine buffer, 2048-token chunks by default when every expert is preloaded (512 otherwise),
  no expert prefetch under full residency.
- Prefill at 22 GiB (full residency, default chunk): **891.9 tok/s** on 1100 tokens
  (pinned llama.cpp on the same ids: 858.34) and **926.9 tok/s** on 8192 (0.3.0: 307.3 /
  212.6 at 4 GiB). At 4 GiB with 512-token chunks: 545.8 / 645.6. Decode unchanged
  (A/B at 22 GiB: 71.09 vs 70.47 for the 0.3.0 binary).
- Tried and reverted: router selection on the GPU (no measurable gain), skipping empty
  pair blocks in the expert kernels (slower).
- `regress_m4.sh`: `logits.long_context_full_residency`; `long_positions.sh` bound 5.0 at
  ≥ 8192 positions (native self-consistency floor 4.45 measured on 0.3.0);
  `bench_m4.sh --only prefill22`; `RL_PREFILL_PROFILE=1`. See `docs/REDLITE_DEV30_PREFILL.md`.

### dev29 — server: state reuse across turns, stop sequences, FIFO queue (M4 Max 48 GiB)

- `redlite-server` keeps the engine state between requests when the new prompt's ids extend
  exactly the ids the state holds (previous prompt + generated tokens that were fed back);
  otherwise it resets and ingests everything (the DeltaNet state cannot be truncated).
  `usage.prompt_tokens_details.cached_tokens` reports the reuse; `--no-reuse` disables it.
  Second turn of a 1185-id conversation: TTFT 3833 → 173 ms (4 GiB cache) and
  3779 → 167 ms (22 GiB), answer identical to 0.3.0 and to `--no-reuse`.
- `stop` (string or up to 4 strings) is supported: output ends before the earliest match,
  `finish_reason: "stop"`; partial matches are held back like incomplete UTF-8.
- Requests are queued FIFO and run by one worker thread; `/health` and `/v1/models` answer
  while a generation runs; `--queue N` (default 16) waiting requests, then `503`.
- Tests: 4 new protocol tests (fake backend, also under ASan/UBSan), selftest cases, and the
  real-model regress check `server.reuse_greedy` (warm turn 2 == cold turn 2, greedy).
  `scripts/dev/server_ttft.py` measures the second-turn latency.
  See `docs/REDLITE_DEV29_SERVER.md`.

### dev28 — Linux CI, release tarball, same-prompt llama.cpp baseline (M4 Max 48 GiB)

- `.github/workflows/ci.yml` runs on every push and pull request: one Linux job with ruff,
  compileall, `make native` and `make sanitize` (warnings fail) and `make test`. macOS
  hosted jobs and `lint.yml` removed; the self-hosted M4 workflow stays manual.
  **Not green on GitHub**: GitHub refuses to start hosted jobs on this account (billing);
  the same commands pass in a Linux container locally.
- `scripts/package_release.sh`: `dist/redlite-<version>-macos-arm64.tar.gz` with
  `redlite-generate`, `redlite-server`, `redlite-engine` (`-mcpu=apple-m1`, macOS ≥ 14),
  `INSTALL.md`, `BUILDINFO`, and a `.sha256` file. `REDLITE_MCPU` / `REDLITE_BUILD_OUT`
  for the build scripts.
- `redlite-ref-llama bench` and `bench_m4.sh --only prefill8192|llama`: llama.cpp is now
  measured on the exact ids of the native benchmark. Baseline (0.3.0, medians of three
  cooled runs): native decode 71.50 / 47.93 tok/s (22 / 4 GiB), prefill 307.30 (1100) and
  212.60 (8192) tok/s; llama.cpp prefill 858.34 / 893.71 tok/s, decode 72.46 tok/s.
  See `docs/REDLITE_DEV28_CI_RELEASE.md`.

## 0.3.0 — 2026-09-30

First release of the **native runtime**: Red Lite's own C11 / Objective-C / Metal
implementation of the Qwen3-Next-80B-A3B graph. At inference time it uses neither
llama.cpp nor Python. It is the work of the 0.3.0.dev3–dev26 entries below. The 0.2
launcher (pinned llama.cpp / CPU mmap runtime) is unchanged and remains the
field-validated path on the M4 Pro 24 GiB.

**What 0.3.0 contains**

- **Engine** (dev18). The dense weights are mapped in place, and the routed experts go
  through a bounded Metal-visible LRU. There are two independent backends: a CPU oracle
  and Metal.
- **Tokenizer, sampler and chat template** (dev18–dev26). The sampler follows llama.cpp's
  chain with the same arithmetic.
- **Batched prefill** (dev20, dev24) and **GPU-routed decode** with full residency
  (dev21, dev23).
- **Decode kernels:**
  - dev22: sub-block GEMV, one encoder per token;
  - dev23: per-layer early-out and pre-gated expert prefetch;
  - dev26: split-K decode attention.
- **Surfaces:** `redlite chat`, `redlite serve --native` / `redlite-server` (OpenAI
  `/v1/chat/completions` with SSE), `redlite-generate`.
- **Validation tooling.** `scripts/regress_m4.sh` has 46 checks: the llama.cpp oracle,
  the CPU oracle, the GGUF fuzz, ASan/UBSan on the model-free tests and on a real chat
  turn, and the model-free kernel self-test.

**Measured, M4 Max 48 GiB** (the README table has the ranges and sources):

| | tok/s |
|---|---:|
| Decode, short context, 22 GiB cache (all experts resident) | 69.8–72.7 |
| Decode, short context, 4 GiB cache | 46.5–50.1 |
| Decode at ~8192 positions, 22 GiB / 4 GiB cache | 56.3–57.4 / 38.8–39.2 |
| Prefill of 1100 tokens, 4 GiB cache | 314.3 |

Greedy output is token-identical to the pinned llama.cpp on the regression prompts.
**Not measured on the M4 Pro 24 GiB since dev18** (27.8 tok/s decode with a 4 GiB cache on
the dev18 build).

**Milestone targets of the 0.3.0 cycle** (M4 Max):

| Milestone | Target | Result |
|---|---|---|
| dev22 | decode ≥ 62 tok/s at 22 GiB | reached (66.2) |
| dev23 | decode ≥ 45 tok/s at 4 GiB | reached (50.1) |
| dev24 | prefill ≥ 300 tok/s on 1100 tokens at 4 GiB | reached (314.3) |
| dev26 | parity and benchmarks at 4096/8192, sanitized chat turn, model-free kernel tests | done |

**Legacy.** The Python streaming oracle (`redlite-stream`, `redlite-ffn` and `redlite-topk`
from Python, dev1–dev8) is frozen. It is kept as a numerical reference and its `--help`
says so. The native stage parity CLIs (dev9–dev17) are regression tools; `redlite-engine`
supersedes them.


### dev26 (completion) — long positions, sanitized chat turn, kernel self-test, split-K attention (M4 Max 48 GiB)

See `docs/REDLITE_DEV26_LONG_CONTEXT.md`. Not measured on the M4 Pro.

- **Parity at 4096 and 8192 positions** against the pinned llama.cpp
  (`scripts/dev/long_positions.sh`; the frozen fixture's ids repeated): argmax 100/100,
  KL ≤ 6.4e-3. The max-logit bound for these positions is 4.0. It was set after measuring
  native token-by-token against native batched prefill at 4096, which already differ by
  2.62.
- **Split-K decode attention** (`attn_gqa_split` + `attn_gqa_merge`, above 256
  positions). Decode at ~8192 positions: 20.1 → 57.4 tok/s at 22 GiB and 17.2 → 38.8 tok/s
  at 4 GiB. At ~4096: 31.5 → 65.5 and 24.8 → 43.6. Short context is unchanged.
  `RL_ENGINE_ATTN_SPLIT=0` restores the old kernel.
- **Sanitized chat turn** (`scripts/dev/sanitize_chat.sh`, regress check
  `generate.sanitize`): ASan+UBSan `redlite-generate` on the real model, with batched
  prefill, prefetch, decode and sampling. No report, and the greedy text is identical to
  the normal build.
- **Model-free kernel self-test** (`redlite-engine kernel-selftest`, regress check
  `selftest.engine_kernels`, also under `make sanitize`) covers:
  - the dev22 sub-block GEMVs and the dev18 block GEMVs against the CPU row dot;
  - the dev23 early-out guard;
  - `rl_copy_f32`;
  - `rl_route` against the CPU router;
  - both decode attention kernels against a double-precision GQA.
- **Fixed.**
  - `compare_dumps.py` reported parity on an empty native dump.
  - `redlite-engine --tokens` silently truncated lists longer than 4096 ids; it now takes
    up to 65 536 and rejects longer lists.

### dev24 — prefill: pre-gated expert prefetch and parallel routing (M4 Max 48 GiB)

Prefill of 1100 tokens with a 4 GiB cache, in the default 512-token chunks, rose from
261.7 to **314.3 tok/s** (target ≥ 300; cooled medians). Parity with llama.cpp on
the 1100-token prompt holds at chunks 96/256/512/1100. Not measured on the M4 Pro. See
`docs/REDLITE_DEV24_PREFILL_OVERLAP.md`.

- **Pre-gated prefetch in the batched prefill.** Layer *l+1*'s router is applied to
  layer *l*'s FFN input. A background thread loads the predicted union of experts
  while layer *l*'s experts and layer *l+1*'s dense pass run on the GPU. Layer *l*'s
  plan stays pinned until the join. `RL_ENGINE_PREFETCH=0` disables it
  (cooled medians: 271.6 off vs 314.3 on).
- **Parallel router selection.** The chunk's softmax top-10 runs on all CPU cores
  (`dispatch_apply`), bit-identical per token: ~180 → ~39 ms per 1100 tokens.
- **`redlite-engine logits`** prints a `prefill split:` line.
- **Dev tooling.** `quick_parity.sh` deletes its outputs before running (it compared a
  stale dump after a crash). `bench_m4.sh --cool S` pauses before each run, because
  back-to-back prefill runs throttle the GPU on this machine (311 → 133 tok/s over five
  runs). Earlier prefill figures were taken without a pause.
- **Race fixed before release.** The predicted logits are double-buffered by layer
  parity, because the prefetch thread reads them while the next dense pass runs.

### dev23 — per-layer early-out and pre-gated prefetch (M4 Max 48 GiB)

Decode with a 4 GiB cache rose from 44.7 to **50.1 tok/s** (target ≥ 45). Decode with full
residency rose from 66.2 to **72.7 tok/s**. Implemented and validated in parity on the
M4 Max with a limited cache; not measured on the M4 Pro. See
`docs/REDLITE_DEV23_EARLY_OUT_PREFETCH.md`.

- **GPU-routed tokens no longer restart.** A miss at layer *f* raises an early-out flag
  (buffer index 30 of every decode kernel). The CPU loads only layer *f*'s experts and
  resumes at *f+1*. The ~72 MiB of per-token state backups and the whole-token redo of
  dev21 are gone.
- **Policy.** The early-out path is tried after a token that loaded nothing; more than 4
  early-outs send the next tokens to the synchronous path.
- **Pre-gated prefetch** in the synchronous path. Layer *l+1*'s router is applied to
  layer *l*'s FFN input, and the predicted experts are loaded while the GPU works.
  Critical-path load time falls from 1.9 to 0.6 ms per token. `RL_ENGINE_PREFETCH=0`
  disables it.
- **Expert pool release check** is now per slot (load generations), so loads into other
  slots are allowed while a plan is in flight.
- **Stats.** `--stats`/`--json` report per-layer early-outs (`early_outs` in the JSON).
- **Measured dead ends.**
  - Spin-waiting on the command-buffer status: 44.6 → 27.5 tok/s.
  - Early-out as the only 4 GiB path: 8.3 tok/s.
  - Periodic probing: 39.5 tok/s.

### dev22 — decode kernels (M4 Max 48 GiB)

Decode with full expert residency rose from 57.6 to **66.2 tok/s** (target ≥ 62;
llama.cpp fully resident: 68). Decode with a 4 GiB cache rose from 39.9 to 44.7 tok/s.
See `docs/REDLITE_DEV22_DECODE_KERNELS.md`.

- **One compute encoder per token** on the GPU-routed path and one per layer on the
  synchronous path. The layer bodies became shared emitters; the in-token blits became
  an `rl_copy_f32` dispatch; `redmetal_topk_pool_encode_device_into` appends the routed
  experts to the caller's encoder. Worth +1.9% on its own.
- **Sub-block decode GEMV.** For Q4_K, IQ2_XXS, Q6_K and F32 (`rl_rows2_*`), one lane
  handles one 32- or 16-value sub-block, with up to 32 lanes per row and `float4` loads.
  This is the main gain. `RL_ENGINE_ROWS2=0` restores the block kernels.
- **Validation.** Parity with the CPU oracle improved (worst layer abs 5.3e-05 →
  1.5e-05, 0 router mismatches). Greedy output is identical to llama.cpp.
  `regress_m4.sh` passes 44/44 after a clean build with 0 warnings.
- **New dev tools.** `scripts/dev/bench_m4.sh` (median decode/prefill benchmark) and
  `scripts/dev/quick_parity.sh` (the parity gate used for every kernel change).

### PR #2 content (dev25/dev26 model-free work, merged 3e2257e)

Product surface (part of the dev25 scope) and model-free robustness work (part
of the dev26 scope). See `docs/REDLITE_DEV25_PRODUCT.md` and
`docs/REDLITE_DEV26_ROBUSTNESS.md`. Implemented and tested model-free on Linux
x86_64 (cloud container, gcc 13.3 / clang 18.1). Validated with the real model on the
M4 Max 48 GiB (macOS 27, commit `7d96db1`): `scripts/regress_m4.sh` 44/44 after a clean
build, plus manual checks of `redlite chat` defaults, `redlite serve --native` streaming
and interactive Ctrl-C. Not run on the M4 Pro. Version unchanged.

- macOS 27 SDK: removed the no-op `didModifyRange:` calls on Shared Metal buffers, so
  `make native` and `make redmetal` are warning-free again. `make sanitize` now works
  on macOS.

- `redlite chat` picks the expert cache from RAM when `--cache-mib` is omitted:
  22528 MiB (full residency, preloaded) at ≥ 40 GiB, 4096 MiB below. `--batch`
  and `--json` are passed through.
- Ctrl-C now stops the current answer. `redlite-generate` closes the answer like
  a `--max-tokens` stop, so the conversation continues. Ctrl-C at the prompt
  quits cleanly, and non-interactive runs exit 130. Previously the Python
  launcher killed the runtime on the first Ctrl-C.
- `redlite-generate --json`: one statistics object per answer on stderr. The
  interactive chat is now English, like the rest of the runtime. The GGUF
  `tokenizer.chat_template` is checked for ChatML (not interpreted) with the
  built-in ChatML as fallback.
- Native OpenAI-compatible server:
  - `redlite-server` provides `/v1/chat/completions` with SSE streaming, plus
    `/v1/models` and `/health`, in portable C on `rl_engine`;
  - `redlite serve --native` launches it;
  - `redlite-server-fake` (echo backend) with `tests/test_native_server.py`
    tests the protocol without a model.
- `regress_m4.sh` gains `server.stream_greedy`, `generate.json` and
  `generate.sigint`.
- Sampler parity with the pinned llama.cpp:
  - the native sampler gains min-p and follows llama.cpp's chain order and float
    arithmetic (top-k → top-p → min-p → temperature → draw);
  - `rl_sampler_distribution()` exposes the exact distribution;
  - `--min-p` is exposed in `redlite-generate` and `redlite chat`, and `min_p` in
    `redlite-server`; the default stays 0.
- `scripts/dev/compare_sampler.py` compares `redlite-sampler-dist` against the
  real pinned libllama chain (`redlite-ref-sampler`). On Linux the probabilities
  are identical on every grid point; tie handling and the PRNG are documented
  differences. `regress_m4.sh` gains `sampler.vs_llama`.

- Fixed the portable `make native` build (DeltaNet state/tail CLIs used Metal
  telemetry outside `#ifdef __APPLE__`). The Linux build is now warning-clean
  under gcc and clang.
- Hardened both GGUF readers against corrupt files:
  - directory, kv and tokenizer-array counts are bounded by the file size;
  - expert-map tensor offsets beyond EOF are rejected (they could wrap
    `data_base + offset`);
  - out-of-range `token_type` values are rejected instead of converted with UB;
  - nested arrays are rejected.
- Added `redlite-gguf-fuzz` (truncations, boundary values and seeded mutations
  against both readers; run by `make native`) and `make sanitize` (ASan + UBSan
  build and run of every model-free test and the fuzz). `regress_m4.sh` gains
  `selftest.gguf_fuzz` and `selftest.sanitize`.

## 0.3.0.dev21 — 2026-09-11

GPU-routed decode with full expert residency on `v0.3-streaming`. See
`docs/REDLITE_DEV21_GPU_ROUTED_DECODE.md`.

- Added a per-layer expert residency table (GPU slot addresses, maintained by
  the runtime on every LRU commit/abort; the LRU now reports evicted keys), a
  Metal router kernel (`rl_route`, same selection rule as the CPU router) and
  a GPU-driven single-token expert encode, so a whole token runs in one
  command buffer with one CPU wait. DeltaNet states are backed up before the
  attempt and restored if any layer selected a non-resident expert, in which
  case the token is redone by the unchanged synchronous path. Attempted only
  after a token with no expert load (`RL_ENGINE_SPECULATIVE=0` disables).
- Full residency: when the cache can hold every routed expert (≥ 21 300 MiB
  for this GGUF) they are all loaded at open (3.7 s from the page cache;
  `RL_ENGINE_PRELOAD=0` disables) and every token is GPU-routed. The pool's
  slabs are attached to the engine queue with an `MTLResidencySet` (macOS
  15+); per-encoder `useResource` over 384 slabs had made Metal redo 22 GiB of
  residency per command buffer (11 tok/s).
- `redlite-engine parity --repeat N` replays the sequence after a reset so the
  warm-cache GPU-routed path is checked against the CPU oracle; `--stats`
  outputs report GPU-routed / fallback / synchronous token counts.
- Validation (M4 Max): 12/12 GPU-routed tokens with identical router ids,
  worst layer abs 5.3e-05, logits abs 1.7e-05 vs the CPU oracle; regression
  36/36 plus `engine.parity.gpu_routed` on ≥ 40 GiB machines.
- Decode on the M4 Max with a 22 GiB cache: 54–57 tok/s (from ~40), GPU-bound
  at 16 ms of GPU time per token; llama.cpp fully resident: 68 tok/s.

## 0.3.0.dev20 — 2026-09-08

Batched prompt ingestion on `v0.3-streaming`. See
`docs/REDLITE_DEV20_BATCHED_PREFILL.md`.

- Added `rl_engine_prefill()`: the Metal backend ingests a prompt in chunks
  (default 512 tokens, `--batch N`) with one dense command buffer per layer
  (batched norms/residuals, DeltaNet conv and delta-rule recurrences iterated
  inside single dispatches, full attention appending the chunk's keys/values
  then causal GQA per token and head, router, shared expert), f32
  dequantization plus a simdgroup-matrix GEMM for the dense matmuls, and one
  bounded-pool plan per chunk-layer holding the union of the selected experts
  (`REDMETAL_TOPK_MAX` 64 → 512; batched gate/up, down and per-token weighted
  sum kernels over (expert, token) pairs). `redlite-generate` and the chat use
  it for the prompt; the CPU oracle prefill is the sequential step sequence.
- Added `redlite-engine prefill MODEL --tokens ... [--batch N] [--cpu]`
  (batched vs token-by-token Metal, and vs the CPU oracle), `redlite-engine
  logits --batch N`, and two regression checks (`prefill.parity.chunks8`,
  `prefill.parity.96`); the long-context oracle check now ingests the first
  1100 positions batched. Suite: 36 checks (35 with `--quick`).
- Validation (M4 Max 48 GiB): router ids identical in all 48 layers, worst
  layer abs 1.9e-05, logits abs 3.1e-05 vs the sequential engine; CPU oracle
  logits abs 1.1e-05; llama.cpp long-context parity at chunk sizes 32–1100;
  greedy output token-identical.
- Prompt throughput on the M4 Max with the 4 GiB cache: 22.6 tok/s token by
  token → 65.7 tok/s (512-token chunks) → 99.1 tok/s (one 1100-token chunk);
  189 tok/s with the experts resident (12 GiB cache). Bound by copying the
  per-layer expert union into the pool (~6.7 GB/s from the page cache).
  Reading experts in place from the mmap (`RL_PREFILL_MAPPED_EXPERTS=1`) is
  numerically identical but 4–10× slower because Metal re-establishes
  residency of each layer's whole expert window per command buffer; kept as
  an opt-in experiment.
- dev20c (2026-09-08): profiling the expert phase showed the bounded-cache
  copy path was never the bottleneck (78 ms of miss copies for a 96-token
  prompt) while the LRU reservation cost 1.8 s: victim selection scanned every
  entry against the whole selection for each miss, and key lookups were linear
  in the capacity. `rl_native_lru_prepare_many` now reserves all hits before
  choosing victims and resident keys are indexed with an open-addressing hash
  (semantics unchanged, model-free LRU tests pass). Prompt ingestion on the
  M4 Max, 4 GiB cache: 96-token chunk 39 → 155 tok/s; 1100-token prompt 65.7 →
  171 tok/s (512-token chunks), capacity-independent (12 GiB: 163 tok/s).
  Decode also benefits slightly (warm 4 GiB: ~40 tok/s). `redlite-engine
  prefill` prints the expert-phase split (LRU, miss copies, commit, GPU wait).
- dev20d (2026-09-08): batched expert kernels re-laid out. Measured on the
  96-token chunk, register tiles over pairs (8-slot: 981 ms; adaptive 4/2/1
  with float4 loads: 370 ms) and over rows (295 ms) did not beat the 312 ms
  baseline; the bound was per-expert load imbalance (most experts serve one
  pair, a few serve 15–19). The kernels are now gridded over slices of up to
  four pairs of one expert with a 4-row register tile and per-group decoders
  (`rm_group8`, same arithmetic as the validated block dots): experts GPU
  312 → 188 ms on the 96-token chunk and 1361 → 826 ms on a 512-token chunk.
  Prompt ingestion on the M4 Max, 4 GiB cache: 1100-token prompt 249 tok/s
  (512-token chunks) / 271 tok/s (one chunk), 512-token chunk 294 tok/s;
  llama.cpp parity at every size, regress_m4.sh 36/36.

## 0.3.0.dev19 — 2026-09-07

Native chat integration on `v0.3-streaming`. See
`docs/REDLITE_DEV19_CLI_CHAT.md`.

- Added `redlite chat [MODEL.gguf]`, a user-facing entry point for the persistent
  native Red Metal runtime introduced in dev18; when omitted, the model path
  resolves to the repository's standard Qwen3-Next file in `models/`.
- Exposed context, expert-cache size, per-answer token limit, system prompt,
  temperature, top-k, top-p, seed, streaming and statistics through stable CLI
  options with conversational defaults.
- Added native runtime discovery to `redlite doctor` and a clear `make native`
  recovery message when the generator has not been built.
- Added parser and command-construction tests while keeping `redlite run` and
  `redlite serve` backward-compatible with the pinned upstream engines.

Hardening round (2026-09-08), from the post-dev18 audit:

- Native defects fixed: `redlite-generate` refuses an empty prompt instead of
  sampling from uninitialised logits and checks its allocations; the tokenizer's
  special-token fragment array is sized for the worst case (`2·len+3`); the GGUF
  reader rejects a `tokenizer.ggml.token_type` array whose length differs from
  the vocabulary (as llama.cpp does) and reports BF16 row bytes correctly; the
  engine rejects `ssm_conv < 2`; the Metal backend no longer allocates the unused
  267 MiB host-side copy of the conv/recurrent/KV state; the CPU oracle runs a
  row job inline when `pthread_create` fails; `redlite-engine --help` exits 0.
- Sampler: top-k uses a single-pass partial selection instead of sorting the
  whole vocabulary per token, and the stage order now matches llama.cpp's
  default chain (top-k → top-p on the untempered distribution → temperature).
  A model-free sampler selftest was added to `redlite-engine-offline-test`.
- Validation: `tests/fixtures/tokenizer_corpus.txt` (30 inputs) is compared
  against `llama_tokenize` in one model load per tool (`--file`); the llama.cpp
  logits comparison now gates on max-abs ≤ 1e-2 and KL ≤ 1e-5 in addition to the
  argmax; the frozen 1200-token fixture `tests/fixtures/long_context_prompt.txt`
  validates the > 1024-key attention path against llama.cpp at positions
  1100–1199 (`--dump-from` / `--dump-last` and `--router-layer` on both dump
  tools; argmax identical at every position, with a documented exact router tie
  at position 1035 that the implementations break differently); the greedy
  comparison derives the prompt length from the tokenizer instead of a
  hard-coded 19. The suite is now 34 checks (33 with `--quick`).
- CI: the M4 workflow's oracle step used to skip silently because the workspace
  checkout never contains `.deps/llama.cpp`; it now reads the bootstrapped
  checkout from the `REDLITE_LLAMA_DIR` repository variable, fails when that
  path is configured but invalid, and emits a warning annotation when unset.
- Documentation corrections: routed expert payload is ~16.9 GiB (not 22 GiB);
  physical footprints were MiB/1000 mislabelled as GiB (5940 MiB = 5.8 GiB,
  4485 MiB = 4.4 GiB, 8583 MiB = 8.4 GiB); the 70 % hit-rate figure belongs to a
  2 GiB cache (1 GiB gives 56 %); the 25.9 tok/s progression rows were measured
  with an 8 GiB cache; the chat template is hard-coded, not interpreted from the
  GGUF; the CPU oracle accumulates row dots in double but carries float32 state.
- Test machine: this round was validated on an **Apple M4 Max with 48 GiB**
  (the M4 Pro 24 GiB of the dev18 record is no longer the local machine). On
  the M4 Max the same 19-token prompt generates at 34–37 tok/s with the
  default 4 GiB cache (29–31 ms step, ~13 ms of it expert miss loading from
  the page cache), 33.6 tok/s with 8 GiB, 39–40 tok/s in fully warm 40-token
  runs, and ~21 tok/s when misses come from the SSD; hit rates and SSD bytes
  per token are prompt-determined and identical to the M4 Pro record.
  Recorded in `benchmarks/m4max-48gb-native-dev19.json`. The 24 GiB
  memory-pressure conclusions of dev18 are unaffected but were not re-measured.
- Resident ceiling on the M4 Max: with a 17 GiB expert cache and the experts
  of the previous turn resident, the third identical chat turn decodes at
  42.6 tok/s (24 ms step); the pinned llama.cpp fully resident measures
  68.0 tok/s decode and 338.6 tok/s prompt processing (`llama-bench`, pp48 /
  tg64). Expert prefetch therefore caps at ~20 % on decode here; the 8–13×
  prompt-ingestion gap makes batched prefill the dev20 milestone.

## 0.3.0.dev18 — 2026-09-03

Native end-to-end Qwen3-Next inference on `v0.3-streaming`. See
`docs/REDLITE_DEV18_ENGINE.md`.

- Added `rl_engine`, a persistent native runtime: mmap'd GGUF, audited per-layer
  tensor table, and two independent stateful backends (CPU oracle with
  double-precision row dots and float32 state, and Metal) with persistent DeltaNet conv/recurrent states and
  full-attention KV caches carried across tokens.
- Added a complete GGUF directory/metadata reader (hyper-parameters, tokenizer
  vocabulary/merges/types, chat template) and scalar CPU decoders for Q8_0,
  Q2_K (token embedding), Q4_K, Q5_K (LM head), Q6_K and IQ2_XXS; real rows
  match gguf-py's dequantization exactly.
- Added the model input/output path: Q2_K embedding lookup, final RMSNorm and
  the untied Q5_K LM head producing all 151 936 logits.
- Added a native byte-level BPE tokenizer (gpt2/qwen2 pre-tokenizer, Unicode
  tables generated from the pinned llama.cpp) and the Instruct chat template;
  identical ids to `llama_tokenize` on 26 inputs.
- Added `redlite-generate`: prompt → template → tokens → prefill → decoder →
  logits → sampler (greedy, temperature, top-k, top-p, seed) → text, stopping on
  `<|im_end|>` / `<|endoftext|>` / `--max-tokens`, with `--stats`
  (timings, expert-cache hit rate, SSD bytes per token, peak RSS, physical
  footprint).
- Added `redlite-generate --interactive`: a persistent terminal chat that loads
  the model once, appends ChatML user/assistant turns to the live native engine,
  preserves DeltaNet/KV state, and supports `/reset`, `/help` and `/quit`.
- Added `redlite-engine` diagnostics: `info`, multi-token CPU-vs-Metal `parity`,
  `logits` activation dumps, `tokenize`.
- Fixed the DeltaNet key/value head pairing: value head `h` now uses key head
  `h / (H_v / H_k)` (pinned llama.cpp repeat-interleave) instead of
  `h % H_k` in the dev15 oracle and Metal kernels; found by the llama.cpp
  activation comparison, re-validated with the dev15 tools.
- Rewrote top-k expert execution as three batched SIMD-lane dispatches (gate+up+SiLU,
  down, weighted sum) with a Metal 3 argument buffer of slot addresses, keeping
  the validated IQ2_XS/IQ1_M block decode; added an encode-into-external-command-buffer
  path so the engine issues one GPU sync per layer.
- Metal engine kernels: SIMD-lane dense row kernels for every dense quant type,
  threadgroup RMSNorm/residual norms, threadgroup-per-head GQA with chunked online
  softmax (any context length), fused DeltaNet state kernel (decay, delta, in-place
  update, output), fused attention q/k norm + RoPE + KV append, concurrent expert
  miss loads.
- Validation on the M4 Pro: 48-layer engine parity over stateful token
  sequences (worst logits abs 1.5e-05, identical router selections and argmax);
  pinned llama.cpp comparison of every layer output, final norm and logits
  (cosine 1.000000, KL ≤ 3e-12, identical top-5); greedy generation identical to
  llama.cpp for 28 tokens (short prompt) and 96 tokens after a 70-token prompt.
- Performance on the M4 Pro (greedy): decode step 39–43 ms, generation
  25.9 tok/s (short prompt, warm 8 GiB expert cache) to 27.8 tok/s (short
  prompt, default 4 GiB cache) / 24.4 tok/s (166-position run, warm 8 GiB
  cache), prompt ingestion 14–20 tok/s token-by-token; expert cache hit rate
  79–95 %, 17–71 MiB SSD expert traffic per token; physical footprint 4.4 GiB
  (4 GiB cache) to 8.4 GiB (fully populated 8 GiB cache).
  Recorded in `benchmarks/m4pro-24gb-native-dev18.json`.
- Added `scripts/regress_m4.sh` (31 checks incl. the pinned llama.cpp
  tokenizer/logits/greedy comparisons) and the corresponding self-hosted M4
  workflow steps; `make native` builds the engine and its offline test.
- Refactored the dev17 decoder-stack loop into `rl_decoder_stack_parity_execute()`.

## 0.3.0.dev17 — 2026-09-03

- Audited 48-layer decoder-stack diagnostic (`redlite-decoder-stack`): dispatches
  the 36 recurrent and 12 full-attention validated blocks from the real GGUF layer
  map and propagates CPU and Metal hidden vectors independently
  (stack max abs 2.38e-04). See `docs/REDLITE_DEV17_DECODER_STACK.md`.

## 0.3.0.dev16 — 2026-09-03

- Real full-attention tensor audit, complete single-token full-attention branch
  (input norm, joint Q/gate projection, K/V, Q/K norm, NeoX RoPE, GQA, KV cache,
  sigmoid gate, Q4_K output) with CPU/Metal parity at contexts 1, 2 and 16, and
  the complete full-attention transformer block. See `docs/REDLITE_DEV16_FULL_ATTENTION.md`.

## 0.3.0.dev15 — 2026-09-03

- Gated DeltaNet bring-up: projection, prestate (conv, L2 norms, beta/gate),
  recurrent state update, gated tail with Q4_K output, complete DeltaNet layer
  and the complete recurrent transformer block, each with an independent CPU
  oracle and M4 field validation. See `docs/REDLITE_DEV15_*.md`.

## 0.3.0.dev14 — 2026-09-03

Shared-expert bring-up milestone on `v0.3-streaming`.

- Recorded real dev13 router field validation: F32 router CPU/Metal parity passes on representative IQ2_XS and IQ1_M layers, ordered top-10 IDs match exactly, and router-selected routed FFN parity passes with zero SSD reads during Metal compute.
- Verified the pinned Qwen3-Next shared-expert formula: `down(SiLU(gate(x)) * up(x))`, multiplied by a separate `sigmoid(ffn_gate_inp_shexp(x))` scalar gate, then added to the routed MoE output.
- Added `redlite-shared-audit`, a standalone native C scanner for exact `ffn_gate_inp_shexp`, `ffn_gate_shexp`, `ffn_up_shexp` and `ffn_down_shexp` tensors.
- The audit reports layer completeness, hidden/shared-FFN dimensions, per-kind GGML type counts, physical spans and offsets, while keeping the dev13 routed path untouched.
- Shared arithmetic is intentionally not implemented until the target GGUF audit reveals the real shared tensor formats and width.

## 0.3.0.dev13 — 2026-09-03

Native Qwen3-Next router bring-up on `v0.3-streaming`.

- Field-audited all 48 real `blk.N.ffn_gate_inp.weight` tensors from the target Bartowski Qwen3-Next-80B-A3B GGUF: every router is F32, shape `(2048, 512)`, exactly 4.000 MiB per layer / 192.000 MiB total.
- Verified pinned llama.cpp routing semantics: F32 router matvec -> softmax over 512 experts -> top-10 -> selected-weight renormalization (`norm_w=true`) -> routed expert weighted sum. Qwen3-Next does not load an extra expert-weight scale, so the generic default causes no post-normalization scaling.
- Added an independent native C F32 router oracle using positional GGUF reads and 2048x512 matrix-vector evaluation.
- Added correctness-first Metal F32 router execution reading the real 4 MiB router directly into shared `MTLBuffer` memory and producing all 512 logits.
- Added native stable softmax, deterministic descending top-k selection, and selected-probability renormalization.
- Added portable synthetic tests for softmax/top-k/renormalization semantics and deterministic tie breaking.
- Added `.deps/redmetal/redlite-router` on macOS with `router-parity` to compare all 512 CPU/GPU logits plus ordered top-10 IDs and normalized weights.
- Added `routed-parity`, which feeds the real GPU-router-selected expert IDs/weights into the already field-validated native resident Metal expert executor and compares the routed output against the independent C expert reference.
- `routed-parity` intentionally excludes Qwen3-Next's separate shared-expert branch; shared-expert integration is the next milestone after real-router parity passes.

## 0.3.0.dev12 — 2026-09-02

Offline native hardening milestone on `v0.3-streaming`.

- Reworked the native top-k LRU into a two-phase prepare/load/commit flow: new expert mappings are not published until every selected miss has loaded successfully.
- Added abort semantics for failed replacement loads. Any touched miss slot is invalidated rather than restoring metadata for previous expert bytes that may have been partially or fully overwritten.
- Kept `rl_native_lru_acquire_many()` as a metadata-only convenience wrapper implemented through the new transaction API.
- Updated the standalone native Metal wrapper to reserve all top-k slots, load only misses, abort the LRU transaction on any `pread` failure, and publish the complete selection only after all loads succeed.
- Added a portable native offline integration test that writes a temporary GGUF v3 fixture in C and runs the production parser, routed ExpertMap, layer-info and expert-layout code against it.
- The synthetic fixture validates alignment metadata, routed gate/up/down detection, outer expert slicing, payload/triplet sizing, layer dimensions and expert-specific physical offsets without Python or a real model file.
- Added a native LRU fault-path test proving that prepared misses are invisible before commit, abort removes metadata for potentially overwritten victim bytes, and a subsequent retry can commit cleanly.
- `make native` now runs both `redlite-native selftest` and `redlite-native-offline-test` on macOS and Linux CI.
- No new Qwen graph functionality is introduced in dev12; actual router integration remains intentionally blocked on real-model field validation of the standalone native path.

## 0.3.0.dev11 — 2026-09-02

Native CPU/GPU parity milestone on `v0.3-streaming`.

- Added a standalone C CPU reference for routed IQ2_XS and IQ1_M row-dot evaluation, using double-precision accumulation and the dev10 embedded canonical quant grids.
- Added native expert reference execution for gate, up, stable `SiLU(gate) * up`, selected down rows and router-weighted top-k accumulation using explicit positional model reads.
- Extended `redlite-native selftest` with exact synthetic arithmetic fixtures: IQ2_XS all-ones decodes to dot `256.0`, while the known IQ1_M block decodes to dot `32.0` for unit inputs.
- Added `redlite-native topk-parity MODEL`, which runs the standalone native Metal top-k path and the standalone native CPU oracle in one process and reports max absolute/relative error plus per-row deltas.
- The native parity command uses the same deterministic input, non-contiguous expert ids, FP32 router weights and tolerance policy as the earlier Python dev8 oracle.
- Python is no longer required for GGUF parsing, expert mapping, LRU scheduling, quant codebook loading, Metal top-k execution or the CPU correctness comparison in the new standalone path.
- Real-model parity remains intentionally pending until the target M4 Pro is available again; CI validates compilation and synthetic CPU arithmetic on both macOS and Linux.

## 0.3.0.dev10 — 2026-09-02

Standalone native Metal integration milestone on `v0.3-streaming`.

- Linked the standalone `redlite-native` executable directly to the resident Red Metal top-k implementation on macOS; the `topk-probe` path contains no Python interpreter and no `ctypes` boundary.
- Added native routed-layer metadata validation for hidden size, FFN size, expert count and common gate/up/down quant type.
- Added compact canonical IQ2_XS and IQ1_S/IQ1_M codebook literals derived from the pinned llama.cpp/GGML source, expanded entirely in native C at runtime.
- Updated `NOTICE.md` to explicitly attribute the retained quant-grid data under the upstream MIT license.
- Added a native Metal runtime wrapper that connects the C top-k-aware LRU to expert miss loading, shared Metal residency and GPU router-weighted accumulation.
- Native execution checks that all selected expert ids are unique, the complete top-k fits in cache, resident slots are not in-flight before dispatch, and no GGUF reads occur during the synchronous Metal command.
- Added cumulative native telemetry for cache hits/misses/evictions, expert loads, SSD bytes/read calls, resident slots, Metal slab allocation and GPU execution time.
- Added `redlite-native topk-probe MODEL` using the same deterministic input, expert ids and normalized router weights as the field-validated dev8 Python oracle so output rows can be compared directly when the target M4 Pro is available again.
- The portable Linux build keeps the GGUF/LRU/codebook selftest but does not link Metal; macOS builds the complete standalone native Metal path.
- Real-model numerical validation of the new standalone path is intentionally deferred until the target Apple Silicon machine is available; dev8 remains the known-good numerical oracle in the meantime.

## 0.3.0.dev9 — 2026-09-02

Native runtime foundation milestone on `v0.3-streaming`.

- Added `redlite-native`, a standalone C executable that does not import or require Python.
- Ported the dependency-free GGUF v2/v3 tensor-directory parser and routed expert mapper from the Python oracle into native C.
- Native expert mapping reproduces merged gate/up/down tensor detection, outer 512-expert slicing, alignment-tail handling, per-expert byte strides, quant type identification and max triplet sizing.
- Added a native hard-bounded LRU metadata scheduler with top-k-aware `acquire_many()`: experts in the current selected set are protected from eviction while remaining misses are filled, and in-flight slots cannot be selected as victims.
- Added `redlite-native inspect MODEL` to report GGUF/routed layout, IQ2_XS/IQ1_M tensor and layer counts, routed payload, aligned slot size and cache capacity without loading Python.
- Added `redlite-native selftest` and `make native`; CI builds and runs the native selftest on both macOS and Linux.
- dev8 remains the numerical oracle for resident Metal top-k execution while the control plane is migrated out of Python. Metal codebooks/router integration and full token generation remain later native-runtime milestones.

## 0.3.0.dev8 — 2026-09-02

Resident routed top-k layer correctness milestone on `v0.3-streaming`.

- Added a native top-k Metal pool that keeps the selected expert set resident and pins all selected slots for the full routed-layer command buffer.
- Added GPU-side router-weighted accumulation; expert outputs no longer return through Python/CPU before the final routed-layer output.
- Added a top-k-aware hard-bounded global LRU. `acquire_many()` protects every expert in the current selection from eviction while remaining misses are loaded.
- Added `redlite-topk parity` with deterministic non-contiguous expert selection and FP32 router weights by default.
- The top-k executor reuses gate/up/activation/output scratch buffers across selected experts while keeping all expert weight triplets in their resident shared-Metal slots.
- Validation asserts zero GGUF reads during the top-k Metal command after residency; a cold top-10 selection should require exactly 30 positional reads (gate/up/down for ten experts).
- Added an independent CPU top-k reference that evaluates every selected expert FFN and performs the same weighted accumulation.
- This remains a scalar correctness-first path; router-network integration, selected-expert SIMD batching, asynchronous prefetch and SSD/GPU overlap remain later milestones.

## 0.3.0.dev7 — 2026-09-02

Resident single-expert FFN milestone on `v0.3-streaming`.

- Moved the field-validated IQ2_XS and IQ1_M kernels into a hard-bounded Metal resident expert pool.
- Added complete resident expert execution: gate matvec, up matvec, `SiLU(gate) * up`, and down matvec.
- Added `redlite-ffn parity` and an independent CPU reference for the complete quantized expert FFN.
- Field validation on Apple M4 Pro passed both routed formats with numerical parity, one resident expert load, three positional reads and zero SSD reads during FFN execution.

## 0.3.0.dev6 — 2026-09-02

Mixed routed-quant arithmetic validation milestone on `v0.3-streaming`.

- Added `quant-audit` and field-validated the actual routed tensor types in the Bartowski Qwen3-Next-80B-A3B IQ2_XXS GGUF: 33 tensors are IQ2_XS and 111 are IQ1_M.
- The 48 routed layers form two clean patterns: layers 0-5 and 43-47 are IQ2_XS for gate/up/down; layers 6-42 are IQ1_M for gate/up/down.
- Corrected the quant audit mapping for GGML type 29 to IQ1_M.
- Added canonical codebook loading from the pinned llama.cpp `gguf-py/gguf/quants.py` instead of maintaining duplicated giant quant tables in Red Lite.
- Added independent CPU row-dot references for IQ2_XS (256 values / 74-byte blocks) and IQ1_M (256 values / 56-byte blocks).
- Added correctness-first Metal row-matvec kernels for both routed formats with runtime dispatch by the tensor's actual GGML type.
- `iq2-parity` is retained as a backward-compatible command name but now auto-dispatches the real routed quant type and reports it explicitly.
- The dev5 cache/address-table/in-flight path remains part of the parity check; mixed-quant arithmetic is isolated until both formats pass numerical field validation, after which the kernels will be moved into the LRU execution pool.
- Added synthetic exact-value decoder tests for IQ2_XS and IQ1_M.

## 0.3.0.dev5 — 2026-09-02

First Red Metal arithmetic/parity milestone on `v0.3-streaming`.

- Added a separate native execution-pool ABI so the field-validated dev4 byte-visibility path remains intact while dev5 is tested.
- Added lazily allocated GPU-visible gate/up/down address tables for resident `(layer, expert)` entries.
- Added native per-slot in-flight tracking; expert loads refuse to overwrite a slot while Metal work still references it, and the dev5 LRU skips in-flight victims.
- Added a correctness-first IQ2_XXS Metal row-matvec kernel using the canonical ggml 256-value/66-byte block format and codebook.
- Added an independent pure-Python IQ2_XXS decoder/reference based on the canonical ggml grid representation and parity sign encoding.
- Added `redlite-stream iq2-parity` to validate real expert tensor shape, byte stride, GPU address binding, in-flight release and numerical GPU/CPU parity on selected rows.
- Added a synthetic IQ2_XXS reference test whose decoded matrix is exactly all ones.
- The kernel is intentionally scalar/correctness-first; SIMD-group optimization, fused gate/up/SwiGLU, down projection and top-k accumulation remain future work.

## 0.3.0.dev4 — 2026-09-02

First native Red Metal residency milestone.

- Added an Objective-C/Metal bridge compiled into `libredmetal.dylib`.
- Added lazily allocated `MTLStorageModeShared` slabs with fixed reusable expert slots and a hard byte budget.
- Routed expert misses can be read directly from the GGUF fd into `MTLBuffer.contents`, eliminating the production-path Python bytearray-to-Metal copy.
- Added `redlite-stream metal-probe` with a Metal compute checksum and independent CPU reread verification.
- Field validation on Apple M4 Pro confirmed 480 routed expert loads, 185 evictions in a 0.25 GiB Metal cache, and exact GPU/CPU checksum parity.

## 0.3.0.dev3 — 2026-09-02

Experimental DS4-aligned expert residency milestone on `v0.3-streaming`.

- Kept the validated Qwen3-Next GGUF expert mapper: 48 layers, 144 routed gate/up/down tensors, 512 experts and ~16.91 GiB routed payload on the reference IQ2_XXS model.
- Replaced mmap-based expert caching with one read-only model fd plus explicit positional `preadv`/`pread` into reusable RAM slots.
- Changed the cache key from individual tensor slices to complete `(layer, expert)` gate+up+down triplets.
- Added a hard byte budget: fixed-size slots are allocated lazily and recycled by global LRU eviction; Red Lite-owned expert slot memory cannot exceed the configured cache budget.
- Added asynchronous prefetch workers and separate demand-hit, demand-miss and prefetch-wait telemetry.
- Added SSD bytes-read, positional-read-call and throughput telemetry to `redlite-stream probe`.
- Added regression tests for direct positional reads, hard cache bounds, eviction/reuse, prefetch and thousands of expert accesses through a single fd.
- dev1's per-expert mmap design and dev2's whole-file mmap/madvise cache are superseded by this explicit-buffer path.
- Metal MoE binding is still intentionally disabled; dev3 validates storage/residency only.

## 0.2.1 — 2026-09-02

Field-validation tuning release based on a controlled Apple M4 Pro / 24 GiB sweep with the 17.97 GiB Qwen3-Next-80B-A3B IQ2_XXS model.

- Recorded repeatable 2K / 4K / 8K Metal sweep results.
- 2048: 258.8 prompt tok/s, 38.0 generation tok/s, +0.00 GiB swap.
- 4096: 247.2 prompt tok/s, 36.4 generation tok/s, +0.00 GiB swap.
- 8192: 236.7 prompt tok/s, 36.9 generation tok/s, +0.00 GiB swap.
- Changed the automatic default from 2K to 4K only for the field-validated Apple M4 Pro / 24 GiB / ~18 GiB resident profile.
- Kept 8K experimental because estimated policy headroom is only 0.28 GiB despite successful zero-swap completion.
- Kept planner status `CRITICAL`; observed success does not redefine the conservative headroom thresholds.
- Added robust macOS swap telemetry parsing and coverage for multiple `vm.swapusage` formats.
- Added a versioned benchmark record under `benchmarks/` and expanded the M4 Pro validation profile/docs.

## 0.2.0 — 2026-09-02

Field-hardening release after the first successful M4 Pro 24 GiB / Qwen3-Next 80B
resident Metal run.

- Added resident statuses: SAFE, TIGHT, CRITICAL, UNSAFE.
- Changed ~18 GiB / 24 GiB default context from 4096 to 2048.
- Added `redlite pressure` for macOS free-memory percentage and swap telemetry.
- Added `redlite sweep` for 2K/4K/8K context-depth benchmarking with JSON output.
- Added `--single-turn` to make scripted `llama-cli` runs exit after one answer.
- Increased default `redlite run` generation limit from 256 to 512 tokens.
- Added CRITICAL-margin warnings for run/server.
- Graceful Ctrl+C handling; interrupted Hugging Face downloads now explain resume.
- Updated Hugging Face install guidance for the modern `hf` CLI package.
- Fixed Bartowski IQ2 filenames inherited from 0.1.1.
- Improved `install.sh` PATH diagnostics for user Python installations on macOS.
- Added real M4 Pro field-observation profile and native Metal streaming roadmap.
