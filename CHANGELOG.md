# Changelog

## Unreleased (after 0.4.0, branches `dev/iq3-kernels` … `dev/iq3m`)

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
