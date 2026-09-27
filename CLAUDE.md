# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

DwarfStar Red Lite runs **one model family on one hardware family**: Qwen3-Next-80B-A3B
(sparse MoE, 80B total / ~3B active) on Apple Silicon Macs, targeting 24 GiB unified
memory. It is not a generic model runner. The reference GGUF is the Bartowski
`Qwen_Qwen3-Next-80B-A3B-Instruct-IQ2_XXS.gguf` (~17.97 GiB), normally kept in
`models/` (gitignored).

Two branches matter:

- `main` — the stable v0.2.x **launcher**: a Python control plane that picks a memory
  policy and shells out to pinned upstream engines (llama.cpp Metal, or a CPU mmap
  "oversized MoE runtime").
- `v0.3-streaming` — the experimental **native Red Metal runtime**: a from-scratch C /
  Objective-C / Metal implementation of the Qwen3-Next graph, built up one validated
  stage at a time. Most active work lives here.

Version is duplicated in `VERSION`, `pyproject.toml` and `redlite/__init__.py`; bump all three.

## Commands

```bash
./scripts/install.sh            # pip install -e . ; exposes redlite, redlite-stream, redlite-ffn, redlite-topk
./bin/redlite doctor            # run the CLI without installing (sets PYTHONPATH)
make test                       # PYTHONPATH=. python3 -m unittest discover -s tests -v
PYTHONPATH=. python3 -m unittest tests.test_planner.PlannerTests.test_24gb_18gib_model_prefers_metal   # single test
ruff check redlite tests --select E9,F63,F7,F82   # the only lint CI enforces (fatal errors only)
make redmetal                   # build .deps/redmetal/libredmetal.dylib (ctypes bridge for the Python dev CLIs; macOS only)
make native                     # build every standalone native executable into .deps/redmetal/ and run selftests
bash scripts/build_engine.sh    # rebuild just the engine (redlite-engine, redlite-generate, engine offline test)
bash scripts/build_decoder_stack.sh   # rebuild just one native tool (one script per tool, see scripts/build_*.sh)
scripts/regress_m4.sh MODEL [--quick] # complete M4 regression suite (38 checks + the GPU-routed decode check on >= 40 GiB machines; --quick skips the 48-layer stack and the 1200-token long-context check)
make sanitize                   # ASan+UBSan build and run of every model-free native test and the GGUF fuzz (macOS or Linux)
bash scripts/dev/build_ref_llama.sh   # dev-only oracle linked against the bootstrapped llama.cpp (never used at runtime)
make bootstrap                  # clone+build the pinned llama.cpp and oversized-moe-runtime into .deps/ (Apple Silicon only, slow)
```

Python tests never need a model file or Metal; they use synthetic fixtures and fake pools.
`make native` also runs model-free C tests (`redlite-native selftest`, the
`*-offline-test` binaries, `redlite-gguf-fuzz`). Any change to the GGUF readers must
keep `make sanitize` clean; the fuzz's invariants (seeds accepted, truncations
rejected, accepted mutants mappable) are the readers' contract. On Linux it builds only the portable subset and the
Metal-only `build_*.sh` scripts exit 0 with a skip message.

Real-model parity tools all follow the same shape and only work on macOS with the GGUF present:

```bash
.deps/redmetal/redlite-generate MODEL --prompt "..." --max-tokens 64 --cache-mib 4096 --stats   # native end-to-end generation (4 GiB cache is the recommended default on 24 GiB)
.deps/redmetal/redlite-engine parity MODEL --tokens 9707,11,1879 --cache-mib 1024 --context 64  # 48-layer CPU-vs-Metal multi-token parity
.deps/redmetal/redlite-engine parity MODEL --tokens 9707,11,1879 --cache-mib 22528 --repeat 2   # dev21 GPU-routed decode (warm/preloaded cache) vs CPU oracle; needs >= 40 GiB
.deps/redmetal/redlite-generate MODEL --prompt "..." --cache-mib 22528 --stats   # full residency: every expert preloaded, GPU-routed tokens (RL_ENGINE_SPECULATIVE=0 / RL_ENGINE_PRELOAD=0 disable)
.deps/redmetal/redlite-engine tokenize MODEL --text "..." --chat
.deps/redmetal/redlite-engine prefill MODEL --tokens 9707,11,1879 --batch 8 [--cpu]            # dev20 batched prefill vs token-by-token Metal (and CPU oracle)
.deps/redmetal/redlite-generate MODEL --prompt "..." --batch 512   # prompt chunk size for the batched prefill (default 512; 1 = token by token)
RL_ENGINE_PROFILE=1 .deps/redmetal/redlite-generate ...   # per-stage GPU time profile
.deps/redmetal/redlite-decoder-stack parity models/Qwen_Qwen3-Next-80B-A3B-Instruct-IQ2_XXS.gguf --position 7 --top-k 10 --cache-mib 256
.deps/redmetal/redlite-attention-block parity MODEL --layer 3 --position 7 --top-k 10 --cache-mib 256
.deps/redmetal/redlite-deltanet-layer  parity MODEL --layer 0
.deps/redmetal/redlite-ffn             parity MODEL --layer 0 --top-k 10 --rows 8 --cache-mib 256
.deps/redmetal/redlite-layer-audit     MODEL --tensors
```

The full list of invocations that constitute "field validation" is the step list in
`.github/workflows/mac-m4-field-validation.yml` (self-hosted M4 Pro runner, reads
`REDLITE_MODEL_PATH`). When you add a new native stage, add its parity step there.

Environment variables: `REDLITE_REDMETAL_LIB` overrides the dylib path for the Python
bridge; `REDLITE_JOBS` sets bootstrap build parallelism; `REDLITE_LLAMA_DIR` points
`regress_m4.sh` and `scripts/dev/build_ref_llama.sh` at a bootstrapped llama.cpp checkout
outside `.deps/` (the M4 workflow reads it from a repository variable of the same name,
because the runner's workspace checkout is cleaned and never contains `.deps/llama.cpp`).
The tokenizer oracle corpus is `tests/fixtures/tokenizer_corpus.txt` (one input per line,
`\n` `\t` `\\` escapes) and the long-context oracle prompt is the frozen
`tests/fixtures/long_context_prompt.txt`; do not regenerate the latter casually, its known
router near-tie at position 1035 is documented in `docs/REDLITE_DEV18_ENGINE.md`.

## Architecture

### Three layers, from stable to experimental

1. **Python control plane** (`redlite/cli.py`, `hardware.py`, `planner.py`, `runner.py`,
   `telemetry.py`, `benchmark.py`, `model_catalog.py`). `plan_for()` measures the real
   GGUF byte size, subtracts a conservative macOS/runtime/context reserve
   (formula in `docs/MEMORY.md`), and chooses `metal-resident` (llama.cpp `-ngl 999`,
   q8 KV cache, no mlock) or `ssd-cpu` (oversized-moe-runtime, mmap + bounded expert
   residency). Resident fits are graded `SAFE / TIGHT / CRITICAL / UNSAFE` by headroom;
   the field-validated M4 Pro 24 GiB + ~18 GiB model profile is special-cased to
   default to 4K context, everything else tight stays at 2K. `runner.py` builds the
   engine command lines; `--dry-run` prints them. Engine args after the redlite args
   are passed through as a remainder.

2. **Python streaming oracle** (`expert_map.py`, `expert_store.py`, `expert_cache.py`,
   `streaming.py`, `quant_*.py`, `iq2_reference.py`, `redmetal*.py`; CLIs
   `redlite-stream`, `redlite-ffn`, `redlite-topk`). A dependency-free GGUF parser,
   per-expert byte-range mapping inside the merged `blk.N.ffn_{gate,up,down}_exps.weight`
   tensors, a hard-bounded LRU of expert slots, CPU reference row-dots for IQ2_XS/IQ1_M,
   and `ctypes` bindings to `libredmetal.dylib`. This was the dev1–dev8 path and remains
   the known-good numerical oracle; it is frozen, not extended.

3. **Standalone native runtime** (`native/`, dev9 onward). Pure C11 plus Objective-C
   Metal, no Python. Each milestone adds one graph stage with an independent CPU
   reference, a Metal implementation, and a `parity` CLI that runs both in one process
   and reports max abs/rel error, SSD bytes read during GPU compute (must be zero after
   residency), and per-substage timings.

4. **Persistent engine** (dev18: `redlite_native_engine*.{h,c}`, `redmetal_engine.m`,
   dev20 batched prefill: `redmetal_engine_prefill.m` + `redmetal_engine_private.h`,
   dev21 GPU-routed decode: `rl_route` kernel + `step_speculative` in `redmetal_engine.m`, residency table in the pool,
   `redlite_native_tokenizer.[ch]`, `redlite_native_sampler.[ch]`,
   `redlite_native_generate_cli.c`). `rl_engine` owns the mmap'd GGUF
   (`redlite_native_gguf_dir.[ch]`), the audited per-layer tensor table and two
   independent stateful backends: a CPU oracle (double-precision row dots, float32 state) and the Metal backend.
   Dense weights are wrapped in place from the mmap; routed experts go through the
   top-k LRU pool with one GPU sync per layer (prepare/encode/release API). The stage
   CLIs stay as regression tools; the engine reuses their kernel arithmetic.

### Native source conventions

File prefixes tell you what a file is:

- `redlite_native_<stage>.h/.c` — portable C: API + CPU reference implementation.
- `redmetal_<stage>.m` — Objective-C Metal kernels/pipelines for that stage (kernel
  source is embedded as strings inside the `.m` files).
- `redlite_native_<stage>_cli.c` — the standalone `parity`/audit executable.
- `redlite_native_<stage>_runtime.c` — reusable entry points extracted from a CLI so
  higher stages can compose it (see `redlite_native_decoder_block.h`,
  `redlite_native_decoder_stack.h`).
- `redlite_native_*_offline_test.c` — model-free tests that write a synthetic GGUF v3
  fixture in C and exercise the real parser/map/LRU code.

Stage dependency order (each builds on the ones before it):
`gguf` (parser) → `model` (expert map, layer info) → `cache` (two-phase prepare/load/commit LRU)
→ `tables` (embedded IQ2_XS/IQ1_S/IQ1_M codebooks, derived from pinned llama.cpp; see `NOTICE.md`)
→ `reference` (CPU quant row-dot and expert FFN) → `metal` + `redmetal_topk.m` (resident top-k expert executor)
→ `router` / `router_exec` (F32 router, softmax, top-10, renorm) → `shared` / `shared_exec` / `iq2_xxs` (shared expert branch)
→ `deltanet_proj` / `deltanet_prestate` / `deltanet_state` / `deltanet_tail` / `deltanet_layer` (Gated DeltaNet)
→ `attention_proj` / attention (GQA + RoPE + KV cache) → `block_ops` (residual, RMSNorm)
→ `recurrent_block_runtime` / `attention_block_runtime` (complete decoder blocks)
→ `layer_map` (audits the GGUF to classify each layer) → `decoder_stack` (48-layer trunk).

**There is no CMake for `native/`.** Every executable is a single `xcrun clang`
invocation in its own `scripts/build_<tool>.sh` that lists the `.c`/`.m` sources
explicitly. Adding a source file means adding it to every build script whose tool links
it, and adding a new tool means a new script plus a line in the `native` target of the
`Makefile`. Flags are `-std=c11 -Wall -Wextra -Wpedantic` with `-mcpu=native` and
`-fobjc-arc`; keep new code warning-clean under those.

`redmetal_topk.m` is shared between `libredmetal.dylib` (Python bridge) and the native
executables, so ABI-visible changes there affect both layers.

### Model facts the code depends on

- 48 layers, hidden 2048; 36 Gated DeltaNet ("recurrent") blocks and 12 full-attention
  blocks at layers `3,7,11,...,47`. The layer map is **audited from tensor names**, never
  assumed from the 3:1 interval.
- 512 routed experts, top-10, selected weights renormalized (`norm_w=true`), no extra
  expert scale. Router is F32 `(2048, 512)` per layer.
- Routed expert quant types in the target GGUF are mixed: layers 0–5 and 43–47 are
  IQ2_XS, layers 6–42 are IQ1_M. Dispatch is by the tensor's actual GGML type.
- DeltaNet pairs value head `h` with key head `h / (H_v / H_k)` (repeat-interleave,
  as in llama.cpp); `h % H_k` is wrong and was fixed in dev18.
- Token embedding is Q2_K, the LM head is an untied Q5_K `output.weight`; tokenizer
  is gpt2/qwen2 BPE with `<|im_end|>` (151645) as EOS and no BOS.
- Shared expert: `down(SiLU(gate(x)) * up(x)) * sigmoid(ffn_gate_inp_shexp(x))`, added
  to the routed output.
- Pinned llama.cpp (commit in `third_party/README.md` and `scripts/bootstrap_macos.sh`)
  is the semantic reference for all of the above. If you change a pin, update both
  places and re-run field validation before calling anything validated.

## Working conventions

- **The pinned llama.cpp is an oracle only.** `scripts/dev/ref_llama/` dumps
  activations/logits/greedy tokens for comparison; it must never be linked into a
  runtime binary, and Python stays development-only.
- **Honesty about validation is a project rule.** Docs and CHANGELOG distinguish
  "implemented", "synthetically tested", and "field-validated" **naming the machine**:
  the dev18 record and the 24 GiB memory policy come from an Apple M4 Pro / 24 GiB;
  since 2026-09-08 the local test machine is an Apple M4 Max / 48 GiB (`redlite doctor`
  prints which one you are on). Never attribute a number to the M4 Pro unless it was
  measured there, and prefer running `scripts/regress_m4.sh` locally over the GitHub
  workflows. Do not
  describe a stage as validated, or claim GPU expert streaming / throughput, unless a
  real-model parity run on Apple Silicon has passed. Elapsed times from parity tools are
  not benchmarks.
- Each milestone (`dev15e`, `dev16`, `dev17`, …) gets a `docs/REDLITE_DEVnn_*.md` with
  a "Scope boundary" section, a CHANGELOG entry, and commits prefixed like
  `dev17:` / `feat(native):` / `ci:` / `docs:`.
- A top-k router ID divergence between CPU and Metal is always a hard failure; small
  float drift is compared cumulatively across the stack, not per substage.
- Benchmarks and field observations are recorded as JSON in `benchmarks/` and
  `configs/mac-m4pro-24gb-observed.json`; keep GiB (binary) in Red Lite output even
  though model hosts quote decimal GB.
