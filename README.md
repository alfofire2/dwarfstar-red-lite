# DwarfStar Red Lite

**Qwen3-Next-80B-A3B on Apple Silicon Macs, with a native Metal runtime written for this one model.**

Red Lite is an independent project inspired by the narrow, hardware-aware philosophy of
DwarfStar/DS4. It is not a generic model runner. It targets one architecture and one
hardware family:

- **Model:** Qwen3-Next-80B-A3B-Instruct. The reference file is Bartowski's
  `Qwen_Qwen3-Next-80B-A3B-Instruct-IQ2_XXS.gguf` (17.97 GiB).
- **Hardware:** Apple Silicon, macOS only. Designed for 24 GiB of unified memory, and
  developed since September 2026 on a 48 GiB M4 Max.
- **Goal:** run an 80B-total / 3B-active sparse MoE locally without pretending that the
  whole model fits in memory.

Qwen3-Next has 48 layers: 36 Gated DeltaNet blocks and 12 full-attention blocks, each
followed by a 512-expert MoE that selects 10 experts per token. Only about 1 GiB of the
weights is dense. The other ~17 GiB are routed experts, of which a token touches about
3%. Red Lite keeps the dense part resident and treats the experts as a cache.

## Two runtimes

| | **Native runtime (0.3)** | **Launcher (0.2)** |
|---|---|---|
| What runs the model | Red Lite's own C11 / Objective-C / Metal implementation of the Qwen3-Next graph | a pinned llama.cpp (Metal), or a pinned CPU mmap runtime for oversized models |
| Memory model | dense weights mapped in place; routed experts in a bounded LRU cache (`--cache-mib`, 4 GiB by default on 24 GiB machines) | whole model resident, or CPU mmap with bounded expert residency |
| Entry points | `redlite chat`, `redlite serve --native`, `redlite-generate`, `redlite-server` | `redlite plan / run / serve / bench` |
| Correctness reference | pinned llama.cpp, used as a test oracle only and never linked | llama.cpp itself |
| Validated on | M4 Max 48 GiB (0.3.0); M4 Pro 24 GiB (dev18 build) | M4 Pro 24 GiB |

## Native runtime: quick start

```bash
make native                     # builds .deps/redmetal/ and runs the model-free self-tests
python3 -m pip install -U huggingface_hub && ./bin/redlite download 24gb --dir models

M=models/Qwen_Qwen3-Next-80B-A3B-Instruct-IQ2_XXS.gguf
.deps/redmetal/redlite-generate $M --prompt "Explain in one sentence why the sky is blue." --stats
./bin/redlite chat --stats      # persistent terminal chat; /reset, /help, /quit; Ctrl-C stops an answer
./bin/redlite serve --native $M --port 8080    # OpenAI-compatible /v1/chat/completions (SSE) and /v1/models
```

**Chat defaults.**

- Context: 4096 positions (`--context`).
- Answer length: 256 tokens.
- Temperature: 0.7 (`--temperature 0` is greedy and deterministic).
- Expert cache: chosen from RAM. With 40 GiB or more it is 22528 MiB: every expert is
  resident and preloaded, and tokens are routed on the GPU in one command buffer.
  Otherwise it is 4096 MiB.
- Prompts are ingested by a batched prefill in 512-token chunks (`--batch`).

**Sampler.** It follows llama.cpp's chain (top-k → top-p → min-p → temperature) with the
same arithmetic. min-p is off by default here and 0.05 in llama.cpp.

**Server.** It handles one request at a time and re-ingests the conversation on every
request.

## Measured performance

**Method.**

- Reference IQ2_XXS GGUF, greedy decode.
- Warm page cache: the whole GGUF fits in the 48 GiB machine's page cache.
- Decode is measured after a short prompt unless a position is given.
- Each number comes from one of the records listed under the table, with its command
  and commit.
- Nothing is extrapolated from one machine to another.

| Machine | Build | Expert cache | Decode, short context | Decode @ ~4096 | Decode @ ~8192 | Prompt ingestion |
|---|---|---|---:|---:|---:|---:|
| M4 Max 48 GiB | native 0.3.0 | 22 GiB (all resident) | 69.8–72.7 tok/s | 63.8–65.5 tok/s | 56.3–57.4 tok/s | 1100 tokens: 314.3 tok/s (4 GiB cache) |
| M4 Max 48 GiB | native 0.3.0 | 4 GiB | 46.5–50.1 tok/s | 42.3–43.6 tok/s | 38.8–39.2 tok/s | 4096 tokens: ~266 tok/s, 8192 tokens: ~203 tok/s |
| M4 Pro 24 GiB | native dev18 (`280f788`) | 4 GiB | 27.8 tok/s | not measured | not measured | 16.3 tok/s (token by token; there was no batched prefill yet) |
| M4 Pro 24 GiB | launcher 0.2 (llama.cpp Metal, resident) | n/a | 36.4–38.0 tok/s | — | — | 236.7–258.8 tok/s (256-token prompts) |

**Where the numbers come from.**

- **M4 Max ranges.** Two measurements of the same code. Short context: the dev23 record
  (03:40) and the 05:4x re-measurement of 2026-09-30. Long positions: two single-run
  repeats. The same build measured 4–7% lower two hours later on the same
  machine, cause not isolated (`docs/REDLITE_DEV26_LONG_CONTEXT.md`).
- **Long positions.** Single cooled runs, `scripts/dev/long_positions.sh`.
- **Prefill.** Median of three cooled runs, `scripts/dev/bench_m4.sh --cool 90`.
- **Records.**
  - `benchmarks/m4max-48gb-native-dev19.json`: dev19–dev26 sections.
  - `benchmarks/m4pro-24gb-native-dev18.json`.
  - `benchmarks/m4pro-24gb-sweep-2026-09-02.json`. The launcher's 4K/8K rows there are
    context sizes (36.4 / 36.9 tok/s), not decode positions.
- **Reference.** The pinned llama.cpp, fully resident on the M4 Max, gives 68.0 tok/s
  decode (`llama-bench` tg64) and 338.6 tok/s prompt (pp48).

**The 0.3.0 native runtime has not been measured on a 24 GiB M4 Pro.** Its 4 GiB
configuration is the one intended for 24 GiB machines. On the 48 GiB machine the page
cache holds the whole GGUF, so expert misses cost a memory copy. On 24 GiB some misses
will be SSD reads.

## Correctness

- **Greedy output** is token-identical to the pinned llama.cpp on the regression prompts
  (`scripts/regress_m4.sh`, 46 checks).
- **Logits** match llama.cpp:
  - after a 1100-token prompt: max-logit difference ≤ 2.0, KL ≤ 2e-2;
  - at 4096 and 8192 positions: every argmax agrees over 100 steps, KL ≤ 6.4e-3.
- **Kernels.** Every kernel change is gated by a CPU-oracle parity run: a double-precision
  implementation of the same graph in the same process. Any router top-k divergence fails
  the gate.
- **Model-free tests.** The GGUF readers are fuzzed. The model-free tests run under
  ASan/UBSan (`make sanitize`), and so does a real chat turn.
- **Details:**
  - `docs/REDLITE_DEV18_ENGINE.md` (the engine);
  - `docs/REDLITE_DEV22_DECODE_KERNELS.md` to `docs/REDLITE_DEV26_LONG_CONTEXT.md` (the
    0.3.0 performance work).

## Limits

- **One model, one layout.** Routed experts must be IQ2_XS or IQ1_M: the reference GGUF
  mixes both. Dense tensors must be F32, Q8_0, Q2_K, Q4_K, Q5_K, Q6_K or IQ2_XXS. Other
  GGUFs of the same model are not supported by the native runtime (the launcher handles
  them).
- **One sequence.** No multi-sequence batching. The server serves one request at a time.
- **Full residency** (`--cache-mib 22528`, GPU-routed decode) needs a Mac with at least
  40 GiB of RAM.
- **Throughput depends on the page cache.** On a machine whose page cache cannot hold the
  GGUF, expert misses become SSD reads. The expert prefetch (`RL_ENGINE_PREFETCH=0` turns
  it off) may help less there, or hurt.
- **Long contexts slow down.** Decode attention is linear in the position. The batched
  prefill's attention was not optimized, so prompt ingestion falls from ~314 tok/s at
  1100 tokens to ~203 tok/s at 8192.
- **Quantization.** IQ2_XXS is a very low-bit quantization. Red Lite reproduces llama.cpp
  on this file; it does not improve the file's quality.

## Tools

| Status | Tools |
|---|---|
| **Product** | `redlite chat`, `redlite serve --native`, `redlite-generate`, `redlite-server`, `redlite-engine` (`info`, `tokenize`, `logits`, `parity`, `prefill`, `kernel-selftest`) |
| **Launcher** (0.2 path, M4 Pro-validated) | `redlite doctor / plan / run / serve / bench / sweep / bootstrap / download` |
| **Regression** | `scripts/regress_m4.sh`; `scripts/dev/quick_parity.sh`, `bench_m4.sh`, `long_positions.sh`, `sanitize_chat.sh`; the stage parity CLIs in `.deps/redmetal/` (`redlite-ffn`, `redlite-deltanet-*`, `redlite-attention*`, `redlite-decoder-stack`, `redlite-native topk-parity`, …). The stage CLIs validated each graph stage (dev9–dev17) and are kept as regression tools; `redlite-engine` supersedes them for inference. |
| **Legacy** | the Python streaming oracle `redlite-stream`, `redlite-ffn` (Python) and `redlite-topk` (dev1–dev8). It is frozen and kept as a numerical reference; its `--help` says so. |

# Launcher (0.2)

The sections below describe the llama.cpp-based launcher: the memory planner, the pinned
engines and the M4 Pro 24 GiB presets. It remains the field-validated path on the 24 GiB
M4 Pro.

## 24 GB recommended configurations

| Mode | Quant | Approx file size | Backend | Trade-off |
|---|---|---:|---|---|
| Fastest practical resident | IQ2_XXS | ~19.3 GB | Metal | Very low quant quality; M4 Pro 24 GiB field profile defaults to 4K, other unvalidated tight 24 GiB profiles remain at 2K |
| Slightly better quant | IQ2_XS | ~22.2 GB | SSD/CPU by default | Too tight for conservative 24 GB Metal budget |
| Quality profile | Q4_K_M | ~48.4 GB | SSD/CPU | Much better quant quality, heavy SSD traffic |

The planner uses the *actual GGUF file size*, not the marketing parameter count.

### Field-validated M4 Pro / 24 GiB presets

For the 17.97 GiB Bartowski IQ2_XXS model on Apple M4 Pro 24 GiB:

- **2K:** conservative
- **4K:** default
- **8K:** experimental

A controlled sweep completed all three depths with zero observed swap growth:

| Context | Prompt tok/s | Generation tok/s | Swap delta |
|---:|---:|---:|---:|
| 2048 | 258.8 | 38.0 | +0.00 GiB |
| 4096 | 247.2 | 36.4 | +0.00 GiB |
| 8192 | 236.7 | 36.9 | +0.00 GiB |

The planner still labels these fits `CRITICAL` because the estimated headroom remains below 1 GiB. See `docs/FIELD_VALIDATION_M4PRO_24GB.md`.

## Requirements

- Apple Silicon Mac (`arm64`)
- macOS
- Xcode Command Line Tools (`xcode-select --install`)
- CMake
- Git
- Python 3.10+
- Fast internal SSD strongly recommended for oversized mode
- Enough free disk for model + build trees

## Quick start

### 1. Install the Red Lite CLI

```bash
cd dwarfstar-red-lite
./scripts/install.sh
redlite doctor
```

Or without installing:

```bash
./bin/redlite doctor
```

### 2. Build the two pinned engines

```bash
redlite bootstrap
```

This builds:

1. a Metal-enabled pinned `llama.cpp` for resident Qwen3-Next;
2. a CPU-only pinned Oversized MoE Runtime for mmap + bounded expert residency.

Dependencies are placed in `.deps/` and are not committed to this project.

### 3. Download the 24 GB profile

```bash
python3 -m pip install -U huggingface_hub
redlite download 24gb --dir models
```

The `24gb` alias selects the IQ2_XXS build.

For a higher-quality oversized model:

```bash
redlite download quality --dir models
```

### 4. Ask the planner what to do

```bash
redlite plan models/Qwen_Qwen3-Next-80B-A3B-Instruct-IQ2_XXS.gguf
```

On the field-validated Apple M4 Pro / 24 GiB profile, the default is now:

```text
mode             : metal-resident
status           : CRITICAL
context          : 4096
budget pass      : YES
```

For other unvalidated tight 24 GiB machines, the planner remains conservative and may choose 2048.

For Q4_K_M the planner should choose:

```text
mode             : ssd-cpu
```

### 5. Run locally

```bash
redlite run models/Qwen_Qwen3-Next-80B-A3B-Instruct-IQ2_XXS.gguf \
  -p "Write a small Python HTTP server and explain it." \
  -n 512
```

### 6. Start an OpenAI-compatible server

```bash
redlite serve models/Qwen_Qwen3-Next-80B-A3B-Instruct-IQ2_XXS.gguf \
  --host 127.0.0.1 \
  --port 8080
```

Then call it as a normal local OpenAI-style endpoint:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "local",
    "messages": [{"role":"user","content":"Hello from Red Lite"}],
    "stream": false
  }'
```

## CLI

```text
redlite doctor
redlite pressure
redlite models
redlite bootstrap
redlite download [24gb|balanced|quality]
redlite plan MODEL.gguf
redlite chat [MODEL.gguf]
redlite run MODEL.gguf
redlite serve MODEL.gguf
redlite bench MODEL.gguf
redlite sweep MODEL.gguf --contexts 2048,4096,8192
```

### Override the execution path

```bash
redlite plan MODEL.gguf --mode metal-resident
redlite plan MODEL.gguf --mode ssd-cpu
```

Red Lite will reject a forced resident plan if its conservative RAM budget says the
model does not fit. `run`/`serve` expose `--force`, but it should be used only for
experiments.

## Why not just call this a DwarfStar fork?

DwarfStar currently has its own narrow tensor layouts and model implementations for
DeepSeek V4 / GLM. Qwen3-Next is a genuinely different graph: Gated DeltaNet,
periodic full attention and a 512-expert MoE. Pretending that changing a model enum
would make DS4 compatible would be misleading.

Red Lite instead adapts the **DwarfStar design approach**:

- one model family, not a generic UI wrapper;
- hardware-specific defaults;
- asymmetric treatment of hot/dense tensors vs routed experts;
- SSD-backed execution when RAM is insufficient;
- integrated local CLI/server experience;
- deterministic, pinned inference engines.

For mathematical correctness of Qwen3-Next, the first usable release (0.2) delegated the
model graph/kernels to a pinned llama.cpp revision that already implements the
architecture. 0.3 adds the native runtime, whose hand-written Gated DeltaNet, attention and
MoE kernels were each validated against a CPU oracle and the pinned llama.cpp before
being used.

## What is and is not validated

This section covers the launcher. For the native runtime, see *Measured performance*,
*Correctness* and *Limits* above.

Red Lite has completed real field validation on an **Apple M4 Pro with 24 GiB unified memory** using the 17.97 GiB Bartowski IQ2_XXS build of Qwen3-Next-80B-A3B-Instruct.

Interactive Metal inference succeeded at 2K with observed generation throughput of roughly **30.7–40.7 tok/s**. A later controlled sweep completed **2K, 4K and 8K with zero observed swap growth**, measuring approximately **38.0, 36.4 and 36.9 generation tok/s** respectively.

The Python control plane and memory-policy tests are also validated. The oversized Q4 path is based on a separately validated 48.41 GB Qwen3-Next run on a 16 GB M1. Field numbers are observations, not guarantees: macOS memory pressure depends on other processes, context size, build revision and GGUF layout.

See:

- `docs/FIELD_VALIDATION_M4PRO_24GB.md`
- `benchmarks/m4pro-24gb-sweep-2026-09-02.json`

## Development

Run tests:

```bash
make test
```

GitHub CI (`.github/workflows/ci.yml`) runs the model-free checks on Linux for every push
and pull request: ruff, `make native`, `make sanitize`, `make test`. Everything that needs
Metal or the model runs locally with `scripts/regress_m4.sh MODEL`.

Build a binary release tarball (Apple Silicon only; `-mcpu=apple-m1`, macOS ≥ 14):

```bash
scripts/package_release.sh          # dist/redlite-<version>-macos-arm64.tar.gz + .sha256
```

Inspect the launch command without executing it:

```bash
redlite run MODEL.gguf --dry-run
redlite serve MODEL.gguf --dry-run
```

See:

- `docs/ARCHITECTURE.md`
- `docs/DS4_ADAPTATION.md`
- `docs/MEMORY.md`
- `docs/BENCHMARK.md`
- `docs/METAL_STREAMING_ROADMAP.md`
- `NOTICE.md`

## License

Red Lite's own code is MIT licensed. Third-party engines retain their upstream MIT
licenses and notices. See `NOTICE.md`.
