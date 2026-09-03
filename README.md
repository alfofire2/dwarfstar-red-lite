# DwarfStar Red Lite

**Apple-Silicon-only runtime profile for running Qwen3-Next-80B-A3B on memory-constrained Macs.**

Red Lite is an independent project inspired by the narrow, hardware-aware philosophy
of DwarfStar/DS4. Instead of trying to be a universal model runner, it targets one
architecture and one hardware family:

- **Model family:** Qwen3-Next
- **Reference model:** Qwen3-Next-80B-A3B-Instruct
- **Hardware:** Apple Silicon macOS only
- **Target machine:** 24 GB unified memory
- **Goal:** make an 80B-total / 3B-active sparse MoE usable locally without pretending
  that a 48 GB model magically fits in 24 GB RAM.

## What makes it "Red Lite"

Qwen3-Next-80B-A3B has 80B total parameters but activates about 3B per token. It has
48 layers, 512 routed experts and selects 10 experts per token. Its hybrid stack uses
three Gated DeltaNet / MoE blocks for each full-attention / MoE block.

That makes it a better 24 GB target than a dense 80B model: the model's *capacity* can
live on SSD while only a working set must be resident for oversized operation.

Red Lite therefore has two deliberately different execution paths:

```
                 +-----------------------------+
                 | Qwen3-Next-80B-A3B GGUF     |
                 +--------------+--------------+
                                |
                         memory planner
                        /               \
                       /                 \
      working set fits                    model > budget
             |                                  |
             v                                  v
 +-----------------------+          +--------------------------+
 | Metal resident mode   |          | SSD-residency mode       |
 | llama.cpp + Metal     |          | mmap + bounded expert LRU|
 | all useful GPU layers |          | Apple CPU / Accelerate   |
 +-----------------------+          +--------------------------+
```

The fallback is intentional. Current oversized Qwen3-Next execution has a validated
CPU mmap/residency path, whereas forcing a too-large model into Metal unified memory
can trigger severe memory pressure or OOM. Red Lite prioritizes *actually completing
inference* over claiming every mode is GPU accelerated.

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

## Native inference (v0.3 development branch)

The `v0.3-streaming` branch adds a native Qwen3-Next runtime that does not use
llama.cpp or Python at inference time. It keeps the ~1.06 GiB of dense weights
resident (mapped in place from the GGUF), streams the 22 GiB of routed experts
through a bounded Metal-visible LRU cache, and runs the whole 48-layer hybrid
DeltaNet / full-attention / MoE stack on Metal.

```bash
make native
.deps/redmetal/redlite-generate models/Qwen_Qwen3-Next-80B-A3B-Instruct-IQ2_XXS.gguf \
  --prompt "Explain in one sentence why the sky is blue." --max-tokens 64 --cache-mib 8192 --stats
```

Options: `--system`, `--raw`, `--max-tokens`, `--temperature`, `--top-k`,
`--top-p`, `--seed`, `--context`, `--cache-mib`, `--no-stream`, `--stats`,
`--tokens-out`. Greedy output is token-identical to the pinned llama.cpp on the
validated prompts; on the M4 Pro / 24 GiB it generates at roughly 24–26 tok/s
with an 8 GiB expert cache (physical footprint ~8.6 GiB). Details, validation
numbers and limits: `docs/REDLITE_DEV18_ENGINE.md`; regression suite:
`scripts/regress_m4.sh MODEL.gguf`.

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

For mathematical correctness of Qwen3-Next, the first usable release delegates the
model graph/kernels to a pinned llama.cpp revision that already implements the
architecture. This avoids shipping an unvalidated hand-written Gated DeltaNet kernel.

## What is and is not validated

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
