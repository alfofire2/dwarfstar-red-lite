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
| Fastest practical resident | IQ2_XXS | ~19.3 GB | Metal | Very low quant quality; 2K is the conservative 24 GiB default, then validate 4K/8K with `redlite sweep` |
| Slightly better quant | IQ2_XS | ~22.2 GB | SSD/CPU by default | Too tight for conservative 24 GB Metal budget |
| Quality profile | Q4_K_M | ~48.4 GB | SSD/CPU | Much better quant quality, heavy SSD traffic |

The planner uses the *actual GGUF file size*, not the marketing parameter count.

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

Typical 24 GB decision:

```text
mode             : metal-resident
status           : CRITICAL
context          : 2048
budget pass      : YES
```

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

Red Lite has completed real field validation on an **Apple M4 Pro with 24 GiB unified memory** using the 17.97 GiB Bartowski IQ2_XXS build of Qwen3-Next-80B-A3B-Instruct. Interactive Metal inference succeeded at 2K context with observed generation throughput of roughly **30.7–40.7 tok/s** during that session.

The Python control plane and memory-policy tests are also validated. The oversized Q4 path is based on a separately validated 48.41 GB Qwen3-Next run on a 16 GB M1. Field numbers are observations, not guarantees: macOS memory pressure depends on other processes, context size, build revision and GGUF layout. See `docs/FIELD_VALIDATION_M4PRO_24GB.md`.

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
