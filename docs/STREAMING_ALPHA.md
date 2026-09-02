# Red Streaming Engine — v0.3 alpha

This branch introduces the first native substrate for SSD-backed routed experts on Apple Silicon.
It is intentionally **not yet wired into token generation**. The alpha exists to validate the
hard parts that can be measured independently before patching the Qwen3-Next execution graph.

## What is implemented

1. A GGUF v2/v3 metadata and tensor indexer (`redlite.gguf`).
2. Detection of Qwen3-Next routed expert tensors (`ffn_down_exps`, merged `ffn_gate_up_exps`, or separate gate/up tensors).
3. Per-layer/per-expert slice manifests using the expert dimension stored as the final tensor axis.
4. A bounded reference LRU using `os.pread()` for policy tests.
5. A native Objective-C++ cache that reads expert slices with `pread()` directly into `MTLStorageModeShared` buffers.
6. Chunked Metal allocations so a future 4–6 GiB cache does not depend on one giant `MTLBuffer`.
7. A background prefetch queue, LRU eviction, hit/miss telemetry, bytes-read and raw pread throughput.
8. Optional `F_NOCACHE` probe mode.

## Commands

```bash
redlite stream-build
redlite expert-index MODEL.gguf
redlite stream-plan MODEL.gguf --cache-gib 4
redlite stream-probe MODEL.gguf --cache-mib 1024 --requests 5000 --hotset 1024 --prefetch 4
redlite stream-probe MODEL.gguf --cache-mib 1024 --requests 5000 --nocache
```

The probe uses deterministic 80/20 hotset traffic. It is **not** presented as a Qwen router trace.
Its purpose is to validate address correctness, bounded unified-memory allocation, asynchronous
prefetch behavior and storage bandwidth on the real GGUF.

## Gate to full inference integration

Before enabling `metal-streaming` as a `redlite run` mode:

- `expert-index` must report the expected 48 layers and 512 experts/layer.
- The native cache must build and allocate cleanly on the target M4 Pro 24 GiB.
- All real expert slice requests must complete without short reads/address errors.
- `--nocache` throughput must be measured at several cache sizes.
- A router-derived trace must replace the synthetic workload.
- llama.cpp graph integration must substitute selected expert slices without CPU/GPU duplication.

Only after these gates will the experimental mode generate tokens.
