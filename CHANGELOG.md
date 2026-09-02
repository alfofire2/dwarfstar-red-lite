# Changelog

## 0.3.0-alpha.1 — 2026-09-02

First Red Streaming Engine substrate. This alpha does not yet replace llama.cpp's routed-expert execution during token generation.

- Added GGUF v2/v3 tensor indexer and Qwen3-Next routed-expert discovery.
- Added per-layer/per-expert slice manifests.
- Added bounded reference `os.pread()` LRU cache.
- Added native Objective-C++ SSD -> `MTLStorageModeShared` expert cache with chunked buffers.
- Added asynchronous expert prefetch queue and LRU eviction.
- Added hit/miss, eviction, bytes-read and pread-throughput telemetry.
- Added optional `F_NOCACHE` storage probe mode.
- Added `expert-index`, `stream-plan`, `stream-build`, and `stream-probe` commands.
- Preserved the M4 Pro 24 GiB field-validated 4K resident default from v0.2.1.
- Expanded the development test suite to 20 tests.

## 0.2.1 — 2026-09-02

Field-validation tuning release based on a controlled Apple M4 Pro / 24 GiB sweep with the 17.97 GiB Qwen3-Next-80B-A3B IQ2_XXS model.

- 2048: 258.8 prompt tok/s, 38.0 generation tok/s, +0.00 GiB swap.
- 4096: 247.2 prompt tok/s, 36.4 generation tok/s, +0.00 GiB swap.
- 8192: 236.7 prompt tok/s, 36.9 generation tok/s, +0.00 GiB swap.
- Changed the automatic default from 2K to 4K only for the validated M4 Pro 24 GiB profile.
- Kept 8K experimental and planner status `CRITICAL`.
- Added robust macOS swap telemetry parsing.

## 0.2.0 — 2026-09-02

Field-hardening release after the first successful M4 Pro 24 GiB / Qwen3-Next 80B resident Metal run.
