# Changelog

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
