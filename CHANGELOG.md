# Changelog

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
