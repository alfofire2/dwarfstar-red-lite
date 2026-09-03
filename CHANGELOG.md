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
