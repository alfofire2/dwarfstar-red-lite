# Red Lite dev26 (partial) — model-free robustness: GGUF fuzz, sanitizers, portable build

Status: **partial**. This document covers only the dev26 items that need neither
Metal nor the model: the sanitizer run of the model-free tests, a GGUF reader fuzz,
and the fixes they led to. It also covers a fix to the portable (Linux) build. The
other dev26 items are not done: 4096/8192-position parity and benchmarks, sampler
distributional parity against llama.cpp, and model-free tests for the dev22–dev24
kernels, which do not exist yet. See "Scope boundary".

## Why

The native runtime opens an 18 GiB file and reads counts, lengths and offsets from
it before anything else, and until now only well-formed fixtures had tested that
code. `make native` on Linux (the portable subset that CLAUDE.md documents) had
also stopped building at some point after dev15, without anyone noticing.

## What changed

- **Portable build restored** (`2ed500b`). `redlite_native_deltanet_{state,tail}_cli.c`
  used the Metal telemetry struct outside its `#ifdef __APPLE__`. The Linux build
  is now also warning-clean under `-Wall -Wextra -Wpedantic`: the misleading
  indentation in the prestate CLI is split, and the Linux `redlite-native` link
  gets the same `-Wno-overlength-strings` as the macOS one.
- **GGUF reader hardening** (`ee80ff7`). The fuzz found three defects:
  1. `rl_native_build_expert_map` sized a `calloc` straight from the header's
     `tensor_count`. A corrupted count requested 56 TiB.
     `rl_gguf_model_open` accepted up to 2^24 tensors or kv entries (~1.8 GiB of
     descriptors) whatever the file size. Both now reject counts the file cannot
     hold (tensor descriptor ≥ 24 bytes, kv ≥ 13 bytes).
  2. The expert map did not check tensor offsets against the file. Two routed
     tensors with offsets near 2^64 sort next to each other, pass the slice check,
     and then `data_base + offset` wraps into the `pread` offsets. Offsets beyond EOF
     are now rejected, as `rl_gguf_model_open` already did.
  3. Tokenizer arrays are now bounded by the bytes left in the file before
     allocation. `token_type` values outside int32 are rejected instead of going
     through an undefined float→int conversion. Nested arrays are rejected in both
     readers, as llama.cpp does.
- **Tooling** (`cad2e69`).
  - `redlite-gguf-fuzz` builds two valid synthetic seeds (full directory with
    tokenizer and chat template; routed triplet plus router). It feeds both readers
    every strict truncation, 4- and 8-byte boundary values at every header offset,
    and seeded random multi-mutations.
    - Seeds must be accepted, and every truncation of the directory seed must be
      rejected.
    - For accepted mutants it mmaps the file, reads the first and last payload
      bytes, and walks every expert layout.
    - `make native` runs 2000 random iterations; `regress_m4.sh` runs 20000.
  - `scripts/sanitize_offline.sh` (`make sanitize`) rebuilds the model-free tests
    and the fuzz with `-fsanitize=address,undefined,float-cast-overflow
    -fno-sanitize-recover=all` and runs them. It is also a new `regress_m4.sh`
    check, `selftest.sanitize`.

## Validation

Machine: **Linux x86_64 cloud container** (Ubuntu 24.04, 4 vCPU, 15 GiB, gcc 13.3,
clang 18.1). There is no Metal and no GGUF on it. Commit `cad2e69`.

| Check | Command | Result |
|---|---|---|
| portable build, gcc | `rm -rf .deps/redmetal && CC=gcc make native` | rc 0, 0 warnings, all self-tests OK |
| portable build, clang | `rm -rf .deps/redmetal && CC=clang make native` | rc 0, 0 warnings, all self-tests OK |
| sanitizers (gcc; the container's clang lacks the ASan runtime) | `CC=gcc make sanitize` | selftest + 4 offline tests + fuzz: no report |
| fuzz, longer | `.deps/redmetal/sanitize/redlite-gguf-fuzz --iterations 100000 --seed {1,0xdeadbeef,424242}` | 3 × 152 374 inputs, 0 reports, 0 accepted truncations |
| Python | `make test`; `ruff check redlite tests --select E9,F63,F7,F82` | 36 OK; clean |

Before the fixes, the same fuzz aborted under ASan on defect 1, and its own
invariant check caught defect 2.

**Not yet run on macOS / M4 Max.** The build and parity side of `regress_m4.sh`
(including the two new checks and the real 18 GiB GGUF through the hardened
readers) has not been executed since these commits. The new rules cannot reject a
well-formed file: 843 tensors and 49 kv against 18 GiB, no nested arrays (llama.cpp
loads the file), every offset inside the file. The suite's model-opening checks
are what confirm it.

No benchmark record is added: nothing here measures throughput. Tool run times
are not benchmarks.

## Scope boundary

- Implemented and tested synthetically on Linux x86_64 only. Nothing here is
  validated on the M4 Max or the M4 Pro, and nothing here changes Metal code, kernel
  arithmetic, decode or prefill.
- The fuzz is a deterministic mutation fuzzer over two small seeds, not
  coverage-guided (no libFuzzer/AFL). It exercises the header, metadata and
  directory parsers and the expert layout, not the tokenizer or the engine's
  tensor-table audit.
- The sanitizer run covers the portable model-free tests only. The dev26 item
  "sanitizer run of a chat turn" needs Metal and the model and is **not done**.
- Also **not done** from dev26: parity and benchmark at 4096/8192 positions,
  sampler distributional parity with llama.cpp, and model-free tests for the
  dev22–dev24 kernels, which do not exist yet.
- dev22–dev25 are not started. Their targets need Metal, the model and llama.cpp
  on the M4 Max: decode ≥ 62 / ≥ 45 tok/s, prefill ≥ 300 tok/s, and parity with
  llama.cpp for the chat and server work. The work was done in a cloud container
  without any of these.
