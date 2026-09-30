# Red Lite dev28 — Linux CI and release packaging

Status: **done, except "green on GitHub": not reached** (GitHub does not start hosted
jobs on this account, see Validation). Branch `dev/0.4` (0.4.0 cycle). Machine for every local number: Apple
M4 Max / 48 GiB (the daily machine). Nothing here was run on an M4 Pro or an M4 Air.

## What changed

- **GitHub CI runs on every push and pull request again** (`.github/workflows/ci.yml`).
  One Linux job (`ubuntu-latest`), no model, no Metal:
  - `ruff check redlite tests --select E9,F63,F7,F82` (the lint the project enforces) and
    `python -m compileall` of `redlite`, `tests` and `scripts`;
  - `make native` — the portable subset of the native tree (GGUF parser/fuzz, CPU
    references, engine offline test, sampler, server core); the Metal-only build scripts
    print a skip message. Any `warning:` in the log fails the job;
  - `make sanitize` — ASan + UBSan build and run of every model-free test, the GGUF fuzz
    and the OpenAI protocol tests against the sanitized server core; warnings fail the job;
  - `make test` — the Python unit tests.
  The macOS hosted jobs and the separate `lint.yml` were removed: the Metal paths cannot be
  exercised without an Apple GPU and the model, so macOS minutes bought nothing that
  `scripts/regress_m4.sh` on the local machine does not already cover. The self-hosted
  `mac-m4-field-validation.yml` stays manual (`workflow_dispatch` only).
- **`scripts/package_release.sh`** builds `dist/redlite-<version>-macos-arm64.tar.gz`
  (`bin/redlite-generate`, `bin/redlite-server`, `bin/redlite-engine`, `INSTALL.md`,
  `LICENSE`, `NOTICE.md`, `VERSION`, `BUILDINFO`) plus a `.sha256` file checked with
  `shasum -a 256 -c`. The binaries are rebuilt into `.deps/package/build` (the development
  build in `.deps/redmetal` is untouched) with `-mcpu=apple-m1` by default instead of
  `-mcpu=native`, so the tarball does not depend on CPU instructions newer than the first
  Apple Silicon generation. The Metal kernels are compiled at start-up on the target GPU.
  The model is never packaged.
- **Build scripts** accept `REDLITE_MCPU` (default `native`) and `build_engine.sh` /
  `build_server.sh` accept `REDLITE_BUILD_OUT` (default `.deps/redmetal`); nothing changes
  for a plain `make native`.
- **Development measurement tools.** `redlite-ref-llama MODEL bench --tokens ... [--max N]`
  times the pinned llama.cpp (its default context parameters: `n_batch` 2048, `n_ubatch`
  512, flash attention auto, f16 KV, fully resident) on exactly the same token ids as the
  native runtime: one prefill, then N greedy tokens one at a time. `scripts/dev/bench_m4.sh`
  gained `--only prefill8192` (the fixture ids repeated to 8192, as `long_positions.sh`
  does), `--only llama` (the same 1100 / 8192 prefill ids and 256 decode tokens after the
  same chat prompt, through llama.cpp) and `--cache-mib N` for the full-residency cache of
  another GGUF.

## 0.4.0 baseline (main 48c96cd = release 0.3.0, M4 Max 48 GiB)

The reference for every 0.4.0 milestone, measured before any 0.4.0 change
(`scripts/dev/bench_m4.sh MODEL --reps 3 --cool 90`, median of three cooled runs; commit
48c96cd, 2026-09-30; record `benchmarks/m4max-48gb-0.4.0.json`):

| | native 0.3.0 | pinned llama.cpp, same ids |
|---|---:|---:|
| decode, 256 tokens, 22 GiB cache (every expert resident) | 71.50 tok/s | 72.46 tok/s |
| decode, 256 tokens, 4 GiB cache | 47.93 tok/s | — |
| prefill 1100 tokens (chunks of 512, 4 GiB cache) | 307.30 tok/s | 858.34 tok/s |
| prefill 8192 tokens (chunks of 512, 4 GiB cache) | 212.60 tok/s | 893.71 tok/s |

llama.cpp runs fully resident with its defaults (`n_batch` 2048, `n_ubatch` 512, flash
attention auto, f16 KV cache); its prefill on the same 1100 ids is 2.8× the native one, far
above the 339 tok/s `llama-bench pp48` figure recorded in dev19 (48 tokens only).

`RL_ENGINE_PROFILE=1` single runs (the profile commits every stage, so its totals are lower
than the bench):

| prefill | tok/s | dense GPU | of which recurrent / attention wall | experts GPU | expert loads |
|---|---:|---:|---|---:|---:|
| 1100 | 279.7 | 1629 ms | 1290 / 474 ms | 1925 ms | 682 ms (overlapped) |
| 8192 | 201.3 | 25629 ms | 8936 / 17273 ms | 13510 ms | 2654 ms |

At 8192 positions the batched attention (`attn_gqa_b`, one threadgroup per token and head)
is the largest single cost; at 1100 the routed experts and the dense GEMMs split the time.
These are the dev30 targets.

## Validation

- **GitHub.** The `CI` workflow had been disabled by hand on GitHub on 2026-09-08 together
  with the others; it was re-enabled (only `CI`; `Lint` is deleted and the M4 workflow stays
  disabled and manual). Its first run, 36706964887 (workflow_dispatch on a scratch branch
  with the dev28 commits), **was not started by GitHub**: *"The job was not started because
  recent account payments have failed or your spending limit needs to be increased."* The
  same message stopped every hosted job of this private repository on 2026-09-08. There is
  no self-hosted Linux runner. **"CI green on GitHub": not reached** — it needs the
  account's billing to be fixed; nothing in the repository can work around it.
- **The same steps in a Linux container** (`python:3.12`, Debian, arm64, under colima on the
  M4 Max; the workflow's commands verbatim): ruff clean, compileall clean, `make native` 0
  warnings, `make sanitize` OK (12 protocol tests), `make test` 57 tests OK.
- **Package.** `scripts/package_release.sh` built `redlite-0.3.0-macos-arm64.tar.gz` (476 KiB)
  with `-mcpu=apple-m1`, `minos 14.0` (`vtool -show-build`); `shasum -a 256 -c` OK; the
  extracted `bin/redlite-generate` produced the regression greedy answer ("... due to
  Ray[leigh scattering]") on the IQ2_XXS model with a 2 GiB cache.
- **Regression.** `rm -rf .deps/redmetal && scripts/regress_m4.sh MODEL`: 46/46, 0
  compiler warnings in `build.native`.

## Scope boundary

- CI covers only what runs without Apple Metal and without the 18 GiB model: parser,
  CPU references, sampler, server protocol, sanitizers, Python. Every Metal kernel and
  every real-model parity check is still validated only by `scripts/regress_m4.sh` on the
  M4 Max 48 GiB.
- The release tarball is built and its binaries start (`--help`) on the M4 Max; the
  `-mcpu=apple-m1` build was **not** run on an M1/M2/M3 machine, nor on the M4 Pro 24 GiB
  or the M4 Air 16 GiB. It is not signed or notarized (INSTALL.md explains the quarantine
  attribute).
- No runtime code changed in dev28.
