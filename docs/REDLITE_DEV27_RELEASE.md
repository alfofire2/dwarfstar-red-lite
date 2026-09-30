# Red Lite dev27 — release 0.3.0

Status: **released as 0.3.0** (2026-09-30).
- The release was prepared on the `v0.3-streaming` branch and tagged `v0.3.0` at `6c1f650`.
- The maintainer merged it into `main` through pull request #1 (merge commit `3cd58a7`),
  and the branch was then deleted.
- The GitHub Release is `v0.3.0`.

## What the release changes

- **Version.** 0.3.0 in `VERSION`, `pyproject.toml` and `redlite/__init__.py`, and in the
  usage text of the product binaries (`redlite-engine`, `redlite-generate`,
  `redlite-server`). The stage parity CLIs keep the dev tag of the milestone that built
  them.
- **CHANGELOG.** The "Unreleased" block became `0.3.0 — 2026-09-30`. It opens with a release
  summary, the measured numbers and the milestone targets of the cycle, followed by every
  dev entry.
- **README, rewritten for new readers:**
  - what the two runtimes are;
  - a native quick start;
  - a per-machine table of measured numbers, each traced to a record;
  - correctness, limits, and the status of every tool (product, launcher, regression,
    legacy).
  - The launcher documentation follows unchanged under its own heading.
- **`docs/README.md`** points to the release summary and marks the legacy material.
- **Legacy marking.** The Python streaming oracle (`redlite-stream`, `redlite-ffn`,
  `redlite-topk`) states in its `--help` that it is the frozen dev1–dev8 reference, and
  names the native binaries.

No runtime code changed in dev27 apart from the version strings.

## Milestone results of the 0.3.0 cycle (M4 Max 48 GiB)

| Milestone | Target | Reached | Commit |
|---|---|---|---|
| dev22 decode kernels | decode ≥ 62 tok/s at 22 GiB | yes, 66.2 | `fdd3c69` (record `314d483`) |
| dev23 early-out + prefetch | decode ≥ 45 tok/s at 4 GiB | yes, 50.1 | `a4e4a26` (record `65a0624`) |
| dev24 prefill overlap | prefill ≥ 300 tok/s, 1100 tokens, 4 GiB | yes, 314.3 | `8ed6d56` (record `e96add9`) |
| dev26 completion | parity + benchmark at 4096/8192; sanitized chat turn; model-free kernel tests | yes; also split-K attention | `45a45e9`, `1b67f87` (record `0f289c2`) |
| dev27 release | 0.3.0; CHANGELOG; README; docs index; legacy marked; PR to main open, not merged | yes | this commit series |

## Final validation (M4 Max 48 GiB, macOS 27)

Run in this order, on the release tree:

| Check | Command | Result |
|---|---|---|
| Python tests | `make test` | 57 tests OK |
| Lint | `ruff check redlite tests --select E9,F63,F7,F82` | all checks passed |
| Python bridge | `rm -rf .deps/redmetal && make redmetal` | rc 0, 0 warnings |
| Native build + self-tests | `make native` | rc 0, 0 warnings |
| Sanitizers | `make sanitize` | rc 0, 0 warnings; no ASan/UBSan report (including the Metal kernel self-test) |
| Regression | `scripts/regress_m4.sh models/Qwen_Qwen3-Next-80B-A3B-Instruct-IQ2_XXS.gguf` | **46/46 PASS** |

The regression includes, among others:

- `generate.vs_llama`: 24 greedy tokens identical to the pinned llama.cpp.
- `logits.long_context_vs_llama`.
- `engine.parity.gpu_routed`.
- `generate.sanitize`.
- `selftest.engine_kernels`.

## Scope boundary

- **Machine.** Every 0.3.0 number is from the M4 Max 48 GiB. The native runtime was last
  measured on the M4 Pro 24 GiB at dev18. The 24 GiB memory policy and the launcher
  presets come from the M4 Pro and are unchanged.
- **Not in this release.** Multi-sequence batching, concurrent server requests, GGUFs with
  other routed-expert quantizations, and an optimized batched-prefill attention.
- **Merge.** Not part of this milestone's work: the maintainer merged the pull request into
  `main` afterwards (#1, `3cd58a7`).
