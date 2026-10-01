# Red Lite dev32 — release 0.4.0

Status: **prepared as 0.4.0** on branch `dev/0.4`; pull request `dev/0.4 → main` open and
**not merged** (merging is the maintainer's decision). Machine: Apple M4 Max 48 GiB.

## What the release changes

- **Version** 0.4.0 in `VERSION`, `pyproject.toml`, `redlite/__init__.py` and the usage
  text of the product binaries (`redlite-engine`, `redlite-generate`, `redlite-server`).
  The stage parity CLIs keep the dev tag of the milestone that built them.
- **CHANGELOG**: the "Unreleased" block became `0.4.0 — 2026-09-30`, with a summary and the
  milestone targets of the cycle, then the dev28–dev31 entries.
- **README**: the two supported GGUFs, `--cache-mib full`, how `redlite chat` picks the
  model, the server's reuse / stop / queue, a measured-performance table per model and per
  cache (native and pinned llama.cpp on the same ids), correctness for both files.
- **docs/README.md** indexes dev28–dev32.
- **Release tarball**: `scripts/package_release.sh` → `dist/redlite-0.4.0-macos-arm64.tar.gz`
  and its `.sha256` (not committed; ready to attach to a GitHub release).

No runtime code changed in dev32 apart from the version strings.

## Milestone results of the 0.4.0 cycle (M4 Max 48 GiB)

| Milestone | Target | Reached | Commits |
|---|---|---|---|
| dev28 CI and distribution | Linux CI on push/PR green on GitHub; mac workflow manual; arm64 tarball + SHA256 + install notes | tarball, manual mac workflow, CI workflow: yes. **Green on GitHub: not reached** — GitHub does not start hosted jobs on this account (billing); the same steps pass in a Linux container | `3912aec` `b84bdd8` `f6d1d1f` `16f0963` |
| dev29 server | reuse when the prompt extends the conversation (greedy identical to cold), TTFT before/after, stop sequences, FIFO queue, fake-engine tests, real check in regress | yes: identical answer (SHA-1 1f94f64826b2 in every configuration), TTFT 3833 → 173 ms (4 GiB), 3779 → 167 ms (22 GiB) | `4a44030` `79b3572` `788a316` `b3d8df2` `8dff5fd` |
| dev30 prefill and experts | prefill 1100 ≥ llama.cpp same prompt; 8192 ≥ 280 tok/s; decode 22 GiB not below 0.3.0 | yes with full residency: 891.9 vs 858.34, 926.9, decode A/B 71.09 vs 70.47. At 4 GiB / 512 chunks 1100 tokens reach 545.8 (below llama.cpp) | `d5d7fd8` `f058a0f` `65bce55` `55e6cbe` `7e96a3e` `aed1ec5` `fca7d78` `8cbdffe` |
| dev31 quality | higher-quality GGUF resident on 48 GiB; missing quant types (CPU ref + Metal); greedy = llama.cpp, KL ≤ 1e-5; perplexity of both; chat picks the best model with ≥ 40 GiB | yes: IQ3_XXS (29,376 MiB resident), KL 1.4e-12, 24/24 greedy, PPL 14.29 vs 16.47 | `388853e` … `d75d080` (`git log dev30-green..dev31-green`) |
| dev32 release | 0.4.0 versions, CHANGELOG, README per model/cache, tarball, PR | yes; PR open, not merged | this series |

## Final validation

On the release tree (versions 0.4.0), in this order:

- `make test`: OK (Ran 64 tests); `ruff check redlite tests scripts --select E9,F63,F7,F82`: clean.
- `rm -rf .deps/redmetal`, then `make redmetal`, `make native`, `make sanitize`: all OK, **0 compiler warnings**.
- `scripts/package_release.sh`: `dist/redlite-0.4.0-macos-arm64.tar.gz` (528 KiB), SHA-256
  `4c34a1c5b6825556a9ce2099455e651c48e81b6911e6848c56cee7ab0d67400c`.
- `scripts/regress_m4.sh` on the IQ2_XXS GGUF: **49/49**; on the IQ3_XXS GGUF: **36 passed,
  0 failed, 13 skipped** (the dev11–dev17 dense stage tools, IQ2_XXS layout only).
- GitHub CI: see the pull request; hosted jobs are refused for billing (dev28).

## Scope boundary

- Everything was measured and validated on the M4 Max 48 GiB only. The 4 GiB-cache
  configuration intended for the M4 Pro 24 GiB uses the same kernels but was not run
  there; neither was anything on an M4 Air 16 GiB.
- The release tarball was built and started on the M4 Max; the `-mcpu=apple-m1` build
  has not run on older Apple Silicon. It is not signed or notarized.
- GitHub CI is configured and would run on every push and PR, but GitHub refuses to
  start hosted jobs for this account until its Actions billing is fixed.
