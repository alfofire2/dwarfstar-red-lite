# Changelog

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
