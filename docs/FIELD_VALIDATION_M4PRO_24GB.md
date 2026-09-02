# Field validation — Apple M4 Pro / 24 GiB

Date: 2026-09-02

Red Lite now has both an interactive field observation and a controlled context-depth sweep on its primary target class.

## Hardware / model

- Apple M4 Pro
- 24 GiB unified memory
- 8 performance CPU cores detected
- Qwen3-Next-80B-A3B-Instruct
- Bartowski IQ2_XXS GGUF
- actual local file size: 17.97 GiB
- llama.cpp Metal backend
- batch / ubatch: 256 / 128
- K/V cache types: q8_0 / q8_0

Metal reported:

- `MTLGPUFamilyApple9`
- simdgroup matrix multiplication enabled
- residency sets enabled
- shared buffers enabled
- recommended maximum working set: 21474.84 MB
- Tensor API disabled on M4 Pro (expected for pre-M5 hardware in this llama.cpp build)

## Interactive observation

Interactive inference completed successfully at 2K context. Observed timings during the session were approximately:

- prompt processing: 36.7–87.5 tokens/s
- generation: 30.7–40.7 tokens/s

These interactive figures are not a controlled benchmark because prompt lengths, cache state and sampling differed between turns.

## Controlled sweep

Command:

```bash
redlite sweep models/Qwen_Qwen3-Next-80B-A3B-Instruct-IQ2_XXS.gguf \
  --contexts 2048,4096,8192 \
  --prompt-tokens 256 \
  --gen-tokens 128 \
  --max-swap-delta 0.5 \
  --output benchmarks/m4pro-24gb.json
```

Observed results:

| Context | Policy status | Estimated headroom | Prompt tok/s | Generation tok/s | Swap delta |
|---:|---|---:|---:|---:|---:|
| 2048 | CRITICAL | +0.61 GiB | 258.8 | 38.0 | +0.00 GiB |
| 4096 | CRITICAL | +0.50 GiB | 247.2 | 36.4 | +0.00 GiB |
| 8192 | CRITICAL | +0.28 GiB | 236.7 | 36.9 | +0.00 GiB |

All three context depths completed and no swap growth was observed during the sweep.

The result is notable because generation throughput stayed essentially flat from 4K to 8K while prompt throughput declined gradually. The policy still labels all three depths `CRITICAL` because its estimated resident margin is below 1 GiB; observed success does not turn a narrow memory margin into a generally safe one.

## Recommended presets for this exact field profile

- **2K — conservative:** largest estimated margin; useful when other applications must remain open.
- **4K — default:** field-validated and preserves more margin than 8K.
- **8K — experimental:** completed with zero observed swap delta, but estimated headroom is only 0.28 GiB. Longer sessions and repeated runs are still needed before promoting it to the default.

Red Lite 0.2.1 therefore changes the automatic default to 4096 only for the observed Apple M4 Pro / 24 GiB / ~18 GiB resident profile. Other unvalidated 24 GiB machines remain on the conservative 2K default.

## Planner status model

| Headroom | Status | Meaning |
|---:|---|---|
| >= 2 GiB | SAFE | comfortable policy margin |
| 1–2 GiB | TIGHT | usable, monitor pressure |
| 0–1 GiB | CRITICAL | resident fit with little estimated margin |
| < 0 GiB | UNSAFE | reject resident mode |

The controlled sweep is also stored in `benchmarks/m4pro-24gb-sweep-2026-09-02.json` and summarized in `configs/mac-m4pro-24gb-observed.json`.
