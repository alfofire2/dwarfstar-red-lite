# Field validation — Apple M4 Pro / 24 GiB

Date: 2026-09-02

Red Lite 0.2 incorporates the first real field result for its primary target class.

## Hardware / model

- Apple M4 Pro
- 24 GiB unified memory
- 8 performance CPU cores detected
- Qwen3-Next-80B-A3B-Instruct
- Bartowski IQ2_XXS GGUF
- actual local file size: 17.97 GiB
- llama.cpp Metal backend
- 2K context
- batch / ubatch: 256 / 128
- K/V cache types: q8_0 / q8_0

## Observed result

Interactive inference completed successfully. Observed interactive timings during the
session were approximately:

- prompt processing: 36.7–87.5 tokens/s
- generation: 30.7–40.7 tokens/s

These numbers are *field observations*, not a controlled benchmark: prompt lengths,
cache state and sampling differed between turns. They establish that the resident
IQ2_XXS profile can execute on the target hardware; they do not establish a universal
performance guarantee.

## Planner correction

The old planner called +0.5–0.6 GiB estimated headroom simply `SAFE`. That is too
optimistic even though the run succeeded. v0.2 uses four resident states:

| Headroom | Status | Meaning |
|---:|---|---|
| >= 2 GiB | SAFE | comfortable policy margin |
| 1–2 GiB | TIGHT | usable, monitor pressure |
| 0–1 GiB | CRITICAL | experimental resident fit |
| < 0 GiB | UNSAFE | reject resident mode |

For a ~18 GiB model on a 24 GiB Mac, the default context is now 2048. 4096 and 8192
are opt-in validation targets.

## Repeatable test

Use the new sweep command:

```bash
redlite sweep models/Qwen_Qwen3-Next-80B-A3B-Instruct-IQ2_XXS.gguf \
  --contexts 2048,4096,8192 \
  --prompt-tokens 256 \
  --gen-tokens 128 \
  --max-swap-delta 0.5 \
  --output benchmarks/m4pro-24gb.json
```

The command uses the pinned Metal `llama-bench`, records the policy status at every
context depth and snapshots macOS swap before/after each run.
