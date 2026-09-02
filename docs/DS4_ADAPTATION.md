# Mapping from DwarfStar / DS4 to Red Lite

Red Lite is not a source fork of DS4 v4. It is an adaptation of its engineering
principles to a much smaller sparse model and a much smaller Mac.

| DwarfStar idea | Red Lite adaptation |
|---|---|
| Deliberately narrow model support | Qwen3-Next only |
| Metal is the primary Mac backend | Metal resident path for models that fit |
| SSD streaming for RAM-limited Macs | Bounded mmap expert residency for oversized Q4 |
| Asymmetric expert quantization matters | Curated low-bit resident and Q4 quality profiles |
| Integrated CLI/server | `redlite run` and `redlite serve` |
| Model-specific memory behavior | Qwen3-Next recurrent/hybrid-aware conservative planner |
| Reproducible performance work | pinned engine commits + benchmark protocol |

## Why Qwen3-Next instead of shrinking DeepSeek V4

Qwen3-Next-80B-A3B already has the desired sparse shape: 80B total, ~3B active,
512 experts and top-10 routing. Reducing DeepSeek V4 itself would require pruning or
retraining weights and validating the altered router/model quality. Using an existing
80B sparse checkpoint preserves an upstream-trained model while solving the hardware
problem at the runtime layer.

## Why v0.2 still uses two proven engines

Writing a new Qwen3-Next Gated DeltaNet + full-attention + MoE Metal runtime from
scratch is possible, but calling it complete before numerical comparisons and real
Mac testing would be irresponsible. v0.2 therefore owns the Apple-specific policy,
UX and reproducibility layer while using pinned, already-Qwen3-Next-capable kernels.

The architectural endpoint remains a native Red Metal expert-streaming backend, but
that should only replace the current engines after reference-logit and long-run tests
pass on actual Apple hardware.
