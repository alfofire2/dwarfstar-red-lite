# Memory policy for 24 GiB Macs

## Why file size matters more than parameter count

An 80B model at BF16 is roughly 160 GB before runtime overhead. Quantization changes
storage dramatically, so Red Lite plans from the *actual local GGUF byte size*.

Representative Qwen3-Next-80B-A3B files are around 19.3 decimal GB for IQ2_XXS and
48.4 decimal GB for Q4_K_M. Source pages often use decimal GB while Red Lite displays
binary GiB.

## Conservative resident budget

```text
resident_budget = physical_ram
                - max(3.5 GiB, 14% of RAM)  # macOS/apps
                - 1.25 GiB                  # runtime/scratch margin
                - context reserve
```

The context reserve is a policy margin, not a byte-exact model of llama.cpp.

## Status, not a binary fantasy

v0.2 distinguishes a model that merely fits from one with comfortable room:

```text
>= 2 GiB headroom   SAFE
1–2 GiB             TIGHT
0–1 GiB             CRITICAL
< 0 GiB             UNSAFE (resident mode rejected)
```

A `CRITICAL` plan is allowed because the 17.97 GiB IQ2_XXS profile has been observed
running on a 24 GiB M4 Pro, but the warning is intentional.

## 24 GiB default

For a resident model >=17 GiB, Red Lite now starts at 2048 context. Move to 4096 or
8192 only after running `redlite sweep` and inspecting swap / Memory Pressure.

## macOS rules

- Close memory-heavy apps before benchmarking.
- Do not use `mlock` on a 24 GiB target.
- Check `redlite pressure` before and after demanding runs.
- Yellow/red Memory Pressure means the profile is not practically healthy even if it
  has not crashed.
- Uncontrolled VM compression/swap is not equivalent to explicit expert streaming.

## Quality warning

IQ2_XXS reaches the resident target by very aggressive quantization. It is the speed /
fit profile, not the quality recommendation. The existing Q4_K_M path uses bounded
SSD-backed expert residency on CPU/Accelerate until native Metal expert streaming is
implemented and validated.
