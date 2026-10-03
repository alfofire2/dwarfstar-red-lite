# Red Lite dev51 — decode on the 24 GiB Mac

Status: in progress (roadmap Phase 2). Measured on the Apple M4 Pro 24 GiB, 2026-10-03, AC power, IQ2_XXS.

## 2a. Expert cache size sweep

`CACHE=<MiB> scripts/dev/small_mac_ab.sh MODEL default`: six prompts × 256 greedy tokens, one process per prompt,
30 s cooling. After each size, a short run reports the footprint, and `vm.swapusage` and `memory_pressure` are
read.

| `--cache-mib` | misses / token | MiB read / token | decode tok/s per prompt | median | footprint |
|---:|---:|---:|---|---:|---:|
| 4096 | 29.5 | 43.1 | 30.29 32.29 30.14 32.75 34.36 32.67 | **32.48** | 5121 MiB |
| 6144 | 16.2 | 23.1 | 30.72 30.14 29.09 31.70 32.44 30.71 | 30.72 | 6453 MiB |
| 8192 | 13.0 | 18.2 | 30.29 30.07 29.55 31.65 32.10 30.87 | 30.58 | 6724 MiB |
| 10240 | 12.6 | 17.5 | 30.72 30.03 29.42 31.76 32.47 31.37 | 31.05 | 6724 MiB |
| 12288 | 12.6 | 17.5 | 30.94 30.32 29.57 31.69 32.57 31.26 | 31.10 | 6724 MiB |
| 14336 | 12.6 | 17.5 | 30.90 30.18 29.50 31.56 32.54 31.29 | 31.10 | 6724 MiB |

No swap at any size, and memory pressure stayed at 78–87 % free.

- **From 8 GiB up the misses stop falling.** Each prompt runs in a fresh process, so the remaining 12.6 misses per
  token are first loads, and the pool never fills past what one 256-token answer touches (6.7 GiB of footprint).
- **A bigger cache does not make decode faster.** Halving the misses (29.5 → 13.0 per token) left decode at
  31–32 tok/s. With the pre-gated prefetch, loading is not what bounds the bounded-cache path on this Mac.
- **The bound is the synchronous path itself.** One GPU round trip per layer, 48 per token. Only full residency
  removes it: the whole token is then routed on the GPU in one command buffer (dev21).
- **The 4 GiB default stays.** It is the fastest and smallest point measured.

## 2b. Every expert resident on 24 GiB (next)

IQ2_XXS at full residency needs 17,316 MiB of expert slots plus 1,083 MiB of dense weights. The M4 Pro's default GPU
working set (`recommendedMaxWorkingSetSize`) is 17.76 GiB, so it does not fit by default. `sysctl
iogpu.wired_limit_mb` raises the limit until the next reboot; it needs `sudo`.

## Scope boundary

- Only the M4 Pro 24 GiB and the IQ2_XXS file were measured.
- The sweep runs one process per prompt; a long-lived chat with a larger cache would see fewer first-load misses.
  Decode did not depend on misses here, so this does not change the conclusion.
- 2b and 2c are not done yet.
