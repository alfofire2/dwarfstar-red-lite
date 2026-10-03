# Red Lite dev51 — decode on the 24 GiB Mac

Status: 2a and 2b done, 2c open (roadmap Phase 2). Measured on the Apple M4 Pro 24 GiB, 2026-10-03, AC power, IQ2_XXS.

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

## 2b. Every expert resident on 24 GiB

**The limit.** IQ2_XXS at full residency needs 17,316 MiB of expert slots plus 1,083 MiB of dense weights. The M4
Pro's default GPU working set (`recommendedMaxWorkingSetSize`) is 17.76 GiB, so it does not fit by default.
`sudo sysctl iogpu.wired_limit_mb=<MiB>` raises the limit until the next reboot; Metal then reports the new value
(20.00 GiB at 20,480).

**Results** (M4 Pro 24 GiB, release 0.5.0 binary, `--cache-mib full`):

| configuration | decode (six prompts × 256 tokens) | footprint | swap |
|---|---:|---:|---:|
| 4 GiB cache, default limit (2a) | 32.48 tok/s median | 5.1 GiB | 0 |
| every expert resident, limit 20,480 MiB | **45.98 tok/s** median (45.54–46.07) | 17.9 GiB | 1.3–1.5 GB, steady |
| + MTP head, limit 22,016 MiB | **52.65 tok/s** median (46.68–55.65) | 19.6 GiB | 1.3 GB, steady |

- **Every token is routed on the GPU** in one command buffer (255 of 255), with no per-layer CPU round trip.
  That is +42 % over the bounded cache, and faster than the llama.cpp launcher on the same Mac (36.4–38.0 tok/s).
- **MTP works on 24 GiB:** +15 % median, up to +21 % on code (55.65 tok/s; 57.44 with acceptance 0.91 in a single
  run), and the output is identical to plain decoding in all six pairs. The 2-row verify costs 30.3 ms against
  ~21.7 ms for a plain step, a higher ratio than on the M4 Max, hence the smaller gain.
- **Costs:**
  - opening the model preloads 17 GiB of experts (13.7 s);
  - macOS moves about 1.3–1.5 GB of other processes to swap; it stayed steady over the runs, and memory was 92 %
    free after the process exited.
  - Other heavy apps compete for the remaining ~4 GiB.

**In the product.**
- `redlite chat` and `redlite serve --native` read `iogpu.wired_limit_mb`. When it is set and the experts, dense
  weights and a 1 GiB margin fit, they use full residency on any Mac; MTP is added when its head fits too.
- With the default limit, nothing changes (4 GiB cache).
- `redlite doctor` prints the limit each configuration needs and the command (IQ2_XXS: 19,429 MiB; with MTP
  21,741 MiB).

**How to enable it** on a 24 GiB Mac (until reboot):

```bash
./bin/redlite doctor                       # prints the needed limit
sudo sysctl iogpu.wired_limit_mb=21741     # IQ2_XXS + MTP head (19429 without MTP)
./bin/redlite chat                         # now full residency (+ MTP if models/ has the head file)
```

To make it permanent, put `iogpu.wired_limit_mb=21741` in `/etc/sysctl.conf`. Not tested here.

## Scope boundary

- Only the M4 Pro 24 GiB and the IQ2_XXS file were measured.
- The sweep runs one process per prompt; a long-lived chat with a larger cache would see fewer first-load misses.
  Decode did not depend on misses here, so this does not change the conclusion.
- **2b** was measured with IQ2_XXS only, at 4096 context positions, with no other heavy application open.
  Long contexts add 48 KiB per position to the footprint.
- The 1 GiB margin was measured, not derived: footprints were 18,358 and 20,098 MiB against limits of 20,480 and
  22,016.
- **2c** (cache-aware routing as the bounded-cache default) is not done yet.
