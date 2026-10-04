# Red Lite dev55 — long contexts on a 24 GiB Mac with every expert resident

Status: on branch `dev55/gpu-budget`. Measured on the Apple M4 Max 48 GiB and the Apple M4 Pro 24 GiB, 2026-10-04.

## The problems

dev51 made every expert resident on a 24 GiB Mac with a raised GPU limit, with a fixed 1 GiB margin sized for a
4K context. Two things went wrong at longer contexts:
- **GPU memory.** With MTP and a 32K context, the prompt ingestion ran out of GPU memory (dev53), because the margin
  did not grow with the context or the prefill chunk.
- **MTP speed.** At a 16.8K-token prompt, MTP decoded slower than plain decoding on the M4 Pro: 32.6 vs 37.2 tok/s.

## A measured model of the GPU need

Footprint on the M4 Max, every expert resident, a 6,989-token prompt:

| context | chunk 512 | chunk 2048 | chunk 512 + MTP | chunk 2048 + MTP |
|---:|---:|---:|---:|---:|
| 16,384 | 18,664 MiB | 19,236 | 20,451 | 21,023 |
| 32,768 | 19,440 | 20,012 | 21,292 | 21,864 |

The fit, plus the mapped dense weights (held by the GPU but not counted in the footprint):

    need = expert slots + dense + 580 + 48 KiB per context position + 572 MiB x (chunk - 512) / 1536 + 1,787 (MTP)

At the 21,741 MiB limit `redlite doctor` recommends, it explains every M4 Pro outcome:

| configuration | model | observed |
|---|---:|---|
| MTP + 32K context + 512-token chunks | 22,302 MiB | out of GPU memory |
| no MTP + 32K context + 2048-token chunks | 21,087 MiB | ran, 19.9 GiB footprint |
| MTP + 4K context + 2048-token chunks | 21,530 MiB | ran |

**In the product.** Under a raised GPU limit at full residency, `redlite chat` and `redlite serve --native` choose the
first configuration that fits: 2048-token chunks with MTP, 512 with MTP, 2048 without, 512 without. M4 Pro at 21,741
MiB:
- 4K context: 2048 + MTP;
- 16K: 512 + MTP;
- 32K: 2048 without MTP.

`redlite doctor` uses the same model (IQ2_XXS-size file: 19,743 MiB without MTP, 21,530 with). An explicit `--batch`
is never overridden.

## Where MTP stops paying (decode tok/s, plain → MTP)

| prompt tokens | M4 Pro 24 GiB | M4 Max 48 GiB |
|---:|---|---|
| 20 | 46.25 → 57.51 (+24 %) | — |
| 1,407 | 44.82 → 49.69 (+11 %) | — |
| 2,805 | 44.66 → 48.12 (+8 %) | — |
| 5,601 | 41.77 → 43.61 (+4 %) | 77.03 → 81.39 (+6 %) |
| 11,193 | 38.74 → 38.97 (0 %) | 74.05 → 76.74 (+4 %) |
| 16,775 | 37.2 → 32.6 (−12 %, dev53) | 66.14 → 67.59 (+2 %) |

- **Why.** The 2-row verify reads the attention cache twice. With half the M4 Max's memory bandwidth, the M4 Pro
  loses the gain at about 11K positions; the M4 Max still gains at 16.8K.
- **New option.** `--mtp-max-context N` (`redlite-generate`, `redlite-server`): an answer that starts past position N
  is decoded without speculation.
- **Defaults.** `redlite chat` and `serve` pass 8192 on Macs below 40 GiB; on 48 GiB there is no limit.

**End to end on the M4 Pro** (21,741 MiB limit, E3 file, the planner's choices):

| configuration | result |
|---|---|
| 32K context, 2048 chunks, no MTP, 16,775-token prompt | prompt 358 tok/s, decode 36.4 tok/s, footprint 19.9 GiB |
| 16K context, 512 chunks, MTP, 11,183-token prompt | MTP not used (past 8192); decode 38.9 tok/s |
| 16K context, 512 chunks, MTP, short prompt | MTP used (acceptance 0.59); decode 48.4 tok/s |

## Steering in the server

`redlite-server` and `redlite serve --native` take `--steer FILE`, `--steer-layers`, `--steer-strength` and
`--steer-tokens`. Every generated token is steered. Unlike `redlite-generate`, the first answer token is not: the
server's prompt goes through the batched prefill and the reused prefix state. With the pirate vector (layers 12–23,
strength 0.4), "Describe a storm at sea" becomes "…sails tore loose as the masts groaned, 'Rattle the riggin'—'tis
the last o'er the hull!'…".

## Scope boundary

- **The GPU need is fitted on one file size** (IQ2_XXS / E3, 17,316 MiB of slots) and verified at one limit on the
  M4 Pro. Other sizes use the same terms but were not measured.
- **The 8192 threshold comes from one prompt family**, the long-context fixture plus a summary request. The crossover
  depends on how predictable the text is.
- **The permanent GPU limit** (`/etc/sysctl.conf`) is not tested: writing it needs more than the agreed sudo rule.
