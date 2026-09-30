# Red Lite dev34 — bounded-cache prefill: fewer expert reloads

Status: **done** on branch `dev/bounded-prefill` (from `dev/iq3-kernels` at 65cabd9). Apple
M4 Max 48 GiB only. The change targets the 4 GiB configuration meant for 24 GiB Macs, which
could not be measured on one (no M4 Pro here).

## Finding

With a bounded expert cache, the batched prefill routes every token of a chunk, loads the union
of the selected experts layer by layer, and the next chunk finds them evicted by the other 47
layers: **every chunk reloads nearly every expert of every layer.** The chunk count, not the
prompt length, sets the bytes loaded. `redlite-engine logits … --cache-mib 4096 --batch N`
(IQ2_XXS, single runs; the prefill split line now reports the loads):

| prompt | chunk 512 (0.4.0 default) | chunk 1024 | chunk 2048 |
|---|---:|---:|---:|
| 1100 tokens: experts loaded | 37,022 (26.3 GB) | 25,153 (17.9 GB) | 16,784 (11.9 GB) |
| 8192 tokens: experts loaded | 226,117 (161 GB) | 126,763 (90 GB) | 69,071 (49 GB) |

On this Mac the GGUF stays in the page cache, so a load is a memory copy (it also competes with
the GPU for bandwidth). On a Mac whose page cache cannot hold the GGUF, loads are SSD reads, so
3.3× fewer bytes should matter more there; that is an expectation, not a measurement.

## Change

`rl_engine_prefill_batch()` returns 2048 for every cache size (dev30 used 2048 only with full
residency; `--batch` still overrides). Footprint cost at 4 GiB with an 8387-token prompt:
5,231 → 5,802 MiB (+571 MiB, `redlite-generate --json` `phys_footprint_mib`). Chunks of 4096
would need more than the 32,768 pairs one expert plan accepts and another ~0.5 GB of scratch;
not attempted.

## Measurements

Cooled, alternating (`--batch 512` vs `--batch 2048`), IQ2_XXS, 4 GiB cache, 3 runs each:

| prompt | chunk 512 | chunk 2048 |
|---|---:|---:|
| 1100 tokens | 540.3 tok/s (540.3 / 543.0 / 520.9) | **795.1 tok/s** (795.1 / 783.2 / 807.0) |
| 8192 tokens | 639.5 tok/s (644.7 / 639.5 / 637.2) | **885.9 tok/s** (888.3 / 885.9 / 869.7) |

The pinned llama.cpp, fully resident, does 858.3 / 893.7 tok/s on the same ids (dev28).

## Parity

1100-position context vs llama.cpp with the default chunk at 4 and 2 GiB: IQ2_XXS worst 0.99 /
KL 3.5e-3, IQ3_XXS 0.57 / KL 3.9e-4 (as with 512-token chunks). `quick_parity.sh`: PASS.
`regress_m4.sh`'s `generate.json` now expects `"batch":2048`.

## Validation

`rm -rf .deps/redmetal`, `make redmetal`, `make native`, `make sanitize`: 0 warnings; `make test`
and ruff green; `scripts/regress_m4.sh`: IQ2_XXS **49/49**, IQ3_XXS **36 passed, 0 failed, 13
skipped** (the IQ2_XXS-layout dense stage tools).

## Scope boundary

- Not measured on a 24 GiB Mac; the benefit there (fewer SSD reads) is inferred from the load
  counts above.
- The +571 MiB of scratch comes out of the memory a 24 GiB Mac has for the page cache.
