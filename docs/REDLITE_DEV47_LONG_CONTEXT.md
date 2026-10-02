# Red Lite dev47 — long context, the M4 Pro build, and the dev44 decision

Status: on branch `dev/sdk-select`. M4 Max 48 GiB, and for the first time since dev18 the M4 Pro 24 GiB over SSH.

## Long context (M4 Max 48 GiB, IQ2_XXS, every expert resident)

KV memory is 48 KiB per position (12 attention layers × K and V × 2 KV heads × 256 × float32), plus a fixed
72 MiB of DeltaNet state: `--context 65536` reports 3147 MiB of state per backend.

**Parity with the pinned llama.cpp at 16384 positions** (`scripts/dev/long_positions.sh --positions 16384
--batch 2048 --cache-mib full`; 100 decode steps after a batched prefill):

- argmax agreement on all 100 positions;
- worst KL 7.857e-05 (the limit is 2e-2);
- worst single-logit difference 6.21, above the 5.0 bound set in dev30 from 8192-position measurements.

The single-logit difference grows with length (4096: 2.6, 8192: 4.3–4.5 for the 0.3.0 runtime's own
self-consistency), so this is the same pattern, not a regression. The bound was not changed: that would need
the native floor at 16K, which was not measured.

**32K and 60K were not compared with llama.cpp.** The oracle tool dumps logits one decode at a time: 16K took
about 2 hours, and 32K reached position 13,650 after 2 hours (killed). See `docs/WHAT_DID_NOT_WORK.md`.

**Speed at 33,551 prompt tokens** (`--context 65536`, AC power):

- prefill 416.7 tok/s (80.5 s);
- decode 25.7 tok/s (66 tok/s at 8K, dev35);
- footprint 21.6 GiB.

An earlier run read 32.6 tok/s prefill and 14.6 tok/s decode on battery at 6 %; it was discarded. At long
context the decode is dominated by attention over float32 K/V. A float16 KV cache would halve that traffic;
it is not implemented.

## dev44 decision: chunk-parallel DeltaNet prefill not built

Per-stage prefill profile at 8387 tokens (`RL_PREFILL_PROFILE=1`, IQ2_XXS, full residency):

| stage | ms |
|---|---:|
| recurrent qkv/z/ba projections | 1551.3 |
| attention rope+gqa | 895.5 |
| recurrent conv / state / tail | 726.9 |
| recurrent ssm_out | 549.9 |
| attention q/k/v | 388.9 |
| shared expert | 304.5 |
| attention o | 185.0 |
| router (+prediction) | 107.9 |

The whole prefill took 10,919 ms. The DeltaNet recurrence is part of the 727 ms conv/state/tail group, which
is at most 6.7 % of the prefill. MLX's chunk-parallel kernel gains 1.3–1.45× on the recurrence (MLX #4020), so
the reachable gain is about 2 % of the prefill. That is not worth a new, numerically different algorithm.

## The M4 Pro 24 GiB build

- **Linker.** The M4 Pro's Command Line Tools (26.6, ld-1267) ship a MacOSX27 SDK whose `.tbd` files that
  linker cannot read: "unknown architecture arm64e.x1-macos". `REDLITE_SDK` (default `macosx`) now selects
  the SDK of every `xcrun` build. The M4 Pro uses `REDLITE_SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk`.
- **First run.** `make native` there: 0 warnings, and every self-test passes, including the Metal kernel
  self-test on the M4 Pro GPU. The IQ2_XXS GGUF was downloaded on the machine (SHA-256 `d132df15…`, as on
  Hugging Face), and the llama.cpp oracle was bootstrapped with `cmake` from pip. The first greedy answer is
  identical to the M4 Max's.
- **Session script.** `scripts/dev/small_mac_session.sh` runs, in one go: the build, the full regression, the
  4 GiB cache A/B, the bounded-cache benchmark and the state restart, all logged to `.deps/session-<date>/`.

## M4 Pro 24 GiB field session (2026-10-02, commit 91481ea, AC power)

`scripts/dev/small_mac_session.sh` on the M4 Pro, IQ2_XXS. This is the first native measurement there since
dev18 (`280f788`).

**Correctness.** `scripts/regress_m4.sh`: 45 passed, 3 failed, 1 skipped.

- **The 3 failures are the llama.cpp oracle's, not the engine's.** Loading all 18 GiB onto Metal,
  `redlite-ref-llama` fails with `kIOGPUCommandBufferCallbackErrorOutOfMemory` on 24 GiB. The machine's
  Python 3.9 also lacks `numpy` for `compare_dumps.py`.
- **Transitive check instead.** The M4 Pro's native outputs were compared bit for bit with the M4 Max's, whose
  outputs match llama.cpp. All four are **identical**:
  - the short logits dump `native.bin`;
  - the 1200-token long-context dump `native.long.bin` (100 positions);
  - the 24 greedy tokens `gen.native.txt`;
  - the tokenizer corpus.

**4 GiB expert cache A/B** (`small_mac_ab.sh`, six prompts × 256 tokens, 60 s cooling, decode tok/s):

| configuration | misses / token | MiB read / token | decode tok/s per prompt | median |
|---|---:|---:|---|---:|
| `default` (dev37 slot classes) | 29.5 | 43.1 | 32.70 32.61 31.16 33.66 35.12 32.66 | **32.68** |
| `uniform` (0.4.0 slots) | 39.3 | 55.7 | 32.05 31.49 30.53 33.44 34.77 32.31 | 32.18 |
| `bias05` (dev46, opt-in) | 22.9 | 33.1 | 34.21 34.07 33.72 34.65 35.00 34.41 | **34.31** |
| `nocache` (`F_NOCACHE`) | 29.5 | 43.1 | 31.68 30.75 28.85 33.52 34.05 33.52 | 32.60 |
| `noprefetch` | 45.5 | 33.3 | 28.14 28.24 27.52 30.43 33.49 30.20 | 29.22 |

- The dev37 slot classes cut misses by 25 % and gain 1.6 % in decode.
- Cache-aware routing at λ = 0.5 gains 5 % over the default (outputs change; engine perplexity unchanged on
  the M4 Max, dev46).
- `F_NOCACHE` makes no difference.
- The prefetch is worth 12 %.
- Miss counts are identical to the M4 Max's (routing is deterministic). A miss costs more here, but much less
  than an SSD read would: most of the 18 GiB file stays in the page cache of the 24 GiB machine.

**Bounded-cache benchmark** (`bench_m4.sh --reps 3 --cool 90`, 4 GiB cache, default 2048-token chunks):

| | M4 Pro 24 GiB, this build | M4 Pro 24 GiB, dev18 (`280f788`) |
|---|---:|---:|
| decode | **33.00 tok/s** | 27.8 tok/s |
| prefill 1100 tokens | **280.8 tok/s** | 16.3 tok/s (token by token) |
| prefill 8192 tokens | **349.8 tok/s** | not measured |

**State restart** (`server_check.py --state-restart`, 2 GiB cache): 4096 of 4212 prompt tokens restored, TTFT
1416 ms after the restart, identical greedy answer.

## Scope boundary

- On the M4 Pro the llama.cpp oracle cannot run (out of GPU memory). Its correctness evidence is the
  bit-identity with the M4 Max's outputs.
- 32K/60K parity against llama.cpp was not run.
- The float16 KV cache is not implemented.
- The 16K single-logit bound is not re-derived.
