# Roadmap

Agreed with the project owner on 2026-10-03, updated on 2026-10-06. Updated as steps finish: each step links its milestone document when
done. Estimates are estimates; only measured numbers go into the results pages.

## Where things stand (0.8.0, 2026-10-06)

- **24 GiB Mac** (M4 Pro), file F2:
  - default, experts from the SSD: decode 34 tok/s;
  - every expert resident (raised GPU limit): 45–46 tok/s, 49–58 with MTP;
  - prompt ingestion about 360 tok/s;
  - the llama.cpp launcher decodes 36–38 tok/s there.
- **48 GiB Mac** (M4 Max), every expert resident:
  - IQ3_XXS 80 tok/s;
  - IQ2_XXS 86 tok/s, 97 with MTP (llama.cpp 72.5);
  - prompt ingestion about 900 tok/s;
  - Red Lite G2 (dev64) is an option with better perplexity at the same speed.
- **Coding agents:** Red Lite CF2 (dev63), Qwen3-Coder-Next at 2 bits, passes 12 of 12 harder repository tasks on
  both Macs. `redlite setup-pi` uses temperature 0.3, where the 2-bit Coder did not loop.
- **Long contexts:** the model's 262K are read correctly (needles at 256K); 64K is the agent window by default.
- **Project site** with the docs and a search: [redlite.alfonsodaniello.it](https://redlite.alfonsodaniello.it/).

## Next

- **Faster first ingestion of long prompts:** at 256K it takes 24 minutes. The prefill attention still reads each
  key/value block once per query head; sharing it across the 8 heads of a KV head needs Q in registers (Metal's
  32 KiB threadgroup memory does not hold the tile).
- **Agreement with llama.cpp past 64K:** the KL rises to 0.025 at 128K. Find which sum drifts.
- **G2 at long context** and a 48 GiB Coder mix (dev64).

## Done in 0.8.0 (dev65)

- Needles at 62K, 127K and 256K tokens (3 / 3 each), speeds on both Macs, llama.cpp agreement to 128K.
- A harder agent suite. The 64K window is the default for agents (16 / 18 against 10 / 18 at 32K).
- Prefill attention −20 % / −29 % at 32K / 64K, bit-identical.

## Done in 0.7.0 (dev63, dev64)

- **Agent loops at 2 bits (dev63):** pass rates 7–9 of 12 at four sampling settings; loops only at temperature 0.7
  and above.
- **A Red Lite quantization of Qwen3-Coder-Next (dev63):** CF2, code perplexity −3.3 %, agent suite 12/12 on both
  Macs.
- **48 GiB Macs (dev64):** G2 (IQ3_S experts), −0.65 % text / −1.4 % code, same speed with MTP; an option because of
  long prompts.

## Phase 1 — release 0.5.0 (done 2026-10-03)

1. `regress_m4.sh` green on the M4 Pro. The three llama.cpp-oracle checks run out of GPU memory there; skip them on
   machines where the oracle does not fit, and compare the native outputs bit for bit with a 48 GiB machine's
   instead.
2. Full checks on both Macs (`scripts/dev/local_ci.sh`, `regress_m4.sh` on both GGUFs, `quick_parity.sh`), then the
   release:
   - version bump in `VERSION`, `pyproject.toml` and `redlite/__init__.py`;
   - tarball (`scripts/package_release.sh`);
   - release notes and tag.

## Phase 2 — decode on 24 GiB Macs

- **2a — done (dev51).** Larger caches cut misses but not decode time. **Expert cache size sweep on the M4 Pro**, 4 → 14 GiB. Measure decode, misses, footprint, swap and memory
  pressure. The 4 GiB default was a cautious choice, never tested at the limit.
- **2b — done (dev51):** 46.0 tok/s, 52.7 with MTP, on the M4 Pro. **Every expert resident on 24 GiB.** IQ2_XXS needs 16.9 GiB of expert slots + 1.06 GiB of dense weights; the
  M4 Pro's default GPU working set is 17.76 GiB. With `iogpu.wired_limit_mb` raised (sudo, reset at reboot) the
  M4 Pro would get:
  - GPU-routed decode;
  - MTP speculation.

  Estimate, not measured: about 40 tok/s plain, since the M4 Pro has half the M4 Max's memory bandwidth.
  Needs a clear guide for users and planner support.
- **2c — done (dev51c):** on by default in `redlite chat` / `serve` with a bounded cache. **Cache-aware routing as the bounded-cache default.** It gains +5 % on the M4 Pro but changes outputs. Decide
  with the Qwen API comparison (dev48) and perplexity, both run on the M4 Pro.

## Phase 3 — long context (float16 KV built and not kept, dev53: see WHAT_DID_NOT_WORK)

- **Float16 KV cache.** It halves the attention cache's memory and traffic: at 33.5K tokens decode is 25.7 tok/s,
  mostly attention, and 64K positions would take 1.5 GiB instead of 3.
- It needs the prefill attention kernel (`attn_fa_b`) rewritten for half storage, and the parity checks against
  llama.cpp re-run.

## Phase 4 — product

- **Activation steering — done (dev52, `--history` in dev52b).** As in ds4:
  - a vector added to the residual stream at chosen layers;
  - strength adjustable during a chat (`/steer`);
  - an "only the first N tokens of the answer" mode;
  - a `--history FILE` of prefilled turns.
- **Low-power mode — measured:** −22 % decode at full residency, little with a bounded cache; energy not measured. **Low-power mode on the M4 Pro:** tok/s and temperature with `pmset lowpowermode`, for laptop users.
- **A Red Lite quantization — first result (dev54):** IQ2_XS on expert layers 37–47 beats Bartowski's scheme at the same size; published in 0.5.3 (`redlite download 24gb`). **A Red Lite quantization:** search a mix of precisions that beats IQ2_XXS at the same size, judged by
  perplexity and the API comparison. After Phase 2.
- **Concurrent server requests — done (dev56):** `--parallel 2` decodes two requests in one pass; total throughput
  +28 % on the M4 Pro (+10 % against MTP). Only with every expert resident.

## Not planned

- More prefill kernel tuning on the M4 Pro (dev49, dev50: a few percent per step).
- Chunk-parallel DeltaNet prefill (dev44): at most ~2 % (dev47).
- IQ3_M; distributed inference across two Macs.
- Hosted CI: removed on 2026-10-03; `scripts/dev/local_ci.sh` runs the same checks.

## Machines

- **M4 Max 48 GiB:** development, every GGUF at full residency.
- **M4 Pro 24 GiB (`ssh m4pro`):** the target class. Builds need `REDLITE_SDK=…/MacOSX26.5.sdk`.
  `sudo` is allowed only for `sysctl iogpu.wired_limit_mb` and `pmset -a lowpowermode`.
