# Red Lite dev38 — decode: expert down-projection lanes and concurrent encoders

Status: **done** on branch `dev/decode-kernels` (from `dev/expert-slots` at 58a58ce). Apple M4 Max 48 GiB only.

## Finding

At full residency a decode token is GPU-bound: 12.1 ms of wall time, 11.7 ms of GPU time (0.44 ms CPU
gap, measured with a temporary per-step print). About 1.3 GB of weights per token in 11.7 ms is ~110
GB/s, a fifth of this chip's bandwidth. Two measurements located the waste:

- **Serialization.** Every decode dispatch ran in a serial encoder. With a concurrent encoder and
  **no** barriers at all (wrong output, an upper bound only) decode went from 74 to 118 tok/s.
- **Routed experts.** `RL_ENGINE_PROFILE=1` (IQ2_XXS): routed experts 4.07 of 11.98 ms per token (34 %),
  at ~85 GB/s against 134–169 GB/s for the dense IQ2_XXS GEMVs. The down projection (2048 rows of 512
  columns per expert) used 32 lanes per row: 16 weights per lane, then a 5-step shuffle reduction.

## Changes

1. **Lanes per row** (`lanes_for_blocks`, `redmetal_topk.m`): at most 4 lanes per 256-value block (was 16).
   The 512-column down rows get 8 lanes of 64 values; the 2048-column gate/up rows keep 32. Sweep of the
   down lanes (single runs, IQ2_XXS): 32: 79.3, 16: 83.6, 8: **84.8**, 4: 83.9 tok/s; gate/up 16 and 8
   lanes were not faster than 32.
2. **Concurrent decode encoders** (`redmetal_engine.m`, default; `RL_ENGINE_CONCURRENT=0` restores
   serial encoders; the profiler always uses serial ones). Every dispatch is followed by a buffer
   barrier except inside groups of mutually independent dispatches: DeltaNet {qkv, z, ba} and {ba, conv}
   then {l2, shift}; attention {q, k, v}; FFN {router, shared scalar, shared gate, shared up}. The
   expert pool barriers its own dependent dispatches when the encoder is concurrent. Prefill encoders
   are serial and unchanged.
3. `scripts/dev/quick_parity.sh` compares a non-reference model with its own llama.cpp dumps
   (`.deps/regress-<name>`) and uses `--cache-mib full`; it compared IQ3_XXS against the IQ2_XXS dumps.

## Measurements (cooled, alternated, same session)

Decode tok/s, full residency, 256 tokens (`tests/fixtures/cache_trace_prompts.txt` line 1):

| model | dev37 | lanes (serial) | lanes + concurrent (default) |
|---|---|---|---|
| IQ3_XXS | 69.02 / 70.37 / 70.39 (median 70.4) | 74.88 / 75.44 / 74.59 (74.9) | 79.08 / 79.03 / 80.54 (**79.1**) |
| IQ2_XXS | 75.26 / 75.36 / 73.53 (75.3) | 78.39 / 80.10 / 81.18 (80.1) | 80.95 / 84.57 / 82.59 (**82.6**) |

IQ3_XXS: +12 %, against 68.5 tok/s for the pinned llama.cpp on the same file (dev33). IQ2_XXS: +10 %.

4 GiB cache, IQ2_XXS (dev37 binary vs default): 33.09 (cold page cache, first run) / 54.88 / 52.51
vs 59.82 / 59.91 / 59.27 tok/s: median **52.5 → 59.8 (+14 %)**.

Absolute numbers move between sessions with the machine's thermal state (the same dev37 binary measured
69–77 tok/s on different hours), so only same-session columns are compared.

## Tried and not kept (see `docs/WHAT_DID_NOT_WORK.md`)

- Two more groups ({ssm_out, conv-state copy}; {layer-output copy, next RMSNorm}): 83.72 vs 83.77 tok/s.
- Overlapping CPU encoding with GPU execution: the CPU gap is 0.44 ms of 12.1 ms, not attempted.

## Validation

Parity with the new defaults: `quick_parity.sh --long --batch 2048` on IQ2_XXS and IQ3_XXS (engine CPU
vs Metal, GPU-routed at full residency, logits and greedy vs llama.cpp, 1200-token long context): PASS.
`redlite-engine kernel-selftest`: OK. `rm -rf .deps/redmetal`, `make redmetal`, `make native`,
`make sanitize`: 0 warnings; `make test` and ruff green; `scripts/regress_m4.sh`: IQ2_XXS **49/49**,
IQ3_XXS **36 passed, 0 failed, 13 skipped**.

## Scope boundary

- M4 Max 48 GiB only. The lane rule and the encoders also run on the M4 Pro 24 GiB path; nothing is
  claimed there.
- The lane change alters the order of the down-projection sums (float rounding), within every parity
  bound above; router ids unchanged in every check.
- The concurrent encoder still has a barrier after most dispatches; the 118 tok/s no-barrier figure is an
  upper bound with wrong output, not a target.
