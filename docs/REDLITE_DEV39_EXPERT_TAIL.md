# Red Lite dev39 — fused expert tail (GPU-routed decode)

Status: **done** on branch `dev/expert-kernels` (from `dev/decode-kernels` at d167976). Apple M4 Max 48 GiB only.

## Change

In a GPU-routed layer the routed experts ended with three dependent dispatches after the down projection:
the pool's weighted sum, `rl_scale_add` and the copy of the layer output. `rl_moe_tail` does all three in one
dispatch (one thread per hidden element): `routed = Σ_e weights[e] · tmp[e][i]` in the pool's order, then
`x = resid + (routed + shared · scalar)` exactly as `rl_scale_add`, then the layer-output copy. The pool skips its
sum when `redmetal_topk_pool_encode_device_into` gets a NULL output and exposes the per-expert outputs with
`redmetal_topk_pool_tmp_buffer`. `RL_ENGINE_FUSE_TAIL=0` restores the separate dispatches. The synchronous
(bounded-cache) path is unchanged.

## Measurements (IQ2_XXS, full residency, 256 tokens, alternated, 60 s cooling)

- **Kept, parallel tail:** 84.39 / 86.91 / 85.52 vs 82.97 / 85.86 / 84.15 tok/s (median 85.5 vs 84.2, +1.6 %).
  Two later pairs (30.13 / 45.35 vs 84.19 / 41.82) ran while other applications used the GPU and are excluded.
- **Reverted, single-threadgroup tail** that also folded in the next layer's RMSNorm (−3 barriers per layer):
  80.01 / 78.21 / 79.32 / 77.91 / 79.12 vs 84.40 / 83.73 / 84.96 / 83.46 / 84.90 (median 79.1 vs 84.4, −6 %):
  the 10-expert sum on one core costs more than the barriers it saves.

## Traps found (pre-existing, see `docs/WHAT_DID_NOT_WORK.md`)

- `redlite-engine parity --tokens 9707,11,1879,0,785,12884,13,151645,198 --cache-mib 2048`: router mismatch (2 ids)
  at position 7 with logit errors ~1e-5, identical with the dev37 engine and with dev38/dev39 switched off: a
  router near-tie on that ad-hoc sequence, not a regression.
- `redlite-engine prefill` over 1100 IQ3_XXS tokens reports `BATCHED PREFILL PARITY: NO` (worst 0.027, 0 router
  mismatches, same argmax); the dev37 engine gives 0.040. Its tolerance is sized for short sequences.

## Not done

IQ3_XXS prefill (837 vs 861 tok/s for llama.cpp, dev33) was not worked on: during this session other
applications used the GPU and the prefill benchmark read 335 tok/s, so no prefill measurement would have been
comparable.

## Validation

`rm -rf .deps/redmetal`, `make redmetal`, `make native`, `make sanitize`: 0 warnings; `make test` and ruff green;
`scripts/regress_m4.sh`: IQ2_XXS **49/49**, IQ3_XXS **36 passed, 0 failed, 13 skipped**; `quick_parity.sh --long
--batch 2048` on both models (engine, GPU-routed, logits and greedy vs llama.cpp, 1200-token long context): PASS.

## Scope boundary

- M4 Max 48 GiB only; the gain is small (+1.6 %, three clean pairs) and only on the GPU-routed path.
- Bit-identical arithmetic to the separate dispatches is intended but not asserted; parity is checked by the
  usual bounds.
