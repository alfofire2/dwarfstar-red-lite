# Red Lite dev14 — shared expert field validation

Target: Apple M4 Pro, 24 GiB unified memory, real `Qwen_Qwen3-Next-80B-A3B-Instruct-IQ2_XXS.gguf`.

## Shared audit

The real model contains 48 complete shared-expert layers. Each layer has:

- `ffn_gate_inp_shexp.weight`: F32 scalar gate, hidden 2048;
- `ffn_gate_shexp.weight`;
- `ffn_up_shexp.weight`;
- `ffn_down_shexp.weight`;
- shared FFN width 512.

The three shared FFN matrices are Q6_K in 24 layers and IQ2_XXS in 24 layers. Total shared payload is about 78 MiB.

## IQ2_XXS field parity

Layer 6, rows 0..7:

- scalar gate CPU/GPU: `0.407262824 / 0.407262862`;
- scalar-gate absolute error: `3.76363e-08`;
- shared max absolute error: `8.18691e-08`;
- shared max relative error: `4.46434e-06`;
- shared parity: YES;
- GGUF reads during Metal shared compute: zero.

## Q6_K FP16-subnormal oracle bug

Initial layer-0 shared parity failed while Metal outputs were stable. An isolated Q6_K matrix probe exposed a nearly exact factor-of-two discrepancy between GPU and CPU reference values.

The fault was in the dev14 CPU half-to-float helper for FP16 subnormals. The exponent reconstruction used `127 - 15 - shift`; the correct half-subnormal exponent is `127 - 14 - shift`. Real Q6_K super-block scales in this model exercise the subnormal path, while the original synthetic test used normal FP16 `d=1.0` and therefore did not expose the bug.

The native routed reference used by earlier dev12/dev13 validation already had a correct FP16-subnormal converter and is not invalidated by this fix.

A new synthetic Q6_K regression test uses a subnormal half scale and is required to pass during `make native`.

## Q6_K isolated field parity after oracle fix

Layer 0, first 8 rows of each real shared matrix:

- gate: max abs `4.26546e-07`, max rel `1.24716e-05`, parity YES;
- up: max abs `1.2666e-07`, max rel `8.6496e-07`, parity YES;
- down: max abs `1.00583e-07`, max rel `1.28063e-06`, parity YES;
- overall Q6 matrix parity: YES.

Representative gate row 1:

- GPU `-0.5087475`;
- CPU `-0.5087476`;
- delta `+1.118e-07`.

## Q6_K complete shared branch parity

Layer 0, rows 0..7:

- scalar gate CPU/GPU: `0.544462628 / 0.544462562`;
- scalar-gate absolute error: `6.61313e-08`;
- shared max absolute error: `1.23826e-07`;
- shared max relative error: `1.26553e-05`;
- shared parity: YES;
- GGUF reads during Metal shared compute: zero.

This validates both shared-expert quant families present in the real model.

## Complete FFN composition checkpoint

After Q6_K and IQ2_XXS shared parity both passed, dev14 adds `redlite-ffn parity` as a correctness-first composition test. It evaluates independently:

1. F32 router CPU and Metal;
2. softmax -> real top-10 -> renormalized weights;
3. routed top-10 expert branch CPU and Metal;
4. gated shared-expert branch CPU and Metal;
5. `FFN = routed + shared` on both paths;
6. final CPU/Metal FFN parity.

The first complete-FFN field targets remain layer 0 (Q6_K shared / IQ2_XS routed) and layer 6 (IQ2_XXS shared / IQ1_M routed). This is a correctness milestone, not yet a fused or performance-optimized full-layer scheduler.
