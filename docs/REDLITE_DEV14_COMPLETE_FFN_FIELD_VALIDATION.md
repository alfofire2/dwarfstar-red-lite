# Red Lite dev14 — complete Qwen3-Next FFN field validation

Target: Apple M4 Pro, 24 GiB unified memory, Bartowski `Qwen_Qwen3-Next-80B-A3B-Instruct-IQ2_XXS.gguf`.

Dev14 closes the complete feed-forward block on the real model. The validated graph is:

`F32 router -> softmax -> top-10 -> renormalized routed MoE + sigmoid-gated shared expert -> FFN output`.

Both CPU and Metal paths are standalone native implementations with no Python or ctypes in the tested path.

## Layer 0

This layer exercises routed IQ2_XS experts plus a Q6_K shared expert.

- router IDs match: YES;
- router max logit error: `4.76837e-07`;
- router max selected-weight error: `8.9407e-08`;
- selected experts: `438,41,204,493,139,451,412,200,367,154`;
- routed cold loads: 10;
- routed positional reads: 30;
- routed SSD reads during Metal compute: 0 bytes / 0 calls;
- routed GPU / CPU: `9.285 / 21.592 ms`;
- routed max abs / rel error: `2.20497e-08 / 4.09202e-06`;
- routed parity: YES;
- shared quant: Q6_K gate/up/down;
- shared scalar gate CPU / GPU: `0.544462628 / 0.544462562`;
- shared read calls: CPU=4, GPU=4;
- shared SSD reads during Metal compute: 0 bytes / 0 calls;
- shared GPU / CPU: `3.017 / 1.787 ms`;
- shared max abs / rel error: `1.23826e-07 / 1.26553e-05`;
- shared parity: YES;
- complete FFN max abs / rel error: `1.32396e-07 / 6.31249e-05`;
- complete FFN parity: YES.

## Layer 6

This layer exercises routed IQ1_M experts plus an IQ2_XXS shared expert.

- router IDs match: YES;
- router max logit error: `4.76837e-07`;
- router max selected-weight error: `2.98023e-08`;
- selected experts: `480,288,135,505,143,344,485,182,111,83`;
- routed cold loads: 10;
- routed positional reads: 30;
- routed SSD reads during Metal compute: 0 bytes / 0 calls;
- routed GPU / CPU: `8.301 / 22.242 ms`;
- routed max abs / rel error: `1.19063e-08 / 2.6241e-06`;
- routed parity: YES;
- shared quant: IQ2_XXS gate/up/down;
- shared scalar gate CPU / GPU: `0.407262824 / 0.407262862`;
- shared read calls: CPU=4, GPU=4;
- shared SSD reads during Metal compute: 0 bytes / 0 calls;
- shared GPU / CPU: `2.926 / 1.736 ms`;
- shared max abs / rel error: `8.18691e-08 / 4.46434e-06`;
- shared parity: YES;
- complete FFN max abs / rel error: `7.44724e-08 / 1.63362e-05`;
- complete FFN parity: YES.

## Q6_K oracle correction

During dev14 bring-up, Q6_K Metal initially appeared to disagree with the native CPU reference. An isolated real-matrix probe showed Metal outputs at exactly twice the CPU result for affected rows. The actual defect was in the CPU FP16-to-FP32 conversion for half subnormal values: the exponent used `127 - 15 - shift` instead of `127 - 14 - shift`.

After fixing the oracle and adding a dedicated FP16-subnormal regression fixture:

- Q6_K gate matrix parity: YES;
- Q6_K up matrix parity: YES;
- Q6_K down matrix parity: YES;
- complete Q6_K shared branch parity: YES.

The routed dev12/dev13 reference converter used a different, correct half-subnormal implementation and is not invalidated by this fix.

## Validation boundary

Dev14 validates the complete Qwen3-Next FFN for representative real layers covering both routed quant families and both shared-expert quant families. It does not yet validate the complete decoder block. The next milestone must cover the surrounding graph:

`RMSNorm -> DeltaNet/full attention -> residual -> RMSNorm -> complete FFN -> residual`.
