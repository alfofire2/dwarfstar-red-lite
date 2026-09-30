# Red Lite dev15a — DeltaNet projection field validation

Target: Apple M4 Pro, 24 GiB unified memory, Bartowski `Qwen_Qwen3-Next-80B-A3B-Instruct-IQ2_XXS.gguf`.

## Layer graph audit

The corrected native layer audit passed on the real GGUF:

- GGUF v3, 843 tensors;
- 456 non-FFN layer tensors, 511.831 MiB physical span;
- 48/48 layers discovered;
- attention pattern: 36 recurrent/DeltaNet, 12 full-attention, 0 mixed, 0 unknown;
- norm pairs: 48/48 after correcting the GGUF suffix to `post_attention_norm.weight`;
- recurrent layers: `0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22,24,25,26,28,29,30,32,33,34,36,37,38,40,41,42,44,45,46`;
- full-attention layers: `3,7,11,15,19,23,27,31,35,39,43,47`.

All layers reported `norms=YES`.

## DeltaNet projection parity — real layer 0

`redlite-deltanet-proj parity ... --layer 0 --rows 8` passed on the real model.

Discovered layout:

- RMS epsilon: `9.99999997e-07`;
- hidden: 2048;
- QKV: IQ2_XXS `(2048,8192)`;
- Z gate: IQ2_XXS `(2048,4096)`;
- beta/alpha: Q8_0 `(2048,64)`.

Telemetry:

- CPU read: `0.703 ms / 4 calls / 33.250 KiB` (only requested rows for quant projections);
- GPU read: `2.788 ms / 4 calls / 6.328 MiB` (full projection tensors into shared Metal buffers);
- SSD during compute: `0 bytes / 0 calls`;
- CPU compute: `0.065 ms`;
- GPU compute: `3.939 ms`.

Parity:

- RMSNorm max abs/rel: `2.01322e-07 / 1.08407e-07`, YES;
- QKV max abs/rel: `1.04662e-06 / 3.72052e-05`, YES;
- Z max abs/rel: `9.24545e-07 / 1.02637e-05`, YES;
- BA max abs/rel: `6.47018e-07 / 6.88752e-07`, YES;
- overall projection parity: YES.

Representative row 0:

- QKV GPU/CPU: `-0.6311109 / -0.6311105`;
- Z GPU/CPU: `-0.4139920 / -0.4139925`;
- BA GPU/CPU: `+0.1260645 / +0.1260644`.

## Validation boundary

Dev15a validates the real-model input RMSNorm and the optimized DeltaNet front-end projections (`attn_qkv`, `attn_gate`, `ssm_ba`), including the first native Q8_0 path. It does not yet validate convolutional recurrent state handling or the Gated DeltaNet matrix-state update.

Next checkpoint: beta/alpha parameter transforms, conv-state shift, real F32 `ssm_conv1d`, SiLU, Q/K/V split, and per-head Q/K L2 normalization before the recurrent delta update.
