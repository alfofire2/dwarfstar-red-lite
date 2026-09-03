# Red Lite dev15 — DeltaNet projection bring-up

Target: Apple M4 Pro, 24 GiB unified memory, Bartowski `Qwen_Qwen3-Next-80B-A3B-Instruct-IQ2_XXS.gguf`.

The dev15 layer audit field run established the real non-FFN graph layout:

- 48 decoder layers;
- 36 recurrent/DeltaNet layers;
- 12 full-attention layers at `3,7,11,15,19,23,27,31,35,39,43,47`;
- no mixed or unknown layers;
- recurrent layers use the same tensor type/shape pattern across the model;
- full-attention layers use the same tensor type/shape pattern across the model.

The initial audit printed `norm pairs: 0/48` because Red Lite looked for `attn_post_norm.weight`, while the target GGUF stores the tensor as `post_attention_norm.weight`. This was an audit-name bug only: every layer contains both `attn_norm.weight` and `post_attention_norm.weight`. The audit has been corrected and now requires all discovered layers to have both norms.

## Pinned DeltaNet order

The pinned llama.cpp Qwen3-Next optimized recurrent path starts with:

1. decoder input RMSNorm;
2. `attn_qkv.weight` projection;
3. separate `attn_gate.weight` projection producing Z;
4. `ssm_ba.weight` projection producing beta/alpha parameters;
5. beta sigmoid and alpha softplus/gating;
6. convolution-state update and SiLU;
7. Q/K L2 normalization and recurrent delta update;
8. gated per-head RMSNorm;
9. `ssm_out.weight` projection.

Dev15 intentionally brings this up in smaller parity checkpoints rather than implementing the complete stateful path at once.

## Field-derived recurrent dimensions

Representative recurrent layer 0:

- hidden: `2048`;
- `attn_norm.weight`: F32 `(2048)`;
- `attn_qkv.weight`: IQ2_XXS `(2048,8192)`;
- `attn_gate.weight`: IQ2_XXS `(2048,4096)`;
- `ssm_ba.weight`: Q8_0 `(2048,64)`;
- `ssm_conv1d.weight`: F32 `(4,8192)`;
- `ssm_dt.bias`: F32 `(32)`;
- `ssm_a`: F32 `(32)`;
- `ssm_norm.weight`: F32 `(128)`;
- `ssm_out.weight`: Q4_K `(4096,2048)`.

These shapes imply 32 value heads of width 128 and 16 key groups of width 128, consistent with the pinned optimized Qwen3-Next graph.

## dev15a projection parity

`redlite-deltanet-proj` validates the stateless front of a real recurrent layer before any recurrent-state arithmetic is introduced:

`input -> RMSNorm -> {QKV IQ2_XXS, Z IQ2_XXS, beta/alpha Q8_0}`

The CPU path is an independent native reference. The Metal path reads the full four real weight tensors into shared Metal buffers before dispatch, executes the RMSNorm and the requested output rows, and performs no GGUF reads during the command buffer.

The tool also adds a standalone Q8_0 CPU decoder fixture and RMSNorm fixture to `make native`.

Build:

```bash
make native
```

Field parity on recurrent layer 0:

```bash
.deps/redmetal/redlite-deltanet-proj parity \
  models/Qwen_Qwen3-Next-80B-A3B-Instruct-IQ2_XXS.gguf \
  --layer 0 \
  --rows 8
```

Expected gate:

```text
RMSNorm ... parity=YES
QKV ... parity=YES
Z ... parity=YES
BA ... parity=YES
projection parity  : YES
```

## Next gates

After real-model dev15a parity passes:

1. implement beta/alpha transform plus convolution/state preparation and compare intermediate Q/K/V/gate/beta tensors;
2. implement the recurrent delta update and gated per-head norm;
3. add Q4_K `ssm_out` projection parity;
4. compose recurrent attention residual + post-attention RMSNorm + the already field-validated dev14 complete FFN + final residual;
5. bring up one representative full-attention layer separately before executing complete 48-layer token flow.
