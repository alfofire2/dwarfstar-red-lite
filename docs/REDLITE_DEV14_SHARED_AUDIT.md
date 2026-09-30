# Red Lite dev14 — Qwen3-Next shared expert bring-up

Dev14 starts after the real Qwen3-Next router and router-selected routed branch passed native field parity on the target Apple M4 Pro.

## Pinned shared-expert semantics

The pinned llama.cpp Qwen3-Next graph computes the shared branch from the same post-attention-normalized FFN input used by the routed MoE branch.

The shared expert uses four tensors per layer:

- `blk.<layer>.ffn_gate_inp_shexp.weight` — scalar shared-expert gate from the hidden state;
- `blk.<layer>.ffn_gate_shexp.weight` — shared FFN gate projection;
- `blk.<layer>.ffn_up_shexp.weight` — shared FFN up projection;
- `blk.<layer>.ffn_down_shexp.weight` — shared FFN down projection.

The pinned graph applies:

1. `shared_ffn = down(SiLU(gate(x)) * up(x))`;
2. `shared_gate = sigmoid(ffn_gate_inp_shexp(x))`;
3. `shared_out = shared_ffn * shared_gate`;
4. `ffn_out = routed_moe_out + shared_out`.

The scalar gate is therefore separate from the internal gated-FFN SiLU gate.

## Real-model field audit

Target GGUF: `Qwen_Qwen3-Next-80B-A3B-Instruct-IQ2_XXS.gguf`.

The standalone native audit found:

- 192 exact shared tensors;
- 48 shared layers;
- 48/48 complete four-tensor sets;
- shape consistency: YES;
- hidden size: 2048;
- shared FFN width: 512;
- total shared payload: 78.000 MiB;
- `ffn_gate_inp_shexp`: F32 on all 48 layers;
- `ffn_gate_shexp`: Q6_K on 24 layers, IQ2_XXS on 24 layers;
- `ffn_up_shexp`: Q6_K on 24 layers, IQ2_XXS on 24 layers;
- `ffn_down_shexp`: Q6_K on 24 layers, IQ2_XXS on 24 layers.

The three FFN matrices use the same quant type within every audited layer. Layer 0 is a Q6_K field-test target; layer 6 is an IQ2_XXS field-test target.

The Q6_K matrix payload is 0.820 MiB per tensor for these shapes. The IQ2_XXS matrix payload is 0.258 MiB per tensor. The F32 scalar-gate vector is 0.008 MiB per layer.

Because the complete shared payload is only 78 MiB, the final runtime should strongly consider keeping all shared-expert weights resident rather than placing them in the routed-expert LRU. Dev14 correctness testing still uses explicit per-layer reads so storage and arithmetic remain easy to audit.

## Native arithmetic implementation

Dev14 implements only the two real shared FFN formats found by the audit:

- Q6_K (`GGML_TYPE_Q6_K = 14`): 256 values / 210-byte block, using the exact pinned ggml block layout and dequantization formula;
- IQ2_XXS (`GGML_TYPE_IQ2_XXS = 16`): 256 values / 66-byte block, using the canonical IQ2_XXS grid already validated during earlier Red Metal bring-up.

A portable native decoder test constructs exact synthetic all-ones blocks for each format and requires a 256-element unit-vector dot product of exactly `256.0`.

`redlite-shared parity MODEL --layer N --rows N` then executes the real shared branch twice:

1. independent native C CPU oracle;
2. correctness-first Metal implementation.

Both paths evaluate the F32 scalar gate, sigmoid, complete gate/up projections, `SiLU(gate) * up`, selected down rows, and scalar gating. Metal reads all four weight tensors before the command buffer; no GGUF reads occur during shared compute.

Field targets:

- layer 0: Q6_K;
- layer 6: IQ2_XXS.

Once both pass numerical parity, dev14 can compose the already validated real-router routed output with the shared output and validate the complete Qwen3-Next FFN result. The first composition may remain correctness-first; permanent shared residency and fused routed+shared scheduling are subsequent runtime optimizations.
