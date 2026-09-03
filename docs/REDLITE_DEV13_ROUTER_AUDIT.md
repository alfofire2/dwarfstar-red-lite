# Red Lite dev13 — native Qwen3-Next router audit

Dev13 starts only after the standalone routed expert subsystem passed real-model native field validation on both IQ2_XS and IQ1_M.

## Pinned Qwen3-Next routing semantics

The pinned llama.cpp Qwen3-Next implementation (`7798007a29a90e3053e799394da48cf53a2f8e0f`) creates one router tensor per routed layer:

`blk.<layer>.ffn_gate_inp.weight`

with logical shape:

`{ hidden_size, expert_count }`

For the target 80B-A3B architecture that means 2048 hidden features and 512 router logits.

The pinned graph calls `build_moe_ffn()` with:

- `n_expert = 512`;
- `n_expert_used = 10`;
- SiLU expert activation;
- `norm_w = true`;
- softmax expert gating.

The effective router sequence is therefore:

1. router matrix-vector product -> 512 logits;
2. softmax across all 512 logits;
3. select top-10 experts;
4. gather those 10 softmax probabilities;
5. because `norm_w=true`, renormalize the selected 10 probabilities by their selected-weight sum;
6. multiply routed expert outputs by the resulting normalized weights and sum.

Dev13 must reproduce this sequence exactly; it must not substitute sigmoid gating, pre-softmax top-k weighting, or unnormalized selected probabilities.

## Why audit precedes arithmetic

The routed gate/up/down tensor quantization was not what the global GGUF filename suggested: the real model contains IQ2_XS and IQ1_M routed tensors. The router tensor type must therefore also be discovered from the actual GGUF instead of assumed from the preset name.

Dev13 adds a standalone native `redlite-router-audit` executable. It scans only exact `blk.N.ffn_gate_inp.weight` names and deliberately rejects the separate shared-expert gate tensor name `ffn_gate_inp_shexp`.

The audit reports:

- router tensor count;
- unique routed-layer coverage;
- common/heterogeneous GGML types;
- logical shape;
- physical byte span and offset for each layer.

A portable synthetic GGUF regression fixture now includes a rank-2 F16 router tensor and proves exact router discovery in CI.

## Next step after real-model audit

Once the target GGUF router type is known, implement only that real type first:

`router bytes -> native/Metal matvec -> stable softmax(512) -> exact top-10 -> selected-weight renormalization`

Then add a `router-parity` command that compares router ids and weights against an independent native CPU oracle before connecting the result to the already validated resident top-k expert executor.
