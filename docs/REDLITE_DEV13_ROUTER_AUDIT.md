# Red Lite dev13 — native Qwen3-Next router bring-up

Dev13 starts only after the standalone routed expert subsystem passed real-model native field validation on both IQ2_XS and IQ1_M.

## Pinned Qwen3-Next routing semantics

The pinned llama.cpp Qwen3-Next implementation (`7798007a29a90e3053e799394da48cf53a2f8e0f`) creates one router tensor per routed layer:

`blk.<layer>.ffn_gate_inp.weight`

with logical shape `{ hidden_size, expert_count }`.

For the target 80B-A3B architecture that means 2048 hidden features and 512 router logits.

The pinned graph calls `build_moe_ffn()` with:

- `n_expert = 512`;
- `n_expert_used = 10`;
- SiLU expert activation;
- `norm_w = true`;
- softmax expert gating;
- default `expert_weights_scale = 0.0`, which means the generic MoE graph does not apply an additional post-normalization weight scale.

The effective router sequence is:

1. router matrix-vector product -> 512 logits;
2. softmax across all 512 logits;
3. select top-10 experts;
4. gather those 10 softmax probabilities;
5. renormalize selected probabilities by their selected-weight sum;
6. use those weights for routed expert accumulation.

## Real-model router audit — field validated

Target model: Bartowski `Qwen3-Next-80B-A3B-Instruct` IQ2_XXS GGUF on Apple M4 Pro / 24 GiB.

Observed result:

- router tensors: 48;
- unique layers: YES;
- consistent shape: YES;
- logical shape: `(2048, 512)`;
- all router tensors: `GGML_TYPE_F32 (0)`;
- each router: exactly 4.000 MiB;
- total router payload: 192.000 MiB;
- layer coverage: 0 through 47.

This removes the need for any quantized router decoder in the target runtime.

## Native F32 router execution

Dev13 adds two independent arithmetic paths.

### CPU oracle

- positional read of the real 4 MiB F32 router tensor;
- independent 2048x512 matrix-vector product;
- stable softmax;
- deterministic top-k;
- selected-weight renormalization.

### Metal path

- positional read of the same real router directly into a shared `MTLBuffer`;
- correctness-first F32 Metal router matvec producing all 512 logits;
- the same native softmax/top-k/renormalization stage.

The CPU and GPU paths read the router independently so the field test checks the real GGUF bytes and the Metal arithmetic rather than comparing aliases of one copied buffer.

## Validation executable

On macOS, `make native` builds:

`.deps/redmetal/redlite-router`

### Router-only parity

```text
redlite-router router-parity MODEL --layer N --top-k 10
```

Checks:

- all 512 GPU logits against the independent CPU oracle;
- exact ordered top-10 expert IDs;
- selected normalized router weights;
- CPU/GPU router read and compute telemetry.

### Real-router routed-MoE parity

```text
redlite-router routed-parity MODEL --layer N --top-k 10 --rows 8 --cache-mib 256
```

After router parity passes, this command feeds the **real router-selected** expert IDs and weights into the existing native resident Metal expert executor and compares the routed output against the independent C expert reference.

This validates:

`real F32 router -> softmax -> top-10 -> renorm -> LRU residency -> SSD expert loads -> IQ2_XS/IQ1_M Metal FFNs -> GPU weighted accumulation`

The command intentionally validates only the routed-expert branch. Qwen3-Next's separate gated shared-expert branch is not integrated yet.

## Portable regression coverage

A platform-independent test verifies:

- stable softmax;
- descending top-k selection;
- selected-weight renormalization to sum 1;
- deterministic lower-index tie breaking.

The synthetic GGUF fixture also proves exact `ffn_gate_inp.weight` discovery without matching `ffn_gate_inp_shexp`.

## Next milestone

If representative IQ2_XS and IQ1_M layers pass `routed-parity`, dev14 will integrate the shared-expert branch. Only after routed + shared FFN parity is established should Red Lite move upward into post-attention RMSNorm/full layer parity and eventually the Gated DeltaNet / attention graph.
