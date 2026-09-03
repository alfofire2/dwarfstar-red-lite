# Red Lite dev13 — native Qwen3-Next router bring-up

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
- softmax expert gating;
- default `expert_weights_scale = 0.0`, which means the generic MoE graph does not apply an additional post-normalization weight scale.

The effective router sequence is therefore:

1. router matrix-vector product -> 512 logits;
2. softmax across all 512 logits;
3. select top-10 experts;
4. gather those 10 softmax probabilities;
5. because `norm_w=true`, renormalize the selected 10 probabilities by their selected-weight sum;
6. multiply routed expert outputs by the resulting normalized weights and sum.

Dev13 reproduces this sequence; it does not substitute sigmoid gating, pre-softmax weighting, or unnormalized selected probabilities.

## Real-model router audit — field result

The target Bartowski Qwen3-Next-80B-A3B GGUF was audited on the Apple M4 Pro target system.

Result:

- 48 exact `blk.N.ffn_gate_inp.weight` tensors;
- one unique router for each routed layer 0-47;
- consistent logical shape `(2048, 512)`;
- all 48 routers are `GGML_TYPE_F32 (0)`;
- each router occupies exactly 4.000 MiB;
- total router payload is 192.000 MiB.

This removes the need for any quantized router decoder in the target path.

## Native F32 router execution

Dev13 now adds two independent router arithmetic paths:

1. CPU oracle:
   - positional read of the real 4 MiB F32 router tensor;
   - independent 2048x512 matrix-vector product;
   - stable softmax;
   - deterministic top-k;
   - selected-weight renormalization.

2. Metal path:
   - positional read of the same real router directly into a shared `MTLBuffer`;
   - correctness-first F32 Metal router matvec producing all 512 logits;
   - the same native softmax/top-k/renormalization stage.

The CPU and GPU paths read the router independently so the field test verifies both real GGUF bytes and Metal arithmetic rather than comparing two views of one copied buffer.

## New validation executable

On macOS, `make native` now builds `.deps/redmetal/redlite-router`.

### Router-only parity

`redlite-router router-parity MODEL --layer N --top-k 10`

Checks:

- all 512 GPU logits against the independent CPU oracle;
- exact ordered top-10 expert IDs;
- selected normalized router weights;
- CPU/GPU router read and compute telemetry.

### Real-router routed-MoE parity

`redlite-router routed-parity MODEL --layer N --top-k 10 --rows 8 --cache-mib 256`

After router parity passes, this command feeds the **real router-selected** expert IDs and weights into the existing native resident Metal expert executor. It compares the routed output against the independent C expert reference.

This validates the chain:

`real F32 router -> softmax -> top-10 -> renorm -> LRU residency -> SSD expert loads -> IQ2_XS/IQ1_M Metal FFNs -> GPU weighted accumulation`

The command intentionally validates only the routed-expert branch. Qwen3-Next's separate gated shared-expert branch is not integrated in dev13 yet.

## Portable regression coverage

A platform-independent test verifies:

- stable softmax;
- descending top-k selection;
- selected-weight renormalization to sum 1;
- deterministic lower-index tie breaking.

The existing synthetic GGUF fixture also proves that exact `ffn_gate_inp.weight` discovery does not accidentally match `ffn_gate_inp_shexp`.

## Next step after field parity

If both a representative IQ2_XS layer and IQ1_M layer pass `routed-parity`, the next milestone is shared-expert integration. Only after routed + shared FFN parity is established should Red Lite move upward into post-attention RMSNorm/full layer parity and eventually the Gated DeltaNet / attention graph.
