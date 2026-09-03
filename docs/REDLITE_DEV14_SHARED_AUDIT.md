# Red Lite dev14 — Qwen3-Next shared expert audit

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

## Why audit first

The global GGUF quant preset has already proven insufficient to infer individual tensor types: routed tensors are a mixture of IQ2_XS and IQ1_M while routers are F32. Dev14 therefore audits the real shared tensors before adding any arithmetic implementation.

`redlite-shared-audit MODEL --layers` scans only the four exact shared-expert tensor names above and reports:

- total tensor count and layer coverage;
- whether all four tensor kinds exist in every discovered layer;
- common hidden size and shared-FFN width;
- GGML type counts separately for gate-input, gate, up and down;
- physical spans and offsets per layer;
- shape-consistency invariants.

The audit intentionally does not confuse `ffn_gate_inp_shexp` with the routed router `ffn_gate_inp`.

## Expected structural shapes

From the pinned model loader:

- gate input: `(hidden)`;
- gate: `(hidden, shared_ffn)`;
- up: `(hidden, shared_ffn)`;
- down: `(shared_ffn, hidden)`.

For the target model the hidden size is already known to be 2048. The actual shared-FFN width and real GGML types are discovered from the GGUF rather than assumed.

## After the field audit

Implement only the actual formats observed in the target GGUF. The next correctness milestone is:

`shared tensor bytes -> native/Metal shared FFN -> sigmoid scalar gate -> gated shared output`

followed by:

`real routed output + real shared output -> complete Qwen3-Next FFN parity`

The routed path validated in dev13 remains unchanged while shared-expert bring-up is isolated.
