# Red Lite dev16 — complete full-attention parity

Target: Apple M4 Pro, 24 GiB unified memory, Bartowski
`Qwen_Qwen3-Next-80B-A3B-Instruct-IQ2_XXS.gguf`.

Dev16 starts after the complete recurrent transformer block passed on the real
model. The layer audit establishes the complementary path:

- 48 total transformer layers;
- 36 recurrent/DeltaNet layers;
- 12 full-attention layers at `3,7,11,15,19,23,27,31,35,39,43,47`;
- zero mixed or unknown layers and 48/48 attention/post-attention norm pairs.

## Pinned full-attention graph

The implementation follows pinned llama.cpp commit
`7798007a29a90e3053e799394da48cf53a2f8e0f`:

1. input RMSNorm;
2. joint query + sigmoid-gate projection;
3. separate key and value projections;
4. per-head query and key RMSNorm;
5. partial NeoX RoPE;
6. grouped-query attention over the persistent K/V cache;
7. sigmoid gate on the attention result;
8. output projection.

The real GGUF geometry is:

- hidden width: 2048;
- query heads: 16;
- K/V heads: 2;
- head width: 256;
- NeoX RoPE dimensions: 64 of 256;
- RoPE frequency base: 10,000,000;
- `attn_q`: IQ2_XXS `(2048,8192)`, interleaved query/gate per head;
- `attn_k` and `attn_v`: Q4_K `(2048,512)`;
- `attn_output`: Q4_K `(4096,2048)`.

`redlite-attention` uses an independent native C oracle and standalone Metal
kernels. The parity fixture starts from deterministic prior K/V slots, appends
the projected current K/V values, executes 16-to-2 GQA with causal softmax,
gates the 4096-value attention result, and compares the complete 2048-value
output. The field suite covers positions 0, 1 and 15 (contexts 1, 2 and 16), so
both the RoPE identity case and non-zero positions are protected.

`redlite-attention-block` composes the validated branch with the attention
residual, real post-attention RMSNorm, full-width top-10 routed MoE plus shared
expert, and final residual. CPU and Metal paths remain independent until each
parity comparison.

## Build and field validation

```bash
make native

.deps/redmetal/redlite-attention parity \
  models/Qwen_Qwen3-Next-80B-A3B-Instruct-IQ2_XXS.gguf \
  --layer 3 \
  --position 7

.deps/redmetal/redlite-attention-block parity \
  models/Qwen_Qwen3-Next-80B-A3B-Instruct-IQ2_XXS.gguf \
  --layer 3 \
  --position 7 \
  --top-k 10 \
  --cache-mib 256
```

Observed on the target M4 Pro:

```text
weight read CPU    : 3.488 ms / 7 calls / 9.760 MiB
weight read GPU    : 1.065 ms / 7 calls / 9.760 MiB
SSD during compute : 0 bytes / 0 calls
CPU / GPU compute  : 25.527 / 8.575 ms
Q+RoPE max abs     : 7.15256e-06
K+RoPE max abs     : 4.52995e-06
GQA softmax max abs: 6.91414e-06
Q4_K out max abs   : 1.52588e-05
FULL ATTENTION     : YES
```

Context regression on the same target:

```text
layer 3 / position 0  / context 1  : FULL ATTENTION YES
layer 3 / position 1  / context 2  : FULL ATTENTION YES
layer 3 / position 15 / context 16 : FULL ATTENTION YES
```

Complete block result:

```text
attention max abs     : 1.52588e-05
KV cache max abs      : 4.52995e-06
post-attn RMSNorm abs : 3.93391e-06
full FFN max abs      : 3.33786e-06
FINAL block max abs   : 1.52588e-05
COMPLETE FULL ATTENTION BLOCK: YES
```

## Scope boundary

This checkpoint validates one complete full-attention transformer block with a
real autoregressive cache append. It does not claim 48-layer scheduling or
end-to-end token-generation parity. The next correctness gate is the native
48-layer scheduler that alternates the 36 recurrent and 12 full-attention block
implementations according to the audited GGUF layer map.
