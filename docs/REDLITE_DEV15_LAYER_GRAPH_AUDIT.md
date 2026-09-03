# Red Lite dev15 — Qwen3-Next layer graph audit

Dev15 starts after complete real-model FFN parity passed on representative routed/shared quant combinations.

## Pinned decoder-block order

The pinned llama.cpp Qwen3-Next graph executes each trunk layer as:

1. RMSNorm with `blk.<layer>.attn_norm.weight`;
2. recurrent gated DeltaNet or full attention, depending on the layer pattern;
3. attention residual add;
4. RMSNorm with `blk.<layer>.attn_post_norm.weight`;
5. complete routed + shared FFN;
6. FFN residual add.

When an explicit recurrent-layer map is absent, the pinned implementation defaults to full attention every fourth layer and recurrent linear attention otherwise. Red Lite does not assume that fallback for the target GGUF; dev15 audits the actual tensor directory first.

## Audit strategy

`redlite-layer-audit` is a standalone native C GGUF tensor-directory scanner. It deliberately excludes all `blk.N.ffn_*` tensors already validated in dev12-dev14 and inspects the remaining per-layer graph tensors.

It reports:

- total non-FFN layer tensors and physical span;
- number of discovered layers;
- recurrent/DeltaNet vs full-attention classification counts;
- exact recurrent and full-attention layer lists;
- presence of both attention RMSNorm tensors per layer;
- optional exact tensor suffix, GGML type, shape, physical span and absolute offset for every layer.

The classifier uses structural signals rather than assuming a fixed interval:

- recurrent: `ssm_*`, `attn_qkv.weight`, `attn_gate.weight`;
- full attention: `attn_q.weight`, `attn_k.weight`, `attn_v.weight`, `attn_output.weight`, `attn_q_norm.weight`, `attn_k_norm.weight`.

FFN tensors are explicitly excluded before classification.

## Build and field command

`make native` now also runs the layer-audit name/classifier selftest and builds:

`.deps/redmetal/redlite-layer-audit`

Field audit:

```bash
.deps/redmetal/redlite-layer-audit \
  models/Qwen_Qwen3-Next-80B-A3B-Instruct-IQ2_XXS.gguf \
  --tensors
```

Expected high-level architecture from the pinned fallback is 48 layers with 36 recurrent/DeltaNet and 12 full-attention layers, but field output is authoritative.

## Next implementation gate

No attention or DeltaNet arithmetic is added until this audit identifies the exact real tensor names, types, dimensions and layer pattern. After field validation, dev15 should bring up one representative recurrent layer first and one full-attention layer second, each with an independent native CPU oracle before composing the complete decoder block.
