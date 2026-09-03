# Red Lite dev15e — complete recurrent DeltaNet layer composition

Status at implementation commit: awaiting M4 field result.

This checkpoint composes the previously field-validated Qwen3-Next recurrent-attention stages into one single-token forward on the real GGUF layer 0:

1. input RMSNorm using `blk.0.attn_norm.weight`
2. full optimized projections:
   - `attn_qkv` IQ2_XXS: 2048 -> 8192
   - `attn_gate` IQ2_XXS: 2048 -> 4096
   - `ssm_ba` Q8_0: 2048 -> 64
3. real F32 conv/prestate path with beta/decay transforms and Q/K L2 normalization
4. recurrent Gated DeltaNet state update using the pinned llama.cpp transposed state layout
5. gated RMSNorm with `ssm_norm.weight` and `SiLU(z)`
6. real Q4_K `ssm_out`: 4096 -> 2048

CPU and Metal start from the same deterministic hidden vector, convolution cache and 2 MiB recurrent cache. Each implementation feeds its own stage output into the next stage; intermediate values are not reset to shared fixtures.

The CLI is:

```bash
.deps/redmetal/redlite-deltanet-layer parity MODEL --layer 0
```

The decisive field line is:

```text
COMPLETE DELTANET   : YES
```

This is still a recurrent-attention branch checkpoint. It does not yet include the transformer residual connection, post-attention RMSNorm, MoE FFN, FFN residual, embedding/output head, or full-attention layers.
