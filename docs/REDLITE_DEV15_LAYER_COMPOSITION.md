# Red Lite dev15e — complete recurrent DeltaNet layer composition

Status: **FIELD VALIDATED on Apple M4 Pro / 24 GiB / ARM64**.

Validated commit: `385557e93d0372c6e209b102bf74a28d8352a13d`

M4 field workflow: `Red Lite M4 Field Validation`, run `33759672217`, success.

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

## M4 field result

```text
runtime             : complete native single-token Qwen3-Next DeltaNet layer parity
layer               : 0
geometry            : hidden=2048 channels=8192 qk=2048 value=4096 heads(k/v)=16/32
state               : conv=96.000 KiB recurrent=2.000 MiB
CPU weight read     : 1.459 ms / 9 calls / 10.954 MiB
CPU full compute    : 37.424 ms
GPU projection      : 3.512 ms compute / 0.576 ms read / 6.328 MiB
GPU prestate/state  : 1.888 / 2.081 ms
GPU tail            : 2.129 ms
SSD during compute  : prestate=0 bytes/0 calls; other stages=0
full QKV abs/rel    : 9.05991e-06 / 0.00359473
full Z abs/rel      : 6.67572e-06 / 0.000288968
full BA abs/rel     : 8.34465e-07 / 4.63736e-06 projection=YES
prestate Q/K/V abs  : 9.23872e-07 / 3.57628e-07 / 1.3411e-07
prestate gate abs   : 3.05176e-05 parity=YES
conv state abs/rel  : 9.05991e-06 / 0.00359473 parity=YES
recurrent abs/rel   : 6.33299e-08 / 0.318283 parity=YES
core output abs/rel : 1.16415e-09 / 0.0012068 parity=YES
gated norm abs/rel  : 6.51926e-08 / 0.00120743 parity=YES
FINAL output abs/rel: 2.8871e-08 / 0.000449752 parity=YES
COMPLETE DELTANET   : YES
```

The large maximum relative error reported for the recurrent state is caused by values close to zero; the absolute maximum error is only `6.33299e-08`, and every element passes the field tolerance.

The complete composed GPU diagnostic performs no SSD access during compute. Its currently separate diagnostic Metal stages total about `9.61 ms` of reported compute time (`3.512 + 1.888 + 2.081 + 2.129 ms`); this is a correctness-first unfused implementation and is not a final token/s estimate.

All existing projection, prestate, recurrent-state, gated-tail, and FFN regression checks also passed in the same M4 workflow run.

## Boundary of this checkpoint

This validates the complete recurrent-attention branch for one deterministic single token, including both recurrent cache updates. It does **not** yet validate a complete transformer block because the following graph remains to be composed around it:

- attention residual add
- `attn_post_norm.weight` RMSNorm
- full 2048-wide MoE + shared-expert FFN execution
- FFN residual add

Full-attention layers, embedding/final norm/LM head, tokenizer/sampling, and multi-token generation remain later milestones.
