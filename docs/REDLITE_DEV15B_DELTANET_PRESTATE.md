# Red Lite dev15b — DeltaNet pre-state parity

Dev15a field-validated the real-model input RMSNorm and optimized QKV/Z/BA projections. Dev15b advances into the stateful Gated DeltaNet path while keeping the recurrent matrix-state update out of scope for one more checkpoint.

## Pinned semantics

For Qwen3-Next recurrent layers, the pinned llama.cpp graph:

1. splits the 64-value beta/alpha projection according to `ssm.group_count` and `ssm.time_step_rank`;
2. computes `beta = sigmoid(b)`;
3. computes `alpha_softplus = softplus(alpha + ssm_dt)`;
4. computes the decay gate `g = alpha_softplus * ssm_a`;
5. reshapes the persistent conv cache to `(d_conv - 1, conv_channels)`;
6. transposes the current QKV projection and concatenates it after the previous conv state;
7. applies real F32 `ssm_conv1d` over the 4-sample window per channel;
8. applies SiLU;
9. splits the result into Q, K, V;
10. applies per-head L2 normalization to Q and K using `1 / max(sqrt(sum(x^2)), eps)`;
11. updates the convolutional cache to the last `d_conv - 1` samples.

The target GGUF metadata is read directly using the canonical keys:

- `qwen3next.ssm.conv_kernel`;
- `qwen3next.ssm.inner_size`;
- `qwen3next.ssm.state_size`;
- `qwen3next.ssm.time_step_rank`;
- `qwen3next.ssm.group_count`;
- `qwen3next.attention.layer_norm_rms_epsilon`.

For the target model these should resolve to the audited layout:

- `d_conv=4`;
- `d_inner=4096`;
- `d_state=128`;
- `dt_rank=32` value heads;
- `n_group=16` key groups;
- conv channels = `4096 + 2*16*128 = 8192`;
- Q = 16×128 = 2048 values;
- K = 16×128 = 2048 values;
- V = 32×128 = 4096 values.

## Isolation strategy

`redlite-deltanet-prestate` intentionally uses deterministic synthetic QKV/BA projection outputs and a deterministic previous conv state, while loading the real layer's F32 `ssm_conv1d`, `ssm_dt`, and `ssm_a` tensors from the GGUF. This separates recurrent-state semantics from the quantized projection kernels already field-validated in dev15a.

The native CPU oracle and Metal path independently calculate:

- all 32 beta values;
- all 32 decay-gate values;
- all 8192 conv+SiLU outputs;
- all 2048 normalized Q values;
- all 2048 normalized K values;
- all 4096 V values;
- the full 3×8192 next convolutional state.

All real tensor reads complete before the Metal command buffer is submitted, so the test also reports zero SSD reads during compute.

## Build and field test

`make native` builds:

`.deps/redmetal/redlite-deltanet-prestate`

Field command:

```bash
.deps/redmetal/redlite-deltanet-prestate parity \
  models/Qwen_Qwen3-Next-80B-A3B-Instruct-IQ2_XXS.gguf \
  --layer 0
```

Expected gate:

- beta parity: YES;
- gate parity: YES;
- conv+SiLU parity: YES;
- Q L2 parity: YES;
- K L2 parity: YES;
- V split parity: YES;
- conv state shift parity: YES;
- prestate parity: YES;
- SSD during compute: 0 bytes / 0 calls.

## Next boundary

After dev15b passes on the real model, dev15c can introduce the actual recurrent matrix state `S[128,128,32]` and validate the single-token autoregressive Gated DeltaNet update from the pinned graph:

`S <- exp(g) * S; d <- beta * (v - S*k); S <- S + k*d^T; out <- S*q/sqrt(128)`.

Only after that state update is field-validated should Red Lite add `ssm_norm` gated normalization and the real Q4_K `ssm_out` projection.
