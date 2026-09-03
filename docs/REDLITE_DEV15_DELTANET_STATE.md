# Red Lite dev15c — Gated DeltaNet recurrent-state parity

## Scope

This checkpoint isolates the single-token recurrent matrix update after the dev15a projection and dev15b prestate stages. It intentionally uses deterministic synthetic Q/K/V, beta, decay-gate and previous recurrent state so the state transition can be validated independently before composing the full recurrent layer.

The field target is the audited Qwen3-Next recurrent geometry:

- state/key dimension: 128
- key/query heads: 16
- value/recurrent heads: 32
- recurrent state: `32 × 128 × 128` F32 = 2 MiB

## Pinned llama.cpp semantics

The implementation follows llama.cpp commit `7798007a29a90e3053e799394da48cf53a2f8e0f`.

For each value head `h`, Q/K are broadcast from key head `h % 16`. The recurrent state uses the transposed physical layout documented by the pinned CPU operator:

`M[j][i] = S[i][j]`

For one token:

1. `M *= exp(gate[h])`
2. `sk[j] = dot(M[j], k)`
3. `delta[j] = (v[j] - sk[j]) * beta[h]`
4. `M[j][i] += delta[j] * k[i]`
5. `out[j] = dot(M[j], q) / sqrt(128)`

The output uses the updated state.

## Native implementation

- `native/redlite_native_deltanet_state.h`
- `native/redlite_native_deltanet_state_cli.c`
- `native/redmetal_deltanet_state.m`
- `scripts/build_deltanet_state.sh`
- executable: `.deps/redmetal/redlite-deltanet-state`

The Metal reference is deliberately split into four ordered dispatches: decay, delta, state update and output. This is a correctness checkpoint, not the final fused performance kernel.

## Field command

```bash
.deps/redmetal/redlite-deltanet-state parity \
  models/Qwen_Qwen3-Next-80B-A3B-Instruct-IQ2_XXS.gguf \
  --layer 0
```

The GitHub self-hosted Apple-Silicon workflow runs this command automatically and records `deltanet-state-layer0.log` plus the diagnostics JSON.

## Gate

The checkpoint passes only when CPU and Metal agree independently on:

- delta vector
- complete 2 MiB next recurrent state
- recurrent output vector

A passing run prints `state parity : YES`.
