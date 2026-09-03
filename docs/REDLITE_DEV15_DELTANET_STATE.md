# Red Lite dev15c — Gated DeltaNet recurrent-state parity

## Scope

This checkpoint isolates the single-token recurrent matrix update after the dev15a projection and dev15b prestate stages. It uses deterministic synthetic Q/K/V, beta, decay-gate and previous recurrent state so the state transition is validated independently before composing the full recurrent layer.

Audited Qwen3-Next recurrent geometry:

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

## M4 Pro ARM64 field validation

Commit `572024cd8fba4f963554493c1c0e989b355f168a`, self-hosted runner `redlite-m4pro`:

```text
runtime            : native C + Metal DeltaNet recurrent-state parity (no Python/ctypes)
layer              : 0
state geometry     : S=128 key_heads=16 value_heads=32
Q/K broadcast      : value_head % key_heads (0..15,0..15)
state layout       : transposed M[j][i] = S[i][j]
state payload      : 2.000 MiB
CPU / GPU compute  : 0.404 / 2.147 ms
delta max abs/rel  : 7.45058e-09 / 9.5523e-05 parity=YES
state max abs/rel  : 1.86265e-09 / 0.00111594 parity=YES
output max abs/rel : 4.65661e-10 / 0.00445835 parity=YES
state parity       : YES
```

The larger relative errors occur only near zero; absolute agreement is at `1e-9` to `1e-10` scale.

## Result

**dev15c passed.** State layout, key-head broadcast, decay, delta update, persistent-state write and recurrent output are field-validated on native Apple Silicon.
