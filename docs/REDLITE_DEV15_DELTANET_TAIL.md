# Red Lite dev15d — DeltaNet gated tail + Q4_K output projection

## Scope

This checkpoint validates the final compute tail of a Qwen3-Next recurrent/DeltaNet layer after the recurrent state update:

`recurrent output → gated RMSNorm → flatten 4096 → Q4_K ssm_out → hidden 2048`

It deliberately isolates the two new primitives before end-to-end recurrent-layer composition.

## Pinned graph semantics

Pinned llama.cpp commit `7798007a29a90e3053e799394da48cf53a2f8e0f` performs:

1. reshape recurrent output and Z into 32 value heads × 128;
2. RMSNorm each 128-value head using the shared `ssm_norm.weight` vector;
3. compute `SiLU(z)`;
4. elementwise multiply normalized recurrent output by `SiLU(z)`;
5. flatten to 4096;
6. apply `ssm_out` to produce hidden size 2048.

Real layer-0 GGUF tensors:

- `blk.0.ssm_norm.weight`: F32, shape `(128)`;
- `blk.0.ssm_out.weight`: Q4_K type 12, shape `(4096,2048)`.

## Q4_K implementation

The independent native decoder follows the pinned canonical layout:

- 256 values per block;
- 144 bytes per block;
- F16 `d` + F16 `dmin`;
- 12 packed 6-bit scale/min bytes;
- 128 bytes of packed 4-bit quants.

A 4096-value row therefore uses 16 blocks = 2304 bytes. The complete `4096 × 2048` `ssm_out` payload is 4.5 MiB.

CPU and Metal decode the packed payload independently. The Metal path operates directly on Q4_K bytes; it does not use CPU-dequantized weights.

## Native files

- `native/redlite_native_deltanet_tail.h`
- `native/redlite_native_deltanet_tail_cli.c`
- `native/redmetal_deltanet_tail.m`
- `scripts/build_deltanet_tail.sh`
- executable `.deps/redmetal/redlite-deltanet-tail`

## Field gate

```bash
.deps/redmetal/redlite-deltanet-tail parity MODEL --layer 0
```

The checkpoint passes only if:

- all 4096 gated-normalized values agree CPU ↔ Metal;
- all 2048 Q4_K projected outputs agree CPU ↔ Metal;
- no SSD reads occur during compute;
- `tail parity : YES`.

After this passes, all mathematical/quantized primitives of the recurrent attention branch are independently validated and can be composed into a complete real DeltaNet layer.
