# Red Lite dev17 — audited 48-layer decoder stack

Target: Apple M4 Pro, 24 GiB unified memory, Bartowski
`Qwen_Qwen3-Next-80B-A3B-Instruct-IQ2_XXS.gguf`.

Dev17 begins after independent CPU/Metal parity passed for both complete decoder
block variants. This checkpoint extracts those validators into reusable native C
entry points and propagates separate CPU and Metal hidden vectors through the
complete transformer trunk.

## Structural dispatch

`redlite-decoder-stack` does not infer the hybrid pattern from a fixed interval.
It scans the real GGUF tensor directory and requires:

- a contiguous layer sequence;
- both attention norm tensors on every layer;
- exactly one recurrent or full-attention structural signature per layer;
- no mixed or unknown layers.

The target model reports 48 layers: 36 recurrent DeltaNet blocks and 12
full-attention blocks at layers `3,7,11,15,19,23,27,31,35,39,43,47`.

## Execution boundary

For one deterministic hidden vector, the diagnostic executes each audited block
in order:

1. recurrent DeltaNet or GQA full attention;
2. attention residual;
3. post-attention RMSNorm;
4. full-width top-10 routed MoE plus shared expert;
5. final residual;
6. pass the complete 2048-value output to the next layer.

CPU and Metal outputs are propagated independently. Isolated block CLIs retain
their strict same-input substage checks; stack execution evaluates cumulative
block parity so normal upstream floating-point error is not misreported as a
local RMSNorm failure. A top-k router ID divergence remains a hard failure.

## Field validation

```bash
make native

.deps/redmetal/redlite-decoder-stack parity \
  models/Qwen_Qwen3-Next-80B-A3B-Instruct-IQ2_XXS.gguf \
  --position 7 \
  --top-k 10 \
  --cache-mib 256
```

Observed on the target M4 Pro:

```text
GGUF layer map       : 48 layers / recurrent=36 full-attention=12
dispatch totals      : recurrent=36 full-attention=12
stack max abs/rel    : 0.000238419 / 0.133136
elapsed              : 9049.723 ms
COMPLETE 48-LAYER DECODER STACK: YES
```

The large maximum relative error is attached to values near zero; the maximum
absolute difference over every 2048-wide layer output is `2.38419e-04`.

## Scope boundary

This is a correctness-first, single-token transformer-trunk diagnostic. Previous
recurrent/KV states are deterministic parity fixtures, and model structures are
reopened per block. It is not yet a persistent multi-token runtime and the
reported elapsed time is not a throughput benchmark.

The next correctness gates are input embedding, final RMSNorm and LM-head logits,
followed by tokenizer/sampling and persistent multi-token state management.
