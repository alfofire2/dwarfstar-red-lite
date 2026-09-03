# Red Lite dev13 — real router field validation

Target: Apple M4 Pro, 24 GiB unified memory

Model: `Qwen_Qwen3-Next-80B-A3B-Instruct-IQ2_XXS.gguf`

This checkpoint validates the real Qwen3-Next F32 router and the router-selected routed-expert branch entirely through the standalone native C/Objective-C/Metal path. Python and `ctypes` are absent from the tested execution path.

## Router audit

All 48 routed layers contain exactly one `blk.<layer>.ffn_gate_inp.weight` tensor.

- tensor count: 48
- shape: `(2048, 512)` for every layer
- GGML type: F32 (`0`) for all 48 tensors
- physical span: 4.000 MiB per layer
- total router payload: 192.000 MiB

## Layer 0 router parity

- CPU router read: 1.311 ms on the first isolated run
- CPU router matvec: 0.841 ms
- GPU router read: 0.463 ms
- GPU router matvec: 3.004 ms
- max logit absolute error: `4.76837e-07`
- max logit relative error: `2.63783e-04`
- top-k IDs match: YES
- max selected-weight absolute error: `8.9407e-08`
- router parity: YES

Selected experts:

`438,41,204,493,139,451,412,200,367,154`

Normalized weights:

`0.2299825,0.1769590,0.1074829,0.0931625,0.0795658,0.0691245,0.0675194,0.0613993,0.0578358,0.0569682`

## Layer 6 router parity

- CPU router read: 1.467 ms on the first isolated run
- CPU router matvec: 0.846 ms
- GPU router read: 0.434 ms
- GPU router matvec: 3.190 ms
- max logit absolute error: `4.76837e-07`
- max logit relative error: `1.56213e-05`
- top-k IDs match: YES
- max selected-weight absolute error: `2.98023e-08`
- router parity: YES

Selected experts:

`480,288,135,505,143,344,485,182,111,83`

Normalized weights:

`0.1826100,0.1416743,0.1046222,0.1015359,0.0955648,0.0899121,0.0741731,0.0731347,0.0686038,0.0681691`

## Real-router routed-expert parity

### Layer 0 / IQ2_XS

- real router parity: YES
- expert loads: 10
- cache hits/misses: 0/10
- positional expert reads: 30
- SSD I/O during top-k compute: 0 bytes / 0 calls
- GPU routed top-k: 9.364 ms
- CPU routed reference: 22.736 ms
- routed max absolute error: `2.20497e-08`
- routed max relative error: `4.09202e-06`
- routed parity: YES

### Layer 6 / IQ1_M

- real router parity: YES
- expert loads: 10
- cache hits/misses: 0/10
- positional expert reads: 30
- SSD I/O during top-k compute: 0 bytes / 0 calls
- GPU routed top-k: 8.721 ms
- CPU routed reference: 23.397 ms
- routed max absolute error: `1.19063e-08`
- routed max relative error: `2.62410e-06`
- routed parity: YES

## What this proves

The following path is field validated on the real model:

`F32 router GGUF -> native CPU/Metal router matvec -> softmax(512) -> exact top-10 -> selected-weight renormalization -> transactional native LRU -> positional expert reads -> resident IQ2_XS/IQ1_M Metal FFNs -> GPU weighted accumulation`

The native CPU oracle independently agrees with both the router selection/weights and routed output.

## What it does not prove yet

Qwen3-Next also computes a shared expert and multiplies it by its own sigmoided scalar gate before adding it to the routed MoE output. That branch is intentionally excluded from dev13 routed parity. Dev14 begins by auditing the shared-expert tensors and then will validate the complete FFN output.
