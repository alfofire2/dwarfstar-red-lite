# Documentation index

Every milestone document states what was implemented, how it was validated (machine,
commit, command) and a **Scope boundary** of what it does not claim. "Validated" always
names the machine: the M4 Pro 24 GiB (field target) or the M4 Max 48 GiB (development
machine). "Tested synthetically" means model-free tests only.

## Start here

| Document | What it covers |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Design goal and architecture of the v0.2 launcher (sparse 80B on a 24 GiB Mac) |
| [MEMORY.md](MEMORY.md) | Memory policy and headroom formula for 24 GiB Macs |
| [BENCHMARK.md](BENCHMARK.md) | v0.2 benchmark protocol and context-depth sweep |
| [REDLITE_DEV18_ENGINE.md](REDLITE_DEV18_ENGINE.md) | The native end-to-end engine: design, validation, limits |
| [FIELD_VALIDATION_M4PRO_24GB.md](FIELD_VALIDATION_M4PRO_24GB.md) | v0.2 launcher field validation on the M4 Pro 24 GiB |

## Native runtime milestones (v0.3, `v0.3-streaming`)

| Milestone | Document |
|---|---|
| dev7–dev8 Red Metal resident experts | [REDMETAL_DEV7.md](REDMETAL_DEV7.md), [REDMETAL_DEV8.md](REDMETAL_DEV8.md) |
| dev9 native foundation | [REDLITE_DEV9_NATIVE.md](REDLITE_DEV9_NATIVE.md) |
| dev10 standalone Metal path | [REDLITE_DEV10_NATIVE_METAL.md](REDLITE_DEV10_NATIVE_METAL.md) |
| dev11 CPU/GPU parity harness | [REDLITE_DEV11_NATIVE_PARITY.md](REDLITE_DEV11_NATIVE_PARITY.md) |
| dev12 field validation, offline hardening | [REDLITE_DEV12_FIELD_VALIDATION.md](REDLITE_DEV12_FIELD_VALIDATION.md), [REDLITE_DEV12_OFFLINE_HARDENING.md](REDLITE_DEV12_OFFLINE_HARDENING.md) |
| dev13 router | [REDLITE_DEV13_ROUTER_AUDIT.md](REDLITE_DEV13_ROUTER_AUDIT.md), [REDLITE_DEV13_FIELD_VALIDATION.md](REDLITE_DEV13_FIELD_VALIDATION.md) |
| dev14 shared expert and complete FFN | [REDLITE_DEV14_SHARED_AUDIT.md](REDLITE_DEV14_SHARED_AUDIT.md), [REDLITE_DEV14_FIELD_VALIDATION.md](REDLITE_DEV14_FIELD_VALIDATION.md), [REDLITE_DEV14_COMPLETE_FFN_FIELD_VALIDATION.md](REDLITE_DEV14_COMPLETE_FFN_FIELD_VALIDATION.md) |
| dev15 Gated DeltaNet | [graph audit](REDLITE_DEV15_LAYER_GRAPH_AUDIT.md), [projection](REDLITE_DEV15_DELTANET_PROJECTION.md), [projection field](REDLITE_DEV15A_DELTANET_PROJECTION_FIELD_VALIDATION.md), [pre-state](REDLITE_DEV15B_DELTANET_PRESTATE.md), [state](REDLITE_DEV15_DELTANET_STATE.md), [tail](REDLITE_DEV15_DELTANET_TAIL.md), [layer](REDLITE_DEV15_LAYER_COMPOSITION.md) |
| dev16 full attention | [REDLITE_DEV16_FULL_ATTENTION.md](REDLITE_DEV16_FULL_ATTENTION.md) |
| dev17 48-layer decoder stack | [REDLITE_DEV17_DECODER_STACK.md](REDLITE_DEV17_DECODER_STACK.md) |
| dev18 end-to-end engine | [REDLITE_DEV18_ENGINE.md](REDLITE_DEV18_ENGINE.md) |
| dev19 `redlite chat` | [REDLITE_DEV19_CLI_CHAT.md](REDLITE_DEV19_CLI_CHAT.md) |
| dev20 batched prefill | [REDLITE_DEV20_BATCHED_PREFILL.md](REDLITE_DEV20_BATCHED_PREFILL.md) |
| dev21 GPU-routed decode | [REDLITE_DEV21_GPU_ROUTED_DECODE.md](REDLITE_DEV21_GPU_ROUTED_DECODE.md) |
| dev22 decode kernels (one encoder per token, sub-block GEMV) | [REDLITE_DEV22_DECODE_KERNELS.md](REDLITE_DEV22_DECODE_KERNELS.md) |
| dev23 per-layer early-out, pre-gated decode prefetch | [REDLITE_DEV23_EARLY_OUT_PREFETCH.md](REDLITE_DEV23_EARLY_OUT_PREFETCH.md) |
| dev24 prefill: expert prefetch overlap, parallel routing | [REDLITE_DEV24_PREFILL_OVERLAP.md](REDLITE_DEV24_PREFILL_OVERLAP.md) |
| dev25 (partial) product surface: chat defaults, native server | [REDLITE_DEV25_PRODUCT.md](REDLITE_DEV25_PRODUCT.md) |
| dev26 (partial) robustness: GGUF fuzz, sanitizers, sampler parity | [REDLITE_DEV26_ROBUSTNESS.md](REDLITE_DEV26_ROBUSTNESS.md) |

## Background and plans

| Document | What it covers |
|---|---|
| [STREAMING_V03.md](STREAMING_V03.md), [V0.3_STREAMING_DEV.md](V0.3_STREAMING_DEV.md) | v0.3 streaming design direction and dev1–dev8 notes |
| [METAL_STREAMING_ROADMAP.md](METAL_STREAMING_ROADMAP.md) | Native Metal + SSD expert streaming roadmap |
| [DS4_ADAPTATION.md](DS4_ADAPTATION.md) | Mapping from DwarfStar / DS4 ideas to Red Lite |
