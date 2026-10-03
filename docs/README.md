# Documentation index

Every milestone document states what was implemented, how it was validated (machine,
commit, command) and a **Scope boundary** of what it does not claim. "Validated" always
names the machine: the M4 Pro 24 GiB (field target) or the M4 Max 48 GiB (development
machine). "Tested synthetically" means model-free tests only.

## Start here

| Document | What it covers |
|---|---|
| [FINDINGS.md](FINDINGS.md) | **Read first.** What we learned, with charts: where a token's time goes, speed by release vs llama.cpp, exact MTP speculation, 24 GiB cache options, long context, correctness, what did not work |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Design goal and architecture of the v0.2 launcher (sparse 80B on a 24 GiB Mac) |
| [MEMORY.md](MEMORY.md) | Memory policy and headroom formula for 24 GiB Macs |
| [BENCHMARK.md](BENCHMARK.md) | v0.2 benchmark protocol and context-depth sweep |
| [../README.md](../README.md) | Release 0.4.1: the two runtimes, the two supported GGUFs, measured performance per model and cache, correctness, limits, tool status |
| [../CHANGELOG.md](../CHANGELOG.md) | 0.4.1, 0.4.0 and 0.3.0 release summaries with milestone targets, then every dev milestone |
| [REDLITE_DEV18_ENGINE.md](REDLITE_DEV18_ENGINE.md) | The native end-to-end engine: design, validation, limits |
| [FIELD_VALIDATION_M4PRO_24GB.md](FIELD_VALIDATION_M4PRO_24GB.md) | v0.2 launcher field validation on the M4 Pro 24 GiB |
| [RESEARCH_2026_10.md](RESEARCH_2026_10.md) | October 2026 survey: current DwarfStar, Qwen3-Coder-Next, Qwen3-Next MTP GGUFs, llama.cpp/MLX/Metal techniques, ranked options |
| [REDLITE_DEV48_API_REFERENCE.md](REDLITE_DEV48_API_REFERENCE.md) | dev48: greedy answers compared word for word with Qwen's own API (IQ2_XXS vs IQ3_XXS) |
| [REDLITE_DEV49_PREFILL_24GB.md](REDLITE_DEV49_PREFILL_24GB.md) | dev49: why the M4 Pro reads prompts at ~350 tok/s (GPU compute, not the disk), expert decode share, a bit-identical decode speed-up |
| [WHAT_DID_NOT_WORK.md](WHAT_DID_NOT_WORK.md) | Every reverted attempt, correctness trap and unreached target, with the measurement that decided it |

## Native runtime milestones of 0.4.0

Developed on the `dev/0.4` branch; M4 Max 48 GiB only.

| Milestone | Document |
|---|---|
| dev28 Linux CI, release tarball, same-prompt llama.cpp baseline | [REDLITE_DEV28_CI_RELEASE.md](REDLITE_DEV28_CI_RELEASE.md) |
| dev29 server: state reuse, stop sequences, FIFO queue | [REDLITE_DEV29_SERVER.md](REDLITE_DEV29_SERVER.md) |
| dev30 batched prefill: tiled attention, matrix experts, faster dense pass | [REDLITE_DEV30_PREFILL.md](REDLITE_DEV30_PREFILL.md) |
| dev31 IQ3_XXS GGUF: new quant types, full residency on 48 GiB | [REDLITE_DEV31_IQ3.md](REDLITE_DEV31_IQ3.md) |
| dev32 release 0.4.0 | [REDLITE_DEV32_RELEASE.md](REDLITE_DEV32_RELEASE.md) |
| dev33 faster IQ kernels (after 0.4.0) | [REDLITE_DEV33_IQ_KERNELS.md](REDLITE_DEV33_IQ_KERNELS.md) |
| dev34 bounded-cache prefill: fewer expert reloads | [REDLITE_DEV34_BOUNDED_PREFILL.md](REDLITE_DEV34_BOUNDED_PREFILL.md) |
| dev35 long-context decode: grouped split-K attention | [REDLITE_DEV35_LONG_DECODE.md](REDLITE_DEV35_LONG_DECODE.md) |
| [REDLITE_DEV36_IQ3M.md](REDLITE_DEV36_IQ3M.md) | dev36: IQ3_M GGUF supported (Q4_K experts), measured, not chosen on 48 GiB |
| [REDLITE_DEV37_EXPERT_SLOTS.md](REDLITE_DEV37_EXPERT_SLOTS.md) | dev37: expert slots of each layer's own size (4 GiB cache −24 % misses; full residency −3.9 GiB) |
| [REDLITE_DEV38_DECODE_DISPATCH.md](REDLITE_DEV38_DECODE_DISPATCH.md) | dev38: expert down lanes and concurrent decode encoders (IQ3_XXS 70.4 → 79.1 tok/s) |
| [REDLITE_DEV39_EXPERT_TAIL.md](REDLITE_DEV39_EXPERT_TAIL.md) | dev39: fused expert tail in GPU-routed decode (+1.6 %) |
| [REDLITE_DEV42_CODER_NEXT.md](REDLITE_DEV42_CODER_NEXT.md) | dev42: Qwen3-Coder-Next (same engine, validated against llama.cpp) |
| [REDLITE_DEV43_STATE_CACHE.md](REDLITE_DEV43_STATE_CACHE.md) | dev43: session state checkpoints on disk (`--state-dir`), dev37 2 GiB prefill fix |
| [REDLITE_DEV45_MTP.md](REDLITE_DEV45_MTP.md) | dev45: speculative decoding with the Qwen3-Next MTP block (+8-30 % decode, identical output) |
| [REDLITE_DEV46_SMALL_MAC_OPTIONS.md](REDLITE_DEV46_SMALL_MAC_OPTIONS.md) | dev46: cache-aware routing bias, F_NOCACHE, engine perplexity, 24 GiB A/B script |
| [REDLITE_DEV47_LONG_CONTEXT.md](REDLITE_DEV47_LONG_CONTEXT.md) | dev47: M4 Pro 24 GiB field session, long context (16K parity, 33.5K speed), dev44 decision, REDLITE_SDK |

## Native runtime milestones (v0.3, released as 0.3.0)

Developed on the `v0.3-streaming` branch, which was merged into `main` with release 0.3.0.

dev9–dev17 built and validated one graph stage each. Their parity CLIs are now regression
tools, and `redlite-engine` (dev18 onward) supersedes them for inference. dev22–dev26 are
the 0.3.0 performance and robustness work.

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
| dev26 robustness: GGUF fuzz, sanitizers, sampler parity | [REDLITE_DEV26_ROBUSTNESS.md](REDLITE_DEV26_ROBUSTNESS.md) |
| dev26 completion: 4096/8192 positions, sanitized chat turn, kernel self-test, split-K attention | [REDLITE_DEV26_LONG_CONTEXT.md](REDLITE_DEV26_LONG_CONTEXT.md) |
| dev27 release 0.3.0: milestone results, final validation | [REDLITE_DEV27_RELEASE.md](REDLITE_DEV27_RELEASE.md) |

## Background and plans

| Document | What it covers |
|---|---|
| [STREAMING_V03.md](STREAMING_V03.md), [V0.3_STREAMING_DEV.md](V0.3_STREAMING_DEV.md) | v0.3 streaming design direction and dev1–dev8 notes (**legacy**: the Python streaming oracle `redlite-stream` / `redlite-ffn` / `redlite-topk` is frozen as a numerical reference) |
| [METAL_STREAMING_ROADMAP.md](METAL_STREAMING_ROADMAP.md) | Native Metal + SSD expert streaming roadmap |
| [DS4_ADAPTATION.md](DS4_ADAPTATION.md) | Mapping from DwarfStar / DS4 ideas to Red Lite |
