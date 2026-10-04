# Red Lite dev56 — two server requests at once

Status: on branch `dev56/parallel-server`. Measured on the Apple M4 Pro 24 GiB and the Apple M4 Max 48 GiB,
2026-10-04.

## Why

Decode reads every active weight once per token. Until now `redlite-server` served one request at a time. A second
request waited for the first one to finish, even though one pass over the weights could produce a token for each.

## How

- **Engine slots.**
  - `rl_engine_slots_enable` allocates a second sequence state: DeltaNet and convolution states, KV caches
    (the MTP block's too), position.
  - `rl_engine_select_slot` swaps it in by pointer, so every existing call (prefill, step, verify, reset, state
    files) works on the selected slot unchanged.
- **Paired step.**
  - `rl_engine_step_pair` decodes one token of each slot in one command buffer.
  - It is MTP's 2-row verify (dev45) with row 1 bound to the other slot's state and position instead of the next
    position of the same sequence.
  - It needs every expert resident, like the verify.
- **Server.**
  - `--parallel 2` runs two worker threads, each with its own slot. The engine runs one call at a time.
  - A decode step posts its token. When the other request is decoding too, it waits up to 20 ms for that one's
    token, and both run as one paired step.
  - **Prefix reuse:** a new request takes the free slot whose conversation it extends.
  - **MTP** is used only while the other slot is idle. When a second request arrives, the first one leaves
    speculation and pairs.
- **CLI.** `redlite serve --native --parallel 2` passes the flag. The GPU plan (dev55) counts the second slot as
  1,536 positions (its 72 MiB DeltaNet state) plus its context.

## Correctness

- **`redlite-engine pair`:** sequence A and sequence B (A reversed, two positions ahead) decoded together, against
  each alone. Every logit row matches.

  | machine | file | rows | max abs logit diff | argmax mismatches |
  |---|---|---:|---:|---:|
  | M4 Max | IQ2_XXS | 80 | 1.38e-5 | 0 |
  | M4 Pro | E3 | 32 | 1.24e-5 | 0 |

- **Server:**
  - two greedy 200-token answers asked at the same time equal the same answers asked one after the other, with
    and without MTP, on both Macs;
  - `server_check.py --parallel` checks this, and that the steps were paired.
- **Regression:** both are checks in `regress_m4.sh` (`engine.pair`, `server.parallel_greedy`; >= 40 GiB machines,
  as for the other full-residency checks).

## Speed

Two greedy requests of 200 tokens, E3 file, every expert resident (one run each; not a benchmark):

| M4 Pro 24 GiB, GPU limit 21,741 MiB | one after the other | at the same time | gain |
|---|---:|---:|---:|
| plain decode, context 4096 | 43.5 tok/s | **55.9 tok/s** | +28 % |
| with MTP, context 4096, 512-token chunks | 51.0 tok/s | **56.0 tok/s** | +10 % |

- **Per request:** each of the two answers runs at about 29 tok/s instead of 46. The gain is in total throughput,
  not per answer.
- **Cost of a pair** (`redlite-engine pair`): 1.63× a single step on the M4 Pro, 1.45× on the M4 Max (33.6 vs
  20.6 ms; 21.9 vs 15.1 ms).
- **M4 Max**, IQ2_XXS, no MTP: 58.4 → 76.8 tok/s (+32 %). A quantization was using the CPU during that run.
- **Memory:** the planner's choice for `--parallel 2` on the M4 Pro (512-token chunks with MTP) ran without
  running out of GPU memory.

## Two things that went wrong first

- **Both requests stayed on MTP and alternated: slower than serving them in turn** (M4 Max: 66 vs 71 tok/s). A
  request on MTP did not count as decoding, so neither switched. Now a request leaves MTP as soon as the other
  slot is busy.
- **No step was paired on the M4 Pro.** The 20 ms wait started when the token was posted, while the other request
  held the GPU for a 20 ms step, so it had always expired. Now it starts when the engine is free.

## Scope boundary

- **Slots:** two, not N. Both slots share one steering strength (the strength of the slot that runs the pair).
- **Full residency only.** With a bounded expert cache `--parallel` is ignored: each row would need its own expert
  loads.
- **Prompt ingestion is not interleaved.** While one request ingests a long prompt, the other one's decode waits.
- **A slow client costs its partner** up to 20 ms per token (the wait for its step).
- The speed numbers are single runs of two requests. No load test with many clients was made.
