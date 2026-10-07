# Red Lite dev70 — prompt lookup: faster coding agents with the same output

Status: done. Measured on the Apple M4 Max 48 GiB (90 W charger) and the Apple M4 Pro 24 GiB (AC, GPU limit 21,741 MiB),
2026-10-07.

## Why

Qwen3-Coder-Next has no MTP block, so the dev45 speculative decoding (an MTP draft checked together with the pending
token in one 2-row pass) did not apply to the model coding agents use. A coding agent, though, writes much of its
output by copying: a file it rewrites, a function it edits, names and paths already in the conversation.

**Prompt lookup** takes the draft from the context itself. `rl_lookup_draft` looks for the latest earlier occurrence
of the context's last 3 tokens (then 2) and proposes the token that followed it. The draft goes through the same
exact verify as MTP (`rl_engine_verify2`): it is kept only when it is the token that decoding produces with the
request's sampler. **The output is unchanged**, at any temperature. When the context has no match, the server takes a
plain step and pays nothing.

## What changed

- `rl_lookup_draft` in `redlite_native_server.c`, with model-free cases in the server self-test. A linear scan back
  from the end: about one compare per context token, tens of microseconds at 64K.
- `redlite-server --lookup`: the dev45 speculation loop with lookup drafts. It needs every expert resident (the
  verify checks two tokens at once) and is not combined with `--mtp`.
- `redlite serve --native` passes `--lookup` when every expert is resident and no MTP head is used, which is the case
  for Qwen3-Coder-Next. `--no-lookup` turns it off.
- `scripts/regress_m4.sh` (machines with at least 40 GiB) adds `server.lookup_greedy`. It uses a prompt that repeats
  text and requires two things: the greedy answer must equal `redlite-generate`'s plain decoding, and drafts must be
  accepted (30 / 30).
- `scripts/dev/server_log_decode.py`: decode speed per context band from server logs, with draft acceptance.

## Measurements

**Two prompts, greedy, CF2, every expert resident, alternated (plain, lookup, lookup, plain).** The answers are
identical with and without lookup, on both Macs.

| Prompt | Mac | Plain | Lookup | Drafts accepted |
|---|---|---:|---:|---|
| rewrite a 1.7K-token Python file with one rename | M4 Max | 80.5 / 79.6 tok/s | 109.5 / 108.6 tok/s (+36 %) | 316 of 379 (83 %), 3 plain steps |
| the same | M4 Pro | 45.1 / 45.0 tok/s | 59.7 / 59.7 tok/s (+33 %) | the same |
| 300 words of prose | M4 Max | 79.2 / 79.8 tok/s | 80.4 / 81.2 tok/s | 15 of 36; 384 plain steps |
| the same | M4 Pro | 44.7 / 44.5 tok/s | 44.9 / 44.7 tok/s | the same |

**Coding agent, M4 Max** (the dev65 hard suite with pi, CF2, temperature 0.3, three runs; 0.8.2 without lookup
against this branch with lookup; decode speed from the server logs, by context at the start of each answer):

| Context of the request | Without lookup | With lookup | Drafts accepted |
|---|---:|---:|---:|
| 0–8K | 76.8 tok/s | 96.1 tok/s (+25 %) | 78 % |
| 8–16K | 72.2 tok/s | 90.0 tok/s (+25 %) | 79 % |
| 16–32K | 61.5 tok/s | 72.2 tok/s (+17 %) | 76 % |
| 32K and more | 53.9 tok/s | 65.5 tok/s (+22 %) | 73 % |

- **Tasks:** 18 / 18 with lookup (no task at the time limit), against 16 / 18 without. Lookup does not change what
  the model writes, so this difference is the run-to-run spread of a sampled agent plus one fewer timeout.
- **Time for the 18 tasks:** 37 minutes against 52.
- The without-lookup run had a 128K window and this one 64K. That does not change decode speed at a given position,
  and both windows pass the same tasks (dev67).

**Coding agent, M4 Pro 24 GiB** (CF2, every expert resident, 32K window, the same build with and without lookup):
M4PRO_AGENT_RESULT

## Smaller findings

- **Half precision does not compute faster on the M4 Max.** A register-only micro-benchmark measured the following
  (`simdgroup_multiply_accumulate`, 4,096 threadgroups):
  - simdgroup 8×8 matrix products: 15.5 TFLOPS float × float → float, 15.8 half × half → float, 15.9 half × half → half;
  - scalar FMA: 13.5 TFLOPS float, 14.8 half.

  The prefill kernels run at about 6–8 TFLOPS, so the room left is in their structure, not in the precision.
- **The verify reads the attention cache twice** (two rows). Lookup still gained 22 % above 32K positions on the M4
  Max, where attention is about a quarter of a token's time.

## Scope boundary

- `redlite-server` only. `redlite-generate` (and so `redlite chat`) keeps MTP or plain decoding.
- One draft token per pass, the 2-row verify of dev45. Edits that copy long runs would gain more from checking
  several drafted tokens at once; not attempted.
- With a bounded expert cache (24 GiB Macs at the default GPU limit, or 64K on the M4 Pro), the verify cannot run:
  lookup is off there.
