# Red Lite dev77 — where coding-agent sessions spend their prefill, and a rewind for re-rendered answers

Status: done. Apple M4 Max 48 GiB, plus checks on the Apple M4 Pro 24 GiB (AC, kept awake), 2026-10-08.

## 1. Agent sessions spend more time on prefill than on decode

The server logs of the dev74 hard-suite runs (pi, CF2, every expert resident, 64K window, prompt lookup, 0.9.3
kernels) were split by the new prompt tokens of each request:

| New tokens per request | Requests | Prefill time | Mean |
|---|---:|---:|---:|
| 16–256 | 109 | 46 s | 0.3–0.5 s |
| 256–4,096 | 123 | 199 s | 1–2.5 s |
| more than 4,096 | 22 | 527 s | 24 s |
| all | 255 | **773 s** | decode: 648 s |

Short chunks are a small share. A prefill of 8 tokens costs 142 ms against 74 ms for 8 decode steps, because every
dense weight is dequantized to float for the chunk. Reading the quantized weights directly halves it at 8 tokens but
loses from 32 tokens on (not kept). Most of the time goes to a few requests that recompute the whole conversation.

## 2. Why the server recomputed whole conversations

`RL_SERVER_DEBUG_REUSE=1` prints where a new prompt stops matching the ids the state holds. One agent run gave three
kinds of restart:

| Divergence | Count | Recomputed | Cause |
|---|---:|---:|---|
| at token ~597 | 5 | 8,376 tokens | a new task: pi's system prompt contains the task's temporary directory |
| at token 5 | 5 | 78,983 tokens | pi compacts the conversation near 49K tokens: a summarizer request with its own system prompt, then a new, shorter conversation |
| inside the last answer | 1 (2–5 in other runs) | 18,364 tokens | pi sends the model's answer back re-rendered (a tool call re-serialized), so the prompt matches the state only up to the end of the previous prompt |

Before dev77 the server reused the state only when the new prompt extended exactly the ids it held: a single
re-rendered token near the end of a 40K-token conversation meant prefilling all 40K again (30–75 s on the M4 Max).

## 3. The rewind

- **Engine:**
  - `rl_engine_mark` keeps a host copy of the DeltaNet states (conv + recurrent, about 75 MB) at the current
    position;
  - `rl_engine_rewind` restores it and moves the position back;
  - KV rows below the mark are not touched by later tokens, so the rewind is exact.
- **Server:**
  - after every prompt prefill the server marks the state;
  - when the next prompt extends that prompt but not the answer the state holds, the server rewinds to the mark and
    prefills only the rest;
  - single-slot servers only (`--parallel 1`, the default).
- **Check:** `scripts/dev/server_rewind_check.py` (regression check `server.rewind_greedy`).
  - Request 1 gets a greedy answer.
  - Request 2 sends that answer with one character changed in its middle, plus a new question.
  - The server must log "back to the end of the previous prompt" and answer exactly as a freshly started server
    does.
  - It passes on the M4 Max and on the M4 Pro.
- **Agent runs** (M4 Max, the same suite and settings, two runs each, sessions differ from run to run):

  | | Before (0.9.3) | With the rewind |
  |---|---:|---:|
  | Rewinds | — | 1 and 2 (6.9K, 30.1K, 28.3K tokens kept) |
  | Full recomputes of more than 4K tokens | 5 (152K tokens, 268 s) | 2 (30K tokens, 49 s: pi's compactions) |
  | Prefill time per run | 390 and 382 s | 218 and 346 s |
  | Tasks passed | 12 / 12 | 11 / 12 |

Compaction restarts cannot be reused: the new conversation holds a summary and the recent turns at new positions.
Section 5 tests a larger window, with which pi compacts later.

**On the 24 GiB M4 Pro** (0.9.3 against 0.9.4 servers, 4 GiB expert cache, 64K window, pi on the M4 Max, four
alternated runs, 2026-10-08/09):

| Run | Rewinds (tokens kept) | Full recomputes over 4K | Prefill rate | Decode | Tasks |
|---|---|---:|---:|---:|---:|
| 0.9.3, run 1 | — | 1 (21K tokens, 63 s) | 233 tok/s | 24.8 tok/s | 6 / 6 |
| 0.9.4, run 1 | 1 (33.0K) | 0 | 161 tok/s | 19.2 tok/s | 5 / 6 |
| 0.9.3, run 2 | — | 0 | 146 tok/s | 17.6 tok/s | 5 / 6 |
| 0.9.4, run 2 | 2 (5.4K, 21.6K) | 0 | 139 tok/s | 16.9 tok/s | 5 / 6 |

- The rewinds kept 60K tokens that would have been prefilled again: 4–7 minutes at this machine's 140–230 tok/s.
- The session totals cannot be compared. Over the three hours, prefill and decode slowed by a third whatever the
  release; the first run was the fastest. The laptop throttled, with no thermal warning, as the M4 Max did in dev76.
- The three failures are the same task (`two_bugs`, at the 900 s limit). It does not loop; it is long:
  - the run 2 of 0.9.4 prefilled 42K new tokens (320 s) and generated 8.8K tokens at 15.6 tok/s (566 s), at
    contexts of 20–45K;
  - on the M4 Max the same task takes 230–600 s, and fails its check in half of the CF2 runs there too (6 of 12);
  - agent runs on a 24 GiB Mac with the 4 GiB cache need `agent_eval.py --timeout 1800`.

## 4. Red Lite 0.9.3 on the 24 GiB M4 Pro

`decode-bench` and `redlite-generate`, CF2, alternated with 0.9.1, AC power, GPU limit 21,741 MiB:

| | 0.9.1 | 0.9.3 | Gain |
|---|---:|---:|---:|
| Decode, 4 GiB expert cache, short / 16K | 32.0 / 28.7 tok/s | 35.7 / 31.9 tok/s | +11–12 % |
| Decode, every expert resident, short / 16K | 48.2 / 42.6 tok/s | 59.1 / 50.9 tok/s | +19–23 % |
| Prompt ingestion 1.4K tokens, 4 GiB / every expert resident | 354 / 403 tok/s | 377 / 447 tok/s | +6–11 % |
| Bartowski IQ2_XXS, six prompts, 4 GiB / every expert resident | 33.9 / 46.3 tok/s | 37.6 / 53.3 tok/s | +11–15 % |

A larger expert cache from the memory `--kv f16` frees was not tried: dev51 measured that 4 → 14 GiB of cache halves
the misses but leaves decode at 31–32 tok/s. The bounded path waits for the GPU once per layer.

## 5. A 128K window

The same suite with a 131,072-token window (the server's context and pi's `contextWindow`), two runs:

| | 64K window (with the rewind) | 128K window |
|---|---:|---:|
| Compactions | 0 and 1 | 0 and 0 |
| Largest context | 35K / 49K | 82K / 46K |
| Prefill time | 218 / 346 s | 413 / 286 s |
| Decode | 80 / 67 tok/s | 66 / 69 tok/s |
| Tasks passed | 6 / 6, 5 / 6 | 4 / 6, 5 / 6 |

pi stops compacting, but the sessions grow longer: one run looped for 406 requests in `debug_session`, and decode is
slower at larger contexts. Two runs do not show a gain, so `redlite setup-pi` keeps its 64K window.

## Scope boundary

- Agent numbers come from one suite on one Mac. Sessions at temperature 0.3 take different paths, so the totals move
  between runs; the restart counts are the robust part.
- The rewind covers single-slot servers.
