# Red Lite dev29 — server: state reuse across turns, stop sequences, FIFO queue

Status: **done** on `dev/0.4`. Implemented in the portable server core and the engine
binding; protocol tested model-free (Linux container and macOS, also under ASan/UBSan);
real-model checks on the Apple M4 Max 48 GiB only. Nothing measured on an M4 Pro or Air.

## What changed

### State reuse when a request extends the previous conversation

`redlite-server` keeps the token ids its engine state holds: the prompt of the last request
plus every generated token that was fed back into the model (a token is fed back only when
another one is decoded after it, so the end-of-turn token, the token that hit `max_tokens`
and the token that completed a stop sequence are not part of the state).

For a new request it tokenizes the whole ChatML conversation as before and calls
`rl_prefix_reuse(history, ids)`:

- **reuse** only when the held ids are a *proper prefix* of the new ids — the new request is
  exactly the previous conversation (its prompt and its answer, as generated) followed by
  new tokens. Only the suffix is prefilled (batched prefill from the current position);
- **otherwise reset** and ingest the whole prompt, as 0.3.0 always did.

The Gated DeltaNet state cannot be rolled back to an earlier position, so there is no
partial reuse: an edited earlier message, a different system prompt, a regenerated answer,
an answer truncated by a stop sequence or modified by the client, or a re-tokenization that
does not give back the same ids all fall back to the reset. A failed request invalidates the
history (the next request resets). `--no-reuse` disables the mechanism.

Responses report the reused ids as `usage.prompt_tokens_details.cached_tokens` (the OpenAI
field); the server log prints `prefill N tok (M cached)` and the time to the first token.

### Stop sequences

`"stop"`: a string or an array of up to 4 non-empty strings (≤ 256 bytes). The text is
scanned before it is sent (`rl_stop_scan`): output up to the earliest match is sent, the
stop sequence and everything after it is dropped, generation ends with
`finish_reason: "stop"`. A tail that could still become a stop sequence is held back (like
an incomplete UTF-8 character) and flushed if generation ends otherwise. 0.3.0 rejected
`stop` with a 400.

### FIFO request queue

The accepting thread now only reads and parses requests. `/health` and `/v1/models` are
answered at once, even while a generation runs (0.3.0 answered them only between
generations). Chat requests go into a FIFO served by one worker thread (the engine is
single-sequence), in arrival order. `--queue N` (default 16) requests may wait behind the
running one; more get `503 server busy`. A queued client that closed its connection is
skipped. On SIGINT the running generation stops through its sink as before and waiting
requests get `503 server is shutting down`.

## Tests

- **Model-free** (`tests/test_native_server.py`, fake echo backend, which simulates the
  engine's state with bytes): stop sequences across chunk boundaries, the earliest match
  winning, multi-byte stop strings, a held-back partial match flushed at the end; reuse
  reported only for a proper extension (not for an edited turn, a repeated prompt, an answer
  cut by a stop sequence, or after a failed request); four concurrent requests completing
  in arrival order with `/health` answered in < 0.5 s meanwhile; `503` with `--queue 1`.
  `rl_server_selftest` covers the stop scan, the prefix rule and the new parse errors.
  16 protocol tests, also run against the ASan/UBSan server by `make sanitize`.
- **Real model** (`scripts/dev/server_check.py`, regress checks `server.stream_greedy` and
  the new `server.reuse_greedy`): turn 1, then turn 2 = turn 1 + its answer + a follow-up,
  greedy. The warm server reuses 46 of 69 prompt ids; a second server with `--no-reuse`
  ingests all 69. Both answers are byte-identical, with the 2 GiB cache of the regression
  and with the 22 GiB full-residency cache (GPU-routed decode).

## Second-turn latency (TTFT)

`python3 scripts/dev/server_ttft.py MODEL --cache-mib C [--bin DIR] [--server-args=--no-reuse]`:
turn 1 = the first 4400 characters of the long-context fixture + "Summarize the text above in
one sentence." (1185 prompt ids, greedy, 32 tokens); turn 2 = turn 1 + its answer + "Which
detail matters most, and why?" (1235 ids). A fresh server per run, 60 s pause before each,
median of 3, time from the request to the first streamed content byte of turn 2. "Before"
is the 0.3.0 `redlite-server` built from `pre-dev28` (= 48c96cd).

| cache | 0.3.0 (before) | dev29 `--no-reuse` | dev29 (reuse) | reused ids |
|---|---:|---:|---:|---:|
| 4 GiB | 3833 ms | 3833 ms | **173 ms** | 1216 of 1235 |
| 22 GiB | 3779 ms | 3845 ms | **167 ms** | 1216 of 1235 |

The turn-2 answer is the same text in every configuration and run (SHA-1 `1f94f64826b2`):
0.3.0, `--no-reuse` and reuse, at both caches. The first turn itself is unchanged (≈ 3.8–4.3 s
for 1185 ids: batched prefill plus the first decode).

## Validation (M4 Max 48 GiB)

- `rm -rf .deps/redmetal && scripts/regress_m4.sh MODEL`: **47/47** (46 + `server.reuse_greedy`),
  0 compiler warnings; `make sanitize` and `make test` (61 tests) green on macOS and in a Linux
  container (`python:3.12`, arm64), `ruff` clean.

## Scope boundary

- Reuse covers exactly one retained conversation (the last one). Two clients alternating
  conversations reset each other every time; there is no prefix cache and no multi-sequence
  batching.
- The reuse decision is on token ids, not text: the answer the client sends back must
  re-tokenize to the ids that were generated. On the regression prompts it does; text that
  BPE splits differently when re-tokenized simply resets (correct, not faster).
- Warm and cold second turns are identical on the regression conversations; in general the
  warm path ingests the first answer token by token (decode) and the cold path in one batch,
  so they are the same computation in a different order, compared here only greedily.
- The queue is FIFO with no priorities or per-client fairness; a slow client sending its
  headers still occupies the accepting thread up to the 30 s receive timeout.
- Measured on the M4 Max 48 GiB only.
