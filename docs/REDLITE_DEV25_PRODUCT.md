# Red Lite dev25 (partial) — product surface: chat defaults, native server, Ctrl-C, JSON stats

Status: **implemented, tested model-free on Linux, and validated with the real model on the
M4 Max 48 GiB** (macOS 27, commit `7d96db1`, `scripts/regress_m4.sh` 44/44 after a clean
build; see "Validation on the M4 Max" below). Nothing here is measured on the M4 Pro.
dev22–dev24, the performance milestones, are not started, so this work comes ahead of
them in time but is numbered by scope.

## What changed

### `redlite chat` defaults (`f4e8ddd`, `2c7ae6c`)

- If `--cache-mib` is not given, `redlite.planner.native_defaults()` picks the cache from
  RAM:
  - **≥ 40 GiB → 22528 MiB.** Every routed expert is resident and preloaded at open, and
    decode uses the GPU-routed path. This is the dev21 setting validated on the M4 Max
    48 GiB.
  - **< 40 GiB → 4096 MiB.** The field-validated bounded cache, as on the M4 Pro 24 GiB.
  - The choice and its reason are printed. An explicit `--cache-mib` always wins.
- `--batch N` (prefill chunk size) and `--json` are passed through to `redlite-generate`.
- **Ctrl-C** now reaches the native runtime. Previously `subprocess.call` turned the
  terminal's SIGINT into a KeyboardInterrupt in the Python launcher and killed
  `redlite-generate`, so Ctrl-C always ended the whole chat. The launcher now ignores
  SIGINT while the child runs.

### `redlite-generate` (`f7e5887`)

- **Ctrl-C stops the current answer.** The answer is closed exactly like a `--max-tokens`
  stop: every emitted token is already in the KV/DeltaNet state, and `<|im_end|>` opens the
  next turn. The conversation therefore stays consistent and the next question can follow.
  - At the `you>` prompt, Ctrl-C quits and the engine is closed normally.
  - Non-interactive runs print their statistics and exit 130.
- **`--json`** writes one statistics object per answer on stderr:
  - prefill and decode tokens, ms and tok/s;
  - GPU-routed tokens and fallbacks;
  - cache hits and misses, expert loads, SSD MiB;
  - peak RSS and physical footprint;
  - cache, context and batch;
  - `finish` = `stop | length | interrupted`.
- **One language.** The interactive chat was Italian while every other message was
  English; it is now English.
- **Chat template.** The GGUF `tokenizer.chat_template` is checked for ChatML
  (`rl_tokenizer_chat_template_source`); if it is missing or not ChatML, the built-in
  ChatML is used. The source is printed with `--stats`.
  - **The template is not interpreted.** There is no Jinja engine. The Qwen3-Next Instruct
    GGUF ships a ChatML template, and the hard-coded ChatML is the format validated
    token-for-token against llama.cpp.

### Native OpenAI-compatible server (`3060e9b`, `b6648c1`)

- `native/redlite_native_server.[ch]` is portable C11 plus POSIX sockets.
  - **Endpoints:** `POST /v1/chat/completions` with `stream` true or false, `GET /v1/models`,
    `GET /health`.
  - **Request handling:** one request at a time, `Connection: close`.
  - **Streaming:** SSE chunks in the OpenAI `chat.completion.chunk` shape (role delta,
    content deltas, then `finish_reason` + `usage` + `[DONE]`).
  - **UTF-8 hold-back:** a character split across tokens is never sent in two chunks.
  - **Rejected with 400 rather than silently ignored:** stop sequences, `n > 1`, tools and
    non-text content.
  - **Accepted extensions:** `developer` messages are treated as `system`, and `top_k` is
    accepted.
  - **Failures:**
    - a failure mid-stream ends the stream with an error event and no `[DONE]`;
    - a client that disconnects stops generation;
    - SIGINT stops the current generation and shuts the server down cleanly.
- `redlite-server MODEL` binds the core to `rl_engine`. Each request resets the engine,
  ingests the ChatML conversation with the batched prefill and decodes like
  `redlite-generate`. The backend is Metal on macOS; `--cpu` selects the CPU oracle, which
  is the only backend elsewhere.
- `redlite serve --native` launches it with the RAM-based cache default, `--context`
  (default 4096) and `--batch`.

## Validation

Machine: **Linux x86_64 cloud container** (Ubuntu 24.04, gcc 13.3, clang 18.1). There is
no Metal and no GGUF on it.

| Check | Command | Result |
|---|---|---|
| protocol, fake engine | `PYTHONPATH=. python3 -m unittest tests.test_native_server` | 12/12. Covers streaming/blocking, split UTF-8, `max_tokens`, defaults, multi-turn ChatML, 400/404/405/500, mid-stream error, client disconnect, SIGINT, 300-mutation body fuzz |
| same, sanitized server | `CC=gcc make sanitize` | no ASan/UBSan report |
| server core selftest | `.deps/redmetal/redlite-server-fake --selftest` | OK |
| launcher | `make test` | 57 OK. Covers RAM-based defaults (48/40/24 GiB), `serve --native` command, and a Ctrl-C test that fails with the old launcher |
| builds | `rm -rf .deps/redmetal && CC={gcc,clang} make native` | rc 0, 0 warnings. `redlite-server` links against the CPU-oracle engine on Linux; `redlite-generate` was compiled by hand the same way |

**Added to `regress_m4.sh` for the Mac:**

- `server.stream_greedy` (`scripts/dev/server_check.py`). The temperature-0 SSE stream
  and the blocking answer must equal `redlite-generate`'s greedy text byte for byte, and
  SIGINT must end the server with exit 0.
- `generate.json` checks that the statistics line is written.
- `generate.sigint`: Ctrl-C mid-answer must give exit 130 and `"finish":"interrupted"`.

### Validation on the M4 Max 48 GiB (commit `7d96db1`)

`rm -rf .deps/redmetal && make native` (0 warnings), then
`scripts/regress_m4.sh models/Qwen_Qwen3-Next-80B-A3B-Instruct-IQ2_XXS.gguf`:
**44/44 PASS**, including `server.stream_greedy` (SSE stream and blocking answer
byte-identical to `redlite-generate` greedy, clean SIGINT), `generate.json` (the real
GGUF's `tokenizer.chat_template` is confirmed ChatML: `"chat_template_from_gguf":true`) and
`generate.sigint` (exit 130, `"finish":"interrupted"`).

Manual checks on the same machine and commit:

- `./bin/redlite chat --dry-run` with no flags picks `--cache-mib 22528` (48.0 GiB RAM,
  full residency).
- `redlite serve --native MODEL --port 8093`, then a streaming `curl` request
  (temperature 0, 40 tokens): SSE chunks, `finish_reason:"stop"`, `usage` and `[DONE]`.
  Every decode token was GPU-routed and SIGINT shut the server down cleanly. The engine
  was ready in 5.6 s with every expert preloaded.
- Interactive `redlite-generate --interactive --json`, Ctrl-C (SIGINT) mid-answer: the
  answer ends with `[interrupted]`, `"finish":"interrupted"`. The next question ("What was
  my previous request?") is answered correctly, so the conversation state survived, and
  `/quit` exits 0.

No benchmark record is added. The server log prints per-request tok/s, but a single
request is an observation, not a benchmark, and throughput is not claimed here.

## Scope boundary

- **Validated on the M4 Max 48 GiB only.** Nothing is run on the M4 Pro 24 GiB. The
  4096 MiB default for machines under 40 GiB is the field-validated dev18 setting, but
  the new launcher logic that picks it has not run there.
- **Throughput is not claimed.** The server's throughput should equal `redlite-generate`'s
  plus HTTP overhead, but that is not measured.
- **The server is single-sequence and stateless.** Every request re-ingests the whole
  conversation; there is no prefix reuse between requests. Concurrent requests queue on
  the accept loop.
- **The server is not hardened for untrusted networks.**
  - It binds 127.0.0.1 by default.
  - Robustness measures: bounded headers (64 KiB) and bodies (8 MiB), a 30 s receive
    timeout, and a JSON nesting limit of 64.
  - There is no TLS, no authentication and no rate limiting.
- **Other limits:**
  - No Jinja chat-template interpretation. Stop sequences, tools and logprobs are not
    supported.
  - Interactive Ctrl-C cannot interrupt a prefill in progress; the answer stops right
    after it.
