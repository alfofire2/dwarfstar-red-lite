# Red Lite dev59 — OpenAI tool calling in the server

Status: done. Tested with the fake backend (Linux and macOS, ASan+UBSan) and with the real models on the Apple M4
Max 48 GiB, 2026-10-04.

## Why

Coding agents and most assistant front ends talk to a model through OpenAI tool calling: the request carries
`tools`, the model answers with `tool_calls`, and the tool results come back as `tool` messages. `redlite-server`
rejected all of it, so no agent could use it.

## What the server does now

- **Requests.**
  - `tools`, with `tool_choice` (`"none"` drops the tools; `"auto"`, `"required"` and named functions act as auto);
  - assistant messages with `tool_calls`, whose `content` may be null;
  - `tool` messages;
  - the legacy `functions` / `function_call` API is rejected with a pointer to `tools`.
- **Prompt.** Rendered exactly as the model's own chat template does, in two formats chosen from the GGUF's template:
  - **Qwen3-Next Instruct** (F2, E3, Bartowski's files): the tools as JSON lines in the system message, calls as
    `<tool_call>{"name": ..., "arguments": ...}</tool_call>`;
  - **Qwen3-Coder** (Qwen3-Coder-Next): the tools as XML, calls as
    `<tool_call><function=NAME><parameter=KEY>value</parameter></function></tool_call>`.
- **Answers.**
  - Text before the first `<tool_call>` streams as content.
  - The calls are collected and returned as `tool_calls` (one SSE delta when streaming), with `finish_reason`
    `"tool_calls"`.
  - In the XML format a value is a JSON string unless the tool's schema gives that parameter another type and the
    value parses as JSON (`"3"` for an integer becomes `3`; `"12"` for a string stays `"12"`).
  - Output that is not well-formed calls goes out as text, so nothing is lost.

## Why the rendering must be exact

In an agent loop every request repeats the whole conversation. The server keeps the engine state when a new prompt
extends exactly the token ids it holds (dev29). If an assistant turn rendered from the returned `tool_calls`
differed by one byte from what the model wrote, every turn would re-read the whole prompt, and coding agents send
10–20K tokens of system prompt. Two things keep it exact:
- the returned `arguments` are the model's own text;
- the separator whitespace before a call is held back, not streamed as content.

## Checks

- **Rendering against Jinja** (`tests/test_native_server_tools.py`): the C renderer is compared byte for byte with
  the two templates copied from the GGUFs (`tests/fixtures/chat_template_*.jinja`), rendered by Jinja. It covers 5
  conversations each: tools with and without a system message, several calls in one turn, consecutive tool results,
  null content, escaped quotes.
- **Parsing:** JSON and XML calls, several calls, streaming, schema types, malformed calls, no tools,
  `tool_choice: "none"`, request validation. 150 mutated bodies per format get 200 or 400, and the servers stay up.
  All of it also runs under ASan+UBSan (`make sanitize`).
- **Round trip with the fake:** the second turn reuses the first turn's state.
- **Real models** (M4 Max, `server_check.py --tools`, now `server.tool_call` in `regress_m4.sh`):

  | model | turn 1 | turn 2 | state reused |
  |---|---|---|---:|
  | Red Lite F2 (full residency) | `get_weather {"city": "Rome"}` | "The current weather in Rome is clear with a temperature of 19°C." | 203 / 230 tokens |
  | Bartowski IQ2_XXS (2 GiB cache) | same | same | 176 / 203 |
  | Qwen3-Coder-Next IQ2_XXS (full residency) | same, XML format | same | 350 / 377 |

## Scope boundary

- **Rendering limits.**
  - Numbers in `tools` keep their written form: Python would print `1E5` as `100000.0`.
  - A list-valued `type` renders as JSON where Jinja prints a Python list.
- **Qwen3-Coder turns with text before a call** are rendered by its template with an extra newline that the model
  did not write. Such turns lose the state reuse; the answer is not affected.
- **`tool_choice: "required"` and named functions** do not force a call.
- **Not measured yet:** whether a 2-bit model is reliable enough for agentic coding. That is the next step, with
  Qwen3-Coder-Next and a real agent.
