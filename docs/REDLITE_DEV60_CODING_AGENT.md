# Red Lite dev60 — a coding agent on the native server

Status: done. Measured on the Apple M4 Max 48 GiB and the Apple M4 Pro 24 GiB, 2026-10-04, with the
[pi](https://github.com/earendil-works/pi) coding agent 0.84.4.

## The question

Is a 2-bit Qwen3-Next good enough to drive a coding agent, and does it run well enough on a 24 GiB Mac? dev59 gave
the server OpenAI tool calling; this milestone measures it with a real agent and fixes what the measurement showed.

## Setup

pi talks to `redlite-server` as an OpenAI-compatible provider. No fork and no plugin: a `models.json` in pi's
config directory is enough (see the README, *Use with a coding agent*). `scripts/dev/agent_eval.py` runs five
tasks, each in a fresh directory, through `pi -p`, and grades them by script:

| task | what the agent must do | graded by |
|---|---|---|
| `new_code` | write `is_prime` and its tests, run them | hidden test cases, and its own tests pass |
| `bug_fix` | fix a failing test in `stats.py` without touching the tests | tests pass, test file unchanged, a hidden case |
| `rename` | rename a function across three files and run the program | no old name left, same output |
| `read_answer` | read a file and say the default port and its environment variable | both facts in the answer |
| `add_flag` | add an argparse `--verbose` flag and run it both ways | exact output with and without the flag |

The server runs with every expert resident and a 32K context. pi sets no temperature, so the server's default
sampling (0.7) applies and every run differs: each configuration was run three times.

## What the measurement found, and the fixes

1. **The state was not reused in agent loops with F2** (0 % on most turns, against 76–87 % with
   Qwen3-Coder-Next).
   - **Cause:** pi sends the call arguments back re-serialized (`{"a":1}` for the model's `{"a": 1}`). The
     Qwen3-Next template renders an argument string as it is, so the turn no longer matched what the model wrote.
   - **Fix:** arguments that parse as JSON are rendered in `json.dumps` form, the form the model writes (dev59b).
2. **The first turn of every task still lost it.**
   - **Cause:** before a call Qwen3-Next leaves a blank line (`text.\n\n<tool_call>`) where the template puts
     one newline.
   - **Fix:** the server now strips only that one newline from the content, so the turn renders back exactly
     (dev59b). `RL_SERVER_DEBUG_REUSE=1` logs where a new prompt diverges from the held state; that is how both
     causes were found.
3. **F2 failed `bug_fix` in 3 of 3 runs, with the right fix.**
   - **Cause:** the model closed the nested arguments of pi's `edit` tool wrongly (`…"}}]}` or `…"}}}}` for
     `…"}]}`), so the call did not parse and came back as text.
   - **Fix:** when everything after the first unmatched bracket is closing brackets, the server replaces that
     tail with the closers the open brackets need (dev60). Any other malformed call still comes back as text.
   - With it, F2 went from 11 to 14 of 15 tasks.

The remaining divergences are between tasks: pi's system prompt contains the working directory, so a new task
re-reads it (about 1,450 tokens: 1.9 s on the M4 Max).

## Results

Three runs of the five tasks; median wall time per task.

| model, Mac | tasks passed | new_code | bug_fix | rename | read_answer | add_flag | prompt reused (multi-turn tasks, median) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-Coder-Next IQ2_XXS, M4 Max | **15 / 15** | 9.8 s | 9.1 s | 11.1 s | 3.4 s | 10.6 s | 81 % |
| Red Lite F2, M4 Max (+ MTP) | 14 / 15 | 12.3 s | 12.5 s | 12.3 s | 3.1 s | 8.3 s | 73 % |
| Red Lite F2, M4 Pro 24 GiB (32K context, no MTP) | **15 / 15** | 21.6 s | 26.3 s | 29.2 s | 5.8 s | 16.0 s | 57 % |
| Qwen3-Coder-Next IQ2_XXS, M4 Pro 24 GiB (32K context) | 14 / 15 | 21.0 s | 15.5 s | 18.6 s | 5.8 s | 13.2 s | 83 % |

- **Failures:**
  - the one F2 failure is a `new_code` run whose own tests failed;
  - the one Qwen3-Coder failure on the M4 Pro is a `rename` run that left the old name in a file.
- **F2 before the fixes:** 11 of 15 on the M4 Max.
- **On the 24 GiB Mac** the GPU plan (dev55) turns MTP off at a 32K context; the tasks take about twice as long
  as on the M4 Max, in line with half the memory bandwidth.
- Record: `benchmarks/m4-dev60-coding-agent.json`.

## Scope boundary

- Five small tasks, three runs each: this shows that the loop works and that the 2-bit models follow the tools.
  It is not a benchmark of coding ability. Larger repositories, long sessions and context compaction were not
  tested.
- **No fork of pi is needed.** A fork would only make sense for something a configuration cannot do, such as
  starting the server or steering. Nothing in this measurement needed one.
- The bracket repair handles one error shape. A call broken in any other way still comes back as text, and the
  agent sees that as the answer.
