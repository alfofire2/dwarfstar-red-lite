# Red Lite dev19 — user-facing native chat command

Dev19 connects the native end-to-end engine from dev18 to the installed Red Lite
command line. Users no longer need to know the internal
`.deps/redmetal/redlite-generate` build path to open a persistent conversation.

## Command

```bash
redlite chat --stats
```

`redlite chat` invokes the native generator with `--interactive`. The model is
opened once; the DeltaNet recurrent state and full-attention KV cache remain live
between turns. With no positional argument it uses the reference GGUF from the
repository's `models/` directory; a different path can still be supplied as
`redlite chat MODEL.gguf`. The default profile is intended for the field-validated M4 Pro
with 24 GiB unified memory:

- 4096 context positions;
- 4096 MiB routed-expert cache;
- 256 generated tokens per answer;
- temperature 0.7, top-k 40 and top-p 0.95.

Pass `--temperature 0` for the greedy path that was compared token-for-token with
the pinned llama.cpp oracle. `--prompt` supplies an optional first user message;
`--system` supplies a system instruction. The native interactive commands remain
`/reset`, `/help` and `/quit`.

## Failure behaviour

The launcher checks Apple Silicon and the model path before starting. If the
native executable is absent, it reports `make native` as the recovery action.
`redlite doctor` now reports `native_redlite_generate` independently from the
pinned llama.cpp and oversized-runtime binaries.

## Validation

- Python parser tests cover defaults and all exposed options.
- A runner test verifies the exact native command without launching inference.
- A real-GGUF one-token smoke test opened the native Metal engine through
  `redlite chat`, completed the first turn and exited cleanly with `/quit`.
- The existing dev18 native build, engine parity and real-model generation tests
  remain the inference correctness gates; this milestone does not change engine
  arithmetic.

## Scope boundary

This command is a terminal chat client. It does not add a native HTTP server,
batched prompt ingestion or new model/quantization support. `redlite run` and
`redlite serve` retain their v0.2 behavior and continue to select the pinned
upstream engines through the memory planner.
