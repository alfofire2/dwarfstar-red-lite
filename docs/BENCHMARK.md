# Benchmark protocol

## Fast field sweep

v0.2 adds a context-depth sweep for the resident Metal path:

```bash
redlite sweep /path/model.gguf \
  --contexts 2048,4096,8192 \
  --prompt-tokens 256 \
  --gen-tokens 128 \
  --max-swap-delta 0.5 \
  --output benchmarks/sweep.json
```

For each depth Red Lite:

1. recalculates the resident budget and status;
2. skips a depth if the policy says it is genuinely unsafe;
3. runs pinned `llama-bench` with the same Metal/cache/batch policy as Red Lite;
4. records prompt/decode throughput;
5. snapshots macOS swap before and after the run;
6. stores raw benchmark rows and hardware metadata in JSON.

Current llama-bench supports JSON output and a context-depth (`-d`) test that prefills
the cache before measuring. This is preferable to judging stability from one chat.

## Single profile

```bash
redlite bench /path/model.gguf -c 2048 --prompt-tokens 256 --gen-tokens 128
```

Record:

- chip and macOS version
- physical RAM
- quant
- context/depth
- prompt processing tok/s
- decode tok/s
- swap delta / Memory Pressure
- temperature if doing long sweeps

## Oversized Q4

The oversized backend uses a deterministic completion workload rather than
`llama-bench`, because the residency policy is part of the end-to-end runtime.
Run at least three fresh launches; filesystem cache state can materially change it.

## Acceptance criteria for a 24 GiB profile

A profile should not be called working merely because it emitted one token:

1. 256 generated tokens without crash.
2. macOS remains responsive.
3. No sustained red Memory Pressure.
4. Swap growth is recorded and judged acceptable for the intended mode.
5. Deterministic prompt produces stable output at temp=0.
6. OpenAI-compatible server completes at least 10 sequential requests.
7. Throughput is always reported with quant and context.
