# Red Lite dev9 — native runtime foundation

Dev9 freezes the Python/ctypes path as a correctness oracle and starts the migration to a DS4-style native runtime.

The new `redlite-native` executable is written in C and has no Python dependency. In this milestone it owns two pieces that previously lived in Python:

1. GGUF v2/v3 tensor-directory parsing and routed-expert mapping;
2. hard-bounded LRU metadata scheduling with top-k and in-flight victim protection.

The field-validated dev8 Metal kernels remain unchanged and continue to be the numerical oracle while the control plane moves native.

## Build

```bash
make native
```

The binary is written to:

```text
.deps/redmetal/redlite-native
```

`make native` also runs the standalone native scheduler selftest.

## Inspect the real Qwen3-Next model

```bash
.deps/redmetal/redlite-native inspect \
  models/Qwen_Qwen3-Next-80B-A3B-Instruct-IQ2_XXS.gguf \
  --cache-mib 256 \
  --layers
```

For the field model, the native parser is expected to reproduce the Python oracle observations:

- GGUF routed layers: 48
- routed tensors: 144
- expert count: 512
- quant tensors: 33 IQ2_XS and 111 IQ1_M
- layer patterns: 11 IQ2_XS layers and 37 IQ1_M layers
- all merged routed tensors slice-safe
- top-10 fits in a 256 MiB cache

The exact routed payload and slot size must also agree with the existing Python planner within normal binary-unit formatting.

## Native selftest

```bash
.deps/redmetal/redlite-native selftest
```

It validates:

- hard capacity enforcement;
- LRU hit/miss/eviction behavior;
- current top-k selection protection during `acquire_many`;
- in-flight slot protection;
- zero Python dependency.

## Architecture after dev9

```text
Development / oracle                 Runtime direction
--------------------                 -----------------
Python GGUF parser       ───────►     native/redlite_native_gguf.c
Python LRU policy        ───────►     native/redlite_native_cache.c
Python CLI planning      ───────►     redlite-native
Python parity reference  retained     tests/oracle only
ctypes Metal probes      retained     correctness oracle
Metal kernels            retained     production kernels
```

Dev9 does **not** yet execute a token entirely through the native binary. The next native milestones are to embed canonical IQ2_XS/IQ1_M codebooks, bind the already validated Metal top-k executor directly to the native cache, then port the Qwen3-Next router and full layer/token scheduling. Python should eventually be unnecessary for normal inference.
