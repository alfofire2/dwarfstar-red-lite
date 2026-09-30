# Red Lite dev11 — native CPU/GPU parity harness

Dev11 removes Python from the correctness harness used to validate the standalone
native routed-expert path.

The hot path remains the dev10 native runtime:

`GGUF -> C ExpertMap -> C top-k LRU -> pread into shared Metal slots -> Metal FFN -> GPU weighted accumulation`

Dev11 adds an independent C CPU oracle alongside it.

## Native CPU quant reference

`native/redlite_native_reference.c` implements correctness-first row-dot references for:

- IQ2_XS (`GGML_TYPE 17`, 256 values / 74-byte block);
- IQ1_M (`GGML_TYPE 29`, 256 values / 56-byte block).

It uses the same canonical quant grids embedded by dev10 but independently evaluates
quantized rows on the CPU using positional reads and double-precision accumulation.
Gate, up, stable `SiLU(gate) * up`, selected down rows and router-weighted top-k
accumulation are all evaluated natively.

The standalone `selftest` now includes synthetic exact-value cases equivalent to the
Python oracle tests:

- IQ2_XS synthetic all-ones block -> dot product `256.0` for 256 unit inputs;
- IQ1_M known synthetic block -> dot product `32.0` for 256 unit inputs.

These tests run on both Linux and macOS CI and do not require a model file.

## Field commands for later

When the target M4 Pro is available again:

```bash
git pull --ff-only
make native
.deps/redmetal/redlite-native selftest
```

Then validate the full standalone native path on IQ2_XS:

```bash
.deps/redmetal/redlite-native topk-parity \
  models/Qwen_Qwen3-Next-80B-A3B-Instruct-IQ2_XXS.gguf \
  --layer 0 \
  --top-k 10 \
  --rows 8 \
  --cache-mib 256
```

And IQ1_M:

```bash
.deps/redmetal/redlite-native topk-parity \
  models/Qwen_Qwen3-Next-80B-A3B-Instruct-IQ2_XXS.gguf \
  --layer 6 \
  --top-k 10 \
  --rows 8 \
  --cache-mib 256
```

Success requires the same storage invariants already field-validated in dev8:

- cold top-10 -> 10 expert loads;
- 30 pool `pread` calls for gate/up/down residency;
- zero pool reads during the Metal top-k command;
- all selected slots released after the synchronous command;
- `parity match: YES` against the native C CPU reference.

Default tolerance remains `1e-3 + 1e-4 * abs(reference)`, matching the earlier Python
parity harness.

## Why keep the CPU oracle

The CPU implementation is intentionally not the production inference path. It exists to
separate arithmetic/layout correctness from Metal scheduling and optimization. Future
SIMD/simdgroup kernels, fused expert execution, async SSD prefetch and real router
integration can be checked against the same standalone native oracle without reintroducing
Python into the runtime.

The next production milestone should still wait for real-model dev10/dev11 field
validation before connecting Qwen3-Next's actual router network.
