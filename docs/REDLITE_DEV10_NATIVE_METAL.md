# Red Lite dev10 — standalone native Metal path

Dev10 moves the validated dev8 routed top-k execution path behind the standalone
`redlite-native` binary. Python remains in the repository as a development oracle,
benchmark harness and regression suite, but it is no longer required by the new
native execution path.

## What is native in dev10

`redlite-native` now owns:

1. GGUF v2/v3 parsing;
2. routed expert tensor discovery and physical expert slicing;
3. routed layer shape/type validation;
4. hard-bounded top-k-aware LRU metadata;
5. canonical IQ2_XS and IQ1_S/IQ1_M codebooks embedded in compact form;
6. expansion of those codebooks without `gguf-py` or Python;
7. expert miss loading through the Red Metal top-k pool;
8. resident gate/up/down execution and GPU router-weighted accumulation;
9. telemetry asserting zero SSD reads while the top-k Metal command executes.

On macOS the standalone executable links the Red Metal top-k Objective-C/Metal
implementation directly. There is no Python interpreter and no `ctypes` boundary in
`topk-probe`.

The Linux build intentionally retains only the portable C parser/LRU/table selftest;
Metal execution is Apple-only.

## Quant grids

The compact canonical grids are adapted from the pinned llama.cpp/GGML source at:

`7798007a29a90e3053e799394da48cf53a2f8e0f`

- IQ2_XS: 512 × 8 values, compact 2-bit symbols mapped to `0x08/0x19/0x2b`.
- IQ1_M: reuses the IQ1_S 2048 × 8 grid, compact 2-bit symbols mapped to `-1/0/+1`.

The standalone runtime expands them once before creating the Metal pool. See
`NOTICE.md` for attribution.

## Commands to run when the target Mac is available

Build:

```bash
git pull --ff-only
make native
.deps/redmetal/redlite-native selftest
```

Native model inspection:

```bash
.deps/redmetal/redlite-native inspect \
  models/Qwen_Qwen3-Next-80B-A3B-Instruct-IQ2_XXS.gguf \
  --cache-mib 256 \
  --layers
```

Expected field invariants from the already validated Python oracle are:

- 144 routed tensors;
- 48 routed layers;
- 512 experts;
- all slices safe;
- 33 IQ2_XS routed tensors and 111 IQ1_M routed tensors;
- 11 all-IQ2_XS routed layers and 37 all-IQ1_M routed layers.

Native Metal top-k probe for an IQ2_XS layer:

```bash
.deps/redmetal/redlite-native topk-probe \
  models/Qwen_Qwen3-Next-80B-A3B-Instruct-IQ2_XXS.gguf \
  --layer 0 \
  --top-k 10 \
  --rows 8 \
  --cache-mib 256
```

Native Metal top-k probe for an IQ1_M layer:

```bash
.deps/redmetal/redlite-native topk-probe \
  models/Qwen_Qwen3-Next-80B-A3B-Instruct-IQ2_XXS.gguf \
  --layer 6 \
  --top-k 10 \
  --rows 8 \
  --cache-mib 256
```

The deterministic input, expert ids and normalized router weights intentionally match
dev8. This lets the standalone output rows be compared directly against the already
field-validated `redlite-topk parity` results.

For a cold top-10 run we expect:

- `expert loads: 10`;
- `cache hits/misses: 0/10`;
- `pread calls: 30`;
- `SSD during top-k: 0 bytes / 0 calls`;
- output rows matching the dev8 GPU values within normal FP32 tolerance.

## What dev10 does not yet claim

Dev10 still uses deterministic expert ids and router weights. It does not yet execute
the real Qwen3-Next router network, the dense/recurrent portions of the model, KV
state, sampling, tokenizer logic, or a full generation graph.

It also remains a scalar correctness-first Metal implementation. SIMD/simdgroup
optimization, asynchronous `pread`, router-driven prefetch and SSD/GPU overlap are
future milestones.

The next major milestone should connect the actual Qwen3-Next router output to this
native top-k path, but only after the standalone dev10 probe is field-validated on the
target M4 Pro.
