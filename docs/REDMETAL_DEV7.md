# Red Metal dev7 — resident single-expert FFN

Dev7 moves the field-validated IQ2_XS and IQ1_M arithmetic into a hard-bounded resident expert cache.

For one `(layer, expert)` miss, gate/up/down are read once from the GGUF directly into one shared Metal slot. The same resident slot is then used for:

1. gate matvec
2. up matvec
3. `SiLU(gate) * up`
4. down matvec

The validation probe intentionally compares only a small selectable set of final down rows, but both gate and up projections are evaluated in full so the intermediate FFN vector is real.

The native pool enforces a hard byte budget, slab allocation, global Python-side LRU recycling and native in-flight slot protection. `expert-ffn-parity` also asserts that the Metal execution phase performs zero additional GGUF reads after residency.

## Field commands

Build and install:

```bash
git pull --ff-only
./scripts/install.sh
make redmetal
make test
```

IQ2_XS layer:

```bash
redlite-ffn parity \
  models/Qwen_Qwen3-Next-80B-A3B-Instruct-IQ2_XXS.gguf \
  --layer 0 --expert 0 --rows 8 --cache-gib 0.25
```

IQ1_M layer:

```bash
redlite-ffn parity \
  models/Qwen_Qwen3-Next-80B-A3B-Instruct-IQ2_XXS.gguf \
  --layer 6 --expert 0 --rows 8 --cache-gib 0.25
```

Success requires `parity match: YES`, `SSD during FFN: 0 bytes`, non-zero gate/up/down GPU addresses and `slot in-flight: NO` after the synchronous wait.

This remains a correctness-first scalar Metal path. SIMD-group kernels, selected-expert batching, router weights, top-k accumulation, asynchronous SSD prefetch and overlap are later milestones.
