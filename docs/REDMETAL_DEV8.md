# Red Metal dev8 — resident routed top-k layer

Dev8 extends the field-validated dev7 single-expert FFN into one correctness-first routed top-k layer.

For a selected set of experts in one routed layer:

1. the hard-bounded LRU makes every selected expert resident;
2. selected experts are protected from eviction while remaining misses are filled;
3. the native executor pins every selected Metal slot for one command buffer;
4. each expert runs gate, up, `SiLU(gate) * up`, and down from its resident slot;
5. a Metal kernel performs `accumulator += router_weight * expert_output` after each expert;
6. only the final routed-layer output is copied back to the CPU.

The first top-10 cold selection should therefore produce 10 expert loads and 30 positional GGUF reads (gate/up/down for every expert), followed by zero SSD reads during the routed-layer command buffer.

The default validation uses ten deterministic non-contiguous expert ids and positive FP32 router weights normalized to approximately 1.0. Explicit `--experts` and `--weights` lists can override these values.

## Field validation

Build and install:

```bash
git pull --ff-only
./scripts/install.sh
make redmetal
make test
```

IQ2_XS routed layer:

```bash
redlite-topk parity \
  models/Qwen_Qwen3-Next-80B-A3B-Instruct-IQ2_XXS.gguf \
  --layer 0 \
  --top-k 10 \
  --rows 8 \
  --cache-gib 0.25
```

IQ1_M routed layer:

```bash
redlite-topk parity \
  models/Qwen_Qwen3-Next-80B-A3B-Instruct-IQ2_XXS.gguf \
  --layer 6 \
  --top-k 10 \
  --rows 8 \
  --cache-gib 0.25
```

Success requires:

- `expert loads: 10` on a cold top-10 selection;
- `pread calls: 30` on that cold selection;
- `SSD during top-k: 0 bytes / 0 calls`;
- `slots in-flight: NO` after the synchronous wait;
- `parity match: YES` against the independent CPU reference.

This is still a correctness-first scalar path. It deliberately does not yet claim production performance. Router-network integration, real token router outputs, SIMD/simdgroup expert kernels, async prefetch, cross-layer scheduling and SSD/GPU overlap remain later milestones.
