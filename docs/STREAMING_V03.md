# Red Lite v0.3 streaming runtime

Status: experimental (`v0.3-streaming` branch)

## Design direction

Red Lite v0.3 follows the storage/residency shape used by DS4 rather than treating virtual-memory mappings as the expert cache.

Validated Qwen3-Next IQ2_XXS layout on the target M4 Pro 24 GiB system:

- 48 layers
- 512 routed experts per layer
- 144 routed tensors (`gate`, `up`, `down`)
- 16.91 GiB routed payload
- merged tensors are physically sliceable per expert
- largest gate+up+down expert triplet is about 0.87 MiB

## dev3 storage path

```text
GGUF on SSD
   |
   | one read-only fd
   v
ExpertStore
   | positional preadv()/pread()
   v
ExpertSlotCache
   | fixed-size reusable slots
   | hard byte budget
   | global LRU by (layer, expert)
   | async prefetch workers
   v
expert gate+up+down bytes in RAM
```

The cache allocates slots lazily, but the number of slots is fixed by:

```text
floor(cache_budget_bytes / max_expert_triplet_bytes)
```

Eviction recycles an existing slot instead of allocating more memory. This makes the Red Lite-owned expert cache deterministic: `allocated_bytes <= budget_bytes`.

## Why dev1/dev2 were replaced

- dev1 created one mmap per expert tensor slice and hit macOS `EMFILE` / `Too many open files` under realistic traces.
- dev2 used one whole-file mmap plus `madvise`. It fixed the file-descriptor failure, but mmap page residency is not a sufficiently deterministic representation of a hard expert-cache budget.
- dev3 uses explicit reads into reusable allocated slots, matching the broad DS4 SSD-streaming design more closely.

## Probe

```bash
redlite-stream probe MODEL.gguf \
  --cache-gib 4 \
  --steps 2 \
  --top-k 10 \
  --prefetch-depth 1 \
  --prefetch-workers 2
```

The probe uses a deterministic synthetic router trace. It measures storage/cache behavior only: demand hits/misses, prefetch waits, evictions, allocated cache bytes, bytes read and positional-read throughput.

It does **not** yet execute MoE kernels on Metal.

## Next Metal milestone

The next stage replaces Python bytearray slots with Metal-visible shared buffers while retaining the same store/cache policy. The router-selected slot addresses will then feed a Qwen3-Next MoE Metal execution path. Correctness will be checked against the pinned llama.cpp Qwen3-Next implementation before performance tuning.
