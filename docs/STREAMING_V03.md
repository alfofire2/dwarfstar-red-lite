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

## dev4 Red Metal path

The first native Metal milestone removes the production-path `bytearray` copy. Expert slots are ranges inside lazily allocated `MTLStorageModeShared` slabs:

```text
GGUF / SSD
   |
   | pread() directly into MTLBuffer.contents
   v
shared Metal slab
   |
   +-- slot N
       +-- gate
       +-- up
       +-- down
   |
   v
Metal compute-visible bytes
```

The hard cache capacity still comes from `floor(cache_budget / slot_bytes)`. Slabs are allocated lazily and never beyond that budget. The Python LRU recycles slot ids; the native bridge reuses the same Metal ranges.

Build the native bridge on Apple Silicon macOS with:

```bash
make redmetal
```

Then verify direct SSD -> shared Metal residency with:

```bash
redlite-stream metal-probe MODEL.gguf \
  --cache-gib 0.5 \
  --steps 1 \
  --top-k 10
```

`metal-probe` performs a deterministic router trace, reads cache misses directly into `MTLBuffer.contents`, and finally runs a small Metal compute kernel over one resident expert. A verification-only CPU reread computes the same sampled FNV-1a checksum; `checksum match: YES` proves that the GPU sees the bytes loaded from the GGUF without a production-path intermediate Python buffer.

The probe kernel is intentionally not a MoE matmul. It validates storage-to-Metal visibility and slab reuse before introducing IQ2_XXS arithmetic.

## Python storage probe

```bash
redlite-stream probe MODEL.gguf \
  --cache-gib 4 \
  --steps 2 \
  --top-k 10 \
  --prefetch-depth 1 \
  --prefetch-workers 2
```

The Python probe remains useful for cache-policy and prefetch experiments. It measures demand hits/misses, prefetch waits, evictions, allocated cache bytes, bytes read and positional-read throughput.

## Next Metal milestone

After `metal-probe` is validated on the target M4 Pro, the next stage is to replace the probe kernel with a Qwen3-Next routed-expert execution path. The native cache will expose the resident slot's gate/up/down buffer offsets to an IQ2_XXS Metal kernel, with in-flight slot protection before eviction. Correctness will be checked against the pinned llama.cpp Qwen3-Next implementation before performance tuning.
