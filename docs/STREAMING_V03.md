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

The cache allocates slots lazily, but the number of slots is fixed by `floor(cache_budget_bytes / max_expert_triplet_bytes)`. Eviction recycles an existing slot instead of allocating more memory.

## Why dev1/dev2 were replaced

- dev1 created one mmap per expert tensor slice and hit macOS `EMFILE` / `Too many open files` under realistic traces.
- dev2 used one whole-file mmap plus `madvise`. It fixed the file-descriptor failure, but mmap page residency is not a sufficiently deterministic representation of a hard expert-cache budget.
- dev3 uses explicit reads into reusable allocated slots, matching the broad DS4 SSD-streaming design more closely.

## dev4 Red Metal residency

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

Build with:

```bash
make redmetal
```

Verify storage-to-GPU visibility with:

```bash
redlite-stream metal-probe MODEL.gguf \
  --cache-gib 0.25 \
  --steps 1 \
  --top-k 10
```

Field validation on the Apple M4 Pro reference machine produced an exact GPU/CPU checksum match while respecting a 0.25 GiB hard Metal cache budget.

## dev5 address tables, lifetime and IQ2 parity

Dev5 adds the first arithmetic execution path while keeping the dev4 probe intact as a regression baseline.

Each resident `(layer, expert)` is installed in three GPU-visible shared address tables:

```text
layer L
  gate[expert] -> slab GPU address + slot base
  up[expert]   -> gate + gate_bytes
  down[expert] -> up + up_bytes
```

Before a slot is reused, the LRU queries native in-flight state. `pread()` refuses to overwrite a slot while a Metal command still references it. Eviction also clears the old address-table entry before the slot is recycled.

The first IQ2 kernel is intentionally correctness-first, not performance-tuned. It uses the canonical ggml IQ2_XXS physical format:

```text
256 values / block
66 bytes / block
  2 bytes  FP16 d
 64 bytes  packed IQ2 codebook/sign/scale data
```

For each tested row, Metal decodes the quantized expert weights directly from the SSD-filled shared slab and computes a FP32 dot product. An independent pure-Python decoder rereads the same GGUF row and computes the reference value.

Run the parity probe with:

```bash
redlite-stream iq2-parity MODEL.gguf \
  --layer 0 \
  --expert 0 \
  --kind gate \
  --rows 8 \
  --cache-gib 0.25
```

The command validates before dispatch that:

- the selected routed tensor is GGML type 16 (`IQ2_XXS`)
- the expert axis and rank match the mapped Qwen3-Next layout
- `ncols` is a multiple of 256
- the physical expert slice length equals `nrows * (ncols / 256 * 66)`
- the gate/up/down GPU addresses are non-zero
- the slot is no longer in-flight after the synchronous parity command returns

The key output is `parity match: YES`, together with max absolute/relative error and per-row GPU/CPU values.

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

After dev5 numerical parity is field-validated on the target M4 Pro, the next path is:

1. vectorize the IQ2_XXS matvec with SIMD-groups while preserving parity
2. run gate and up from the resident address table
3. fuse SiLU(gate) * up
4. add the down projection
5. accumulate the router top-k weighted expert outputs
6. overlap expert `pread` with useful Metal work
7. compare complete routed-FFN output with the pinned llama.cpp Qwen3-Next implementation

Full generation is not claimed until that routed-FFN parity exists.
