# Architecture

## Design goal

DwarfStar Red Lite is intentionally narrow: Qwen3-Next on Apple Silicon. The key
constraint is that "80B" describes total sparse capacity, while a 24 GB machine
cannot hold a normal Q4 copy of all 80B weights in unified memory.

## Qwen3-Next shape

Reference configuration used by this project:

- 80B total parameters / ~3B activated
- hidden size 2048
- 48 decoder layers
- hybrid pattern repeated 12 times:
  - 3 × Gated DeltaNet -> MoE
  - 1 × Gated Attention -> MoE
- 512 routed experts
- top-10 routed experts per token
- one shared expert
- expert intermediate size 512

This is sparse enough that storage residency can be treated independently from
active compute.

## Control plane

`redlite` is the control plane. It does not duplicate model weights and does not
invent a second GGUF format.

1. Detect Apple Silicon and physical RAM.
2. Measure the real GGUF file size.
3. Reserve memory for macOS, runtime scratch and context state.
4. Select one of two engines:
   - Metal resident
   - CPU oversized mmap/residency
5. Launch the matching pinned backend with safe defaults.

## Metal resident path

The resident path uses upstream Qwen3-Next kernels and Metal graph execution from
llama.cpp. Red Lite uses:

- `-ngl 999` to place supported graph work on Metal;
- Flash Attention when supported;
- q8 KV/cache types as a memory-quality compromise;
- modest batch/ubatch on 24 GB machines;
- no `mlock`, because the OS must retain freedom to manage unified memory.

This path is intended for the ~19 GB IQ2_XXS 80B quant on a 24 GB Mac, initially at
2K context on especially tight 24 GiB profiles; 4K/8K are measured opt-in steps.

## SSD-residency path

For a ~48 GB Q4 model, forcing Metal residency is the wrong memory policy on 24 GB.
The oversized path keeps the GGUF mmap-backed and uses bounded zero-copy residency
for hot routed-expert pages. Cold experts remain pageable and are faulted from SSD.

The consequence is important: this mode is **CPU/Accelerate today**, not a fake
"Metal streaming" claim. It trades I/O for quantization quality and capacity.

## Future native Red Metal path

A future `red-metal` backend can merge both paths:

- dense/shared tensors permanently resident in unified memory;
- routed experts stored in an SSD-friendly packed file;
- async `pread` / dispatch_io into a bounded shared expert cache;
- router-driven prefetch for the next MoE layer;
- Metal quantized matvec/GEMM directly from cache buffers;
- adaptive global LRU rather than rigid per-layer quotas;
- memory-pressure feedback using macOS pressure APIs;
- no CPU/GPU duplicate expert copies.

That work requires numerical validation against a reference implementation on real
Apple hardware, so it is deliberately not represented as finished in v0.2. See `METAL_STREAMING_ROADMAP.md`.
