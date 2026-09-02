# Native Metal + SSD expert streaming roadmap

Red Lite 0.2 does **not** claim to have solved native Metal expert streaming yet.
The quality path remains the separately validated CPU/Accelerate mmap-residency
backend. This document defines the next native backend instead of hiding the gap.

## Goal

Run a Qwen3-Next-80B-A3B quantization substantially larger than unified memory while:

1. keeping dense/shared/non-expert tensors resident;
2. storing routed MoE experts in a file-backed SSD region;
3. selecting experts from the router before they are needed;
4. asynchronously staging selected expert tensors into a bounded unified-memory cache;
5. executing those experts with Metal without a second CPU/GPU copy;
6. evicting cold experts under a global LRU / pressure-aware policy.

## Why this cannot be a launcher flag

Qwen3-Next combines Gated DeltaNet, periodic full attention and 512 routed experts.
A correct streaming backend needs to control tensor storage and lifetime inside the
model execution graph. Merely asking macOS to swap a 48 GiB Metal allocation is not
expert streaming and can destroy latency.

## v0.2 groundwork

The control plane now exposes the pieces needed to validate such a backend later:

- explicit `CRITICAL/TIGHT/SAFE` resident states;
- repeatable context-depth sweep with JSON results;
- swap telemetry;
- a conservative 24 GiB field profile;
- pinned Metal and oversized engines kept separate;
- no claim that stock mixed Metal/CPU expert offload is equivalent to streaming.

## Native backend milestones

### M1 — GGUF expert map

Build an index at load time containing, per layer and expert:

- gate/up tensor file offset and byte length;
- down tensor file offset and byte length;
- quant type / block geometry;
- target Metal buffer alignment.

Dense tensors and router weights remain normal resident tensors.

### M2 — unified expert cache

Implement a bounded cache whose entries are Metal-visible shared buffers. The cache
must have a hard byte quota rather than relying on VM pressure.

Suggested first target for 24 GiB machines: benchmark 4, 6 and 8 GiB quotas rather
than hardcoding a winner.

### M3 — asynchronous prefetch

After layer L's router result is known, prefetch experts for L (and, when prediction
is reliable, L+1) using asynchronous positional reads. Maintain a small queue of
in-flight reads and deduplicate requests for already resident experts.

### M4 — Metal quantized MoE kernels

Consume the staged quantized blocks directly from the expert cache. Avoid expanding
an expert to FP16 in unified memory. The first implementation should support only the
quant types selected for Red Lite, not every GGUF quant.

### M5 — pressure feedback

When macOS pressure worsens or swap begins to increase, reduce the expert-cache quota
and prefetch distance. When pressure is healthy, cautiously grow within the configured
maximum.

### M6 — correctness gate

Before calling the backend usable:

- compare logits/tokens with the pinned reference backend at temperature 0;
- run at least 256 generated tokens without a cache miss correctness failure;
- test repeated expert eviction/reload;
- validate 2K/4K/8K context;
- compare throughput against the CPU SSD path and IQ2_XXS resident Metal path.

Until these milestones are implemented and tested on real Apple Silicon, `ssd-cpu`
is the honest quality fallback.
