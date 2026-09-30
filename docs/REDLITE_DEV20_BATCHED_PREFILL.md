# Red Lite dev20 — batched prompt ingestion (native Metal prefill)

Target machine for this milestone: Apple M4 Max, 48 GiB (see `CLAUDE.md` on
naming the machine). Model: Bartowski `Qwen_Qwen3-Next-80B-A3B-Instruct-IQ2_XXS.gguf`.

## What changed

`rl_engine_prefill()` ingests a token sequence in chunks (`prefill_batch`,
default 512; `--batch N` on `redlite-generate` / `redlite chat`; 1 = the old
token-by-token path). The CPU oracle prefill is the sequential step sequence.
On the Metal backend (`native/redmetal_engine_prefill.m`) every layer runs:

1. one command buffer for the dense work of the whole chunk: per-token RMSNorm
   and residuals, Gated DeltaNet with the causal conv and the gated delta rule
   iterated over the chunk *inside* single dispatches (`dn_conv_seq`,
   `dn_state_seq`, state updated in place), full attention appending all
   keys/values (`attn_qk_prep_b`) then causal GQA per (token, head)
   (`attn_gqa_b`, same chunked online softmax as decode), router, shared expert;
2. the router selection of every token on the CPU (`rl_native_router_select_softmax_topk`);
3. the routed experts of the chunk as one bounded-pool plan holding the union of
   the selected experts (up to 512 per plan, `REDMETAL_TOPK_MAX`), executed by
   three batched dispatches: `redmetal_topk_gateup_b` / `redmetal_topk_down_b`
   over (expert, token) pairs sorted by expert, and `redmetal_topk_sum_b`
   accumulating each token's top-k in router selection order (identical
   summation order to decode and to the CPU oracle). When a chunk's union
   exceeds the plan limit (min(512, half the cache slots)) the chunk is split
   into consecutive token groups;
4. the residual add of the whole chunk, then the LM head for the last token.

Dense matmuls dequantize the weight to f32 once per chunk (`rl_dq_*`, functor
decoders sharing the validated block arithmetic) and run a simdgroup-matrix
GEMM (`rl_gemm_wt`, 32 rows × 32 tokens per threadgroup, f32 accumulate).
Lane-row batched kernels (`rl_rowsb_*`) remain as the fallback for shapes the
GEMM does not cover.

## Validation (`redlite-engine prefill`, regression checks `prefill.parity.*`)

Batched prefill versus the token-by-token Metal engine after a reset, last
token of the sequence, plus the CPU oracle with `--cpu`:

| prompt | chunks | router ids (48 layers) | worst layer abs | logits abs | argmax |
|---|---|---|---:|---:|---|
| 19-token chat prompt | 8 | identical | 1.2e-05 | 3.1e-05 | identical |
| same, CPU oracle vs batched | 8 | identical | — | 1.1e-05 | identical |
| 96 tokens of the long fixture | 32 and 96 | identical | 1.9e-05 | 2.1e-05 | identical |

Against the pinned llama.cpp on the frozen 1200-token fixture (first 1100
positions batched, positions 1100–1199 token by token, `logits.long_context_vs_llama`):
argmax identical at every position; logits max abs 6.2e-05, KL 8.5e-12 with
chunks of 32; `ORACLE LOGITS PARITY: YES` at chunks of 64, 128, 256, 512 and
1100. Greedy generation after batched prefill is token-identical to the
dev18 record and to llama.cpp (`generate.vs_llama`).

## Throughput (M4 Max 48 GiB, prompt tokens per second)

Before the LRU fix below (dev20b as first committed):

| prompt | path | tok/s |
|---|---|---:|
| 1100 tokens, 4 GiB expert cache | token by token (dev18) | 22.6 |
| 1100 tokens, 4 GiB | chunks of 32 / 128 / 256 / 512 | 33.7 / 40.3 / 49.7 / 65.7 |
| 1100 tokens, 4 GiB | one chunk of 1100 | 99.1 |
| 96 tokens, 4 GiB, chunk 96 | experts not resident | 38.1 |
| 96 tokens, 12 GiB, chunk 96 | experts resident (no loads) | 189.2 |

Profiling the expert phase of the 96-token chunk (`expert phase split` line of
`redlite-engine prefill`) showed that copying the misses from the page cache
took 78 ms of wall time for the whole prompt while the LRU reservation took
1807 ms: `rl_native_lru_prepare_many` scanned every cache entry against the
whole selection for each miss (capacity × selection × misses ≈ 4×10⁸ key
compares per 300-expert plan), and key lookups were linear in the capacity.
dev20c reserves all hits before any victim is chosen (so the selection never
needs to be scanned) and indexes resident keys with an open-addressing hash;
both changes preserve the LRU semantics and pass the model-free LRU tests.
After the fix (same machine, `ORACLE LOGITS PARITY: YES` at every size):

| prompt | path | tok/s |
|---|---|---:|
| 96 tokens, 4 GiB, chunk 96 | LRU reserve 31 ms, miss copies 78 ms, experts GPU 312 ms | **155.4** |
| 1100 tokens, 4 GiB | chunks of 128 / 256 / 512 / 1100 | 152.4 / 163.9 / **171.4** / 166.3 |
| 1100 tokens, 12 GiB | chunks of 512 | 163.4 |
| reference: pinned llama.cpp fully resident (`llama-bench` pp48) | | 338.6 |

After dev20c (96 tokens, one chunk, 618 ms): experts GPU 312 ms (3.3 ms/token),
dense GPU 139 ms (1.5 ms/token), miss copies 78 ms, LRU 31 ms, router 18 ms.
The copy path itself (`pread` from the page cache into the pool,
`dispatch_apply` over the misses) is not the bottleneck; the expert kernels are.

### dev20d: expert kernel layout

The batched gate/up and down kernels were re-laid out step by step on the
96-token chunk (experts GPU ms per 48 layers; every variant kept parity):

| variant | experts GPU | note |
|---|---:|---|
| one thread per (expert, row, lane), pairs of the expert in sequence (dev20b) | 312 | baseline |
| eight-pair register tile per thread | 981 | padding to 8 slots with ~3 pairs per expert; 8 scalar activation loads per weight |
| adaptive 4/2/1-pair tiles, float4 activation loads | 370 | activation traffic is not the bound |
| 4-row register tile (one activation load serves 4 rows) | 295 | neither is row reuse |
| grid over pairs instead of experts (4-row tile) | 256 | the bound was load imbalance: 115 of ~300 experts serve one pair, a few serve 15–19, and a thread walked its expert's pairs serially |
| grid over slices of 2 pairs of one expert | 182 | one decoded weight serves both pairs, imbalance bounded |
| **slices of 4 pairs** (`REDMETAL_TOPK_SLICE`, `RM_S`) | **188 / 826 (512-token chunk, was 1361)** | kept: same at 96 tokens, 1.6× better at 512 |

The kernels decode each IQ2_XS / IQ1_M block one 8-value group at a time
(`rm_group8`, the same arithmetic as the validated block dots) for four rows in
lockstep and apply it to up to four tokens with float4 activation loads.

Throughput after dev20d (M4 Max, 4 GiB cache, `ORACLE LOGITS PARITY: YES` at
every size): 96-token chunk 201 tok/s; 512-token chunk 294 tok/s (experts GPU
1.6 ms/token, dense 1.2 ms/token); 1100-token prompt 231 / 249 / 271 tok/s with
chunks of 256 / 512 / 1100 — versus 22.6 tok/s token by token before dev20 and
339 tok/s for the pinned llama.cpp fully resident. The default chunk stays at
512 tokens (scratch ≈ 0.26 MiB per chunk token plus the pair buffers).

Reading the experts in place from the mmap instead of copying them
(`RL_PREFILL_MAPPED_EXPERTS=1`, `redmetal_topk_pool_encode_mapped`) is
numerically identical but 4–10× slower here: Metal re-establishes residency of
each layer's whole ~350 MB expert window for every command buffer (~3.3 s per
48-layer chunk regardless of chunk length). It stays as an opt-in experiment.

## Scope boundary

- Validated on the M4 Max 48 GiB only; the 24 GiB M4 Pro numbers of dev18 were
  not re-measured. On a 24 GiB machine the same code runs with the default
  512-token chunks (scratch ≈ 135 MiB) and the page cache, not the GPU, bounds
  the expert copies.
- Decode is unchanged (one token, deferred expert buffer); the batched kernels
  are used for prompt ingestion only.
- Prompt ingestion is now within 1.4× of llama.cpp's fully resident prefill;
  the remaining split at a 512-token chunk is experts 1.6 ms/token, dense
  1.2 ms/token, router selection 0.2 ms/token. Overlapping the miss copies
  with GPU work is not worth it at ~80 ms per 96-token prompt.
- No batched sampling, no multi-sequence batching.
