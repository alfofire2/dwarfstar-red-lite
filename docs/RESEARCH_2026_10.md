# Research notes, October 2026: where Red Lite can still improve

A literature and source survey done on 2026-10-01, after release 0.4.1. It covers the current DwarfStar
(antirez/ds4), newer Qwen models and MTP, llama.cpp / MLX status on Apple Silicon, and recent MoE /
Metal techniques. Everything here is **somebody else's measurement** unless it says "Red Lite". Nothing
in this file is validated on our machines.

## 1. DwarfStar (antirez/ds4) today

Read at HEAD `0aaea5a` (2026-09-20).

- **Qwen3-Next-80B is not supported by ds4.** Its Qwen model is Qwen3.8-Flash-Next (`qwen4exp`): 125B main
  parameters plus 51B n-gram embeddings plus a 4B MTP module. It has hidden 2560, hyper-connections, sparse
  attention and a vision tower. The Q2 profile keeps 41.73 GiB resident and is meant for 64 GB Macs.
  ds4 does not stream its experts from SSD. **Not runnable on 24 or 48 GiB.**
  Sources: `docs/QWEN38_FLASH_NEXT.md`, `docs/MODELS.md`, https://github.com/antirez/ds4.
- **MTP speculation for Qwen** (`docs/SPECULATIVE_DECODING.md`, `ds4.c`
  `ds4_session_qwen4_spec_cycle` / `qwen4_spec_depth`):
  - **Draft depth.** It drafts one token. A second, chained draft is added only while the rolling acceptance
    of the first draft is perfect (8 of the last 8). It disengages when fewer than 6 of the last 8 are
    accepted, or after 2 second-draft rejections in a row.
  - **Verify.** The verify is a 2- or 3-row forward on the *decode* kernels ("few-row matvec"). Commit
    891c005 reports C=1 50.5 → 55.6 tok/s against 51.3 plain. On M3 Ultra, generation went from about
    54.7 tok/s to 64.8 / 90.5 / 68.9 tok/s with MTP.
  - **Recurrent state.** The GDN scan kernel writes the state snapshot *inside the kernel*, after verify
    rows 0 and 1. A rejected draft restores it by **swapping buffer pointers**, not by copying or
    replaying. The KV rows of rejected tokens are simply overwritten.
  - **Sampling.** Exact sampling (`--mtp-exact-sampling`) treats the draft as a point-mass proposal.
- **Disk KV checkpoints.** They are keyed by the SHA1 of the rendered byte prefix. Snapshots are aligned to
  2048-token prefill chunks, from at least 512 tokens. For Qwen the payload is the token ids, the last
  logits, and per layer either the GDN state + conv history or the K/V rows (`ds4_kvstore.[ch]`).
- **Metal.**
  - The router, top-k and shared-gate run in one kernel.
  - The shared expert runs as an extra slot of the MoE kernel.
  - Up to four projections are computed per dispatch.
  - The weight type is a function constant.
  - The command buffer is flushed early (after layer 2) while the host encodes the rest.
- **SSD streaming (DeepSeek/GLM only).**
  - Experts are read with `pread` into Metal buffers by 9–18 threads.
  - The cache tracks per-expert hotness, halved every 16 tokens.
  - A bounded prefill bonus fixed a 10× decode collapse (commit b6af0ad).
- **A possible trap.** ds4's GDN kernel pairs value head `h` with key head `h % Hk` for *its* tiled GGUF
  order. Red Lite's `h / (Hv/Hk)` is right for the Qwen3-Next GGUFs (dev18). Do not copy theirs.

## 2. Models

- **Qwen3-Coder-Next** (`Qwen/Qwen3-Coder-Next`, Apache-2.0, 2026-02-03) is a drop-in. Its config equals
  Qwen3-Next-80B-A3B except `rope_theta` 5e6 (instead of 1e7), and it has no MTP tensors.
  - It is non-thinking, with SWE-bench Verified 70.6.
  - `bartowski/Qwen_Qwen3-Coder-Next-GGUF` uses the same sizes as our files: IQ2_XXS 17.97 GiB,
    IQ3_XXS 29.55 GiB.
- **Qwen3.5/3.6-35B-A3B** have 40 layers and 256 experts, so they are a different shape. Qwen3.8 open
  weights are 27B dense and 2.4T-A95B. There is no other Qwen "Next" model of our shape.
- **Bartowski quants of Qwen3-Next-80B-A3B-Instruct** (GiB): IQ1_M 16.11, IQ2_XXS 17.97 (ours), IQ2_XS
  20.69, IQ2_S 21.76, IQ2_M 24.31, IQ3_XXS 29.55 (ours), IQ3_XS 30.76, IQ4_XS 39.91.
  - ilintar's perplexity (dataset not stated): IQ2_XXS 10.25, IQ2_M 9.11, IQ3_XS 8.33, Q8_0 8.15.

## 3. Qwen3-Next MTP

- **What it is.** One full-attention decoder layer with its own 512-expert MoE (`mtp.layers.0.*`), plus an
  `fc` that maps 2×2048 → 2048 from the normed embedding of token t+1 and the hidden state h_t. It shares
  `embed_tokens` and the LM head with the trunk.
- **Our files.** Bartowski's and Unsloth's GGUFs drop it.
- **MTP-only GGUFs.**
  - `a4lg/Qwen3-Next-80B-A3B-Instruct-MTP-ONLY-GGUF`: Q4_K_M 1.40, Q6_K 1.75, Q8_0 2.26 GiB.
  - `yomaytk/...-MTP-HEAD-GGUF`: Q8_0 2.26 GiB.
- **llama.cpp support.** PR #25589 (merged 2026-08-03) adds Qwen3-Next MTP (`--spec-type draft-mtp`).
  Reported on an M5 Max (Q4_K_M): about 91 → 109 tok/s (+20 %). Acceptance was 0.66–0.99, and 0.93–1.00
  with `--spec-draft-n-max 2 --spec-draft-p-min 0.6` at temperature 0.
- **State rollback.** Speculation on a Qwen3-Next-family model only paid off once the DeltaNet/conv
  snapshots stayed on the device (llama.cpp PR #28118, Strix Halo). Before that change it was a 5× loss
  (6.2 t/s). After it: 41.5 t/s at n=3 against 32.4 t/s without drafts.

## 4. Techniques with evidence

| Technique | Evidence | Relevance to Red Lite |
|---|---|---|
| Narrow-row IQ mul_mv (several lanes per 32-weight chunk when rows have < 1024 columns) | llama.cpp #28692: −37 % kernel time at k=256/512, +3.3 % tg (M5) | **Already done** for the 512-column expert down rows (dev38: 75.3 → 82.6 tok/s) |
| MoE fusions (top-k, weighted reduce, RMS_NORM+SCALE, SSM_CONV+SiLU) | llama.cpp #28948: +12–16 % tg on qwen35moe; #29134: −6 % elsewhere from lost concurrency | Mostly done (GPU routing dev21, tail fusion dev39); our single-threadgroup fusion lost 6 % (dev39) |
| Fused launches / "megakernel" segments | scratchy #150: −7.3 % ms/token at 986 → 405 launches; persistent kernels are unsafe on Metal | Realistic ceiling ~5–8 % |
| Chunk-parallel DeltaNet (WY form, chunk 64, simdgroup matrices) | MLX #4020: 1.3–1.45× over sequential on M1 Max / M5 Max, bitwise-equal perplexity | Prefill recurrence (`dn_state_seq_sg`) |
| q/k L2 norm as `x * rsqrt(Σx² + eps)` | llama.cpp #28068 (2026-09-06), affects qwen3next | Our pin predates it; parity reference question, not speed |
| Parallel `pread`, page cache instead of a custom pool, no mmap of cold experts, no `dispatch_io`; prefetch can slow the GPU | Flash-MoE (Qwen3.5-397B, M3 Max 48 GB): mmap 5× slower, `dispatch_io` −70 %, dropping the custom cache +38 %, speculative prefetch −38 % | 24 GiB mode: must be A/B'd **on the M4 Pro** |
| Cache-aware routing bias (`z + λ·Δ·cached`) | arXiv 2412.00099: >50 % fewer misses, PPL +0.1–3 % (8–64-expert models) | 24 GiB mode, opt-in only (changes outputs) |
| Model-free drafts (prompt lookup, SuffixDecoding) | arXiv 2411.04975: up to 5.3× on agentic/code, small on chat | Cheapest draft source; needs the multi-row verify path |
| Untrained reduced-top-k self-drafts | DraftExpert 2607.24434: acceptance 22–46 % | Weak; not worth it |

## 5. Ranked options for Red Lite (Red Lite's assessment, not measured yet)

1. **Speculative decoding with the Qwen3-Next MTP head.** Load the MTP layer from an MTP-only GGUF (about
   1.4–2.3 GiB). Verify 2–3 rows with few-row decode kernels and snapshot the DeltaNet/conv state inside
   the kernels, with a pointer-swap restore (the ds4 design). Gate the draft depth on rolling acceptance.
   Reference gains: +20 % (llama.cpp, M5 Max) to +65 % on deterministic text (ds4). It needs a new graph
   stage (one attention + MoE layer), multi-row decode kernels and the snapshot logic, and the GGUF
   download. Prompt-lookup drafts can reuse the same verify path.
2. **Qwen3-Coder-Next.** The same engine plus `rope_theta` from the GGUF, at the same sizes as our files.
   It needs a download and the llama.cpp oracle checks.
3. **Disk checkpoints of the session state**, keyed by a prompt-prefix hash: the 36 DeltaNet states, the
   conv history, and the 12 attention layers' K/V. Restarts and long system prompts no longer need a
   re-prefill.
4. **Chunk-parallel DeltaNet prefill** (MLX #4020 design). The gain is bounded by the recurrence's share
   of prefill.
5. **24 GiB mode, on the M4 Pro:**
   - A/B the expert I/O (pread threads vs the current loads, page cache vs pool, prefetch on/off).
   - Then the opt-in cache-aware routing with a perplexity gate.

## 6. antirez on DwarfStar (video, 2026-10-02)

Source: "IL VERO FUTURO DELL'AI? LLM LOCALI, STATI UNITI e CINA con Salvatore Sanfilippo" (TechDale, 2026-10-02,
1:28:35, https://youtu.be/d0p_AviYH7M); the technical part is minutes 17-63. Everything below is **what was said**,
read from YouTube's automatic Italian subtitles. Nothing here was measured by Red Lite, and numbers heard through
the auto-transcript may be wrong.

**How ds4 is built and checked**
- **Quality reference: the producer's own API.** antirez buys tokens from the model maker (DeepSeek), asks about
  1000 questions, and checks that ds4's answers stay close to the API's, "word for word" (min 24-25). Red Lite's
  reference is the pinned llama.cpp, which shares the same quantized file. A check against the maker's API would
  measure what llama.cpp cannot: the quantization's distance from the real model.
- **Few model families, on purpose.** Other engines chase every new model and leave the old ones broken or slow.
  ds4 supports three families, which lets it regroup the multiplications to saturate the GPU (min 21, 25).
- **Small codebase as a feature.** A few hundred thousand lines or less, so a user's coding agent can read it and
  change it for their own need (min 26). Red Lite's narrow scope serves the same goal.
- **The engine author should own the quantization.** Who quantizes, who writes the engine and who writes the
  agents are different people who do not talk to each other (min 20). ds4 controls all three.
- Old NVIDIA cards (A100, L40S) that vLLM/SGLang no longer support (min 26-27); distributed inference over an RDMA
  cable between two M5 Max laptops (min 47). Neither applies to Red Lite.

**Steering (min 27-33)**
- **Single-direction vector steering**, from the public "single direction" refusal paper. ds4 loads a vector
  (antirez released one for DeepSeek V4 Flash), and `/steer` changes its strength during an interactive chat.
- **Steer only the start of the answer.** Apply the vector at the start and then remove it. The model is
  autoregressive: once it has started answering it keeps going. A permanently "abliterated" file is made
  "dumber" by the fixed strength (his claim).
- **Prefilled conversation:** a hand-written text file of user/assistant turns that the model reads as its own
  past, with no steering.
- His stance: the technique is public and easy to add elsewhere; most engines leave it out for political
  reasons; he calls it experimental.

**Speeds quoted and his usability thresholds (min 52-62)**

| | quoted |
|---|---|
| decode | 20 tok/s "usable", 10 boring, 60/100/200 good |
| reading (prefill) | 500-600 tok/s starts being "interesting", 1000-2000 "very usable" |
| DGX Station (~120 k EUR), DeepSeek V4 Flash | prefill 23,000 tok/s, decode 250 tok/s (one session) |
| DGX Spark, DeepSeek V4 Flash | prefill ~1000 tok/s, decode ~20 tok/s (heard as "vendi" = venti) |
| M5 Max 128 GB | fully resident model ~40 tok/s; DeepSeek 4.1 (about twice V4, "engram" embeddings) streamed from SSD ~20 tok/s; Kimi K3 streamed ~3 tok/s |
| M3 Max | about half the inference speed of an M5 Max |

- **Prefill matters most for agents.** A coding agent mostly *reads*: tool outputs, files, grep results (min 51).
- **SSD streaming is for the occasional bigger model.** Buy the memory the everyday model needs fully resident
  (min 59-60).
- **Serving many users** does not divide the single-session speed: concurrent sessions share the loaded parts of
  the model (min 53).
- **Lower power on laptops.** Running a MacBook at lower power keeps it cooler and makes it last. macOS only
  offers automatic / low / high (min 55).
- ds4 has users in production: companies with sensitive data doing coding and security work with ds4 +
  DeepSeek (min 41). ds4 can also drive a logged-in Chrome for web searches (min 44).

**What it means for Red Lite (assessment, not measured)**
1. **24 GiB prompt ingestion is below his "interesting" line.** The M4 Pro reads 280-350 tok/s with a 4 GiB
   cache (dev47); the M4 Max reads ~900 at full residency. Agents mostly read, so 24 GiB prefill is the most
   useful next target.
2. **A quality check against Qwen's own API** (DashScope or another host of Qwen3-Next-80B-A3B-Instruct):
   - a fixed question set, greedy, with top logprobs where available;
   - measures how far IQ2_XXS and IQ3_XXS are from the full-precision model;
   - complements the llama.cpp oracle;
   - needs an API key and a small token budget.
3. **Optional steering.** A vector added to the residual stream at chosen layers, with strength changeable in
   chat and an "only the first N tokens" mode. Separately, a `--history FILE` of prefilled turns. Cheap in the
   engine; whether to ship it is the project owner's call.
4. **Low-power mode on the M4 Pro.** Measure tok/s and power with low-power mode on; a cheap experiment for
   laptops.
5. **Batching concurrent server requests** so they share expert loads; the server is FIFO, one request at a
   time.

## Sources

DwarfStar: https://github.com/antirez/ds4 (docs/QWEN38_FLASH_NEXT.md, docs/SPECULATIVE_DECODING.md,
docs/SSD_STREAMING.md, docs/MODELS.md), https://dwarfstar.sh/ ·
Qwen: https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct, https://huggingface.co/Qwen/Qwen3-Coder-Next,
https://huggingface.co/Qwen/Qwen3.8-Flash-Next, https://github.com/QwenLM/Qwen3.8 ·
MTP GGUFs: https://huggingface.co/a4lg/Qwen3-Next-80B-A3B-Instruct-MTP-ONLY-GGUF,
https://huggingface.co/yomaytk/Qwen3-Next-80B-A3B-Instruct-MTP-HEAD-GGUF ·
llama.cpp: PRs #25589, #28118, #28692, #28948, #29134, #28068, #20361, #19375 (github.com/ggml-org/llama.cpp) ·
MLX: https://github.com/ml-explore/mlx/pull/4020 ·
Flash-MoE: https://github.com/danveloper/flash-moe · scratchy: https://github.com/AI-native-Systems-Research/scratchy/pull/150 ·
Papers: arXiv 2412.00099 (cache-aware routing), 2411.04975 (SuffixDecoding), 2607.24434 (DraftExpert),
2505.19645 (MoESD), 2607.12696 (EcoSpec), 2607.00501 (BaseRT) ·
Quants: https://huggingface.co/bartowski/Qwen_Qwen3-Next-80B-A3B-Instruct-GGUF,
https://huggingface.co/ilintar/Qwen3-Next-80B-A3B-Instruct-GGUF.
