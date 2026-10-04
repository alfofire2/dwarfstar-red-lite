# Red Lite dev61 — the 24 GiB Mac without a raised GPU limit: agents and the IQ3_XXS file from the SSD

Status: done. Measured on the Apple M4 Pro 24 GiB (macOS 26.7) at the default GPU limit, 2026-10-04.

## Why

A Red Lite user who never raises the GPU limit (it needs `sudo`) runs with the bounded expert cache.
- **What stays in memory:** the dense weights and a 4 GiB LRU of experts.
- **What the SSD serves:** every other expert a token needs is read from the file.

dev60 measured coding agents only with every expert resident. This milestone measures that default user, with pi
installed on the same Mac. It also tries the 29.6 GiB IQ3_XXS file, which cannot be resident on 24 GiB at all.

## pi on the 24 GiB Mac

- **Install and configure:**
  - `brew install node`;
  - `npm install -g @earendil-works/pi-coding-agent` (pi 1.0.2);
  - `redlite setup-pi --port 8091`, which writes the `redlite` provider into `~/.pi/agent/models.json`, keeps
    other providers and keeps the file private (0600; it may hold other providers' API keys).
- **Server:** `redlite serve --native --context 32768 --port 8091`. At the default GPU limit the planner picks
  the 4 GiB cache with cache-aware routing. The engine is ready in 0.1 s: no experts are preloaded.
- **Run:** `scripts/dev/agent_eval.py --agent-dir ~/.pi/agent`, with the five tasks of dev60, three runs per
  model, agent and server on the same Mac.

| M4 Pro 24 GiB | passed | new_code | bug_fix | rename | read_answer | add_flag | sum of medians | prompt reused |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| F2, every expert resident (dev60) | 15/15 | 21.6 s | 26.3 s | 29.2 s | 5.8 s | 16.0 s | 99 s | 57 % |
| **F2, 4 GiB cache** | **15/15** | 21.4 s | 27.1 s | 43.4 s | 6.7 s | 20.6 s | 119 s | 76 % |
| Qwen3-Coder-Next, every expert resident (dev60) | 14/15 | 21.0 s | 15.5 s | 18.6 s | 5.8 s | 13.2 s | 74 s | 83 % |
| **Qwen3-Coder-Next, 4 GiB cache** | **15/15** | 20.9 s | 29.2 s | 40.4 s | 7.0 s | 14.7 s | 112 s | 84 % |

- **The tasks pass as often from the SSD as resident:** 30 of 30. With a 4 GiB cache the server uses about
  5 GiB instead of 18–20 GiB, so the agent and other applications have room.
- **They take longer:** +20 % (F2) and +51 % (Qwen3-Coder-Next) over the five tasks. Most of it is in the
  multi-step ones: `rename` 29 → 43 s and 19 → 40 s.
- **The cost of the resident runs:** a raised GPU limit and 12–20 s of preloading at every server start.

## IQ3_XXS from the SSD

Bartowski's IQ3_XXS (31.7 GB, 29.6 GiB) is the file 48 GiB Macs keep resident: perplexity 14.29, against 15.38 for
F2 and 16.47 for Bartowski's IQ2_XXS. On a 24 GiB Mac it cannot be resident, but the bounded cache streams it.

**Decode.** `small_mac_ab.sh` measured the six cache-trace prompts × 256 tokens, with cache-aware routing (λ 0.5,
the product default with a bounded cache) and 30 s of cooling before each prompt.

| M4 Pro 24 GiB, default GPU limit | cache | experts read per token | MiB read per token | decode (median of 6) |
|---|---:|---:|---:|---:|
| F2 | 4 GiB | 29.9 | 40.7 | 34.0 tok/s |
| **IQ3_XXS** | 4 GiB | 46.4 | 107.4 | **29.3 tok/s** |
| IQ3_XXS | 8 GiB | 17.6 | 40.5 | 27.9 tok/s |
| IQ3_XXS | 12 GiB | 12.1 | 27.6 | 28.1 tok/s |

**Prompt ingestion** (`bench_m4.sh --only prefill4`, 4 GiB cache, chunks of 2048, median of 2):

| | 1,100 tokens | 8,192 tokens |
|---|---:|---:|
| F2 | 350 tok/s | 367 tok/s |
| IQ3_XXS | 208 tok/s | 334 tok/s |

- **The better file costs only 14 % of decode speed.**
  - **Reads:** IQ3_XXS reads 2.6× more from the SSD per token (107 MiB), yet decodes at 29.3 tok/s against 34.0.
  - **Why:** as in dev51, the bounded-cache path is bound by its per-layer CPU-GPU round trip, not by the reads.
  - **Prompts:** long prompts amortize the loads (−9 % at 8,192 tokens); short ones cost more (−41 % at 1,100).
- **A bigger cache does not help.** 8 or 12 GiB cut the reads 2.6–3.8× but decode at 27.9–28.1 tok/s, slightly
  slower than 4 GiB. The 4 GiB default stays.
- **Memory:** swap did not move during the runs (751 MB in use throughout, from before).

**What a 24 GiB user can choose** (M4 Pro):

| | perplexity | decode | needs |
|---|---:|---:|---|
| F2, every expert resident | 15.38 | 44.5 tok/s (49–58 with MTP) | the raised GPU limit; ~18–20 GiB in use |
| F2, 4 GiB cache | 15.38 | 34.0 tok/s | nothing |
| **IQ3_XXS, 4 GiB cache** | **14.29** | 29.3 tok/s | nothing; 31.7 GB of disk |

`redlite chat` keeps choosing F2 on 24 GiB; IQ3_XXS is a choice: `redlite download 48gb`, then
`redlite chat PATH/Qwen_Qwen3-Next-80B-A3B-Instruct-IQ3_XXS.gguf`.

## Scope boundary

- Same five tasks as dev60, three runs each: a check that the loop works on the default configuration, not a
  coding benchmark.
- **Agent runs:** 759 MB of swap in use after both, not separated from earlier resident runs on the same boot.
- **IQ3_XXS quality** is the M4 Max measurement (exact routing). With a bounded cache the cache-aware routing
  changes some expert choices. dev46 found no perplexity change for IQ2_XXS; IQ3_XXS with the bias was not
  measured.
- **IQ3_XXS with the coding agent** was not run.
- **Single machine:** only the M4 Pro; single runs of six prompts per cache size.
