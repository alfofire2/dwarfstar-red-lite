# Red Lite dev76 — MLX measured again, one more prefill step, three ideas weighed

Status: done. Apple M4 Max 48 GiB only, 2026-10-08.

## 1. Red Lite and MLX, the same day

dev72 compared the two on different days and display states. This time MLX (`mlx-lm` 0.32.0,
`NexVeridian/Qwen3-Coder-Next-3bit`, 32.5 GB) and Red Lite 0.9.3 (CF2, 18 GiB, every expert resident) ran interleaved,
with 60 s of cooling between them, on the dev72 prompts. The prompts are a 1,738-token file rewrite (`planner.py` of
v0.9.1) and a 33-token prose question. Each tool was warmed first; greedy decoding, 300 tokens, three rounds.

| | Red Lite CF2 | MLX 3-bit |
|---|---:|---:|
| Decode | **99.3 tok/s** (98.96–99.46) | 87–90 tok/s |
| Decode, file rewrite with prompt lookup | **125 tok/s** | — |
| Prompt ingestion, 1,738 tokens | 1,070 tok/s | **1,435 tok/s** |
| Prompt ingestion, 33 tokens | 164 tok/s | **370 tok/s** |
| Memory | 18.2 GiB footprint | 36.8 GB peak |

- **Decode:** Red Lite is now 11–14 % faster than MLX's 3-bit file, with half the memory.
- **Prompt ingestion:** MLX is still a third faster on long prompts and twice as fast on very short ones. Red Lite's
  short-prompt time (0.2 s for 33 tokens) is mostly fixed per-layer work.
- **A trap on the way:** twelve Red Lite runs back to back, right after six MLX runs, fell from 99 to 43 tok/s. The
  machine reported no thermal warning. With 60 s of cooling every run was within 0.3 %. A remote-desktop viewer kept
  the display composited (WindowServer at 60 % CPU) during these runs; both tools ran in that state.

## 2. Prefill: two DeltaNet state rows per simdgroup

The prefill recurrence (`dn_state_seq_sg`, one simdgroup per state row, sequential over the chunk's tokens) now
handles two rows of one head per simdgroup (`dn_state_seq_sg2`). The rows share the k, q, gate and beta loads, and
their two chains interleave. Every row's arithmetic is unchanged: the logits are bit-identical.

- **Stage time:** 89.6 → 74.0 ms for 1,090 tokens. Four rows per simdgroup: 77.4 ms.
- **CF2 prefill:** 1,105 → 1,093 ms (six alternated pairs, all faster).
- `RL_PREFILL_STATE_ROWS=1` restores one row.

## 3. Weighed and not done

- **Gate and up as two passes over 32-pair slices:** slower (WHAT_DID_NOT_WORK). It is the fifth larger-tile layout
  that loses.
- **A chunked (WY) form of the DeltaNet prefill recurrence:** after section 2 the whole recurrence-and-tail stage is
  74 ms of 1,090, so a rewrite could save a few percent at most. It would change the arithmetic and need its own
  validation. Not started.
- **A faster-to-decode expert format:** the decode experts spend their time decoding IQ1_M / IQ2_XS weights.
  - A Red Lite-only type would make the files unreadable by llama.cpp.
  - Standard Q2_K decodes with plain shifts, but is 50 % larger: about 26 GiB of experts, too much for a 24 GiB Mac.
  - Either needs a new quality evaluation; it is a product decision, not a kernel change. Not started.

## Scope boundary

- M4 Max only (the M4 Pro 24 GiB was not available).
- One pair of prompts against MLX, one MLX quantization.
