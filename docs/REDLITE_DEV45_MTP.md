# Red Lite dev45 — speculative decoding with the Qwen3-Next MTP block

Status: **done** on branch `dev/mtp`. Apple M4 Max 48 GiB, every expert resident. Opt-in with `--mtp FILE` on
`redlite-generate` (one-shot mode).

## Why

At full residency a decode token is GPU-bound: about 12 ms per step, of which matrix weight decoding is a large
part. Checking two tokens in one pass decodes each dense weight once for both. The Qwen3-Next checkpoint has
a multi-token-prediction (MTP) block that drafts the next token. The research survey
(`docs/RESEARCH_2026_10.md`) found:

- that block is distributed separately as `a4lg/Qwen3-Next-80B-A3B-Instruct-MTP-ONLY-GGUF`;
- the current DwarfStar (antirez/ds4) verifies MTP drafts with in-kernel state snapshots and a pointer-swap
  restore.

Our IQ2_XXS / IQ3_XXS GGUFs do not include the block.

## What it is

**The MTP block** (`--mtp FILE`; `redlite download mtp` fetches the Q8_0 file, 2.26 GiB, SHA-256 checked).

- **Contents:** block `blk.48` of the MTP file, with one gated full-attention layer and a 512-expert MoE whose
  experts are Q8_0. It also carries `nextn.eh_proj` / `enorm` / `hnorm` / `shared_head_norm` and its own Q8_0
  token embedding and output head.
- **Input:** `eh_proj · [enorm(embed(t+1)), hnorm(h_t)]`. Here h_t is the trunk's hidden state **after** its
  final norm: drafts are accepted 0.742 of the time, against 0.672 with the pre-norm state (measured).
- **Q8_0 experts:** supported in the shared decoder, bit-identical to the dense Q8_0 dequantizer (offline
  test). All 512 are kept resident in a small pool of their own.
- **KV cache:** the block has its own, filled only at the positions where it runs. Positions are compact
  (0, 1, 2, … per pass). Skipping the prompt changed acceptance from 0.770 to 0.758 and from 0.617 to 0.617,
  so the prompt is never run through it.

**The 2-row verify** (`rl_engine_verify2` / `rl_engine_verify_commit`) runs the pending token u and the draft d
in one pass.

- **Dense products:** two-vector kernels (IQ2_XXS, Q4_K, Q5_K, Q6_K, Q8_0, F32) decode each weight once and
  compute each row in exactly the 1-row kernel's order. Other types fall back to two 1-row products.
- **DeltaNet:** 2-row kernels (`dn_ba_params2`, `dn_conv_shift2`, `dn_qk_l2_2`, `dn_state2`, `dn_tail_norm2`)
  run row 1 from row 0's state. They write the state after row 0 to snapshot buffers from inside the kernels.
- **Attention:** runs per row (row 1 sees row 0's K/V).
- **Experts:** both rows' experts go in one router dispatch (`rl_route2`). The 20 (expert, row) pairs share
  one pool encode, with a two-input gate/up pipeline (function constant), and one tail (`rl_moe_tail2`).
- **Rejection:** a rejected draft is undone by swapping the DeltaNet state pointers with the snapshots.
  Row 1's K/V rows are simply overwritten later.

**Acceptance.** The draft is accepted when the token sampled from u's logits equals d. The output is then exactly
what two ordinary steps give, with any sampler. In greedy mode it is identical, token for token.

## Measurements (M4 Max 48 GiB, full residency, greedy, 256 tokens, alternated runs with 40 s cooling)

IQ2_XXS, the six prompts of `tests/fixtures/cache_trace_prompts.txt`:

| prompt | plain tok/s | `--mtp` tok/s | gain | acceptance | output |
|---|---:|---:|---:|---:|---|
| transformer explanation | 80.77 | 94.26 | +16.7 % | 0.716 | identical |
| Python LRU cache | 81.20 | 102.25 | +25.9 % | 0.841 | identical |
| train arithmetic | 79.88 | 103.84 | +30.0 % | 0.841 | identical |
| Italian short story | 80.45 | 87.28 | +8.5 % | 0.578 | identical |
| fantasy story | 80.60 | 88.64 | +10.0 % | 0.618 | identical |
| 40 sleep tips | 80.61 | 99.87 | +23.9 % | 0.814 | identical |

IQ3_XXS (Python prompt, single pair, its dense types still on the fallback): 75.20 → 92.29 tok/s (+22.7 %),
acceptance 0.868, identical.

Cost per cycle (code prompt): MTP draft 1.44 ms; 2-row verify 19.4 ms in the first version, 17.6 ms with the
DeltaNet kernels, 16.2 ms with the merged experts (one plain step: about 12 ms).

## Tried and kept or not

- Calling the 1-vector dot twice and relying on the compiler to share the decode: 1.74× one vector (not
  shared). Explicit two-vector functions are used.
- Snapshot copies (75 MB per cycle) measured free (19.42 vs 19.39 ms without them); the in-kernel snapshot is
  kept for the dispatch count.

## Validation

GATES_LINE

## Scope boundary

- M4 Max 48 GiB, full residency only. With a bounded cache, verifying two tokens loads the union of their
  experts, so `--mtp` is refused unless every expert is resident (`rl_engine_verify2` checks it). Not run on
  the M4 Pro.
- One draft token per pass. Deeper chains (DS4's 3-row verify) are not implemented.
- `redlite-generate` one-shot mode only. Interactive mode and the server do not use it yet.
- The MTP head file is for Qwen3-Next-80B-A3B-Instruct. With Qwen3-Coder-Next (which has no MTP block) the
  output stays exact, but acceptance was not measured.
- IQ3-family two-vector kernels are not written (the IQ3_XXS file runs those products twice).
