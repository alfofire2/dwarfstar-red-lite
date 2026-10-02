# Red Lite dev43 — session state checkpoints on disk

Status: **done** on branch `dev/state-cache` (from `main` at 186b827, release 0.4.1 plus docs). Apple M4 Max
48 GiB only.

## Why

Qwen3-Next's session state is small:
- 36 DeltaNet layers, each with a 2 MiB recurrent state plus its conv history (72 MiB in all);
- 12 attention layers' K/V rows, 48 KiB per token.

Re-running a long prompt is not small: about 4.6 s for 4200 tokens on this Mac. The current DwarfStar keeps
prefix checkpoints on disk, keyed by the prompt prefix ("KV cache as a disk citizen", `ds4_kvstore.[ch]`,
see `docs/RESEARCH_2026_10.md`). dev43 adapts that idea.

## Design

- `rl_engine_state_write` / `rl_engine_state_read` (GPU backend) copy the shared Metal buffers: conv and
  recurrent states, plus the first `position` K/V rows of each attention layer.
- `redlite_native_statecache.[ch]` frames a file as follows:
  - **Header:** model tag (GGUF size + shape), build tag, prefill chunk, number of ids, vocabulary,
    state bytes.
  - **Body:** the prompt ids, the last token's logits, then the state.
- `rl_statecache_prefill`:
  1. Restore the longest stored prefix of the prompt, or reset if there is none.
  2. Prefill up to the largest multiple of the prefill chunk (2048) and store a checkpoint there.
  3. Prefill the rest.
- **Why only multiples of the chunk.** A restored session ingests the rest with the same chunk boundaries as
  a cold run. Its state is therefore the cold run's state.
- **What is ignored.** Files from another model, another build or another chunk size. Files that cannot be
  read are deleted. The directory is trimmed to `--state-max-mib` (default 8192), least recently used
  first.
- **Where it is used:**
  - `redlite-generate --state-dir DIR` (one-shot mode);
  - `redlite-server --state-dir DIR`, used when the in-memory reuse of dev29 does not apply (first
    request, a different conversation, after a restart).

## Measurements (IQ2_XXS, full residency, `--context 8192`)

| case | prompt | prefill / TTFT | restored |
|---|---:|---:|---:|
| `redlite-generate`, cold | 4209 | 4602.5 ms | 0 |
| first run with `--state-dir` (stores 4096) | 4209 | 4730.4 ms | 0 |
| second run | 4209 | **387.4 ms** | 4096 |
| different question, same first 4096 tokens | 4207 | 4610.4 → **377.9 ms** | 4096 |
| `redlite-server`, first request | 4212 | TTFT 4928 ms | 0 |
| `redlite-server` after a restart | 4212 | TTFT **556 ms** | 4096 |

Greedy outputs are identical in every pair (32 tokens). The checkpoint at 4096 tokens is 268 MiB on disk;
storing it adds about 0.13 s.

## Bug found and fixed (dev37)

With a 2 GiB cache and prompts long enough to fill a 2048-token chunk, prefill failed with
`prefill expert prefetch failed: native expert load failed`.
- **Cause.** dev37's size classes left the class of the 11 IQ2_XS layers about 660 slots. One chunk needs up
  to 512 experts per layer while the next layer's 512 are prefetched. The uniform pool of 0.4.0 had 2361
  slots.
- **Fix.** Size classes are used only when every class holds at least 2 × 512 + 64 slots; otherwise the
  slots are uniform as before. At the default 4 GiB, IQ2_XXS keeps its classes (smallest 1331). The prefill
  prefetch also checks the capacity of the next layer's class, counting the current layer's pinned experts
  only when they share it.
- **Coverage.** The default 4 GiB path was not affected. The new regression check `server.state_restart`
  runs a 4212-token prompt with a 2 GiB cache.

## Validation

`rm -rf .deps/redmetal`, `make redmetal`, `make native`, `make sanitize`: 0 warnings; `make test` and ruff green; `scripts/regress_m4.sh`: IQ2_XXS **50/50** (new `server.state_restart`), IQ3_XXS **37 passed, 0 failed, 13 skipped**; `quick_parity.sh --long --batch 2048` on both models: PASS.

## Scope boundary

- M4 Max 48 GiB only. On a 24 GiB Mac the files compete with the page cache like any other file I/O. Not
  measured there.
- GPU backend only. Interactive `redlite-generate -i` does not use it (the server does).
- Checkpoints are only made at multiples of the prefill chunk (2048 tokens by default). Shorter prompts are
  never stored.
- A rebuild invalidates every stored file: kernels decide the float rounding, so states are only
  comparable within one build.
