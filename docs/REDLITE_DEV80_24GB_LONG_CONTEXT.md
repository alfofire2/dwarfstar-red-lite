# Red Lite dev80: long contexts on the 24 GiB Mac

Status: done. Apple M4 Pro 24 GiB (AC, GPU limit 21,741 MiB) and Apple M4 Max 48 GiB, 2026-10-09, Red Lite 0.9.5.

## 1. Coding agents with every expert resident

dev77 ran the hard agent suite on the M4 Pro with the 4 GiB expert cache (64K window), where 64K and every expert do
not fit together. After `redlite gpu-limit`, `redlite serve --native --context 65536 --kv f16` keeps every expert
resident: the half KV cache takes 1.5 GiB at 64K instead of 3. The float cache would need 22,077 MiB against the
21,741 MiB limit.

Same suite, limits (900 s per task) and temperature (0.3), pi on the M4 Max, CF2:

| Setup | Runs | Decode | Prefill | Tasks |
|---|---:|---:|---:|---:|
| 4 GiB cache, float KV (dev77, 0.9.3 / 0.9.4) | 4 | 16.9–24.8 tok/s | 139–233 tok/s | 6, 5, 5, 5 of 6 |
| Every expert resident, `--kv f16` (0.9.5) | 2 | 35.4 / 50.7 tok/s | 218 / 307 tok/s | 5, 5 of 6 |

- **Run 1** took 15 minutes for the six tasks (its `two_bugs` failed its check). The 4 GiB runs took 37–56 minutes.
- **Run 2** was slower: `two_bugs` passed after 795 s and 64 requests, and `debug_session` hit the 900 s limit after
  79 requests. The second run of a session was also the slower one in dev77.
- **The server now says so.** When the float cache is what keeps full residency out at the requested context and
  the half one would fit, `redlite serve --native` and `redlite chat` print a one-line tip for `--kv f16`. It is not
  made the default: dev74 measured logit drift for the half cache on CF2 (KL up to 5.8e-2 at 1.1K positions).

## 2. Decode as the context grows

`decode-bench`, CF2, M4 Pro, ms per token:

| Position | 4 GiB cache, float KV | Every expert resident, half KV | Every expert resident, float KV |
|---:|---:|---:|---:|
| 64 | 45.1 | 16.5 | — |
| 16K | 44.5 | 18.9 | 19.6 |
| 32K | 50.2 | 21.7 | 23.5 |
| 48K | 53.2 | 23.8 | — |

- **Every expert resident:** decode falls from 60.6 to 42.0 tok/s by 48K. The extra time is attention:
  - with the float cache it reads ~230 GB/s at 32K, close to the M4 Pro's bandwidth;
  - with the half cache it reads half the bytes but only reaches ~150 GB/s, so it is 8 % faster, not twice.
  - A faster half-precision decode attention kernel could give about 10 % at 32K, for `--kv f16` only. Not attempted.
- **4 GiB cache:** expert reads dominate. These runs came right after the full-residency agent runs had emptied the
  macOS file cache, so most reads went to the SSD: 45 ms per token at position 64, against 27 ms the day before with
  the file cache warm. Attention adds about the same 8 ms at 48K as with full residency.
- **The 15.6 tok/s of `two_bugs` in dev77** therefore came from reads more than from attention:
  - expert misses on a cold file cache;
  - attention at 20–45K;
  - the slowdown of a long session.

## 3. Prefill attention at long contexts (M4 Max)

At 32K, attention is about 44 % of the prefill (12 attention layers at 48.7 ms against 48 expert layers at 15.4 ms,
per 512 tokens). Three kernel rewrites gave nothing; they are in `WHAT_DID_NOT_WORK.md`. The next step there is the
per-kernel GPU counters, not another rewrite.

## Scope boundary

- **Agent runs:** two runs per setup on one suite, one Mac; sessions at temperature 0.3 differ from run to run.
- **Positions:** `decode-bench` jumps to the position without a prefill (random state), so it times the kernels, not
  the quality.
- **The `--kv f16` tip:** it changes no default.
