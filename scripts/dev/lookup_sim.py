#!/usr/bin/env python3
"""dev71: simulate prompt lookup with K drafted tokens per verify on server traces (RL_SERVER_TRACE).

    python3 scripts/dev/lookup_sim.py TRACE [--cost-extra-row 0.35]

Each trace line is "start id,id,...": the request's ids in the engine state at the end (prompt + answer). Speculation
is exact, so the answer is what was generated whatever K is; the simulation replays it. At the pending token q the draft
chain copies the tokens that followed the latest earlier occurrence of h[q-2..q] (then h[q-1..q]), up to K tokens known
at that point. A pass verifies q and the chain and accepts the longest matching prefix r: q advances by r + 1. Without a
draft, a plain step advances by 1. Cost: a plain step 1, a pass with R rows 1 + extra * (R - 1) (verify2 measured 1.35).
"""
from __future__ import annotations

import argparse


def simulate(h: list[int], start: int, k: int, extra: float) -> tuple[int, float, int]:
    """(tokens, cost, passes) for the answer h[start:] with up to k drafts per pass"""
    last3: dict = {}
    last2: dict = {}
    indexed = 0   # occurrences ending at e < indexed are in the maps

    def index_to(q: int) -> None:   # occurrences ending at e <= q - 1
        nonlocal indexed
        while indexed <= q - 1:
            e = indexed
            if e >= 2:
                last3[(h[e - 2], h[e - 1], h[e])] = e
            if e >= 1:
                last2[(h[e - 1], h[e])] = e
            indexed += 1

    q, cost, passes, n = start, 0.0, 0, len(h)
    while q < n:
        index_to(q)
        e = last3.get((h[q - 2], h[q - 1], h[q])) if q >= 2 else None
        if e is None and q >= 1:
            e = last2.get((h[q - 1], h[q]))
        chain = [h[e + j] for j in range(1, k + 1) if e is not None and e + j <= q] if e is not None else []
        if not chain:
            q += 1; cost += 1.0; continue
        r = 0
        while r < len(chain) and q + 1 + r < n and h[q + 1 + r] == chain[r]:
            r += 1
        q += r + 1; cost += 1.0 + extra * len(chain); passes += 1
    return n - start, cost, passes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace")
    ap.add_argument("--cost-extra-row", type=float, default=0.35)
    args = ap.parse_args()
    reqs = []
    for line in open(args.trace):
        s, ids = line.split(" ", 1)
        reqs.append((int(s), [int(x) for x in ids.strip().split(",") if x]))
    print(f"{len(reqs)} requests, {sum(len(h) - s for s, h in reqs)} answer tokens")
    for k in (1, 2, 4, 8):
        tok = cost = 0.0
        for s, h in reqs:
            t, c, _ = simulate(h, s, k, args.cost_extra_row)
            tok += t; cost += c
        print(f"K={k}: {tok / cost:.3f} tokens per plain-step cost (1.0 = plain decoding)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
