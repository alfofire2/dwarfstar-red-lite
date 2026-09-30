#!/usr/bin/env python3
"""Sampler distributional parity: native redlite-sampler-dist vs the pinned llama.cpp chain.

    python3 scripts/dev/compare_sampler.py [--bin .deps/redmetal] [--draws 40000]

Model-free. For several synthetic logit vectors over the Qwen3-Next vocabulary (151 936)
and a grid of temperature / top-k / top-p / min-p it checks:
  1. the final candidate set is identical (same ids) and every probability agrees within
     --prob-tol (both sides compute in float32). Exception, reported as "tie": when tokens
     with exactly equal logits straddle a cut (top-k boundary, nucleus cut, greedy mode),
     llama.cpp keeps whichever ids std::partial_sort leaves in front (implementation-
     defined) while the native sampler keeps the lowest ids. Then the sorted probability
     lists must still agree and every differing id must carry the boundary logit;
  2. N seeded draws from each implementation fit the exact distribution (chi-square with
     Wilson-Hilferty, plus an exact Poisson tail for the pooled rare tokens; p-value >
     --min-pvalue). The PRNGs differ (native splitmix64,
     llama.cpp mt19937), so individual draws are not compared.
Prints "SAMPLER PARITY: YES" on success.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import random
import struct
import subprocess
import sys
import tempfile

VOCAB = 151936


def make_logits(kind: str, seed: int) -> list[float]:
    r = random.Random(seed)
    v = [r.gauss(0.0, 2.5) for _ in range(VOCAB)]
    if kind == "peaked":        # one dominant token (typical confident step)
        v[r.randrange(VOCAB)] = 25.0
    elif kind == "top3":        # a few strong candidates
        for j, x in enumerate((18.0, 17.5, 16.2)):
            v[1000 * (j + 1) + seed] = x
    elif kind == "flat":        # many plausible tokens (creative step)
        for _ in range(300):
            v[r.randrange(VOCAB)] = r.uniform(9.0, 12.0)
    elif kind == "ties":        # exact ties around the cut-offs
        for j in range(60):
            v[500 + j] = 12.0 if j % 2 else 11.0
    return v


def tie_equivalent(a: dict[int, float], b: dict[int, float], logits: list[float], tol: float) -> bool:
    """Same distribution up to the identity of equally-scored tokens at the cut."""
    if len(a) != len(b):
        return False
    pa, pb = sorted(a.values(), reverse=True), sorted(b.values(), reverse=True)
    if any(abs(x - y) > tol for x, y in zip(pa, pb)):
        return False
    boundary = min(logits[i] for i in a)
    return all(logits[i] == boundary for i in set(a) ^ set(b))


def run(binary: Path, logits: Path, params: list[str], extra: list[str] | None = None) -> dict[int, float]:
    out = subprocess.run([str(binary), str(logits), *params, *(extra or [])], capture_output=True, text=True, check=True).stdout
    res: dict[int, float] = {}
    for line in out.splitlines():
        a, b = line.split()
        res[int(a)] = float(b)
    return res


def poisson_upper_tail(k: float, lam: float) -> float:
    """P(X >= k) for X ~ Poisson(lam)."""
    if k <= 0:
        return 1.0
    term, cdf = math.exp(-lam), 0.0
    for i in range(int(k)):
        cdf += term
        term *= lam / (i + 1)
    return max(0.0, 1.0 - cdf)


def chi2_pvalue(observed: dict[int, float], expected_p: dict[int, float], n: int) -> float:
    """Goodness of fit of the draws to the exact distribution.

    Cells with an expected count >= 5 get a chi-square test (Wilson-Hilferty). The rare
    tokens are pooled; if the pool still expects fewer than 5 draws, the chi-square
    approximation is invalid for it, so it is folded into the smallest regular cell and
    its own count is checked with an exact Poisson upper tail instead. The smaller of the
    two p-values is returned.
    """
    if any(t not in expected_p for t in observed):
        return 0.0  # drew a token outside the support
    cells_o, cells_e, rest_o, rest_e = [], [], 0.0, 0.0
    for tok, p in expected_p.items():
        e = p * n
        o = observed.get(tok, 0.0)
        if e < 5.0:
            rest_o += o
            rest_e += e
        else:
            cells_o.append(o)
            cells_e.append(e)
    p_rare = 1.0
    if rest_e >= 5.0 or (rest_e > 0 and not cells_e):
        cells_o.append(rest_o)
        cells_e.append(rest_e)
    elif rest_e > 0:
        p_rare = poisson_upper_tail(rest_o, rest_e)
        j = min(range(len(cells_e)), key=cells_e.__getitem__)
        cells_o[j] += rest_o
        cells_e[j] += rest_e
    dof = len(cells_e) - 1
    if dof <= 0:
        return p_rare
    x2 = sum((o - e) ** 2 / e for o, e in zip(cells_o, cells_e))
    z = ((x2 / dof) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * dof))) / math.sqrt(2.0 / (9.0 * dof))
    return min(p_rare, 0.5 * math.erfc(z / math.sqrt(2.0)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=str(Path(__file__).resolve().parents[2] / ".deps" / "redmetal"))
    ap.add_argument("--draws", type=int, default=40000)
    ap.add_argument("--prob-tol", type=float, default=1e-6)
    ap.add_argument("--min-pvalue", type=float, default=1e-4)
    ap.add_argument("--kinds", default="peaked,top3,flat,ties", help="comma-separated logit fixtures")
    args = ap.parse_args()
    native = Path(args.bin) / "redlite-sampler-dist"
    ref = Path(args.bin) / "redlite-ref-sampler"

    grid = [
        # temperature, top_k, top_p, min_p
        (0.7, 40, 0.95, 0.0),    # redlite chat defaults
        (0.8, 40, 0.95, 0.05),   # llama.cpp defaults
        (1.0, 0, 1.0, 0.0),      # plain softmax over the whole vocabulary
        (1.3, 200, 0.9, 0.0),
        (0.5, 1, 1.0, 0.0),
        (1.0, 0, 0.5, 0.1),
        (1.0, 50, 1.0, 0.2),
        (0.0, 40, 0.95, 0.05),   # greedy
    ]
    all_kinds = ["peaked", "top3", "flat", "ties"]  # fixture seeds depend on this order
    kinds = [k for k in all_kinds if k in args.kinds.split(",")]
    ok = True
    worst = 0.0
    ties = 0
    with tempfile.TemporaryDirectory() as tmp:
        for kind in kinds:
            ki = all_kinds.index(kind)
            path = Path(tmp) / f"{kind}.f32"
            raw = make_logits(kind, 17 + ki)
            path.write_bytes(struct.pack(f"<{VOCAB}f", *raw))
            logits = list(struct.unpack(f"<{VOCAB}f", path.read_bytes()))  # float32-rounded
            for t, k, p, m in grid:
                params = ["--temperature", str(t), "--top-k", str(k), "--top-p", str(p), "--min-p", str(m)]
                a = run(native, path, params)
                b = run(ref, path, params)
                tie = False
                if t <= 0.0:
                    # greedy: llama.cpp keeps the full chain with -inf logits; compare the mode only
                    ma, mb = max(a, key=a.get), max(b, key=b.get)
                    same = ma == mb
                    tie = not same and logits[ma] == logits[mb]
                    diff = 0.0
                else:
                    same = set(a) == set(b)
                    diff = max(abs(a[i] - b[i]) for i in a) if same else 0.0
                    tie = not same and tie_equivalent(a, b, logits, args.prob_tol)
                worst = max(worst, diff)
                line = f"{kind:6} T={t:<4} k={k:<4} p={p:<5} min_p={m:<5} candidates={len(a):>6}/{len(b):<6} max|dp|={diff:.2e}"
                if tie:
                    ties += 1
                    line += f" tie: {len(set(a) ^ set(b)) // 2 if t > 0 else 1} ids differ among equal logits at the cut"
                if (not same and not tie) or diff > args.prob_tol:
                    ok = False
                    print("FAIL", line)
                    continue
                if t > 0.0 and len(a) > 1 and args.draws:
                    # without top-k both sides sort the whole vocabulary on every draw: fewer draws
                    n = args.draws if k > 0 else max(1000, args.draws // 20)
                    da = run(native, path, params, ["--draws", str(n), "--seed", "7"])
                    db = run(ref, path, params, ["--draws", str(n), "--seed", "7"])
                    pa, pb = chi2_pvalue(da, a, n), chi2_pvalue(db, b, n)
                    line += f" chi2 p native={pa:.3f} llama={pb:.3f}"
                    if pa < args.min_pvalue or pb < args.min_pvalue:
                        ok = False
                        print("FAIL", line)
                        continue
                print("ok  ", line)
    print(f"worst probability difference: {worst:.3e}; equal-logit ties resolved differently at a cut: {ties}")
    print(f"SAMPLER PARITY: {'YES' if ok else 'NO'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
