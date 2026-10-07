#!/usr/bin/env python3
"""dev70: decode speed per context band from redlite-server logs (one line per request), with the speculation
summary that follows each request when there is one.

    python3 scripts/dev/server_log_decode.py SERVER_LOG [SERVER_LOG ...]

Context = cached + prefill tokens at the start of the answer. Prints, per log and band, requests, decoded tokens,
aggregate tok/s (tokens / time) and, for lookup or MTP, the share of drafts accepted.
"""
from __future__ import annotations

import re
import sys

REQ = re.compile(r"prefill (\d+) tok \((\d+) cached\).*decode (\d+) tok (\d+) ms")
SPEC = re.compile(r"speculation: (\d+) cycles, (\d+) drafts accepted")
BANDS = [(0, 8192), (8192, 16384), (16384, 32768), (32768, 1 << 30)]


def bands(path: str):
    rows, last = [], None
    for line in open(path, errors="replace"):
        m = REQ.search(line)
        if m:
            ctx = int(m.group(1)) + int(m.group(2))
            last = {"ctx": ctx, "tok": int(m.group(3)), "ms": int(m.group(4)), "cyc": 0, "acc": 0}
            rows.append(last)
            continue
        s = SPEC.search(line)
        if s and last is not None:
            last["cyc"], last["acc"] = int(s.group(1)), int(s.group(2))
    out = []
    for lo, hi in BANDS:
        b = [r for r in rows if lo <= r["ctx"] < hi and r["tok"] > 1]
        tok, ms, cyc, acc = (sum(r[k] for r in b) for k in ("tok", "ms", "cyc", "acc"))
        out.append((lo, hi, len(b), tok, tok / ms * 1000 if ms else 0.0, acc / cyc if cyc else None))
    return out


def main() -> int:
    for path in sys.argv[1:]:
        print(path)
        for lo, hi, n, tok, rate, acc in bands(path):
            if n:
                band = f"{lo // 1024}K-{hi // 1024}K" if hi < 1 << 30 else f"{lo // 1024}K+"
                print(f"  {band:>8}  {n:4d} requests  {tok:7d} tokens  {rate:6.1f} tok/s" + (f"  accepted {acc:.2f}" if acc is not None else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
