#!/usr/bin/env python3
"""dev65: long-context check through a running redlite-server: three needles in a haystack of this repository's code.

    redlite-server MODEL --cache-mib full --context 131072 --port 8095 2> server.log &
    python3 scripts/dev/needle_check.py --port 8095 --tokens 120000 --server-log server.log

The haystack is the repository's own source (C, Objective-C, Python), cut to about --tokens tokens (3.2 characters
per token, the measured ratio for this code). Three passphrases sit in code comments at 10 %, 50 % and 90 % of it;
the question at the end asks for all three. Prints one JSON line: found needles, prompt tokens, wall time and the
server's prefill / decode speeds for the request.
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHARS_PER_TOKEN = 3.2
NEEDLES = [(0.10, "the first passphrase is AMBER-FALCON-4182"),
           (0.50, "the second passphrase is COBALT-WILLOW-9057"),
           (0.90, "the third passphrase is SCARLET-ORCHID-2639")]
QUESTION = ("Three comments in the code above each state a passphrase (first, second and third). "
            "List the three passphrases exactly, one per line, and nothing else.")


def haystack(chars: int) -> str:
    parts, total = [], 0
    files = sorted(ROOT.glob("native/*.c")) + sorted(ROOT.glob("native/*.m")) + sorted(ROOT.glob("redlite/*.py"))
    while total < chars:                      # repeat the corpus if one pass is not long enough
        for f in files:
            text = f"// ===== {f.relative_to(ROOT)} =====\n" + f.read_text(encoding="utf-8", errors="replace")
            parts.append(text)
            total += len(text)
            if total >= chars:
                break
    return "".join(parts)[:chars]


def build_prompt(tokens: int) -> str:
    text = haystack(int(tokens * CHARS_PER_TOKEN))
    for frac, needle in sorted(NEEDLES, reverse=True):     # insert from the end so earlier offsets stay valid
        at = text.rfind("\n", 0, int(len(text) * frac)) + 1
        text = text[:at] + f"/* NOTE: {needle}. */\n" + text[at:]
    return text + "\n\n" + QUESTION


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8095)
    ap.add_argument("--tokens", type=int, required=True, help="approximate prompt length in tokens")
    ap.add_argument("--server-log", help="the server's stderr, to read its prefill/decode line for this request")
    ap.add_argument("--max-tokens", type=int, default=64)
    args = ap.parse_args()
    body = {"model": "x", "messages": [{"role": "user", "content": build_prompt(args.tokens)}],
            "max_tokens": args.max_tokens, "temperature": 0}
    req = urllib.request.Request(f"http://127.0.0.1:{args.port}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t = time.time()
    resp = json.load(urllib.request.urlopen(req, timeout=4 * 3600))
    wall = time.time() - t
    answer = resp["choices"][0]["message"]["content"] or ""
    codes = [n.split()[-1] for _, n in NEEDLES]
    out = {"tokens_target": args.tokens, "prompt_tokens": resp.get("usage", {}).get("prompt_tokens"),
           "found": [c in answer for c in codes], "wall_s": round(wall, 1), "answer": answer.strip()[:200]}
    if args.server_log:
        lines = [l for l in Path(args.server_log).read_text(errors="replace").splitlines() if "prefill" in l and "decode" in l]
        if lines:
            m = re.search(r"prefill (\d+) tok .*?\(([0-9.]+) tok/s\).*decode (\d+) tok \d+ ms \(([0-9.]+) tok/s", lines[-1])
            if m:
                out.update(prefill_tok_s=float(m.group(2)), decode_tok_s=float(m.group(4)))
    print(json.dumps(out))
    return 0 if all(out["found"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
