#!/usr/bin/env python3
"""dev29 measurement: first-token latency of a second chat turn through redlite-server.

    python3 scripts/dev/server_ttft.py MODEL [--bin DIR] [--cache-mib 4096] [--reps 3] [--cool 60]
                                             [--server-args=--no-reuse] [--first-chars 4400]

Turn 1: the user sends the first ~N tokens of tests/fixtures/long_context_prompt.txt and asks for a
one-sentence summary (greedy, 32 tokens). Turn 2: the same conversation plus the answer and a short
follow-up question. The script times turn 2 from the request to the first streamed content byte.
Each repetition starts a fresh server (so the page cache is warm but the engine state is only what
turn 1 left). --server-args=--no-reuse measures the reset-and-ingest-everything behaviour; a
0.3.0 redlite-server (no reuse) can be measured with --bin pointing at its build. Prints each run
and the median. Development measurement, not a regression check.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shlex
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from server_check import start_server, stop_server, stream_chat  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--bin", default=str(ROOT / ".deps" / "redmetal"))
    ap.add_argument("--cache-mib", default="4096")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--cool", type=float, default=60.0)
    ap.add_argument("--server-args", default="")
    ap.add_argument("--first-chars", type=int, default=4400, help="characters of the fixture in turn 1 (~1000 tokens)")
    args = ap.parse_args()
    text = (ROOT / "tests" / "fixtures" / "long_context_prompt.txt").read_text()[: args.first_chars]
    first = [{"role": "user", "content": text + "\n\nSummarize the text above in one sentence."}]
    base = {"temperature": 0, "max_tokens": 32}
    ttfts, answers = [], set()
    for rep in range(args.reps):
        time.sleep(args.cool)
        srv, port = start_server(Path(args.bin), args.model, args.cache_mib, *shlex.split(args.server_args))
        if not srv:
            return 1
        try:
            reply, u1, t1 = stream_chat(port, {**base, "messages": first})
            second = first + [{"role": "assistant", "content": reply}, {"role": "user", "content": "Which detail matters most, and why?"}]
            answer, u2, t2 = stream_chat(port, {**base, "messages": second})
        finally:
            stop_server(srv)
        cached = u2.get("prompt_tokens_details", {}).get("cached_tokens", 0)
        print(f"run {rep + 1}: turn1 prompt {u1.get('prompt_tokens')} ttft {t1 * 1000:.0f} ms | "
              f"turn2 prompt {u2.get('prompt_tokens')} cached {cached} ttft {t2 * 1000:.0f} ms")
        ttfts.append(t2 * 1000)
        answers.add(answer)
    digest = hashlib.sha1(sorted(answers)[0].encode()).hexdigest()[:12]
    print(f"turn2 ttft median: {statistics.median(ttfts):.0f} ms ({'same answer every run' if len(answers) == 1 else 'ANSWERS DIFFER'}, answer sha1 {digest})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
