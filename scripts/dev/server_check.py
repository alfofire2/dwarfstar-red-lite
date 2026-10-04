#!/usr/bin/env python3
"""Real-model check of redlite-server (dev tool, used by scripts/regress_m4.sh).

    python3 scripts/dev/server_check.py MODEL [--bin .deps/redmetal] [--cache-mib 2048]

1. greedy reference: redlite-generate MODEL --prompt P --temperature 0 --no-stream
2. starts redlite-server MODEL --port 0 with the same cache, waits for it to listen
3. POST /v1/chat/completions stream=true temperature=0: the concatenated deltas must equal
   the reference text byte for byte, the stream must end with finish_reason + [DONE]
4. the same request with stream=false must return the same content
5. dev29 state reuse: a second turn that extends that conversation (first answer + a new user
   message) is answered with usage.prompt_tokens_details.cached_tokens > 0 (the first turn's ids
   are reused, only the new suffix is prefilled); a second server started with --no-reuse
   answers the same second turn from a reset engine (cached_tokens == 0). The two greedy answers
   must be identical, and the first-token latencies of both are printed
6. SIGINT: the server must exit 0 ("shut down cleanly")
Prints "SERVER CHECK: YES" on success.
"""
from __future__ import annotations

import argparse
import http.client
import json
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

PROMPT = "Why is the sky blue? Answer in one sentence."
FOLLOW_UP = "Now explain it to a five-year-old, in two sentences."


def post(port: int, body: dict) -> tuple[int, bytes]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
    conn.request("POST", "/v1/chat/completions", body=json.dumps(body).encode(),
                 headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    return resp.status, data


def stream_chat(port: int, body: dict) -> tuple[str, dict, float]:
    """Streams one request; returns (content, final usage, seconds to the first content delta)."""
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
    t0 = time.monotonic()
    conn.request("POST", "/v1/chat/completions", body=json.dumps({**body, "stream": True}).encode(),
                 headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    content, usage, ttft, buf = "", {}, -1.0, b""
    while True:
        chunk = resp.read1(65536) if hasattr(resp, "read1") else resp.read(65536)
        if not chunk:
            break
        buf += chunk
        while b"\n\n" in buf:
            block, buf = buf.split(b"\n\n", 1)
            payload = block.decode("utf-8")[len("data: "):]
            if payload == "[DONE]":
                continue
            event = json.loads(payload)
            delta = event["choices"][0]["delta"].get("content", "")
            if delta and ttft < 0:
                ttft = time.monotonic() - t0
            content += delta
            usage = event.get("usage", usage)
    conn.close()
    return content, usage, ttft


SERVER_EXTRA: list = []   # dev45: --mtp FILE appends ["--mtp", FILE] to every server started here


def start_server(bin_dir: Path, model: str, cache_mib: str, *extra: str):
    srv = subprocess.Popen(
        [str(bin_dir / "redlite-server"), model, "--port", "0", "--cache-mib", cache_mib, *SERVER_EXTRA, *extra],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    line = srv.stdout.readline()
    m = re.search(r":(\d+) ", line)
    if not m:
        print(f"server did not start: {line!r}\n{srv.stderr.read()}")
        srv.kill()
        return None, 0
    return srv, int(m.group(1))


def stop_server(srv) -> tuple[int, str]:
    srv.send_signal(signal.SIGINT)
    try:
        rc = srv.wait(timeout=120)
    except subprocess.TimeoutExpired:
        srv.kill()
        rc = -1
    return rc, srv.stderr.read()


def multi_turn(bin_dir: Path, model: str, cache_mib: str, port: int, max_tokens: int) -> bool:
    first = [{"role": "user", "content": PROMPT}]
    base = {"temperature": 0, "max_tokens": max_tokens}
    reply, _, _ = stream_chat(port, {**base, "messages": first})
    second = first + [{"role": "assistant", "content": reply}, {"role": "user", "content": FOLLOW_UP}]
    warm, warm_usage, warm_ttft = stream_chat(port, {**base, "messages": second})
    cold_srv, cold_port = start_server(bin_dir, model, cache_mib, "--no-reuse")
    if not cold_srv:
        return False
    try:
        cold, cold_usage, cold_ttft = stream_chat(cold_port, {**base, "messages": second})
    finally:
        cold_rc, cold_err = stop_server(cold_srv)
    warm_cached = warm_usage.get("prompt_tokens_details", {}).get("cached_tokens", -1)
    cold_cached = cold_usage.get("prompt_tokens_details", {}).get("cached_tokens", -1)
    print(f"turn 2 warm: cached={warm_cached}/{warm_usage.get('prompt_tokens')} ttft={warm_ttft * 1000:.0f} ms content={warm!r}")
    print(f"turn 2 cold: cached={cold_cached}/{cold_usage.get('prompt_tokens')} ttft={cold_ttft * 1000:.0f} ms content={cold!r}")
    ok = warm_cached > 0 and cold_cached == 0 and warm == cold and bool(warm) and cold_rc == 0
    print(f"multi-turn reuse: {'identical greedy answer' if ok else 'MISMATCH'}")
    return ok


def state_restart(bin_dir: Path, model: str, cache_mib: str, max_tokens: int) -> bool:
    """dev43: a long prompt on a server with --state-dir, then the same request after a restart: the second
    server restores the stored chunk-aligned prefix (cached_tokens > 0) and answers identically."""
    import tempfile
    fixture = (Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "long_context_prompt.txt").read_text()
    body = {"messages": [{"role": "user", "content": fixture * 3 + "\nSummarize the text above in two sentences."}],
            "temperature": 0, "max_tokens": max_tokens}
    with tempfile.TemporaryDirectory() as state_dir:
        answers = []
        for run in ("first", "restart"):
            srv, port = start_server(bin_dir, model, cache_mib, "--state-dir", state_dir, "--context", "8192")
            if not srv:
                return False
            try:
                content, usage, ttft = stream_chat(port, body)
            finally:
                rc, _ = stop_server(srv)
            cached = usage.get("prompt_tokens_details", {}).get("cached_tokens", -1)
            print(f"state {run}: cached={cached}/{usage.get('prompt_tokens')} ttft={ttft * 1000:.0f} ms rc={rc} content={content!r}")
            answers.append((content, cached, rc))
        ok = (answers[0][1] == 0 and answers[1][1] >= 2048 and answers[0][0] == answers[1][0] and bool(answers[0][0])
              and answers[0][2] == 0 and answers[1][2] == 0)
    print(f"state restart: {'identical greedy answer' if ok else 'MISMATCH'}")
    return ok


def parallel(bin_dir: Path, model: str, max_tokens: int) -> bool:
    """dev56: --parallel 2 at full residency. Two greedy answers asked at the same time equal the same answers asked
    one after the other, and their decode steps ran as pairs."""
    import threading
    srv, port = start_server(bin_dir, model, "full", "--parallel", "2")
    if not srv:
        return False
    prompts = [PROMPT, "Write a haiku about the sea."]
    bodies = [{"messages": [{"role": "user", "content": p}], "temperature": 0, "max_tokens": max_tokens} for p in prompts]
    alone, together = [None, None], [None, None]
    try:
        for i, b in enumerate(bodies):
            alone[i] = json.loads(post(port, b)[1])["choices"][0]["message"]["content"]

        def ask(i):
            together[i] = json.loads(post(port, bodies[i])[1])["choices"][0]["message"]["content"]
        threads = [threading.Thread(target=ask, args=(i,)) for i in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=600)
    finally:
        rc, err = stop_server(srv)
    paired = sum(int(m) for m in re.findall(r"(\d+) paired", err))
    for i in range(2):
        print(f"prompt {i}: {'identical' if alone[i] == together[i] else 'DIFFERENT'}: {together[i]!r}")
    print(f"paired decode steps: {paired}")
    return rc == 0 and alone == together and None not in alone and paired > 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--bin", default=str(Path(__file__).resolve().parents[2] / ".deps" / "redmetal"))
    ap.add_argument("--cache-mib", default="2048")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--state-restart", action="store_true", help="only the dev43 --state-dir restart check")
    ap.add_argument("--parallel", action="store_true", help="only the dev56 --parallel 2 check (full residency)")
    ap.add_argument("--mtp", help="dev45: run every server with --mtp FILE (needs --cache-mib full); outputs must not change")
    args = ap.parse_args()
    bin_dir = Path(args.bin)
    if args.mtp:
        SERVER_EXTRA.extend(["--mtp", args.mtp])
    if args.parallel:
        ok = parallel(bin_dir, args.model, args.max_tokens)
        print(f"SERVER PARALLEL CHECK: {'YES' if ok else 'NO'}")
        return 0 if ok else 1
    if args.state_restart:
        ok = state_restart(bin_dir, args.model, args.cache_mib, args.max_tokens)
        print(f"SERVER STATE CHECK: {'YES' if ok else 'NO'}")
        return 0 if ok else 1

    ref = subprocess.run(
        [str(bin_dir / "redlite-generate"), args.model, "--prompt", PROMPT, "--max-tokens", str(args.max_tokens),
         "--temperature", "0", "--cache-mib", args.cache_mib, "--no-stream"],
        capture_output=True, check=True,
    ).stdout
    expected = ref[:-1].decode("utf-8", errors="replace") if ref.endswith(b"\n") else ref.decode("utf-8", errors="replace")
    print(f"reference ({len(expected)} chars): {expected!r}")

    srv = subprocess.Popen(
        [str(bin_dir / "redlite-server"), args.model, "--port", "0", "--cache-mib", args.cache_mib, *SERVER_EXTRA],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    ok = False
    try:
        line = srv.stdout.readline()
        m = re.search(r":(\d+) ", line)
        if not m:
            print(f"server did not start: {line!r}\n{srv.stderr.read()}")
            return 1
        port = int(m.group(1))
        body = {"messages": [{"role": "user", "content": PROMPT}], "temperature": 0, "max_tokens": args.max_tokens}

        status, data = post(port, {**body, "stream": True})
        blocks = [b for b in data.decode("utf-8").split("\n\n") if b.strip()]
        done = blocks and blocks[-1] == "data: [DONE]"
        events = [json.loads(b[len("data: "):]) for b in blocks if b != "data: [DONE]"]
        streamed = "".join(e["choices"][0]["delta"].get("content", "") for e in events)
        finish = events[-1]["choices"][0].get("finish_reason") if events else None
        print(f"stream  : status={status} done={bool(done)} finish={finish} content={streamed!r}")

        status2, data2 = post(port, body)
        plain = json.loads(data2)["choices"][0]["message"]["content"] if status2 == 200 else None
        print(f"blocking: status={status2} content={plain!r}")

        ok = status == 200 and bool(done) and finish in ("stop", "length") and streamed == expected and plain == expected
        ok = multi_turn(bin_dir, args.model, args.cache_mib, port, args.max_tokens) and ok
    finally:
        srv.send_signal(signal.SIGINT)
        try:
            rc = srv.wait(timeout=120)
        except subprocess.TimeoutExpired:
            srv.kill()
            rc = -1
        err = srv.stderr.read()
        print(err.strip())
        ok = ok and rc == 0 and "shut down cleanly" in err
    print(f"SERVER CHECK: {'YES' if ok else 'NO'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
