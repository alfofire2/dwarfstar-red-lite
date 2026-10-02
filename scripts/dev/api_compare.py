#!/usr/bin/env python3
"""dev48: compare Red Lite's greedy answers with the model maker's own API, word for word.

    python3 scripts/dev/api_compare.py fetch                 # once: API answers -> tests/fixtures/qwen_api_reference.json
    python3 scripts/dev/api_compare.py compare MODEL [--cache-mib full] [--json OUT]

The reference is Alibaba Cloud Model Studio's qwen3-next-80b-a3b-instruct with temperature 0 and top_k 1
(temperature 0 alone is not deterministic there; its logprobs are misaligned, so only text is compared).
The API key is read from $DASHSCOPE_API_KEY or ~/.config/redlite/dashscope_key and never printed.
The llama.cpp oracle checks that the engine runs the quantized file correctly; this measures how far the
quantized file is from the model as its maker serves it.
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROMPTS = ROOT / "tests" / "fixtures" / "api_prompts.txt"
REFERENCE = ROOT / "tests" / "fixtures" / "qwen_api_reference.json"
API_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"
API_MODEL = "qwen3-next-80b-a3b-instruct"
MAX_TOKENS = 128
CATEGORIES = ["facts", "science", "reasoning", "math", "code", "italian", "writing", "instructions"]


def load_prompts() -> list[str]:
    return [line for line in PROMPTS.read_text().splitlines() if line and not line.startswith("#")]


def category(index: int) -> str:
    return CATEGORIES[min(index // 30, len(CATEGORIES) - 1)]


def api_key() -> str:
    key = os.environ.get("DASHSCOPE_API_KEY") or Path("~/.config/redlite/dashscope_key").expanduser().read_text()
    return key.strip()


def fetch(_args) -> int:
    ref = json.loads(REFERENCE.read_text()) if REFERENCE.exists() else {
        "model": API_MODEL, "endpoint": API_URL, "params": {"temperature": 0, "top_k": 1, "max_tokens": MAX_TOKENS},
        "fetched": time.strftime("%Y-%m-%d"), "answers": {}}
    key = api_key()
    todo = [p for p in load_prompts() if p not in ref["answers"]]
    for n, prompt in enumerate(todo, 1):
        body = {"model": API_MODEL, "messages": [{"role": "user", "content": prompt}], **ref["params"]}
        for attempt in range(5):
            try:
                req = urllib.request.Request(API_URL, data=json.dumps(body).encode(), headers={
                    "Authorization": "Bearer " + key, "Content-Type": "application/json"})
                r = json.load(urllib.request.urlopen(req, timeout=120))
                break
            except (urllib.error.URLError, TimeoutError) as e:
                print(f"  retry {attempt + 1}: {e}", file=sys.stderr)
                time.sleep(5 * (attempt + 1))
        else:
            sys.exit(f"API failed on: {prompt}")
        c = r["choices"][0]
        ref["answers"][prompt] = {"text": c["message"]["content"], "finish_reason": c["finish_reason"],
                                  "prompt_tokens": r["usage"]["prompt_tokens"],
                                  "completion_tokens": r["usage"]["completion_tokens"]}
        REFERENCE.write_text(json.dumps(ref, ensure_ascii=False, indent=1) + "\n")
        print(f"{n}/{len(todo)} {prompt[:60]}")
    print(f"{len(ref['answers'])} reference answers in {REFERENCE.relative_to(ROOT)}")
    return 0


def start_server(bin_dir: Path, model: str, cache_mib: str, log: Path):
    srv = subprocess.Popen([str(bin_dir / "redlite-server"), model, "--port", "0", "--cache-mib", cache_mib],
                           stdout=subprocess.PIPE, stderr=log.open("w"), text=True)
    assert srv.stdout is not None
    m = re.search(r":(\d+) ", srv.stdout.readline())
    if not m:
        srv.kill()
        sys.exit(f"server did not start, see {log}")
    return srv, int(m.group(1))


def chat(port: int, prompt: str) -> dict:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=900)
    body = {"messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": MAX_TOKENS}
    conn.request("POST", "/v1/chat/completions", body=json.dumps(body).encode(),
                 headers={"Content-Type": "application/json"})
    r = json.loads(conn.getresponse().read())
    conn.close()
    return {"text": r["choices"][0]["message"]["content"], "prompt_tokens": r["usage"]["prompt_tokens"]}


def common_words(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a.split(), b.split()):
        if x != y:
            break
        n += 1
    return n


def compare(args) -> int:
    ref = json.loads(REFERENCE.read_text())["answers"]
    prompts = load_prompts()
    missing = [p for p in prompts if p not in ref]
    if missing:
        sys.exit(f"{len(missing)} prompts have no reference answer: run `api_compare.py fetch`")
    log = Path(args.json or "/tmp/api_compare").with_suffix(".server.log")
    srv, port = start_server(Path(args.bin), args.model, args.cache_mib, log)
    rows = []
    try:
        for i, prompt in enumerate(prompts):
            got = chat(port, prompt)
            want = ref[prompt]
            words = len(want["text"].split())
            same = common_words(got["text"], want["text"])
            rows.append({"prompt": prompt, "category": category(i), "identical": got["text"] == want["text"],
                         "common_words": same, "api_words": words, "template_ok": got["prompt_tokens"] == want["prompt_tokens"],
                         "text": got["text"]})
            print(f"{i + 1:3d} {'=' if rows[-1]['identical'] else f'{same:3d}/{words:<3d}'} {prompt[:60]}", flush=True)
    finally:
        srv.send_signal(signal.SIGINT)
        srv.wait(timeout=120)
    frac = [r["common_words"] / max(r["api_words"], 1) for r in rows]
    summary = {
        "model": Path(args.model).name, "cache_mib": args.cache_mib, "prompts": len(rows),
        "identical": sum(r["identical"] for r in rows),
        "median_common_prefix_fraction": round(statistics.median(frac), 3),
        "mean_common_prefix_fraction": round(statistics.mean(frac), 3),
        "first_word_differs": sum(r["common_words"] == 0 for r in rows),
        "template_token_count_mismatch": sum(not r["template_ok"] for r in rows),
        "identical_by_category": {c: f"{sum(r['identical'] for r in rows if r['category'] == c)}/"
                                     f"{sum(r['category'] == c for r in rows)}" for c in CATEGORIES},
    }
    print(json.dumps(summary, indent=1))
    if args.json:
        Path(args.json).write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1) + "\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch").set_defaults(fn=fetch)
    c = sub.add_parser("compare")
    c.add_argument("model")
    c.add_argument("--bin", default=str(ROOT / ".deps" / "redmetal"))
    c.add_argument("--cache-mib", default="full")
    c.add_argument("--json")
    c.set_defaults(fn=compare)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
