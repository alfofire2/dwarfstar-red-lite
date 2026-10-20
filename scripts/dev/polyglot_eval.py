#!/usr/bin/env python3
"""dev85: Aider's polyglot benchmark (Exercism exercises) against an OpenAI-compatible server (redlite serve --native).

Each exercise: the model gets the instructions and the files to complete, answers with every file it changes in full,
the exercise's tests run; on a failure the model sees the end of the test output and tries once more (pass@2, as in
Aider's benchmark). Languages: python, javascript, rust, cpp. The reference solutions in .meta are never shown.

    git clone --depth 1 https://github.com/Aider-AI/polyglot-benchmark .deps/polyglot
    (JS once) mkdir .deps/polyglot-jsenv && cp .deps/polyglot/javascript/exercises/practice/affine-cipher/{package.json,babel.config.js} \\
              .deps/polyglot-jsenv && (cd .deps/polyglot-jsenv && npm install)
    python3 scripts/dev/polyglot_eval.py --reference                        # the harness itself: official solutions
    python3 scripts/dev/polyglot_eval.py --port 8080 --out .deps/pg-cf2.jsonl  # a model (resumable)
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / ".deps" / "polyglot"
JSENV = ROOT / ".deps" / "polyglot-jsenv"
LANGS = ("python", "javascript", "rust", "cpp")
FENCE = {"python": "python", "javascript": "javascript", "rust": "rust", "cpp": "cpp"}

SYSTEM = ("You are an expert software engineer. Solve the programming exercise by editing the given files. "
          "For every file you change, write its path on a line by itself, then a fenced code block with the complete "
          "new content of that file. Do not change the tests. Keep the existing function and class names.")


def exercises(langs, names):
    for lang in langs:
        for d in sorted((BENCH / lang / "exercises" / "practice").iterdir()):
            if d.is_dir() and (not names or d.name in names):
                yield lang, d


def config(ex):
    return json.loads((ex / ".meta" / "config.json").read_text())["files"]


def prepare(lang, ex, work):
    """the exercise without its .meta and .approaches; tests that Exercism skips by default are switched on"""
    shutil.copytree(ex, work, ignore=shutil.ignore_patterns(".meta", ".approaches", "node_modules"))
    for t in config(ex)["test"]:
        p = work / t
        s = p.read_text()
        if lang == "javascript":
            s = re.sub(r"\bxtest\(", "test(", s)
            s = re.sub(r"\bxit\(", "it(", s)
        elif lang == "rust":
            s = re.sub(r"^\s*#\[ignore\]\s*$\n?", "", s, flags=re.M)
        p.write_text(s)
    if lang == "javascript":
        (work / "node_modules").symlink_to(JSENV / "node_modules")


def run_tests(lang, work, timeout=180):
    env = dict(os.environ, CARGO_NET_OFFLINE="true", CARGO_TARGET_DIR=str(work / "target"))
    if lang == "python":
        cmds = [[sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]]
    elif lang == "javascript":
        cmds = [[str(JSENV / "node_modules" / ".bin" / "jest"), "--rootDir", str(work), "--ci"]]
    elif lang == "rust":
        cmds = [["cargo", "test", "--quiet", "--", "--include-ignored"]]
    else:
        cmds = [["cmake", "-S", ".", "-B", "build", "-DEXERCISM_RUN_ALL_TESTS=1"], ["cmake", "--build", "build"]]
    out = []
    for c in cmds:
        try:
            r = subprocess.run(c, cwd=work, env=env, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False, "\n".join(out) + f"\n(timed out after {timeout} s: {' '.join(c)})"
        out.append(r.stdout + r.stderr)
        if r.returncode != 0:
            return False, "\n".join(out)
    return True, "\n".join(out)


def instructions(ex):
    docs = ex / ".docs"
    text = (docs / "instructions.md").read_text()
    if (docs / "instructions.append.md").exists():
        text += "\n\n" + (docs / "instructions.append.md").read_text()
    return text


def files_message(lang, work, files):
    parts = []
    for f in files:
        parts.append(f"{f}\n```{FENCE[lang] if not f.endswith(('.toml', '.h')) else ''}\n{(work / f).read_text()}```")
    return "\n\n".join(parts)


def apply_answer(text, work, files):
    """write each fenced block to the solution file named on the line before it; a single unnamed block goes to the
    only (or the first matching-extension) solution file. Returns the files written."""
    written = []
    for m in re.finditer(r"(?:^|\n)([^\n`]*)\n```[^\n]*\n(.*?)```", text, flags=re.S):
        label, body = m.group(1).strip().strip("*`#: "), m.group(2)
        target = next((f for f in files if label and (label == f or label.endswith("/" + f) or label.endswith(Path(f).name))), None)
        if target is None:
            code = [f for f in files if not f.endswith(".toml")]
            if len(code) == 1:
                target = code[0]
        if target and target not in written:
            (work / target).write_text(body)
            written.append(target)
    return written


def chat(port, model, messages, temperature, max_tokens, timeout=1800):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", method="POST",
                                 headers={"Content-Type": "application/json"},
                                 data=json.dumps({"model": model, "messages": messages, "temperature": temperature,
                                                  "max_tokens": max_tokens, "stream": False}).encode())
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    return d["choices"][0]["message"].get("content") or "", d.get("usage") or {}


def reference(lang, ex, work):
    """copy the official solution over the solution files (harness check)"""
    cfg = config(ex)
    for sol, exm in zip(sorted(f for f in cfg["solution"] if not f.endswith(".toml")),
                        sorted(f for f in cfg["example"] if not f.endswith(".toml"))):
        shutil.copy(ex / exm, work / sol)


def main():
    ap = argparse.ArgumentParser(description="Aider polyglot benchmark against a local OpenAI-compatible server")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--model", default="qwen3-next-80b-a3b-redlite")
    ap.add_argument("--langs", default=",".join(LANGS))
    ap.add_argument("--only", default="", help="comma-separated exercise names")
    ap.add_argument("--limit", type=int, default=0, help="at most N exercises per language")
    ap.add_argument("--tries", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=6144)
    ap.add_argument("--out", default=str(ROOT / ".deps" / "polyglot-results.jsonl"))
    ap.add_argument("--reference", action="store_true", help="run the official solutions instead of a model")
    args = ap.parse_args()
    done = set()
    if os.path.exists(args.out) and not args.reference:
        done = {(e["lang"], e["name"]) for e in map(json.loads, open(args.out))}
    names = set(filter(None, args.only.split(",")))
    count = {}
    for lang, ex in exercises(args.langs.split(","), names):
        count[lang] = count.get(lang, 0) + 1
        if (args.limit and count[lang] > args.limit) or (lang, ex.name) in done:
            continue
        files = config(ex)["solution"]
        t0 = time.time()
        rec = {"lang": lang, "name": ex.name, "passed_at": 0, "tries": [], "tokens_in": 0, "tokens_out": 0}
        with tempfile.TemporaryDirectory(prefix="pg-") as tmp:
            work = Path(tmp) / ex.name
            prepare(lang, ex, work)
            if args.reference:
                reference(lang, ex, work)
                ok, out = run_tests(lang, work)
                rec["passed_at"] = 1 if ok else 0
                if not ok:
                    rec["output"] = out[-1500:]
            else:
                messages = [{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": instructions(ex) + "\n\n# Files\n\n" + files_message(lang, work, files)}]
                for attempt in range(1, args.tries + 1):
                    try:
                        answer, usage = chat(args.port, args.model, messages, args.temperature, args.max_tokens)
                    except Exception as e:   # a server error ends this exercise, the run goes on
                        rec["error"] = str(e)
                        break
                    rec["tokens_in"] += usage.get("prompt_tokens", 0)
                    rec["tokens_out"] += usage.get("completion_tokens", 0)
                    written = apply_answer(answer, work, files)
                    ok, out = run_tests(lang, work)
                    rec["tries"].append({"written": written, "ok": ok})
                    if ok:
                        rec["passed_at"] = attempt
                        break
                    tail = "\n".join(out.strip().splitlines()[-60:])
                    messages += [{"role": "assistant", "content": answer},
                                 {"role": "user", "content": f"The tests failed:\n```\n{tail}\n```\nFix the code. Reply with "
                                  "the complete content of every file you change, in the same format."}]
        rec["seconds"] = round(time.time() - t0, 1)
        print(f"{lang:10s} {ex.name:28s} {'pass@' + str(rec['passed_at']) if rec['passed_at'] else 'FAIL':7s} {rec['seconds']:7.1f} s",
              flush=True)
        if not args.reference:
            with open(args.out, "a") as f:
                f.write(json.dumps(rec) + "\n")
        elif not rec["passed_at"]:
            print(rec.get("output", ""), flush=True)
    if not args.reference and os.path.exists(args.out):
        summary(args.out)


def summary(path):
    rows = [json.loads(line) for line in open(path)]
    print(f"\n{'lang':10s} {'n':>4s} {'pass@1':>8s} {'pass@2':>8s} {'errors':>7s} {'median s':>9s}")
    for lang in LANGS + ("all",):
        r = [x for x in rows if lang == "all" or x["lang"] == lang]
        if not r:
            continue
        p1 = sum(x["passed_at"] == 1 for x in r)
        p2 = sum(x["passed_at"] >= 1 for x in r)
        s = sorted(x["seconds"] for x in r)
        print(f"{lang:10s} {len(r):4d} {100 * p1 / len(r):7.1f}% {100 * p2 / len(r):7.1f}% {sum('error' in x for x in r):7d} "
              f"{s[len(s) // 2]:9.1f}")


if __name__ == "__main__":
    main()
