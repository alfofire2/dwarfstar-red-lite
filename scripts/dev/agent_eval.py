#!/usr/bin/env python3
"""dev60: a small coding-agent check of a running redlite-server, driven by the pi coding agent.

Each task starts in a fresh directory with its fixture files, runs `pi -p TASK` against the server, and is graded by
a script (tests that must pass, files that must or must not change, an answer that must contain given facts).
Per task: pass/fail, wall time, and from the server's log the requests, prompt tokens and the share that reused the
engine state.

    redlite-server MODEL --port 8091 --cache-mib full --context 32768 2> server.log &
    python3 scripts/dev/agent_eval.py --server-log server.log --json results.json

pi reads its provider from PI_CODING_AGENT_DIR/models.json (--agent-dir writes one for 127.0.0.1:PORT).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

TASKS = [
    {
        "id": "new_code",
        "prompt": "Create primes.py with a function is_prime(n) and a unittest file test_primes.py with a few tests, "
                  "then run the tests with python3 -m unittest and report the result.",
        "files": {},
        "check": "import primes\n"
                 "assert [n for n in range(-3, 30) if primes.is_prime(n)] == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]\n"
                 "assert primes.is_prime(7919) and not primes.is_prime(7917)\n"
                 "import subprocess, sys\n"
                 "r = subprocess.run([sys.executable, '-m', 'unittest', '-q'], capture_output=True, text=True)\n"
                 "assert r.returncode == 0 and 'Ran 0 tests' not in r.stderr, r.stderr\n",
    },
    {
        "id": "bug_fix",
        "prompt": "The tests in test_stats.py fail. Find and fix the bug in stats.py. Do not change the tests.",
        "files": {
            "stats.py": "def mean(xs):\n    return sum(xs) / len(xs)\n\n\n"
                        "def median(xs):\n    s = sorted(xs)\n    n = len(s)\n    mid = n // 2\n"
                        "    if n % 2:\n        return s[mid]\n    return (s[mid] + s[mid + 1]) / 2\n",
            "test_stats.py": "import unittest\nfrom stats import mean, median\n\n\nclass T(unittest.TestCase):\n"
                             "    def test_mean(self):\n        self.assertEqual(mean([1, 2, 3]), 2)\n\n"
                             "    def test_median_odd(self):\n        self.assertEqual(median([3, 1, 2]), 2)\n\n"
                             "    def test_median_even(self):\n        self.assertEqual(median([4, 1, 3, 2]), 2.5)\n",
        },
        "unchanged": ["test_stats.py"],
        "check": "import subprocess, sys\n"
                 "r = subprocess.run([sys.executable, '-m', 'unittest', '-q'], capture_output=True, text=True)\n"
                 "assert r.returncode == 0, r.stderr\n"
                 "from stats import median\nassert median([10, 1, 7, 3, 5, 9]) == 6\n",
    },
    {
        "id": "rename",
        "prompt": "Rename the function get_data to load_records everywhere in this project (definition and every "
                  "use). Then run python3 main.py to check that it still works.",
        "files": {
            "utils.py": "def get_data():\n    return [('ada', 36), ('linus', 28), ('grace', 45)]\n",
            "report.py": "from utils import get_data\n\n\ndef oldest():\n    return max(get_data(), key=lambda r: r[1])[0]\n",
            "main.py": "from utils import get_data\nfrom report import oldest\n\n"
                       "print(len(get_data()), 'records; oldest:', oldest())\n",
        },
        "check": "import pathlib, subprocess, sys\n"
                 "src = ''.join(p.read_text() for p in pathlib.Path('.').glob('*.py') if p.name != 'check_task.py')\n"
                 "assert 'get_data' not in src and 'def load_records' in src\n"
                 "r = subprocess.run([sys.executable, 'main.py'], capture_output=True, text=True)\n"
                 "assert r.stdout.strip() == '3 records; oldest: grace', r.stdout + r.stderr\n",
    },
    {
        "id": "read_answer",
        "prompt": "Read server.py. What port does it listen on by default, and which environment variable overrides "
                  "it? Answer in one sentence; do not modify any file.",
        "files": {
            "server.py": "import os\nimport http.server\n\nDEFAULT_PORT = 8443\n\n\ndef port():\n"
                         "    return int(os.environ.get('APP_LISTEN_PORT', DEFAULT_PORT))\n\n\n"
                         "if __name__ == '__main__':\n"
                         "    http.server.HTTPServer(('', port()), http.server.SimpleHTTPRequestHandler).serve_forever()\n",
        },
        "unchanged": ["server.py"],
        "answer": ["8443", "APP_LISTEN_PORT"],
    },
    {
        "id": "add_flag",
        "prompt": "Add a --verbose flag to cli.py (use argparse). When it is given, print 'verbose on' before the "
                  "normal output; without it the output must not change. Run it both ways to show it works.",
        "files": {
            "cli.py": "import sys\n\n\ndef main():\n    print('hello from cli')\n\n\n"
                      "if __name__ == '__main__':\n    main()\n",
        },
        "check": "import subprocess, sys\n"
                 "a = subprocess.run([sys.executable, 'cli.py'], capture_output=True, text=True).stdout\n"
                 "b = subprocess.run([sys.executable, 'cli.py', '--verbose'], capture_output=True, text=True).stdout\n"
                 "assert a.strip() == 'hello from cli', a\n"
                 "assert b.strip().splitlines() == ['verbose on', 'hello from cli'], b\n",
    },
]

# dev62: harder tasks on a copy of this repository (git archive of HEAD): new code in a large file, reading C,
# a planted bug, and one long session of four prompts in the same pi session.
REPO_TASKS = [
    {
        "id": "models_json",
        "repo": True,
        "prompt": "Add a --json option to the `redlite models` command (redlite/cli.py): instead of the table it prints "
                  "a JSON list with one object per variant (key, repo, filename, nominal_gb, quality, recommended_mode). "
                  "Add a unit test for it in tests/test_cli.py, then run "
                  "python3 -m unittest discover -s tests -p test_cli.py and make sure it passes.",
        "check": "import json, subprocess, sys, os\n"
                 "sys.path.insert(0, '.')\n"
                 "from redlite.model_catalog import VARIANTS\n"
                 "env = {**os.environ, 'PYTHONPATH': '.'}\n"
                 "out = subprocess.run([sys.executable, '-m', 'redlite.cli', 'models', '--json'], capture_output=True, text=True, env=env)\n"
                 "d = json.loads(out.stdout)\n"
                 "assert isinstance(d, list) and len(d) == len(VARIANTS), out.stdout[:200]\n"
                 "assert {x['key'] for x in d} == set(VARIANTS) and all('filename' in x and 'repo' in x for x in d)\n"
                 "r = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_cli.py'], capture_output=True, text=True, env=env)\n"
                 "assert r.returncode == 0, r.stderr[-400:]\n"
                 "t = open('tests/test_cli.py').read()\n"
                 "assert 'models' in t and '--json' in t\n",
    },
    {
        "id": "explain_stop",
        "repo": True,
        "prompt": "In native/redlite_native_server.c, how are stop sequences handled while an answer is streamed? Name "
                  "the function that finds them, and say what happens to generated text that could be the start of a "
                  "stop sequence. Answer briefly; do not modify any file.",
        "unchanged_tree": True,
        "answer": ["rl_stop_scan"],
        "answer_any": ["hold", "held", "keep", "kept", "buffer", "pending"],
    },
    {
        "id": "fix_planner",
        "repo": True,
        "patch": [("redlite/planner.py", "NATIVE_SLOT_STATE_POSITIONS * (parallel - 1)", "NATIVE_SLOT_STATE_POSITIONS * parallel")],
        "prompt": "A test in tests/test_native_defaults.py fails. Find the cause in the redlite package and fix it; do "
                  "not change the tests. Run python3 -m unittest discover -s tests -p test_native_defaults.py to confirm.",
        "unchanged": ["tests/test_native_defaults.py"],
        "check": "import subprocess, sys, os\n"
                 "r = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_native_defaults.py'],"
                 " capture_output=True, text=True, env={**os.environ, 'PYTHONPATH': '.'})\n"
                 "assert r.returncode == 0, r.stderr[-400:]\n",
    },
    {
        "id": "long_session",
        "repo": True,
        "prompts": [
            "Read redlite/planner.py and explain in three sentences what native_gpu_plan returns and in which order "
            "it tries the configurations.",
            "Now read the function _gpu_tuning in redlite/cli.py and say how it uses native_gpu_plan.",
            "Which test class in tests/test_native_defaults.py covers native_gpu_plan? Name it.",
            "Add one more test method to that class: with a GPU limit of 1 MiB nothing fits, so native_gpu_plan must "
            "return None. Then run python3 -m unittest discover -s tests -p test_native_defaults.py.",
        ],
        "answer": ["GpuPlanTests"],
        "check": "import subprocess, sys, os, re\n"
                 "src = open('tests/test_native_defaults.py').read()\n"
                 "cls = src[src.index('class GpuPlanTests'):]\n"
                 "cls = cls[:cls.index('\\nclass ', 1)] if '\\nclass ' in cls[1:] else cls\n"
                 "assert 'None' in cls and len(re.findall(r'def test_', cls)) >= 2, 'no new test in GpuPlanTests'\n"
                 "r = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_native_defaults.py'],"
                 " capture_output=True, text=True, env={**os.environ, 'PYTHONPATH': '.'})\n"
                 "assert r.returncode == 0, r.stderr[-400:]\n",
    },
]

# dev65: harder tasks, still on a copy of this repository: two bugs in two modules, a rename across files, tests graded
# by mutants, a C feature compiled and exercised by the check, reading C, and a three-prompt debugging session.
_FAKE_BUILD = ("import subprocess, sys, os, re, json, tempfile, time, urllib.request, urllib.error\n"
               "cc = 'clang'\n"
               "b = os.path.join(tempfile.mkdtemp(), 'fake')\n"
               "subprocess.run([cc, '-O1', '-std=c11', '-D_POSIX_C_SOURCE=200809L', '-Wall', '-Wextra', '-Werror', '-Inative',\n"
               "                'native/redlite_native_server.c', 'native/redlite_native_server_fake.c', '-lm', '-pthread', '-o', b], check=True)\n"
               "p = subprocess.Popen([b, '--port', '0'], stdout=subprocess.PIPE, text=True)\n"
               "port = int(re.search(r'127\\.0\\.0\\.1:(\\d+)', p.stdout.readline()).group(1))\n"
               "def post(n):\n"
               "    body = json.dumps({'model': 'x', 'messages': [{'role': 'user', 'content': 'hi'}], 'max_tokens': n}).encode()\n"
               "    req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', body, {'Content-Type': 'application/json'})\n"
               "    try:\n"
               "        r = urllib.request.urlopen(req, timeout=20); return r.status, r.read().decode()\n"
               "    except urllib.error.HTTPError as e:\n"
               "        return e.code, e.read().decode()\n")
_MUTANTS = ("import subprocess, sys, os\n"
            "def run():\n"
            "    return subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_quant_mix.py'],\n"
            "                          capture_output=True, text=True, env={**os.environ, 'PYTHONPATH': '.'})\n"
            "r = run(); assert r.returncode == 0 and 'Ran 0 tests' not in r.stderr, 'tests do not pass: ' + r.stderr[-300:]\n"
            "src = open('scripts/dev/quant_mix.py').read()\n"
            "killed = []\n"
            "for old, new in [('range(int(a), int(b or a) + 1)', 'range(int(a), int(b or a))'),\n"
            "                 ('layers(where) if where else None', 'layers(where) if where else set()')]:\n"
            "    assert src.count(old) == 1, 'quant_mix.py was changed: ' + old\n"
            "    open('scripts/dev/quant_mix.py', 'w').write(src.replace(old, new))\n"
            "    killed.append(run().returncode != 0)\n"
            "    open('scripts/dev/quant_mix.py', 'w').write(src)\n"
            "assert all(killed), f'mutants not caught: {killed}'\n")
HARD_TASKS = [
    {
        "id": "two_bugs",
        "repo": True,
        "patch": [("redlite/model_catalog.py", '"24gb": "redlite_f2",', '"24gb": "redlite_e3",'),
                  ("redlite/planner.py", "NATIVE_KV_MIB_PER_POS = 48.0 / 1024.0", "NATIVE_KV_MIB_PER_POS = 4.8 / 1024.0")],
        "prompt": "Some tests in tests/test_native_defaults.py fail. Find every cause in the redlite package and fix them; "
                  "do not change the tests. Run python3 -m unittest discover -s tests -p test_native_defaults.py to confirm.",
        "unchanged": ["tests/test_native_defaults.py"],
        "check": "import subprocess, sys, os\n"
                 "r = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_native_defaults.py'],"
                 " capture_output=True, text=True, env={**os.environ, 'PYTHONPATH': '.'})\n"
                 "assert r.returncode == 0, r.stderr[-400:]\n",
    },
    {
        "id": "rename",
        "repo": True,
        "prompt": "Rename the function native_plan_context in redlite/planner.py to plan_context_positions, everywhere it "
                  "is defined, imported or called in redlite/ and tests/. Then run "
                  "python3 -m unittest discover -s tests -p 'test_native_*.py' and python3 -m unittest discover -s tests "
                  "-p test_cli.py and make sure they pass.",
        "check": "import subprocess, sys, os, pathlib\n"
                 "left = [str(p) for d in ('redlite', 'tests') for p in pathlib.Path(d).rglob('*.py') if 'native_plan_context' in p.read_text()]\n"
                 "assert not left, f'old name left in {left}'\n"
                 "assert 'def plan_context_positions(' in open('redlite/planner.py').read()\n"
                 "env = {**os.environ, 'PYTHONPATH': '.'}\n"
                 "for pat in ('test_native_defaults.py', 'test_cli.py'):\n"
                 "    r = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-p', pat], capture_output=True, text=True, env=env)\n"
                 "    assert r.returncode == 0, pat + ': ' + r.stderr[-300:]\n",
    },
    {
        "id": "write_tests",
        "repo": True,
        "prompt": "Write a new file tests/test_quant_mix.py with unit tests for the functions layers() and parse_set() in "
                  "scripts/dev/quant_mix.py (import it by putting scripts/dev on sys.path; do not change quant_mix.py). "
                  "Cover single layers, ranges like 24-47, comma lists like 0-5,43-47, and specs with and without the "
                  "@LAYERS part. Run python3 -m unittest discover -s tests -p test_quant_mix.py.",
        "unchanged": ["scripts/dev/quant_mix.py"],
        "check": _MUTANTS,
    },
    {
        "id": "c_limit",
        "repo": True,
        "prompt": "In the C server (native/redlite_native_server.c), requests may ask for up to 1048576 tokens. Lower "
                  "the limit: a max_tokens (or max_completion_tokens) above 65536 must be rejected with HTTP 400 and an "
                  "error message containing 'max_tokens must be at most 65536'. 65536 itself stays allowed. Add a test "
                  "for it to tests/test_native_server.py (it builds a fake server from the same C file) and run "
                  "python3 -m unittest discover -s tests -p test_native_server.py.",
        "check": _FAKE_BUILD +
                 "try:\n"
                 "    code, text = post(70000)\n"
                 "    assert code == 400 and 'max_tokens must be at most 65536' in text, (code, text[:200])\n"
                 "    code, text = post(65536)\n"
                 "    assert code == 200, (code, text[:200])\n"
                 "finally:\n"
                 "    p.kill()\n"
                 "assert '65536' in open('tests/test_native_server.py').read(), 'no test added'\n"
                 "r = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_native_server.py'],"
                 " capture_output=True, text=True, env={**os.environ, 'PYTHONPATH': '.'})\n"
                 "assert r.returncode == 0, r.stderr[-400:]\n",
    },
    {
        "id": "explain_kv_pad",
        "repo": True,
        "prompt": "The Metal engine allocates its attention KV cache with RL_ENGINE_KV_PAD extra positions past the "
                  "context. How many positions is that, and which part of the engine needs them, and why? Look in "
                  "native/. Answer briefly; do not modify any file.",
        "unchanged_tree": True,
        "answer": ["32"],
        "answer_any": ["tile", "tiled", "block"],
    },
    {
        "id": "debug_session",
        "repo": True,
        "patch": [("redlite/model_catalog.py", "key = ALIASES.get(name.lower(), name.lower())", "key = ALIASES.get(name, name)")],
        "prompts": [
            "`redlite download CODER` fails with 'Unknown variant', while `redlite download coder` works. Find the cause "
            "in the redlite package and explain it in one sentence. Do not change anything yet.",
            "Fix it so that variant names are case-insensitive, and add a test for an upper-case name to "
            "tests/test_native_defaults.py.",
            "Also add an alias coder-24gb that points to the same variant as coder, with a test. Then run "
            "python3 -m unittest discover -s tests -p test_native_defaults.py.",
        ],
        "answer_any": ["lower", "case"],
        "check": "import subprocess, sys, os\n"
                 "sys.path.insert(0, '.')\n"
                 "from redlite.model_catalog import resolve_variant\n"
                 "assert resolve_variant('CODER').key == resolve_variant('coder').key == 'redlite_coder_cf2'\n"
                 "assert resolve_variant('coder-24gb').key == 'redlite_coder_cf2'\n"
                 "assert 'coder-24gb' in open('tests/test_native_defaults.py').read(), 'no test for the alias'\n"
                 "r = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_native_defaults.py'],"
                 " capture_output=True, text=True, env={**os.environ, 'PYTHONPATH': '.'})\n"
                 "assert r.returncode == 0, r.stderr[-400:]\n",
    },
]
SUITES = {"basic": TASKS, "repo": REPO_TASKS, "hard": HARD_TASKS}
ROOT = Path(__file__).resolve().parents[2]

REQ = re.compile(r"request \d+: prompt=(\d+) cached=(\d+) completion=(\d+) finish=(\w+)")


def write_agent_dir(path: Path, port: int, model_id: str, context: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "models.json").write_text(json.dumps({"providers": {"redlite": {
        "baseUrl": f"http://127.0.0.1:{port}/v1", "api": "openai-completions", "apiKey": "redlite",
        "compat": {"supportsDeveloperRole": False, "supportsReasoningEffort": False, "supportsStore": False,
                   "supportsStrictMode": False, "maxTokensField": "max_tokens"},
        "models": [{"id": model_id, "name": "Red Lite (local)", "reasoning": False, "input": ["text"],
                    "contextWindow": context, "maxTokens": 4096,
                    "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}}]}}}, indent=2))


def run_task(task: dict, args) -> dict:
    with tempfile.TemporaryDirectory() as d:
        work = Path(d)
        if task.get("repo"):
            archive = subprocess.run(["git", "-C", str(ROOT), "archive", "HEAD"], capture_output=True, check=True).stdout
            subprocess.run(["tar", "-x", "-C", str(work)], input=archive, check=True)
            for name, old, new in task.get("patch", []):
                text = (work / name).read_text()
                assert text.count(old) == 1, (name, old)
                (work / name).write_text(text.replace(old, new))
            task = {**task, "files": {n: (work / n).read_text() for n in task.get("unchanged", [])}}
            snapshot = {p: p.read_bytes() for p in work.rglob("*") if p.is_file()} if task.get("unchanged_tree") else None
        else:
            snapshot = None
        for name, text in task["files"].items():
            (work / name).write_text(text)
        log_start = Path(args.server_log).stat().st_size
        env = {**os.environ, "PI_CODING_AGENT_DIR": args.agent_dir, "PI_OFFLINE": "1"}
        t0 = time.monotonic()
        answer, rc, err = "", 0, ""
        prompts = task.get("prompts") or [task["prompt"]]
        session = ["--no-session"] if len(prompts) == 1 else ["--session-dir", str(work / ".pi-sessions"), "--session-id", "eval"]
        for k, prompt in enumerate(prompts):   # several prompts: one pi session, continued
            try:
                out = subprocess.run(["pi", "--provider", "redlite", "--model", args.model_id, *session,
                                      "--no-extensions", "--no-skills", "--no-context-files", "-p", prompt],
                                     cwd=work, env=env, stdin=subprocess.DEVNULL,   # pi -p reads a piped stdin
                                     capture_output=True, text=True,
                                     timeout=max(1, args.timeout - (time.monotonic() - t0)))
                answer, rc, err = answer + out.stdout, out.returncode, out.stderr
            except subprocess.TimeoutExpired:
                rc = -1
            if rc != 0:
                break
        wall = time.monotonic() - t0
        with open(args.server_log, encoding="utf-8", errors="replace") as f:
            f.seek(log_start)
            reqs = [tuple(int(x) for x in m.groups()[:3]) for m in REQ.finditer(f.read())]
        problems = []
        if rc != 0:
            problems.append(f"pi exit {rc}" + (": " + err.strip().splitlines()[-1][:160] if err.strip() else ""))
        for name in task.get("unchanged", []):
            if (work / name).read_text() != task["files"][name]:
                problems.append(f"{name} was modified")
        for fact in task.get("answer", []):
            if fact not in answer:
                problems.append(f"answer lacks {fact!r}")
        if task.get("answer_any") and not any(w in answer.lower() for w in task["answer_any"]):
            problems.append(f"answer lacks any of {task['answer_any']}")
        if snapshot is not None:
            changed = [str(p.relative_to(work)) for p, b in snapshot.items() if not p.exists() or p.read_bytes() != b]
            if changed:
                problems.append(f"modified {changed[:3]}")
        if "check" in task:
            (work / "check_task.py").write_text(task["check"])
            # no bytecode cache: a fix of the same length written within a second of a test run kept the stale .pyc
            r = subprocess.run([sys.executable, "check_task.py"], cwd=work, capture_output=True, text=True, timeout=120,
                               env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
            if r.returncode != 0:
                problems.append("check: " + (r.stderr.strip().splitlines() or ["failed"])[-1][:200])
    prompt = sum(r[0] for r in reqs)
    cached = sum(r[1] for r in reqs)
    return {"id": task["id"], "pass": not problems, "problems": problems, "wall_s": round(wall, 1),
            "requests": len(reqs), "prompt_tokens": prompt, "cached_tokens": cached,
            "completion_tokens": sum(r[2] for r in reqs), "answer": answer.strip()[-300:]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server-log", required=True, help="the server's stderr, appended to while the tasks run")
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--model-id", default="qwen3-next-80b-a3b-redlite")
    ap.add_argument("--context", type=int, default=32768)
    ap.add_argument("--agent-dir", help="pi config dir (default: a models.json for --port in a temporary dir)")
    ap.add_argument("--suite", choices=sorted(SUITES), default="basic", help="basic: 5 small tasks; repo: 4 on a copy of this repository; hard: 6 harder ones (dev65)")
    ap.add_argument("--tasks", help="comma-separated task ids (default: all)")
    ap.add_argument("--timeout", type=int, default=900, help="seconds per task")
    ap.add_argument("--json", help="write the results here")
    args = ap.parse_args()
    tmp = None
    if not args.agent_dir:
        tmp = tempfile.TemporaryDirectory()
        args.agent_dir = tmp.name
        write_agent_dir(Path(tmp.name), args.port, args.model_id, args.context)
    tasks = [t for t in SUITES[args.suite] if not args.tasks or t["id"] in args.tasks.split(",")]
    results = []
    for t in tasks:
        r = run_task(t, args)
        results.append(r)
        reuse = r["cached_tokens"] / r["prompt_tokens"] if r["prompt_tokens"] else 0.0
        print(f"{'PASS' if r['pass'] else 'FAIL'}  {r['id']:12s} {r['wall_s']:6.1f} s  {r['requests']:2d} requests  "
              f"prompt {r['prompt_tokens']:6d} tok, {reuse:4.0%} reused  {'; '.join(r['problems'])}", flush=True)
    passed = sum(r["pass"] for r in results)
    print(f"{passed}/{len(results)} tasks passed")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
