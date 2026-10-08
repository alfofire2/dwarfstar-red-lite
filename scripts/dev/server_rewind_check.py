#!/usr/bin/env python3
"""dev77: the server returns to the end of the previous prompt when a client re-renders the model's answer.

Request 1 gets a greedy answer; request 2 sends that answer with one character changed in its middle (as when an
agent re-serializes a tool call) plus a new question. The server must rewind to the mark left after request 1's prompt
("back to the end of the previous prompt") and answer exactly as a freshly started server does.

    python3 scripts/dev/server_rewind_check.py MODEL [BIN_DIR]
"""
import json, subprocess, sys, urllib.request
MODEL = sys.argv[1]
BIN = sys.argv[2] if len(sys.argv) > 2 else ".deps/redmetal"
def start(log):
    p = subprocess.Popen([BIN + "/redlite-server", MODEL, "--port", "8099", "--cache-mib", "full", "--context", "8192"], stdout=subprocess.PIPE, stderr=open(log, "w"), text=True)
    for _ in range(300):
        line = p.stdout.readline()
        if "listening" in line: return p
    raise SystemExit("server did not start")
def chat(msgs):
    body = json.dumps({"model": "x", "messages": msgs, "max_tokens": 80, "temperature": 0}).encode()
    r = urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8099/v1/chat/completions", body, {"Content-Type": "application/json"}), timeout=600)
    return json.loads(r.read())["choices"][0]["message"]["content"]
q1 = "Write a Python function that returns the n-th Fibonacci number iteratively."
q2 = "Now add type hints and a docstring to it."
p = start("/tmp/rw_warm.log")
a1 = chat([{"role": "user", "content": q1}])
msgs2 = [{"role": "user", "content": q1}, {"role": "assistant", "content": a1[:30] + " " + a1[30:]}, {"role": "user", "content": q2}]
warm = chat(msgs2)
p.send_signal(2); p.wait()
p = start("/tmp/rw_cold.log")
cold = chat(msgs2)
p.send_signal(2); p.wait()
log = open("/tmp/rw_warm.log").read()
rewound = "back to the end of the previous prompt" in log
print("rewound:", rewound)
print("SERVER REWIND CHECK:", "YES" if rewound and warm == cold else "NO")
if warm != cold: print(repr(warm[:200])); print(repr(cold[:200]))
