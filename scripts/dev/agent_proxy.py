#!/usr/bin/env python3
"""dev83: record what a coding agent asks the server and what the model answers, to study agent loops.

A pass-through proxy (stdlib only) between the agent and `redlite serve --native`. Every /v1/chat/completions
request is forwarded unchanged; streamed answers are passed on event by event as they arrive. One JSON line per
request goes to --log:
- the request body;
- the answer as the client saw it (text and tool calls rebuilt from the stream);
- the finish reason and the wall time.

    python3 scripts/dev/agent_proxy.py --port 8090 --upstream 127.0.0.1:8080 --log agent.jsonl
    (point the agent at port 8090)
"""
import argparse
import http.client
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOCK = threading.Lock()


def rebuild(events):
    """the assistant message of a list of streamed chat.completion.chunk objects"""
    text, calls, finish = [], {}, None
    for ev in events:
        for choice in ev.get("choices", []):
            delta = choice.get("delta") or {}
            if delta.get("content"):
                text.append(delta["content"])
            for tc in delta.get("tool_calls") or []:
                c = calls.setdefault(tc.get("index", 0), {"id": None, "name": "", "arguments": ""})
                c["id"] = tc.get("id") or c["id"]
                fn = tc.get("function") or {}
                c["name"] += fn.get("name") or ""
                c["arguments"] += fn.get("arguments") or ""
            finish = choice.get("finish_reason") or finish
    return {"content": "".join(text), "tool_calls": [calls[k] for k in sorted(calls)], "finish_reason": finish}


class Proxy(BaseHTTPRequestHandler):
    upstream: tuple = ("127.0.0.1", 8080)
    log_path = "agent.jsonl"

    def log_message(self, format, *args):   # quiet
        pass

    def _forward(self, method):
        t0 = time.time()
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0)) if method == "POST" else b""
        conn = http.client.HTTPConnection(*self.upstream, timeout=3600)
        headers = {k: v for k, v in self.headers.items() if k.lower() not in ("host", "content-length", "connection")}
        conn.request(method, self.path, body=body or None, headers=headers)
        resp = conn.getresponse()
        self.send_response(resp.status)
        for k, v in resp.getheaders():
            if k.lower() not in ("transfer-encoding", "connection", "content-length"):
                self.send_header(k, v)
        self.send_header("Connection", "close")
        self.end_headers()
        streamed = "text/event-stream" in (resp.getheader("Content-Type") or "")
        events, raw = [], b""
        while True:
            line = resp.readline() if streamed else resp.read()
            if not line:
                break
            self.wfile.write(line)
            self.wfile.flush()
            if streamed and line.startswith(b"data: ") and line.strip() != b"data: [DONE]":
                try:
                    events.append(json.loads(line[6:]))
                except ValueError:
                    pass
            elif not streamed:
                raw += line
        conn.close()
        if method == "POST" and self.path.startswith("/v1/chat/completions"):
            if streamed:
                answer = rebuild(events)
            else:
                try:
                    msg = json.loads(raw)["choices"][0]
                    answer = {"content": msg["message"].get("content") or "", "finish_reason": msg.get("finish_reason"),
                              "tool_calls": [{"id": t.get("id"), "name": t["function"]["name"], "arguments": t["function"]["arguments"]}
                                             for t in msg["message"].get("tool_calls") or []]}
                except (ValueError, KeyError, IndexError):
                    answer = {"raw": raw.decode("utf-8", "replace")}
            try:
                request = json.loads(body)
            except ValueError:
                request = {"raw": body.decode("utf-8", "replace")}
            entry = {"t": round(t0, 3), "seconds": round(time.time() - t0, 3), "status": resp.status,
                     "request": request, "answer": answer}
            with LOCK, open(self.log_path, "a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def do_POST(self):
        self._forward("POST")

    def do_GET(self):
        self._forward("GET")


def main():
    ap = argparse.ArgumentParser(description="record agent requests and answers through a pass-through proxy")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--upstream", default="127.0.0.1:8080", help="host:port of redlite serve --native")
    ap.add_argument("--log", default="agent.jsonl")
    args = ap.parse_args()
    host, port = args.upstream.rsplit(":", 1)
    Proxy.upstream, Proxy.log_path = (host, int(port)), args.log
    ThreadingHTTPServer(("127.0.0.1", args.port), Proxy).serve_forever()


if __name__ == "__main__":
    main()
