"""OpenAI protocol tests for the native HTTP server core (native/redlite_native_server.c).

The server core is compiled together with the deterministic echo backend
(native/redlite_native_server_fake.c) into a temporary binary, so no model file
and no Metal are needed; the tests are skipped when no C compiler is available.
"""
from __future__ import annotations

import http.client
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "native"


def _compiler() -> str | None:
    cc = os.environ.get("CC")
    if cc and shutil.which(cc):
        return cc
    for name in ("cc", "clang", "gcc"):
        if shutil.which(name):
            return name
    return None


class FakeServer:
    def __init__(self, binary: Path, *args: str):
        self.proc = subprocess.Popen(
            [str(binary), "--port", "0", *args],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        line = self.proc.stdout.readline()
        m = re.search(r"http://127\.0\.0\.1:(\d+)", line)
        if not m:
            self.proc.kill()
            raise RuntimeError(f"server did not start: {line!r} {self.proc.stderr.read()}")
        self.port = int(m.group(1))

    def request(self, method: str, path: str, body=None, raw: bytes | None = None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        payload = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        headers = {"Content-Type": "application/json"} if payload is not None else {}
        conn.request(method, path, body=payload, headers=headers)
        resp = conn.getresponse()
        data = resp.read()
        conn.close()
        return resp, data

    def stop(self) -> int:
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGINT)
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.proc.stdout.close()
        self.proc.stderr.close()
        return self.proc.returncode


def parse_sse(data: bytes) -> tuple[list[dict], bool]:
    events, done = [], False
    for block in data.decode("utf-8").split("\n\n"):
        if not block.strip():
            continue
        assert block.startswith("data: "), block
        payload = block[len("data: "):]
        if payload == "[DONE]":
            done = True
            continue
        events.append(json.loads(payload))
    return events, done


def user(content: str, **extra) -> dict:
    return {"model": "redlite", "messages": [{"role": "user", "content": content}], **extra}


@unittest.skipUnless(_compiler(), "no C compiler available")
class NativeServerProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        prebuilt = os.environ.get("REDLITE_SERVER_FAKE_BIN")  # e.g. the sanitizer build from scripts/sanitize_offline.sh
        if prebuilt:
            cls.binary = Path(prebuilt)
            cls.server = FakeServer(cls.binary)
            return
        cls.binary = Path(cls.tmp.name) / "redlite-server-fake"
        subprocess.run(
            [_compiler(), "-O1", "-std=c11", "-D_POSIX_C_SOURCE=200809L", "-Wall", "-Wextra", "-Wpedantic", "-Werror",
             f"-I{NATIVE}", str(NATIVE / "redlite_native_server.c"), str(NATIVE / "redlite_native_server_fake.c"),
             "-lm", "-pthread", "-o", str(cls.binary)],
            check=True, capture_output=True,
        )
        cls.server = FakeServer(cls.binary)

    @classmethod
    def tearDownClass(cls):
        cls.server.stop()
        cls.tmp.cleanup()

    def test_selftest(self):
        out = subprocess.run([str(self.binary), "--selftest"], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)

    def test_health_and_models(self):
        resp, data = self.server.request("GET", "/health")
        self.assertEqual(resp.status, 200)
        self.assertEqual(json.loads(data), {"status": "ok"})
        resp, data = self.server.request("GET", "/v1/models")
        self.assertEqual(resp.status, 200)
        models = json.loads(data)
        self.assertEqual(models["object"], "list")
        self.assertEqual(models["data"][0]["id"], "redlite-fake-echo")

    def test_non_streaming_completion(self):
        resp, data = self.server.request("POST", "/v1/chat/completions", user("Ciao è 😀 \"q\"\n"))
        self.assertEqual(resp.status, 200)
        self.assertEqual(resp.getheader("Content-Type"), "application/json")
        body = json.loads(data)
        self.assertEqual(body["object"], "chat.completion")
        self.assertTrue(body["id"].startswith("chatcmpl-"))
        choice = body["choices"][0]
        self.assertEqual(choice["message"]["role"], "assistant")
        self.assertEqual(choice["message"]["content"], "Echo: Ciao è 😀 \"q\"\n")
        self.assertEqual(choice["finish_reason"], "stop")
        prompt = "<|im_start|>user\nCiao è 😀 \"q\"\n<|im_end|>\n<|im_start|>assistant\n"
        usage = body["usage"]
        self.assertEqual(usage["prompt_tokens"], len(prompt.encode()))
        self.assertEqual(usage["completion_tokens"], -(-len("Echo: Ciao è 😀 \"q\"\n".encode()) // 3))
        self.assertEqual(usage["total_tokens"], usage["prompt_tokens"] + usage["completion_tokens"])

    def test_streaming_completion_reassembles_split_utf8(self):
        text = "àèìòù 😀 日本語"
        resp, data = self.server.request("POST", "/v1/chat/completions", user(text, stream=True))
        self.assertEqual(resp.status, 200)
        self.assertEqual(resp.getheader("Content-Type"), "text/event-stream")
        events, done = parse_sse(data)
        self.assertTrue(done)
        self.assertEqual(events[0]["choices"][0]["delta"], {"role": "assistant", "content": ""})
        self.assertTrue(all(e["object"] == "chat.completion.chunk" for e in events))
        self.assertEqual(len({e["id"] for e in events}), 1)
        content = "".join(e["choices"][0]["delta"].get("content", "") for e in events)
        self.assertEqual(content, "Echo: " + text)
        self.assertNotIn("�", content)  # no chunk ever split a code point
        self.assertEqual(events[-1]["choices"][0]["finish_reason"], "stop")
        self.assertEqual(events[-1]["choices"][0]["delta"], {})
        self.assertIn("usage", events[-1])

    def test_max_tokens_finishes_with_length(self):
        resp, data = self.server.request("POST", "/v1/chat/completions", user("abcdefghijkl", max_tokens=2))
        body = json.loads(data)
        self.assertEqual(body["choices"][0]["message"]["content"], "Echo: ")
        self.assertEqual(body["choices"][0]["finish_reason"], "length")
        self.assertEqual(body["usage"]["completion_tokens"], 2)
        resp, data = self.server.request("POST", "/v1/chat/completions", user("abcdefghijkl", max_tokens=2, stream=True))
        events, done = parse_sse(data)
        self.assertTrue(done)
        self.assertEqual(events[-1]["choices"][0]["finish_reason"], "length")

    def test_server_defaults_and_overrides_reach_the_backend(self):
        _, data = self.server.request("POST", "/v1/chat/completions", user("__params__", max_tokens=200))
        self.assertEqual(json.loads(data)["choices"][0]["message"]["content"],
                         "temperature=0.70 top_p=0.95 top_k=40 min_p=0.00 max_tokens=200 seed=none/0 presence=0.00 frequency=0.00")
        _, data = self.server.request("POST", "/v1/chat/completions",
                                      user("__params__", temperature=0, top_p=0.5, top_k=3, min_p=0.05, seed=9, max_tokens=200,
                                           presence_penalty=1.5, frequency_penalty=-0.5))
        self.assertEqual(json.loads(data)["choices"][0]["message"]["content"],
                         "temperature=0.00 top_p=0.50 top_k=3 min_p=0.05 max_tokens=200 seed=9 presence=1.50 frequency=-0.50")

    def test_multi_turn_and_system_messages_build_chatml(self):
        body = {"messages": [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "U1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": [{"type": "text", "text": "U"}, {"type": "text", "text": "2"}]},
        ]}
        _, data = self.server.request("POST", "/v1/chat/completions", body)
        out = json.loads(data)
        prompt = ("<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nU1<|im_end|>\n"
                  "<|im_start|>assistant\nA1<|im_end|>\n<|im_start|>user\nU2<|im_end|>\n<|im_start|>assistant\n")
        self.assertEqual(out["usage"]["prompt_tokens"], len(prompt.encode()))
        self.assertEqual(out["choices"][0]["message"]["content"], "Echo: U2")

    def assert_error(self, resp, data, status: int, needle: str):
        self.assertEqual(resp.status, status, data)
        err = json.loads(data)["error"]
        self.assertIn(needle, err["message"])
        self.assertIn(err["type"], {"invalid_request_error", "server_error"})

    def test_request_errors(self):
        resp, data = self.server.request("POST", "/v1/chat/completions", raw=b"{not json")
        self.assertEqual(resp.status, 400)
        resp, data = self.server.request("POST", "/v1/chat/completions", {"messages": []})
        self.assert_error(resp, data, 400, "at least one")
        resp, data = self.server.request("POST", "/v1/chat/completions", user("x", stop=["a", "b", "c", "d", "e"]))
        self.assert_error(resp, data, 400, "at most 4")
        resp, data = self.server.request("POST", "/v1/chat/completions", user("x", stop=""))
        self.assert_error(resp, data, 400, "stop")
        resp, data = self.server.request("GET", "/v1/chat/completions")
        self.assert_error(resp, data, 405, "POST")
        resp, data = self.server.request("GET", "/nope")
        self.assert_error(resp, data, 404, "unknown")
        resp, data = self.server.request("POST", "/v1/chat/completions", user("__too_long__"))
        self.assert_error(resp, data, 400, "context")
        resp, data = self.server.request("POST", "/v1/chat/completions", user("__fail__", stream=True))
        self.assert_error(resp, data, 500, "fake backend failure")

    def test_mid_stream_failure_sends_error_event_without_done(self):
        resp, data = self.server.request("POST", "/v1/chat/completions", user("__fail_late__", stream=True))
        self.assertEqual(resp.status, 200)
        events, done = parse_sse(data)
        self.assertFalse(done)
        self.assertIn("error", events[-1])
        self.assertEqual(events[-1]["error"]["type"], "server_error")

    def test_server_survives_a_client_that_disconnects_mid_stream(self):
        slow = FakeServer(self.binary, "--token-delay-ms", "50")
        try:
            s = socket.create_connection(("127.0.0.1", slow.port), timeout=5)
            body = json.dumps(user("x" * 90, stream=True)).encode()
            s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                      + f"Content-Length: {len(body)}\r\n\r\n".encode() + body)
            self.assertIn(b"text/event-stream", s.recv(4096))
            s.close()
            time.sleep(0.3)
            resp, data = slow.request("POST", "/v1/chat/completions", user("after"))
            self.assertEqual(resp.status, 200)
            self.assertEqual(json.loads(data)["choices"][0]["message"]["content"], "Echo: after")
        finally:
            slow.stop()

    def test_mutated_bodies_get_200_or_400_and_the_server_stays_up(self):
        import random
        rng = random.Random(1234)
        seed = json.dumps({"stream": False, "max_tokens": 3, "temperature": 0.5,
                           "messages": [{"role": "system", "content": "s\u00e8"},
                                        {"role": "user", "content": [{"type": "text", "text": "hi \ud83d\ude00"}]}]},
                          ensure_ascii=True).encode()
        alphabet = b'{}[]",:\\u0123456789eE+-.tfnrl \x00\xff\xc3'
        for _ in range(300):
            body = bytearray(seed)
            for _ in range(rng.randint(1, 4)):
                op = rng.randrange(3)
                at = rng.randrange(len(body)) if body else 0
                if op == 0 and body:
                    body[at] = rng.choice(alphabet)
                elif op == 1:
                    body[at:at] = bytes([rng.choice(alphabet)])
                elif body:
                    del body[at:at + rng.randint(1, 8)]
            resp, data = self.server.request("POST", "/v1/chat/completions", raw=bytes(body))
            self.assertIn(resp.status, (200, 400), data)
            json.loads(data)  # every answer is valid JSON, even for garbage input
        resp, _ = self.server.request("GET", "/health")
        self.assertEqual(resp.status, 200)

    def test_stop_sequences_cut_the_completion(self):
        # "Echo: one, two, three" is emitted in 3-byte chunks: ", t" spans chunk boundaries
        _, data = self.server.request("POST", "/v1/chat/completions", user("one, two, three", stop=", t"))
        body = json.loads(data)
        self.assertEqual(body["choices"][0]["message"]["content"], "Echo: one")
        self.assertEqual(body["choices"][0]["finish_reason"], "stop")
        _, data = self.server.request("POST", "/v1/chat/completions",
                                      user("one, two, three", stop=["zzz", "hree", "wo"], stream=True))
        events, done = parse_sse(data)
        self.assertTrue(done)
        content = "".join(e["choices"][0]["delta"].get("content", "") for e in events)
        self.assertEqual(content, "Echo: one, t")  # earliest match ("wo") wins, nothing after it is sent
        self.assertEqual(events[-1]["choices"][0]["finish_reason"], "stop")
        # a held-back partial match that never completes is flushed at the end
        _, data = self.server.request("POST", "/v1/chat/completions", user("abc ta", stop="tail"))
        self.assertEqual(json.loads(data)["choices"][0]["message"]["content"], "Echo: abc ta")
        # multi-byte stop sequence, split across chunks
        _, data = self.server.request("POST", "/v1/chat/completions", user("x è 😀 y", stop="😀"))
        self.assertEqual(json.loads(data)["choices"][0]["message"]["content"], "Echo: x è ")

    def test_state_reuse_only_when_the_prompt_extends_the_previous_turn(self):
        srv = FakeServer(self.binary)
        try:
            def chat(messages, **extra):
                resp, data = srv.request("POST", "/v1/chat/completions", {"messages": messages, **extra})
                self.assertEqual(resp.status, 200, data)
                return json.loads(data)
            first = [{"role": "user", "content": "hi"}]
            a = chat(first)
            self.assertEqual(a["usage"]["prompt_tokens_details"]["cached_tokens"], 0)
            reply = a["choices"][0]["message"]["content"]
            second = first + [{"role": "assistant", "content": reply}, {"role": "user", "content": "more"}]
            b = chat(second)
            state = len(("<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n" + reply).encode())
            self.assertEqual(b["usage"]["prompt_tokens_details"]["cached_tokens"], state)
            # an edited earlier turn does not extend the state: full reset
            edited = [{"role": "user", "content": "HI"}, {"role": "assistant", "content": reply}, {"role": "user", "content": "more"}]
            self.assertEqual(chat(edited)["usage"]["prompt_tokens_details"]["cached_tokens"], 0)
            # the same prompt again (not a proper extension) is not reused either
            self.assertEqual(chat(edited)["usage"]["prompt_tokens_details"]["cached_tokens"], 0)
            # a reply cut by a stop sequence differs from the fed-back tokens: no reuse on the next turn
            c = chat(first, stop="ho")
            self.assertEqual(c["choices"][0]["message"]["content"], "Ec")
            nxt = first + [{"role": "assistant", "content": "Ec"}, {"role": "user", "content": "x"}]
            self.assertEqual(chat(nxt)["usage"]["prompt_tokens_details"]["cached_tokens"], 0)
            # a failed request leaves no reusable state
            chat(first)
            srv.request("POST", "/v1/chat/completions", {"messages": first + [{"role": "assistant", "content": "Echo: hi"}, {"role": "user", "content": "__fail__"}]})
            again = first + [{"role": "assistant", "content": "Echo: hi"}, {"role": "user", "content": "ok"}]
            self.assertEqual(chat(again)["usage"]["prompt_tokens_details"]["cached_tokens"], 0)
        finally:
            srv.stop()

    def test_concurrent_requests_run_in_arrival_order(self):
        import threading
        srv = FakeServer(self.binary, "--token-delay-ms", "20")
        try:
            done, lock = [], threading.Lock()
            def worker(tag):
                resp, data = srv.request("POST", "/v1/chat/completions", user(tag * 12))
                with lock:
                    done.append((tag, resp.status, json.loads(data)["choices"][0]["message"]["content"]))
            threads = []
            for tag in "ABCD":
                t = threading.Thread(target=worker, args=(tag,))
                t.start()
                threads.append(t)
                time.sleep(0.05)  # arrival order A, B, C, D
            # /health is answered while the queue is busy
            t0 = time.monotonic()
            resp, _ = srv.request("GET", "/health")
            self.assertEqual(resp.status, 200)
            self.assertLess(time.monotonic() - t0, 0.5)
            for t in threads:
                t.join(timeout=30)
            self.assertEqual([d[0] for d in done], list("ABCD"))
            for tag, status, content in done:
                self.assertEqual(status, 200)
                self.assertEqual(content, "Echo: " + tag * 12)
        finally:
            srv.stop()

    def test_two_workers_overlap_generations(self):
        """dev56: --workers 2 runs two requests at the same time (the real server pairs their decode steps)."""
        import threading
        srv = FakeServer(self.binary, "--token-delay-ms", "20", "--workers", "2")
        try:
            t0 = time.monotonic()
            srv.request("POST", "/v1/chat/completions", user("S" * 60))
            single = time.monotonic() - t0
            out = []
            def worker(tag):
                resp, data = srv.request("POST", "/v1/chat/completions", user(tag * 60))
                out.append((resp.status, json.loads(data)["choices"][0]["message"]["content"]))
            threads = [threading.Thread(target=worker, args=(t,)) for t in "AB"]
            t0 = time.monotonic()
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30)
            both = time.monotonic() - t0
            self.assertEqual(sorted(out), [(200, "Echo: " + "A" * 60), (200, "Echo: " + "B" * 60)])
            self.assertLess(both, 1.6 * single)   # serial would take 2x
        finally:
            srv.stop()

    def test_full_queue_answers_503(self):
        import threading
        srv = FakeServer(self.binary, "--token-delay-ms", "40", "--queue", "1")
        try:
            results = []
            def worker():
                resp, data = srv.request("POST", "/v1/chat/completions", user("x" * 30))
                results.append(resp.status)
            threads = [threading.Thread(target=worker) for _ in range(2)]
            for t in threads:
                t.start()
                time.sleep(0.05)  # one running, one waiting
            resp, data = srv.request("POST", "/v1/chat/completions", user("third"))
            self.assert_error(resp, data, 503, "busy")
            for t in threads:
                t.join(timeout=30)
            self.assertEqual(results, [200, 200])
        finally:
            srv.stop()

    def test_sigint_shuts_down_cleanly(self):
        srv = FakeServer(self.binary)
        resp, _ = srv.request("GET", "/health")
        self.assertEqual(resp.status, 200)
        srv.proc.send_signal(signal.SIGINT)
        srv.proc.wait(timeout=10)
        err = srv.proc.stderr.read()
        self.assertEqual(srv.stop(), 0)
        self.assertIn("shut down cleanly", err)


if __name__ == "__main__":
    unittest.main()
