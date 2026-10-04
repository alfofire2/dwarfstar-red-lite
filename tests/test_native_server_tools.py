"""dev59: OpenAI tool calling in the native server core, against the fake backend.

- Prompt rendering: the C renderer must give exactly what the models' own Jinja chat templates give
  (tests/fixtures/chat_template_*.jinja, copied from the GGUFs), for the Qwen3-Next JSON format and the
  Qwen3-Coder XML format. Skipped without jinja2.
- Output: tool calls the "model" writes (the fake's __emit__: mode) come back as OpenAI tool_calls.
- Round trip: the next turn, rendered from the returned tool_calls, extends the previous state exactly.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from test_native_server import FakeServer, NATIVE, _compiler, parse_sse

FIXTURES = Path(__file__).resolve().parent / "fixtures"

try:
    import jinja2
except ImportError:  # pragma: no cover
    jinja2 = None

WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "  Current weather for a city.\n",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string", "description": "City name"},
        "days": {"type": "integer", "description": "Forecast days", "minimum": 1},
        "units": {"type": "string", "enum": ["metric", "imperial"]}},
        "required": ["city"]}}}
SEARCH = {"type": "function", "function": {"name": "search", "parameters": {
    "type": "object", "properties": {"query": {"type": "string"}}}}}


def render_jinja(template: str, messages: list, tools: list | None) -> str:
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
    env.filters["tojson"] = lambda x, **_: json.dumps(x, ensure_ascii=False)
    return env.from_string(template).render(messages=messages, tools=tools, add_generation_prompt=True)


def as_template_messages(messages: list, xml: bool) -> list:
    """What the template sees: OpenAI messages, with arguments as objects for the XML template (it iterates them)."""
    out = []
    for m in messages:
        m = json.loads(json.dumps(m))
        if m.get("content") is None:
            m["content"] = ""
        for c in m.get("tool_calls", []):
            if xml and isinstance(c["function"]["arguments"], str):
                c["function"]["arguments"] = json.loads(c["function"]["arguments"])
        out.append(m)
    return out


CONVERSATIONS = [
    ([{"role": "user", "content": "Hi"}], None),
    ([{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Weather in Rome?"}], [WEATHER, SEARCH]),
    ([{"role": "user", "content": "Weather in Rome?"}], [WEATHER]),
    ([{"role": "system", "content": "Be brief."},
      {"role": "user", "content": "Weather in Rome and Oslo?"},
      {"role": "assistant", "content": "Checking.", "tool_calls": [
          {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Rome\", \"days\": 2}"}},
          {"id": "c2", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Oslo\"}"}}]},
      {"role": "tool", "tool_call_id": "c1", "content": "18 C, sunny"},
      {"role": "tool", "tool_call_id": "c2", "content": "4 C, rain"},
      {"role": "user", "content": "Thanks — and in °F?"}], [WEATHER]),
    ([{"role": "user", "content": "Search it"},
      {"role": "assistant", "content": None, "tool_calls": [
          {"id": "c1", "type": "function", "function": {"name": "search", "arguments": "{\"query\": \"a \\\"quoted\\\" thing\"}"}}]},
      {"role": "tool", "content": "nothing found"}], [SEARCH]),
]


@unittest.skipUnless(_compiler(), "no C compiler available")
class ToolCallTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        prebuilt = os.environ.get("REDLITE_SERVER_FAKE_BIN")   # the sanitizer build (scripts/sanitize_offline.sh)
        binary = Path(prebuilt) if prebuilt else Path(cls.tmp.name) / "redlite-server-fake"
        cls.binary = binary
        if not prebuilt:
            subprocess.run(
                [_compiler(), "-O1", "-std=c11", "-D_POSIX_C_SOURCE=200809L", "-Wall", "-Wextra", "-Wpedantic", "-Werror",
                 f"-I{NATIVE}", str(NATIVE / "redlite_native_server.c"), str(NATIVE / "redlite_native_server_fake.c"),
                 "-lm", "-pthread", "-o", str(binary)],
                check=True, capture_output=True,
            )
        cls.json_srv = FakeServer(binary, "--tool-format", "json")
        cls.xml_srv = FakeServer(binary, "--tool-format", "xml")

    @classmethod
    def tearDownClass(cls):
        cls.json_srv.stop()
        cls.xml_srv.stop()
        cls.tmp.cleanup()

    def chat(self, srv, messages, tools=None, **extra):
        body = {"model": "redlite", "messages": messages, **extra}
        if tools is not None:
            body["tools"] = tools
        resp, data = srv.request("POST", "/v1/chat/completions", body)
        return resp.status, json.loads(data)

    def rendered(self, srv, messages, tools):
        status, out = self.chat(srv, messages + [{"role": "user", "content": "__prompt__"}], tools, max_tokens=100000)
        self.assertEqual(status, 200, out)
        return bytes.fromhex(out["choices"][0]["message"]["content"]).decode("utf-8")

    @unittest.skipUnless(jinja2, "jinja2 not installed")
    def test_prompt_matches_the_jinja_templates(self):
        for name, srv, xml in (("chat_template_qwen3_next_instruct.jinja", self.json_srv, False),
                               ("chat_template_qwen3_coder_next.jinja", self.xml_srv, True)):
            template = (FIXTURES / name).read_text()
            for messages, tools in CONVERSATIONS:
                with self.subTest(template=name, messages=len(messages), tools=bool(tools)):
                    full = messages + [{"role": "user", "content": "__prompt__"}]
                    expected = render_jinja(template, as_template_messages(full, xml), tools)
                    self.assertEqual(self.rendered(srv, messages, tools), expected)

    def test_tool_choice_none_drops_the_tools(self):
        with_tools = self.rendered(self.json_srv, [{"role": "user", "content": "Hi"}], [WEATHER])
        self.assertIn("<tools>", with_tools)
        status, out = self.chat(self.json_srv, [{"role": "user", "content": "Hi"}, {"role": "user", "content": "__prompt__"}],
                                [WEATHER], tool_choice="none", max_tokens=100000)
        self.assertNotIn("<tools>", bytes.fromhex(out["choices"][0]["message"]["content"]).decode())

    def test_json_call_is_returned_as_tool_calls(self):
        emitted = 'Let me check.\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "Rome", "days": 2}}\n</tool_call>'
        status, out = self.chat(self.json_srv, [{"role": "user", "content": "__emit__:" + emitted}], [WEATHER])
        self.assertEqual(status, 200, out)
        choice = out["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertEqual(choice["message"]["content"], "Let me check.")
        [call] = choice["message"]["tool_calls"]
        self.assertEqual(call["type"], "function")
        self.assertEqual(call["function"]["name"], "get_weather")
        self.assertEqual(call["function"]["arguments"], '{"city": "Rome", "days": 2}')
        self.assertTrue(call["id"].startswith("call_"))

    def test_two_calls_and_no_content(self):
        emitted = ('<tool_call>\n{"name": "search", "arguments": {"query": "a"}}\n</tool_call>\n'
                   '<tool_call>\n{"name": "search", "arguments": {"query": "b"}}\n</tool_call>')
        status, out = self.chat(self.json_srv, [{"role": "user", "content": "__emit__:" + emitted}], [SEARCH])
        msg = out["choices"][0]["message"]
        self.assertIsNone(msg["content"])
        self.assertEqual([json.loads(c["function"]["arguments"])["query"] for c in msg["tool_calls"]], ["a", "b"])
        self.assertEqual(len({c["id"] for c in msg["tool_calls"]}), 2)

    def test_streaming_sends_content_then_tool_calls(self):
        emitted = 'Sure.\n<tool_call>\n{"name": "search", "arguments": {"query": "x"}}\n</tool_call>'
        resp, data = self.json_srv.request("POST", "/v1/chat/completions", {
            "messages": [{"role": "user", "content": "__emit__:" + emitted}], "tools": [SEARCH], "stream": True})
        events, done = parse_sse(data)
        self.assertTrue(done)
        deltas = [e["choices"][0]["delta"] for e in events]
        self.assertEqual("".join(d.get("content", "") for d in deltas), "Sure.")
        calls = [c for d in deltas for c in d.get("tool_calls", [])]
        self.assertEqual([(c["index"], c["function"]["name"], c["function"]["arguments"]) for c in calls],
                         [(0, "search", '{"query": "x"}')])
        self.assertEqual(events[-1]["choices"][0]["finish_reason"], "tool_calls")

    def test_xml_call_uses_the_schema_types(self):
        emitted = ('<tool_call>\n<function=get_weather>\n<parameter=city>\nNew York\n</parameter>\n'
                   '<parameter=days>\n3\n</parameter>\n<parameter=units>\n12\n</parameter>\n</function>\n</tool_call>')
        status, out = self.chat(self.xml_srv, [{"role": "user", "content": "__emit__:" + emitted}], [WEATHER])
        [call] = out["choices"][0]["message"]["tool_calls"]
        self.assertEqual(call["function"]["name"], "get_weather")
        # days is an integer in the schema; units is a string even though "12" parses as a number
        self.assertEqual(json.loads(call["function"]["arguments"]), {"city": "New York", "days": 3, "units": "12"})

    def test_wrong_closing_brackets_are_repaired(self):
        """dev60: the 2-bit model writes the right arguments but closes them wrong ('}}]}', '}}}}', or not at all)."""
        good = {"path": "stats.py", "edits": [{"oldText": "a]}", "newText": 'b{"x"'}]}
        body = json.dumps({"name": "edit", "arguments": good})
        assert body.endswith('"}]}}')
        for tail in ('"}}]}', '"}}}}', '"}]}}}', '"}]', '"}]}}\n'):
            emitted = "<tool_call>\n" + body[:-len('"}]}}')] + tail + "\n</tool_call>"
            with self.subTest(tail=tail):
                status, out = self.chat(self.json_srv, [{"role": "user", "content": "__emit__:" + emitted}], [SEARCH])
                choice = out["choices"][0]
                self.assertEqual(choice["finish_reason"], "tool_calls", choice)
                self.assertEqual(json.loads(choice["message"]["tool_calls"][0]["function"]["arguments"]), good)
        # an error that is not only in the closers is not guessed at
        status, out = self.chat(self.json_srv, [{"role": "user", "content": "__emit__:" +
                                '<tool_call>\n{"name": "edit", "arguments": {"path" "x"}}\n</tool_call>'}], [SEARCH])
        self.assertEqual(out["choices"][0]["finish_reason"], "stop")

    def test_malformed_call_comes_back_as_text(self):
        emitted = 'Trying.\n<tool_call>\n{"name": "search", "arguments": {"query": '
        status, out = self.chat(self.json_srv, [{"role": "user", "content": "__emit__:" + emitted}], [SEARCH])
        choice = out["choices"][0]
        self.assertEqual(choice["finish_reason"], "stop")
        self.assertNotIn("tool_calls", choice["message"])
        self.assertEqual(choice["message"]["content"], 'Trying.<tool_call>\n{"name": "search", "arguments": {"query": ')

    def test_without_tools_the_text_is_untouched(self):
        emitted = 'a <tool_call> b'
        status, out = self.chat(self.json_srv, [{"role": "user", "content": "__emit__:" + emitted}])
        self.assertEqual(out["choices"][0]["message"]["content"], emitted)

    def test_request_validation(self):
        status, out = self.chat(self.json_srv, [{"role": "user", "content": None}])
        self.assertEqual(status, 400)
        status, out = self.chat(self.json_srv, [{"role": "user", "content": "x"}], [{"type": "function", "function": {}}])
        self.assertEqual(status, 400)
        status, out = self.chat(self.json_srv, [{"role": "user", "content": "x"}], functions=[{"name": "f"}])
        self.assertEqual(status, 400)

    def test_mutated_tool_bodies_get_200_or_400_and_the_servers_stay_up(self):
        import random
        rng = random.Random(59)
        messages, tools = CONVERSATIONS[3]
        base = json.dumps({"messages": messages, "tools": tools, "max_tokens": 4}).encode()
        for srv in (self.json_srv, self.xml_srv):
            for _ in range(150):
                body = bytearray(base)
                for _ in range(rng.randint(1, 4)):
                    i = rng.randrange(len(body))
                    op = rng.random()
                    if op < 0.4:
                        body[i] = rng.randrange(256)
                    elif op < 0.7:
                        del body[i:i + rng.randint(1, 8)]
                    else:
                        body[i:i] = bytes(rng.choice(b'{}[]",:\\0123456789null') for _ in range(rng.randint(1, 4)))
                resp, _ = srv.request("POST", "/v1/chat/completions", raw=bytes(body))
                self.assertIn(resp.status, (200, 400))
            resp, _ = srv.request("GET", "/health")
            self.assertEqual(resp.status, 200)

    def test_blank_line_before_a_call_still_extends_the_state(self):
        """Qwen3-Next writes 'text.\\n\\n<tool_call>'; the template puts back one newline, so the content keeps one."""
        srv = FakeServer(self.binary, "--tool-format", "json")
        try:
            emitted = 'Reading it.\n\n<tool_call>\n{"name": "search", "arguments": {"query": "x"}}\n</tool_call>'
            first = [{"role": "user", "content": "__emit__:" + emitted}]
            body = {"messages": first, "tools": [SEARCH], "max_tokens": 1000}
            resp, data = srv.request("POST", "/v1/chat/completions", body)
            msg = json.loads(data)["choices"][0]["message"]
            self.assertEqual(msg["content"], "Reading it.\n")
            second = first + [msg, {"role": "tool", "tool_call_id": msg["tool_calls"][0]["id"], "content": "none"}]
            resp, data = srv.request("POST", "/v1/chat/completions", {**body, "messages": second})
            self.assertGreater(json.loads(data)["usage"]["prompt_tokens_details"]["cached_tokens"], 0)
        finally:
            srv.stop()

    def test_reserialized_arguments_still_extend_the_state(self):
        """Agents send back the arguments re-serialized ({"city":"Rome"}); the prompt uses json.dumps form anyway."""
        srv = FakeServer(self.binary, "--tool-format", "json")
        try:
            emitted = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Rome", "days": 2}}\n</tool_call>'
            first = [{"role": "user", "content": "__emit__:" + emitted}]
            body = {"messages": first, "tools": [WEATHER], "max_tokens": 1000}
            resp, data = srv.request("POST", "/v1/chat/completions", body)
            msg = json.loads(data)["choices"][0]["message"]
            for c in msg["tool_calls"]:
                c["function"]["arguments"] = json.dumps(json.loads(c["function"]["arguments"]), separators=(",", ":"))
            second = first + [msg, {"role": "tool", "tool_call_id": msg["tool_calls"][0]["id"], "content": "18 C"}]
            resp, data = srv.request("POST", "/v1/chat/completions", {**body, "messages": second})
            self.assertGreater(json.loads(data)["usage"]["prompt_tokens_details"]["cached_tokens"], 0)
        finally:
            srv.stop()

    def test_next_turn_extends_the_state_exactly(self):
        """The assistant turn rendered from the returned tool_calls equals what the model wrote, so the engine keeps
        its state (cached_tokens) instead of re-reading the whole prompt in an agent loop."""
        srv = FakeServer(self.binary, "--tool-format", "json")
        try:
            emitted = 'Checking.\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "Rome"}}\n</tool_call>'
            first = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "__emit__:" + emitted}]
            body = {"messages": first, "tools": [WEATHER], "max_tokens": 1000}
            resp, data = srv.request("POST", "/v1/chat/completions", body)
            msg = json.loads(data)["choices"][0]["message"]
            second = first + [msg, {"role": "tool", "tool_call_id": msg["tool_calls"][0]["id"], "content": "18 C"}]
            resp, data = srv.request("POST", "/v1/chat/completions", {**body, "messages": second})
            usage = json.loads(data)["usage"]
            self.assertGreater(usage["prompt_tokens_details"]["cached_tokens"], 0)
        finally:
            srv.stop()


if __name__ == "__main__":
    unittest.main()
