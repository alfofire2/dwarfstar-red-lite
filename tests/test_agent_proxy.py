import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("agent_proxy", ROOT / "scripts" / "dev" / "agent_proxy.py")
agent_proxy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent_proxy)


class RebuildTests(unittest.TestCase):
    def test_text_and_split_tool_call_arguments(self):
        chunks = [
            {"choices": [{"delta": {"content": "Let me "}}]},
            {"choices": [{"delta": {"content": "look."}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "read", "arguments": '{"pa'}}]}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'th": "a.py"}'}}]}}]},
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        ]
        a = agent_proxy.rebuild(chunks)
        self.assertEqual(a["content"], "Let me look.")
        self.assertEqual(a["tool_calls"], [{"id": "c1", "name": "read", "arguments": '{"path": "a.py"}'}])
        self.assertEqual(a["finish_reason"], "tool_calls")


if __name__ == "__main__":
    unittest.main()
