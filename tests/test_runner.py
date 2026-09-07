import contextlib
import io
from pathlib import Path
import unittest
from unittest.mock import patch

from redlite.runner import run_native_chat


class NativeChatRunnerTests(unittest.TestCase):
    @patch("redlite.runner.native_generate", return_value=Path("/tmp/redlite-generate"))
    def test_builds_native_interactive_command(self, _native_generate):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = run_native_chat(
                "model.gguf", context=2048, cache_mib=1024, max_tokens=64,
                temperature=0.7, top_k=40, top_p=0.9, seed=42,
                system="Rispondi in italiano", prompt="Ciao", stats=True,
                no_stream=True, dry_run=True,
            )

        self.assertEqual(result, 0)
        command = output.getvalue()
        self.assertIn("/tmp/redlite-generate model.gguf --interactive", command)
        self.assertIn("--context 2048", command)
        self.assertIn("--cache-mib 1024", command)
        self.assertIn("--max-tokens 64", command)
        self.assertIn("--temperature 0.7", command)
        self.assertIn("--top-k 40", command)
        self.assertIn("--top-p 0.9", command)
        self.assertIn("--seed 42", command)
        self.assertIn("--system 'Rispondi in italiano'", command)
        self.assertIn("--prompt Ciao", command)
        self.assertIn("--stats", command)
        self.assertIn("--no-stream", command)


if __name__ == "__main__":
    unittest.main()
