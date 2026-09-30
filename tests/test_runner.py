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


class ForegroundCallTests(unittest.TestCase):
    def test_ctrl_c_reaches_only_the_native_child(self):
        import signal
        import sys
        from redlite.runner import call_native_foreground

        # The child delivers SIGINT to its parent and to itself (what the terminal does to the
        # foreground group) and exits 130 from its own handler; the parent must neither die nor
        # raise KeyboardInterrupt.
        child = (
            "import os, signal, sys, time\n"
            "signal.signal(signal.SIGINT, lambda *a: sys.exit(130))\n"
            "os.kill(os.getppid(), signal.SIGINT)\n"
            "time.sleep(0.2)\n"
            "os.kill(os.getpid(), signal.SIGINT)\n"
            "time.sleep(5)\n"
        )
        before = signal.getsignal(signal.SIGINT)
        rc = call_native_foreground([sys.executable, "-c", child])
        self.assertEqual(rc, 130)
        self.assertIs(signal.getsignal(signal.SIGINT), before)


if __name__ == "__main__":
    unittest.main()
