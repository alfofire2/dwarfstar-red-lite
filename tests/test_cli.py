import unittest

from redlite.cli import build_parser


class CliTests(unittest.TestCase):
    def setUp(self):
        self.p = build_parser()

    def test_run_regular_args_not_swallowed(self):
        a = self.p.parse_args(["run", "x.gguf", "-p", "hello", "-n", "10", "--single-turn", "--dry-run"])
        self.assertEqual(a.prompt, "hello")
        self.assertEqual(a.tokens, 10)
        self.assertTrue(a.single_turn)
        self.assertTrue(a.dry_run)

    def test_engine_args_are_remainder(self):
        a = self.p.parse_args(["run", "x.gguf", "--engine-args", "--temp", "0.5"])
        self.assertEqual(a.extra, ["--temp", "0.5"])

    def test_native_chat_defaults(self):
        a = self.p.parse_args(["chat"])
        self.assertIsNone(a.model)  # dev31: chosen at run time among models/ (redlite.planner.select_native_model)
        self.assertEqual(a.context, 4096)
        self.assertIsNone(a.cache_mib)  # chosen from RAM at run time (redlite.planner.native_defaults)
        self.assertIsNone(a.batch)
        self.assertEqual(a.max_tokens, 256)
        self.assertEqual(a.temperature, 0.7)
        self.assertEqual(a.top_k, 40)
        self.assertEqual(a.top_p, 0.95)

    def test_native_chat_accepts_model_override(self):
        a = self.p.parse_args(["chat", "x.gguf"])
        self.assertEqual(a.model, "x.gguf")

    def test_native_chat_options(self):
        a = self.p.parse_args([
            "chat", "x.gguf", "--prompt", "ciao", "--system", "Rispondi in italiano",
            "-c", "2048", "--cache-mib", "1024", "-n", "64",
            "--temperature", "0", "--top-k", "0", "--top-p", "1", "--seed", "42",
            "--stats", "--no-stream", "--dry-run", "--batch", "64",
        ])
        self.assertEqual(a.batch, 64)
        self.assertEqual(a.prompt, "ciao")
        self.assertEqual(a.system, "Rispondi in italiano")
        self.assertEqual(a.context, 2048)
        self.assertEqual(a.cache_mib, 1024)
        self.assertEqual(a.max_tokens, 64)
        self.assertEqual(a.temperature, 0.0)
        self.assertEqual(a.top_k, 0)
        self.assertEqual(a.top_p, 1.0)
        self.assertEqual(a.seed, 42)
        self.assertTrue(a.stats)
        self.assertTrue(a.no_stream)
        self.assertTrue(a.dry_run)

    def test_sweep_contexts(self):
        a = self.p.parse_args(["sweep", "x.gguf", "--contexts", "2048,4096,8192"])
        self.assertEqual(a.contexts, [2048, 4096, 8192])

    def test_models_json_flag(self):
        import json
        import sys
        from io import StringIO
        import unittest
        from unittest.mock import patch

        # Test --json flag outputs JSON list
        a = self.p.parse_args(["models", "--json"])
        self.assertTrue(a.json)

        # Mock sys.stdout to capture output
        with patch('sys.stdout', new=StringIO()) as out:
            # Need to re-run cmd_models since it's a function, not a method
            from redlite.cli import cmd_models
            cmd_models(a)
            output = out.getvalue()
            data = json.loads(output)
            self.assertIsInstance(data, list)
            self.assertGreater(len(data), 0)
            self.assertEqual(set(data[0].keys()), {"key", "repo", "filename", "nominal_gb", "quality", "recommended_mode"})


if __name__ == "__main__":
    unittest.main()


class SteerArgsTests(unittest.TestCase):
    def test_chat_passes_steering_options(self):
        import argparse
        from redlite.cli import _steer_args
        a = argparse.Namespace(steer="v.f32", steer_layers="12-23", steer_strength=0.3, steer_tokens=None, history="h.txt")
        self.assertEqual(_steer_args(a), ["--steer", "v.f32", "--steer-layers", "12-23", "--steer-strength", "0.3", "--history", "h.txt"])
        self.assertEqual(_steer_args(argparse.Namespace(steer=None, steer_layers=None, steer_strength=None, steer_tokens=None)), [])


class InstalledLayoutTests(unittest.TestCase):
    """dev57: REDLITE_MODELS overrides the models folder; native binaries are also found on PATH."""

    def test_models_dir_override(self):
        import os
        import subprocess
        import sys
        out = subprocess.check_output([sys.executable, "-c", "from redlite.cli import NATIVE_MODELS_DIR; print(NATIVE_MODELS_DIR)"],
                                      env={**os.environ, "REDLITE_MODELS": "/tmp/rl-models-test"}, text=True)
        self.assertEqual(out.strip(), "/tmp/rl-models-test")

    def test_native_binary_from_path(self):
        import os
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from redlite import runner
        with tempfile.TemporaryDirectory() as d:
            exe = Path(d) / "redlite-generate"
            exe.write_text("#!/bin/sh\n"); exe.chmod(0o755)
            with patch.object(runner, "REDMETAL_BIN", Path(d) / "missing"), patch.dict(os.environ, {"PATH": d}):
                self.assertEqual(runner.native_generate(), exe)


class DetectMissingDirTests(unittest.TestCase):
    def test_detect_on_missing_nested_dir(self):
        from redlite.hardware import detect
        self.assertGreater(detect("/tmp/rl-missing-a/b/c").free_disk_bytes, 0)


class SetupPiTests(unittest.TestCase):
    """dev61: redlite setup-pi adds the provider and keeps the others."""

    def test_merges_into_existing_models_json(self):
        import json
        import os
        import subprocess
        import sys
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "models.json").write_text(json.dumps({"providers": {"ollama": {"baseUrl": "x"}}}))
            out = subprocess.run([sys.executable, "-m", "redlite.cli", "setup-pi", "--port", "8091", "--context", "16384"],
                                 env={**os.environ, "PI_CODING_AGENT_DIR": d}, capture_output=True, text=True)
            self.assertEqual(out.returncode, 0, out.stderr)
            cfg = json.loads((Path(d) / "models.json").read_text())
            self.assertEqual(set(cfg["providers"]), {"ollama", "redlite"})
            red = cfg["providers"]["redlite"]
            self.assertEqual(red["baseUrl"], "http://127.0.0.1:8091/v1")
            self.assertEqual(red["models"][0]["contextWindow"], 16384)

    def test_permissions_are_kept_or_private(self):
        import os
        import stat
        import subprocess
        import sys
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            env = {**os.environ, "PI_CODING_AGENT_DIR": d}
            cmd = [sys.executable, "-m", "redlite.cli", "setup-pi"]
            subprocess.run(cmd, env=env, check=True, capture_output=True)
            path = Path(d) / "models.json"
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)   # new file: private
            path.chmod(0o640)
            subprocess.run(cmd, env=env, check=True, capture_output=True)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o640)   # existing file: unchanged
