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


if __name__ == "__main__":
    unittest.main()


class SteerArgsTests(unittest.TestCase):
    def test_chat_passes_steering_options(self):
        import argparse
        from redlite.cli import _steer_args
        a = argparse.Namespace(steer="v.f32", steer_layers="12-23", steer_strength=0.3, steer_tokens=None, history="h.txt")
        self.assertEqual(_steer_args(a), ["--steer", "v.f32", "--steer-layers", "12-23", "--steer-strength", "0.3", "--history", "h.txt"])
        self.assertEqual(_steer_args(argparse.Namespace(steer=None, steer_layers=None, steer_strength=None, steer_tokens=None)), [])
