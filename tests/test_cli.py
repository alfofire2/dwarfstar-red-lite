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

    def test_sweep_contexts(self):
        a = self.p.parse_args(["sweep", "x.gguf", "--contexts", "2048,4096,8192"])
        self.assertEqual(a.contexts, [2048, 4096, 8192])


if __name__ == "__main__":
    unittest.main()
