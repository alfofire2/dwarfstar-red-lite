import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("polyglot_eval", ROOT / "scripts" / "dev" / "polyglot_eval.py")
pg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pg)


class ApplyAnswerTests(unittest.TestCase):
    def test_answer_formats(self):
        with tempfile.TemporaryDirectory() as t:
            w = Path(t)
            (w / "src").mkdir()
            for f in ("a.py", "src/lib.rs", "Cargo.toml", "x.h", "x.cpp"):
                (w / f).write_text("old")
            self.assertEqual(pg.apply_answer("```python\nA\n```", w, ["a.py"]), ["a.py"])   # unnamed, at the start
            self.assertEqual(pg.apply_answer("Here:\nsrc/lib.rs\n```rust\nR\n```\nCargo.toml\n```toml\nT\n```", w,
                                             ["src/lib.rs", "Cargo.toml"]), ["src/lib.rs", "Cargo.toml"])
            self.assertEqual(pg.apply_answer("**x.h**\n```cpp\nH\n```\n\n`x.cpp`\n```cpp\nC\n```", w, ["x.cpp", "x.h"]),
                             ["x.h", "x.cpp"])
            self.assertEqual([(w / f).read_text() for f in ("a.py", "src/lib.rs", "x.h", "x.cpp")], ["A\n", "R\n", "H\n", "C\n"])
            self.assertEqual(pg.apply_answer("two blocks, no names\n```cpp\nX\n```\n```cpp\nY\n```", w, ["x.cpp", "x.h"]), [])


if __name__ == "__main__":
    unittest.main()
