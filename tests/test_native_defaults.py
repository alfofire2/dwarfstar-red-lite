import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from redlite.cli import main
from redlite.hardware import GIB, HardwareInfo
from redlite.planner import (
    NATIVE_BOUNDED_CACHE_MIB,
    NATIVE_FULL_RESIDENCY_CACHE_MIB,
    native_defaults,
)


def _hw(ram_gib: float) -> HardwareInfo:
    return HardwareInfo(
        system="Darwin", machine="arm64", chip="Apple M4", ram_bytes=int(ram_gib * GIB),
        logical_cpus=12, perf_cpus=8, free_disk_bytes=500 * GIB, cwd="/",
    )


class NativeDefaultsTests(unittest.TestCase):
    def test_48gb_gets_full_residency(self):
        d = native_defaults(48 * GIB)
        self.assertTrue(d.full_residency)
        self.assertEqual(d.cache_mib, NATIVE_FULL_RESIDENCY_CACHE_MIB)
        self.assertEqual(d.cache_mib, 22528)

    def test_threshold_is_inclusive_at_40gb(self):
        self.assertTrue(native_defaults(40 * GIB).full_residency)
        self.assertFalse(native_defaults(40 * GIB - 1).full_residency)

    def test_24gb_gets_bounded_4gb_cache(self):
        d = native_defaults(24 * GIB)
        self.assertFalse(d.full_residency)
        self.assertEqual(d.cache_mib, NATIVE_BOUNDED_CACHE_MIB)
        self.assertEqual(d.cache_mib, 4096)


class NativeChatDefaultsCliTests(unittest.TestCase):
    def _run(self, ram_gib: float, *extra: str) -> str:
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "m.gguf"
            model.write_bytes(b"GGUF")
            out = io.StringIO()
            with patch("redlite.cli.detect", return_value=_hw(ram_gib)), \
                 patch("redlite.runner.native_generate", return_value=Path("/x/redlite-generate")), \
                 contextlib.redirect_stdout(out):
                self.assertEqual(main(["chat", str(model), "--dry-run", *extra]), 0)
            return out.getvalue()

    def test_chat_without_flags_on_48gb_uses_full_residency(self):
        text = self._run(48)
        self.assertIn("--cache-mib 22528", text)
        self.assertIn("full expert residency", text)
        self.assertNotIn("--batch", text)

    def test_chat_without_flags_on_24gb_uses_4gb(self):
        text = self._run(24)
        self.assertIn("--cache-mib 4096", text)
        self.assertIn("bounded 4 GiB", text)

    def test_explicit_cache_and_batch_win(self):
        text = self._run(48, "--cache-mib", "1024", "--batch", "128")
        self.assertIn("--cache-mib 1024", text)
        self.assertIn("--batch 128", text)
        self.assertNotIn("full expert residency", text)


if __name__ == "__main__":
    unittest.main()
