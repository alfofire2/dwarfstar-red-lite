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
    NativeResidency,
    native_defaults,
    native_residency,
    select_native_model,
)

# a GGUF whose experts need 21312 MiB (the IQ2_XXS reference file) and 1.06 GiB of dense weights
IQ2 = NativeResidency(21312, int(1.06 * GIB))
IQ3 = NativeResidency(29376, int(1.42 * GIB))


def _hw(ram_gib: float) -> HardwareInfo:
    return HardwareInfo(
        system="Darwin", machine="arm64", chip="Apple M4", ram_bytes=int(ram_gib * GIB),
        logical_cpus=12, perf_cpus=8, free_disk_bytes=500 * GIB, cwd="/",
    )


class NativeDefaultsTests(unittest.TestCase):
    def test_48gb_gets_full_residency_from_the_payload(self):
        with patch("redlite.planner.native_residency", return_value=IQ3):
            d = native_defaults(48 * GIB, "m.gguf")
        self.assertTrue(d.full_residency)
        self.assertEqual(d.cache_mib, 29376)
        self.assertIn("29376 MiB from the file's expert payload", d.reason)

    def test_threshold_is_inclusive_at_40gb(self):
        with patch("redlite.planner.native_residency", return_value=IQ2):
            self.assertTrue(native_defaults(40 * GIB, "m.gguf").full_residency)
            self.assertFalse(native_defaults(40 * GIB - 1, "m.gguf").full_residency)

    def test_full_residency_must_fit_the_working_set(self):
        with patch("redlite.planner.native_residency", return_value=IQ3):
            d = native_defaults(40 * GIB, "m.gguf")   # 29376 MiB + 1.42 GiB > 70% of 40 GiB
        self.assertFalse(d.full_residency)
        self.assertEqual(d.cache_mib, NATIVE_BOUNDED_CACHE_MIB)

    def test_unreadable_model_falls_back_to_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "x.gguf"
            bad.write_bytes(b"not a gguf")
            self.assertIsNone(native_residency(bad))
            d = native_defaults(48 * GIB, bad)
        self.assertFalse(d.full_residency)
        self.assertIn("could not be read", d.reason)

    def test_best_model_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            self.assertIsNone(select_native_model(d, 48 * GIB))
            for n in ("Qwen_Qwen3-Next-80B-A3B-Instruct-IQ2_XXS.gguf", "Qwen_Qwen3-Next-80B-A3B-Instruct-IQ3_XXS.gguf"):
                (d / n).write_bytes(b"GGUF")
            sizes = {"Qwen_Qwen3-Next-80B-A3B-Instruct-IQ3_XXS.gguf": IQ3, "Qwen_Qwen3-Next-80B-A3B-Instruct-IQ2_XXS.gguf": IQ2}
            with patch("redlite.planner.native_residency", side_effect=lambda p: sizes[Path(p).name]):
                self.assertEqual(select_native_model(d, 48 * GIB).name, "Qwen_Qwen3-Next-80B-A3B-Instruct-IQ3_XXS.gguf")
                self.assertEqual(select_native_model(d, 40 * GIB).name, "Qwen_Qwen3-Next-80B-A3B-Instruct-IQ2_XXS.gguf")
                self.assertEqual(select_native_model(d, 24 * GIB).name, "Qwen_Qwen3-Next-80B-A3B-Instruct-IQ2_XXS.gguf")

    def test_iq3m_does_not_get_full_residency_on_48gb(self):
        # dev36: measured on the M4 Max 48 GiB, full residency of IQ3_M ran out of GPU memory
        with patch("redlite.planner.native_residency", return_value=NativeResidency(34944, int(1.6 * GIB))):
            self.assertFalse(native_defaults(48 * GIB, "m.gguf").full_residency)
            self.assertTrue(native_defaults(64 * GIB, "m.gguf").full_residency)

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
                 patch("redlite.planner.native_residency", return_value=IQ2), \
                 patch("redlite.runner.native_generate", return_value=Path("/x/redlite-generate")), \
                 contextlib.redirect_stdout(out):
                self.assertEqual(main(["chat", str(model), "--dry-run", *extra]), 0)
            return out.getvalue()

    def test_chat_without_flags_on_48gb_uses_full_residency(self):
        text = self._run(48)
        self.assertIn("--cache-mib 21312", text)
        self.assertIn("full expert residency", text)
        self.assertNotIn("--batch", text)

    def test_chat_without_flags_on_24gb_uses_4gb(self):
        text = self._run(24)
        self.assertIn("--cache-mib 4096", text)
        self.assertIn("bounded 4 GiB", text)

    def test_explicit_cache_and_batch_win(self):
        text = self._run(48, "--cache-mib", "1024", "--batch", "128", "--json", "--min-p", "0.05")
        self.assertIn("--cache-mib 1024", text)
        self.assertIn("--json", text)
        self.assertIn("--min-p 0.05", text)
        self.assertIn("--batch 128", text)
        self.assertNotIn("full expert residency", text)


class NativeServeCliTests(unittest.TestCase):
    def _run(self, ram_gib: float, *extra: str) -> str:
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "m.gguf"
            model.write_bytes(b"GGUF")
            out = io.StringIO()
            with patch("redlite.cli.detect", return_value=_hw(ram_gib)), \
                 patch("redlite.planner.native_residency", return_value=IQ2), \
                 patch("redlite.runner.native_server", return_value=Path("/x/redlite-server")), \
                 contextlib.redirect_stdout(out):
                self.assertEqual(main(["serve", str(model), "--native", "--dry-run", *extra]), 0)
            return out.getvalue()

    def test_serve_native_builds_the_redlite_server_command(self):
        text = self._run(48, "--port", "9000", "-c", "8192", "--batch", "256")
        self.assertIn("/x/redlite-server", text)
        self.assertIn("--host 127.0.0.1 --port 9000 --context 8192 --cache-mib 21312 --batch 256", text)
        self.assertNotIn("llama-server", text)

    def test_serve_native_on_24gb_defaults_to_4gb_and_4096_context(self):
        text = self._run(24)
        self.assertIn("--context 4096 --cache-mib 4096", text)


if __name__ == "__main__":
    unittest.main()


class GpuLimitTests(unittest.TestCase):
    """dev51: a raised iogpu.wired_limit_mb allows full residency below 40 GiB of RAM (M4 Pro 24 GiB, IQ2_XXS)."""
    IQ2_REAL = NativeResidency(17316, 1083 * 1024 * 1024)

    def test_24gb_with_raised_limit_gets_full_residency(self):
        with patch("redlite.planner.native_residency", return_value=self.IQ2_REAL):
            d = native_defaults(24 * GIB, "/m/Qwen_Qwen3-Next-80B-A3B-Instruct-IQ2_XXS.gguf", wired_mib=20480)
        self.assertTrue(d.full_residency)
        self.assertEqual(d.cache_mib, 17316)

    def test_24gb_with_default_limit_stays_bounded(self):
        with patch("redlite.planner.native_residency", return_value=self.IQ2_REAL):
            d = native_defaults(24 * GIB, "/m/Qwen_Qwen3-Next-80B-A3B-Instruct-IQ2_XXS.gguf", wired_mib=0)
        self.assertFalse(d.full_residency)
        self.assertEqual(d.cache_mib, 4096)

    def test_limit_too_low_stays_bounded(self):
        with patch("redlite.planner.native_residency", return_value=self.IQ2_REAL):
            d = native_defaults(24 * GIB, "/m/Qwen_Qwen3-Next-80B-A3B-Instruct-IQ2_XXS.gguf", wired_mib=18432)
        self.assertFalse(d.full_residency)

    def test_needed_limit_includes_margin(self):
        from redlite.planner import native_full_residency_mib
        self.assertEqual(native_full_residency_mib(self.IQ2_REAL), 17316 + 1083 + 1024)
        self.assertEqual(native_full_residency_mib(self.IQ2_REAL, 2423911040), 17316 + 1083 + 2312 + 1024)

    def test_mtp_needs_room_for_the_head(self):
        import tempfile
        from redlite.planner import NATIVE_MTP_FILE, native_mtp_file
        with tempfile.TemporaryDirectory() as d:
            model = Path(d) / "Qwen_Qwen3-Next-80B-A3B-Instruct-IQ2_XXS.gguf"
            model.write_bytes(b"")
            with open(Path(d) / NATIVE_MTP_FILE, "wb") as f:
                f.truncate(2423911040)
            with patch("redlite.planner.native_residency", return_value=self.IQ2_REAL):
                self.assertIsNone(native_mtp_file(model, "full", ram_bytes=24 * GIB, wired_mib=20480))
                self.assertIsNotNone(native_mtp_file(model, "full", ram_bytes=24 * GIB, wired_mib=22016))
                self.assertIsNotNone(native_mtp_file(model, "full"))   # no budget given: unchanged dev45 behaviour


class RouteBiasDefaultTests(unittest.TestCase):
    """dev51 2c: cache-aware routing (lambda 0.5) by default only with a bounded cache, never over a user setting."""

    def _run(self, cache_mib, exact=False, env=None):
        import argparse
        import os
        from redlite.cli import _route_bias_env
        with patch.dict(os.environ, env or {}, clear=False):
            os.environ.pop("RL_ROUTE_CACHE_BIAS", None) if env is None else None
            with patch("redlite.planner.native_residency", return_value=NativeResidency(17316, GIB)), \
                 contextlib.redirect_stdout(io.StringIO()):
                _route_bias_env("/m/x.gguf", cache_mib, argparse.Namespace(exact_routing=exact))
            return os.environ.get("RL_ROUTE_CACHE_BIAS")

    def test_bounded_cache_gets_the_bias(self):
        self.assertEqual(self._run(4096), "0.5")

    def test_full_residency_does_not(self):
        self.assertIsNone(self._run(17316))
        self.assertIsNone(self._run("full"))

    def test_exact_routing_and_user_setting_win(self):
        self.assertIsNone(self._run(4096, exact=True))
        self.assertEqual(self._run(4096, env={"RL_ROUTE_CACHE_BIAS": "0"}), "0")
