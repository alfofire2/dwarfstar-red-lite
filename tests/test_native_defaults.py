import contextlib
import os
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
                 patch("redlite.cli.gpu_wired_limit_mib", return_value=0), \
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

    def test_chat_full_residency_uses_prompt_lookup(self):
        text = self._run(48)
        self.assertIn("--lookup", text)
        self.assertNotIn("--lookup", self._run(48, "--no-lookup").split("redlite-generate", 1)[-1])
        self.assertIn("--lookup", self._run(24))   # dev72: a bounded cache verifies too

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
                 patch("redlite.cli.gpu_wired_limit_mib", return_value=0), \
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
        self.assertIn("--lookup", text)   # dev72: a bounded cache verifies too, with exact routing
        self.assertNotIn("cache-aware expert routing", text)

    def test_serve_native_full_residency_uses_prompt_lookup(self):
        text = self._run(48)
        self.assertIn("--lookup", text)
        self.assertIn("prompt lookup speculative decoding", text)

    def test_serve_native_no_lookup(self):
        self.assertNotIn("--lookup", self._run(48, "--no-lookup").split("redlite-server", 1)[1])


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
        # dev55 measured model: slots + dense + 580 + 48 KiB per position + 572 MiB for 2048-token chunks (+1787 MTP)
        self.assertEqual(native_full_residency_mib(self.IQ2_REAL), 17316 + 1083 + 580 + 192 + 572)
        self.assertEqual(native_full_residency_mib(self.IQ2_REAL, mtp=True), 17316 + 1083 + 580 + 192 + 572 + 1787)

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

    def test_prompt_lookup_keeps_routing_exact(self):
        import argparse
        import os
        from redlite.cli import _route_bias_env
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RL_ROUTE_CACHE_BIAS", None)
            with patch("redlite.planner.native_residency", return_value=NativeResidency(17316, GIB)), \
                 contextlib.redirect_stdout(io.StringIO()):
                _route_bias_env("/m/x.gguf", 4096, argparse.Namespace(exact_routing=False), lookup=True)
            self.assertIsNone(os.environ.get("RL_ROUTE_CACHE_BIAS"))

    def test_exact_routing_and_user_setting_win(self):
        self.assertIsNone(self._run(4096, exact=True))
        self.assertEqual(self._run(4096, env={"RL_ROUTE_CACHE_BIAS": "0"}), "0")


class RedLiteMixPreferenceTests(unittest.TestCase):
    """dev54: the E3 mix is preferred over Bartowski's IQ2_XXS of the same size; IQ3_XXS still wins on 48 GiB."""

    def test_24gb_prefers_e3(self):
        from redlite.planner import select_native_model
        with tempfile.TemporaryDirectory() as d:
            for n in ("Qwen_Qwen3-Next-80B-A3B-Instruct-IQ2_XXS.gguf", "Qwen3-Next-80B-A3B-Instruct-RedLite-E3.gguf"):
                (Path(d) / n).write_bytes(b"")
            self.assertEqual(select_native_model(d, 24 * GIB).name, "Qwen3-Next-80B-A3B-Instruct-RedLite-E3.gguf")

    def test_48gb_still_prefers_iq3_when_it_fits(self):
        from redlite.planner import select_native_model
        with tempfile.TemporaryDirectory() as d:
            for n in ("Qwen_Qwen3-Next-80B-A3B-Instruct-IQ3_XXS.gguf", "Qwen3-Next-80B-A3B-Instruct-RedLite-E3.gguf"):
                (Path(d) / n).write_bytes(b"")
            with patch("redlite.planner.native_residency", return_value=NativeResidency(28800, GIB)):
                self.assertEqual(select_native_model(d, 48 * GIB).name, "Qwen_Qwen3-Next-80B-A3B-Instruct-IQ3_XXS.gguf")

    def test_24gb_alias_downloads_e3(self):
        from redlite.model_catalog import resolve_variant
        self.assertEqual(resolve_variant("24gb").repo, "alfodaniello/Qwen3-Next-80B-A3B-Instruct-RedLite-GGUF")
        self.assertEqual(resolve_variant("bartowski-24gb").filename, "Qwen_Qwen3-Next-80B-A3B-Instruct-IQ2_XXS.gguf")


class GpuPlanTests(unittest.TestCase):
    """dev55: the measured need explains the M4 Pro outcomes at a 21,741 MiB limit (IQ2_XXS-size file)."""
    RES = NativeResidency(17316, 1083 * 1024 * 1024)

    def test_m4pro_outcomes(self):
        from redlite.planner import native_full_residency_mib
        self.assertGreater(native_full_residency_mib(self.RES, 32768, 512, True), 21741)    # observed: out of GPU memory
        self.assertLessEqual(native_full_residency_mib(self.RES, 32768, 2048, False), 21741)  # observed: ran
        self.assertLessEqual(native_full_residency_mib(self.RES, 4096, 2048, True), 21741)    # observed: ran (52.7 tok/s)

    def test_plan_prefers_mtp_then_big_chunks(self):
        from redlite.planner import native_gpu_plan
        self.assertEqual(native_gpu_plan(self.RES, 21741, 4096, True), (2048, True))
        self.assertEqual(native_gpu_plan(self.RES, 21741, 32768, True), (2048, False))   # MTP dropped at 32K
        self.assertEqual(native_gpu_plan(self.RES, 21741, 16384, True), (512, True))     # smaller chunks keep MTP
        self.assertIsNone(native_gpu_plan(self.RES, 19000, 4096, True))


class ContextAwareFitTests(unittest.TestCase):
    """dev64: at the default GPU limit the 70 % RAM rule counts the context's KV cache (G2 on the M4 Max 48 GiB)."""
    G2 = NativeResidency(30528, 1457 * 1024 * 1024)       # measured: swapped with MTP and a 25K prompt, not without
    IQ3_XXS = NativeResidency(28800, 1457 * 1024 * 1024)  # measured: no swap with MTP and a 25K prompt

    def test_g2_keeps_mtp_at_short_context_only(self):
        from redlite.planner import native_full_residency_fits
        self.assertTrue(native_full_residency_fits(48 * GIB, self.G2, 0, 4096, mtp=True))
        self.assertFalse(native_full_residency_fits(48 * GIB, self.G2, 0, 32768, mtp=True))
        self.assertTrue(native_full_residency_fits(48 * GIB, self.G2, 0, 32768, mtp=False))

    def test_iq3_xxs_keeps_mtp_at_32k(self):
        from redlite.planner import native_full_residency_fits
        self.assertTrue(native_full_residency_fits(48 * GIB, self.IQ3_XXS, 0, 32768, mtp=True))

    def test_serve_drops_mtp_for_g2_at_32k(self):
        from redlite import cli
        args = type("A", (), {"no_mtp": False, "batch": None, "parallel": 1, "context": 32768})()
        with tempfile.TemporaryDirectory() as t:
            model = Path(t) / "Qwen3-Next-80B-A3B-Instruct-RedLite-G2.gguf"
            model.write_bytes(b"x")
            (Path(t) / "Qwen3-Next-80B-A3B-Instruct-MTP-ONLY-Q8_0.gguf").write_bytes(b"x")
            out = io.StringIO()
            with patch("redlite.planner.native_residency", return_value=self.G2), \
                 patch("redlite.cli.gpu_wired_limit_mib", return_value=0), \
                 patch("redlite.cli.detect", return_value=_hw(48)), contextlib.redirect_stdout(out):
                self.assertEqual(cli._gpu_tuning(model, "full", args, 32768)[1], None)
                args.context = 4096
                self.assertIsNotNone(cli._gpu_tuning(model, "full", args, 4096)[1])
        self.assertIn("MTP off", out.getvalue())


class MtpMaxContextTests(unittest.TestCase):
    def test_limit_only_below_40gb(self):
        from redlite.planner import native_mtp_max_context
        self.assertEqual(native_mtp_max_context(24 * GIB), 8192)
        self.assertIsNone(native_mtp_max_context(48 * GIB))


class ParallelPlanTests(unittest.TestCase):
    """dev56: two slots are planned as extra positions (second KV cache + 72 MiB DeltaNet state)."""

    def test_plan_context(self):
        from redlite.planner import native_plan_context
        self.assertEqual(native_plan_context(4096), 4096)
        self.assertEqual(native_plan_context(4096, 2), 8192 + 1536)
        self.assertEqual(native_plan_context(2048, 2), 4096 + 1536)


class RedLiteF2PreferenceTests(unittest.TestCase):
    """dev58: F2 is preferred over E3 and Bartowski's IQ2_XXS; `24gb` downloads it, `e3` still names E3."""

    def test_f2_preferred(self):
        import tempfile
        from pathlib import Path
        from redlite.planner import select_native_model
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            for n in ("Qwen_Qwen3-Next-80B-A3B-Instruct-IQ2_XXS.gguf", "Qwen3-Next-80B-A3B-Instruct-RedLite-E3.gguf",
                      "Qwen3-Next-80B-A3B-Instruct-RedLite-F2.gguf"):
                (d / n).write_bytes(b"x")
            self.assertEqual(select_native_model(d, 24 * GIB).name, "Qwen3-Next-80B-A3B-Instruct-RedLite-F2.gguf")

    def test_aliases(self):
        from redlite.model_catalog import resolve_variant
        self.assertEqual(resolve_variant("24gb").filename, "Qwen3-Next-80B-A3B-Instruct-RedLite-F2.gguf")
        self.assertEqual(resolve_variant("e3").filename, "Qwen3-Next-80B-A3B-Instruct-RedLite-E3.gguf")
        # dev63/dev64
        self.assertEqual(resolve_variant("coder").filename, "Qwen3-Coder-Next-RedLite-CF2.gguf")
        self.assertEqual(resolve_variant("coder").repo, "alfodaniello/Qwen3-Coder-Next-RedLite-GGUF")
        self.assertEqual(resolve_variant("bartowski-coder").filename, "Qwen_Qwen3-Coder-Next-IQ2_XXS.gguf")
        self.assertEqual(resolve_variant("48gb").filename, "Qwen_Qwen3-Next-80B-A3B-Instruct-IQ3_XXS.gguf")
        self.assertEqual(resolve_variant("48gb-g2").filename, "Qwen3-Next-80B-A3B-Instruct-RedLite-G2.gguf")

    def test_aliases_case_insensitive(self):
        from redlite.model_catalog import resolve_variant
        # dev72: variant names should be case-insensitive
        self.assertEqual(resolve_variant("CODER").filename, "Qwen3-Coder-Next-RedLite-CF2.gguf")
        self.assertEqual(resolve_variant("24GB").filename, "Qwen3-Next-80B-A3B-Instruct-RedLite-F2.gguf")
        self.assertEqual(resolve_variant("MTP").filename, "Qwen3-Next-80B-A3B-Instruct-MTP-ONLY-Q8_0.gguf")
        self.assertEqual(resolve_variant("CF2").filename, "Qwen3-Coder-Next-RedLite-CF2.gguf")

    def test_coder_24gb_alias(self):
        from redlite.model_catalog import resolve_variant
        # dev72: coder-24gb is an alias for coder (same variant)
        self.assertEqual(resolve_variant("coder-24gb").key, resolve_variant("coder").key)
        self.assertEqual(resolve_variant("CODER-24GB").filename, "Qwen3-Coder-Next-RedLite-CF2.gguf")

    def test_g2_preferred_on_48gb_when_present(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from redlite import planner
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            for n in ("Qwen3-Next-80B-A3B-Instruct-RedLite-G2.gguf", "Qwen_Qwen3-Next-80B-A3B-Instruct-IQ3_XXS.gguf",
                      "Qwen3-Next-80B-A3B-Instruct-RedLite-F2.gguf"):
                (d / n).write_bytes(b"x")
            g2 = NativeResidency(30528, 1457 * 1024 * 1024)
            with patch.object(planner, "native_residency", return_value=g2):
                self.assertEqual(planner.select_native_model(d, 48 * GIB).name, "Qwen3-Next-80B-A3B-Instruct-RedLite-G2.gguf")
                # 24 GiB with a raised limit: G2 does not fit, F2 is chosen
                self.assertEqual(planner.select_native_model(d, 24 * GIB, 21741).name,
                                 "Qwen3-Next-80B-A3B-Instruct-RedLite-F2.gguf")


class PreferenceUnderGpuLimitTests(unittest.TestCase):
    """dev58b: with a raised GPU limit (full-residency path) F2 is chosen too; 0.5.6 picked E3 there."""

    def test_f2_under_raised_limit(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from redlite import planner
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            for n in planner.NATIVE_SMALL_MODELS:
                (d / n).write_bytes(b"x")
            with patch.object(planner, "native_residency", return_value=object()), \
                 patch.object(planner, "native_full_residency_fits", return_value=True):
                self.assertEqual(planner.select_native_model(d, 24 * GIB, 21741).name,
                                 "Qwen3-Next-80B-A3B-Instruct-RedLite-F2.gguf")


class HalfKvPlanTests(unittest.TestCase):
    """dev75: --kv f16 (RL_KV_F16=1) halves the KV term of the GPU plan; dev74 M4 Pro outcomes at a 21,741 MiB limit."""
    RES = NativeResidency(17316, 1083 * 1024 * 1024)

    def test_half_kv_halves_the_kv_term(self):
        import os
        from redlite.planner import native_full_residency_mib, native_plan_context
        with patch.dict(os.environ, {"RL_KV_F16": "1"}):
            self.assertEqual(native_full_residency_mib(self.RES), 17316 + 1083 + 580 + 96 + 572)
            self.assertEqual(native_plan_context(4096, 2), 8192 + 3072)
            self.assertLessEqual(native_full_residency_mib(self.RES, 32768, 512, True), 21741)    # dev74 observed: ran
            self.assertLessEqual(native_full_residency_mib(self.RES, 16384, 2048, True), 21741)   # dev74 observed: ran
            self.assertGreater(native_full_residency_mib(self.RES, 32768, 2048, True), 21741)    # dev74 observed: out of memory
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RL_KV_F16", None)
            self.assertEqual(native_plan_context(4096, 2), 8192 + 1536)

    def test_kv_option(self):
        from redlite.cli import build_parser
        p = build_parser()
        self.assertEqual(p.parse_args(["chat"]).kv, "f32")
        self.assertEqual(p.parse_args(["serve", "--native", "--kv", "f16"]).kv, "f16")


class GpuLimitCliTests(unittest.TestCase):
    """dev79: redlite gpu-limit and the chat tip"""

    def _gpu_limit(self, *extra: str, daemon_exists: bool = False, wired: int = 0) -> str:
        with tempfile.TemporaryDirectory() as tmp:
            daemon = Path(tmp) / "com.redlite.gpulimit.plist"
            if daemon_exists:
                daemon.write_text("x")
            out = io.StringIO()
            with patch("redlite.cli.detect", return_value=_hw(24)), \
                 patch("redlite.cli._gpu_limit_need", return_value=(19000, 21741)), \
                 patch("redlite.cli.gpu_wired_limit_mib", return_value=wired), \
                 patch("redlite.cli.GPU_LIMIT_DAEMON", daemon), \
                 contextlib.redirect_stdout(out):
                self.assertEqual(main(["gpu-limit", "--dry-run", *extra]), 0)
            return out.getvalue()

    def test_sets_the_mtp_need_until_reboot(self):
        text = self._gpu_limit()
        self.assertIn("$ sudo /usr/sbin/sysctl iogpu.wired_limit_mb=21741", text)
        self.assertNotIn("launchctl", text)

    def test_boot_installs_the_launch_daemon(self):
        text = self._gpu_limit("--boot", "--mib", "20000")
        self.assertIn("iogpu.wired_limit_mb=20000", text)
        self.assertIn("sudo install -m 644 -o root -g wheel", text)
        self.assertIn("$ sudo launchctl bootstrap system", text)
        self.assertNotIn("bootout", text)
        self.assertIn("bootout", self._gpu_limit("--boot", daemon_exists=True))   # replaced, not doubled

    def test_off_restores_the_default_and_removes_the_daemon(self):
        self.assertIn("iogpu.wired_limit_mb=0", self._gpu_limit("--off"))
        text = self._gpu_limit("--off", daemon_exists=True)
        self.assertIn("bootout system/com.redlite.gpulimit", text)
        self.assertIn("sudo rm -f", text)

    def test_nothing_to_do_when_the_limit_is_already_enough(self):
        text = self._gpu_limit(wired=21741)
        self.assertIn("nothing to do", text)
        self.assertNotIn("sysctl", text)

    def test_refuses_a_limit_that_leaves_macos_less_than_2_gib(self):
        with patch("redlite.cli.detect", return_value=_hw(24)), contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(["gpu-limit", "--dry-run", "--mib", "23000"])
        with patch("redlite.cli.detect", return_value=_hw(24)), contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(["gpu-limit", "--dry-run", "--mib", "0"])

    def test_chat_tip_only_with_a_bounded_cache(self):
        def chat(ram_gib: float) -> str:
            with tempfile.TemporaryDirectory() as tmp:
                model = Path(tmp) / "m.gguf"
                model.write_bytes(b"GGUF")
                out = io.StringIO()
                with patch("redlite.cli.detect", return_value=_hw(ram_gib)), \
                     patch("redlite.cli.gpu_wired_limit_mib", return_value=0), \
                     patch("redlite.planner.native_residency", return_value=NativeResidency(17316, int(1.06 * GIB))), \
                     patch("redlite.runner.native_generate", return_value=Path("/x/redlite-generate")), \
                     contextlib.redirect_stdout(out):
                    self.assertEqual(main(["chat", str(model), "--dry-run"]), 0)
                return out.getvalue()
        self.assertIn("redlite gpu-limit", chat(24))
        self.assertNotIn("redlite gpu-limit", chat(48))


class KvF16TipTests(unittest.TestCase):
    """dev80: serve suggests --kv f16 when only the float KV cache keeps full residency out"""

    def _serve(self, *extra: str, wired: int = 21741) -> str:
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "m.gguf"
            model.write_bytes(b"GGUF")
            out = io.StringIO()
            with patch("redlite.cli.detect", return_value=_hw(24)), \
                 patch("redlite.cli.gpu_wired_limit_mib", return_value=wired), \
                 patch("redlite.planner.native_residency", return_value=NativeResidency(17316, 1083 * 1024 * 1024)), \
                 patch("redlite.runner.native_server", return_value=Path("/x/redlite-server")), \
                 patch.dict(os.environ, {}, clear=False), \
                 contextlib.redirect_stdout(out):
                os.environ.pop("RL_KV_F16", None)
                self.assertEqual(main(["serve", str(model), "--native", "--dry-run", *extra]), 0)
            return out.getvalue()

    def test_64k_at_the_raised_limit_suggests_half_kv(self):
        text = self._serve("-c", "65536")
        self.assertIn("--cache-mib 4096", text)
        self.assertIn("--kv f16", text)
        self.assertNotIn("RL_KV_F16", os.environ)   # the check leaves the environment as it was

    def test_no_tip_when_full_residency_already_fits_or_half_kv_is_on(self):
        self.assertNotIn("--kv f16", self._serve("-c", "4096"))
        self.assertNotIn("tip: with --kv f16", self._serve("-c", "65536", wired=0))   # default limit: gpu-limit first
        text = self._serve("-c", "65536", "--kv", "f16")
        self.assertIn("full expert residency", text)
        self.assertNotIn("tip: with --kv f16", text)


class SmallMacTests(unittest.TestCase):
    """dev82: below the 24 GiB target"""

    def _chat(self, ram_gib: float) -> str:
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "m.gguf"
            model.write_bytes(b"GGUF")
            out = io.StringIO()
            with patch("redlite.cli.detect", return_value=_hw(ram_gib)), \
                 patch("redlite.cli.gpu_wired_limit_mib", return_value=0), \
                 patch("redlite.planner.native_residency", return_value=NativeResidency(16992, 1083 * 1024 * 1024)), \
                 patch("redlite.runner.native_generate", return_value=Path("/x/redlite-generate")), \
                 contextlib.redirect_stdout(out):
                self.assertEqual(main(["chat", str(model), "--dry-run"]), 0)
            return out.getvalue()

    def test_16gb_gets_the_note_and_no_full_residency_tips(self):
        text = self._chat(16)
        self.assertIn("below Red Lite's 24 GiB target", text)
        self.assertNotIn("redlite gpu-limit", text)
        self.assertNotIn("--kv f16", text)
        self.assertNotIn("below Red Lite's 24 GiB target", self._chat(24))

    def test_gpu_limit_on_16gb_says_why(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "Qwen_Qwen3-Next-80B-A3B-Instruct-IQ2_XXS.gguf").write_bytes(b"GGUF")
            err = io.StringIO()
            with patch("redlite.cli.detect", return_value=_hw(16)), patch("redlite.cli.NATIVE_MODELS_DIR", Path(tmp)), \
                 patch("redlite.planner.native_residency", return_value=NativeResidency(17316, 1083 * 1024 * 1024)), \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err), self.assertRaises(SystemExit):
                main(["gpu-limit", "--dry-run"])
        self.assertIn("streams the experts from the SSD", err.getvalue())
        self.assertNotIn("40 GiB or more", err.getvalue())
