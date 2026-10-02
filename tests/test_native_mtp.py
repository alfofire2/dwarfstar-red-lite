import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from redlite.planner import NATIVE_MTP_FILE, NativeResidency, native_mtp_file

GIB = 1024**3


class NativeMtpTests(unittest.TestCase):
    def _dir(self, model_name: str, with_head: bool) -> Path:
        d = Path(tempfile.mkdtemp())
        (d / model_name).write_bytes(b"GGUF")
        if with_head:
            (d / NATIVE_MTP_FILE).write_bytes(b"GGUF")
        return d

    def test_full_residency_instruct_with_head(self):
        d = self._dir("Qwen_Qwen3-Next-80B-A3B-Instruct-IQ2_XXS.gguf", True)
        with patch("redlite.planner.native_residency", return_value=NativeResidency(17316, GIB)):
            model = d / "Qwen_Qwen3-Next-80B-A3B-Instruct-IQ2_XXS.gguf"
            self.assertEqual(native_mtp_file(model, 17316), d / NATIVE_MTP_FILE)
            self.assertEqual(native_mtp_file(model, "full"), d / NATIVE_MTP_FILE)
            self.assertIsNone(native_mtp_file(model, 4096))          # bounded cache: the verify needs every expert
            self.assertIsNone(native_mtp_file(model, 17316, disabled=True))

    def test_needs_head_and_instruct(self):
        with patch("redlite.planner.native_residency", return_value=NativeResidency(17316, GIB)):
            d = self._dir("Qwen_Qwen3-Next-80B-A3B-Instruct-IQ2_XXS.gguf", False)
            self.assertIsNone(native_mtp_file(d / "Qwen_Qwen3-Next-80B-A3B-Instruct-IQ2_XXS.gguf", "full"))
            d = self._dir("Qwen_Qwen3-Coder-Next-IQ2_XXS.gguf", True)
            self.assertIsNone(native_mtp_file(d / "Qwen_Qwen3-Coder-Next-IQ2_XXS.gguf", "full"))


if __name__ == "__main__":
    unittest.main()
