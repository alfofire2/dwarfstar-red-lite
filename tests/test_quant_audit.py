import unittest
from unittest.mock import patch

from redlite.expert_map import TensorInfo
from redlite.quant_audit import audit_routed_quantization


class QuantAuditTests(unittest.TestCase):
    def test_audit_uses_span_bytes_and_reports_patterns(self):
        infos = [
            TensorInfo(
                name="blk.0.ffn_gate_exps.weight",
                shape=(2048, 768, 512),
                ggml_type=17,
                relative_offset=0,
                absolute_offset=4096,
                span_bytes=1234,
            ),
            TensorInfo(
                name="blk.0.ffn_up_exps.weight",
                shape=(2048, 768, 512),
                ggml_type=17,
                relative_offset=1234,
                absolute_offset=5330,
                span_bytes=2345,
            ),
            TensorInfo(
                name="blk.0.ffn_down_exps.weight",
                shape=(768, 2048, 512),
                ggml_type=16,
                relative_offset=3579,
                absolute_offset=7675,
                span_bytes=3456,
            ),
        ]
        with patch("redlite.quant_audit.read_tensor_directory", return_value=(3, 32, infos)):
            data = audit_routed_quantization("dummy.gguf")

        self.assertEqual(data["routed_tensor_count"], 3)
        self.assertEqual(data["layers"], 1)
        self.assertEqual(data["type_counts"], {"IQ2_XS": 2, "IQ2_XXS": 1})
        self.assertEqual(data["by_kind"]["gate"], {"IQ2_XS": 1})
        self.assertEqual(data["by_kind"]["up"], {"IQ2_XS": 1})
        self.assertEqual(data["by_kind"]["down"], {"IQ2_XXS": 1})
        self.assertEqual(data["layer_patterns"], {"IQ2_XS/IQ2_XS/IQ2_XXS": 1})
        self.assertEqual(data["tensors"][0]["bytes"], 1234)


if __name__ == "__main__":
    unittest.main()
