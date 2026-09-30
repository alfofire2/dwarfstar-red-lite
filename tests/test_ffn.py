import math
import unittest

from redlite.redmetal_ffn_parity import _silu_mul


class FfnTests(unittest.TestCase):
    def test_silu_mul_known_values(self):
        values = _silu_mul([0.0, 1.0, -1.0], [2.0, 2.0, 2.0])
        self.assertEqual(values[0], 0.0)
        self.assertAlmostEqual(values[1], 2.0 / (1.0 + math.exp(-1.0)), places=7)
        expected_neg = -2.0 * math.exp(-1.0) / (1.0 + math.exp(-1.0))
        self.assertAlmostEqual(values[2], expected_neg, places=7)

    def test_silu_mul_rejects_shape_mismatch(self):
        with self.assertRaises(ValueError):
            _silu_mul([1.0], [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
