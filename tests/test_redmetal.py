import struct
import unittest

from redlite.iq2_reference import deterministic_input, iq2_xxs_row_dot
from redlite.quant_reference import quant_row_dot
from redlite.redmetal import redmetal_built, redmetal_library_path
from redlite.redmetal_streaming import sampled_fnv1a


class RedMetalTests(unittest.TestCase):
    def test_sampled_fnv1a_is_deterministic(self):
        payload = bytearray((i * 17 + 3) & 0xFF for i in range(10000))
        self.assertEqual(sampled_fnv1a(payload), sampled_fnv1a(payload))
        payload[-1] ^= 1
        self.assertNotEqual(sampled_fnv1a(payload), sampled_fnv1a(payload[:-1] + bytes([payload[-1] ^ 1])))

    def test_native_library_lookup_is_lazy(self):
        path = redmetal_library_path()
        self.assertTrue(path.name.endswith(".dylib"))
        self.assertIsInstance(redmetal_built(), bool)

    def test_iq2_reference_known_all_ones_block(self):
        # Legacy dev5 IQ2_XXS reference sanity check.
        block = struct.pack("<e", 1.0) + bytes(64)
        x = deterministic_input(256)
        self.assertAlmostEqual(iq2_xxs_row_dot(block, x), sum(x), places=6)

    def test_iq2_xs_reference_known_all_ones_block(self):
        # IQ2_XS: d=1, all 32 packed q words=0, all nibble scales=0.
        # With a synthetic grid containing only 8s, db=0.125 and every
        # decoded weight is exactly 1.0.
        block = struct.pack("<e", 1.0) + bytes(64) + bytes(8)
        grid = (8,) * (512 * 8)
        self.assertAlmostEqual(quant_row_dot(block, 17, [1.0] * 256, grid), 256.0, places=6)

    def test_iq1_m_reference_known_block(self):
        # IQ1_M has no standalone d field. Encode fp16(1.0)=0x3c00 in
        # the top nibbles of the four scale words, keep all 3-bit scales
        # zero, qh delta bit clear, and use a synthetic zero grid.
        # Each decoded weight is then 1 * (2*0+1) * (0 + 0.125)=0.125.
        scales = struct.pack("<4H", 0x0000, 0x0000, 0xC000, 0x3000)
        block = bytes(32) + bytes(16) + scales
        grid = (0,) * (2048 * 8)
        self.assertAlmostEqual(quant_row_dot(block, 29, [1.0] * 256, grid), 32.0, places=6)


if __name__ == "__main__":
    unittest.main()
