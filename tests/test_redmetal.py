import struct
import unittest

from redlite.iq2_reference import deterministic_input, iq2_xxs_row_dot
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
        # d=1.0, all grid indices=0, sign/scale word=0. Grid[0] is eight
        # 0x08 values and db=1*(0.5+0)*0.25=0.125, so every decoded
        # weight is exactly 1.0.
        block = struct.pack("<e", 1.0) + bytes(64)
        x = deterministic_input(256)
        self.assertAlmostEqual(iq2_xxs_row_dot(block, x), sum(x), places=6)


if __name__ == "__main__":
    unittest.main()
