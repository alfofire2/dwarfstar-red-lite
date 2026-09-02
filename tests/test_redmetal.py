import unittest

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


if __name__ == "__main__":
    unittest.main()
