import unittest

from redlite.telemetry import _parse_swapusage


class TelemetryTests(unittest.TestCase):
    def test_parse_standard_macos_swapusage(self):
        used, total = _parse_swapusage("total = 2048.00M  used = 512.00M  free = 1536.00M  (encrypted)")
        self.assertAlmostEqual(used, 0.5)
        self.assertAlmostEqual(total, 2.0)

    def test_parse_prefixed_and_mixed_units(self):
        used, total = _parse_swapusage("vm.swapusage: total = 2.00G used = 128.00M free = 1.88G")
        self.assertAlmostEqual(used, 0.125)
        self.assertAlmostEqual(total, 2.0)

    def test_parse_gib_suffix(self):
        used, total = _parse_swapusage("total = 4.00GiB used = 1.25GiB free = 2.75GiB")
        self.assertAlmostEqual(used, 1.25)
        self.assertAlmostEqual(total, 4.0)

    def test_none(self):
        self.assertEqual(_parse_swapusage(None), (None, None))


if __name__ == "__main__":
    unittest.main()
