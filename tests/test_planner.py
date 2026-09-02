import os
from pathlib import Path
import tempfile
import unittest

from redlite.hardware import HardwareInfo, GIB
from redlite.planner import plan_for


def fake_hw(ram_gib=24, disk_gib=100, chip="Apple M4 Pro"):
    return HardwareInfo(
        system="Darwin",
        machine="arm64",
        chip=chip,
        ram_bytes=int(ram_gib * GIB),
        logical_cpus=14,
        perf_cpus=8,
        free_disk_bytes=int(disk_gib * GIB),
        cwd="/tmp",
    )


class PlannerTests(unittest.TestCase):
    def _model(self, gib):
        f = tempfile.NamedTemporaryFile(delete=False)
        f.close()
        os.truncate(f.name, int(gib * GIB))
        self.addCleanup(lambda: Path(f.name).unlink(missing_ok=True))
        return f.name

    def test_24gb_18gib_model_prefers_metal(self):
        p = plan_for(fake_hw(24), self._model(17.8), context=4096)
        self.assertEqual(p.mode, "metal-resident")
        self.assertTrue(p.safe)
        self.assertEqual(p.status, "CRITICAL")

    def test_m4pro_24gb_18gib_profile_uses_4k_default(self):
        p = plan_for(fake_hw(24, chip="Apple M4 Pro"), self._model(17.97))
        self.assertEqual(p.context, 4096)
        self.assertEqual(p.threads, 8)
        self.assertEqual((p.batch, p.ubatch), (256, 128))
        self.assertIn("2K/4K/8K Metal sweep", p.reason)

    def test_unvalidated_24gb_18gib_profile_stays_at_2k(self):
        p = plan_for(fake_hw(24, chip="Apple M3 Pro"), self._model(17.97))
        self.assertEqual(p.context, 2048)

    def test_roomier_resident_profile_can_be_safe(self):
        p = plan_for(fake_hw(24), self._model(15.0), context=2048)
        self.assertEqual(p.status, "SAFE")
        self.assertTrue(p.safe)

    def test_24gb_45gib_model_prefers_ssd(self):
        p = plan_for(fake_hw(24), self._model(45.0), context=4096)
        self.assertEqual(p.mode, "ssd-cpu")
        self.assertEqual(p.status, "SSD")
        self.assertTrue(p.safe)

    def test_forced_metal_rejects_oversized_model(self):
        p = plan_for(fake_hw(24), self._model(45.0), context=4096, force_mode="metal-resident")
        self.assertFalse(p.safe)
        self.assertEqual(p.status, "UNSAFE")

    def test_low_disk_rejects_oversized(self):
        p = plan_for(fake_hw(24, disk_gib=2), self._model(45.0), context=4096)
        self.assertFalse(p.safe)
        self.assertEqual(p.status, "UNSAFE")


if __name__ == "__main__":
    unittest.main()
