import os
from pathlib import Path
import tempfile
import unittest

from redlite.expert_cache import CacheKey, MmapExpertCache
from redlite.expert_map import TensorInfo, _expert_layout


class StreamingTests(unittest.TestCase):
    def test_merged_expert_tensor_slices_evenly(self):
        t = TensorInfo(
            name="blk.7.ffn_gate_exps.weight",
            shape=(512, 2048, 512),
            ggml_type=1,
            relative_offset=0,
            absolute_offset=4096,
            span_bytes=512 * 4096,
        )
        e = _expert_layout(t, 512, 32)
        self.assertIsNotNone(e)
        self.assertTrue(e.slice_safe)
        off, size = e.expert_slice(10)
        self.assertEqual(size, 4096)
        self.assertEqual(off, 4096 + 10 * 4096)

    def test_missing_outer_expert_axis_is_rejected(self):
        t = TensorInfo("blk.0.ffn_up_exps.weight", (512, 2048), 1, 0, 4096, 512 * 4096)
        e = _expert_layout(t, 512, 32)
        self.assertFalse(e.slice_safe)

    def test_mmap_cache_is_bounded_and_evicts(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(os.urandom(1024 * 1024))
            name = f.name
        self.addCleanup(lambda: Path(name).unlink(missing_ok=True))
        with MmapExpertCache(name, 300 * 1024) as cache:
            for i in range(4):
                view = cache.acquire(CacheKey(0, i, "gate"), i * 128 * 1024, 128 * 1024)
                self.assertEqual(len(view), 128 * 1024)
                view.release()
            self.assertGreater(cache.stats.evictions, 0)
            self.assertLessEqual(cache.stats.mapped_bytes, cache.budget_bytes)

    def test_thousands_of_entries_share_one_file_mapping(self):
        # Regression for macOS EMFILE: dev1 created a separate mmap/file handle
        # for each expert slice. The cache must scale in entry count without
        # scaling OS file descriptors.
        size = 16 * 1024 * 1024
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.truncate(size)
            name = f.name
        self.addCleanup(lambda: Path(name).unlink(missing_ok=True))

        with MmapExpertCache(name, 12 * 1024 * 1024) as cache:
            for i in range(2500):
                offset = (i * 4096) % (size - 4096)
                view = cache.acquire(CacheKey(i // 512, i, "gate"), offset, 4096)
                self.assertEqual(len(view), 4096)
                view.release()
            self.assertEqual(cache.stats.misses, 2500)
            self.assertLessEqual(cache.stats.mapped_bytes, cache.budget_bytes)
            self.assertGreater(len(cache.entries), 1000)


if __name__ == "__main__":
    unittest.main()
