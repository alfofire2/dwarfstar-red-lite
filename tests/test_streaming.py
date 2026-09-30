import os
from pathlib import Path
import tempfile
import unittest

from redlite.expert_cache import CacheKey, ExpertSlotCache
from redlite.expert_map import TensorInfo, _expert_layout
from redlite.expert_store import ExpertLayout, ExpertPart, ExpertStore


class StreamingTests(unittest.TestCase):
    def test_merged_expert_tensor_slices_evenly(self):
        tensor = TensorInfo(
            name="blk.7.ffn_gate_exps.weight",
            shape=(512, 2048, 512),
            ggml_type=1,
            relative_offset=0,
            absolute_offset=4096,
            span_bytes=512 * 4096,
        )
        expert = _expert_layout(tensor, 512, 32)
        self.assertIsNotNone(expert)
        self.assertTrue(expert.slice_safe)
        offset, size = expert.expert_slice(10)
        self.assertEqual(size, 4096)
        self.assertEqual(offset, 4096 + 10 * 4096)

    def test_missing_outer_expert_axis_is_rejected(self):
        tensor = TensorInfo(
            "blk.0.ffn_up_exps.weight",
            (512, 2048),
            1,
            0,
            4096,
            512 * 4096,
        )
        expert = _expert_layout(tensor, 512, 32)
        self.assertFalse(expert.slice_safe)

    def _model_file(self, size=1024 * 1024):
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.write(bytes((i % 251 for i in range(size))))
        tmp.close()
        self.addCleanup(lambda: Path(tmp.name).unlink(missing_ok=True))
        return tmp.name

    @staticmethod
    def _layout(layer, expert, base, part_size=1024):
        return ExpertLayout(
            layer=layer,
            expert=expert,
            parts=(
                ExpertPart("gate", base, part_size, 0),
                ExpertPart("up", base + part_size, part_size, part_size),
                ExpertPart("down", base + 2 * part_size, part_size, 2 * part_size),
            ),
            total_bytes=3 * part_size,
        )

    def test_expert_store_uses_positional_reads_into_slot(self):
        name = self._model_file()
        layout = self._layout(0, 0, 8192)
        slot = bytearray(4096)
        with ExpertStore(name) as store:
            view = store.load(layout, slot)
            with open(name, "rb") as source:
                source.seek(8192)
                expected = source.read(layout.total_bytes)
            self.assertEqual(bytes(view), expected)
            view.release()
            self.assertEqual(store.stats.bytes_read, layout.total_bytes)
            self.assertEqual(store.stats.read_calls, 3)

    def test_slot_cache_is_hard_bounded_and_reuses_evicted_slots(self):
        name = self._model_file()
        slot_bytes = 4096
        with ExpertSlotCache(name, slot_bytes * 2, slot_bytes, prefetch_workers=1) as cache:
            for expert in range(3):
                layout = self._layout(0, expert, expert * 4096)
                view = cache.acquire(CacheKey(0, expert), layout)
                view.release()
            stats = cache.telemetry()
            self.assertEqual(stats["allocated_slots"], 2)
            self.assertEqual(stats["allocated_bytes"], slot_bytes * 2)
            self.assertEqual(stats["resident_slots"], 2)
            self.assertGreaterEqual(stats["evictions"], 1)
            self.assertLessEqual(stats["allocated_bytes"], stats["budget_bytes"])

    def test_prefetch_loads_expert_before_demand(self):
        name = self._model_file()
        layout = self._layout(0, 1, 4096)
        with ExpertSlotCache(name, 4 * 4096, 4096, prefetch_workers=1) as cache:
            future = cache.prefetch(CacheKey(0, 1), layout)
            self.assertIsNotNone(future)
            cache.wait_prefetch()
            view = cache.acquire(CacheKey(0, 1), layout)
            view.release()
            stats = cache.telemetry()
            self.assertEqual(stats["prefetch_submitted"], 1)
            self.assertEqual(stats["hits"], 1)
            self.assertEqual(stats["misses"], 0)

    def test_thousands_of_accesses_keep_fixed_slot_count(self):
        size = 16 * 1024 * 1024
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.truncate(size)
            name = tmp.name
        self.addCleanup(lambda: Path(name).unlink(missing_ok=True))

        slot_bytes = 4096
        capacity = 32
        with ExpertSlotCache(
            name,
            capacity * slot_bytes,
            slot_bytes,
            prefetch_workers=2,
        ) as cache:
            for i in range(2500):
                base = (i * 4096) % (size - 4096)
                layout = ExpertLayout(
                    layer=i // 512,
                    expert=i,
                    parts=(ExpertPart("gate", base, 4096, 0),),
                    total_bytes=4096,
                )
                view = cache.acquire(CacheKey(i // 512, i), layout)
                view.release()
            stats = cache.telemetry()
            self.assertEqual(stats["misses"], 2500)
            self.assertEqual(stats["allocated_slots"], capacity)
            self.assertEqual(stats["resident_slots"], capacity)
            self.assertEqual(stats["allocated_bytes"], capacity * slot_bytes)
            self.assertGreater(stats["evictions"], 2000)


if __name__ == "__main__":
    unittest.main()
