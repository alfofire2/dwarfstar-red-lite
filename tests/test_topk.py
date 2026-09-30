import unittest
from unittest.mock import patch

from redlite.expert_cache import CacheKey
from redlite.expert_store import ExpertLayout, ExpertPart
from redlite.redmetal_topk_cache import TopKMetalExpertCache
from redlite.redmetal_topk_parity import default_expert_ids, default_router_weights


class _FakePool:
    def __init__(self, *_args, **_kwargs):
        self.capacity = 3
        self.loaded = []

    def close(self):
        pass

    def slot_inflight(self, _slot_id):
        return False

    def load(self, slot_id, layout):
        self.loaded.append((slot_id, layout.expert))
        return {"bytes_read": layout.total_bytes, "elapsed_ms": 0.0}

    def telemetry(self):
        return {
            "capacity": self.capacity,
            "slab_count": 1,
            "allocated_bytes": 0,
            "bytes_read": 0,
            "read_calls": 0,
            "read_ms": 0.0,
        }


def _layout(expert: int) -> ExpertLayout:
    parts = (
        ExpertPart("gate", 0, 10, 0),
        ExpertPart("up", 10, 10, 10),
        ExpertPart("down", 20, 10, 20),
    )
    return ExpertLayout(0, expert, parts, 30)


class TopKTests(unittest.TestCase):
    def test_default_selection_is_unique_and_weights_normalized(self):
        experts = default_expert_ids(10, 512)
        weights = default_router_weights(10)
        self.assertEqual(len(experts), 10)
        self.assertEqual(len(set(experts)), 10)
        self.assertEqual(len(weights), 10)
        self.assertAlmostEqual(sum(weights), 1.0, places=6)
        self.assertTrue(all(value > 0.0 for value in weights))

    def test_acquire_many_does_not_evict_current_selection(self):
        with patch("redlite.redmetal_topk_cache.RedMetalTopKPool", _FakePool):
            cache = TopKMetalExpertCache("dummy.gguf", 1024, 128)
            first = [(CacheKey(0, expert), _layout(expert)) for expert in (0, 1, 2)]
            cache.acquire_many(first)
            second = [(CacheKey(0, expert), _layout(expert)) for expert in (1, 2, 3)]
            cache.acquire_many(second)
            self.assertEqual(set(cache.entries), {CacheKey(0, 1), CacheKey(0, 2), CacheKey(0, 3)})
            self.assertEqual(cache.stats.evictions, 1)
            self.assertNotIn(CacheKey(0, 0), cache.entries)


if __name__ == "__main__":
    unittest.main()
