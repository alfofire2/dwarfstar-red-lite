from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .expert_cache import CacheKey
from .expert_store import ExpertLayout
from .redmetal import RedMetalPool


@dataclass
class MetalCacheStats:
    hits: int = 0
    misses: int = 0
    loads: int = 0
    evictions: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MetalEntry:
    slot_id: int
    payload_bytes: int


class MetalExpertCache:
    """LRU policy whose backing slots are ranges of shared MTLBuffer slabs."""

    def __init__(
        self,
        model: str | Path,
        budget_bytes: int,
        slot_bytes: int,
        *,
        slots_per_slab: int = 64,
    ) -> None:
        self.pool = RedMetalPool(
            model,
            budget_bytes,
            slot_bytes,
            slots_per_slab=slots_per_slab,
        )
        self.capacity = self.pool.capacity
        self.entries: OrderedDict[CacheKey, MetalEntry] = OrderedDict()
        self.next_slot = 0
        self.stats = MetalCacheStats()

    def __enter__(self) -> "MetalExpertCache":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self.entries.clear()
        self.pool.close()

    def acquire(self, key: CacheKey, layout: ExpertLayout) -> MetalEntry:
        existing = self.entries.pop(key, None)
        if existing is not None:
            self.entries[key] = existing
            self.stats.hits += 1
            return existing

        self.stats.misses += 1
        if self.next_slot < self.capacity:
            slot_id = self.next_slot
            self.next_slot += 1
        else:
            _, victim = self.entries.popitem(last=False)
            slot_id = victim.slot_id
            self.stats.evictions += 1

        self.pool.load(slot_id, layout)
        entry = MetalEntry(slot_id=slot_id, payload_bytes=layout.total_bytes)
        self.entries[key] = entry
        self.stats.loads += 1
        return entry

    def get(self, key: CacheKey) -> MetalEntry | None:
        return self.entries.get(key)

    def resident_keys(self) -> list[CacheKey]:
        return list(self.entries.keys())

    def telemetry(self) -> dict[str, Any]:
        data = self.stats.to_dict()
        data.update(
            {
                "resident_slots": len(self.entries),
                "used_slot_ids": self.next_slot,
                "slot_capacity": self.capacity,
                "metal": self.pool.telemetry(),
            }
        )
        return data
