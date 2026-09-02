from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .expert_cache import CacheKey
from .expert_store import ExpertLayout
from .redmetal import RedMetalError
from .redmetal_exec import RedMetalExecPool


@dataclass
class ExecCacheStats:
    hits: int = 0
    misses: int = 0
    loads: int = 0
    evictions: int = 0
    blocked_victims: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExecMetalEntry:
    slot_id: int
    payload_bytes: int


class ExecMetalExpertCache:
    """Global LRU backed by shared Metal slabs with in-flight-safe recycling.

    Every resident `(layer, expert)` is also installed in a GPU-visible address
    table. A slot is never overwritten while native Metal work reports it as
    in-flight. This is the lifetime rule required before routed expert kernels
    can run asynchronously.
    """

    def __init__(
        self,
        model: str | Path,
        budget_bytes: int,
        slot_bytes: int,
        *,
        slots_per_slab: int = 64,
    ) -> None:
        self.pool = RedMetalExecPool(
            model,
            budget_bytes,
            slot_bytes,
            slots_per_slab=slots_per_slab,
        )
        self.capacity = self.pool.capacity
        self.entries: OrderedDict[CacheKey, ExecMetalEntry] = OrderedDict()
        self.next_slot = 0
        self.stats = ExecCacheStats()

    def __enter__(self) -> "ExecMetalExpertCache":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        for key in list(self.entries):
            try:
                self.pool.unbind(key.layer, key.expert)
            except Exception:
                pass
        self.entries.clear()
        self.pool.close()

    def _recyclable_victim(self) -> tuple[CacheKey, ExecMetalEntry]:
        for key, entry in self.entries.items():
            if self.pool.slot_inflight(entry.slot_id):
                self.stats.blocked_victims += 1
                continue
            return key, entry
        raise RedMetalError("all Metal expert cache slots are currently in-flight")

    def acquire(self, key: CacheKey, layout: ExpertLayout) -> ExecMetalEntry:
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
            victim_key, victim = self._recyclable_victim()
            self.entries.pop(victim_key)
            self.pool.unbind(victim_key.layer, victim_key.expert)
            slot_id = victim.slot_id
            self.stats.evictions += 1

        self.pool.load(slot_id, layout)
        self.pool.bind(key.layer, key.expert, slot_id, layout)
        entry = ExecMetalEntry(slot_id=slot_id, payload_bytes=layout.total_bytes)
        self.entries[key] = entry
        self.stats.loads += 1
        return entry

    def get(self, key: CacheKey) -> ExecMetalEntry | None:
        return self.entries.get(key)

    def addresses(self, key: CacheKey) -> tuple[int, int, int] | None:
        if key not in self.entries:
            return None
        return self.pool.addresses(key.layer, key.expert)

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
