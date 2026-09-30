from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .expert_cache import CacheKey
from .expert_store import ExpertLayout
from .redmetal import RedMetalError
from .redmetal_ffn import RedMetalFfnPool


@dataclass
class FfnCacheStats:
    hits: int = 0
    misses: int = 0
    loads: int = 0
    evictions: int = 0
    blocked_victims: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FfnMetalEntry:
    slot_id: int
    payload_bytes: int


class FfnMetalExpertCache:
    """Hard-bounded LRU whose resident slots are directly executable by Metal."""

    def __init__(
        self,
        model: str | Path,
        budget_bytes: int,
        slot_bytes: int,
        *,
        slots_per_slab: int = 64,
    ) -> None:
        self.pool = RedMetalFfnPool(
            model,
            budget_bytes,
            slot_bytes,
            slots_per_slab=slots_per_slab,
        )
        self.capacity = self.pool.capacity
        self.entries: OrderedDict[CacheKey, FfnMetalEntry] = OrderedDict()
        self.layouts: dict[CacheKey, ExpertLayout] = {}
        self.next_slot = 0
        self.stats = FfnCacheStats()

    def __enter__(self) -> "FfnMetalExpertCache":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self.entries.clear()
        self.layouts.clear()
        self.pool.close()

    def _recyclable_victim(self) -> tuple[CacheKey, FfnMetalEntry]:
        for key, entry in self.entries.items():
            if self.pool.slot_inflight(entry.slot_id):
                self.stats.blocked_victims += 1
                continue
            return key, entry
        raise RedMetalError("all resident FFN Metal slots are currently in-flight")

    def acquire(self, key: CacheKey, layout: ExpertLayout) -> FfnMetalEntry:
        existing = self.entries.pop(key, None)
        if existing is not None:
            self.entries[key] = existing
            self.layouts[key] = layout
            self.stats.hits += 1
            return existing

        self.stats.misses += 1
        if self.next_slot < self.capacity:
            slot_id = self.next_slot
            self.next_slot += 1
        else:
            victim_key, victim = self._recyclable_victim()
            self.entries.pop(victim_key)
            self.layouts.pop(victim_key, None)
            slot_id = victim.slot_id
            self.stats.evictions += 1

        self.pool.load(slot_id, layout)
        entry = FfnMetalEntry(slot_id=slot_id, payload_bytes=layout.total_bytes)
        self.entries[key] = entry
        self.layouts[key] = layout
        self.stats.loads += 1
        return entry

    def execute(
        self,
        key: CacheKey,
        ggml_type: int,
        hidden_size: int,
        ffn_size: int,
        output_row_start: int,
        output_row_count: int,
        input_values: list[float],
    ) -> tuple[list[float], float]:
        entry = self.entries.get(key)
        layout = self.layouts.get(key)
        if entry is None or layout is None:
            raise KeyError(f"expert {key} is not resident")
        self.entries.move_to_end(key)
        return self.pool.execute(
            entry.slot_id,
            layout,
            ggml_type,
            hidden_size,
            ffn_size,
            output_row_start,
            output_row_count,
            input_values,
        )

    def addresses(self, key: CacheKey) -> tuple[int, int, int] | None:
        entry = self.entries.get(key)
        layout = self.layouts.get(key)
        if entry is None or layout is None:
            return None
        return self.pool.addresses(entry.slot_id, layout)

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
