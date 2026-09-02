from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .expert_cache import CacheKey
from .expert_store import ExpertLayout
from .redmetal import RedMetalError
from .redmetal_topk import RedMetalTopKPool


@dataclass
class TopKCacheStats:
    hits: int = 0
    misses: int = 0
    loads: int = 0
    evictions: int = 0
    blocked_inflight: int = 0
    protected_victims: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TopKMetalEntry:
    slot_id: int
    payload_bytes: int


class TopKMetalExpertCache:
    """Hard-bounded global LRU with protection for the current selected expert set."""

    def __init__(
        self,
        model: str | Path,
        budget_bytes: int,
        slot_bytes: int,
        *,
        slots_per_slab: int = 64,
    ) -> None:
        self.pool = RedMetalTopKPool(
            model,
            budget_bytes,
            slot_bytes,
            slots_per_slab=slots_per_slab,
        )
        self.capacity = self.pool.capacity
        self.entries: OrderedDict[CacheKey, TopKMetalEntry] = OrderedDict()
        self.layouts: dict[CacheKey, ExpertLayout] = {}
        self.next_slot = 0
        self.stats = TopKCacheStats()

    def __enter__(self) -> "TopKMetalExpertCache":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self.entries.clear()
        self.layouts.clear()
        self.pool.close()

    def _recyclable_victim(
        self,
        protected: set[CacheKey],
    ) -> tuple[CacheKey, TopKMetalEntry]:
        for key, entry in self.entries.items():
            if key in protected:
                self.stats.protected_victims += 1
                continue
            if self.pool.slot_inflight(entry.slot_id):
                self.stats.blocked_inflight += 1
                continue
            return key, entry
        raise RedMetalError("no recyclable Metal slot remains outside the current top-k set")

    def acquire(
        self,
        key: CacheKey,
        layout: ExpertLayout,
        *,
        protected: set[CacheKey] | None = None,
    ) -> TopKMetalEntry:
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
            victim_key, victim = self._recyclable_victim(protected or set())
            self.entries.pop(victim_key)
            self.layouts.pop(victim_key, None)
            slot_id = victim.slot_id
            self.stats.evictions += 1

        self.pool.load(slot_id, layout)
        entry = TopKMetalEntry(slot_id=slot_id, payload_bytes=layout.total_bytes)
        self.entries[key] = entry
        self.layouts[key] = layout
        self.stats.loads += 1
        return entry

    def acquire_many(
        self,
        items: Iterable[tuple[CacheKey, ExpertLayout]],
    ) -> list[TopKMetalEntry]:
        pairs = list(items)
        keys = [key for key, _ in pairs]
        if len(set(keys)) != len(keys):
            raise ValueError("top-k expert keys must be unique")
        if len(keys) > self.capacity:
            raise RedMetalError(
                f"top-k requires {len(keys)} resident experts but cache capacity is only {self.capacity}"
            )
        protected = set(keys)
        return [self.acquire(key, layout, protected=protected) for key, layout in pairs]

    def execute(
        self,
        keys: list[CacheKey],
        router_weights: list[float],
        ggml_type: int,
        hidden_size: int,
        ffn_size: int,
        output_row_start: int,
        output_row_count: int,
        input_values: list[float],
    ) -> tuple[list[float], float]:
        if len(keys) != len(router_weights):
            raise ValueError("expert and router-weight counts must match")
        entries: list[TopKMetalEntry] = []
        layouts: list[ExpertLayout] = []
        for key in keys:
            entry = self.entries.get(key)
            layout = self.layouts.get(key)
            if entry is None or layout is None:
                raise KeyError(f"expert {key} is not resident")
            entries.append(entry)
            layouts.append(layout)
        for key in keys:
            self.entries.move_to_end(key)
        return self.pool.execute(
            [entry.slot_id for entry in entries],
            layouts,
            router_weights,
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
