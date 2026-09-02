from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
import threading
from typing import Any

from .expert_store import ExpertLayout, ExpertStore


@dataclass(frozen=True)
class CacheKey:
    layer: int
    expert: int


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    loads: int = 0
    evictions: int = 0
    prefetch_submitted: int = 0
    prefetch_hits: int = 0
    prefetch_waits: int = 0
    allocated_slots: int = 0
    resident_slots: int = 0
    allocated_bytes: int = 0
    peak_allocated_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _Entry:
    slot: bytearray
    payload_bytes: int


class ExpertSlotCache:
    """Hard-bounded DS4-style cache of explicit expert buffers.

    The cache owns a single ExpertStore/file descriptor. Expert triplets are read
    with positional I/O into fixed-size reusable bytearray slots. Slots are
    allocated lazily, never exceeding floor(budget / slot_size), and are recycled
    by a global LRU keyed by (layer, expert).

    This deliberately avoids using mmap residency as the cache budget: the number
    reported by the cache is the actual amount of slot memory allocated by Red
    Lite, making the limit deterministic before Metal buffers are introduced.
    """

    def __init__(
        self,
        path: str | Path,
        budget_bytes: int,
        slot_bytes: int,
        *,
        prefetch_workers: int = 2,
    ):
        if budget_bytes <= 0:
            raise ValueError("budget_bytes must be positive")
        if slot_bytes <= 0:
            raise ValueError("slot_bytes must be positive")
        self.budget_bytes = int(budget_bytes)
        self.slot_bytes = int(slot_bytes)
        self.slot_count = self.budget_bytes // self.slot_bytes
        if self.slot_count < 1:
            raise ValueError(
                f"cache budget {self.budget_bytes} cannot fit one {self.slot_bytes}-byte expert slot"
            )

        self.store = ExpertStore(path)
        self.entries: OrderedDict[CacheKey, _Entry] = OrderedDict()
        self.free_slots: list[bytearray] = []
        self.inflight: dict[CacheKey, Future[_Entry]] = {}
        self.stats = CacheStats()
        self._lock = threading.RLock()
        workers = max(1, min(int(prefetch_workers), self.slot_count))
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="redlite-expert")
        self._closed = False

    def __enter__(self) -> "ExpertSlotCache":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=False)
        with self._lock:
            self.inflight.clear()
            self.entries.clear()
            self.free_slots.clear()
            self.stats.resident_slots = 0
        self.store.close()

    def _take_slot_locked(self) -> bytearray:
        if self.free_slots:
            return self.free_slots.pop()

        if self.stats.allocated_slots < self.slot_count:
            slot = bytearray(self.slot_bytes)
            self.stats.allocated_slots += 1
            self.stats.allocated_bytes = self.stats.allocated_slots * self.slot_bytes
            self.stats.peak_allocated_bytes = max(
                self.stats.peak_allocated_bytes,
                self.stats.allocated_bytes,
            )
            return slot

        if not self.entries:
            raise RuntimeError("all cache slots are busy with in-flight loads")

        _, victim = self.entries.popitem(last=False)
        self.stats.evictions += 1
        self.stats.resident_slots = len(self.entries)
        return victim.slot

    def _load_new(self, key: CacheKey, layout: ExpertLayout) -> _Entry:
        if layout.total_bytes > self.slot_bytes:
            raise ValueError(
                f"expert ({layout.total_bytes} bytes) exceeds fixed slot size ({self.slot_bytes} bytes)"
            )

        with self._lock:
            existing = self.entries.get(key)
            if existing is not None:
                self.entries.move_to_end(key)
                return existing
            slot = self._take_slot_locked()

        try:
            view = self.store.load(layout, slot)
            view.release()
        except Exception:
            with self._lock:
                self.free_slots.append(slot)
            raise

        entry = _Entry(slot=slot, payload_bytes=layout.total_bytes)
        with self._lock:
            self.entries[key] = entry
            self.entries.move_to_end(key)
            self.stats.loads += 1
            self.stats.resident_slots = len(self.entries)
        return entry

    def _entry_view(self, entry: _Entry) -> memoryview:
        return memoryview(entry.slot)[: entry.payload_bytes]

    def acquire(self, key: CacheKey, layout: ExpertLayout) -> memoryview:
        with self._lock:
            if self._closed:
                raise RuntimeError("cache is closed")
            existing = self.entries.get(key)
            if existing is not None:
                self.entries.move_to_end(key)
                self.stats.hits += 1
                return self._entry_view(existing)
            future = self.inflight.get(key)
            if future is None:
                self.stats.misses += 1

        if future is not None:
            self.stats.prefetch_waits += 1
            entry = future.result()
            with self._lock:
                self.inflight.pop(key, None)
                self.entries.move_to_end(key)
                self.stats.hits += 1
            return self._entry_view(entry)

        entry = self._load_new(key, layout)
        return self._entry_view(entry)

    def prefetch(self, key: CacheKey, layout: ExpertLayout) -> Future[_Entry] | None:
        with self._lock:
            if self._closed:
                raise RuntimeError("cache is closed")
            if key in self.entries:
                self.entries.move_to_end(key)
                self.stats.prefetch_hits += 1
                return None
            existing = self.inflight.get(key)
            if existing is not None:
                self.stats.prefetch_hits += 1
                return existing
            self.stats.prefetch_submitted += 1
            future = self._executor.submit(self._load_new, key, layout)
            self.inflight[key] = future
            return future

    def wait_prefetch(self) -> None:
        while True:
            with self._lock:
                pending = list(self.inflight.items())
            if not pending:
                return
            for key, future in pending:
                future.result()
                with self._lock:
                    self.inflight.pop(key, None)

    def resident_keys(self) -> list[CacheKey]:
        with self._lock:
            return list(self.entries.keys())

    def telemetry(self) -> dict[str, Any]:
        with self._lock:
            data = self.stats.to_dict()
            data.update(
                {
                    "slot_bytes": self.slot_bytes,
                    "slot_capacity": self.slot_count,
                    "budget_bytes": self.budget_bytes,
                    "store": self.store.stats.to_dict(),
                }
            )
            return data
