from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, asdict
import os
import threading
import time
from typing import Iterable

from .gguf import ExpertRecord


@dataclass
class CacheStats:
    requests: int = 0
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    bytes_read: int = 0
    read_seconds: float = 0.0

    @property
    def hit_rate(self) -> float:
        return self.hits / self.requests if self.requests else 0.0

    @property
    def read_gib_s(self) -> float:
        return (self.bytes_read / (1024 ** 3)) / self.read_seconds if self.read_seconds > 0 else 0.0

    def to_dict(self):
        d = asdict(self)
        d["hit_rate"] = self.hit_rate
        d["read_gib_s"] = self.read_gib_s
        return d


class ExpertLRUCache:
    """Bounded reference cache using os.pread.

    This is intentionally a correctness/reference implementation. The native Metal
    cache in native/ stores the same expert records directly in MTLStorageModeShared
    buffers. Keeping this implementation small makes cache policy unit-testable on
    non-macOS CI hosts.
    """

    def __init__(self, model_path: str, capacity_bytes: int):
        if capacity_bytes <= 0:
            raise ValueError("capacity_bytes must be positive")
        self.fd = os.open(model_path, os.O_RDONLY)
        self.capacity_bytes = int(capacity_bytes)
        self.used_bytes = 0
        self._items: OrderedDict[tuple[int, int], bytes] = OrderedDict()
        self._lock = threading.Lock()
        self.stats = CacheStats()

    def close(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def _read_record(self, rec: ExpertRecord) -> bytes:
        chunks = []
        start = time.perf_counter()
        for seg in rec.segments:
            b = os.pread(self.fd, seg.length, seg.offset)
            if len(b) != seg.length:
                raise IOError(f"short pread for layer={rec.layer} expert={rec.expert}")
            chunks.append(b)
        elapsed = time.perf_counter() - start
        payload = b"".join(chunks)
        self.stats.bytes_read += len(payload)
        self.stats.read_seconds += elapsed
        return payload

    def get(self, rec: ExpertRecord) -> bytes:
        key = (rec.layer, rec.expert)
        with self._lock:
            self.stats.requests += 1
            cached = self._items.pop(key, None)
            if cached is not None:
                self.stats.hits += 1
                self._items[key] = cached
                return cached
            self.stats.misses += 1

        payload = self._read_record(rec)
        if len(payload) > self.capacity_bytes:
            return payload

        with self._lock:
            existing = self._items.pop(key, None)
            if existing is not None:
                self._items[key] = existing
                return existing
            while self._items and self.used_bytes + len(payload) > self.capacity_bytes:
                _old_key, old = self._items.popitem(last=False)
                self.used_bytes -= len(old)
                self.stats.evictions += 1
            self._items[key] = payload
            self.used_bytes += len(payload)
        return payload

    def warm(self, records: Iterable[ExpertRecord], limit: int | None = None) -> None:
        for i, rec in enumerate(records):
            if limit is not None and i >= limit:
                break
            self.get(rec)
