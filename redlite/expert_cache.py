from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
import mmap
import os
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CacheKey:
    layer: int
    expert: int
    kind: str


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    mapped_bytes: int = 0
    peak_mapped_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _Range:
    logical_offset: int
    logical_length: int
    page_offset: int
    page_length: int


class MmapExpertCache:
    """Bounded LRU over ranges of one whole-file GGUF mmap.

    dev1 created one mmap per expert tensor slice. Python/macOS keeps a file
    descriptor associated with each mapping, so a realistic MoE trace could hit
    RLIMIT_NOFILE long before reaching the byte budget.

    dev2 maps the GGUF exactly once and treats expert residency as page-range
    advice on that mapping:

      miss     -> MADV_WILLNEED(range)
      eviction -> MADV_DONTNEED(range)

    `mapped_bytes` therefore means bytes currently accounted to the residency
    working set, rounded to VM pages. It is not virtual address-space size (the
    entire file is mapped virtually but remains demand-paged).
    """

    def __init__(self, path: str | Path, budget_bytes: int):
        if budget_bytes <= 0:
            raise ValueError("budget_bytes must be positive")

        self.path = str(Path(path).expanduser().resolve())
        self.fd = os.open(self.path, os.O_RDONLY)
        self.file_size = os.fstat(self.fd).st_size
        if self.file_size <= 0:
            os.close(self.fd)
            raise ValueError("cannot mmap an empty model file")

        self.budget_bytes = int(budget_bytes)
        self.page_size = mmap.PAGESIZE
        self.mapping = mmap.mmap(
            self.fd,
            0,
            flags=mmap.MAP_PRIVATE,
            prot=mmap.PROT_READ,
        )
        self.entries: OrderedDict[CacheKey, _Range] = OrderedDict()
        self.stats = CacheStats()

    def close(self) -> None:
        self.entries.clear()
        try:
            self.mapping.close()
        finally:
            os.close(self.fd)

    def __enter__(self) -> "MmapExpertCache":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _page_range(self, offset: int, length: int) -> tuple[int, int]:
        if offset < 0 or length <= 0 or offset + length > self.file_size:
            raise ValueError(
                f"invalid expert range offset={offset} length={length} file_size={self.file_size}"
            )
        page = self.page_size
        start = (offset // page) * page
        end = min(self.file_size, ((offset + length + page - 1) // page) * page)
        return start, end - start

    def _advise(self, advice: int, start: int, length: int) -> None:
        if not hasattr(self.mapping, "madvise"):
            return
        try:
            # Python exposes mmap.madvise(option, start=0, length=0) on Unix.
            # start/length are page-aligned here for Darwin compatibility.
            self.mapping.madvise(advice, start, length)
        except (OSError, TypeError, ValueError):
            # Advice is an optimization. Correctness must not depend on the OS
            # accepting a particular madvise flavor.
            pass

    def _evict_until(self, needed: int) -> None:
        if needed > self.budget_bytes:
            raise ValueError(
                f"single expert range ({needed} bytes) exceeds cache budget "
                f"({self.budget_bytes} bytes)"
            )

        while self.stats.mapped_bytes + needed > self.budget_bytes and self.entries:
            _, rng = self.entries.popitem(last=False)
            self.stats.mapped_bytes -= rng.page_length
            self.stats.evictions += 1
            if hasattr(mmap, "MADV_DONTNEED"):
                self._advise(mmap.MADV_DONTNEED, rng.page_offset, rng.page_length)

    def acquire(
        self,
        key: CacheKey,
        offset: int,
        length: int,
        *,
        eager: bool = True,
    ) -> memoryview:
        existing = self.entries.pop(key, None)
        if existing is not None:
            self.entries[key] = existing
            self.stats.hits += 1
            return memoryview(self.mapping)[offset : offset + length]

        page_offset, page_length = self._page_range(offset, length)
        self.stats.misses += 1
        self._evict_until(page_length)

        if eager and hasattr(mmap, "MADV_WILLNEED"):
            self._advise(mmap.MADV_WILLNEED, page_offset, page_length)

        rng = _Range(offset, length, page_offset, page_length)
        self.entries[key] = rng
        self.stats.mapped_bytes += page_length
        self.stats.peak_mapped_bytes = max(
            self.stats.peak_mapped_bytes,
            self.stats.mapped_bytes,
        )
        return memoryview(self.mapping)[offset : offset + length]

    def prefetch(self, key: CacheKey, offset: int, length: int) -> None:
        view = self.acquire(key, offset, length, eager=True)
        view.release()

    def resident_keys(self) -> list[CacheKey]:
        return list(self.entries.keys())
