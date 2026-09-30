from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import threading
from typing import Any

from .expert_map import ExpertMap


@dataclass(frozen=True)
class ExpertPart:
    kind: str
    file_offset: int
    length: int
    slot_offset: int


@dataclass(frozen=True)
class ExpertLayout:
    layer: int
    expert: int
    parts: tuple[ExpertPart, ...]
    total_bytes: int


@dataclass
class StoreStats:
    read_calls: int = 0
    bytes_read: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_expert_layouts(expert_map: ExpertMap) -> dict[tuple[int, int], ExpertLayout]:
    by_layer: dict[int, dict[str, Any]] = {}
    for tensor in expert_map.expert_tensors:
        if tensor.slice_safe:
            by_layer.setdefault(tensor.layer, {})[tensor.kind] = tensor

    layouts: dict[tuple[int, int], ExpertLayout] = {}
    for layer in expert_map.layers:
        tensors = by_layer.get(layer, {})
        missing = [kind for kind in ("gate", "up", "down") if kind not in tensors]
        if missing:
            raise ValueError(f"layer {layer} is missing routed expert tensors: {', '.join(missing)}")

        for expert in range(expert_map.expert_count):
            slot_offset = 0
            parts: list[ExpertPart] = []
            for kind in ("gate", "up", "down"):
                file_offset, length = tensors[kind].expert_slice(expert)
                parts.append(ExpertPart(kind, file_offset, length, slot_offset))
                slot_offset += length
            layouts[(layer, expert)] = ExpertLayout(layer, expert, tuple(parts), slot_offset)

    return layouts


class ExpertStore:
    """Single-fd positional reader for routed experts.

    DS4-style SSD streaming keeps model-file access explicit instead of creating
    one mmap per expert. This store owns exactly one read-only file descriptor and
    copies selected expert ranges directly into caller-provided cache slots.
    """

    def __init__(self, path: str | Path):
        self.path = str(Path(path).expanduser().resolve())
        self.fd = os.open(self.path, os.O_RDONLY)
        self.file_size = os.fstat(self.fd).st_size
        self.stats = StoreStats()
        self._stats_lock = threading.Lock()

    def close(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1

    def __enter__(self) -> "ExpertStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _pread_into(self, offset: int, dest: memoryview) -> None:
        if offset < 0 or offset + len(dest) > self.file_size:
            raise ValueError(
                f"invalid expert read offset={offset} length={len(dest)} file_size={self.file_size}"
            )

        remaining = len(dest)
        cursor = 0
        while remaining:
            target = dest[cursor : cursor + remaining]
            if hasattr(os, "preadv"):
                n = os.preadv(self.fd, [target], offset + cursor)
            else:
                chunk = os.pread(self.fd, remaining, offset + cursor)
                n = len(chunk)
                target[:n] = chunk
            if n <= 0:
                raise EOFError(
                    f"short read at offset={offset + cursor}; wanted {remaining} more bytes"
                )
            with self._stats_lock:
                self.stats.read_calls += 1
                self.stats.bytes_read += n
            cursor += n
            remaining -= n

    def load(self, layout: ExpertLayout, slot: bytearray) -> memoryview:
        if len(slot) < layout.total_bytes:
            raise ValueError(
                f"cache slot is {len(slot)} bytes but expert needs {layout.total_bytes} bytes"
            )
        view = memoryview(slot)
        for part in layout.parts:
            target = view[part.slot_offset : part.slot_offset + part.length]
            self._pread_into(part.file_offset, target)
            target.release()
        return view[: layout.total_bytes]
