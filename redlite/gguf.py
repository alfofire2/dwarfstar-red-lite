from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import os
import re
import struct
from typing import Any, BinaryIO


GGUF_MAGIC = b"GGUF"
GGUF_DEFAULT_ALIGNMENT = 32

_UINT8, _INT8, _UINT16, _INT16, _UINT32, _INT32, _FLOAT32, _BOOL, _STRING, _ARRAY, _UINT64, _INT64, _FLOAT64 = range(13)
_ROUTED_RE = re.compile(r"^blk\.(\d+)\.ffn_(down|gate|up|gate_up)_exps\.weight$")


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dimensions: tuple[int, ...]
    type_id: int
    relative_offset: int
    absolute_offset: int
    span_bytes: int

    @property
    def expert_count(self) -> int | None:
        if _ROUTED_RE.match(self.name) and self.dimensions:
            return int(self.dimensions[-1])
        return None

    @property
    def layer(self) -> int | None:
        m = _ROUTED_RE.match(self.name)
        return int(m.group(1)) if m else None

    @property
    def role(self) -> str | None:
        m = _ROUTED_RE.match(self.name)
        return m.group(2) if m else None

    def expert_stride_bytes(self) -> int:
        n = self.expert_count
        if not n:
            raise ValueError(f"tensor is not a routed expert tensor: {self.name}")
        stride = self.span_bytes // n
        if stride <= 0:
            raise ValueError(f"invalid expert stride for {self.name}")
        return stride


@dataclass(frozen=True)
class ExpertSegment:
    tensor_name: str
    role: str
    offset: int
    length: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExpertRecord:
    layer: int
    expert: int
    segments: tuple[ExpertSegment, ...]

    @property
    def total_bytes(self) -> int:
        return sum(s.length for s in self.segments)

    @property
    def key(self) -> str:
        return f"{self.layer}:{self.expert}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "expert": self.expert,
            "total_bytes": self.total_bytes,
            "segments": [s.to_dict() for s in self.segments],
        }


@dataclass(frozen=True)
class GGUFIndex:
    path: str
    version: int
    tensor_count: int
    metadata_count: int
    alignment: int
    data_offset: int
    metadata: dict[str, Any]
    tensors: tuple[TensorInfo, ...]

    @property
    def file_size(self) -> int:
        return os.path.getsize(self.path)

    @property
    def routed_tensors(self) -> tuple[TensorInfo, ...]:
        return tuple(t for t in self.tensors if _ROUTED_RE.match(t.name))

    @property
    def routed_bytes(self) -> int:
        return sum(t.span_bytes for t in self.routed_tensors)

    @property
    def non_routed_bytes_estimate(self) -> int:
        return max(0, self.file_size - self.routed_bytes)

    def expert_records(self) -> tuple[ExpertRecord, ...]:
        grouped: dict[tuple[int, int], list[ExpertSegment]] = {}
        for tensor in self.routed_tensors:
            n_expert = tensor.expert_count or 0
            stride = tensor.expert_stride_bytes()
            for expert in range(n_expert):
                key = (tensor.layer or 0, expert)
                grouped.setdefault(key, []).append(ExpertSegment(
                    tensor_name=tensor.name,
                    role=tensor.role or "unknown",
                    offset=tensor.absolute_offset + expert * stride,
                    length=stride,
                ))
        records = [ExpertRecord(layer, expert, tuple(sorted(segs, key=lambda s: s.role)))
                   for (layer, expert), segs in grouped.items()]
        return tuple(sorted(records, key=lambda r: (r.layer, r.expert)))


def _read_exact(f: BinaryIO, n: int) -> bytes:
    b = f.read(n)
    if len(b) != n:
        raise ValueError("truncated GGUF")
    return b


def _u32(f: BinaryIO) -> int:
    return struct.unpack("<I", _read_exact(f, 4))[0]


def _u64(f: BinaryIO) -> int:
    return struct.unpack("<Q", _read_exact(f, 8))[0]


def _gguf_string(f: BinaryIO) -> str:
    n = _u64(f)
    return _read_exact(f, n).decode("utf-8")


def _value(f: BinaryIO, type_id: int) -> Any:
    fixed = {
        _UINT8: ("<B", 1), _INT8: ("<b", 1), _UINT16: ("<H", 2), _INT16: ("<h", 2),
        _UINT32: ("<I", 4), _INT32: ("<i", 4), _FLOAT32: ("<f", 4), _BOOL: ("<?", 1),
        _UINT64: ("<Q", 8), _INT64: ("<q", 8), _FLOAT64: ("<d", 8),
    }
    if type_id in fixed:
        fmt, size = fixed[type_id]
        return struct.unpack(fmt, _read_exact(f, size))[0]
    if type_id == _STRING:
        return _gguf_string(f)
    if type_id == _ARRAY:
        elem_type = _u32(f)
        n = _u64(f)
        return [_value(f, elem_type) for _ in range(n)]
    raise ValueError(f"unsupported GGUF metadata type {type_id}")


def _align(value: int, alignment: int) -> int:
    if alignment <= 0 or alignment & (alignment - 1):
        raise ValueError(f"invalid GGUF alignment {alignment}")
    return (value + alignment - 1) & ~(alignment - 1)


def read_index(path: str | os.PathLike[str]) -> GGUFIndex:
    p = Path(path).expanduser().resolve()
    with p.open("rb") as f:
        if _read_exact(f, 4) != GGUF_MAGIC:
            raise ValueError(f"not a GGUF file: {p}")
        version = _u32(f)
        if version not in {2, 3}:
            raise ValueError(f"unsupported GGUF version {version}")
        tensor_count = _u64(f)
        metadata_count = _u64(f)
        metadata: dict[str, Any] = {}
        for _ in range(metadata_count):
            key = _gguf_string(f)
            type_id = _u32(f)
            metadata[key] = _value(f, type_id)

        raw: list[tuple[str, tuple[int, ...], int, int]] = []
        for _ in range(tensor_count):
            name = _gguf_string(f)
            n_dims = _u32(f)
            dims = tuple(_u64(f) for _ in range(n_dims))
            type_id = _u32(f)
            offset = _u64(f)
            raw.append((name, dims, type_id, offset))

        alignment = int(metadata.get("general.alignment", GGUF_DEFAULT_ALIGNMENT))
        data_offset = _align(f.tell(), alignment)

    file_size = p.stat().st_size
    data_size = file_size - data_offset
    if data_size < 0:
        raise ValueError("GGUF data offset exceeds file size")

    ordered = sorted(raw, key=lambda item: item[3])
    spans: dict[str, int] = {}
    for i, (name, _dims, _type, off) in enumerate(ordered):
        next_off = ordered[i + 1][3] if i + 1 < len(ordered) else data_size
        if next_off < off:
            raise ValueError("invalid GGUF tensor offsets")
        spans[name] = next_off - off

    tensors = tuple(TensorInfo(
        name=name,
        dimensions=dims,
        type_id=type_id,
        relative_offset=offset,
        absolute_offset=data_offset + offset,
        span_bytes=spans[name],
    ) for name, dims, type_id, offset in raw)

    return GGUFIndex(
        path=str(p), version=version, tensor_count=tensor_count, metadata_count=metadata_count,
        alignment=alignment, data_offset=data_offset, metadata=metadata, tensors=tensors,
    )


def write_expert_manifest(index: GGUFIndex, path: str | os.PathLike[str]) -> Path:
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# redlite-expert-manifest-v1",
        f"# model={index.path}",
        f"# data_offset={index.data_offset}",
        "# layer\texpert\ttotal_bytes\tsegments(offset:length,...)",
    ]
    for rec in index.expert_records():
        segs = ",".join(f"{s.offset}:{s.length}" for s in rec.segments)
        lines.append(f"{rec.layer}\t{rec.expert}\t{rec.total_bytes}\t{segs}")
    out.write_text("\n".join(lines) + "\n")
    return out
