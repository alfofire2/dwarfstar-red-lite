from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import re
import struct
from typing import Any, BinaryIO

GGUF_MAGIC = b"GGUF"
DEFAULT_ALIGNMENT = 32

_U8, _I8, _U16, _I16, _U32, _I32, _F32, _BOOL, _STRING, _ARRAY, _U64, _I64, _F64 = range(13)
_EXPERT_RE = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$")


@dataclass(frozen=True)
class TensorInfo:
    name: str
    shape: tuple[int, ...]
    ggml_type: int
    relative_offset: int
    absolute_offset: int
    span_bytes: int


@dataclass(frozen=True)
class ExpertTensor:
    layer: int
    kind: str
    name: str
    experts: int
    expert_axis: int
    tensor_offset: int
    tensor_span_bytes: int
    payload_bytes: int
    tail_padding_bytes: int
    expert_stride_bytes: int
    slice_safe: bool
    reason: str

    def expert_slice(self, expert_id: int) -> tuple[int, int]:
        if not self.slice_safe:
            raise ValueError(f"{self.name} is not slice-safe: {self.reason}")
        if not 0 <= expert_id < self.experts:
            raise IndexError(expert_id)
        return self.tensor_offset + expert_id * self.expert_stride_bytes, self.expert_stride_bytes

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExpertMap:
    path: str
    version: int
    alignment: int
    tensor_count: int
    expert_count: int
    expert_tensors: tuple[ExpertTensor, ...]

    @property
    def layers(self) -> tuple[int, ...]:
        return tuple(sorted({t.layer for t in self.expert_tensors}))

    @property
    def total_routed_payload_bytes(self) -> int:
        return sum(t.payload_bytes for t in self.expert_tensors)

    @property
    def all_slice_safe(self) -> bool:
        return bool(self.expert_tensors) and all(t.slice_safe for t in self.expert_tensors)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "version": self.version,
            "alignment": self.alignment,
            "tensor_count": self.tensor_count,
            "expert_count": self.expert_count,
            "layers": list(self.layers),
            "total_routed_payload_bytes": self.total_routed_payload_bytes,
            "all_slice_safe": self.all_slice_safe,
            "expert_tensors": [t.to_dict() for t in self.expert_tensors],
        }


def _read_exact(f: BinaryIO, n: int) -> bytes:
    data = f.read(n)
    if len(data) != n:
        raise ValueError("truncated GGUF")
    return data


def _u32(f: BinaryIO) -> int:
    return struct.unpack("<I", _read_exact(f, 4))[0]


def _u64(f: BinaryIO) -> int:
    return struct.unpack("<Q", _read_exact(f, 8))[0]


def _string(f: BinaryIO) -> str:
    n = _u64(f)
    if n > 1 << 30:
        raise ValueError("unreasonable GGUF string length")
    return _read_exact(f, n).decode("utf-8")


def _skip_scalar(f: BinaryIO, typ: int) -> Any:
    sizes = {_U8:1,_I8:1,_U16:2,_I16:2,_U32:4,_I32:4,_F32:4,_BOOL:1,_U64:8,_I64:8,_F64:8}
    if typ == _STRING:
        return _string(f)
    if typ == _ARRAY:
        subtype = _u32(f)
        count = _u64(f)
        if count > 1 << 30:
            raise ValueError("unreasonable GGUF array length")
        for _ in range(count):
            _skip_scalar(f, subtype)
        return None
    size = sizes.get(typ)
    if size is None:
        raise ValueError(f"unsupported GGUF metadata type {typ}")
    raw = _read_exact(f, size)
    if typ == _U32:
        return struct.unpack("<I", raw)[0]
    if typ == _U64:
        return struct.unpack("<Q", raw)[0]
    return None


def _metadata(f: BinaryIO, count: int) -> dict[str, Any]:
    result = {}
    for _ in range(count):
        key = _string(f)
        typ = _u32(f)
        result[key] = _skip_scalar(f, typ)
    return result


def _tensor_directory(f: BinaryIO, count: int):
    tensors = []
    for _ in range(count):
        name = _string(f)
        n_dims = _u32(f)
        if n_dims > 8:
            raise ValueError(f"unreasonable tensor rank {n_dims}")
        shape = tuple(_u64(f) for _ in range(n_dims))
        ggml_type = _u32(f)
        offset = _u64(f)
        tensors.append((name, shape, ggml_type, offset))
    return tensors


def read_tensor_directory(path: str | Path) -> tuple[int, int, list[TensorInfo]]:
    p = Path(path).expanduser().resolve()
    size = p.stat().st_size
    with p.open("rb") as f:
        if _read_exact(f, 4) != GGUF_MAGIC:
            raise ValueError(f"{p} is not a GGUF file")
        version = _u32(f)
        if version not in {2, 3}:
            raise ValueError(f"unsupported GGUF version {version}")
        tensor_count = _u64(f)
        kv_count = _u64(f)
        meta = _metadata(f, kv_count)
        raw = _tensor_directory(f, tensor_count)
        alignment = int(meta.get("general.alignment") or DEFAULT_ALIGNMENT)
        pos = f.tell()
        data_base = (pos + alignment - 1) // alignment * alignment

    by_offset = sorted(raw, key=lambda x: x[3])
    infos = []
    for i, (name, shape, typ, rel) in enumerate(by_offset):
        next_rel = by_offset[i + 1][3] if i + 1 < len(by_offset) else size - data_base
        span = max(0, next_rel - rel)
        infos.append(TensorInfo(name, shape, typ, rel, data_base + rel, span))
    return version, alignment, infos


def _expert_layout(t: TensorInfo, expected_experts: int, alignment: int) -> ExpertTensor | None:
    m = _EXPERT_RE.match(t.name)
    if not m:
        return None
    layer, kind = int(m.group(1)), m.group(2)
    axes = [i for i, d in enumerate(t.shape) if d == expected_experts]
    if not axes:
        return ExpertTensor(layer, kind, t.name, expected_experts, -1, t.absolute_offset, t.span_bytes, t.span_bytes, 0, 0, False, f"shape {t.shape} has no {expected_experts}-expert axis")
    axis = axes[-1]
    axis_ok = axis == len(t.shape) - 1
    remainder = t.span_bytes % expected_experts
    tail_padding = remainder if remainder < alignment else 0
    payload = t.span_bytes - tail_padding
    divisible = payload > 0 and payload % expected_experts == 0
    stride = payload // expected_experts if divisible else 0
    safe = axis_ok and divisible
    reason = "contiguous outer expert axis and equal per-expert byte stride" if safe else f"axis_ok={axis_ok}, divisible={divisible}, shape={t.shape}, span={t.span_bytes}"
    return ExpertTensor(layer, kind, t.name, expected_experts, axis, t.absolute_offset, t.span_bytes, payload, tail_padding, stride, safe, reason)


def build_expert_map(path: str | Path, expected_experts: int = 512) -> ExpertMap:
    version, alignment, infos = read_tensor_directory(path)
    tensors = tuple(e for t in infos if (e := _expert_layout(t, expected_experts, alignment)) is not None)
    return ExpertMap(str(Path(path).expanduser().resolve()), version, alignment, len(infos), expected_experts, tensors)
