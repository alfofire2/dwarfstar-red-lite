from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import subprocess
from typing import Any

from .gguf import read_index, write_expert_manifest
from .runner import ROOT

STREAM_BUILD = ROOT / ".deps" / "red-stream" / "red-stream-probe"


@dataclass(frozen=True)
class StreamingPlan:
    model_gib: float
    routed_expert_gib: float
    non_routed_gib: float
    cache_gib: float
    estimated_resident_gib: float
    ram_gib: float
    reserve_gib: float
    headroom_gib: float
    expert_records: int
    layers: int
    safe: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def streaming_plan(model: str, ram_gib: float, cache_gib: float, reserve_gib: float = 4.5) -> StreamingPlan:
    idx = read_index(model)
    records = idx.expert_records()
    routed = idx.routed_bytes / (1024 ** 3)
    model_gib = idx.file_size / (1024 ** 3)
    non_routed = max(0.0, model_gib - routed)
    resident = non_routed + cache_gib
    headroom = ram_gib - reserve_gib - resident
    layers = len({r.layer for r in records})
    return StreamingPlan(model_gib, routed, non_routed, cache_gib, resident, ram_gib,
                         reserve_gib, headroom, len(records), layers, headroom >= 0)


def build_manifest(model: str, output: str) -> tuple[Path, dict[str, Any]]:
    idx = read_index(model)
    path = write_expert_manifest(idx, output)
    records = idx.expert_records()
    sizes = [r.total_bytes for r in records]
    summary = {
        "model": idx.path,
        "gguf_version": idx.version,
        "tensor_count": idx.tensor_count,
        "routed_tensor_count": len(idx.routed_tensors),
        "routed_expert_gib": idx.routed_bytes / (1024 ** 3),
        "expert_records": len(records),
        "layers": len({r.layer for r in records}),
        "experts_per_layer": max((r.expert for r in records), default=-1) + 1,
        "expert_record_min_kib": min(sizes) / 1024 if sizes else 0,
        "expert_record_max_kib": max(sizes) / 1024 if sizes else 0,
        "manifest": str(path),
    }
    return path, summary


def native_probe(model: str, manifest: str, cache_mib: int, requests: int, hotset: int,
                 prefetch: int, seed: int, nocache: bool, dry_run: bool = False) -> int:
    if not STREAM_BUILD.exists():
        raise FileNotFoundError("native streaming probe not built; run: redlite stream-build")
    cmd = [str(STREAM_BUILD), "--model", model, "--manifest", manifest,
           "--cache-mib", str(cache_mib), "--requests", str(requests),
           "--hotset", str(hotset), "--prefetch", str(prefetch), "--seed", str(seed)]
    if nocache:
        cmd.append("--nocache")
    print("[redlite]", " ".join(cmd))
    if dry_run:
        return 0
    return subprocess.call(cmd)
