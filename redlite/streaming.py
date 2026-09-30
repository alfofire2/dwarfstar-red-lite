from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import random
import time
from typing import Any

from .expert_cache import CacheKey, ExpertSlotCache
from .expert_map import ExpertMap, build_expert_map
from .expert_store import build_expert_layouts

GIB = 1024 ** 3
MIB = 1024 ** 2


@dataclass(frozen=True)
class StreamingPlan:
    model: str
    cache_gib: float
    cache_bytes: int
    expert_count: int
    layers: int
    routed_tensor_count: int
    routed_payload_gib: float
    slice_safe: bool
    max_expert_triplet_mib: float
    slot_bytes: int
    slot_capacity: int
    ready_for_probe: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_streaming_plan(
    model: str | Path,
    cache_gib: float = 4.0,
    expected_experts: int = 512,
) -> tuple[ExpertMap, StreamingPlan]:
    if cache_gib <= 0:
        raise ValueError("cache_gib must be positive")

    expert_map = build_expert_map(model, expected_experts)
    layouts = build_expert_layouts(expert_map) if expert_map.all_slice_safe else {}
    max_triplet = max((layout.total_bytes for layout in layouts.values()), default=0)
    cache_bytes = int(cache_gib * GIB)
    slot_capacity = cache_bytes // max_triplet if max_triplet else 0
    safe = expert_map.all_slice_safe and bool(expert_map.layers) and bool(layouts)

    reason = (
        "GGUF routed experts are sliceable; explicit single-fd positional reads can fill a "
        "hard-bounded LRU of reusable expert slots. Metal binding is not enabled yet."
        if safe
        else "Merged expert tensor layout is not safely sliceable into gate/up/down triplets."
    )

    return expert_map, StreamingPlan(
        model=str(Path(model).expanduser().resolve()),
        cache_gib=cache_gib,
        cache_bytes=cache_bytes,
        expert_count=expected_experts,
        layers=len(expert_map.layers),
        routed_tensor_count=len(expert_map.expert_tensors),
        routed_payload_gib=expert_map.total_routed_payload_bytes / GIB,
        slice_safe=safe,
        max_expert_triplet_mib=max_triplet / MIB,
        slot_bytes=max_triplet,
        slot_capacity=slot_capacity,
        ready_for_probe=safe and slot_capacity >= expected_experts // 8,
        reason=reason,
    )


def probe_streaming(
    model: str | Path,
    cache_gib: float = 4.0,
    steps: int = 4,
    top_k: int = 10,
    seed: int = 1337,
    prefetch_depth: int = 1,
    prefetch_workers: int = 2,
) -> dict[str, Any]:
    expert_map, plan = make_streaming_plan(model, cache_gib)
    if not plan.ready_for_probe:
        raise ValueError(plan.reason)
    if steps <= 0 or top_k <= 0 or top_k > expert_map.expert_count:
        raise ValueError("invalid steps/top_k")
    if prefetch_depth < 0:
        raise ValueError("prefetch_depth must be >= 0")

    layouts = build_expert_layouts(expert_map)
    rng = random.Random(seed)

    # Precompute the synthetic routing trace so future layer selections are known
    # to the probe. This is a benchmark aid only; the real runtime will get the
    # next expert ids from Qwen3-Next's router and overlap those reads with useful
    # Metal work, following the same broad strategy as DS4.
    trace: list[tuple[int, tuple[int, ...]]] = []
    for _ in range(steps):
        for layer in expert_map.layers:
            chosen = tuple(rng.sample(range(expert_map.expert_count), top_k))
            trace.append((layer, chosen))

    checksum = 0
    route_events = 0
    started = time.perf_counter()

    with ExpertSlotCache(
        model,
        plan.cache_bytes,
        plan.slot_bytes,
        prefetch_workers=prefetch_workers,
    ) as cache:
        for index, (layer, chosen) in enumerate(trace):
            if prefetch_depth:
                for future_index in range(
                    index + 1,
                    min(len(trace), index + 1 + prefetch_depth),
                ):
                    future_layer, future_experts = trace[future_index]
                    for expert in future_experts:
                        key = CacheKey(future_layer, expert)
                        cache.prefetch(key, layouts[(future_layer, expert)])

            for expert in chosen:
                route_events += 1
                key = CacheKey(layer, expert)
                view = cache.acquire(key, layouts[(layer, expert)])
                if len(view):
                    checksum ^= int(view[0])
                    checksum ^= int(view[-1])
                view.release()

        cache.wait_prefetch()
        elapsed = time.perf_counter() - started
        telemetry = cache.telemetry()
        resident = len(cache.resident_keys())

    store_stats = telemetry["store"]
    bytes_read = int(store_stats["bytes_read"])
    return {
        "plan": plan.to_dict(),
        "probe": {
            "steps": steps,
            "top_k": top_k,
            "seed": seed,
            "prefetch_depth": prefetch_depth,
            "prefetch_workers": prefetch_workers,
            "route_events": route_events,
            "elapsed_sec": elapsed,
            "route_events_per_sec": route_events / elapsed if elapsed else None,
            "ssd_read_gib": bytes_read / GIB,
            "ssd_read_mib_per_sec": (bytes_read / MIB) / elapsed if elapsed else None,
            "resident_slots": resident,
            "checksum": checksum,
            "cache": telemetry,
            "note": (
                "Synthetic router trace using explicit positional SSD reads into a hard-bounded "
                "reusable LRU slot cache. This validates the storage/residency path only; Metal "
                "MoE execution is not connected yet."
            ),
        },
    }
