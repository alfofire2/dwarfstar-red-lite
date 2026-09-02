from __future__ import annotations

from pathlib import Path
import random
import time
from typing import Any

from .expert_cache import CacheKey
from .expert_store import ExpertStore, build_expert_layouts
from .redmetal import redmetal_device_name, redmetal_library_path
from .redmetal_cache import MetalExpertCache
from .streaming import GIB, MIB, make_streaming_plan


def sampled_fnv1a(data: bytes | bytearray | memoryview) -> int:
    view = memoryview(data)
    h = 2166136261
    if len(view) == 0:
        return h
    for index in range(0, len(view), 4096):
        h = ((h ^ int(view[index])) * 16777619) & 0xFFFFFFFF
    h = ((h ^ int(view[-1])) * 16777619) & 0xFFFFFFFF
    view.release()
    return h


def probe_redmetal(
    model: str | Path,
    cache_gib: float = 0.5,
    steps: int = 1,
    top_k: int = 10,
    seed: int = 1337,
    slots_per_slab: int = 64,
) -> dict[str, Any]:
    expert_map, plan = make_streaming_plan(model, cache_gib)
    if not plan.ready_for_probe:
        raise ValueError(plan.reason)
    if steps <= 0 or top_k <= 0 or top_k > expert_map.expert_count:
        raise ValueError("invalid steps/top_k")
    if slots_per_slab <= 0:
        raise ValueError("slots_per_slab must be positive")

    layouts = build_expert_layouts(expert_map)
    rng = random.Random(seed)
    trace: list[tuple[int, tuple[int, ...]]] = []
    for _ in range(steps):
        for layer in expert_map.layers:
            trace.append(
                (layer, tuple(rng.sample(range(expert_map.expert_count), top_k)))
            )

    route_events = 0
    last_key: CacheKey | None = None
    started = time.perf_counter()

    with MetalExpertCache(
        model,
        plan.cache_bytes,
        plan.slot_bytes,
        slots_per_slab=slots_per_slab,
    ) as cache:
        for layer, chosen in trace:
            for expert in chosen:
                route_events += 1
                key = CacheKey(layer, expert)
                cache.acquire(key, layouts[(layer, expert)])
                last_key = key

        elapsed = time.perf_counter() - started
        if last_key is None:
            raise RuntimeError("empty Red Metal routing trace")
        entry = cache.get(last_key)
        if entry is None:
            raise RuntimeError("final Red Metal expert unexpectedly evicted")

        gpu = cache.pool.gpu_probe(entry.slot_id, entry.payload_bytes)
        telemetry = cache.telemetry()

        # Verification-only reread. The production Red Metal data path above does
        # not pass through this bytearray; it exists solely to prove that the GPU
        # sees the same bytes that were read from GGUF into MTLBuffer.contents.
        layout = layouts[(last_key.layer, last_key.expert)]
        slot = bytearray(plan.slot_bytes)
        with ExpertStore(model) as store:
            view = store.load(layout, slot)
            cpu_checksum = sampled_fnv1a(view)
            view.release()

    metal = telemetry["metal"]
    bytes_read = int(metal["bytes_read"])
    return {
        "plan": plan.to_dict(),
        "probe": {
            "device": redmetal_device_name(),
            "library": str(redmetal_library_path()),
            "steps": steps,
            "top_k": top_k,
            "seed": seed,
            "slots_per_slab": slots_per_slab,
            "route_events": route_events,
            "elapsed_sec": elapsed,
            "route_events_per_sec": route_events / elapsed if elapsed else None,
            "ssd_read_gib": bytes_read / GIB,
            "ssd_read_mib_per_sec": (bytes_read / MIB) / elapsed if elapsed else None,
            "gpu_probe_checksum": int(gpu["checksum"]),
            "cpu_verify_checksum": int(cpu_checksum),
            "gpu_checksum_match": int(gpu["checksum"]) == int(cpu_checksum),
            "gpu_probe_ms": float(gpu["elapsed_ms"]),
            "last_layer": last_key.layer,
            "last_expert": last_key.expert,
            "cache": telemetry,
            "note": (
                "Expert misses are read directly from the GGUF fd into CPU-visible shared "
                "MTLBuffer slab ranges. A Metal compute kernel samples one resident expert "
                "and is compared with a verification-only CPU reread. MoE matmul is not yet "
                "connected."
            ),
        },
    }
