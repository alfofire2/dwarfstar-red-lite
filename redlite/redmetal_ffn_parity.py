from __future__ import annotations

import math
from pathlib import Path
import time
from typing import Any

from .expert_cache import CacheKey
from .expert_map import read_tensor_directory
from .expert_store import build_expert_layouts
from .quant_reference import deterministic_input, reference_rows, row_bytes_for_type
from .quant_tables import SUPPORTED_ROUTED_TYPES, load_quant_grid, quant_type_name
from .redmetal_ffn_cache import FfnMetalExpertCache
from .streaming import GIB, make_streaming_plan


def _layer_tensors(model: str | Path, layer: int):
    _, _, infos = read_tensor_directory(model)
    wanted = {
        kind: f"blk.{layer}.ffn_{kind}_exps.weight"
        for kind in ("gate", "up", "down")
    }
    found = {}
    for info in infos:
        for kind, name in wanted.items():
            if info.name == name:
                found[kind] = info
    missing = sorted(set(wanted) - set(found))
    if missing:
        raise ValueError(f"missing routed tensors for layer {layer}: {missing}")
    return found


def _silu_mul(gate: list[float], up: list[float]) -> list[float]:
    if len(gate) != len(up):
        raise ValueError("gate/up reference size mismatch")
    out = []
    for g, u in zip(gate, up):
        if g >= 0.0:
            sig = 1.0 / (1.0 + math.exp(-g))
        else:
            eg = math.exp(g)
            sig = eg / (1.0 + eg)
        out.append((g * sig) * u)
    return out


def expert_ffn_parity_probe(
    model: str | Path,
    *,
    layer: int = 0,
    expert: int = 0,
    row_start: int = 0,
    rows: int = 8,
    cache_gib: float = 0.25,
    slots_per_slab: int = 64,
    atol: float = 1e-3,
    rtol: float = 1e-4,
) -> dict[str, Any]:
    if rows <= 0 or row_start < 0:
        raise ValueError("rows must be positive and row_start non-negative")
    model = str(Path(model).expanduser().resolve())
    expert_map, plan = make_streaming_plan(model, cache_gib)
    if not plan.slice_safe:
        raise ValueError(plan.reason)
    if layer not in expert_map.layers:
        raise ValueError(f"layer {layer} is not routed")
    if not 0 <= expert < expert_map.expert_count:
        raise ValueError(f"expert must be in [0, {expert_map.expert_count - 1}]")

    tensors = _layer_tensors(model, layer)
    types = {int(info.ggml_type) for info in tensors.values()}
    if len(types) != 1:
        raise ValueError(f"layer {layer} has mixed gate/up/down quant types: {sorted(types)}")
    ggml_type = types.pop()
    if ggml_type not in SUPPORTED_ROUTED_TYPES:
        raise ValueError(f"unsupported routed quant type {quant_type_name(ggml_type)}")

    gate_shape = tuple(int(v) for v in tensors["gate"].shape)
    up_shape = tuple(int(v) for v in tensors["up"].shape)
    down_shape = tuple(int(v) for v in tensors["down"].shape)
    if len(gate_shape) != 3 or len(up_shape) != 3 or len(down_shape) != 3:
        raise ValueError("routed FFN tensors must be rank 3")
    hidden_size, ffn_size, experts = gate_shape
    if up_shape != gate_shape:
        raise ValueError(f"gate/up shape mismatch: {gate_shape} vs {up_shape}")
    if down_shape != (ffn_size, hidden_size, experts):
        raise ValueError(
            f"unexpected down shape {down_shape}; expected {(ffn_size, hidden_size, experts)}"
        )
    if experts != expert_map.expert_count:
        raise ValueError(f"shape reports {experts} experts, map reports {expert_map.expert_count}")
    if row_start + rows > hidden_size:
        raise ValueError(f"output rows [{row_start}, {row_start + rows}) exceed hidden size {hidden_size}")

    layouts = build_expert_layouts(expert_map)
    layout = layouts[(layer, expert)]
    parts = {part.kind: part for part in layout.parts}
    gate_row_bytes = row_bytes_for_type(ggml_type, hidden_size)
    down_row_bytes = row_bytes_for_type(ggml_type, ffn_size)
    expected = {
        "gate": ffn_size * gate_row_bytes,
        "up": ffn_size * gate_row_bytes,
        "down": hidden_size * down_row_bytes,
    }
    for kind, length in expected.items():
        if parts[kind].length != length:
            raise ValueError(
                f"{kind} expert slice is {parts[kind].length} bytes, expected {length} "
                f"for {quant_type_name(ggml_type)}"
            )

    input_values = deterministic_input(hidden_size)
    key = CacheKey(layer, expert)
    budget_bytes = int(cache_gib * GIB)

    with FfnMetalExpertCache(
        model,
        budget_bytes,
        plan.slot_bytes,
        slots_per_slab=slots_per_slab,
    ) as cache:
        entry = cache.acquire(key, layout)
        addresses = cache.addresses(key)
        if addresses is None or not all(addresses):
            raise RuntimeError("resident FFN expert did not expose GPU addresses")
        before_execute = cache.pool.telemetry()
        gpu, gpu_ms = cache.execute(
            key,
            ggml_type,
            hidden_size,
            ffn_size,
            row_start,
            rows,
            input_values,
        )
        after_execute = cache.pool.telemetry()
        if after_execute["bytes_read"] != before_execute["bytes_read"]:
            raise RuntimeError("FFN execution performed unexpected SSD reads after expert residency")
        inflight_after_wait = cache.pool.slot_inflight(entry.slot_id)
        telemetry = cache.telemetry()

    grid = load_quant_grid(ggml_type)
    cpu_t0 = time.perf_counter()
    gate_cpu = reference_rows(
        model,
        parts["gate"].file_offset,
        ggml_type,
        hidden_size,
        0,
        ffn_size,
        input_values,
        grid,
    )
    up_cpu = reference_rows(
        model,
        parts["up"].file_offset,
        ggml_type,
        hidden_size,
        0,
        ffn_size,
        input_values,
        grid,
    )
    activated = _silu_mul(gate_cpu, up_cpu)
    cpu = reference_rows(
        model,
        parts["down"].file_offset,
        ggml_type,
        ffn_size,
        row_start,
        rows,
        activated,
        grid,
    )
    cpu_ms = (time.perf_counter() - cpu_t0) * 1000.0

    errors = [abs(a - b) for a, b in zip(gpu, cpu)]
    rel_errors = [err / max(abs(ref), 1e-12) for err, ref in zip(errors, cpu)]
    max_abs = max(errors, default=0.0)
    max_rel = max(rel_errors, default=0.0)
    match = all(err <= atol + rtol * abs(ref) for err, ref in zip(errors, cpu))

    return {
        "model": model,
        "layer": layer,
        "expert": expert,
        "quant_name": quant_type_name(ggml_type),
        "ggml_type": ggml_type,
        "hidden_size": hidden_size,
        "ffn_size": ffn_size,
        "gate_shape": list(gate_shape),
        "down_shape": list(down_shape),
        "row_start": row_start,
        "rows_tested": rows,
        "slot_id": entry.slot_id,
        "gpu_addresses": {"gate": addresses[0], "up": addresses[1], "down": addresses[2]},
        "inflight_after_wait": inflight_after_wait,
        "gpu_ms": gpu_ms,
        "cpu_reference_ms": cpu_ms,
        "gpu": gpu,
        "cpu": cpu,
        "max_abs_error": max_abs,
        "max_rel_error": max_rel,
        "atol": atol,
        "rtol": rtol,
        "match": match,
        "cache": telemetry,
        "ssd_reads_during_execute": after_execute["bytes_read"] - before_execute["bytes_read"],
        "note": (
            "Full single-expert routed FFN parity: the expert triplet is loaded once into a hard-bounded "
            "shared Metal LRU slot, then gate and up matvecs, SiLU(gate)*up, and selected down rows run "
            "without rereading the GGUF. CPU reference independently evaluates the same quantized expert."
        ),
    }
