from __future__ import annotations

import math
from pathlib import Path
import struct
import time
from typing import Any, Sequence

from .expert_cache import CacheKey
from .expert_map import read_tensor_directory
from .expert_store import ExpertLayout, build_expert_layouts
from .quant_reference import deterministic_input, reference_rows, row_bytes_for_type
from .quant_tables import SUPPORTED_ROUTED_TYPES, load_quant_grid, quant_type_name
from .redmetal_topk_cache import TopKMetalExpertCache
from .streaming import GIB, make_streaming_plan


def _layer_tensors(model: str | Path, layer: int):
    _, _, infos = read_tensor_directory(model)
    wanted = {kind: f"blk.{layer}.ffn_{kind}_exps.weight" for kind in ("gate", "up", "down")}
    found = {}
    for info in infos:
        for kind, name in wanted.items():
            if info.name == name:
                found[kind] = info
    missing = sorted(set(wanted) - set(found))
    if missing:
        raise ValueError(f"missing routed tensors for layer {layer}: {missing}")
    return found


def _silu_mul(gate: Sequence[float], up: Sequence[float]) -> list[float]:
    if len(gate) != len(up):
        raise ValueError("gate/up reference size mismatch")
    out: list[float] = []
    for g, u in zip(gate, up):
        if g >= 0.0:
            sig = 1.0 / (1.0 + math.exp(-g))
        else:
            eg = math.exp(g)
            sig = eg / (1.0 + eg)
        out.append((g * sig) * u)
    return out


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def default_expert_ids(top_k: int, expert_count: int) -> list[int]:
    if top_k <= 0 or top_k > expert_count:
        raise ValueError(f"top_k must be in [1, {expert_count}]")
    result: list[int] = []
    candidate = 7 % expert_count
    step = 47
    while len(result) < top_k:
        if candidate not in result:
            result.append(candidate)
        candidate = (candidate + step) % expert_count
    return result


def default_router_weights(top_k: int) -> list[float]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    raw = [float(top_k - i) for i in range(top_k)]
    total = sum(raw)
    return [_f32(value / total) for value in raw]


def _validate_layout(
    layout: ExpertLayout,
    ggml_type: int,
    hidden_size: int,
    ffn_size: int,
) -> None:
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


def _reference_expert(
    model: str,
    layout: ExpertLayout,
    ggml_type: int,
    hidden_size: int,
    ffn_size: int,
    row_start: int,
    rows: int,
    input_values: Sequence[float],
    grid: Sequence[int],
) -> list[float]:
    parts = {part.kind: part for part in layout.parts}
    gate = reference_rows(
        model,
        parts["gate"].file_offset,
        ggml_type,
        hidden_size,
        0,
        ffn_size,
        input_values,
        grid,
    )
    up = reference_rows(
        model,
        parts["up"].file_offset,
        ggml_type,
        hidden_size,
        0,
        ffn_size,
        input_values,
        grid,
    )
    activated = _silu_mul(gate, up)
    return reference_rows(
        model,
        parts["down"].file_offset,
        ggml_type,
        ffn_size,
        row_start,
        rows,
        activated,
        grid,
    )


def topk_layer_parity_probe(
    model: str | Path,
    *,
    layer: int = 0,
    top_k: int = 10,
    expert_ids: Sequence[int] | None = None,
    router_weights: Sequence[float] | None = None,
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

    selected = list(expert_ids) if expert_ids is not None else default_expert_ids(top_k, expert_map.expert_count)
    if not selected:
        raise ValueError("at least one expert must be selected")
    if len(selected) > 64:
        raise ValueError("top-k executor currently supports at most 64 experts")
    if len(set(selected)) != len(selected):
        raise ValueError("selected expert ids must be unique")
    if any(expert < 0 or expert >= expert_map.expert_count for expert in selected):
        raise ValueError(f"expert ids must be in [0, {expert_map.expert_count - 1}]")
    if expert_ids is not None and top_k != 10 and top_k != len(selected):
        raise ValueError("when both top_k and expert_ids are provided, their counts must match")
    top_k = len(selected)

    weights = list(router_weights) if router_weights is not None else default_router_weights(top_k)
    if len(weights) != top_k:
        raise ValueError("router-weight count must match selected experts")
    if any(not math.isfinite(value) for value in weights):
        raise ValueError("router weights must be finite")
    weights = [_f32(value) for value in weights]

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
        raise ValueError(f"unexpected down shape {down_shape}; expected {(ffn_size, hidden_size, experts)}")
    if experts != expert_map.expert_count:
        raise ValueError(f"shape reports {experts} experts, map reports {expert_map.expert_count}")
    if row_start + rows > hidden_size:
        raise ValueError(f"output rows [{row_start}, {row_start + rows}) exceed hidden size {hidden_size}")

    all_layouts = build_expert_layouts(expert_map)
    keys = [CacheKey(layer, expert) for expert in selected]
    layouts = [all_layouts[(layer, expert)] for expert in selected]
    for layout in layouts:
        _validate_layout(layout, ggml_type, hidden_size, ffn_size)

    input_values = deterministic_input(hidden_size)
    budget_bytes = int(cache_gib * GIB)

    with TopKMetalExpertCache(
        model,
        budget_bytes,
        plan.slot_bytes,
        slots_per_slab=slots_per_slab,
    ) as cache:
        entries = cache.acquire_many(zip(keys, layouts))
        addresses = [cache.addresses(key) for key in keys]
        if any(value is None or not all(value) for value in addresses):
            raise RuntimeError("one or more resident top-k experts did not expose GPU addresses")
        before_execute = cache.pool.telemetry()
        gpu, gpu_ms = cache.execute(
            keys,
            weights,
            ggml_type,
            hidden_size,
            ffn_size,
            row_start,
            rows,
            input_values,
        )
        after_execute = cache.pool.telemetry()
        ssd_bytes_during_execute = after_execute["bytes_read"] - before_execute["bytes_read"]
        ssd_calls_during_execute = after_execute["read_calls"] - before_execute["read_calls"]
        if ssd_bytes_during_execute or ssd_calls_during_execute:
            raise RuntimeError("top-k execution performed unexpected SSD reads after expert residency")
        inflight_after_wait = [cache.pool.slot_inflight(entry.slot_id) for entry in entries]
        telemetry = cache.telemetry()

    grid = load_quant_grid(ggml_type)
    cpu_t0 = time.perf_counter()
    cpu = [0.0] * rows
    for weight, layout in zip(weights, layouts):
        expert_output = _reference_expert(
            model,
            layout,
            ggml_type,
            hidden_size,
            ffn_size,
            row_start,
            rows,
            input_values,
            grid,
        )
        for index, value in enumerate(expert_output):
            cpu[index] += float(weight) * value
    cpu_ms = (time.perf_counter() - cpu_t0) * 1000.0

    errors = [abs(a - b) for a, b in zip(gpu, cpu)]
    rel_errors = [err / max(abs(ref), 1e-12) for err, ref in zip(errors, cpu)]
    max_abs = max(errors, default=0.0)
    max_rel = max(rel_errors, default=0.0)
    match = all(err <= atol + rtol * abs(ref) for err, ref in zip(errors, cpu))

    return {
        "model": model,
        "layer": layer,
        "top_k": top_k,
        "experts": selected,
        "router_weights": weights,
        "router_weight_sum": sum(weights),
        "quant_name": quant_type_name(ggml_type),
        "ggml_type": ggml_type,
        "hidden_size": hidden_size,
        "ffn_size": ffn_size,
        "row_start": row_start,
        "rows_tested": rows,
        "slot_ids": [entry.slot_id for entry in entries],
        "gpu_addresses": [
            {"gate": value[0], "up": value[1], "down": value[2]}  # type: ignore[index]
            for value in addresses
        ],
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
        "ssd_bytes_during_execute": ssd_bytes_during_execute,
        "ssd_calls_during_execute": ssd_calls_during_execute,
        "note": (
            "Correctness-first routed top-k layer parity: all selected experts are made resident first, "
            "their slots remain pinned for one Metal command buffer, each complete FFN runs from resident "
            "gate/up/down bytes, and router-weighted accumulation stays on GPU until the final layer output."
        ),
    }
