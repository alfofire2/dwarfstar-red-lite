from __future__ import annotations

from pathlib import Path
from typing import Any

from .expert_cache import CacheKey
from .expert_map import read_tensor_directory
from .expert_store import build_expert_layouts
from .iq2_reference import (
    IQ2_XXS_BLOCK_BYTES,
    IQ2_XXS_BLOCK_VALUES,
    deterministic_input,
    reference_rows,
)
from .redmetal_exec_cache import ExecMetalExpertCache
from .streaming import GIB, make_streaming_plan


def _tensor_info(model: str | Path, layer: int, kind: str):
    _, _, infos = read_tensor_directory(model)
    name = f"blk.{layer}.ffn_{kind}_exps.weight"
    for info in infos:
        if info.name == name:
            return info
    raise ValueError(f"tensor {name!r} was not found in the GGUF")


def iq2_parity_probe(
    model: str | Path,
    *,
    layer: int = 0,
    expert: int = 0,
    kind: str = "gate",
    row_start: int = 0,
    rows: int = 8,
    cache_gib: float = 0.25,
    slots_per_slab: int = 64,
    atol: float = 1e-3,
    rtol: float = 1e-4,
) -> dict[str, Any]:
    if kind not in {"gate", "up", "down"}:
        raise ValueError("kind must be gate, up, or down")
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

    tensor = _tensor_info(model, layer, kind)
    if tensor.ggml_type != 16:
        raise ValueError(
            f"{tensor.name} is GGML type {tensor.ggml_type}, expected IQ2_XXS type 16"
        )
    if len(tensor.shape) != 3 or tensor.shape[-1] != expert_map.expert_count:
        raise ValueError(f"unexpected routed tensor shape {tensor.shape}")

    ncols = int(tensor.shape[0])
    nrows = int(tensor.shape[1])
    if ncols <= 0 or ncols % IQ2_XXS_BLOCK_VALUES:
        raise ValueError(f"IQ2_XXS ncols={ncols} is not a positive multiple of 256")
    if row_start + rows > nrows:
        raise ValueError(f"row range [{row_start}, {row_start + rows}) exceeds {nrows} rows")

    layouts = build_expert_layouts(expert_map)
    layout = layouts[(layer, expert)]
    parts = {part.kind: part for part in layout.parts}
    part = parts[kind]
    row_bytes = (ncols // IQ2_XXS_BLOCK_VALUES) * IQ2_XXS_BLOCK_BYTES
    expected_matrix_bytes = nrows * row_bytes
    if part.length != expected_matrix_bytes:
        raise ValueError(
            f"{tensor.name} expert slice is {part.length} bytes, but shape {tensor.shape} "
            f"implies {expected_matrix_bytes} IQ2_XXS bytes ({nrows} x {row_bytes})"
        )

    input_values = deterministic_input(ncols)
    key = CacheKey(layer, expert)
    budget_bytes = int(cache_gib * GIB)

    with ExecMetalExpertCache(
        model,
        budget_bytes,
        plan.slot_bytes,
        slots_per_slab=slots_per_slab,
    ) as cache:
        entry = cache.acquire(key, layout)
        addresses = cache.addresses(key)
        if addresses is None or not all(addresses):
            raise RuntimeError("expert was loaded but GPU address table is empty")
        gpu, gpu_ms = cache.pool.iq2_rows(
            entry.slot_id,
            part.slot_offset,
            ncols,
            nrows,
            row_start,
            rows,
            input_values,
        )
        inflight_after_wait = cache.pool.slot_inflight(entry.slot_id)
        telemetry = cache.telemetry()

    cpu = reference_rows(
        model,
        part.file_offset,
        ncols,
        row_start,
        rows,
        input_values,
    )

    errors = [abs(a - b) for a, b in zip(gpu, cpu)]
    rel_errors = [err / max(abs(ref), 1e-12) for err, ref in zip(errors, cpu)]
    max_abs = max(errors, default=0.0)
    max_rel = max(rel_errors, default=0.0)
    per_row_ok = [err <= atol + rtol * abs(ref) for err, ref in zip(errors, cpu)]
    match = all(per_row_ok)

    return {
        "model": model,
        "layer": layer,
        "expert": expert,
        "kind": kind,
        "tensor": tensor.name,
        "shape": list(tensor.shape),
        "ggml_type": tensor.ggml_type,
        "ncols": ncols,
        "nrows": nrows,
        "row_start": row_start,
        "rows_tested": rows,
        "row_bytes": row_bytes,
        "expert_matrix_bytes": part.length,
        "slot_id": entry.slot_id,
        "gpu_addresses": {
            "gate": addresses[0],
            "up": addresses[1],
            "down": addresses[2],
        },
        "inflight_after_wait": inflight_after_wait,
        "gpu_ms": gpu_ms,
        "gpu": gpu,
        "cpu": cpu,
        "max_abs_error": max_abs,
        "max_rel_error": max_rel,
        "atol": atol,
        "rtol": rtol,
        "match": match,
        "cache": telemetry,
        "note": (
            "Correctness-first single-expert IQ2_XXS row matvec. GPU reads the quantized "
            "matrix from an SSD-filled shared Metal slab; CPU reference independently decodes "
            "the same GGUF rows in Python. This is not yet the fused routed MoE path."
        ),
    }
