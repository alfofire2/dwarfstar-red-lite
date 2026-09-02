from __future__ import annotations

from pathlib import Path
from typing import Any

from .expert_cache import CacheKey
from .expert_map import read_tensor_directory
from .expert_store import build_expert_layouts
from .quant_reference import deterministic_input, reference_rows, row_bytes_for_type
from .quant_tables import SUPPORTED_ROUTED_TYPES, load_quant_grid, quant_type_name
from .redmetal_exec_cache import ExecMetalExpertCache
from .redmetal_quant import RedMetalQuantBridge
from .streaming import GIB, make_streaming_plan


def _tensor_info(model: str | Path, layer: int, kind: str):
    _, _, infos = read_tensor_directory(model)
    name = f"blk.{layer}.ffn_{kind}_exps.weight"
    for info in infos:
        if info.name == name:
            return info
    raise ValueError(f"tensor {name!r} was not found in the GGUF")


def quant_parity_probe(
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
    if tensor.ggml_type not in SUPPORTED_ROUTED_TYPES:
        raise ValueError(
            f"{tensor.name} is GGML type {tensor.ggml_type} ({quant_type_name(tensor.ggml_type)}); "
            f"supported routed types are {sorted(SUPPORTED_ROUTED_TYPES)}"
        )
    if len(tensor.shape) != 3 or tensor.shape[-1] != expert_map.expert_count:
        raise ValueError(f"unexpected routed tensor shape {tensor.shape}")

    ncols = int(tensor.shape[0])
    nrows = int(tensor.shape[1])
    if ncols <= 0 or ncols % 256:
        raise ValueError(f"{quant_type_name(tensor.ggml_type)} ncols={ncols} is not a positive multiple of 256")
    if row_start + rows > nrows:
        raise ValueError(f"row range [{row_start}, {row_start + rows}) exceeds {nrows} rows")

    layouts = build_expert_layouts(expert_map)
    layout = layouts[(layer, expert)]
    parts = {part.kind: part for part in layout.parts}
    part = parts[kind]
    row_bytes = row_bytes_for_type(tensor.ggml_type, ncols)
    expected_matrix_bytes = nrows * row_bytes
    if part.length != expected_matrix_bytes:
        raise ValueError(
            f"{tensor.name} expert slice is {part.length} bytes, but shape {tensor.shape} "
            f"implies {expected_matrix_bytes} {quant_type_name(tensor.ggml_type)} bytes "
            f"({nrows} x {row_bytes})"
        )

    input_values = deterministic_input(ncols)
    grid = load_quant_grid(tensor.ggml_type)
    key = CacheKey(layer, expert)
    budget_bytes = int(cache_gib * GIB)

    # Keep the dev5 residency/address-table path in the parity test. Arithmetic is
    # intentionally isolated in RedMetalQuantBridge until both real routed formats
    # are numerically validated on the target M4 Pro.
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
        inflight_after_wait = cache.pool.slot_inflight(entry.slot_id)
        telemetry = cache.telemetry()

    bridge = RedMetalQuantBridge()
    gpu, io_ms, gpu_ms = bridge.rows(
        model,
        part.file_offset,
        part.length,
        tensor.ggml_type,
        ncols,
        nrows,
        row_start,
        rows,
        input_values,
        grid,
    )
    cpu = reference_rows(
        model,
        part.file_offset,
        tensor.ggml_type,
        ncols,
        row_start,
        rows,
        input_values,
        grid,
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
        "type_name": quant_type_name(tensor.ggml_type),
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
        "io_ms": io_ms,
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
            f"Correctness-first single-expert {quant_type_name(tensor.ggml_type)} row matvec. "
            "The main dev5 cache still validates SSD-filled shared Metal slots, GPU address tables, "
            "and in-flight lifetime. The isolated arithmetic bridge rereads only the selected matrix "
            "directly into shared Metal memory so IQ2_XS and IQ1_M can be proven numerically before "
            "their kernels are integrated into the LRU execution pool."
        ),
    }


# Backward-compatible name used by the current CLI command.
iq2_parity_probe = quant_parity_probe
