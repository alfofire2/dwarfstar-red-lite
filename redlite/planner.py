from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from .hardware import HardwareInfo, GIB


@dataclass(frozen=True)
class Plan:
    mode: str
    status: str
    context: int
    threads: int
    batch: int
    ubatch: int
    model_gib: float
    ram_gib: float
    os_reserve_gib: float
    runtime_reserve_gib: float
    context_reserve_gib: float
    resident_budget_gib: float
    headroom_gib: float
    safe: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def context_reserve_gib(context: int) -> float:
    # Qwen3-Next mixes recurrent/linear-attention and full-attention blocks. This
    # is a policy reserve, not an attempt to reproduce llama.cpp allocation byte
    # for byte. Real memory pressure remains the source of truth at runtime.
    return 0.55 + (context / 8192.0) * 0.45


def _is_field_validated_m4pro_24gb(chip: str, ram_gib: float, model_gib: float) -> bool:
    return (
        "Apple M4 Pro" in chip
        and 23.5 <= ram_gib <= 24.5
        and 17.5 <= model_gib <= 18.3
    )


def _default_context(ram_gib: float, model_gib: float, chip: str = "") -> int:
    # A controlled Red Lite sweep on Apple M4 Pro / 24 GiB with the 17.97 GiB
    # Qwen3-Next IQ2_XXS model completed 2K, 4K and 8K with zero observed swap
    # delta. Use 4K as the field-validated default for that exact target class;
    # keep 8K opt-in because the planner's estimated margin is still very small.
    if ram_gib <= 16:
        return 2048
    if ram_gib <= 24:
        if model_gib >= 17.0:
            return 4096 if _is_field_validated_m4pro_24gb(chip, ram_gib, model_gib) else 2048
        return 4096
    if ram_gib <= 36:
        return 8192
    if model_gib > ram_gib:
        return 8192
    return 16384


def _resident_status(headroom_gib: float) -> str:
    if headroom_gib < 0.0:
        return "UNSAFE"
    if headroom_gib < 1.0:
        return "CRITICAL"
    if headroom_gib < 2.0:
        return "TIGHT"
    return "SAFE"


def plan_for(hw: HardwareInfo, model_path: str | Path, context: int | None = None, force_mode: str = "auto") -> Plan:
    p = Path(model_path).expanduser()
    if not p.exists():
        raise FileNotFoundError(p)
    size = p.stat().st_size
    model_gib = size / GIB
    ram_gib = hw.ram_gib
    ctx = context or _default_context(ram_gib, model_gib, hw.chip)
    if ctx <= 0:
        raise ValueError("context must be greater than zero")

    os_reserve = max(3.5, ram_gib * 0.14)
    runtime_reserve = 1.25
    ctx_reserve = context_reserve_gib(ctx)
    resident_budget = max(0.0, ram_gib - os_reserve - runtime_reserve - ctx_reserve)
    headroom = resident_budget - model_gib
    resident_safe = headroom >= 0.0
    field_validated = _is_field_validated_m4pro_24gb(hw.chip, ram_gib, model_gib)

    if force_mode == "auto":
        mode = "metal-resident" if resident_safe else "ssd-cpu"
    elif force_mode in {"metal-resident", "ssd-cpu"}:
        mode = force_mode
    else:
        raise ValueError("mode must be auto, metal-resident, or ssd-cpu")

    if mode == "metal-resident":
        status = _resident_status(headroom)
        safe = resident_safe
        if not safe:
            reason = (
                f"Model is {model_gib:.2f} GiB but conservative resident budget is "
                f"{resident_budget:.2f} GiB. Use ssd-cpu or reduce model/context."
            )
        elif status == "CRITICAL" and field_validated:
            reason = (
                f"Model fits by only {headroom:.2f} GiB. This M4 Pro 24 GiB class has "
                "completed a 2K/4K/8K Metal sweep with zero observed swap delta; policy "
                "remains CRITICAL because estimated margin is still small."
            )
        elif status == "CRITICAL":
            reason = (
                f"Model fits by only {headroom:.2f} GiB. This profile is experimental "
                "and has little margin for other applications or larger context."
            )
        elif status == "TIGHT":
            reason = (
                f"Model fits with {headroom:.2f} GiB estimated headroom. Keep memory-heavy "
                "apps closed and watch macOS Memory Pressure."
            )
        else:
            reason = (
                f"Model fits conservative unified-memory budget with {headroom:.2f} GiB "
                "estimated headroom. Metal is preferred."
            )
    else:
        ratio = model_gib / max(ram_gib, 0.01)
        disk_floor = max(4.0, model_gib * 0.08)
        if hw.free_disk_gib < disk_floor:
            safe = False
            status = "UNSAFE"
            reason = (
                f"Free SSD is {hw.free_disk_gib:.1f} GiB; oversized execution wants at least "
                f"{disk_floor:.1f} GiB of additional free headroom."
            )
        else:
            safe = True
            status = "SSD"
            reason = (
                f"Model/RAM ratio is {ratio:.2f}x. Use bounded mmap expert residency on "
                "Apple-Silicon CPU/Accelerate to preserve model quality without OOM."
            )

    threads = max(1, hw.perf_cpus)
    if ram_gib <= 24:
        batch, ubatch = 256, 128
    else:
        batch, ubatch = 512, 256

    return Plan(
        mode=mode,
        status=status,
        context=ctx,
        threads=threads,
        batch=batch,
        ubatch=ubatch,
        model_gib=model_gib,
        ram_gib=ram_gib,
        os_reserve_gib=os_reserve,
        runtime_reserve_gib=runtime_reserve,
        context_reserve_gib=ctx_reserve,
        resident_budget_gib=resident_budget,
        headroom_gib=headroom,
        safe=safe,
        reason=reason,
    )


# Native runtime (redlite chat / redlite-generate) defaults. The full-residency cache holds
# every routed expert of the reference IQ2_XXS GGUF (the engine preloads them at open when
# the cache is >= 21 300 MiB, docs/REDLITE_DEV21_GPU_ROUTED_DECODE.md); it was validated on
# the M4 Max 48 GiB. Below 40 GiB the field-validated 4 GiB bounded cache is used.
NATIVE_FULL_RESIDENCY_MIN_RAM_GIB = 40.0
NATIVE_FULL_RESIDENCY_CACHE_MIB = 22528
NATIVE_BOUNDED_CACHE_MIB = 4096


@dataclass(frozen=True)
class NativeDefaults:
    cache_mib: int
    full_residency: bool
    reason: str


def native_defaults(ram_bytes: int) -> NativeDefaults:
    ram_gib = ram_bytes / GIB
    if ram_gib >= NATIVE_FULL_RESIDENCY_MIN_RAM_GIB:
        return NativeDefaults(
            NATIVE_FULL_RESIDENCY_CACHE_MIB, True,
            f"{ram_gib:.1f} GiB RAM >= {NATIVE_FULL_RESIDENCY_MIN_RAM_GIB:.0f} GiB: "
            "full expert residency, preloaded at open, GPU-routed decode",
        )
    return NativeDefaults(
        NATIVE_BOUNDED_CACHE_MIB, False,
        f"{ram_gib:.1f} GiB RAM < {NATIVE_FULL_RESIDENCY_MIN_RAM_GIB:.0f} GiB: bounded 4 GiB expert cache",
    )
