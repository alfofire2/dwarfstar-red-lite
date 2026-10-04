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
NATIVE_BOUNDED_CACHE_MIB = 4096
# Metal's recommendedMaxWorkingSetSize is ~78% of RAM on Apple Silicon (37.44 GiB on the M4 Max 48 GiB):
# a full-residency cache plus the mapped dense weights must stay below this fraction of RAM.
# dev36: was 0.75. On the 48 GiB M4 Max the IQ3_M file (34944 MiB + 1.6 GiB dense = 74% of RAM) passed 0.75 but its
# GPU-routed decode fell from 46.8 to 6.0 tok/s on the second run and then failed with kIOGPUCommandBufferCallbackErrorOutOfMemory;
# IQ3_XXS (63% of RAM) is stable there
NATIVE_WORKING_SET_FRACTION = 0.70
NATIVE_SLOT_ALIGNMENT = 4096
# dev51/dev55: with a raised GPU limit (sysctl iogpu.wired_limit_mb > 0) full residency is allowed on any Mac whose limit
# holds what it needs. dev55 measured the need on the M4 Max (IQ2_XXS-size file, every expert resident; process footprint
# plus the mapped dense weights, which the GPU holds but the footprint does not count): it explains every M4 Pro outcome
# (limit 21,741: MTP + 32K context + 512-token chunks out of GPU memory at 22,302; no MTP + 32K + 2048 fine at 21,087;
# MTP + 4K + 2048 fine at 21,530).
NATIVE_GPU_BASE_MIB = 580                       # pool, scratch and buffers above slots + dense, at 512-token prefill chunks
NATIVE_KV_MIB_PER_POS = 48.0 / 1024.0           # 12 attention layers x K,V x 512 floats per position
NATIVE_BATCH_MIB_PER_TOKEN = 572.0 / 1536.0     # prefill scratch per token of chunk above 512 (2048: +572 MiB)
NATIVE_MTP_MIB = 1787                           # the resident MTP block (Q8_0 head file)
# dev31: native models in preference order (better quality first); `redlite chat` without a model path takes the
# first one present whose full residency fits, else the 24 GiB reference file
NATIVE_MODEL_PREFERENCE = (
    "Qwen_Qwen3-Next-80B-A3B-Instruct-IQ3_XXS.gguf",
    "Qwen3-Next-80B-A3B-Instruct-RedLite-E3.gguf",
    "Qwen_Qwen3-Next-80B-A3B-Instruct-IQ2_XXS.gguf",
)
NATIVE_REFERENCE_MODEL = "Qwen_Qwen3-Next-80B-A3B-Instruct-IQ2_XXS.gguf"
# dev54: the 24 GiB files in preference order: the Red Lite E3 mix (same size and types, better quality), then Bartowski's
NATIVE_SMALL_MODELS = ("Qwen3-Next-80B-A3B-Instruct-RedLite-F2.gguf", "Qwen3-Next-80B-A3B-Instruct-RedLite-E3.gguf",
                       NATIVE_REFERENCE_MODEL)   # dev58 F2, dev54 E3, then Bartowski's IQ2_XXS


@dataclass(frozen=True)
class NativeResidency:
    cache_mib: int          # expert cache that holds every routed expert of the file
    dense_bytes: int        # everything else in the file (mapped dense weights, embeddings, head)


def native_residency(model_path: str | Path) -> NativeResidency | None:
    """Full-residency cache of a GGUF from its real expert payload (dev31).

    Same formula as rl_engine_full_residency_mib() in the native engine: experts x the sum over layers of the layer's
    gate+up+down expert triplet rounded up to 4 KiB (dev37 slot size classes; before, every slot had the largest size). None when the file is not a readable Qwen3-Next MoE GGUF.
    """
    from .expert_map import build_expert_map

    try:
        emap = build_expert_map(model_path)
    except (OSError, ValueError):
        return None
    if not emap.all_slice_safe or not emap.expert_tensors:
        return None
    triplets: dict[int, int] = {}
    for t in emap.expert_tensors:
        triplets[t.layer] = triplets.get(t.layer, 0) + t.expert_stride_bytes
    # dev37: every layer's slots have that layer's own aligned size (native slot size classes)
    total = sum(-(-b // NATIVE_SLOT_ALIGNMENT) * NATIVE_SLOT_ALIGNMENT for b in triplets.values()) * emap.expert_count
    size = Path(model_path).stat().st_size
    return NativeResidency(-(-total // (1024 * 1024)), max(0, size - emap.total_routed_payload_bytes))


def native_full_residency_mib(residency: NativeResidency, context: int = 4096, batch: int = 2048, mtp: bool = False) -> int:
    """GPU memory full residency needs (dev55 model): the value to give iogpu.wired_limit_mb."""
    need = (residency.cache_mib + residency.dense_bytes / (1024 * 1024) + NATIVE_GPU_BASE_MIB + context * NATIVE_KV_MIB_PER_POS
            + max(0, batch - 512) * NATIVE_BATCH_MIB_PER_TOKEN + (NATIVE_MTP_MIB if mtp else 0))
    return int(-(-need // 1))


def native_full_residency_fits(ram_bytes: int, residency: NativeResidency, wired_mib: int = 0, context: int = 4096,
                               batch: int = 512, mtp: bool = False) -> bool:
    if wired_mib > 0:
        return native_full_residency_mib(residency, context, batch, mtp) <= wired_mib
    extra = NATIVE_MTP_MIB * 1024 * 1024 if mtp else 0
    return residency.cache_mib * 1024 * 1024 + residency.dense_bytes + extra <= ram_bytes * NATIVE_WORKING_SET_FRACTION


NATIVE_SLOT_STATE_POSITIONS = 1536   # dev56: a second slot's DeltaNet state (72 MiB) in 48 KiB KV positions


def native_plan_context(context: int, parallel: int = 1) -> int:
    """dev56: positions the GPU plan sizes for when `parallel` slots each hold a `context`-position state."""
    return context * parallel + NATIVE_SLOT_STATE_POSITIONS * (parallel - 1)


def native_gpu_plan(residency: NativeResidency, wired_mib: int, context: int, mtp_available: bool) -> tuple[int, bool] | None:
    """dev55: under a raised GPU limit, the (prefill chunk, MTP) that fits, preferring MTP, then 2048-token chunks."""
    for batch, mtp in ((2048, True), (512, True), (2048, False), (512, False)):
        if (mtp_available or not mtp) and native_full_residency_mib(residency, context, batch, mtp) <= wired_mib:
            return batch, mtp
    return None


@dataclass(frozen=True)
class NativeDefaults:
    cache_mib: int
    full_residency: bool
    reason: str


def native_defaults(ram_bytes: int, model_path: str | Path | None = None, wired_mib: int = 0, context: int = 4096) -> NativeDefaults:
    ram_gib = ram_bytes / GIB
    if (ram_gib >= NATIVE_FULL_RESIDENCY_MIN_RAM_GIB or wired_mib > 0) and model_path is not None:
        res = native_residency(model_path)
        budget = (f"GPU limit {wired_mib} MiB (iogpu.wired_limit_mb)" if wired_mib > 0
                  else f"{ram_gib:.1f} GiB RAM >= {NATIVE_FULL_RESIDENCY_MIN_RAM_GIB:.0f} GiB")
        if res is None:
            reason = f"{budget}, but the expert payload of {Path(model_path).name} could not be read"
        elif native_full_residency_fits(ram_bytes, res, wired_mib, context):
            return NativeDefaults(
                res.cache_mib, True,
                f"{budget}: full expert residency ({res.cache_mib} MiB from the file's expert payload), "
                f"preloaded at open, GPU-routed decode",
            )
        elif wired_mib > 0:
            reason = (f"{budget}: full residency of {Path(model_path).name} at context {context} needs "
                      f"{native_full_residency_mib(res, context, 512)} MiB")
        else:
            reason = (f"{ram_gib:.1f} GiB RAM: full residency of {Path(model_path).name} needs {res.cache_mib} MiB + "
                      f"{res.dense_bytes / GIB:.1f} GiB dense, above {NATIVE_WORKING_SET_FRACTION:.0%} of RAM")
        return NativeDefaults(NATIVE_BOUNDED_CACHE_MIB, False, reason + "; bounded 4 GiB expert cache")
    if ram_gib >= NATIVE_FULL_RESIDENCY_MIN_RAM_GIB:
        return NativeDefaults(NATIVE_BOUNDED_CACHE_MIB, False, f"{ram_gib:.1f} GiB RAM, no model to size: bounded 4 GiB expert cache")
    return NativeDefaults(
        NATIVE_BOUNDED_CACHE_MIB, False,
        f"{ram_gib:.1f} GiB RAM < {NATIVE_FULL_RESIDENCY_MIN_RAM_GIB:.0f} GiB: bounded 4 GiB expert cache",
    )


def select_native_model(models_dir: str | Path, ram_bytes: int, wired_mib: int = 0) -> Path | None:
    """dev31: the best native model present in models_dir for this machine.

    With >= 40 GiB of RAM the first file of NATIVE_MODEL_PREFERENCE whose full residency fits is chosen; otherwise
    (or when none fits) the 24 GiB reference file, else the first known file present.
    """
    d = Path(models_dir)
    present = [d / name for name in NATIVE_MODEL_PREFERENCE if (d / name).is_file()]
    if not present:
        return None
    if ram_bytes / GIB >= NATIVE_FULL_RESIDENCY_MIN_RAM_GIB or wired_mib > 0:
        for p in present:
            res = native_residency(p)
            if res is not None and native_full_residency_fits(ram_bytes, res, wired_mib):
                return p
    for name in NATIVE_SMALL_MODELS:
        if (d / name).is_file():
            return d / name
    return present[-1]


NATIVE_MTP_FILE = "Qwen3-Next-80B-A3B-Instruct-MTP-ONLY-Q8_0.gguf"
# dev55: below 40 GiB (M4 Pro: half the M4 Max's bandwidth) MTP stops paying at long context: +24 % at 20 prompt tokens,
# +4 % at 5.6K, 0 % at 11.2K, -12 % at 16.8K. The M4 Max still gains at 16.8K (+2 %), so it has no limit.
NATIVE_MTP_MAX_CONTEXT_SMALL = 8192


def native_mtp_max_context(ram_bytes: int) -> int | None:
    return NATIVE_MTP_MAX_CONTEXT_SMALL if ram_bytes / GIB < NATIVE_FULL_RESIDENCY_MIN_RAM_GIB else None


def native_mtp_file(model_path: str | Path, cache_mib: int | str, disabled: bool = False,
                    ram_bytes: int | None = None, wired_mib: int = 0) -> Path | None:
    """dev45: the MTP head to pass as --mtp, or None.

    Used only for the Qwen3-Next-80B-A3B-Instruct files it was trained with, when the head file sits next to the
    model and the cache holds every expert (the 2-row verify needs full residency). Its output equals plain
    decoding, so there is no quality trade-off to opt into.
    """
    if disabled:
        return None
    model = Path(model_path)
    head = model.parent / NATIVE_MTP_FILE
    if "Qwen3-Next-80B-A3B-Instruct" not in model.name or not head.is_file():
        return None
    res = native_residency(model)
    if res is None:
        return None
    full = str(cache_mib) == "full" or (str(cache_mib).isdigit() and int(cache_mib) >= res.cache_mib)
    if not full:
        return None
    # dev51: the head (2.26 GiB) must fit the GPU budget next to the resident experts (24 GiB Macs with a raised limit)
    if ram_bytes is not None and not native_full_residency_fits(ram_bytes, res, wired_mib, mtp=True):
        return None
    return head
