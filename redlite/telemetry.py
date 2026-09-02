from __future__ import annotations

from dataclasses import dataclass, asdict
import re
import subprocess
from typing import Any


@dataclass(frozen=True)
class MemorySnapshot:
    free_percent: int | None
    swap_used_gib: float | None
    swap_total_gib: float | None
    raw_pressure: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _run(cmd: list[str]) -> str | None:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception:
        return None


def _to_gib(value: float, unit: str) -> float:
    unit = unit.upper()
    if unit == "G":
        return value
    if unit == "M":
        return value / 1024.0
    if unit == "K":
        return value / (1024.0 * 1024.0)
    return value / (1024.0 ** 3)


def snapshot() -> MemorySnapshot:
    pressure = _run(["memory_pressure", "-Q"])
    free_percent = None
    if pressure:
        m = re.search(r"System-wide memory free percentage:\s*(\d+)%", pressure)
        if m:
            free_percent = int(m.group(1))

    swap = _run(["sysctl", "-n", "vm.swapusage"])
    swap_used = swap_total = None
    if swap:
        tm = re.search(r"total\s*=\s*([0-9.]+)([KMGT])", swap, re.I)
        um = re.search(r"used\s*=\s*([0-9.]+)([KMGT])", swap, re.I)
        if tm:
            swap_total = _to_gib(float(tm.group(1)), tm.group(2))
        if um:
            swap_used = _to_gib(float(um.group(1)), um.group(2))

    return MemorySnapshot(
        free_percent=free_percent,
        swap_used_gib=swap_used,
        swap_total_gib=swap_total,
        raw_pressure=pressure,
    )
