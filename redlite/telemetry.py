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
    unit = unit.upper().replace("IB", "").replace("B", "")
    if unit in {"G", ""}:
        return value if unit == "G" else value / (1024.0 ** 3)
    if unit == "T":
        return value * 1024.0
    if unit == "M":
        return value / 1024.0
    if unit == "K":
        return value / (1024.0 * 1024.0)
    return value / (1024.0 ** 3)


def _parse_swapusage(text: str | None) -> tuple[float | None, float | None]:
    if not text:
        return None, None

    # Common macOS forms include:
    #   total = 2048.00M  used = 0.00M  free = 2048.00M
    #   vm.swapusage: total = 2.00G used = 128.00M ...
    # Be deliberately permissive about whitespace and optional B/iB suffixes.
    number = r"([0-9]+(?:\.[0-9]+)?)"
    unit = r"([KMGTP]?(?:i?B)?)"
    tm = re.search(rf"\btotal\s*=\s*{number}\s*{unit}", text, re.I)
    um = re.search(rf"\bused\s*=\s*{number}\s*{unit}", text, re.I)

    def convert(match: re.Match[str] | None) -> float | None:
        if not match:
            return None
        value = float(match.group(1))
        raw_unit = match.group(2) or "B"
        return _to_gib(value, raw_unit)

    return convert(um), convert(tm)


def snapshot() -> MemorySnapshot:
    pressure = _run(["memory_pressure", "-Q"])
    free_percent = None
    if pressure:
        m = re.search(r"System-wide memory free percentage:\s*(\d+)%", pressure)
        if m:
            free_percent = int(m.group(1))

    swap = _run(["sysctl", "-n", "vm.swapusage"])
    if swap is None:
        # Some macOS builds/tools behave differently with -n; the prefixed form
        # is equivalent for our parser and gives us a second chance.
        swap = _run(["sysctl", "vm.swapusage"])
    swap_used, swap_total = _parse_swapusage(swap)

    return MemorySnapshot(
        free_percent=free_percent,
        swap_used_gib=swap_used,
        swap_total_gib=swap_total,
        raw_pressure=pressure,
    )
