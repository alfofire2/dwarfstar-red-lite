from __future__ import annotations

from dataclasses import dataclass
import os
import platform
import shutil
import subprocess
from pathlib import Path

GIB = 1024 ** 3


def _sysctl(key: str) -> str | None:
    try:
        return subprocess.check_output(["sysctl", "-n", key], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def gpu_wired_limit_mib() -> int:
    """dev51: `sysctl iogpu.wired_limit_mb` (the GPU working-set limit an admin can raise until reboot); 0 = macOS default."""
    raw = _sysctl("iogpu.wired_limit_mb") if platform.system() == "Darwin" else None
    return int(raw) if raw and raw.isdigit() else 0


@dataclass(frozen=True)
class HardwareInfo:
    system: str
    machine: str
    chip: str
    ram_bytes: int
    logical_cpus: int
    perf_cpus: int
    free_disk_bytes: int
    cwd: str

    @property
    def is_apple_silicon(self) -> bool:
        return self.system == "Darwin" and self.machine in {"arm64", "aarch64"}

    @property
    def ram_gib(self) -> float:
        return self.ram_bytes / GIB

    @property
    def free_disk_gib(self) -> float:
        return self.free_disk_bytes / GIB


def detect(path: str | os.PathLike[str] = ".") -> HardwareInfo:
    system = platform.system()
    machine = platform.machine().lower()
    ram = 0
    chip = platform.processor() or "unknown"

    if system == "Darwin":
        raw = _sysctl("hw.memsize")
        if raw and raw.isdigit():
            ram = int(raw)
        brand = _sysctl("machdep.cpu.brand_string")
        if brand:
            chip = brand
        # Apple Silicon often exposes the chip name here on newer macOS.
        hw_model = _sysctl("hw.model")
        if chip in {"arm", "arm64", "unknown", ""} and hw_model:
            chip = hw_model
    elif hasattr(os, "sysconf"):
        try:
            ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except Exception:
            ram = 0

    logical = os.cpu_count() or 1
    perf = logical
    if system == "Darwin":
        raw_perf = _sysctl("hw.perflevel0.physicalcpu")
        if raw_perf and raw_perf.isdigit():
            perf = max(1, int(raw_perf))

    target = Path(path).expanduser().resolve()
    if not target.exists():
        target = target.parent
    disk = shutil.disk_usage(target)

    return HardwareInfo(
        system=system,
        machine=machine,
        chip=chip,
        ram_bytes=ram,
        logical_cpus=logical,
        perf_cpus=perf,
        free_disk_bytes=disk.free,
        cwd=str(target),
    )
