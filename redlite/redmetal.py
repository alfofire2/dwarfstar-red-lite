from __future__ import annotations

import ctypes
import os
from pathlib import Path
import sys
from typing import Any

from .expert_store import ExpertLayout


class RedMetalError(RuntimeError):
    pass


def _default_library_path() -> Path:
    override = os.environ.get("REDLITE_REDMETAL_LIB")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[1] / ".deps" / "redmetal" / "libredmetal.dylib"


def redmetal_library_path() -> Path:
    return _default_library_path()


def redmetal_built() -> bool:
    return sys.platform == "darwin" and _default_library_path().is_file()


def _load_library() -> ctypes.CDLL:
    if sys.platform != "darwin":
        raise RedMetalError("Red Metal requires macOS on Apple Silicon")
    path = _default_library_path()
    if not path.is_file():
        raise RedMetalError(
            f"Red Metal native library not found at {path}. Run make redmetal first."
        )

    lib = ctypes.CDLL(str(path))
    lib.redmetal_abi_version.argtypes = []
    lib.redmetal_abi_version.restype = ctypes.c_uint32
    lib.redmetal_last_error.argtypes = []
    lib.redmetal_last_error.restype = ctypes.c_char_p
    lib.redmetal_system_device_name.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
    lib.redmetal_system_device_name.restype = ctypes.c_int

    lib.redmetal_pool_create.argtypes = [
        ctypes.c_char_p,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint32,
    ]
    lib.redmetal_pool_create.restype = ctypes.c_void_p
    lib.redmetal_pool_destroy.argtypes = [ctypes.c_void_p]
    lib.redmetal_pool_destroy.restype = None

    for name, restype in (
        ("redmetal_pool_capacity", ctypes.c_uint32),
        ("redmetal_pool_slab_count", ctypes.c_uint32),
        ("redmetal_pool_allocated_bytes", ctypes.c_uint64),
        ("redmetal_pool_bytes_read", ctypes.c_uint64),
        ("redmetal_pool_read_calls", ctypes.c_uint64),
        ("redmetal_pool_read_ms", ctypes.c_double),
    ):
        fn = getattr(lib, name)
        fn.argtypes = [ctypes.c_void_p]
        fn.restype = restype

    lib.redmetal_pool_load_expert.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.redmetal_pool_load_expert.restype = ctypes.c_int

    lib.redmetal_pool_gpu_probe.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.redmetal_pool_gpu_probe.restype = ctypes.c_int

    abi = lib.redmetal_abi_version()
    if abi != 1:
        raise RedMetalError(f"unsupported Red Metal ABI {abi}")
    return lib


def _last_error(lib: ctypes.CDLL) -> str:
    value = lib.redmetal_last_error()
    return value.decode("utf-8", errors="replace") if value else "unknown Red Metal error"


def redmetal_device_name() -> str:
    lib = _load_library()
    buf = ctypes.create_string_buffer(256)
    if not lib.redmetal_system_device_name(buf, len(buf)):
        raise RedMetalError(_last_error(lib))
    return buf.value.decode("utf-8", errors="replace")


class RedMetalPool:
    def __init__(
        self,
        model: str | Path,
        budget_bytes: int,
        slot_bytes: int,
        *,
        slots_per_slab: int = 64,
    ) -> None:
        if budget_bytes <= 0 or slot_bytes <= 0:
            raise ValueError("budget_bytes and slot_bytes must be positive")
        if slots_per_slab <= 0:
            raise ValueError("slots_per_slab must be positive")

        # DS4 rounds streaming slab slots to a VM page. Red Lite does the same:
        # it keeps every MTLBuffer binding offset naturally page-aligned while
        # preserving a deterministic hard budget.
        page = 4096
        aligned_slot_bytes = ((int(slot_bytes) + page - 1) // page) * page

        self.model = str(Path(model).expanduser().resolve())
        self.lib = _load_library()
        self.handle = self.lib.redmetal_pool_create(
            os.fsencode(self.model),
            int(budget_bytes),
            aligned_slot_bytes,
            int(slots_per_slab),
        )
        if not self.handle:
            raise RedMetalError(_last_error(self.lib))
        self.payload_slot_bytes = int(slot_bytes)
        self.slot_bytes = aligned_slot_bytes

    def close(self) -> None:
        if self.handle:
            self.lib.redmetal_pool_destroy(self.handle)
            self.handle = None

    def __enter__(self) -> "RedMetalPool":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def capacity(self) -> int:
        return int(self.lib.redmetal_pool_capacity(self.handle))

    def load(self, slot_id: int, layout: ExpertLayout) -> dict[str, Any]:
        parts = {part.kind: part for part in layout.parts}
        missing = [kind for kind in ("gate", "up", "down") if kind not in parts]
        if missing:
            raise ValueError(f"expert layout is missing {', '.join(missing)}")
        gate, up, down = parts["gate"], parts["up"], parts["down"]
        bytes_read = ctypes.c_uint64()
        elapsed_ms = ctypes.c_double()
        ok = self.lib.redmetal_pool_load_expert(
            self.handle,
            int(slot_id),
            gate.file_offset,
            gate.length,
            up.file_offset,
            up.length,
            down.file_offset,
            down.length,
            ctypes.byref(bytes_read),
            ctypes.byref(elapsed_ms),
        )
        if not ok:
            raise RedMetalError(_last_error(self.lib))
        return {"bytes_read": int(bytes_read.value), "elapsed_ms": float(elapsed_ms.value)}

    def gpu_probe(self, slot_id: int, payload_bytes: int) -> dict[str, Any]:
        checksum = ctypes.c_uint32()
        elapsed_ms = ctypes.c_double()
        ok = self.lib.redmetal_pool_gpu_probe(
            self.handle,
            int(slot_id),
            int(payload_bytes),
            ctypes.byref(checksum),
            ctypes.byref(elapsed_ms),
        )
        if not ok:
            raise RedMetalError(_last_error(self.lib))
        return {"checksum": int(checksum.value), "elapsed_ms": float(elapsed_ms.value)}

    def telemetry(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "payload_slot_bytes": self.payload_slot_bytes,
            "slot_stride_bytes": self.slot_bytes,
            "slab_count": int(self.lib.redmetal_pool_slab_count(self.handle)),
            "allocated_bytes": int(self.lib.redmetal_pool_allocated_bytes(self.handle)),
            "bytes_read": int(self.lib.redmetal_pool_bytes_read(self.handle)),
            "read_calls": int(self.lib.redmetal_pool_read_calls(self.handle)),
            "read_ms": float(self.lib.redmetal_pool_read_ms(self.handle)),
        }
