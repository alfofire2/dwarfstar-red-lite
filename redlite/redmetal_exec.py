from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Any, Sequence

from .expert_store import ExpertLayout
from .redmetal import RedMetalError, redmetal_library_path


class RedMetalExecPool:
    """ctypes wrapper for the dev5 in-flight-safe Metal execution pool."""

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
        path = redmetal_library_path()
        if not path.is_file():
            raise RedMetalError(f"Red Metal native library not found at {path}. Run make redmetal first.")
        self.lib = ctypes.CDLL(str(path))
        self._configure()
        if self.lib.redmetal_exec_abi_version() != 1:
            raise RedMetalError(f"unsupported Red Metal execution ABI {self.lib.redmetal_exec_abi_version()}")
        self.model = str(Path(model).expanduser().resolve())
        self.handle = self.lib.redmetal_exec_pool_create(
            os.fsencode(self.model),
            int(budget_bytes),
            int(slot_bytes),
            int(slots_per_slab),
        )
        if not self.handle:
            raise RedMetalError(self._last_error())

    def _configure(self) -> None:
        lib = self.lib
        lib.redmetal_exec_abi_version.argtypes = []
        lib.redmetal_exec_abi_version.restype = ctypes.c_uint32
        lib.redmetal_exec_last_error.argtypes = []
        lib.redmetal_exec_last_error.restype = ctypes.c_char_p
        lib.redmetal_exec_pool_create.argtypes = [
            ctypes.c_char_p,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint32,
        ]
        lib.redmetal_exec_pool_create.restype = ctypes.c_void_p
        lib.redmetal_exec_pool_destroy.argtypes = [ctypes.c_void_p]
        lib.redmetal_exec_pool_destroy.restype = None
        for name, restype in (
            ("redmetal_exec_pool_capacity", ctypes.c_uint32),
            ("redmetal_exec_pool_slab_count", ctypes.c_uint32),
            ("redmetal_exec_pool_allocated_bytes", ctypes.c_uint64),
            ("redmetal_exec_pool_bytes_read", ctypes.c_uint64),
            ("redmetal_exec_pool_read_calls", ctypes.c_uint64),
            ("redmetal_exec_pool_read_ms", ctypes.c_double),
        ):
            fn = getattr(lib, name)
            fn.argtypes = [ctypes.c_void_p]
            fn.restype = restype
        lib.redmetal_exec_pool_load_expert.argtypes = [
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
        lib.redmetal_exec_pool_load_expert.restype = ctypes.c_int
        lib.redmetal_exec_pool_bind_expert.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
        ]
        lib.redmetal_exec_pool_bind_expert.restype = ctypes.c_int
        lib.redmetal_exec_pool_unbind_expert.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
        ]
        lib.redmetal_exec_pool_unbind_expert.restype = ctypes.c_int
        lib.redmetal_exec_pool_bound_addresses.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
        ]
        lib.redmetal_exec_pool_bound_addresses.restype = ctypes.c_int
        lib.redmetal_exec_pool_slot_inflight.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        lib.redmetal_exec_pool_slot_inflight.restype = ctypes.c_int
        lib.redmetal_exec_pool_iq2_xxs_rows.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint64,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_double),
        ]
        lib.redmetal_exec_pool_iq2_xxs_rows.restype = ctypes.c_int

    def _last_error(self) -> str:
        raw = self.lib.redmetal_exec_last_error()
        return raw.decode("utf-8", errors="replace") if raw else "unknown Red Metal execution error"

    def close(self) -> None:
        if self.handle:
            self.lib.redmetal_exec_pool_destroy(self.handle)
            self.handle = None

    def __enter__(self) -> "RedMetalExecPool":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def capacity(self) -> int:
        return int(self.lib.redmetal_exec_pool_capacity(self.handle))

    def load(self, slot_id: int, layout: ExpertLayout) -> dict[str, Any]:
        parts = {part.kind: part for part in layout.parts}
        gate, up, down = parts["gate"], parts["up"], parts["down"]
        read = ctypes.c_uint64()
        ms = ctypes.c_double()
        ok = self.lib.redmetal_exec_pool_load_expert(
            self.handle,
            slot_id,
            gate.file_offset,
            gate.length,
            up.file_offset,
            up.length,
            down.file_offset,
            down.length,
            ctypes.byref(read),
            ctypes.byref(ms),
        )
        if not ok:
            raise RedMetalError(self._last_error())
        return {"bytes_read": int(read.value), "elapsed_ms": float(ms.value)}

    def bind(self, layer: int, expert: int, slot_id: int, layout: ExpertLayout) -> None:
        parts = {part.kind: part for part in layout.parts}
        ok = self.lib.redmetal_exec_pool_bind_expert(
            self.handle,
            layer,
            expert,
            slot_id,
            parts["gate"].length,
            parts["up"].length,
            parts["down"].length,
        )
        if not ok:
            raise RedMetalError(self._last_error())

    def unbind(self, layer: int, expert: int) -> None:
        if not self.lib.redmetal_exec_pool_unbind_expert(self.handle, layer, expert):
            raise RedMetalError(self._last_error())

    def addresses(self, layer: int, expert: int) -> tuple[int, int, int] | None:
        gate = ctypes.c_uint64()
        up = ctypes.c_uint64()
        down = ctypes.c_uint64()
        ok = self.lib.redmetal_exec_pool_bound_addresses(
            self.handle,
            layer,
            expert,
            ctypes.byref(gate),
            ctypes.byref(up),
            ctypes.byref(down),
        )
        if not ok:
            return None
        return int(gate.value), int(up.value), int(down.value)

    def slot_inflight(self, slot_id: int) -> bool:
        return bool(self.lib.redmetal_exec_pool_slot_inflight(self.handle, slot_id))

    def iq2_rows(
        self,
        slot_id: int,
        matrix_slot_offset: int,
        ncols: int,
        nrows: int,
        row_start: int,
        row_count: int,
        input_values: Sequence[float],
    ) -> tuple[list[float], float]:
        if len(input_values) != ncols:
            raise ValueError(f"input has {len(input_values)} values, expected {ncols}")
        inp = (ctypes.c_float * ncols)(*map(float, input_values))
        out = (ctypes.c_float * row_count)()
        ms = ctypes.c_double()
        ok = self.lib.redmetal_exec_pool_iq2_xxs_rows(
            self.handle,
            slot_id,
            matrix_slot_offset,
            ncols,
            nrows,
            row_start,
            row_count,
            inp,
            ncols,
            out,
            row_count,
            ctypes.byref(ms),
        )
        if not ok:
            raise RedMetalError(self._last_error())
        return [float(x) for x in out], float(ms.value)

    def telemetry(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "slab_count": int(self.lib.redmetal_exec_pool_slab_count(self.handle)),
            "allocated_bytes": int(self.lib.redmetal_exec_pool_allocated_bytes(self.handle)),
            "bytes_read": int(self.lib.redmetal_exec_pool_bytes_read(self.handle)),
            "read_calls": int(self.lib.redmetal_exec_pool_read_calls(self.handle)),
            "read_ms": float(self.lib.redmetal_exec_pool_read_ms(self.handle)),
        }
