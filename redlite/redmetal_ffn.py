from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Any, Sequence

from .expert_store import ExpertLayout
from .quant_tables import load_quant_grid
from .redmetal import RedMetalError, redmetal_library_path


class RedMetalFfnPool:
    """ctypes wrapper for the resident single-expert FFN Metal pool."""

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
        if self.lib.redmetal_ffn_abi_version() != 1:
            raise RedMetalError(f"unsupported Red Metal FFN ABI {self.lib.redmetal_ffn_abi_version()}")

        iq2 = load_quant_grid(17)
        iq1 = load_quant_grid(29)
        self._iq2 = (ctypes.c_int8 * len(iq2))(*iq2)
        self._iq1 = (ctypes.c_int8 * len(iq1))(*iq1)
        self.model = str(Path(model).expanduser().resolve())
        self.handle = self.lib.redmetal_ffn_pool_create(
            os.fsencode(self.model),
            int(budget_bytes),
            int(slot_bytes),
            int(slots_per_slab),
            self._iq2,
            len(iq2),
            self._iq1,
            len(iq1),
        )
        if not self.handle:
            raise RedMetalError(self._last_error())

    def _configure(self) -> None:
        lib = self.lib
        lib.redmetal_ffn_abi_version.argtypes = []
        lib.redmetal_ffn_abi_version.restype = ctypes.c_uint32
        lib.redmetal_ffn_last_error.argtypes = []
        lib.redmetal_ffn_last_error.restype = ctypes.c_char_p
        lib.redmetal_ffn_pool_create.argtypes = [
            ctypes.c_char_p,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_int8),
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_int8),
            ctypes.c_uint32,
        ]
        lib.redmetal_ffn_pool_create.restype = ctypes.c_void_p
        lib.redmetal_ffn_pool_destroy.argtypes = [ctypes.c_void_p]
        lib.redmetal_ffn_pool_destroy.restype = None
        for name, restype in (
            ("redmetal_ffn_pool_capacity", ctypes.c_uint32),
            ("redmetal_ffn_pool_slab_count", ctypes.c_uint32),
            ("redmetal_ffn_pool_allocated_bytes", ctypes.c_uint64),
            ("redmetal_ffn_pool_bytes_read", ctypes.c_uint64),
            ("redmetal_ffn_pool_read_calls", ctypes.c_uint64),
            ("redmetal_ffn_pool_read_ms", ctypes.c_double),
        ):
            fn = getattr(lib, name)
            fn.argtypes = [ctypes.c_void_p]
            fn.restype = restype
        lib.redmetal_ffn_pool_slot_inflight.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        lib.redmetal_ffn_pool_slot_inflight.restype = ctypes.c_int
        lib.redmetal_ffn_pool_load_expert.argtypes = [
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
        lib.redmetal_ffn_pool_load_expert.restype = ctypes.c_int
        lib.redmetal_ffn_pool_slot_addresses.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
        ]
        lib.redmetal_ffn_pool_slot_addresses.restype = ctypes.c_int
        lib.redmetal_ffn_pool_execute_expert.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint64,
            ctypes.c_uint64,
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
        lib.redmetal_ffn_pool_execute_expert.restype = ctypes.c_int

    def _last_error(self) -> str:
        raw = self.lib.redmetal_ffn_last_error()
        return raw.decode("utf-8", errors="replace") if raw else "unknown Red Metal FFN error"

    def close(self) -> None:
        if self.handle:
            self.lib.redmetal_ffn_pool_destroy(self.handle)
            self.handle = None

    def __enter__(self) -> "RedMetalFfnPool":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def capacity(self) -> int:
        return int(self.lib.redmetal_ffn_pool_capacity(self.handle))

    def slot_inflight(self, slot_id: int) -> bool:
        return bool(self.lib.redmetal_ffn_pool_slot_inflight(self.handle, slot_id))

    def load(self, slot_id: int, layout: ExpertLayout) -> dict[str, Any]:
        parts = {part.kind: part for part in layout.parts}
        gate, up, down = parts["gate"], parts["up"], parts["down"]
        read = ctypes.c_uint64()
        ms = ctypes.c_double()
        ok = self.lib.redmetal_ffn_pool_load_expert(
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

    def addresses(self, slot_id: int, layout: ExpertLayout) -> tuple[int, int, int]:
        parts = {part.kind: part for part in layout.parts}
        gate = ctypes.c_uint64()
        up = ctypes.c_uint64()
        down = ctypes.c_uint64()
        ok = self.lib.redmetal_ffn_pool_slot_addresses(
            self.handle,
            slot_id,
            parts["gate"].length,
            parts["up"].length,
            parts["down"].length,
            ctypes.byref(gate),
            ctypes.byref(up),
            ctypes.byref(down),
        )
        if not ok:
            raise RedMetalError(self._last_error())
        return int(gate.value), int(up.value), int(down.value)

    def execute(
        self,
        slot_id: int,
        layout: ExpertLayout,
        ggml_type: int,
        hidden_size: int,
        ffn_size: int,
        output_row_start: int,
        output_row_count: int,
        input_values: Sequence[float],
    ) -> tuple[list[float], float]:
        if len(input_values) != hidden_size:
            raise ValueError(f"input has {len(input_values)} values, expected {hidden_size}")
        parts = {part.kind: part for part in layout.parts}
        inp = (ctypes.c_float * hidden_size)(*map(float, input_values))
        out = (ctypes.c_float * output_row_count)()
        ms = ctypes.c_double()
        ok = self.lib.redmetal_ffn_pool_execute_expert(
            self.handle,
            slot_id,
            int(ggml_type),
            parts["gate"].length,
            parts["up"].length,
            parts["down"].length,
            hidden_size,
            ffn_size,
            output_row_start,
            output_row_count,
            inp,
            hidden_size,
            out,
            output_row_count,
            ctypes.byref(ms),
        )
        if not ok:
            raise RedMetalError(self._last_error())
        return [float(x) for x in out], float(ms.value)

    def telemetry(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "slab_count": int(self.lib.redmetal_ffn_pool_slab_count(self.handle)),
            "allocated_bytes": int(self.lib.redmetal_ffn_pool_allocated_bytes(self.handle)),
            "bytes_read": int(self.lib.redmetal_ffn_pool_bytes_read(self.handle)),
            "read_calls": int(self.lib.redmetal_ffn_pool_read_calls(self.handle)),
            "read_ms": float(self.lib.redmetal_ffn_pool_read_ms(self.handle)),
        }
