from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Sequence

from .redmetal import RedMetalError, redmetal_library_path


class RedMetalQuantBridge:
    def __init__(self) -> None:
        path = redmetal_library_path()
        if not path.is_file():
            raise RedMetalError(f"Red Metal native library not found at {path}. Run make redmetal first.")
        self.lib = ctypes.CDLL(str(path))
        self._configure()
        abi = int(self.lib.redmetal_quant_abi_version())
        if abi != 1:
            raise RedMetalError(f"unsupported Red Metal quant ABI {abi}")

    def _configure(self) -> None:
        lib = self.lib
        lib.redmetal_quant_abi_version.argtypes = []
        lib.redmetal_quant_abi_version.restype = ctypes.c_uint32
        lib.redmetal_quant_last_error.argtypes = []
        lib.redmetal_quant_last_error.restype = ctypes.c_char_p
        lib.redmetal_quant_rows.argtypes = [
            ctypes.c_char_p,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_int8),
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]
        lib.redmetal_quant_rows.restype = ctypes.c_int

    def _last_error(self) -> str:
        raw = self.lib.redmetal_quant_last_error()
        return raw.decode("utf-8", errors="replace") if raw else "unknown Red Metal quant error"

    def rows(
        self,
        model: str | Path,
        matrix_file_offset: int,
        matrix_bytes: int,
        ggml_type: int,
        ncols: int,
        nrows: int,
        row_start: int,
        row_count: int,
        input_values: Sequence[float],
        grid: Sequence[int],
    ) -> tuple[list[float], float, float]:
        if len(input_values) != ncols:
            raise ValueError(f"input has {len(input_values)} values, expected {ncols}")
        inp = (ctypes.c_float * ncols)(*map(float, input_values))
        grid_buf = (ctypes.c_int8 * len(grid))(*map(int, grid))
        out = (ctypes.c_float * row_count)()
        io_ms = ctypes.c_double()
        gpu_ms = ctypes.c_double()
        ok = self.lib.redmetal_quant_rows(
            str(Path(model).expanduser().resolve()).encode(),
            int(matrix_file_offset),
            int(matrix_bytes),
            int(ggml_type),
            int(ncols),
            int(nrows),
            int(row_start),
            int(row_count),
            inp,
            int(ncols),
            grid_buf,
            int(len(grid)),
            out,
            int(row_count),
            ctypes.byref(io_ms),
            ctypes.byref(gpu_ms),
        )
        if not ok:
            raise RedMetalError(self._last_error())
        return [float(value) for value in out], float(io_ms.value), float(gpu_ms.value)
