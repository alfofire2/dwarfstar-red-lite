from __future__ import annotations

import os
from pathlib import Path
import struct
from typing import Sequence

from .quant_tables import load_quant_grid, quant_type_name

QK_IQ = 256
QUANT_BLOCK_BYTES = {
    17: 74,  # IQ2_XS: half d + 32 uint16 qs + 8 nibble-scale bytes
    29: 56,  # IQ1_M: 32 qs + 16 qh + 8 packed scale bytes
}


def deterministic_input(ncols: int) -> list[float]:
    return [((i * 17 + 3) % 61 - 30) / 32.0 for i in range(ncols)]


def block_bytes_for_type(ggml_type: int) -> int:
    try:
        return QUANT_BLOCK_BYTES[int(ggml_type)]
    except KeyError as exc:
        raise ValueError(f"unsupported routed GGML type {ggml_type}") from exc


def row_bytes_for_type(ggml_type: int, ncols: int) -> int:
    if ncols <= 0 or ncols % QK_IQ:
        raise ValueError(f"ncols={ncols} must be a positive multiple of {QK_IQ}")
    return (ncols // QK_IQ) * block_bytes_for_type(ggml_type)


def _sign_byte(sign7: int) -> int:
    sign7 &= 0x7F
    return sign7 | ((sign7.bit_count() & 1) << 7)


def _iq2_xs_row_dot(row: bytes, input_values: Sequence[float], grid: Sequence[int]) -> float:
    ncols = len(input_values)
    expected = row_bytes_for_type(17, ncols)
    if len(row) != expected:
        raise ValueError(f"IQ2_XS row has {len(row)} bytes, expected {expected}")
    if len(grid) != 512 * 8:
        raise ValueError(f"IQ2_XS grid has {len(grid)} values, expected 4096")

    acc = 0.0
    for ib in range(ncols // QK_IQ):
        block = row[ib * 74 : (ib + 1) * 74]
        d = float(struct.unpack_from("<e", block, 0)[0])
        qs = struct.unpack_from("<32H", block, 2)
        scales = block[66:74]
        for group in range(16):
            scale = (scales[group // 2] >> (4 * (group & 1))) & 0x0F
            db = d * (0.5 + float(scale)) * 0.25
            for part in range(2):
                q = qs[2 * group + part]
                grid_index = q & 0x01FF
                signs = _sign_byte(q >> 9)
                base = ib * QK_IQ + group * 16 + part * 8
                gbase = grid_index * 8
                for j in range(8):
                    sign = -1.0 if signs & (1 << j) else 1.0
                    acc += float(input_values[base + j]) * (db * float(grid[gbase + j]) * sign)
    return acc


def _iq1_m_row_dot(row: bytes, input_values: Sequence[float], grid: Sequence[int]) -> float:
    ncols = len(input_values)
    expected = row_bytes_for_type(29, ncols)
    if len(row) != expected:
        raise ValueError(f"IQ1_M row has {len(row)} bytes, expected {expected}")
    if len(grid) != 2048 * 8:
        raise ValueError(f"IQ1_M grid has {len(grid)} values, expected 16384")

    acc = 0.0
    for ib in range(ncols // QK_IQ):
        block = row[ib * 56 : (ib + 1) * 56]
        qs = block[0:32]
        qh = block[32:48]
        sc = struct.unpack_from("<4H", block, 48)
        d_bits = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000)
        d = float(struct.unpack("<e", struct.pack("<H", d_bits))[0])

        for group in range(32):
            nibble = (qh[group // 2] >> (4 * (group & 1))) & 0x0F
            grid_index = int(qs[group]) | ((nibble & 0x07) << 8)
            delta = -0.125 if nibble & 0x08 else 0.125
            scale_index = group // 2
            scale = (sc[scale_index // 4] >> (3 * (scale_index & 3))) & 0x07
            dl = d * float(2 * scale + 1)
            base = ib * QK_IQ + group * 8
            gbase = grid_index * 8
            for j in range(8):
                weight = dl * (float(grid[gbase + j]) + delta)
                acc += float(input_values[base + j]) * weight
    return acc


def quant_row_dot(
    row: bytes,
    ggml_type: int,
    input_values: Sequence[float],
    grid: Sequence[int] | None = None,
) -> float:
    values = load_quant_grid(ggml_type) if grid is None else grid
    if ggml_type == 17:
        return _iq2_xs_row_dot(row, input_values, values)
    if ggml_type == 29:
        return _iq1_m_row_dot(row, input_values, values)
    raise ValueError(f"unsupported routed quant type {quant_type_name(ggml_type)}")


def reference_rows(
    model: str | Path,
    matrix_file_offset: int,
    ggml_type: int,
    ncols: int,
    row_start: int,
    row_count: int,
    input_values: Sequence[float],
    grid: Sequence[int] | None = None,
) -> list[float]:
    if len(input_values) != ncols:
        raise ValueError("reference input size mismatch")
    row_bytes = row_bytes_for_type(ggml_type, ncols)
    values = load_quant_grid(ggml_type) if grid is None else grid
    fd = os.open(str(Path(model).expanduser().resolve()), os.O_RDONLY)
    try:
        result = []
        for row in range(row_start, row_start + row_count):
            data = os.pread(fd, row_bytes, matrix_file_offset + row * row_bytes)
            if len(data) != row_bytes:
                raise EOFError(f"short {quant_type_name(ggml_type)} reference read for row {row}")
            result.append(quant_row_dot(data, ggml_type, input_values, values))
        return result
    finally:
        os.close(fd)
