from __future__ import annotations

import os
from pathlib import Path
import struct
from typing import Sequence


_GRID_HEX = (
    b"00000200050008000a00110014002000220028002a0041004400500058006100"
    b"6400800082008a00a20001010401100115014001840198010002020222028202"
    b"010404041004210424044004420448046004810484049004a404000502050805"
    b"200546056905800591050906100640068406a406000805080808140828084108"
    b"440850085208880804094009020a140a01100410101021104010601084109010"
    b"951000110811201150115a118011241245120014081420142514491480141815"
    b"6215001616160118041810184018811800190519a019511a002002200a204420"
    b"6120802082202921482100220222012404241024402456240025412564259026"
    b"082820289428442a014004401040184021402440404048405640604081408440"
    b"9040004120416141804185410142104248425642684200440844204480449944"
    b"124524450046014804481048404845480049584961498249454a904a00500850"
    b"1150195020508050885004514251a4519152905492540a550156545600581158"
    b"195864584059085a046010604060686000615561186260620064056410651265"
    b"84654268008002800a8041808280048118814081118201840484108415844084"
    b"608400854685948509864086608602880489118a0490109024904090a1901691"
    b"8091459200942294449451958198209902a050a085a009a100a218a450a804a9"
)
_GRID_MAP = (0x08, 0x19, 0x2B)
IQ2_XXS_BLOCK_VALUES = 256
IQ2_XXS_BLOCK_BYTES = 66


def _build_grid() -> tuple[tuple[int, ...], ...]:
    packed = bytes.fromhex(_GRID_HEX.decode("ascii"))
    values: list[int] = []
    for byte in packed:
        for shift in (0, 2, 4, 6):
            code = (byte >> shift) & 0x03
            if code >= len(_GRID_MAP):
                raise ValueError("invalid canonical IQ2_XXS grid code")
            values.append(_GRID_MAP[code])
    if len(values) != 256 * 8:
        raise ValueError(f"unexpected IQ2_XXS grid size {len(values)}")
    return tuple(tuple(values[i * 8 : (i + 1) * 8]) for i in range(256))


IQ2_XXS_GRID = _build_grid()


def _sign_byte(sign7: int) -> int:
    sign7 &= 0x7F
    return sign7 | ((sign7.bit_count() & 1) << 7)


def deterministic_input(ncols: int) -> list[float]:
    return [((i * 17 + 3) % 61 - 30) / 32.0 for i in range(ncols)]


def iq2_xxs_row_dot(row: bytes, input_values: Sequence[float]) -> float:
    ncols = len(input_values)
    if ncols <= 0 or ncols % IQ2_XXS_BLOCK_VALUES:
        raise ValueError("IQ2_XXS input length must be a positive multiple of 256")
    blocks = ncols // IQ2_XXS_BLOCK_VALUES
    expected = blocks * IQ2_XXS_BLOCK_BYTES
    if len(row) != expected:
        raise ValueError(f"IQ2_XXS row has {len(row)} bytes, expected {expected}")

    acc = 0.0
    for ib in range(blocks):
        block = row[ib * IQ2_XXS_BLOCK_BYTES : (ib + 1) * IQ2_XXS_BLOCK_BYTES]
        d = struct.unpack_from("<e", block, 0)[0]
        q = struct.unpack_from("<32H", block, 2)
        for group in range(8):
            qi = group * 4
            auxg = q[qi] | (q[qi + 1] << 16)
            auxs = q[qi + 2] | (q[qi + 3] << 16)
            db = float(d) * (0.5 + float(auxs >> 28)) * 0.25
            for part in range(4):
                grid = IQ2_XXS_GRID[(auxg >> (8 * part)) & 0xFF]
                signs = _sign_byte((auxs >> (7 * part)) & 0x7F)
                base = ib * 256 + group * 32 + part * 8
                for j in range(8):
                    sign = -1.0 if signs & (1 << j) else 1.0
                    acc += float(input_values[base + j]) * (db * float(grid[j]) * sign)
    return acc


def reference_rows(
    model: str | Path,
    matrix_file_offset: int,
    ncols: int,
    row_start: int,
    row_count: int,
    input_values: Sequence[float],
) -> list[float]:
    if len(input_values) != ncols:
        raise ValueError("reference input size mismatch")
    row_bytes = (ncols // IQ2_XXS_BLOCK_VALUES) * IQ2_XXS_BLOCK_BYTES
    fd = os.open(str(Path(model).expanduser().resolve()), os.O_RDONLY)
    try:
        values = []
        for row in range(row_start, row_start + row_count):
            data = os.pread(fd, row_bytes, matrix_file_offset + row * row_bytes)
            if len(data) != row_bytes:
                raise EOFError(f"short IQ2 reference read for row {row}")
            values.append(iq2_xxs_row_dot(data, input_values))
        return values
    finally:
        os.close(fd)
