from __future__ import annotations

import ast
from functools import lru_cache
from math import ceil, log2
from pathlib import Path
from typing import Sequence


SUPPORTED_ROUTED_TYPES = {
    17: "IQ2_XS",
    29: "IQ1_M",
}

# IQ1_M intentionally reuses the canonical IQ1_S codebook.
_GRID_CLASS = {
    17: "IQ2_XS",
    29: "IQ1_S",
}


def default_llama_quants_path() -> Path:
    root = Path(__file__).resolve().parents[1]
    return root / ".deps" / "llama.cpp" / "gguf-py" / "gguf" / "quants.py"


def _class_literals(source: str, class_name: str) -> dict[str, object]:
    tree = ast.parse(source)
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        values: dict[str, object] = {}
        for child in node.body:
            if not isinstance(child, ast.Assign) or len(child.targets) != 1:
                continue
            target = child.targets[0]
            if not isinstance(target, ast.Name):
                continue
            if target.id not in {"grid_shape", "grid_map", "grid_hex"}:
                continue
            values[target.id] = ast.literal_eval(child.value)
        return values
    raise ValueError(f"class {class_name!r} not found in pinned llama.cpp quants.py")


def _ascii_hex_bytes(grid_hex: object) -> bytes:
    """Normalize gguf-py's grid_hex literal to one ASCII-hex byte string.

    Current llama.cpp quant classes use an implicitly concatenated bytes literal,
    so ast.literal_eval() returns one ``bytes`` object. Older/generated forms may
    use a tuple/list of bytes or strings; accept those too for robustness.
    """
    if isinstance(grid_hex, bytes):
        return grid_hex
    if isinstance(grid_hex, bytearray):
        return bytes(grid_hex)
    if isinstance(grid_hex, str):
        return grid_hex.encode("ascii")
    if isinstance(grid_hex, Sequence):
        parts: list[bytes] = []
        for part in grid_hex:
            if isinstance(part, bytes):
                parts.append(part)
            elif isinstance(part, bytearray):
                parts.append(bytes(part))
            elif isinstance(part, str):
                parts.append(part.encode("ascii"))
            else:
                raise TypeError(
                    f"unsupported quant grid_hex sequence item {type(part).__name__}"
                )
        return b"".join(parts)
    raise TypeError(f"unsupported quant grid_hex value {type(grid_hex).__name__}")


def _decode_grid(shape: Sequence[int], grid_map: Sequence[int], grid_hex: object) -> tuple[int, ...]:
    if len(shape) != 2 or not shape[0] or not shape[1]:
        raise ValueError(f"invalid quant grid shape {tuple(shape)}")
    if len(grid_map) < 2:
        raise ValueError("quant grid map must contain at least two values")

    bits_per_elem = ceil(log2(len(grid_map)))
    if bits_per_elem <= 0 or 8 % bits_per_elem:
        raise ValueError(f"unsupported grid encoding width {bits_per_elem}")
    elems_per_byte = 8 // bits_per_elem
    mask = (1 << bits_per_elem) - 1

    ascii_hex = _ascii_hex_bytes(grid_hex)
    packed = bytes.fromhex(ascii_hex.decode("ascii"))
    result: list[int] = []
    for byte in packed:
        for shift in range(0, 8, bits_per_elem):
            code = (byte >> shift) & mask
            if code >= len(grid_map):
                raise ValueError(f"invalid grid code {code} for map of size {len(grid_map)}")
            result.append(int(grid_map[code]))

    expected = int(shape[0]) * int(shape[1])
    if len(result) != expected:
        raise ValueError(f"decoded grid has {len(result)} values, expected {expected}")
    if len(result) != len(packed) * elems_per_byte:
        raise ValueError("quant grid decode accounting mismatch")
    return tuple(result)


@lru_cache(maxsize=8)
def _load_quant_grid_cached(ggml_type: int, source_path: str) -> tuple[int, ...]:
    if ggml_type not in _GRID_CLASS:
        raise ValueError(f"GGML type {ggml_type} is not a supported routed quant type")
    path = Path(source_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"pinned llama.cpp quant source not found at {path}. Run redlite bootstrap first."
        )
    values = _class_literals(path.read_text(encoding="utf-8"), _GRID_CLASS[ggml_type])
    missing = {"grid_shape", "grid_map", "grid_hex"} - values.keys()
    if missing:
        raise ValueError(f"pinned quant class is missing {sorted(missing)}")
    return _decode_grid(
        values["grid_shape"],  # type: ignore[arg-type]
        values["grid_map"],  # type: ignore[arg-type]
        values["grid_hex"],
    )


def load_quant_grid(ggml_type: int, source_path: str | Path | None = None) -> tuple[int, ...]:
    path = Path(source_path) if source_path is not None else default_llama_quants_path()
    return _load_quant_grid_cached(int(ggml_type), str(path.expanduser().resolve()))


def quant_type_name(ggml_type: int) -> str:
    return SUPPORTED_ROUTED_TYPES.get(int(ggml_type), f"TYPE_{ggml_type}")
