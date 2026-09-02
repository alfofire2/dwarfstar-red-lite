from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import re
from typing import Any

from .expert_map import read_tensor_directory

_ROUTED_RE = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$")

GGML_TYPE_NAMES = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    9: "Q8_1",
    10: "Q2_K",
    11: "Q3_K",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    15: "Q8_K",
    16: "IQ2_XXS",
    17: "IQ2_XS",
    18: "IQ3_XXS",
    19: "IQ1_S",
    20: "IQ4_NL",
    21: "IQ3_S",
    22: "IQ2_S",
    23: "IQ4_XS",
    30: "BF16",
}


def ggml_type_name(value: int) -> str:
    return GGML_TYPE_NAMES.get(value, f"TYPE_{value}")


def audit_routed_quantization(model: str | Path) -> dict[str, Any]:
    model = str(Path(model).expanduser().resolve())
    _, _, infos = read_tensor_directory(model)

    rows: list[dict[str, Any]] = []
    for info in infos:
        match = _ROUTED_RE.match(info.name)
        if not match:
            continue
        layer = int(match.group(1))
        kind = match.group(2)
        rows.append(
            {
                "layer": layer,
                "kind": kind,
                "name": info.name,
                "ggml_type": info.ggml_type,
                "type_name": ggml_type_name(info.ggml_type),
                "shape": list(info.shape),
                "bytes": info.n_bytes,
            }
        )

    rows.sort(key=lambda item: (item["layer"], ("gate", "up", "down").index(item["kind"])))
    total = Counter(row["type_name"] for row in rows)
    by_kind: dict[str, Counter[str]] = defaultdict(Counter)
    layer_patterns: Counter[str] = Counter()
    layer_detail: dict[int, dict[str, str]] = defaultdict(dict)

    for row in rows:
        by_kind[row["kind"]][row["type_name"]] += 1
        layer_detail[row["layer"]][row["kind"]] = row["type_name"]

    for layer in sorted(layer_detail):
        kinds = layer_detail[layer]
        pattern = "/".join(kinds.get(kind, "MISSING") for kind in ("gate", "up", "down"))
        layer_patterns[pattern] += 1

    return {
        "model": model,
        "routed_tensor_count": len(rows),
        "layers": len(layer_detail),
        "type_counts": dict(sorted(total.items())),
        "by_kind": {kind: dict(sorted(by_kind[kind].items())) for kind in ("gate", "up", "down")},
        "layer_patterns": dict(sorted(layer_patterns.items())),
        "layer_detail": {str(layer): layer_detail[layer] for layer in sorted(layer_detail)},
        "tensors": rows,
    }
