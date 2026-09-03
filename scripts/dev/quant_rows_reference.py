#!/usr/bin/env python3
"""Development-only oracle: dequantize selected GGUF rows with gguf-py (pinned llama.cpp).

Prints the same digests as `redlite-engine-offline-test MODEL` so the native
Q2_K/Q5_K/Q4_K/Q6_K/Q8_0/IQ2_XXS decoders can be cross-checked independently.
Python is never used at inference time.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, ".deps", "llama.cpp", "gguf-py"))

import numpy as np  # noqa: E402
from gguf import GGUFReader  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

TARGETS = [
    ("token_embd.weight", 0),
    ("output.weight", 151645),
    ("output_norm.weight", 0),
    ("blk.0.ssm_out.weight", 5),
    ("blk.3.attn_k.weight", 7),
    ("blk.0.ffn_gate_shexp.weight", 3),
    ("blk.0.ssm_ba.weight", 1),
    ("blk.0.attn_qkv.weight", 9),
]


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: quant_rows_reference.py MODEL.gguf", file=sys.stderr)
        return 2
    reader = GGUFReader(sys.argv[1])
    by_name = {t.name: t for t in reader.tensors}
    for name, row in TARGETS:
        t = by_name[name]
        ncols = int(t.shape[0])
        # gguf-py exposes data as [rows..., row_bytes]; select the row and dequantize it alone
        raw = np.asarray(t.data)
        if raw.dtype != np.uint8:
            vals = raw.reshape(-1, ncols)[row].astype(np.float64)
        else:
            row_bytes = raw.reshape(-1, raw.shape[-1])[row:row + 1]
            vals = dequantize(row_bytes, t.tensor_type).reshape(-1).astype(np.float64)
        x = np.array([((i * 7 + 3) % 31 - 15) / 16.0 for i in range(ncols)], dtype=np.float64)
        print(f"{name}[row {row}]: type={t.tensor_type.name} sum={vals.sum():.9g} abs_sum={np.abs(vals).sum():.9g} "
              f"first=[{vals[0]:.9g} {vals[1]:.9g} {vals[2]:.9g} {vals[3]:.9g}] last={vals[-1]:.9g}")
        print(f"{name}[row {row}]: dequant_dot={float(vals @ x):.9g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
