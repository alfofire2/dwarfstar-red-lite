#!/usr/bin/env python3
"""dev54: build a Red Lite expert-mix GGUF from Bartowski's Q8_0 and imatrix (pinned llama.cpp, dev tool only).

    python3 scripts/dev/quant_mix.py --like models/Qwen_Qwen3-Next-80B-A3B-Instruct-IQ2_XXS.gguf \\
        --q8 models/q8/Qwen_Qwen3-Next-80B-A3B-Instruct-Q8_0-00001-of-00003.gguf \\
        --imatrix models/q8/Qwen_Qwen3-Next-80B-A3B-Instruct-imatrix.gguf \\
        --iq2xs-layers 37-47 --out models/Qwen3-Next-80B-A3B-Instruct-RL-IQ2-E3.gguf

Every tensor gets the type it has in --like (Bartowski's IQ2_XXS: dense IQ2_XXS/Q4_K/Q6_K/Q8_0, Q2_K embedding,
Q5_K head), except the routed experts: IQ2_XS on --iq2xs-layers, IQ1_M elsewhere (the two expert types the native
engine runs at this size). The pinned llama.cpp's own IQ2_XXS recipe differs (IQ2_XXS experts, Q4_K attention) and
is not used. Quantizing from Q8_0 takes ~40 min on an M4 Max.

dev63, other sizes (the 48 GiB IQ3_XXS file as the template):
    --keep-experts               the experts keep their --like types (no IQ2_XS / IQ1_M rule)
    --set ffn_down_exps=iq3_s@0-47   a tensor (name without blk.N.) gets a type on some layers (all without @)
    --dense-map iq3_xxs=q8_0     every non-expert tensor of one type gets another
    --base-type IQ3_XXS          llama-quantize's base type (all tensors are listed anyway)
    --print-types                print the per-tensor types and exit
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / ".deps" / "llama.cpp" / "gguf-py"))


def layers(spec: str) -> set[int]:
    out: set[int] = set()
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out.update(range(int(a), int(b or a) + 1))
    return out


def type_lines(like: Path, iq2xs: set[int], dense_type: str | None = None, keep_experts: bool = False,
               sets: list[tuple[str, str, set[int] | None]] = (), dense_map: dict[str, str] | None = None) -> list[str]:
    from gguf import GGUFReader
    lines = []
    for t in GGUFReader(str(like)).tensors:
        ty = t.tensor_type.name.lower()
        if ty == "f32":
            continue
        m = re.match(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$", t.name)
        if m:
            if not keep_experts:
                ty = "iq2_xs" if int(m.group(1)) in iq2xs else "iq1_m"
        elif dense_type and ty == "iq2_xxs":   # dev58: the 2-bit dense projections every token reads
            ty = dense_type
        elif dense_map and ty in dense_map:      # dev63
            ty = dense_map[ty]
        lm = re.match(r"blk\.(\d+)\.(.+)\.weight$", t.name)
        for name, new, where in sets:            # dev63: explicit overrides, last one wins
            if lm and lm.group(2) == name and (where is None or int(lm.group(1)) in where):
                ty = new
        lines.append(f"^{re.escape(t.name)}$={ty}")
    return lines


def parse_set(spec: str) -> tuple[str, str, set[int] | None]:
    name, _, rest = spec.partition("=")
    ty, _, where = rest.partition("@")
    return name, ty.lower(), layers(where) if where else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--like", required=True, help="GGUF whose per-tensor types are copied (Bartowski IQ2_XXS)")
    ap.add_argument("--q8", required=True, help="first split of the Q8_0 source")
    ap.add_argument("--imatrix", required=True)
    ap.add_argument("--iq2xs-layers", default="37-47", help="expert layers at IQ2_XS, e.g. 37-47 or 0-5,43-47")
    ap.add_argument("--dense-type", help="dev58: type for the dense tensors that are IQ2_XXS in --like (e.g. iq3_xxs, q4_K)")
    ap.add_argument("--keep-experts", action="store_true", help="dev63: experts keep the --like types")
    ap.add_argument("--set", action="append", default=[], help="dev63: NAME=TYPE[@LAYERS], e.g. ffn_down_exps=iq3_s@24-47")
    ap.add_argument("--dense-map", action="append", default=[], help="dev63: FROM=TO for non-expert tensors, e.g. iq3_xxs=q8_0")
    ap.add_argument("--base-type", default="IQ2_XXS")
    ap.add_argument("--print-types", action="store_true")
    ap.add_argument("--out", required=False)
    ap.add_argument("--quantize", default=str(ROOT / ".deps" / "llama.cpp" / "build-ppl" / "bin" / "llama-quantize"))
    args = ap.parse_args()
    dense_map = dict(x.lower().split("=", 1) for x in args.dense_map)
    lines = type_lines(Path(args.like), layers(args.iq2xs_layers), args.dense_type, args.keep_experts,
                       [parse_set(x) for x in args.set], dense_map)
    if args.print_types:
        print("\n".join(lines))
        return 0
    if not args.out:
        ap.error("--out is required")
    with tempfile.NamedTemporaryFile("w", suffix=".types", delete=False) as f:
        f.write("\n".join(lines) + "\n")
    print(f"{len(lines)} tensor types" + ("" if args.keep_experts else f"; experts IQ2_XS on layers {args.iq2xs_layers}, IQ1_M elsewhere"))
    return subprocess.call([args.quantize, "--allow-requantize", "--imatrix", args.imatrix, "--tensor-type-file", f.name,
                            args.q8, args.out, args.base_type])


if __name__ == "__main__":
    sys.exit(main())
