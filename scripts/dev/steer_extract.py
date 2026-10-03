#!/usr/bin/env python3
"""dev52: extract an activation steering vector from two prompt sets (difference of means).

    python3 scripts/dev/steer_extract.py MODEL --pos POS.txt --neg NEG.txt --layer 24 --out vector.f32

POS.txt / NEG.txt hold one user message per line. Each message is wrapped in the chat template, run through
`redlite-engine logits`, and the residual stream entering layer --layer at the last prompt position is read from the
dump. The vector is mean(pos) - mean(neg): `hidden` little-endian float32 values, the format of
`redlite-generate --steer FILE`. Strength 1 adds the full mean difference at every steered layer.
"""
from __future__ import annotations

import argparse
import math
import subprocess
import sys
import tempfile
from array import array
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def residual_at(bin_dir: Path, model: str, text: str, layer: int, hidden: int, layers: int, vocab: int, cache: str) -> array:
    ids = subprocess.check_output([str(bin_dir / "redlite-engine"), "tokenize", model, "--text", text, "--chat"],
                                  text=True, stderr=subprocess.DEVNULL).splitlines()[0]
    n = len(ids.split(","))
    with tempfile.NamedTemporaryFile(suffix=".bin") as dump:
        subprocess.run([str(bin_dir / "redlite-engine"), "logits", model, "--tokens", ids, "--backend", "gpu",
                        "--dump-from", str(n - 1), "--out", dump.name, "--cache-mib", cache, "--context", str(n + 8)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        data = array("f")
        with open(dump.name, "rb") as f:
            data.frombytes(f.read())
    rec = hidden * (layers + 2) + vocab
    if len(data) < rec:
        sys.exit(f"dump too short for {text[:40]!r}")
    last = data[len(data) - rec:]
    # record: [embed][layer 0 out]...[layer n-1 out][final norm][logits]; the input of `layer` is the previous output
    start = hidden * layer   # layer 0 input = embed at offset 0, layer L input = layer L-1 output at hidden * L
    return last[start:start + hidden]


def mean(vectors: list[array]) -> list[float]:
    return [sum(v[i] for v in vectors) / len(vectors) for i in range(len(vectors[0]))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--pos", required=True)
    ap.add_argument("--neg", required=True)
    ap.add_argument("--layer", type=int, default=24)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bin", default=str(ROOT / ".deps" / "redmetal"))
    ap.add_argument("--cache-mib", default="4096")
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--layers", type=int, default=48)
    ap.add_argument("--vocab", type=int, default=151936)
    args = ap.parse_args()
    sets = []
    for path in (args.pos, args.neg):
        lines = [x for x in Path(path).read_text().splitlines() if x.strip()]
        if not lines:
            sys.exit(f"{path} has no prompts")
        sets.append([residual_at(Path(args.bin), args.model, x, args.layer, args.hidden, args.layers, args.vocab, args.cache_mib)
                     for x in lines])
        print(f"{path}: {len(lines)} prompts")
    vec = [p - n for p, n in zip(mean(sets[0]), mean(sets[1]))]
    norm = math.sqrt(sum(x * x for x in vec))
    ref = math.sqrt(sum(x * x for x in mean(sets[1])))
    array("f", vec).tofile(open(args.out, "wb"))
    print(f"wrote {args.out}: layer {args.layer}, |v| = {norm:.3f} (|mean neg| = {ref:.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
