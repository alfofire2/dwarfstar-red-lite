#!/usr/bin/env python3
"""Development-only: compare two per-token activation dumps (native engine vs pinned llama.cpp).

Dump layout per token (f32): [hidden] embed, [n_layer][hidden] layer outputs,
[hidden] final norm, [vocab] logits.
"""
import argparse
import sys

import numpy as np


def stats(a: np.ndarray, b: np.ndarray):
    d = np.abs(a - b)
    denom = np.maximum(np.abs(b), 1e-12)
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    return float(d.max()), float((d / denom).max()), cos, int(d.argmax())


def softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("native")
    ap.add_argument("reference")
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--layers", type=int, default=48)
    ap.add_argument("--vocab", type=int, default=151936)
    ap.add_argument("--all-layers", action="store_true")
    args = ap.parse_args()
    rec = args.hidden * (args.layers + 2) + args.vocab
    a = np.fromfile(args.native, dtype=np.float32)
    b = np.fromfile(args.reference, dtype=np.float32)
    if a.size % rec or b.size % rec:
        print(f"bad dump sizes: {a.size} {b.size} (record {rec})", file=sys.stderr)
        return 1
    n = min(a.size // rec, b.size // rec)
    a = a[: n * rec].reshape(n, rec)
    b = b[: n * rec].reshape(n, rec)
    worst_logit = 0.0
    ok = True
    for t in range(n):
        ea, eb = a[t, : args.hidden], b[t, : args.hidden]
        print(f"token {t}: embed max_abs={stats(ea, eb)[0]:.3e}")
        for l in range(args.layers):
            o = args.hidden * (1 + l)
            la, lb = a[t, o : o + args.hidden], b[t, o : o + args.hidden]
            mx, rel, cos, idx = stats(la, lb)
            scale = float(np.abs(lb).max())
            if args.all_layers or l in (0, 1, 2, 3, 7, 23, 46, 47):
                print(f"  layer {l:02d}: max_abs={mx:.4e} (ref |max| {scale:.3f}, rel-to-scale {mx / max(scale, 1e-12):.2e}) cos={cos:.6f}")
        o = args.hidden * (1 + args.layers)
        fa, fb = a[t, o : o + args.hidden], b[t, o : o + args.hidden]
        mx, rel, cos, idx = stats(fa, fb)
        print(f"  final_norm: max_abs={mx:.4e} cos={cos:.6f}")
        la, lb = a[t, o + args.hidden :], b[t, o + args.hidden :]
        mx, rel, cos, idx = stats(la, lb)
        pa, pb = softmax(la.astype(np.float64)), softmax(lb.astype(np.float64))
        kl = float(np.sum(pb * (np.log(pb + 1e-30) - np.log(pa + 1e-30))))
        ta, tb = np.argsort(-la)[:5], np.argsort(-lb)[:5]
        print(f"  logits: max_abs={mx:.4e} at {idx} (native {la[idx]:.4f} ref {lb[idx]:.4f}) cos={cos:.6f} KL(ref||native)={kl:.3e}")
        print(f"          argmax native={int(la.argmax())} ref={int(lb.argmax())} top5 native={ta.tolist()} ref={tb.tolist()}")
        worst_logit = max(worst_logit, mx)
        if int(la.argmax()) != int(lb.argmax()):
            ok = False
    print(f"worst logits max_abs: {worst_logit:.4e}")
    print("ARGMAX AGREEMENT:", "YES" if ok else "NO")
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
