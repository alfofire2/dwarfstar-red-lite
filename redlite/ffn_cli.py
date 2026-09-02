from __future__ import annotations

import argparse
import json

from .redmetal_ffn_parity import expert_ffn_parity_probe


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="redlite-ffn",
        description="Red Metal resident single-expert FFN validation tools",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    parity = sub.add_parser("parity", help="Run gate+up->SiLU*up->down parity from one resident expert slot")
    parity.add_argument("model")
    parity.add_argument("--layer", type=int, default=0)
    parity.add_argument("--expert", type=int, default=0)
    parity.add_argument("--row-start", type=int, default=0)
    parity.add_argument("--rows", type=int, default=8)
    parity.add_argument("--cache-gib", type=float, default=0.25)
    parity.add_argument("--slots-per-slab", type=int, default=64)
    parity.add_argument("--atol", type=float, default=1e-3)
    parity.add_argument("--rtol", type=float, default=1e-4)
    parity.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    data = expert_ffn_parity_probe(
        args.model,
        layer=args.layer,
        expert=args.expert,
        row_start=args.row_start,
        rows=args.rows,
        cache_gib=args.cache_gib,
        slots_per_slab=args.slots_per_slab,
        atol=args.atol,
        rtol=args.rtol,
    )
    if args.json:
        print(json.dumps(data, indent=2))
        return 0 if data["match"] else 2

    cache = data["cache"]
    metal = cache["metal"]
    addrs = data["gpu_addresses"]
    print(f"quant type        : {data['quant_name']} ({data['ggml_type']})")
    print(f"layer/expert      : {data['layer']}/{data['expert']}")
    print(f"hidden / ffn      : {data['hidden_size']} / {data['ffn_size']}")
    print(f"gate shape        : {tuple(data['gate_shape'])}")
    print(f"down shape        : {tuple(data['down_shape'])}")
    print(f"output rows       : {data['row_start']}..{data['row_start'] + data['rows_tested'] - 1}")
    print(f"slot              : {data['slot_id']}/{cache['slot_capacity']}")
    print(f"Metal slabs       : {metal['slab_count']}")
    print(f"Metal allocated   : {metal['allocated_bytes']/(1024**3):.3f} GiB")
    print(f"GPU addr gate     : 0x{addrs['gate']:016x}")
    print(f"GPU addr up       : 0x{addrs['up']:016x}")
    print(f"GPU addr down     : 0x{addrs['down']:016x}")
    print(f"slot in-flight    : {'YES' if data['inflight_after_wait'] else 'NO'}")
    print(f"expert loads      : {cache['loads']}")
    print(f"SSD read          : {metal['bytes_read']/(1024**2):.3f} MiB")
    print(f"pread calls       : {metal['read_calls']}")
    print(f"SSD during FFN    : {data['ssd_reads_during_execute']} bytes")
    print(f"GPU full FFN      : {data['gpu_ms']:.3f} ms")
    print(f"CPU reference     : {data['cpu_reference_ms']:.3f} ms")
    print(f"max abs error     : {data['max_abs_error']:.6g}")
    print(f"max rel error     : {data['max_rel_error']:.6g}")
    print(f"parity match      : {'YES' if data['match'] else 'NO'}")
    for index, (gpu, cpu) in enumerate(zip(data['gpu'], data['cpu'])):
        row = data['row_start'] + index
        print(f"row {row:4d}        : gpu={gpu:+.7f} cpu={cpu:+.7f} delta={gpu-cpu:+.3e}")
    print(f"note              : {data['note']}")
    return 0 if data["match"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
