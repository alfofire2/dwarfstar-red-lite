from __future__ import annotations

import argparse

from .redmetal_topk_parity import topk_layer_parity_probe


def _parse_ints(value: str) -> list[int]:
    try:
        result = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("experts must be a comma-separated integer list") from exc
    if not result:
        raise argparse.ArgumentTypeError("experts list cannot be empty")
    return result


def _parse_floats(value: str) -> list[float]:
    try:
        result = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("weights must be a comma-separated number list") from exc
    if not result:
        raise argparse.ArgumentTypeError("weights list cannot be empty")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="redlite-topk",
        description="Red Metal routed top-k layer validation tools",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    parity = sub.add_parser(
        "parity",
        help="Run resident top-k expert FFNs plus GPU router-weighted accumulation",
    )
    parity.add_argument("model")
    parity.add_argument("--layer", type=int, default=0)
    parity.add_argument("--top-k", type=int, default=10)
    parity.add_argument("--experts", type=_parse_ints)
    parity.add_argument("--weights", type=_parse_floats)
    parity.add_argument("--row-start", type=int, default=0)
    parity.add_argument("--rows", type=int, default=8)
    parity.add_argument("--cache-gib", type=float, default=0.25)
    parity.add_argument("--slots-per-slab", type=int, default=64)
    parity.add_argument("--atol", type=float, default=1e-3)
    parity.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args(argv)

    data = topk_layer_parity_probe(
        args.model,
        layer=args.layer,
        top_k=args.top_k,
        expert_ids=args.experts,
        router_weights=args.weights,
        row_start=args.row_start,
        rows=args.rows,
        cache_gib=args.cache_gib,
        slots_per_slab=args.slots_per_slab,
        atol=args.atol,
        rtol=args.rtol,
    )

    cache = data["cache"]
    metal = cache["metal"]
    print(f"quant type        : {data['quant_name']} ({data['ggml_type']})")
    print(f"layer / top-k     : {data['layer']} / {data['top_k']}")
    print(f"hidden / ffn      : {data['hidden_size']} / {data['ffn_size']}")
    print("experts           : " + ",".join(str(value) for value in data["experts"]))
    print("router weights    : " + ",".join(f"{value:.6f}" for value in data["router_weights"]))
    print(f"weight sum        : {data['router_weight_sum']:.7f}")
    print(f"output rows       : {data['row_start']}..{data['row_start'] + data['rows_tested'] - 1}")
    print("slots             : " + ",".join(str(value) for value in data["slot_ids"]))
    print(f"resident slots    : {cache['resident_slots']}/{cache['slot_capacity']}")
    print(f"Metal slabs       : {metal['slab_count']}")
    print(f"Metal allocated   : {metal['allocated_bytes']/(1024**3):.3f} GiB")
    print(f"expert loads      : {cache['loads']}")
    print(f"cache hits/misses : {cache['hits']}/{cache['misses']}")
    print(f"evictions         : {cache['evictions']}")
    print(f"SSD read          : {metal['bytes_read']/(1024**2):.3f} MiB")
    print(f"pread calls       : {metal['read_calls']}")
    print(f"SSD during top-k  : {data['ssd_bytes_during_execute']} bytes / {data['ssd_calls_during_execute']} calls")
    print(f"slots in-flight   : {'YES' if any(data['inflight_after_wait']) else 'NO'}")
    print(f"GPU top-k layer   : {data['gpu_ms']:.3f} ms")
    print(f"CPU reference     : {data['cpu_reference_ms']:.3f} ms")
    print(f"max abs error     : {data['max_abs_error']:.6g}")
    print(f"max rel error     : {data['max_rel_error']:.6g}")
    print(f"parity match      : {'YES' if data['match'] else 'NO'}")
    for index, (gpu, cpu) in enumerate(zip(data["gpu"], data["cpu"])):
        print(
            f"row {data['row_start'] + index:4d}        : "
            f"gpu={gpu:+.7f} cpu={cpu:+.7f} delta={gpu-cpu:+.3e}"
        )
    print(f"note              : {data['note']}")
    return 0 if data["match"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
