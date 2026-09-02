from __future__ import annotations

import argparse
import json

from .streaming import make_streaming_plan, probe_streaming


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="redlite-stream",
        description="Experimental Qwen3-Next expert residency tools",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    plan_cmd = sub.add_parser("plan")
    plan_cmd.add_argument("model")
    plan_cmd.add_argument("--cache-gib", type=float, default=4.0)
    plan_cmd.add_argument("--json", action="store_true")

    probe_cmd = sub.add_parser("probe")
    probe_cmd.add_argument("model")
    probe_cmd.add_argument("--cache-gib", type=float, default=4.0)
    probe_cmd.add_argument("--steps", type=int, default=4)
    probe_cmd.add_argument("--top-k", type=int, default=10)
    probe_cmd.add_argument("--prefetch-depth", type=int, default=1)
    probe_cmd.add_argument("--prefetch-workers", type=int, default=2)
    probe_cmd.add_argument("--json", action="store_true")

    metal_cmd = sub.add_parser(
        "metal-probe",
        help="Read routed experts directly into shared Metal slabs and verify GPU visibility",
    )
    metal_cmd.add_argument("model")
    metal_cmd.add_argument("--cache-gib", type=float, default=0.5)
    metal_cmd.add_argument("--steps", type=int, default=1)
    metal_cmd.add_argument("--top-k", type=int, default=10)
    metal_cmd.add_argument("--slots-per-slab", type=int, default=64)
    metal_cmd.add_argument("--json", action="store_true")

    parity_cmd = sub.add_parser(
        "iq2-parity",
        help="Run a correctness-first IQ2_XXS single-expert Metal matvec parity test",
    )
    parity_cmd.add_argument("model")
    parity_cmd.add_argument("--layer", type=int, default=0)
    parity_cmd.add_argument("--expert", type=int, default=0)
    parity_cmd.add_argument("--kind", choices=("gate", "up", "down"), default="gate")
    parity_cmd.add_argument("--row-start", type=int, default=0)
    parity_cmd.add_argument("--rows", type=int, default=8)
    parity_cmd.add_argument("--cache-gib", type=float, default=0.25)
    parity_cmd.add_argument("--slots-per-slab", type=int, default=64)
    parity_cmd.add_argument("--atol", type=float, default=1e-3)
    parity_cmd.add_argument("--rtol", type=float, default=1e-4)
    parity_cmd.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)

    if args.cmd == "plan":
        _, plan = make_streaming_plan(args.model, args.cache_gib)
        data = plan.to_dict()
    elif args.cmd == "probe":
        data = probe_streaming(
            args.model,
            args.cache_gib,
            args.steps,
            args.top_k,
            prefetch_depth=args.prefetch_depth,
            prefetch_workers=args.prefetch_workers,
        )
    elif args.cmd == "metal-probe":
        from .redmetal_streaming import probe_redmetal

        data = probe_redmetal(
            args.model,
            args.cache_gib,
            args.steps,
            args.top_k,
            slots_per_slab=args.slots_per_slab,
        )
    else:
        from .redmetal_parity import iq2_parity_probe

        data = iq2_parity_probe(
            args.model,
            layer=args.layer,
            expert=args.expert,
            kind=args.kind,
            row_start=args.row_start,
            rows=args.rows,
            cache_gib=args.cache_gib,
            slots_per_slab=args.slots_per_slab,
            atol=args.atol,
            rtol=args.rtol,
        )

    if args.json:
        print(json.dumps(data, indent=2))
        return 0

    if args.cmd == "plan":
        print(f"cache             : {data['cache_gib']:.2f} GiB")
        print(f"layers            : {data['layers']}")
        print(f"routed tensors    : {data['routed_tensor_count']}")
        print(f"routed payload    : {data['routed_payload_gib']:.2f} GiB")
        print(f"slice safe        : {'YES' if data['slice_safe'] else 'NO'}")
        print(f"max expert triplet: {data['max_expert_triplet_mib']:.2f} MiB")
        print(f"slot size         : {data['slot_bytes']/(1024**2):.2f} MiB")
        print(f"slot capacity     : {data['slot_capacity']}")
        print(f"probe ready       : {'YES' if data['ready_for_probe'] else 'NO'}")
        print(f"reason            : {data['reason']}")
        return 0

    if args.cmd == "iq2-parity":
        cache = data["cache"]
        metal = cache["metal"]
        addrs = data["gpu_addresses"]
        print(f"tensor            : {data['tensor']}")
        print(f"shape             : {tuple(data['shape'])}")
        print(f"layer/expert/kind : {data['layer']}/{data['expert']}/{data['kind']}")
        print(f"rows tested       : {data['row_start']}..{data['row_start'] + data['rows_tested'] - 1}")
        print(f"row bytes         : {data['row_bytes']}")
        print(f"slot              : {data['slot_id']}/{cache['slot_capacity']}")
        print(f"Metal slabs       : {metal['slab_count']}")
        print(f"Metal allocated   : {metal['allocated_bytes']/(1024**3):.3f} GiB")
        print(f"GPU addr gate     : 0x{addrs['gate']:016x}")
        print(f"GPU addr up       : 0x{addrs['up']:016x}")
        print(f"GPU addr down     : 0x{addrs['down']:016x}")
        print(f"slot in-flight    : {'YES' if data['inflight_after_wait'] else 'NO'}")
        print(f"GPU matvec        : {data['gpu_ms']:.3f} ms")
        print(f"max abs error     : {data['max_abs_error']:.6g}")
        print(f"max rel error     : {data['max_rel_error']:.6g}")
        print(f"parity match      : {'YES' if data['match'] else 'NO'}")
        for index, (gpu, cpu) in enumerate(zip(data['gpu'], data['cpu'])):
            print(f"row {data['row_start'] + index:4d}        : gpu={gpu:+.7f} cpu={cpu:+.7f} delta={gpu-cpu:+.3e}")
        print(f"note              : {data['note']}")
        return 0 if data["match"] else 2

    probe = data["probe"]
    stats = probe["cache"]

    if args.cmd == "metal-probe":
        metal = stats["metal"]
        print(f"device            : {probe['device']}")
        print(f"route events      : {probe['route_events']}")
        print(f"elapsed           : {probe['elapsed_sec']:.3f} s")
        print(f"route events/s    : {probe['route_events_per_sec']:.1f}")
        print(f"cache hits/misses : {stats['hits']}/{stats['misses']}")
        print(f"evictions         : {stats['evictions']}")
        print(f"resident slots    : {stats['resident_slots']}/{stats['slot_capacity']}")
        print(f"Metal slabs       : {metal['slab_count']}")
        print(f"Metal allocated   : {metal['allocated_bytes']/(1024**3):.2f} GiB")
        print(f"SSD read          : {probe['ssd_read_gib']:.2f} GiB")
        print(f"SSD throughput    : {probe['ssd_read_mib_per_sec']:.1f} MiB/s")
        print(f"pread calls       : {metal['read_calls']}")
        print(f"GPU probe         : {probe['gpu_probe_ms']:.3f} ms")
        print(f"GPU checksum      : 0x{probe['gpu_probe_checksum']:08x}")
        print(f"CPU verify        : 0x{probe['cpu_verify_checksum']:08x}")
        print(f"checksum match    : {'YES' if probe['gpu_checksum_match'] else 'NO'}")
        print(f"note              : {probe['note']}")
        return 0

    store = stats["store"]
    print(f"route events      : {probe['route_events']}")
    print(f"elapsed           : {probe['elapsed_sec']:.3f} s")
    print(f"route events/s    : {probe['route_events_per_sec']:.1f}")
    print(f"cache hits/misses : {stats['hits']}/{stats['misses']}")
    print(f"prefetch submitted: {stats['prefetch_submitted']}")
    print(f"prefetch waits    : {stats['prefetch_waits']}")
    print(f"evictions         : {stats['evictions']}")
    print(f"resident slots    : {probe['resident_slots']}/{stats['slot_capacity']}")
    print(f"allocated cache   : {stats['allocated_bytes']/(1024**3):.2f} GiB")
    print(f"SSD read          : {probe['ssd_read_gib']:.2f} GiB")
    print(f"SSD throughput    : {probe['ssd_read_mib_per_sec']:.1f} MiB/s")
    print(f"pread calls       : {store['read_calls']}")
    print(f"note              : {probe['note']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
