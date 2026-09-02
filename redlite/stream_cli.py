from __future__ import annotations
import argparse, json
from .streaming import make_streaming_plan, probe_streaming

def main(argv=None)->int:
    p=argparse.ArgumentParser(prog='redlite-stream',description='Experimental Qwen3-Next expert residency tools')
    sub=p.add_subparsers(dest='cmd',required=True)
    s=sub.add_parser('plan'); s.add_argument('model'); s.add_argument('--cache-gib',type=float,default=4.0); s.add_argument('--json',action='store_true')
    s=sub.add_parser('probe'); s.add_argument('model'); s.add_argument('--cache-gib',type=float,default=4.0); s.add_argument('--steps',type=int,default=4); s.add_argument('--top-k',type=int,default=10); s.add_argument('--json',action='store_true')
    a=p.parse_args(argv)
    if a.cmd=='plan':
        _,plan=make_streaming_plan(a.model,a.cache_gib); d=plan.to_dict()
    else:
        d=probe_streaming(a.model,a.cache_gib,a.steps,a.top_k)
    if a.json: print(json.dumps(d,indent=2))
    else:
        if a.cmd=='plan':
            print(f"cache             : {d['cache_gib']:.2f} GiB")
            print(f"layers            : {d['layers']}")
            print(f"routed tensors    : {d['routed_tensor_count']}")
            print(f"routed payload    : {d['routed_payload_gib']:.2f} GiB")
            print(f"slice safe        : {'YES' if d['slice_safe'] else 'NO'}")
            print(f"max expert triplet: {d['max_expert_triplet_mib']:.2f} MiB")
            print(f"cache capacity    : ~{d['estimated_triplets_in_cache']} expert triplets")
            print(f"probe ready       : {'YES' if d['ready_for_probe'] else 'NO'}")
            print(f"reason            : {d['reason']}")
        else:
            pr=d['probe']; st=pr['cache']
            print(f"route events      : {pr['route_events']}")
            print(f"elapsed           : {pr['elapsed_sec']:.3f} s")
            print(f"route events/s    : {pr['route_events_per_sec']:.1f}")
            print(f"cache hits/misses : {st['hits']}/{st['misses']}")
            print(f"evictions         : {st['evictions']}")
            print(f"peak mapped       : {st['peak_mapped_bytes']/(1024**3):.2f} GiB")
            print(f"note              : {pr['note']}")
    return 0

if __name__=='__main__': raise SystemExit(main())
