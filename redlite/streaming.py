from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import random, time
from typing import Any
from .expert_cache import CacheKey, MmapExpertCache
from .expert_map import ExpertMap, build_expert_map
GIB=1024**3

@dataclass(frozen=True)
class StreamingPlan:
    model:str; cache_gib:float; cache_bytes:int; expert_count:int; layers:int; routed_tensor_count:int; routed_payload_gib:float; slice_safe:bool; max_expert_triplet_mib:float; estimated_triplets_in_cache:int; ready_for_probe:bool; reason:str
    def to_dict(self)->dict[str,Any]: return asdict(self)

def make_streaming_plan(model:str|Path,cache_gib:float=4.0,expected_experts:int=512)->tuple[ExpertMap,StreamingPlan]:
    if cache_gib<=0: raise ValueError('cache_gib must be positive')
    em=build_expert_map(model,expected_experts); by_layer={}
    for t in em.expert_tensors:
        if t.slice_safe: by_layer[t.layer]=by_layer.get(t.layer,0)+t.expert_stride_bytes
    max_triplet=max(by_layer.values(),default=0); cb=int(cache_gib*GIB); cap=cb//max_triplet if max_triplet else 0
    safe=em.all_slice_safe and bool(em.layers)
    reason=('GGUF routed expert tensors are physically sliceable; mmap/LRU probe enabled. Metal binding is not enabled yet.' if safe else 'Merged expert tensor layout is not safely sliceable.')
    return em,StreamingPlan(str(Path(model).expanduser().resolve()),cache_gib,cb,expected_experts,len(em.layers),len(em.expert_tensors),em.total_routed_payload_bytes/GIB,safe,max_triplet/(1024**2),cap,safe and cap>=10,reason)

def probe_streaming(model:str|Path,cache_gib:float=4.0,steps:int=4,top_k:int=10,seed:int=1337,touch_bytes:int=4096)->dict[str,Any]:
    em,plan=make_streaming_plan(model,cache_gib)
    if not plan.ready_for_probe: raise ValueError(plan.reason)
    if steps<=0 or top_k<=0 or top_k>em.expert_count: raise ValueError('invalid steps/top_k')
    lt={}
    for t in em.expert_tensors:
        if t.slice_safe: lt.setdefault(t.layer,[]).append(t)
    rng=random.Random(seed); start=time.perf_counter(); touched=0; routes=0
    with MmapExpertCache(model,plan.cache_bytes) as cache:
        for _ in range(steps):
            for layer in em.layers:
                for expert in rng.sample(range(em.expert_count),top_k):
                    routes+=1
                    for t in lt[layer]:
                        off,n=t.expert_slice(expert); v=cache.acquire(CacheKey(layer,expert,t.kind),off,n)
                        k=min(touch_bytes,len(v))
                        if k: _=v[0]; _=v[k-1]; touched+=k
                        v.release()
        elapsed=time.perf_counter()-start; stats=cache.stats.to_dict(); resident=len(cache.entries)
    return {'plan':plan.to_dict(),'probe':{'steps':steps,'top_k':top_k,'route_events':routes,'elapsed_sec':elapsed,'route_events_per_sec':routes/elapsed if elapsed else None,'touched_bytes':touched,'resident_windows':resident,'cache':stats,'note':'Synthetic router trace; validates GGUF expert slicing and bounded SSD-backed residency only. No Metal MoE execution yet.'}}
