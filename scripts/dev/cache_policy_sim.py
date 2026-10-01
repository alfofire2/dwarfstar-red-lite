#!/usr/bin/env python3
"""dev37: replay RL_ROUTE_TRACE routing traces through expert cache policies (hit rate is machine-independent).

Trace format: per decoded token, n_layer x top_k uint16 expert ids, layer-major (redlite-generate RL_ROUTE_TRACE=file).
Every layer's selection is one transaction, as in rl_native_lru_prepare_many: its resident keys are reserved
before any victim is chosen, so a selection never evicts its own experts.

usage: cache_policy_sim.py --slots 4723 [--layers 48 --topk 10] trace.bin [...]
"""
import argparse
import heapq
from collections import OrderedDict
import math
import struct
import sys


def load(path, layers, topk):
    data = open(path, "rb").read()
    n = len(data) // 2
    ids = struct.unpack(f"<{n}H", data[: 2 * n])
    per_tok = layers * topk
    steps = []
    for t in range(n // per_tok):
        base = t * per_tok
        for layer in range(layers):
            sel = ids[base + layer * topk: base + (layer + 1) * topk]
            steps.append([layer * 512 + e for e in sel])
    return steps


class LRU:
    """The engine's policy: evict the oldest stamp."""

    def __init__(self, slots):
        self.slots, self.od = slots, OrderedDict()

    def access(self, sel):
        hits = sum(1 for k in sel if k in self.od)
        for k in sel:
            if k in self.od:
                self.od.move_to_end(k)
        for k in sel:
            if k not in self.od:
                if len(self.od) >= self.slots:
                    self.od.popitem(last=False)   # the selection's own keys were just moved to the end
                self.od[k] = True
        return hits


class Decayed:
    """Frequency with exponential decay (half-life in transactions): score = sum over uses of 2^(-age/half).

    Ordering two keys by their decayed scores equals ordering log(score at last use) + lam * last_use, which does not
    change with time, so a lazy min-heap holds the victim order.
    """

    def __init__(self, slots, half):
        self.slots, self.t, self.lam = slots, 0, math.log(2.0) / half
        self.logs, self.last, self.key, self.heap = {}, {}, {}, []

    def access(self, sel):
        self.t += 1
        hits = sum(1 for k in sel if k in self.key)
        for k in sel:
            if k in self.logs:   # resident or remembered (ghost) history
                prev = self.logs[k] - self.lam * (self.t - self.last[k])
                self.logs[k] = prev + math.log1p(math.exp(-prev)) if prev > -50 else 0.0
            else:
                self.logs[k] = 0.0
            self.last[k] = self.t
        for k in sel:
            if k not in self.key and len(self.key) >= self.slots:
                kept = []
                while True:
                    kv, victim = heapq.heappop(self.heap)
                    if self.key.get(victim) != kv:
                        continue   # stale entry
                    if victim in sel:
                        kept.append((kv, victim))   # reserved by this selection
                        continue
                    del self.key[victim]
                    break
                for e in kept:
                    heapq.heappush(self.heap, e)
            kv = self.logs[k] + self.lam * self.t
            self.key[k] = kv
            heapq.heappush(self.heap, (kv, k))
        if len(self.heap) > 8 * self.slots:   # drop stale heap entries
            self.heap = [(v, k) for k, v in self.key.items()]
            heapq.heapify(self.heap)
        return hits


class SLRU:
    """Segmented LRU: new keys enter a probation segment, a second use promotes them to a protected segment of
    `protect` x slots; victims come from probation first (scan resistant, keeps recency)."""

    def __init__(self, slots, protect):
        self.slots, self.cap_p = slots, int(slots * protect)
        self.prob, self.prot = OrderedDict(), OrderedDict()

    def access(self, sel):
        hits = sum(1 for k in sel if k in self.prob or k in self.prot)
        for k in sel:
            if k in self.prot:
                self.prot.move_to_end(k)
            elif k in self.prob:
                del self.prob[k]
                self.prot[k] = True
                while len(self.prot) > self.cap_p:   # demote the protected LRU to probation's MRU end
                    d, _ = self.prot.popitem(last=False)
                    self.prob[d] = True
        for k in sel:
            if k in self.prob or k in self.prot:
                continue
            if len(self.prob) + len(self.prot) >= self.slots:
                src = self.prob if self.prob else self.prot
                victim = next(v for v in src if v not in sel)
                del src[victim]
            self.prob[k] = True
        return hits


def belady(steps, slots):
    """Optimal offline replacement (evict the resident key used farthest in the future): the upper bound."""
    nxt_use = [{} for _ in steps]
    later = {}
    for i in range(len(steps) - 1, -1, -1):
        nxt_use[i] = {k: later.get(k, math.inf) for k in steps[i]}
        for k in steps[i]:
            later[k] = i
    res, nu, heap, hits = set(), {}, [], 0
    for i, sel in enumerate(steps):
        for k in sel:
            if k in res:
                hits += 1
        for k in sel:
            nu[k] = nxt_use[i][k]
        for k in sel:
            if k not in res:
                while len(res) >= slots:
                    negnext, victim = heapq.heappop(heap)
                    if victim in res and -negnext == nu[victim] and victim not in sel:
                        res.discard(victim)
                res.add(k)
        for k in sel:
            heapq.heappush(heap, (-nu[k], k))
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", type=int, required=True)
    ap.add_argument("--layers", type=int, default=48)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--halves", default="250,1000")
    ap.add_argument("traces", nargs="+")
    a = ap.parse_args()
    rows = []
    for path in a.traces:
        steps = load(path, a.layers, a.topk)
        total = sum(len(s) for s in steps)
        res = {"LRU": sum(map(LRU(a.slots).access, steps))}
        for h in a.halves.split(","):
            pol = Decayed(a.slots, float(h))
            res[f"decay{h}"] = sum(map(pol.access, steps))
        for f in (0.5, 0.8, 0.95):
            res[f"slru{f}"] = sum(map(SLRU(a.slots, f).access, steps))
        res["OPT"] = belady(steps, a.slots)
        rows.append((path, total, res))
        print(path, " ".join(f"{k}={v / total:.4f}" for k, v in res.items()), flush=True)
    keys = rows[0][2].keys()
    tot = sum(r[1] for r in rows)
    print("ALL", " ".join(f"{k}={sum(r[2][k] for r in rows) / tot:.4f}" for k in keys))
    print("misses/token", " ".join(f"{k}={(tot - sum(r[2][k] for r in rows)) / (tot / (a.layers * a.topk)):.1f}" for k in keys))


if __name__ == "__main__":
    sys.exit(main())


def partitioned(steps, budget_bytes, big_layers, big_bytes, small_bytes, big_share):
    """dev37: two LRU pools, one per slot size class; big_share of the byte budget goes to the big class."""
    big = LRU(int(budget_bytes * big_share) // big_bytes)
    small = LRU(int(budget_bytes * (1.0 - big_share)) // small_bytes)
    hits = 0
    for sel in steps:
        hits += (big if sel[0] // 512 in big_layers else small).access(sel)
    return hits, big.slots, small.slots
