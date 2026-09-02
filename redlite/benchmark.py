from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any

from .planner import Plan, plan_for
from .hardware import HardwareInfo
from .runner import LLAMA_BIN, _first_existing, _quote
from .telemetry import snapshot


@dataclass
class SweepResult:
    context: int
    status: str
    safe: bool
    headroom_gib: float
    returncode: int | None
    prompt_ts: float | None
    generation_ts: float | None
    swap_before_gib: float | None
    swap_after_gib: float | None
    raw: list[dict[str, Any]]
    skipped_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _extract_rates(rows: list[dict[str, Any]]) -> tuple[float | None, float | None]:
    pp = tg = None
    for row in rows:
        n_prompt = int(row.get("n_prompt", 0) or 0)
        n_gen = int(row.get("n_gen", 0) or 0)
        rate = row.get("avg_ts")
        if rate is None:
            continue
        if n_prompt > 0 and n_gen == 0:
            pp = float(rate)
        if n_gen > 0 and n_prompt == 0:
            tg = float(rate)
    return pp, tg


def _bench_command(model: str, plan: Plan, prompt_tokens: int, gen_tokens: int, repetitions: int) -> list[str]:
    bench = _first_existing([LLAMA_BIN / "llama-bench"])
    return [
        str(bench),
        "-m", model,
        "-ngl", "999",
        "-p", str(prompt_tokens),
        "-n", str(gen_tokens),
        "-d", str(plan.context),
        "-b", str(plan.batch),
        "-ub", str(plan.ubatch),
        "-t", str(plan.threads),
        "-ctk", "q8_0",
        "-ctv", "q8_0",
        "-fa", "on",
        "-r", str(repetitions),
        "-o", "json",
    ]


def run_sweep(
    hw: HardwareInfo,
    model: str,
    contexts: list[int],
    prompt_tokens: int,
    gen_tokens: int,
    repetitions: int,
    output: str | None,
    max_swap_delta_gib: float = 0.5,
    min_free_percent: int = 10,
    dry_run: bool = False,
) -> int:
    results: list[SweepResult] = []
    rc = 0

    print("context  status     headroom   prompt t/s   gen t/s   swap delta")
    print("-------  ---------  ---------  -----------  --------  ----------")

    for ctx in contexts:
        plan = plan_for(hw, model, ctx, "metal-resident")
        if not plan.safe:
            results.append(SweepResult(ctx, plan.status, False, plan.headroom_gib, None, None, None, None, None, [], plan.reason))
            print(f"{ctx:7d}  {plan.status:9}  {plan.headroom_gib:+8.2f}  {'SKIPPED':>11}  {'-':>8}  {'-':>10}")
            continue

        cmd = _bench_command(model, plan, prompt_tokens, gen_tokens, repetitions)
        if dry_run:
            print("[redlite]", " ".join(_quote(x) for x in cmd))
            results.append(SweepResult(ctx, plan.status, True, plan.headroom_gib, 0, None, None, None, None, []))
            continue

        before = snapshot()
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        after = snapshot()
        rows: list[dict[str, Any]] = []
        if proc.stdout.strip():
            try:
                parsed = json.loads(proc.stdout)
                rows = parsed if isinstance(parsed, list) else [parsed]
            except json.JSONDecodeError:
                print(proc.stdout)
        if proc.stderr.strip():
            print(proc.stderr, end="" if proc.stderr.endswith("\n") else "\n")
        pp, tg = _extract_rates(rows)
        delta = None
        if before.swap_used_gib is not None and after.swap_used_gib is not None:
            delta = after.swap_used_gib - before.swap_used_gib

        results.append(SweepResult(
            ctx, plan.status, True, plan.headroom_gib, proc.returncode, pp, tg,
            before.swap_used_gib, after.swap_used_gib, rows,
        ))
        pp_s = f"{pp:.1f}" if pp is not None else "?"
        tg_s = f"{tg:.1f}" if tg is not None else "?"
        d_s = f"{delta:+.2f} GiB" if delta is not None else "?"
        print(f"{ctx:7d}  {plan.status:9}  {plan.headroom_gib:+8.2f}  {pp_s:>11}  {tg_s:>8}  {d_s:>10}")
        if proc.returncode != 0:
            rc = proc.returncode
            break
        if delta is not None and delta > max_swap_delta_gib:
            print(f"[redlite] stopping sweep: swap grew by {delta:.2f} GiB (limit {max_swap_delta_gib:.2f} GiB).")
            break
        if after.free_percent is not None and after.free_percent < min_free_percent:
            print(f"[redlite] stopping sweep: system-wide free memory is {after.free_percent}% (floor {min_free_percent}%).")
            break

    if output:
        payload = {
            "schema": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": str(Path(model)),
            "hardware": {
                "chip": hw.chip,
                "ram_gib": hw.ram_gib,
                "performance_cpus": hw.perf_cpus,
            },
            "prompt_tokens": prompt_tokens,
            "generation_tokens": gen_tokens,
            "repetitions": repetitions,
            "max_swap_delta_gib": max_swap_delta_gib,
            "min_free_percent": min_free_percent,
            "results": [r.to_dict() for r in results],
        }
        path = Path(output).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2))
        print(f"results          : {path}")

    return rc
