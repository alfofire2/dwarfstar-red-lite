from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from . import __version__
from .hardware import detect
from .model_catalog import VARIANTS, resolve_variant
from .planner import plan_for
from .runner import engine_status, run_completion, run_server, run_bench, ROOT
from .telemetry import snapshot
from .benchmark import run_sweep
from .streaming import build_manifest, native_probe, streaming_plan


def _die(msg: str, code: int = 2) -> None:
    print(f"redlite: {msg}", file=sys.stderr)
    raise SystemExit(code)


def _require_apple(hw) -> None:
    if not hw.is_apple_silicon:
        _die(f"Apple Silicon required (detected {hw.system}/{hw.machine}).")


def cmd_doctor(args) -> int:
    hw = detect(ROOT)
    status = engine_status()
    mem = snapshot() if hw.is_apple_silicon else None
    out = {
        "version": __version__, "apple_silicon": hw.is_apple_silicon, "system": hw.system,
        "machine": hw.machine, "chip": hw.chip, "ram_gib": round(hw.ram_gib, 2),
        "logical_cpus": hw.logical_cpus, "performance_cpus": hw.perf_cpus,
        "free_disk_gib": round(hw.free_disk_gib, 2), "memory": mem.to_dict() if mem else None,
        "dependencies": {
            "git": shutil.which("git") is not None, "cmake": shutil.which("cmake") is not None,
            "xcode_clang": shutil.which("clang") is not None,
            "hf": shutil.which("hf") is not None or shutil.which("huggingface-cli") is not None,
        }, "engines": status,
    }
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        print(f"DwarfStar Red Lite {__version__}")
        print(f"Apple Silicon : {'YES' if hw.is_apple_silicon else 'NO'}")
        print(f"Chip          : {hw.chip}")
        print(f"RAM           : {hw.ram_gib:.2f} GiB")
        print(f"Perf CPUs     : {hw.perf_cpus}")
        print(f"Free SSD      : {hw.free_disk_gib:.1f} GiB")
        if mem:
            print(f"Memory free % : {mem.free_percent if mem.free_percent is not None else 'unknown'}%" if mem.free_percent is not None else "Memory free % : unknown")
            print(f"Swap used     : {mem.swap_used_gib:.2f} GiB" if mem.swap_used_gib is not None else "Swap used     : unknown")
        for k, v in out["dependencies"].items(): print(f"{k:13}: {'OK' if v else 'MISSING'}")
        for k, v in status.items(): print(f"{k:18}: {'READY' if v else 'NOT BUILT'}")
    return 0 if hw.is_apple_silicon else 1


def cmd_pressure(args) -> int:
    hw = detect(ROOT); _require_apple(hw); mem = snapshot()
    if args.json: print(json.dumps(mem.to_dict(), indent=2))
    else:
        print(f"memory free % : {mem.free_percent if mem.free_percent is not None else 'unknown'}")
        print(f"swap used     : {mem.swap_used_gib:.2f} GiB" if mem.swap_used_gib is not None else "swap used     : unknown")
        print(f"swap total    : {mem.swap_total_gib:.2f} GiB" if mem.swap_total_gib is not None else "swap total    : unknown")
    return 0


def _make_plan(args):
    hw = detect(Path(args.model).parent); _require_apple(hw)
    try: plan = plan_for(hw, args.model, args.context, args.mode)
    except (FileNotFoundError, ValueError) as e: _die(str(e))
    return hw, plan


def _print_plan(plan) -> None:
    print(f"mode             : {plan.mode}")
    print(f"status           : {plan.status}")
    print(f"model            : {plan.model_gib:.2f} GiB")
    print(f"unified RAM      : {plan.ram_gib:.2f} GiB")
    print(f"resident budget  : {plan.resident_budget_gib:.2f} GiB")
    print(f"headroom         : {plan.headroom_gib:+.2f} GiB")
    print(f"context          : {plan.context}")
    print(f"threads          : {plan.threads}")
    print(f"batch/ubatch     : {plan.batch}/{plan.ubatch}")
    print(f"budget pass      : {'YES' if plan.safe else 'NO'}")
    print(f"reason           : {plan.reason}")


def cmd_plan(args) -> int:
    _, plan = _make_plan(args)
    print(json.dumps(plan.to_dict(), indent=2)) if args.json else _print_plan(plan)
    return 0 if plan.safe else 3


def cmd_bootstrap(args) -> int:
    hw = detect(ROOT); _require_apple(hw); env = os.environ.copy()
    if args.jobs: env["REDLITE_JOBS"] = str(args.jobs)
    return subprocess.call([str(ROOT / "scripts" / "bootstrap_macos.sh")], cwd=ROOT, env=env)


def _hf_binary() -> list[str]:
    if shutil.which("hf"): return ["hf", "download"]
    if shutil.which("huggingface-cli"): return ["huggingface-cli", "download"]
    _die("Hugging Face CLI not found. Install: python3 -m pip install -U huggingface_hub")


def cmd_download(args) -> int:
    try: v = resolve_variant(args.variant)
    except ValueError as e: _die(str(e))
    dest = Path(args.dir).expanduser().resolve(); dest.mkdir(parents=True, exist_ok=True)
    cmd = [*_hf_binary(), v.repo, v.filename, "--local-dir", str(dest)]
    print(f"variant : {v.key} ({v.nominal_gb:.1f} GB, quality={v.quality})"); print("command :", " ".join(cmd))
    if args.dry_run: return 0
    try: return subprocess.call(cmd)
    except KeyboardInterrupt:
        print("\nDownload interrupted. Re-run the same command to resume from the Hugging Face cache.", file=sys.stderr); return 130


def cmd_models(args) -> int:
    print("variant      approx GB   quality               default mode")
    for v in VARIANTS.values(): print(f"{v.key:12} {v.nominal_gb:9.2f}   {v.quality:20} {v.recommended_mode}")
    return 0


def cmd_run(args) -> int:
    _, plan = _make_plan(args); _print_plan(plan)
    if not plan.safe and not args.force: _die("Plan rejected by safety budget. Use --force only if you accept macOS memory pressure/OOM risk.", 3)
    if plan.status == "CRITICAL" and not args.quiet_warning: print("[redlite] warning: CRITICAL resident margin; close memory-heavy apps and monitor 'redlite pressure'.")
    return run_completion(args.model, plan, args.prompt, args.tokens, args.extra, args.dry_run, args.single_turn)


def cmd_serve(args) -> int:
    _, plan = _make_plan(args); _print_plan(plan)
    if not plan.safe and not args.force: _die("Plan rejected by safety budget. Use --force only if you accept macOS memory pressure/OOM risk.", 3)
    if plan.status == "CRITICAL" and not args.quiet_warning: print("[redlite] warning: CRITICAL resident margin; server concurrency should remain 1 on 24 GiB.")
    return run_server(args.model, plan, args.host, args.port, args.extra, args.dry_run)


def cmd_bench(args) -> int:
    _, plan = _make_plan(args); _print_plan(plan)
    if not plan.safe and not args.force: _die("Unsafe plan; benchmark not started.", 3)
    return run_bench(args.model, plan, args.prompt_tokens, args.gen_tokens, args.dry_run)


def _parse_contexts(value: str) -> list[int]:
    try: contexts = [int(v.strip()) for v in value.split(",") if v.strip()]
    except ValueError as e: raise argparse.ArgumentTypeError("contexts must be comma-separated integers") from e
    if not contexts or any(v <= 0 for v in contexts): raise argparse.ArgumentTypeError("contexts must contain positive integers")
    return contexts


def cmd_sweep(args) -> int:
    hw = detect(Path(args.model).parent); _require_apple(hw)
    return run_sweep(hw, args.model, args.contexts, args.prompt_tokens, args.gen_tokens,
                     args.repetitions, args.output, args.max_swap_delta, args.min_free_percent, args.dry_run)


def cmd_stream_build(args) -> int:
    hw = detect(ROOT); _require_apple(hw)
    return subprocess.call([str(ROOT / "scripts" / "build_streaming_macos.sh")], cwd=ROOT)


def cmd_expert_index(args) -> int:
    try: manifest, summary = build_manifest(args.model, args.manifest)
    except (OSError, ValueError) as e: _die(str(e))
    if args.json: print(json.dumps(summary, indent=2))
    else:
        print(f"GGUF             : {summary['gguf_version']}")
        print(f"tensors          : {summary['tensor_count']}")
        print(f"routed tensors   : {summary['routed_tensor_count']}")
        print(f"routed experts   : {summary['routed_expert_gib']:.2f} GiB")
        print(f"layers           : {summary['layers']}")
        print(f"experts/layer    : {summary['experts_per_layer']}")
        print(f"expert records   : {summary['expert_records']}")
        print(f"record size      : {summary['expert_record_min_kib']:.1f}-{summary['expert_record_max_kib']:.1f} KiB")
        print(f"manifest         : {manifest}")
    return 0


def cmd_stream_plan(args) -> int:
    hw = detect(Path(args.model).parent); _require_apple(hw)
    try: plan = streaming_plan(args.model, hw.ram_gib, args.cache_gib, args.reserve_gib)
    except (OSError, ValueError) as e: _die(str(e))
    if args.json: print(json.dumps(plan.to_dict(), indent=2))
    else:
        print(f"model             : {plan.model_gib:.2f} GiB")
        print(f"routed experts    : {plan.routed_expert_gib:.2f} GiB")
        print(f"non-routed est.   : {plan.non_routed_gib:.2f} GiB")
        print(f"expert cache      : {plan.cache_gib:.2f} GiB")
        print(f"resident estimate : {plan.estimated_resident_gib:.2f} GiB")
        print(f"OS/runtime reserve: {plan.reserve_gib:.2f} GiB")
        print(f"headroom          : {plan.headroom_gib:+.2f} GiB")
        print(f"expert records    : {plan.expert_records} across {plan.layers} layers")
        print(f"budget pass       : {'YES' if plan.safe else 'NO'}")
    return 0 if plan.safe else 3


def cmd_stream_probe(args) -> int:
    hw = detect(Path(args.model).parent); _require_apple(hw)
    try:
        manifest, summary = build_manifest(args.model, str(Path(args.manifest).expanduser()))
        print(f"[redlite] indexed {summary['expert_records']} expert records; routed payload {summary['routed_expert_gib']:.2f} GiB")
        return native_probe(args.model, str(manifest), args.cache_mib, args.requests, args.hotset,
                            args.prefetch, args.seed, args.nocache, args.dry_run)
    except (OSError, ValueError, FileNotFoundError) as e: _die(str(e))


def _add_plan_args(p):
    p.add_argument("model", help="Path to a Qwen3-Next GGUF")
    p.add_argument("--mode", choices=["auto", "metal-resident", "ssd-cpu"], default="auto")
    p.add_argument("-c", "--context", type=int, default=None)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="redlite", description="Apple-Silicon-only oversized MoE runtime launcher")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("doctor", help="Check Apple Silicon hardware, memory and dependencies"); s.add_argument("--json", action="store_true"); s.set_defaults(func=cmd_doctor)
    s = sub.add_parser("pressure", help="Show current macOS memory pressure and swap"); s.add_argument("--json", action="store_true"); s.set_defaults(func=cmd_pressure)
    s = sub.add_parser("models", help="List curated Qwen3-Next 80B variants"); s.set_defaults(func=cmd_models)
    s = sub.add_parser("plan", help="Calculate RAM/SSD execution plan"); _add_plan_args(s); s.add_argument("--json", action="store_true"); s.set_defaults(func=cmd_plan)
    s = sub.add_parser("bootstrap", help="Build pinned Metal + oversized engines"); s.add_argument("-j", "--jobs", type=int); s.set_defaults(func=cmd_bootstrap)
    s = sub.add_parser("download", help="Download a curated GGUF from Hugging Face"); s.add_argument("variant", nargs="?", default="24gb"); s.add_argument("--dir", default="models"); s.add_argument("--dry-run", action="store_true"); s.set_defaults(func=cmd_download)

    s = sub.add_parser("run", help="Run local completion/chat"); _add_plan_args(s)
    s.add_argument("-p", "--prompt", default="Hello! Introduce yourself briefly."); s.add_argument("-n", "--tokens", type=int, default=512)
    s.add_argument("--single-turn", action="store_true"); s.add_argument("--quiet-warning", action="store_true"); s.add_argument("--force", action="store_true"); s.add_argument("--dry-run", action="store_true")
    s.add_argument("--engine-args", dest="extra", nargs=argparse.REMAINDER, default=[]); s.set_defaults(func=cmd_run)

    s = sub.add_parser("serve", help="Start OpenAI-compatible HTTP server"); _add_plan_args(s)
    s.add_argument("--host", default="127.0.0.1"); s.add_argument("--port", type=int, default=8080); s.add_argument("--quiet-warning", action="store_true"); s.add_argument("--force", action="store_true"); s.add_argument("--dry-run", action="store_true")
    s.add_argument("--engine-args", dest="extra", nargs=argparse.REMAINDER, default=[]); s.set_defaults(func=cmd_serve)

    s = sub.add_parser("bench", help="Benchmark selected execution profile"); _add_plan_args(s); s.add_argument("--prompt-tokens", type=int, default=256); s.add_argument("--gen-tokens", type=int, default=128); s.add_argument("--force", action="store_true"); s.add_argument("--dry-run", action="store_true"); s.set_defaults(func=cmd_bench)

    s = sub.add_parser("sweep", help="Benchmark multiple resident context depths with swap telemetry"); s.add_argument("model"); s.add_argument("--contexts", type=_parse_contexts, default=[2048,4096,8192]); s.add_argument("--prompt-tokens", type=int, default=256); s.add_argument("--gen-tokens", type=int, default=128); s.add_argument("-r", "--repetitions", type=int, default=2); s.add_argument("--max-swap-delta", type=float, default=0.5); s.add_argument("--min-free-percent", type=int, default=10); s.add_argument("--output", default="benchmarks/redlite-sweep.json"); s.add_argument("--dry-run", action="store_true"); s.set_defaults(func=cmd_sweep)

    s = sub.add_parser("expert-index", help="Index routed Qwen3-Next expert slices inside a GGUF"); s.add_argument("model"); s.add_argument("--manifest", default=".redlite/expert-manifest.tsv"); s.add_argument("--json", action="store_true"); s.set_defaults(func=cmd_expert_index)
    s = sub.add_parser("stream-plan", help="Estimate resident memory for experimental Metal expert streaming"); s.add_argument("model"); s.add_argument("--cache-gib", type=float, default=4.0); s.add_argument("--reserve-gib", type=float, default=4.5); s.add_argument("--json", action="store_true"); s.set_defaults(func=cmd_stream_plan)
    s = sub.add_parser("stream-build", help="Build native SSD -> Metal expert cache probe"); s.set_defaults(func=cmd_stream_build)
    s = sub.add_parser("stream-probe", help="Exercise real GGUF expert slices through the native Metal cache"); s.add_argument("model"); s.add_argument("--manifest", default=".redlite/expert-manifest.tsv"); s.add_argument("--cache-mib", type=int, default=1024); s.add_argument("--requests", type=int, default=5000); s.add_argument("--hotset", type=int, default=1024); s.add_argument("--prefetch", type=int, default=4); s.add_argument("--seed", type=int, default=42); s.add_argument("--nocache", action="store_true"); s.add_argument("--dry-run", action="store_true"); s.set_defaults(func=cmd_stream_probe)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try: return int(args.func(args))
    except KeyboardInterrupt:
        print("\n[redlite] interrupted.", file=sys.stderr); return 130


if __name__ == "__main__":
    raise SystemExit(main())
