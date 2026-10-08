from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path
import shutil
import subprocess
import sys

from . import __version__
from .hardware import GIB, _sysctl, detect, gpu_wired_limit_mib
from .model_catalog import VARIANTS, resolve_variant
from .planner import native_defaults, native_mtp_max_context, native_plan_context, plan_for, select_native_model
from .runner import engine_status, run_completion, run_server, run_bench, run_native_chat, run_native_server, ROOT
from .telemetry import snapshot
from .benchmark import run_sweep

# dev57: models next to the code in a source checkout, else in ~/.redlite/models (Homebrew / pip installs);
# REDLITE_MODELS overrides both
NATIVE_MODELS_DIR = Path(os.environ.get("REDLITE_MODELS") or (
    ROOT / "models" if (ROOT / "pyproject.toml").is_file() else Path.home() / ".redlite" / "models")).expanduser()
DEFAULT_NATIVE_MODEL = NATIVE_MODELS_DIR / "Qwen_Qwen3-Next-80B-A3B-Instruct-IQ2_XXS.gguf"


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
        "version": __version__,
        "apple_silicon": hw.is_apple_silicon,
        "system": hw.system,
        "machine": hw.machine,
        "chip": hw.chip,
        "ram_gib": round(hw.ram_gib, 2),
        "logical_cpus": hw.logical_cpus,
        "performance_cpus": hw.perf_cpus,
        "free_disk_gib": round(hw.free_disk_gib, 2),
        "memory": mem.to_dict() if mem else None,
        "dependencies": {
            "git": shutil.which("git") is not None,
            "cmake": shutil.which("cmake") is not None,
            "xcode_clang": shutil.which("clang") is not None,
            "hf": shutil.which("hf") is not None or shutil.which("huggingface-cli") is not None,
        },
        "engines": status,
    }
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        print(f"DwarfStar Red Lite {__version__}")
        print(f"Apple Silicon : {'YES' if hw.is_apple_silicon else 'NO'}")
        print(f"Chip          : {hw.chip}")
        if platform.system() == "Darwin":   # for bug reports (the macOS 27.0.1 GPU-driver panics, docs/WHAT_DID_NOT_WORK.md)
            print(f"macOS         : {platform.mac_ver()[0]} ({_sysctl('hw.model') or 'unknown model'})")
        print(f"RAM           : {hw.ram_gib:.2f} GiB")
        print(f"Perf CPUs     : {hw.perf_cpus}")
        print(f"Free SSD      : {hw.free_disk_gib:.1f} GiB")
        if mem:
            fp = f"{mem.free_percent}%" if mem.free_percent is not None else "unknown"
            sw = f"{mem.swap_used_gib:.2f} GiB" if mem.swap_used_gib is not None else "unknown"
            print(f"Memory free % : {fp}")
            print(f"Swap used     : {sw}")
        wired = gpu_wired_limit_mib()
        boot = GPU_LIMIT_DAEMON.is_file()
        print(f"GPU limit     : {f'{wired} MiB (iogpu.wired_limit_mb, ' + ('set at every boot by ' + str(GPU_LIMIT_DAEMON) if boot else 'until reboot') + ')' if wired else 'macOS default'}")
        _print_gpu_advice(hw, wired)
        for k, v in out["dependencies"].items():
            print(f"{k:13}: {'OK' if v else 'MISSING'}")
        for k, v in status.items():
            print(f"{k:18}: {'READY' if v else 'NOT BUILT'}")
    return 0 if hw.is_apple_silicon else 1


def cmd_pressure(args) -> int:
    hw = detect(ROOT)
    _require_apple(hw)
    mem = snapshot()
    if args.json:
        print(json.dumps(mem.to_dict(), indent=2))
    else:
        print(f"memory free % : {mem.free_percent if mem.free_percent is not None else 'unknown'}")
        print(f"swap used     : {mem.swap_used_gib:.2f} GiB" if mem.swap_used_gib is not None else "swap used     : unknown")
        print(f"swap total    : {mem.swap_total_gib:.2f} GiB" if mem.swap_total_gib is not None else "swap total    : unknown")
    return 0


def _make_plan(args):
    hw = detect(Path(args.model).parent)
    _require_apple(hw)
    try:
        plan = plan_for(hw, args.model, args.context, args.mode)
    except (FileNotFoundError, ValueError) as e:
        _die(str(e))
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
    if args.json:
        print(json.dumps(plan.to_dict(), indent=2))
    else:
        _print_plan(plan)
    return 0 if plan.safe else 3


def cmd_bootstrap(args) -> int:
    hw = detect(ROOT)
    _require_apple(hw)
    script = ROOT / "scripts" / "bootstrap_macos.sh"
    env = os.environ.copy()
    if args.jobs:
        env["REDLITE_JOBS"] = str(args.jobs)
    return subprocess.call([str(script)], cwd=ROOT, env=env)


def _hf_binary() -> list[str]:
    if shutil.which("hf"):
        return ["hf", "download"]
    if shutil.which("huggingface-cli"):
        return ["huggingface-cli", "download"]
    _die("Hugging Face CLI not found. Install: brew install hf (or python3 -m pip install -U huggingface_hub)")


SERVER_MODEL_ID = "qwen3-next-80b-a3b-redlite"   # the id redlite-server reports in /v1/models
AGENT_TEMPERATURE = 0.3                          # dev63: setup-pi's sampling temperature for agent loops


def pi_provider(port: int, context: int, host: str = "127.0.0.1") -> dict:
    """dev61: the pi coding agent's provider entry for a local redlite-server (docs/REDLITE_DEV60_CODING_AGENT.md)."""
    return {
        "baseUrl": f"http://{host}:{port}/v1", "api": "openai-completions", "apiKey": "redlite",
        "compat": {"supportsDeveloperRole": False, "supportsReasoningEffort": False, "supportsStore": False,
                   "supportsStrictMode": False, "maxTokensField": "max_tokens"},
        "models": [{"id": SERVER_MODEL_ID, "name": "Red Lite (local)", "reasoning": False, "input": ["text"],
                    "contextWindow": context, "maxTokens": min(4096, context // 4),
                    # dev63: the 2-bit Qwen3-Coder-Next looped 4 times in 36 agent sessions at temperature >= 0.7 and
                    # never at 0.3 (M4 Pro, repo suite); pi sends this with every request
                    "samplingParams": {"temperature": AGENT_TEMPERATURE},
                    "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}}],
    }


def cmd_setup_pi(args) -> int:
    """Adds (or updates) the "redlite" provider in pi's models.json; other providers are kept."""
    agent_dir = Path(os.environ.get("PI_CODING_AGENT_DIR") or Path.home() / ".pi" / "agent").expanduser()
    path = agent_dir / "models.json"
    config = {}
    if path.is_file():
        try:
            config = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            _die(f"{path} is not valid JSON ({e}); fix or move it, then run redlite setup-pi again")
    config.setdefault("providers", {})["redlite"] = pi_provider(args.port, args.context)
    agent_dir.mkdir(parents=True, exist_ok=True)
    # models.json may hold other providers' API keys: keep the file's permissions, and create it private (0600)
    mode = path.stat().st_mode & 0o777 if path.is_file() else 0o600
    tmp = path.with_suffix(".json.tmp")
    tmp.unlink(missing_ok=True)
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(fd, "w") as f:
        f.write(json.dumps(config, indent=2) + "\n")
    os.chmod(tmp, mode)   # the umask may have narrowed it
    tmp.replace(path)
    print(f"pi provider \"redlite\" written to {path} (server http://127.0.0.1:{args.port}, context {args.context}, "
          f"temperature {AGENT_TEMPERATURE})")
    print(f"start the server:  redlite serve --native --context {args.context} --port {args.port}")
    print(f"then:              pi --provider redlite --model {SERVER_MODEL_ID}")
    return 0


def cmd_download(args) -> int:
    try:
        v = resolve_variant(args.variant)
    except ValueError as e:
        _die(str(e))
    dest = Path(args.dir).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [*_hf_binary(), v.repo, v.filename, "--local-dir", str(dest)]
    print(f"variant : {v.key} ({v.nominal_gb:.1f} GB, quality={v.quality})")
    print("command :", " ".join(cmd))
    if args.dry_run:
        return 0
    try:
        return subprocess.call(cmd)
    except KeyboardInterrupt:
        print("\nDownload interrupted. Re-run the same command to resume from the Hugging Face cache.", file=sys.stderr)
        return 130


def cmd_models(args) -> int:
    if args.json:
        variants = []
        for v in VARIANTS.values():
            variants.append({
                "key": v.key,
                "repo": v.repo,
                "filename": v.filename,
                "nominal_gb": v.nominal_gb,
                "quality": v.quality,
                "recommended_mode": v.recommended_mode
            })
        print(json.dumps(variants, indent=2))
        return 0
    
    print("variant      approx GB   quality               default mode")
    for v in VARIANTS.values():
        print(f"{v.key:12} {v.nominal_gb:9.2f}   {v.quality:20} {v.recommended_mode}")
    return 0


def cmd_run(args) -> int:
    _, plan = _make_plan(args)
    _print_plan(plan)
    if not plan.safe and not args.force:
        _die("Plan rejected by safety budget. Use --force only if you accept macOS memory pressure/OOM risk.", 3)
    if plan.status == "CRITICAL" and not args.quiet_warning:
        print("[redlite] warning: CRITICAL resident margin; close memory-heavy apps and monitor 'redlite pressure'.")
    return run_completion(args.model, plan, args.prompt, args.tokens, args.extra, args.dry_run, args.single_turn)


def _mtp_for(model, cache_mib, args, quiet: bool = False) -> str | None:
    """dev45: pass the MTP head when it applies (full residency, Instruct file, head present); --no-mtp disables"""
    from .planner import native_mtp_file
    head = native_mtp_file(model, cache_mib, disabled=getattr(args, "no_mtp", False),
                           ram_bytes=detect(Path(model).parent).ram_bytes, wired_mib=gpu_wired_limit_mib())
    if head and not quiet:
        print(f"[redlite] MTP speculative decoding with {head.name} (same output as plain decoding; --no-mtp disables)")
    return str(head) if head else None


GPU_LIMIT_DAEMON = Path("/Library/LaunchDaemons/com.redlite.gpulimit.plist")   # dev55b: sets the limit at boot


def _print_gpu_advice(hw, wired: int) -> None:
    """dev51: on Macs where full residency does not fit by default, the GPU limit that would allow it."""
    from .planner import NATIVE_MTP_FILE, NATIVE_SMALL_MODELS, native_full_residency_mib, native_residency
    model = next((NATIVE_MODELS_DIR / n for n in NATIVE_SMALL_MODELS if (NATIVE_MODELS_DIR / n).is_file()), None)
    if hw.ram_gib >= 40 or model is None:
        return
    res = native_residency(model)
    if res is None:
        return
    need = native_full_residency_mib(res)
    head = NATIVE_MODELS_DIR / NATIVE_MTP_FILE
    need_mtp = native_full_residency_mib(res, mtp=True) if head.is_file() else None
    if need > hw.ram_bytes / (1024 * 1024) - 2048:
        return
    print(f"Full residency: needs a GPU limit of {need} MiB"
          + (f" ({need_mtp} MiB with MTP)" if need_mtp else "")
          + f"; now {'OK' if wired >= need else 'not enabled'}")
    if wired < need:
        print(f"  enable until reboot: sudo sysctl iogpu.wired_limit_mb={need_mtp or need}")
        print("  enable at every boot: the LaunchDaemon in docs/REDLITE_DEV51_24GB_DECODE.md")


ROUTE_CACHE_BIAS_DEFAULT = "0.5"


def _gpu_tuning(model, cache_mib, args, context: int):
    """dev55: (prefill chunk, MTP head) for this run. Under a raised GPU limit at full residency the chunk and MTP are
    chosen so the measured need (planner.native_full_residency_mib) fits; otherwise the defaults stand."""
    from .planner import native_gpu_plan, native_residency
    head = _mtp_for(model, cache_mib, args, quiet=True)
    wired = gpu_wired_limit_mib()
    res = native_residency(model)
    full = _full_residency(res, cache_mib)
    batch = args.batch
    parallel = getattr(args, "parallel", 1) or 1
    where = f"context {args.context or 4096}" + (f", {parallel} parallel requests" if parallel > 1 else "")
    if wired > 0 and full and batch is None:
        plan = native_gpu_plan(res, wired, context, head is not None)
        if plan is not None:
            batch, use_mtp = plan
            if head and not use_mtp:
                print(f"[redlite] MTP off: it does not fit the GPU limit ({wired} MiB) at {where}")
                head = None
            if batch != 2048:
                print(f"[redlite] prefill chunks of {batch} tokens to fit the GPU limit ({wired} MiB) at {where}")
    if wired == 0 and full and head:   # dev64: at the default limit, MTP must fit the RAM rule at this context too
        from .planner import NATIVE_WORKING_SET_FRACTION, native_full_residency_fits
        if not native_full_residency_fits(detect(Path(model).parent).ram_bytes, res, 0, context, mtp=True):
            print(f"[redlite] MTP off: with it, {Path(model).name} at {where} is above "
                  f"{NATIVE_WORKING_SET_FRACTION:.0%} of RAM")
            head = None
    if head:
        print(f"[redlite] MTP speculative decoding with {Path(head).name} (same output as plain decoding; --no-mtp disables)")
    return batch, head


def _kv_env(args) -> None:
    """dev75: --kv f16 stores the native KV cache as half (RL_KV_F16=1 for the binary and the planner): half the
    context memory, +9 % decode at 32K and +13 % at 64K on the M4 Max (dev74). Not the default: logits drift from the
    float cache (KL up to 5.8e-2 on CF2 at 1.1K positions), see docs/REDLITE_DEV74_DECODE_KERNELS.md."""
    import os
    if getattr(args, "kv", "f32") == "f16":
        os.environ["RL_KV_F16"] = "1"
    if os.environ.get("RL_KV_F16") == "1":
        print("[redlite] half-precision KV cache (half the context memory; outputs can differ slightly from the float cache)")


def _lookup_for(model, cache_mib, mtp, args) -> bool:
    """dev70: prompt lookup speculation when there is no MTP head (Qwen3-Coder-Next), same output; --no-lookup disables.
    Every expert resident: +15-25 % decode in coding-agent sessions on the M4 Max. dev72: with a bounded cache too (the
    verify loads both rows' experts), which needs exact routing: +7-10 % over cache-aware routing on the M4 Pro 24 GiB
    with the 4 GiB cache, so it replaces that default. A user's RL_ROUTE_CACHE_BIAS keeps it off."""
    import os
    lookup = mtp is None and not getattr(args, "no_lookup", False) and "RL_ROUTE_CACHE_BIAS" not in os.environ
    if lookup:
        print("[redlite] prompt lookup speculative decoding (same output as plain decoding; --no-lookup disables)")
    return lookup


def _full_residency(res, cache_mib) -> bool:
    """the expert cache holds every expert (res = planner.native_residency of the model)"""
    return res is not None and (str(cache_mib) == "full" or (str(cache_mib).isdigit() and int(cache_mib) >= res.cache_mib))


def _route_bias_env(model, cache_mib, args, lookup: bool = False) -> None:
    """dev51 (roadmap 2c): cache-aware routing by default with a bounded expert cache.

    lambda = 0.5 is +5 % decode on the M4 Pro 24 GiB with a 4 GiB cache, with no measured quality loss: perplexity
    17.586 -> 17.583 (dev46) and the same agreement with Qwen's own API on 235 prompts (dev51). It changes which
    experts are picked when scores are close, so the binaries stay exact by default and --exact-routing turns it off.
    With every expert resident there are no misses and the bias does nothing, so it is not set.
    """
    import os
    from .planner import native_residency
    if lookup or getattr(args, "exact_routing", False) or "RL_ROUTE_CACHE_BIAS" in os.environ:
        return   # dev72: prompt lookup needs exact routing and gains more
    res = native_residency(model)
    full = str(cache_mib) == "full" or (res is not None and str(cache_mib).isdigit() and int(cache_mib) >= res.cache_mib)
    if not full:
        os.environ["RL_ROUTE_CACHE_BIAS"] = ROUTE_CACHE_BIAS_DEFAULT
        print(f"[redlite] cache-aware expert routing (lambda {ROUTE_CACHE_BIAS_DEFAULT}, bounded cache); --exact-routing disables")


def _steer_args(args) -> list[str]:
    """dev52: redlite-generate steering options (and dev52b --history) from `redlite chat`"""
    out: list[str] = []
    for flag, value in (("--steer", args.steer), ("--steer-layers", args.steer_layers),
                        ("--steer-strength", args.steer_strength), ("--steer-tokens", args.steer_tokens),
                        ("--history", getattr(args, "history", None))):
        if value is not None:
            out += [flag, str(value)]
    return out


def cmd_chat(args) -> int:
    hw = detect(Path(args.model).parent if args.model else NATIVE_MODELS_DIR)
    _require_apple(hw)
    if args.model:
        model = Path(args.model).expanduser()
    else:
        # dev31: the best model present in models/ that this machine can hold
        picked = select_native_model(NATIVE_MODELS_DIR, hw.ram_bytes, gpu_wired_limit_mib())
        if picked is None:
            _die(f"No native model found in {NATIVE_MODELS_DIR}. Download one: redlite download 24gb (and redlite download mtp)")
        model = picked
        print(f"[redlite] model {model.name} (best native model in {NATIVE_MODELS_DIR} for {hw.ram_bytes / GIB:.0f} GiB RAM)")
    if not model.is_file():
        _die(f"Model not found: {model}")
    _kv_env(args)
    cache_mib = args.cache_mib
    if cache_mib is None:
        defaults = native_defaults(hw.ram_bytes, model, gpu_wired_limit_mib(), args.context)
        cache_mib = defaults.cache_mib
        print(f"[redlite] expert cache {cache_mib} MiB ({defaults.reason}); override with --cache-mib")
    batch, mtp = _gpu_tuning(model, cache_mib, args, args.context)
    lookup = _lookup_for(model, cache_mib, mtp, args)
    _route_bias_env(model, cache_mib, args, lookup)
    try:
        return run_native_chat(
            str(model), args.context, cache_mib, args.max_tokens,
            args.temperature, args.top_k, args.top_p, args.seed,
            args.system, args.prompt, args.stats, args.no_stream, args.dry_run,
            batch=batch, json_stats=args.json, min_p=args.min_p,
            mtp=mtp, steer=_steer_args(args), mtp_max_context=native_mtp_max_context(hw.ram_bytes), lookup=lookup,
        )
    except FileNotFoundError:
        _die("Native Red Lite runtime not built. Run: make native")


def cmd_serve(args) -> int:
    if args.native:
        return _serve_native(args)
    if not args.model:
        _die("redlite serve needs a model path (redlite serve --native picks one from the models folder)")
    _, plan = _make_plan(args)
    _print_plan(plan)
    if not plan.safe and not args.force:
        _die("Plan rejected by safety budget. Use --force only if you accept macOS memory pressure/OOM risk.", 3)
    if plan.status == "CRITICAL" and not args.quiet_warning:
        print("[redlite] warning: CRITICAL resident margin; server concurrency should remain 1 on 24 GiB.")
    return run_server(args.model, plan, args.host, args.port, args.extra, args.dry_run)


def _serve_native(args) -> int:
    hw = detect(Path(args.model).parent if args.model else NATIVE_MODELS_DIR)
    _require_apple(hw)
    if args.model:
        model = Path(args.model).expanduser()
    else:   # dev60: as redlite chat, the best model present that this Mac can hold
        model = select_native_model(NATIVE_MODELS_DIR, hw.ram_bytes, gpu_wired_limit_mib())
        if model is None:
            _die(f"No native model found in {NATIVE_MODELS_DIR}. Download one: redlite download 24gb (and redlite download mtp)")
        print(f"[redlite] model {model.name} (best native model in {NATIVE_MODELS_DIR} for {hw.ram_bytes / GIB:.0f} GiB RAM)")
    if not model.is_file():
        _die(f"Model not found: {model}")
    _kv_env(args)
    cache_mib = args.cache_mib
    context = args.context or 4096
    # dev56: a second slot holds another KV cache and DeltaNet state (72 MiB = 1536 positions at 48 KiB): the GPU
    # plan sizes for those positions too
    plan_context = native_plan_context(context, args.parallel)
    if cache_mib is None:
        defaults = native_defaults(hw.ram_bytes, model, gpu_wired_limit_mib(), plan_context)
        cache_mib = defaults.cache_mib
        print(f"[redlite] expert cache {cache_mib} MiB ({defaults.reason}); override with --cache-mib")
    batch, mtp = _gpu_tuning(model, cache_mib, args, plan_context)
    lookup = _lookup_for(model, cache_mib, mtp, args)
    _route_bias_env(model, cache_mib, args, lookup)
    try:
        return run_native_server(
            str(model), args.host, args.port, context, cache_mib,
            batch=batch, dry_run=args.dry_run, mtp=mtp, mtp_max_context=native_mtp_max_context(hw.ram_bytes),
            steer=_steer_args(args), parallel=args.parallel, lookup=lookup,
        )
    except FileNotFoundError:
        _die("Native Red Lite server not built. Run: make native")


def cmd_bench(args) -> int:
    _, plan = _make_plan(args)
    _print_plan(plan)
    if not plan.safe and not args.force:
        _die("Unsafe plan; benchmark not started.", 3)
    return run_bench(args.model, plan, args.prompt_tokens, args.gen_tokens, args.dry_run)


def _parse_contexts(value: str) -> list[int]:
    try:
        contexts = [int(v.strip()) for v in value.split(",") if v.strip()]
    except ValueError as e:
        raise argparse.ArgumentTypeError("contexts must be comma-separated integers") from e
    if not contexts or any(v <= 0 for v in contexts):
        raise argparse.ArgumentTypeError("contexts must contain positive integers")
    return contexts


def cmd_sweep(args) -> int:
    hw = detect(Path(args.model).parent)
    _require_apple(hw)
    return run_sweep(
        hw, args.model, args.contexts, args.prompt_tokens, args.gen_tokens,
        args.repetitions, args.output, args.max_swap_delta, args.min_free_percent,
        args.dry_run,
    )


def _add_plan_args(p, model_optional: bool = False):
    p.add_argument("model", nargs="?" if model_optional else None,
                   help="Path to a Qwen3-Next GGUF" + (" (--native: default, the best one in the models folder)" if model_optional else ""))
    p.add_argument("--mode", choices=["auto", "metal-resident", "ssd-cpu"], default="auto")
    p.add_argument("-c", "--context", type=int, default=None)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="redlite", description="Apple-Silicon-only oversized MoE runtime launcher")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("doctor", help="Check Apple Silicon hardware, memory and dependencies")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_doctor)

    s = sub.add_parser("pressure", help="Show current macOS memory pressure and swap")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_pressure)

    s = sub.add_parser("models", help="List curated Qwen3-Next 80B variants")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_models)

    s = sub.add_parser("plan", help="Calculate RAM/SSD execution plan")
    _add_plan_args(s)
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_plan)

    s = sub.add_parser("bootstrap", help="Build pinned Metal + oversized engines")
    s.add_argument("-j", "--jobs", type=int)
    s.set_defaults(func=cmd_bootstrap)

    s = sub.add_parser("setup-pi", help="Configure the pi coding agent for a local redlite serve --native")
    s.add_argument("--port", type=int, default=8080)
    # dev65: 64K, the agent window that measured best (hard suite, CF2, M4 Max: 16/18 at 64K, 10/18 at 32K)
    s.add_argument("--context", type=int, default=65536, help="the --context the server runs with (default 65536)")
    s.set_defaults(func=cmd_setup_pi)

    s = sub.add_parser("download", help="Download a curated GGUF from Hugging Face")
    s.add_argument("variant", nargs="?", default="24gb")
    s.add_argument("--dir", default=str(NATIVE_MODELS_DIR), help=f"Destination (default: {NATIVE_MODELS_DIR})")
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(func=cmd_download)

    s = sub.add_parser("run", help="Run local completion/chat")
    _add_plan_args(s)
    s.add_argument("-p", "--prompt", default="Hello! Introduce yourself briefly.")
    s.add_argument("-n", "--tokens", type=int, default=512)
    s.add_argument("--single-turn", action="store_true", help="Exit after the first generated answer")
    s.add_argument("--quiet-warning", action="store_true")
    s.add_argument("--force", action="store_true")
    s.add_argument("--dry-run", action="store_true")
    s.add_argument("--engine-args", dest="extra", nargs=argparse.REMAINDER, default=[], help="Remaining arguments are passed to the selected backend")
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("chat", help="Chat with the persistent native Red Lite runtime")
    s.add_argument(
        "model", nargs="?", default=None,
        help=f"Path to the Qwen3-Next GGUF (default: the best one in {NATIVE_MODELS_DIR}: IQ3_XXS when RAM >= 40 GiB "
             "and its experts fit, else IQ2_XXS)",
    )
    s.add_argument("-p", "--prompt", help="Optional first user message")
    s.add_argument("--system", help="Optional system prompt")
    s.add_argument("-c", "--context", type=int, default=4096, help="Context positions (default: 4096)")
    s.add_argument(
        "--cache-mib", type=int, default=None,
        help="Routed-expert cache in MiB (default: every expert resident and preloaded when RAM >= 40 GiB and the "
             "model's expert payload fits, e.g. 17316 for IQ2_XXS / 28800 for IQ3_XXS; otherwise 4096)",
    )
    s.add_argument("--batch", type=int, default=None, help="Prompt tokens per batched prefill chunk (default: 2048; 1 = token by token)")
    s.add_argument("-n", "--max-tokens", type=int, default=256, help="Maximum tokens per answer (default: 256)")
    s.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (default: 0.7; 0 = greedy)")
    s.add_argument("--top-k", type=int, default=40, help="Top-k sampling candidates (default: 40; 0 = off)")
    s.add_argument("--top-p", type=float, default=0.95, help="Nucleus probability (default: 0.95)")
    s.add_argument("--min-p", type=float, default=None, help="Drop candidates below this fraction of the top probability (default: off; llama.cpp uses 0.05)")
    s.add_argument("--no-mtp", action="store_true", help="Do not use the MTP head for speculative decoding even when it applies")
    s.add_argument("--no-lookup", action="store_true",
                   help="No prompt lookup speculative decoding (on by default with every expert resident and no MTP)")
    s.add_argument("--exact-routing", action="store_true", help="Bounded cache: pick experts exactly as the model does (no cache-aware routing)")
    s.add_argument("--kv", choices=("f32", "f16"), default="f32",
                   help="KV cache precision (default f32; f16 halves the context memory and is faster at long contexts, "
                        "outputs can differ slightly)")
    s.add_argument("--steer", default=None, help="Activation steering vector (scripts/dev/steer_extract.py); /steer S in the chat changes the strength")
    s.add_argument("--steer-layers", default=None, help="Steered layers A-B (default 16-31)")
    s.add_argument("--steer-strength", type=float, default=None, help="Steering strength (default 1; typical 0.2-0.5 over 8-12 layers)")
    s.add_argument("--steer-tokens", type=int, default=None, help="Steer only the first N tokens of each answer")
    s.add_argument("--history", default=None, help="Prefilled conversation: a text file of 'user: ...' / 'assistant: ...' turns read before the first message")
    s.add_argument("--seed", type=int, default=0, help="Sampling seed (default: fixed native seed)")
    s.add_argument("--stats", action="store_true", help="Print per-turn runtime statistics")
    s.add_argument("--json", action="store_true", help="Write per-turn statistics as JSON lines on stderr")
    s.add_argument("--no-stream", action="store_true", help="Print each answer only when complete")
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(func=cmd_chat)

    s = sub.add_parser("serve", help="Start OpenAI-compatible HTTP server")
    _add_plan_args(s, model_optional=True)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8080)
    s.add_argument("--native", action="store_true",
                   help="Serve with the native Red Lite runtime (redlite-server) instead of llama-server")
    s.add_argument("--cache-mib", type=int, default=None,
                   help="--native: routed-expert cache in MiB (default: the model's full-residency cache when RAM >= 40 GiB "
                        "and it fits, otherwise 4096)")
    s.add_argument("--batch", type=int, default=None, help="--native: prompt tokens per batched prefill chunk (default: 2048)")
    s.add_argument("--no-mtp", action="store_true", help="--native: do not use the MTP head for speculative decoding")
    s.add_argument("--no-lookup", action="store_true",
                   help="--native: no prompt lookup speculative decoding (on by default with every expert resident and no MTP)")
    s.add_argument("--exact-routing", action="store_true", help="--native, bounded cache: no cache-aware expert routing")
    s.add_argument("--kv", choices=("f32", "f16"), default="f32",
                   help="--native: KV cache precision (default f32; f16 halves the context memory, outputs can differ slightly)")
    s.add_argument("--steer", default=None, help="--native: activation steering vector for the generated tokens")
    s.add_argument("--steer-layers", default=None, help="--native: steered layers A-B (default 16-31)")
    s.add_argument("--steer-strength", type=float, default=None, help="--native: steering strength (default 1)")
    s.add_argument("--steer-tokens", type=int, default=None, help="--native: steer only the first N tokens of each answer")
    s.add_argument("--parallel", type=int, choices=(1, 2), default=1,
                   help="--native: requests served at the same time (2 needs every expert resident; their decode steps share "
                        "one pass over the weights)")
    s.add_argument("--quiet-warning", action="store_true")
    s.add_argument("--force", action="store_true")
    s.add_argument("--dry-run", action="store_true")
    s.add_argument("--engine-args", dest="extra", nargs=argparse.REMAINDER, default=[])
    s.set_defaults(func=cmd_serve)

    s = sub.add_parser("bench", help="Benchmark selected execution profile")
    _add_plan_args(s)
    s.add_argument("--prompt-tokens", type=int, default=256)
    s.add_argument("--gen-tokens", type=int, default=128)
    s.add_argument("--force", action="store_true")
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(func=cmd_bench)

    s = sub.add_parser("sweep", help="Benchmark multiple resident context depths with swap telemetry")
    s.add_argument("model", help="Path to a Qwen3-Next GGUF")
    s.add_argument("--contexts", type=_parse_contexts, default=[2048, 4096, 8192], help="comma-separated depths; default 2048,4096,8192")
    s.add_argument("--prompt-tokens", type=int, default=256)
    s.add_argument("--gen-tokens", type=int, default=128)
    s.add_argument("-r", "--repetitions", type=int, default=2)
    s.add_argument("--max-swap-delta", type=float, default=0.5, help="stop after a depth grows swap by more than this many GiB")
    s.add_argument("--min-free-percent", type=int, default=10, help="stop if macOS system-wide free memory falls below this percentage")
    s.add_argument("--output", default="benchmarks/redlite-sweep.json")
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(func=cmd_sweep)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\n[redlite] interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
