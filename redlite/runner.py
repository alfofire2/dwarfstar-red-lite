from __future__ import annotations

import os
import shutil
from pathlib import Path
import signal
import subprocess
from typing import Iterable

from .planner import Plan

ROOT = Path(__file__).resolve().parents[1]
DEPS = ROOT / ".deps"
LLAMA_BIN = DEPS / "llama.cpp" / "build" / "bin"
OMR_BIN = DEPS / "oversized-moe-runtime" / "build" / "bin"
REDMETAL_BIN = DEPS / "redmetal"


def _first_existing(paths: Iterable[Path]) -> Path:
    for p in paths:
        if p.exists() and os.access(p, os.X_OK):
            return p
    raise FileNotFoundError("No compatible engine binary found. Run: redlite bootstrap")


def llama_cli() -> Path:
    return _first_existing([LLAMA_BIN / "llama-cli", LLAMA_BIN / "llama-completion"])


def llama_server() -> Path:
    return _first_existing([LLAMA_BIN / "llama-server"])


def oversized_cli() -> Path:
    return _first_existing([OMR_BIN / "oversized-moe", OMR_BIN / "oversized-moe-run"])


def _native(name: str) -> Path:
    """dev57: the native binary from a source checkout (.deps/redmetal), else from PATH (Homebrew, release tarball)."""
    found = shutil.which(name)
    return _first_existing([REDMETAL_BIN / name] + ([Path(found)] if found else []))


def native_generate() -> Path:
    return _native("redlite-generate")


def native_server() -> Path:
    return _native("redlite-server")


def engine_status() -> dict[str, bool]:
    return {
        "native_redlite_generate": (REDMETAL_BIN / "redlite-generate").exists() or shutil.which("redlite-generate") is not None,
        "metal_llama_cli": any(p.exists() for p in [LLAMA_BIN / "llama-cli", LLAMA_BIN / "llama-completion"]),
        "metal_llama_server": (LLAMA_BIN / "llama-server").exists(),
        "oversized_moe": any(p.exists() for p in [OMR_BIN / "oversized-moe", OMR_BIN / "oversized-moe-run"]),
    }


def call_native_foreground(cmd: list[str]) -> int:
    """Run a native binary that handles Ctrl-C itself (stop the answer / shut down cleanly).

    subprocess.call would turn the terminal's SIGINT into a KeyboardInterrupt here and kill
    the child, so the parent ignores SIGINT while the child runs with the default disposition.
    """
    previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        proc = subprocess.Popen(cmd, preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL))
        return proc.wait()
    finally:
        signal.signal(signal.SIGINT, previous)


def run_native_chat(
    model: str,
    context: int,
    cache_mib: int,
    max_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    seed: int,
    system: str | None = None,
    prompt: str | None = None,
    stats: bool = False,
    no_stream: bool = False,
    dry_run: bool = False,
    batch: int | None = None,
    json_stats: bool = False,
    min_p: float | None = None,
    mtp: str | None = None,
    mtp_max_context: int | None = None,
    steer: list[str] | None = None,
    lookup: bool = False,
) -> int:
    cmd = [
        str(native_generate()), model, "--interactive",
        "--context", str(context),
        "--cache-mib", str(cache_mib),
        "--max-tokens", str(max_tokens),
        "--temperature", str(temperature),
        "--top-k", str(top_k),
        "--top-p", str(top_p),
        "--seed", str(seed),
    ]
    if system:
        cmd.extend(["--system", system])
    if prompt:
        cmd.extend(["--prompt", prompt])
    if stats:
        cmd.append("--stats")
    if no_stream:
        cmd.append("--no-stream")
    if batch is not None:
        cmd.extend(["--batch", str(batch)])
    if json_stats:
        cmd.append("--json")
    if min_p is not None:
        cmd.extend(["--min-p", str(min_p)])
    if mtp:
        cmd.extend(["--mtp", mtp])
        if mtp_max_context:
            cmd.extend(["--mtp-max-context", str(mtp_max_context)])
    if steer:
        cmd.extend(steer)
    if lookup:
        cmd.append("--lookup")
    print("[redlite]", " ".join(_quote(x) for x in cmd))
    if dry_run:
        return 0
    return call_native_foreground(cmd)


def run_native_server(
    model: str,
    host: str,
    port: int,
    context: int,
    cache_mib: int,
    batch: int | None = None,
    dry_run: bool = False,
    mtp: str | None = None,
    mtp_max_context: int | None = None,
    steer: list[str] | None = None,
    parallel: int = 1,
    lookup: bool = False,
) -> int:
    cmd = [
        str(native_server()), model,
        "--host", host,
        "--port", str(port),
        "--context", str(context),
        "--cache-mib", str(cache_mib),
    ]
    if batch is not None:
        cmd.extend(["--batch", str(batch)])
    if mtp:
        cmd.extend(["--mtp", mtp])
        if mtp_max_context:
            cmd.extend(["--mtp-max-context", str(mtp_max_context)])
    if steer:
        cmd.extend(steer)
    if parallel > 1:
        cmd.extend(["--parallel", str(parallel)])
    if lookup:
        cmd.append("--lookup")
    print("[redlite]", " ".join(_quote(x) for x in cmd))
    if dry_run:
        return 0
    return call_native_foreground(cmd)


def metal_common(model: str, plan: Plan) -> list[str]:
    return [
        "-m", model,
        "-ngl", "999",
        "--flash-attn", "on",
        "-c", str(plan.context),
        "-b", str(plan.batch),
        "-ub", str(plan.ubatch),
        "-t", str(plan.threads),
        "-tb", str(plan.threads),
        "-ctk", "q8_0",
        "-ctv", "q8_0",
        "--no-warmup",
    ]


def run_completion(model: str, plan: Plan, prompt: str, tokens: int, extra: list[str] | None = None, dry_run: bool = False, single_turn: bool = False) -> int:
    extra = extra or []
    if plan.mode == "metal-resident":
        cmd = [str(llama_cli()), *metal_common(model, plan), "-n", str(tokens), "-p", prompt]
        if single_turn:
            cmd.append("--single-turn")
        cmd.extend(extra)
    else:
        exe = oversized_cli()
        # v0.2 exposes a multi-command `oversized-moe` executable. Keep compatibility
        # with a possible standalone oversized-moe-run binary.
        if exe.name == "oversized-moe":
            cmd = [str(exe), "run", model, "-t", str(plan.threads), "-tb", str(plan.threads), "-c", str(plan.context), "-n", str(tokens), "-p", prompt, *extra]
        else:
            cmd = [str(exe), model, "-t", str(plan.threads), "-tb", str(plan.threads), "-c", str(plan.context), "-n", str(tokens), "-p", prompt, *extra]
    print("[redlite]", " ".join(_quote(x) for x in cmd))
    if dry_run:
        return 0
    return subprocess.call(cmd)


def run_server(model: str, plan: Plan, host: str, port: int, extra: list[str] | None = None, dry_run: bool = False) -> int:
    extra = extra or []
    if plan.mode == "metal-resident":
        cmd = [str(llama_server()), *metal_common(model, plan), "--host", host, "--port", str(port), *extra]
    else:
        exe = oversized_cli()
        if exe.name == "oversized-moe":
            cmd = [str(exe), "serve", model, "--host", host, "--port", str(port), "-t", str(plan.threads), "-c", str(plan.context), *extra]
        else:
            # v0.2 documented build always contains multi-command executable; fail
            # clearly instead of silently launching the wrong binary.
            raise RuntimeError("Oversized server requires the v0.2 oversized-moe executable. Re-run bootstrap.")
    print("[redlite]", " ".join(_quote(x) for x in cmd))
    if dry_run:
        return 0
    return subprocess.call(cmd)


def run_bench(model: str, plan: Plan, prompt_tokens: int, gen_tokens: int, dry_run: bool = False) -> int:
    if plan.mode == "ssd-cpu":
        # OMR has no dedicated bench front-end; use deterministic completion timing.
        prompt = "The following is a deterministic local inference benchmark. " * max(1, prompt_tokens // 10)
        return run_completion(model, plan, prompt, gen_tokens, ["--temp", "0", "--seed", "42"], dry_run)
    bench = _first_existing([LLAMA_BIN / "llama-bench"])
    cmd = [
        str(bench), "-m", model, "-ngl", "999",
        "-p", str(prompt_tokens), "-n", str(gen_tokens),
        "-d", str(plan.context),
        "-b", str(plan.batch), "-ub", str(plan.ubatch),
        "-t", str(plan.threads),
        "-ctk", "q8_0", "-ctv", "q8_0", "-fa", "on",
        "-r", "3",
    ]
    print("[redlite]", " ".join(_quote(x) for x in cmd))
    if dry_run:
        return 0
    return subprocess.call(cmd)


def _quote(s: str) -> str:
    if not s or any(c.isspace() for c in s):
        return "'" + s.replace("'", "'\\''") + "'"
    return s
