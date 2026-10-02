#!/bin/bash
# 24 GiB Mac field session (M4 Pro): build, full regression, 4 GiB cache A/B, bounded-cache benchmark, state restart.
# Run from the repo root after `make bootstrap`; set REDLITE_SDK if the Command Line Tools SDK is newer than their linker.
# Everything is logged to .deps/m4pro-session-<date>/; nothing here needs more than the IQ2_XXS GGUF.
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
M="${MODEL:-models/Qwen_Qwen3-Next-80B-A3B-Instruct-IQ2_XXS.gguf}"
D=".deps/session-$(date +%Y%m%d-%H%M)"; mkdir -p "$D"
echo "machine: $(sysctl -n machdep.cpu.brand_string) $(($(sysctl -n hw.memsize) / 1073741824)) GiB, commit $(git rev-parse --short HEAD)" | tee "$D/README.txt"
[ -f "$M" ] || { echo "missing $M (./bin/redlite download 24gb --dir models)"; exit 1; }
make native > "$D/native.log" 2>&1 && echo "native build: OK" | tee -a "$D/README.txt"
[ -x .deps/redmetal/redlite-ref-llama ] || echo "note: no llama.cpp oracle (scripts/dev/build_ref_llama.sh); regress skips those checks" | tee -a "$D/README.txt"
# 1. correctness: the full regression suite (llama.cpp oracle when bootstrapped)
scripts/regress_m4.sh "$M" > "$D/regress.log" 2>&1; grep "regression summary" "$D/regress.log" | tee -a "$D/README.txt"
# 2. 4 GiB expert cache A/B: dev37 slot classes, dev46 cache bias / F_NOCACHE, prefetch
COOL=60 scripts/dev/small_mac_ab.sh "$M" default uniform bias05 nocache noprefetch 2>&1 | tee "$D/small_mac_ab.log" | tee -a "$D/README.txt"
# 3. bounded-cache benchmark with the release method (decode, prefill 1100/8192 at 4 GiB, default 2048 chunks)
for o in decode4 prefill4; do scripts/dev/bench_m4.sh "$M" --reps 3 --only $o --cool 90 2>&1 | grep -E "median|commit"; done | tee "$D/bench.log" | tee -a "$D/README.txt"
# 4. dev43 state checkpoints across a server restart (4212-token prompt, 2 GiB cache)
python3 scripts/dev/server_check.py "$M" --state-restart 2>&1 | tail -3 | tee "$D/state_restart.log" | tee -a "$D/README.txt"
echo "done: $D"
