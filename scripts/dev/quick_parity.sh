#!/usr/bin/env bash
# Development gate for kernel/path changes (a subset of regress_m4.sh, same commands):
#   engine.parity            48-layer CPU oracle vs Metal, synchronous path (1 GiB cache)
#   engine.parity.gpu_routed CPU oracle vs the GPU-routed decode (22 GiB, full residency, --repeat 2)
#   logits.vs_llama          native logits vs the pinned llama.cpp dump from the last regress_m4.sh run
#   greedy.gpu_routed        24 greedy tokens with every expert resident == the llama.cpp greedy ids
#   long_context (optional)  --long: 1100-token batched prefill + 100 steps vs llama.cpp (chunk --batch N, default 1100)
# Needs .deps/regress/{ref.bin,ref.long.bin,gen.ref.txt} from a previous full regress_m4.sh run.
#   scripts/dev/quick_parity.sh MODEL [--long] [--batch N]
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="$ROOT/.deps/redmetal"; OUT="$ROOT/.deps/quick"
MODEL="${1:?usage: $0 MODEL [--long] [--batch N]}"; shift
# dev38: the llama.cpp dumps of the same model (regress_m4.sh writes non-reference models to .deps/regress-<name>)
MODEL_BASE="$(basename "$MODEL" .gguf)"
if [[ "$MODEL_BASE" == "Qwen_Qwen3-Next-80B-A3B-Instruct-IQ2_XXS" ]]; then LOG="$ROOT/.deps/regress"; else LOG="$ROOT/.deps/regress-$MODEL_BASE"; fi
LONG=0; BATCH=1100
while [[ $# -gt 0 ]]; do case "$1" in --long) LONG=1; shift;; --batch) BATCH="$2"; shift 2;; *) echo "unknown $1"; exit 2;; esac; done
mkdir -p "$OUT"
rm -f "$OUT"/*.bin "$OUT"/*.txt "$OUT"/*.log   # a crashed run must never be compared against a stale dump
FAIL=0
check() { # name pattern cmd...
  local name="$1" pat="$2"; shift 2
  if "$@" >"$OUT/$name.log" 2>&1 && grep -q -- "$pat" "$OUT/$name.log"; then echo "PASS  $name"; else echo "FAIL  $name ($OUT/$name.log)"; FAIL=1; fi
}
check engine.parity "MULTI-TOKEN ENGINE PARITY: YES" "$BIN/redlite-engine" parity "$MODEL" --tokens 9707,11,1879,0,785,12884 --cache-mib 1024 --context 64
check engine.parity.gpu_routed "MULTI-TOKEN ENGINE PARITY: YES" "$BIN/redlite-engine" parity "$MODEL" --tokens 9707,11,1879,0,785,12884 --cache-mib full --context 64 --repeat 2
grep -q "GPU-routed tokens     : 12 speculative" "$OUT/engine.parity.gpu_routed.log" || { echo "FAIL  engine.parity.gpu_routed.count"; FAIL=1; }
"$BIN/redlite-engine" logits "$MODEL" --tokens 9707,11,1879 --backend gpu --out "$OUT/native.bin" --cache-mib 1024 --context 64 >"$OUT/logits.native.log" 2>&1
check logits.vs_llama "ORACLE LOGITS PARITY: YES" python3 "$ROOT/scripts/dev/compare_dumps.py" "$OUT/native.bin" "$LOG/ref.bin"
PROMPT="Explain in one sentence why the sky is blue."
"$BIN/redlite-generate" "$MODEL" --prompt "$PROMPT" --max-tokens 24 --cache-mib full --no-stream --tokens-out "$OUT/gen.gpu.txt" >"$OUT/gen.gpu.log" 2>&1
if [[ -f "$LOG/gen.native.txt" ]] && cmp -s "$OUT/gen.gpu.txt" "$LOG/gen.native.txt"; then echo "PASS  greedy.gpu_routed (== regress greedy, identical to llama.cpp)"; else echo "FAIL  greedy.gpu_routed"; FAIL=1; fi
if [[ "$LONG" == 1 ]]; then
  IDS="$("$BIN/redlite-engine" tokenize "$MODEL" --text "$(cat "$ROOT/tests/fixtures/long_context_prompt.txt")" --no-special | head -1 | cut -d',' -f1-1200)"
  "$BIN/redlite-engine" logits "$MODEL" --tokens "$IDS" --backend gpu --out "$OUT/native.long.bin" --dump-from 1100 --batch "$BATCH" --cache-mib 4096 --context 1536 >"$OUT/logits.long.native.log" 2>&1
  check "long_context.batch$BATCH" "ORACLE LOGITS PARITY: YES" python3 "$ROOT/scripts/dev/compare_dumps.py" "$OUT/native.long.bin" "$LOG/ref.long.bin" --max-logit-abs 2.0 --max-kl 2e-2
fi
exit $FAIL
