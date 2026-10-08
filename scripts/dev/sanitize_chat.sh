#!/usr/bin/env bash
# dev26: one real chat turn under AddressSanitizer + UndefinedBehaviorSanitizer (macOS, Metal, model).
#
#   scripts/dev/sanitize_chat.sh MODEL [--cache-mib N] [--max-tokens N]
#
# Builds redlite-generate with -fsanitize=address,undefined into .deps/redmetal/sanitize/ and runs a
# chat turn whose prompt is long enough for two batched-prefill chunks (--batch 64): tokenizer, chat
# template, batched prefill (with the dev24 prefetch thread), the synchronous decode with the dev23
# prefetch, the sampler (greedy, then a seeded top-k/top-p/min-p draw) and the JSON stats. Any
# sanitizer report aborts the run. The greedy text must equal the non-sanitized build's.
# LeakSanitizer is unsupported on Apple platforms; the Metal framework itself is not instrumented.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
N="$ROOT/native"; OUT="$ROOT/.deps/redmetal/sanitize"
MODEL="${1:?usage: $0 MODEL [--cache-mib N] [--max-tokens N]}"; shift
CACHE=4096; MAXTOK=32
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cache-mib) CACHE="$2"; shift 2;;
    --max-tokens) MAXTOK="$2"; shift 2;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
done
[[ "$(uname -s)" == "Darwin" ]] || { echo "sanitize_chat: macOS only (Metal)"; exit 0; }
mkdir -p "$OUT"
xcrun --sdk "${REDLITE_SDK:-macosx}" clang -O1 -g -std=c11 -D_FILE_OFFSET_BITS=64 -D_POSIX_C_SOURCE=200809L \
  -Wall -Wextra -Wpedantic -Wno-overlength-strings -Wno-gnu-conditional-omitted-operand -Wno-nullability-extension \
  -fsanitize=address,undefined,float-cast-overflow -fno-sanitize-recover=all -fno-omit-frame-pointer \
  -fobjc-arc -I"$N" \
  "$N/redlite_native_generate_cli.c" "$N/redlite_native_sampler.c" \
  "$N/redlite_native_engine.c" "$N/redlite_native_engine_cpu.c" "$N/redlite_native_tokenizer.c" \
  "$N/redlite_native_gguf_dir.c" "$N/redlite_native_quant_cpu.c" "$N/redlite_native_iq3.c" "$N/redlite_native_layer_map.c" \
  "$N/redlite_native_router.c" "$N/redlite_native_router_exec.c" "$N/redlite_native_shared_exec.c" \
  "$N/redlite_native_iq2_xxs.c" "$N/redlite_native_gguf.c" "$N/redlite_native_cache.c" \
  "$N/redlite_native_model.c" "$N/redlite_native_tables.c" "$N/redlite_native_reference.c" \
  "$N/redlite_native_metal.c" "$N/redlite_native_statecache.c" "$N/redmetal_topk.m" "$N/redmetal_router.m" \
  "$N/redmetal_engine.m" "$N/redmetal_engine_prefill.m" \
  -framework Foundation -framework Metal -framework MetalPerformanceShaders -lm -lpthread -o "$OUT/redlite-generate"
export ASAN_OPTIONS="${ASAN_OPTIONS:-abort_on_error=1:detect_leaks=0}"
export UBSAN_OPTIONS="${UBSAN_OPTIONS:-print_stacktrace=1:halt_on_error=1}"
PROMPT="$(head -c 1400 "$ROOT/tests/fixtures/long_context_prompt.txt") Summarize the text above in two sentences."
COMMON=(--prompt "$PROMPT" --max-tokens "$MAXTOK" --cache-mib "$CACHE" --batch 64 --no-stream --json)
"$OUT/redlite-generate" "$MODEL" "${COMMON[@]}" >"$OUT/chat.greedy.txt" 2>"$OUT/chat.greedy.err"
"$ROOT/.deps/redmetal/redlite-generate" "$MODEL" "${COMMON[@]}" >"$OUT/chat.greedy.ref.txt" 2>/dev/null
cmp -s "$OUT/chat.greedy.txt" "$OUT/chat.greedy.ref.txt" || { echo "sanitized greedy text differs from the normal build"; exit 1; }
"$OUT/redlite-generate" "$MODEL" "${COMMON[@]}" --temperature 0.8 --top-k 40 --top-p 0.95 --min-p 0.05 --seed 7 \
  >"$OUT/chat.sampled.txt" 2>"$OUT/chat.sampled.err"
grep -q '"prompt_tokens"' "$OUT/chat.greedy.err" && grep -q '"prompt_tokens"' "$OUT/chat.sampled.err"
echo "prompt tokens: $(grep -o '"prompt_tokens":[0-9]*' "$OUT/chat.greedy.err" | cut -d: -f2), greedy == normal build"
echo "SANITIZED CHAT TURN: OK ($OUT)"
