#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/.deps/redmetal"
mkdir -p "$OUT"

CC_BIN="${CC:-cc}"
FLAGS=(-O2 -std=c11 -D_FILE_OFFSET_BITS=64 -D_POSIX_C_SOURCE=200809L -Wall -Wextra -Wpedantic -I"$ROOT/native")
OFFLINE=(
  "$ROOT/native/redlite_native_engine_offline_test.c"
  "$ROOT/native/redlite_native_gguf_dir.c"
  "$ROOT/native/redlite_native_quant_cpu.c"
  "$ROOT/native/redlite_native_shared_exec.c"
  "$ROOT/native/redlite_native_iq2_xxs.c"
  "$ROOT/native/redlite_native_sampler.c"
)
ENGINE=(
  "$ROOT/native/redlite_native_engine_cli.c"
  "$ROOT/native/redlite_native_engine.c"
  "$ROOT/native/redlite_native_engine_cpu.c"
  "$ROOT/native/redlite_native_tokenizer.c"
  "$ROOT/native/redlite_native_gguf_dir.c"
  "$ROOT/native/redlite_native_quant_cpu.c"
  "$ROOT/native/redlite_native_layer_map.c"
  "$ROOT/native/redlite_native_router.c"
  "$ROOT/native/redlite_native_router_exec.c"
  "$ROOT/native/redlite_native_shared_exec.c"
  "$ROOT/native/redlite_native_iq2_xxs.c"
  "$ROOT/native/redlite_native_gguf.c"
  "$ROOT/native/redlite_native_cache.c"
  "$ROOT/native/redlite_native_model.c"
  "$ROOT/native/redlite_native_tables.c"
  "$ROOT/native/redlite_native_reference.c"
  "$ROOT/native/redlite_native_metal.c"
)

if [[ "$(uname -s)" == "Darwin" ]]; then
  xcrun --sdk macosx clang "${FLAGS[@]}" -mcpu=native "${OFFLINE[@]}" -lm -o "$OUT/redlite-engine-offline-test"
  xcrun --sdk macosx clang \
    -O3 -std=c11 -D_FILE_OFFSET_BITS=64 -D_POSIX_C_SOURCE=200809L \
    -Wall -Wextra -Wpedantic -Wno-overlength-strings -Wno-gnu-conditional-omitted-operand -Wno-nullability-extension \
    -mcpu=native -fobjc-arc -I"$ROOT/native" \
    "${ENGINE[@]}" \
    "$ROOT/native/redmetal_topk.m" \
    "$ROOT/native/redmetal_router.m" \
    "$ROOT/native/redmetal_engine.m" \
    "$ROOT/native/redmetal_engine_prefill.m" \
    "$ROOT/native/redmetal_engine_selftest.m" \
    -framework Foundation -framework Metal -lm -lpthread \
    -o "$OUT/redlite-engine"
  "$OUT/redlite-engine" --help >/dev/null || true
  "$OUT/redlite-engine" kernel-selftest
  GEN=("${ENGINE[@]:1}")
  xcrun --sdk macosx clang \
    -O3 -std=c11 -D_FILE_OFFSET_BITS=64 -D_POSIX_C_SOURCE=200809L \
    -Wall -Wextra -Wpedantic -Wno-overlength-strings -Wno-gnu-conditional-omitted-operand -Wno-nullability-extension \
    -mcpu=native -fobjc-arc -I"$ROOT/native" \
    "$ROOT/native/redlite_native_generate_cli.c" \
    "$ROOT/native/redlite_native_sampler.c" \
    "${GEN[@]}" \
    "$ROOT/native/redmetal_topk.m" \
    "$ROOT/native/redmetal_router.m" \
    "$ROOT/native/redmetal_engine.m" \
    "$ROOT/native/redmetal_engine_prefill.m" \
    -framework Foundation -framework Metal -lm -lpthread \
    -o "$OUT/redlite-generate"
  "$OUT/redlite-generate" --help >/dev/null
else
  "$CC_BIN" "${FLAGS[@]}" -march=native "${OFFLINE[@]}" -lm -o "$OUT/redlite-engine-offline-test"
fi

"$OUT/redlite-engine-offline-test"
echo "Built $OUT/redlite-engine-offline-test"
# model-free sampler distribution tool (compared with llama.cpp by scripts/dev/compare_sampler.py)
if [[ "$(uname -s)" == "Darwin" ]]; then SD_CC=(xcrun --sdk macosx clang -mcpu=native); else SD_CC=("$CC_BIN" -march=native); fi
"${SD_CC[@]}" "${FLAGS[@]}" "$ROOT/native/redlite_native_sampler_dist_cli.c" "$ROOT/native/redlite_native_sampler.c" \
  -lm -o "$OUT/redlite-sampler-dist"
echo "Built $OUT/redlite-sampler-dist"
if [[ "$(uname -s)" == "Darwin" ]]; then echo "Built $OUT/redlite-engine"; echo "Built $OUT/redlite-generate"; fi
