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
)

if [[ "$(uname -s)" == "Darwin" ]]; then
  xcrun --sdk macosx clang "${FLAGS[@]}" -mcpu=native "${OFFLINE[@]}" -lm -o "$OUT/redlite-engine-offline-test"
else
  "$CC_BIN" "${FLAGS[@]}" -march=native "${OFFLINE[@]}" -lm -o "$OUT/redlite-engine-offline-test"
fi

"$OUT/redlite-engine-offline-test"
echo "Built $OUT/redlite-engine-offline-test"
