#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/.deps/redmetal"
mkdir -p "$OUT"

CC_BIN="${CC:-cc}"
COMMON=(
  "$ROOT/native/redlite_native_main.c"
  "$ROOT/native/redlite_native_gguf.c"
  "$ROOT/native/redlite_native_cache.c"
  "$ROOT/native/redlite_native_model.c"
  "$ROOT/native/redlite_native_tables.c"
)

if [[ "$(uname -s)" == "Darwin" ]]; then
  xcrun --sdk macosx clang \
    -O3 \
    -std=c11 \
    -D_FILE_OFFSET_BITS=64 \
    -D_POSIX_C_SOURCE=200809L \
    -Wall -Wextra -Wpedantic \
    -mcpu=native \
    -fobjc-arc \
    -I"$ROOT/native" \
    "${COMMON[@]}" \
    "$ROOT/native/redlite_native_metal.c" \
    "$ROOT/native/redmetal_topk.m" \
    -framework Foundation \
    -framework Metal \
    -o "$OUT/redlite-native"
else
  "$CC_BIN" \
    -O3 \
    -std=c11 \
    -D_FILE_OFFSET_BITS=64 \
    -D_POSIX_C_SOURCE=200809L \
    -Wall -Wextra -Wpedantic \
    -march=native \
    -I"$ROOT/native" \
    "${COMMON[@]}" \
    -o "$OUT/redlite-native"
fi

"$OUT/redlite-native" selftest

echo "Built $OUT/redlite-native"
