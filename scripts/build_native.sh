#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/.deps/redmetal"
mkdir -p "$OUT"

CC_BIN="${CC:-cc}"
if [[ "$(uname -s)" == "Darwin" ]]; then
  NATIVE_FLAG="-mcpu=native"
else
  NATIVE_FLAG="-march=native"
fi

"$CC_BIN" \
  -O3 \
  -std=c11 \
  -D_FILE_OFFSET_BITS=64 \
  -Wall -Wextra -Wpedantic \
  "$NATIVE_FLAG" \
  -I"$ROOT/native" \
  "$ROOT/native/redlite_native_main.c" \
  "$ROOT/native/redlite_native_gguf.c" \
  "$ROOT/native/redlite_native_cache.c" \
  -o "$OUT/redlite-native"

"$OUT/redlite-native" selftest

echo "Built $OUT/redlite-native"
