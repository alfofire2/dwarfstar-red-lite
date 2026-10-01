#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "Red Metal can only be built on macOS." >&2
  exit 1
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/.deps/redmetal"
mkdir -p "$OUT"

xcrun --sdk macosx clang \
  -O2 \
  -fobjc-arc \
  -dynamiclib \
  -I"$ROOT/native" \
  "$ROOT/native/redmetal.m" \
  "$ROOT/native/redmetal_exec.m" \
  "$ROOT/native/redmetal_quant.m" \
  "$ROOT/native/redmetal_ffn.m" \
  "$ROOT/native/redmetal_topk.m" \
  "$ROOT/native/redlite_native_iq3.c" \
  -framework Foundation \
  -framework Metal \
  -o "$OUT/libredmetal.dylib"

echo "Built $OUT/libredmetal.dylib"
