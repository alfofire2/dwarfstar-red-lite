#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/.deps/redmetal"
mkdir -p "$OUT"

if [[ "$(uname -s)" == "Darwin" ]]; then
  xcrun --sdk macosx clang \
    -O2 -std=c11 -D_FILE_OFFSET_BITS=64 -D_POSIX_C_SOURCE=200809L \
    -Wall -Wextra -Wpedantic -mcpu=native \
    "$ROOT/native/redlite_native_layer_audit_cli.c" \
    -o "$OUT/redlite-layer-audit"
else
  "${CC:-cc}" \
    -O2 -std=c11 -D_FILE_OFFSET_BITS=64 -D_POSIX_C_SOURCE=200809L \
    -Wall -Wextra -Wpedantic -march=native \
    "$ROOT/native/redlite_native_layer_audit_cli.c" \
    -o "$OUT/redlite-layer-audit"
fi

"$OUT/redlite-layer-audit" --selftest
"$OUT/redlite-layer-audit" --help >/dev/null

echo "Built $OUT/redlite-layer-audit"
