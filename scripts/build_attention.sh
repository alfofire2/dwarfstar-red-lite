#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/.deps/redmetal"
mkdir -p "$OUT"

CC_BIN="${CC:-cc}"
COMMON=(
  "$ROOT/native/redlite_native_attention_cli.c"
  "$ROOT/native/redlite_native_shared_exec.c"
  "$ROOT/native/redlite_native_iq2_xxs.c"
)

if [[ "$(uname -s)" == "Darwin" ]]; then
  xcrun --sdk macosx clang \
    -O3 -std=c11 -D_FILE_OFFSET_BITS=64 -D_POSIX_C_SOURCE=200809L \
    -Wall -Wextra -Wpedantic -Wno-overlength-strings -Wno-gnu-conditional-omitted-operand \
    -mcpu="${REDLITE_MCPU:-native}" -fobjc-arc -I"$ROOT/native" \
    "${COMMON[@]}" \
    "$ROOT/native/redmetal_attention_proj.m" \
    -framework Foundation -framework Metal -lm \
    -o "$OUT/redlite-attention"
else
  "$CC_BIN" \
    -O2 -std=c11 -D_FILE_OFFSET_BITS=64 -D_POSIX_C_SOURCE=200809L \
    -Wall -Wextra -Wpedantic -march=native -I"$ROOT/native" \
    "${COMMON[@]}" -lm \
    -o "$OUT/redlite-attention"
fi

"$OUT/redlite-attention" --selftest

echo "Built $OUT/redlite-attention"
