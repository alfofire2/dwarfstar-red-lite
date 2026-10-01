#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/.deps/redmetal"
mkdir -p "$OUT"

if [[ "$(uname -s)" == "Darwin" ]]; then
  xcrun --sdk macosx clang \
    -O2 -std=c11 -D_FILE_OFFSET_BITS=64 -D_POSIX_C_SOURCE=200809L \
    -Wall -Wextra -Wpedantic -Wno-overlength-strings -Wno-gnu-conditional-omitted-operand -mcpu="${REDLITE_MCPU:-native}" \
    "$ROOT/native/redlite_native_deltanet_prestate_cli.c" \
    "$ROOT/native/redmetal_deltanet_prestate.m" \
    -framework Foundation -framework Metal -lm \
    -o "$OUT/redlite-deltanet-prestate"
else
  "${CC:-cc}" \
    -O2 -std=c11 -D_FILE_OFFSET_BITS=64 -D_POSIX_C_SOURCE=200809L \
    -Wall -Wextra -Wpedantic -march=native \
    "$ROOT/native/redlite_native_deltanet_prestate_cli.c" \
    -lm -o "$OUT/redlite-deltanet-prestate"
fi

"$OUT/redlite-deltanet-prestate" --selftest
"$OUT/redlite-deltanet-prestate" --help >/dev/null || true

echo "Built $OUT/redlite-deltanet-prestate"
