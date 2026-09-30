#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/.deps/redmetal"
mkdir -p "$OUT"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "redlite-deltanet-layer requires macOS Metal; skipping non-Darwin build"
  exit 0
fi

xcrun --sdk macosx clang \
  -O3 -std=c11 -D_FILE_OFFSET_BITS=64 -D_POSIX_C_SOURCE=200809L \
  -Wall -Wextra -Wpedantic -Wno-overlength-strings -Wno-gnu-conditional-omitted-operand \
  -mcpu=native -fobjc-arc -I"$ROOT/native" \
  "$ROOT/native/redlite_native_deltanet_layer_cli.c" \
  "$ROOT/native/redlite_native_shared_exec.c" \
  "$ROOT/native/redlite_native_iq2_xxs.c" \
  "$ROOT/native/redmetal_deltanet_proj_full.m" \
  "$ROOT/native/redmetal_deltanet_prestate.m" \
  "$ROOT/native/redmetal_deltanet_state.m" \
  "$ROOT/native/redmetal_deltanet_tail.m" \
  -framework Foundation -framework Metal -lm \
  -o "$OUT/redlite-deltanet-layer"

echo "Built $OUT/redlite-deltanet-layer"
