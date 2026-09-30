#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/.deps/redmetal"
mkdir -p "$OUT"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "redlite-attention-block requires macOS Metal; skipping non-Darwin build"
  exit 0
fi

xcrun --sdk macosx clang \
  -O3 -std=c11 -D_FILE_OFFSET_BITS=64 -D_POSIX_C_SOURCE=200809L \
  -Wall -Wextra -Wpedantic -Wno-overlength-strings -Wno-gnu-conditional-omitted-operand -Wno-nullability-extension \
  -mcpu="${REDLITE_MCPU:-native}" -fobjc-arc -I"$ROOT/native" \
  "$ROOT/native/redlite_native_attention_block_cli.c" \
  "$ROOT/native/redlite_native_router.c" \
  "$ROOT/native/redlite_native_router_exec.c" \
  "$ROOT/native/redlite_native_shared.c" \
  "$ROOT/native/redlite_native_shared_exec.c" \
  "$ROOT/native/redlite_native_iq2_xxs.c" \
  "$ROOT/native/redlite_native_gguf.c" \
  "$ROOT/native/redlite_native_cache.c" \
  "$ROOT/native/redlite_native_model.c" \
  "$ROOT/native/redlite_native_tables.c" \
  "$ROOT/native/redlite_native_reference.c" \
  "$ROOT/native/redlite_native_metal.c" \
  "$ROOT/native/redmetal_topk.m" \
  "$ROOT/native/redmetal_router.m" \
  "$ROOT/native/redmetal_shared.m" \
  "$ROOT/native/redmetal_attention_proj.m" \
  "$ROOT/native/redmetal_block_ops.m" \
  -framework Foundation -framework Metal -lm \
  -o "$OUT/redlite-attention-block"

echo "Built $OUT/redlite-attention-block"
