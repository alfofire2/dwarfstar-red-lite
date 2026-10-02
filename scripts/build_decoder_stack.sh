#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/.deps/redmetal"
mkdir -p "$OUT"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "redlite-decoder-stack requires macOS Metal; skipping non-Darwin build"
  exit 0
fi

xcrun --sdk "${REDLITE_SDK:-macosx}" clang \
  -O3 -std=c11 -D_FILE_OFFSET_BITS=64 -D_POSIX_C_SOURCE=200809L \
  -Wall -Wextra -Wpedantic -Wno-overlength-strings -Wno-gnu-conditional-omitted-operand -Wno-nullability-extension \
  -mcpu="${REDLITE_MCPU:-native}" -fobjc-arc -I"$ROOT/native" \
  "$ROOT/native/redlite_native_decoder_stack_cli.c" \
  "$ROOT/native/redlite_native_decoder_stack_runtime.c" \
  "$ROOT/native/redlite_native_layer_map.c" \
  "$ROOT/native/redlite_native_recurrent_block_runtime.c" \
  "$ROOT/native/redlite_native_attention_block_runtime.c" \
  "$ROOT/native/redlite_native_router.c" \
  "$ROOT/native/redlite_native_router_exec.c" \
  "$ROOT/native/redlite_native_shared.c" \
  "$ROOT/native/redlite_native_shared_exec.c" \
  "$ROOT/native/redlite_native_iq2_xxs.c" \
  "$ROOT/native/redlite_native_gguf.c" \
  "$ROOT/native/redlite_native_cache.c" \
  "$ROOT/native/redlite_native_model.c" \
  "$ROOT/native/redlite_native_tables.c" \
  "$ROOT/native/redlite_native_reference.c" "$ROOT/native/redlite_native_iq3.c" \
  "$ROOT/native/redlite_native_metal.c" \
  "$ROOT/native/redmetal_topk.m" \
  "$ROOT/native/redmetal_router.m" \
  "$ROOT/native/redmetal_shared.m" \
  "$ROOT/native/redmetal_deltanet_proj_full.m" \
  "$ROOT/native/redmetal_deltanet_prestate.m" \
  "$ROOT/native/redmetal_deltanet_state.m" \
  "$ROOT/native/redmetal_deltanet_tail.m" \
  "$ROOT/native/redmetal_attention_proj.m" \
  "$ROOT/native/redmetal_block_ops.m" \
  -framework Foundation -framework Metal -lm \
  -o "$OUT/redlite-decoder-stack"

"$OUT/redlite-decoder-stack" --help >/dev/null

echo "Built $OUT/redlite-decoder-stack"
