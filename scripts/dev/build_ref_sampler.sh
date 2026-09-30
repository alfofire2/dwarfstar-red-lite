#!/usr/bin/env bash
# Development-only: build the pinned-llama.cpp sampler oracle (redlite-ref-sampler).
# Needs only libllama from the pinned checkout (no model); works on macOS and Linux.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LLAMA="${REDLITE_LLAMA_DIR:-$ROOT/.deps/llama.cpp}"
BIN="$LLAMA/build/bin"
OUT="$ROOT/.deps/redmetal"
mkdir -p "$OUT"
if [[ "$(uname -s)" == "Darwin" ]]; then
  LIB="$BIN/libllama.dylib"; CXX_CMD=(xcrun --sdk macosx clang++)
else
  LIB="$BIN/libllama.so"; CXX_CMD=("${CXX:-c++}")
fi
if [[ ! -f "$LIB" ]]; then
  echo "pinned llama.cpp build not found at $BIN (run: redlite bootstrap, or set REDLITE_LLAMA_DIR)" >&2
  exit 2
fi
"${CXX_CMD[@]}" -std=c++17 -O2 -Wall -Wextra \
  -I"$LLAMA/include" -I"$LLAMA/ggml/include" \
  "$ROOT/scripts/dev/ref_llama/redlite_ref_sampler.cpp" \
  -L"$BIN" -lllama -lggml -lggml-base \
  -Wl,-rpath,"$BIN" \
  -o "$OUT/redlite-ref-sampler"
echo "Built $OUT/redlite-ref-sampler"
