#!/usr/bin/env bash
# Development-only: build the pinned-llama.cpp reference oracle (not part of `make native`).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LLAMA="${REDLITE_LLAMA_DIR:-$ROOT/.deps/llama.cpp}"   # a bootstrapped checkout outside .deps can be shared (CI runner)
BIN="$LLAMA/build/bin"
OUT="$ROOT/.deps/redmetal"
mkdir -p "$OUT"
if [[ ! -f "$BIN/libllama.dylib" ]]; then
  echo "pinned llama.cpp build not found at $BIN (run: redlite bootstrap, or set REDLITE_LLAMA_DIR)" >&2
  exit 2
fi
xcrun --sdk macosx clang++ -std=c++17 -O2 \
  -I"$LLAMA/include" -I"$LLAMA/ggml/include" \
  "$ROOT/scripts/dev/ref_llama/redlite_ref_llama.cpp" \
  -L"$BIN" -lllama -lggml -lggml-base \
  -Wl,-rpath,"$BIN" \
  -o "$OUT/redlite-ref-llama"
echo "Built $OUT/redlite-ref-llama"
