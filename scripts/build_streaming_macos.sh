#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/.deps/red-stream"
mkdir -p "$OUT"
if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "red-stream requires Apple Silicon macOS" >&2
  exit 2
fi
clang++ -std=c++17 -O3 -fobjc-arc \
  -framework Foundation -framework Metal \
  -pthread \
  "$ROOT/native/red_stream_cache.mm" "$ROOT/native/red_stream_probe.mm" \
  -o "$OUT/red-stream-probe"
echo "Built $OUT/red-stream-probe"
