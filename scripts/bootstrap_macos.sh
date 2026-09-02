#!/bin/bash
set -euo pipefail

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
DEPS="$ROOT/.deps"
JOBS="${REDLITE_JOBS:-$(sysctl -n hw.logicalcpu 2>/dev/null || echo 4)}"

LLAMA_REPO="https://github.com/ggml-org/llama.cpp.git"
LLAMA_PIN="7798007a29a90e3053e799394da48cf53a2f8e0f"
OMR_REPO="https://github.com/dimitripavlov/oversized-moe-runtime.git"
OMR_PIN="18815840f02bd860c2cbb5c6f0ff89e97a837229"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "error: DwarfStar Red Lite supports Apple Silicon macOS only." >&2
  exit 2
fi
for tool in git cmake clang; do
  command -v "$tool" >/dev/null || { echo "error: missing $tool" >&2; exit 2; }
done
mkdir -p "$DEPS"

checkout_pinned() {
  local repo="$1" dir="$2" pin="$3"
  if [[ ! -d "$dir/.git" ]]; then
    git clone --filter=blob:none "$repo" "$dir"
  fi
  git -C "$dir" fetch --depth 1 origin "$pin"
  git -C "$dir" checkout --detach "$pin"
}

echo "==> Building Metal resident engine (llama.cpp @ $LLAMA_PIN)"
checkout_pinned "$LLAMA_REPO" "$DEPS/llama.cpp" "$LLAMA_PIN"
cmake -S "$DEPS/llama.cpp" -B "$DEPS/llama.cpp/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=ON \
  -DGGML_ACCELERATE=ON \
  -DGGML_NATIVE=ON \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=ON
cmake --build "$DEPS/llama.cpp/build" --target llama-cli llama-server llama-bench -j "$JOBS"

echo "==> Building oversized CPU/SSD engine (Oversized MoE Runtime @ $OMR_PIN)"
checkout_pinned "$OMR_REPO" "$DEPS/oversized-moe-runtime" "$OMR_PIN"
cmake -S "$DEPS/oversized-moe-runtime" -B "$DEPS/oversized-moe-runtime/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=OFF \
  -DGGML_ACCELERATE=ON \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_SUBPROCESS=ON \
  -DLLAMA_BUILD_TESTS=ON
cmake --build "$DEPS/oversized-moe-runtime/build" \
  --target oversized-moe llama-completion llama-server \
  -j "$JOBS"

echo "==> Done"
echo "Run: $ROOT/bin/redlite doctor"
