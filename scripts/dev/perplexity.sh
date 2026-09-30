#!/usr/bin/env bash
# dev31: perplexity of one or more GGUF files with the pinned llama.cpp (oracle only, never a runtime).
#
#   scripts/dev/perplexity.sh MODEL.gguf [MODEL2.gguf ...] [--ctx 512]
#
# Text: tests/fixtures/perplexity_corpus.txt, frozen (the docs/*.md of commit 8cbdffe concatenated in
# name order; sha256 6948b2c3...). The same text, context and llama.cpp build are used for every file,
# so the numbers compare the quantizations, not the text. llama-perplexity is built on demand from the
# bootstrapped checkout into its own tree (.deps/llama.cpp/build-ppl, same flags as bootstrap_macos.sh,
# OpenSSL/curl off) so the oracle libraries in build/ are not touched.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LLAMA="${REDLITE_LLAMA_DIR:-$ROOT/.deps/llama.cpp}"
BUILD="$LLAMA/build-ppl"
CORPUS="$ROOT/tests/fixtures/perplexity_corpus.txt"
CTX=512
MODELS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ctx) CTX="$2"; shift 2;;
    *) MODELS+=("$1"); shift;;
  esac
done
[[ ${#MODELS[@]} -gt 0 ]] || { echo "usage: $0 MODEL.gguf [MODEL2.gguf ...] [--ctx N]" >&2; exit 2; }
if [[ ! -x "$BUILD/bin/llama-perplexity" ]]; then
  cmake -S "$LLAMA" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DGGML_ACCELERATE=ON -DGGML_NATIVE=ON \
    -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_OPENSSL=OFF -DLLAMA_CURL=OFF >/dev/null
  cmake --build "$BUILD" --target llama-perplexity -j "${REDLITE_JOBS:-8}" >/dev/null
fi
echo "llama.cpp $(git -C "$LLAMA" rev-parse --short HEAD), corpus sha256 $(shasum -a 256 "$CORPUS" | cut -c1-12), ctx $CTX"
for M in "${MODELS[@]}"; do
  LOG="$ROOT/.deps/ppl.$(basename "$M" .gguf).ctx$CTX.log"
  "$BUILD/bin/llama-perplexity" -m "$M" -f "$CORPUS" -c "$CTX" -ngl 999 >"$LOG" 2>&1
  echo "$(basename "$M"): $(grep -o 'Final estimate: PPL = .*' "$LOG" || echo "no estimate, see $LOG")"
done
