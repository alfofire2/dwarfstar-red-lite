#!/usr/bin/env bash
# Build and run the model-free native tests under AddressSanitizer + UndefinedBehaviorSanitizer.
# Portable C only (no Metal): runs on macOS (xcrun clang) and Linux (CC, default cc).
# Any sanitizer report aborts the test (-fno-sanitize-recover=all) and fails the script.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/.deps/redmetal/sanitize"
N="$ROOT/native"
mkdir -p "$OUT"

if [[ "$(uname -s)" == "Darwin" ]]; then
  CC_CMD=(xcrun --sdk macosx clang)
else
  CC_CMD=("${CC:-cc}")
fi

FLAGS=(-O1 -g -std=c11 -D_FILE_OFFSET_BITS=64 -D_POSIX_C_SOURCE=200809L
  -Wall -Wextra -Wpedantic -Wno-overlength-strings -I"$N"
  -fsanitize=address,undefined,float-cast-overflow -fno-sanitize-recover=all -fno-omit-frame-pointer)

build() {
  local name="$1"; shift
  "${CC_CMD[@]}" "${FLAGS[@]}" "$@" -lm -o "$OUT/$name"
}

NATIVE_MAIN=("$N/redlite_native_main.c" "$N/redlite_native_gguf.c" "$N/redlite_native_cache.c"
  "$N/redlite_native_model.c" "$N/redlite_native_tables.c" "$N/redlite_native_reference.c"
  "$N/redlite_native_router.c")
if [[ "$(uname -s)" == "Darwin" ]]; then
  # the macOS redlite-native also carries the Metal top-k path (as in build_native.sh)
  "${CC_CMD[@]}" "${FLAGS[@]}" -Wno-gnu-conditional-omitted-operand -Wno-nullability-extension \
    -fobjc-arc "${NATIVE_MAIN[@]}" \
    "$N/redlite_native_metal.c" "$N/redmetal_topk.m" \
    -framework Foundation -framework Metal -lm -o "$OUT/redlite-native"
else
  build redlite-native "${NATIVE_MAIN[@]}"
fi
build redlite-native-offline-test \
  "$N/redlite_native_offline_test.c" "$N/redlite_native_gguf.c" "$N/redlite_native_cache.c" \
  "$N/redlite_native_model.c" "$N/redlite_native_router.c"
build redlite-router-offline-test \
  "$N/redlite_native_router_offline_test.c" "$N/redlite_native_router_exec.c"
build redlite-shared-exec-offline-test \
  "$N/redlite_native_shared_exec_offline_test.c" "$N/redlite_native_shared_exec.c" \
  "$N/redlite_native_iq2_xxs.c"
build redlite-engine-offline-test \
  "$N/redlite_native_engine_offline_test.c" "$N/redlite_native_gguf_dir.c" \
  "$N/redlite_native_quant_cpu.c" "$N/redlite_native_shared_exec.c" \
  "$N/redlite_native_iq2_xxs.c" "$N/redlite_native_sampler.c"
build redlite-server-fake \
  "$N/redlite_native_server.c" "$N/redlite_native_server_fake.c" -pthread
build redlite-sampler-dist \
  "$N/redlite_native_sampler_dist_cli.c" "$N/redlite_native_sampler.c"
build redlite-gguf-fuzz \
  "$N/redlite_native_gguf_fuzz.c" "$N/redlite_native_gguf.c" "$N/redlite_native_gguf_dir.c"
if [[ "$(uname -s)" == "Darwin" ]]; then
  # redlite-engine for its model-free Metal kernel self-test (dev22-dev23 kernels vs the CPU reference)
  "${CC_CMD[@]}" "${FLAGS[@]}" -Wno-gnu-conditional-omitted-operand -Wno-nullability-extension -fobjc-arc \
    "$N/redlite_native_engine_cli.c" "$N/redlite_native_engine.c" "$N/redlite_native_engine_cpu.c" \
    "$N/redlite_native_tokenizer.c" "$N/redlite_native_gguf_dir.c" "$N/redlite_native_quant_cpu.c" \
    "$N/redlite_native_layer_map.c" "$N/redlite_native_router.c" "$N/redlite_native_router_exec.c" \
    "$N/redlite_native_shared_exec.c" "$N/redlite_native_iq2_xxs.c" "$N/redlite_native_gguf.c" \
    "$N/redlite_native_cache.c" "$N/redlite_native_model.c" "$N/redlite_native_tables.c" \
    "$N/redlite_native_reference.c" "$N/redlite_native_metal.c" "$N/redmetal_topk.m" "$N/redmetal_router.m" \
    "$N/redmetal_engine.m" "$N/redmetal_engine_prefill.m" "$N/redmetal_engine_selftest.m" \
    -framework Foundation -framework Metal -lm -lpthread -o "$OUT/redlite-engine"
fi

export ASAN_OPTIONS="${ASAN_OPTIONS:-detect_leaks=1:abort_on_error=1}"
export UBSAN_OPTIONS="${UBSAN_OPTIONS:-print_stacktrace=1:halt_on_error=1}"
# LeakSanitizer is unsupported on Apple platforms.
if [[ "$(uname -s)" == "Darwin" ]]; then ASAN_OPTIONS="${ASAN_OPTIONS/detect_leaks=1/detect_leaks=0}"; fi

"$OUT/redlite-native" selftest
"$OUT/redlite-native-offline-test"
"$OUT/redlite-router-offline-test"
"$OUT/redlite-shared-exec-offline-test"
"$OUT/redlite-engine-offline-test"
"$OUT/redlite-gguf-fuzz" --iterations "${REDLITE_FUZZ_ITERATIONS:-2000}"
"$OUT/redlite-server-fake" --selftest
if [[ "$(uname -s)" == "Darwin" ]]; then "$OUT/redlite-engine" kernel-selftest; fi
# OpenAI protocol tests against the sanitized HTTP server core (any report aborts the server and fails a test).
(cd "$ROOT" && REDLITE_SERVER_FAKE_BIN="$OUT/redlite-server-fake" PYTHONPATH="$ROOT" \
  python3 -m unittest discover -s tests -p test_native_server.py)

echo "Sanitized offline tests: OK ($OUT)"
