#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/.deps/redmetal"
mkdir -p "$OUT"

CC_BIN="${CC:-cc}"
COMMON=(
  "$ROOT/native/redlite_native_main.c"
  "$ROOT/native/redlite_native_gguf.c"
  "$ROOT/native/redlite_native_cache.c"
  "$ROOT/native/redlite_native_model.c"
  "$ROOT/native/redlite_native_tables.c"
  "$ROOT/native/redlite_native_reference.c"
  "$ROOT/native/redlite_native_router.c"
)
OFFLINE_TEST=(
  "$ROOT/native/redlite_native_offline_test.c"
  "$ROOT/native/redlite_native_gguf.c"
  "$ROOT/native/redlite_native_cache.c"
  "$ROOT/native/redlite_native_model.c"
  "$ROOT/native/redlite_native_router.c"
)
ROUTER_AUDIT=(
  "$ROOT/native/redlite_native_router_cli.c"
  "$ROOT/native/redlite_native_router.c"
)
SHARED_AUDIT=(
  "$ROOT/native/redlite_native_shared_cli.c"
  "$ROOT/native/redlite_native_shared.c"
  "$ROOT/native/redlite_native_router.c"
)
ROUTER_OFFLINE=(
  "$ROOT/native/redlite_native_router_offline_test.c"
  "$ROOT/native/redlite_native_router_exec.c"
)
ROUTER_PARITY_COMMON=(
  "$ROOT/native/redlite_native_routed_cli.c"
  "$ROOT/native/redlite_native_router.c"
  "$ROOT/native/redlite_native_router_exec.c"
  "$ROOT/native/redlite_native_gguf.c"
  "$ROOT/native/redlite_native_cache.c"
  "$ROOT/native/redlite_native_model.c"
  "$ROOT/native/redlite_native_tables.c"
  "$ROOT/native/redlite_native_reference.c"
)

if [[ "$(uname -s)" == "Darwin" ]]; then
  xcrun --sdk macosx clang \
    -O3 \
    -std=c11 \
    -D_FILE_OFFSET_BITS=64 \
    -D_POSIX_C_SOURCE=200809L \
    -Wall -Wextra -Wpedantic \
    -Wno-overlength-strings \
    -Wno-gnu-conditional-omitted-operand \
    -mcpu=native \
    -fobjc-arc \
    -I"$ROOT/native" \
    "${COMMON[@]}" \
    "$ROOT/native/redlite_native_metal.c" \
    "$ROOT/native/redmetal_topk.m" \
    -framework Foundation \
    -framework Metal \
    -lm \
    -o "$OUT/redlite-native"

  xcrun --sdk macosx clang \
    -O2 -std=c11 -D_FILE_OFFSET_BITS=64 -D_POSIX_C_SOURCE=200809L \
    -Wall -Wextra -Wpedantic -mcpu=native -I"$ROOT/native" \
    "${OFFLINE_TEST[@]}" \
    -o "$OUT/redlite-native-offline-test"

  xcrun --sdk macosx clang \
    -O2 -std=c11 -D_FILE_OFFSET_BITS=64 -D_POSIX_C_SOURCE=200809L \
    -Wall -Wextra -Wpedantic -mcpu=native -I"$ROOT/native" \
    "${ROUTER_AUDIT[@]}" \
    -o "$OUT/redlite-router-audit"

  xcrun --sdk macosx clang \
    -O2 -std=c11 -D_FILE_OFFSET_BITS=64 -D_POSIX_C_SOURCE=200809L \
    -Wall -Wextra -Wpedantic -mcpu=native -I"$ROOT/native" \
    "${SHARED_AUDIT[@]}" \
    -o "$OUT/redlite-shared-audit"

  xcrun --sdk macosx clang \
    -O2 -std=c11 -D_FILE_OFFSET_BITS=64 -D_POSIX_C_SOURCE=200809L \
    -Wall -Wextra -Wpedantic -mcpu=native -I"$ROOT/native" \
    "${ROUTER_OFFLINE[@]}" -lm \
    -o "$OUT/redlite-router-offline-test"

  xcrun --sdk macosx clang \
    -O3 -std=c11 -D_FILE_OFFSET_BITS=64 -D_POSIX_C_SOURCE=200809L \
    -Wall -Wextra -Wpedantic -Wno-overlength-strings -Wno-gnu-conditional-omitted-operand \
    -mcpu=native -fobjc-arc -I"$ROOT/native" \
    "${ROUTER_PARITY_COMMON[@]}" \
    "$ROOT/native/redlite_native_metal.c" \
    "$ROOT/native/redmetal_topk.m" \
    "$ROOT/native/redmetal_router.m" \
    -framework Foundation -framework Metal -lm \
    -o "$OUT/redlite-router"
else
  "$CC_BIN" \
    -O3 \
    -std=c11 \
    -D_FILE_OFFSET_BITS=64 \
    -D_POSIX_C_SOURCE=200809L \
    -Wall -Wextra -Wpedantic \
    -march=native \
    -I"$ROOT/native" \
    "${COMMON[@]}" \
    -lm \
    -o "$OUT/redlite-native"

  "$CC_BIN" \
    -O2 -std=c11 -D_FILE_OFFSET_BITS=64 -D_POSIX_C_SOURCE=200809L \
    -Wall -Wextra -Wpedantic -march=native -I"$ROOT/native" \
    "${OFFLINE_TEST[@]}" \
    -o "$OUT/redlite-native-offline-test"

  "$CC_BIN" \
    -O2 -std=c11 -D_FILE_OFFSET_BITS=64 -D_POSIX_C_SOURCE=200809L \
    -Wall -Wextra -Wpedantic -march=native -I"$ROOT/native" \
    "${ROUTER_AUDIT[@]}" \
    -o "$OUT/redlite-router-audit"

  "$CC_BIN" \
    -O2 -std=c11 -D_FILE_OFFSET_BITS=64 -D_POSIX_C_SOURCE=200809L \
    -Wall -Wextra -Wpedantic -march=native -I"$ROOT/native" \
    "${SHARED_AUDIT[@]}" \
    -o "$OUT/redlite-shared-audit"

  "$CC_BIN" \
    -O2 -std=c11 -D_FILE_OFFSET_BITS=64 -D_POSIX_C_SOURCE=200809L \
    -Wall -Wextra -Wpedantic -march=native -I"$ROOT/native" \
    "${ROUTER_OFFLINE[@]}" -lm \
    -o "$OUT/redlite-router-offline-test"
fi

"$OUT/redlite-native" selftest
"$OUT/redlite-native-offline-test"
"$OUT/redlite-router-audit" --help >/dev/null
"$OUT/redlite-shared-audit" --help >/dev/null
"$OUT/redlite-router-offline-test"
if [[ "$(uname -s)" == "Darwin" ]]; then
  "$OUT/redlite-router" --help >/dev/null
fi

echo "Built $OUT/redlite-native"
echo "Built $OUT/redlite-router-audit"
echo "Built $OUT/redlite-shared-audit"
if [[ "$(uname -s)" == "Darwin" ]]; then
  echo "Built $OUT/redlite-router"
fi
