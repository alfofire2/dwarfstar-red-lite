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
SHARED_EXEC_OFFLINE=(
  "$ROOT/native/redlite_native_shared_exec_offline_test.c"
  "$ROOT/native/redlite_native_shared_exec.c"
  "$ROOT/native/redlite_native_iq2_xxs.c"
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
SHARED_PARITY_COMMON=(
  "$ROOT/native/redlite_native_shared_parity_cli.c"
  "$ROOT/native/redlite_native_shared.c"
  "$ROOT/native/redlite_native_shared_exec.c"
  "$ROOT/native/redlite_native_iq2_xxs.c"
)
Q6_PROBE_COMMON=(
  "$ROOT/native/redlite_native_q6_probe_cli.c"
  "$ROOT/native/redlite_native_shared.c"
  "$ROOT/native/redlite_native_shared_exec.c"
  "$ROOT/native/redlite_native_iq2_xxs.c"
)
FFN_PARITY_COMMON=(
  "$ROOT/native/redlite_native_ffn_parity_cli.c"
  "$ROOT/native/redlite_native_router.c"
  "$ROOT/native/redlite_native_router_exec.c"
  "$ROOT/native/redlite_native_shared.c"
  "$ROOT/native/redlite_native_shared_exec.c"
  "$ROOT/native/redlite_native_iq2_xxs.c"
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
    -Wno-gnu-conditional-omitted-operand -Wno-nullability-extension \
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
    -O2 -std=c11 -D_FILE_OFFSET_BITS=64 -D_POSIX_C_SOURCE=200809L \
    -Wall -Wextra -Wpedantic -mcpu=native -I"$ROOT/native" \
    "${SHARED_EXEC_OFFLINE[@]}" -lm \
    -o "$OUT/redlite-shared-exec-offline-test"

  xcrun --sdk macosx clang \
    -O3 -std=c11 -D_FILE_OFFSET_BITS=64 -D_POSIX_C_SOURCE=200809L \
    -Wall -Wextra -Wpedantic -Wno-overlength-strings -Wno-gnu-conditional-omitted-operand -Wno-nullability-extension \
    -mcpu=native -fobjc-arc -I"$ROOT/native" \
    "${ROUTER_PARITY_COMMON[@]}" \
    "$ROOT/native/redlite_native_metal.c" \
    "$ROOT/native/redmetal_topk.m" \
    "$ROOT/native/redmetal_router.m" \
    -framework Foundation -framework Metal -lm \
    -o "$OUT/redlite-router"

  xcrun --sdk macosx clang \
    -O3 -std=c11 -D_FILE_OFFSET_BITS=64 -D_POSIX_C_SOURCE=200809L \
    -Wall -Wextra -Wpedantic -Wno-overlength-strings -Wno-gnu-conditional-omitted-operand -Wno-nullability-extension \
    -mcpu=native -fobjc-arc -I"$ROOT/native" \
    "${SHARED_PARITY_COMMON[@]}" \
    "$ROOT/native/redmetal_shared.m" \
    -framework Foundation -framework Metal -lm \
    -o "$OUT/redlite-shared"

  xcrun --sdk macosx clang \
    -O3 -std=c11 -D_FILE_OFFSET_BITS=64 -D_POSIX_C_SOURCE=200809L \
    -Wall -Wextra -Wpedantic -Wno-overlength-strings -Wno-gnu-conditional-omitted-operand -Wno-nullability-extension \
    -mcpu=native -fobjc-arc -I"$ROOT/native" \
    "${Q6_PROBE_COMMON[@]}" \
    "$ROOT/native/redmetal_q6_probe.m" \
    -framework Foundation -framework Metal -lm \
    -o "$OUT/redlite-q6-probe"

  xcrun --sdk macosx clang \
    -O3 -std=c11 -D_FILE_OFFSET_BITS=64 -D_POSIX_C_SOURCE=200809L \
    -Wall -Wextra -Wpedantic -Wno-overlength-strings -Wno-gnu-conditional-omitted-operand -Wno-nullability-extension \
    -mcpu=native -fobjc-arc -I"$ROOT/native" \
    "${FFN_PARITY_COMMON[@]}" \
    "$ROOT/native/redlite_native_metal.c" \
    "$ROOT/native/redmetal_topk.m" \
    "$ROOT/native/redmetal_router.m" \
    "$ROOT/native/redmetal_shared.m" \
    -framework Foundation -framework Metal -lm \
    -o "$OUT/redlite-ffn"
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

  "$CC_BIN" \
    -O2 -std=c11 -D_FILE_OFFSET_BITS=64 -D_POSIX_C_SOURCE=200809L \
    -Wall -Wextra -Wpedantic -march=native -I"$ROOT/native" \
    "${SHARED_EXEC_OFFLINE[@]}" -lm \
    -o "$OUT/redlite-shared-exec-offline-test"
fi

"$OUT/redlite-native" selftest
"$OUT/redlite-native-offline-test"
"$OUT/redlite-router-audit" --help >/dev/null
"$OUT/redlite-shared-audit" --help >/dev/null
"$OUT/redlite-router-offline-test"
"$OUT/redlite-shared-exec-offline-test"
if [[ "$(uname -s)" == "Darwin" ]]; then
  "$OUT/redlite-router" --help >/dev/null
  "$OUT/redlite-shared" --help >/dev/null
  "$OUT/redlite-q6-probe" --help >/dev/null
  "$OUT/redlite-ffn" --help >/dev/null
fi

echo "Built $OUT/redlite-native"
echo "Built $OUT/redlite-router-audit"
echo "Built $OUT/redlite-shared-audit"
if [[ "$(uname -s)" == "Darwin" ]]; then
  echo "Built $OUT/redlite-router"
  echo "Built $OUT/redlite-shared"
  echo "Built $OUT/redlite-q6-probe"
  echo "Built $OUT/redlite-ffn"
fi
