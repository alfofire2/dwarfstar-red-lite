#!/usr/bin/env bash
# Build the native OpenAI-compatible HTTP server:
#   redlite-server-fake  server core + deterministic echo backend (protocol test tool, every OS)
#   redlite-server       server core + rl_engine (Metal on macOS; CPU oracle only elsewhere)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/.deps/redmetal"
N="$ROOT/native"
mkdir -p "$OUT"

FLAGS=(-std=c11 -D_FILE_OFFSET_BITS=64 -D_POSIX_C_SOURCE=200809L -Wall -Wextra -Wpedantic -I"$N")
ENGINE=(
  "$N/redlite_native_engine.c"
  "$N/redlite_native_engine_cpu.c"
  "$N/redlite_native_tokenizer.c"
  "$N/redlite_native_sampler.c"
  "$N/redlite_native_gguf_dir.c"
  "$N/redlite_native_quant_cpu.c"
  "$N/redlite_native_layer_map.c"
  "$N/redlite_native_router.c"
  "$N/redlite_native_router_exec.c"
  "$N/redlite_native_shared_exec.c"
  "$N/redlite_native_iq2_xxs.c"
  "$N/redlite_native_gguf.c"
  "$N/redlite_native_cache.c"
  "$N/redlite_native_model.c"
  "$N/redlite_native_tables.c"
  "$N/redlite_native_reference.c"
)

if [[ "$(uname -s)" == "Darwin" ]]; then
  xcrun --sdk macosx clang -O2 "${FLAGS[@]}" -mcpu=native \
    "$N/redlite_native_server.c" "$N/redlite_native_server_fake.c" \
    -lm -o "$OUT/redlite-server-fake"
  xcrun --sdk macosx clang \
    -O3 "${FLAGS[@]}" -Wno-overlength-strings -Wno-gnu-conditional-omitted-operand -Wno-nullability-extension \
    -mcpu=native -fobjc-arc \
    "$N/redlite_native_server_cli.c" "$N/redlite_native_server.c" \
    "${ENGINE[@]}" "$N/redlite_native_metal.c" \
    "$N/redmetal_topk.m" "$N/redmetal_router.m" "$N/redmetal_engine.m" "$N/redmetal_engine_prefill.m" \
    -framework Foundation -framework Metal -lm -lpthread \
    -o "$OUT/redlite-server"
else
  CC_BIN="${CC:-cc}"
  "$CC_BIN" -O2 "${FLAGS[@]}" -march=native \
    "$N/redlite_native_server.c" "$N/redlite_native_server_fake.c" \
    -lm -o "$OUT/redlite-server-fake"
  "$CC_BIN" -O2 "${FLAGS[@]}" -Wno-overlength-strings -march=native \
    "$N/redlite_native_server_cli.c" "$N/redlite_native_server.c" \
    "${ENGINE[@]}" \
    -lm -lpthread -o "$OUT/redlite-server"
fi

"$OUT/redlite-server-fake" --selftest
"$OUT/redlite-server" --help >/dev/null
echo "Built $OUT/redlite-server-fake"
echo "Built $OUT/redlite-server"
