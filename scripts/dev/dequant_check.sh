#!/usr/bin/env bash
# dev31: the native CPU dequantization of real tensors vs ggml's own to_float, bit for bit.
#
#   scripts/dev/dequant_check.sh MODEL [TENSOR ...]
#
# For each tensor (default: one tensor of every quant type the file uses outside F32), rows 0..7 and
# 777..784 are dequantized by `redlite-engine dequant` and by `redlite-ref-llama MODEL dequant` (pinned
# llama.cpp, oracle only) and compared with cmp. Prints "DEQUANT PARITY: YES" when every row is identical.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="$ROOT/.deps/redmetal"
MODEL="${1:?usage: $0 MODEL [TENSOR ...]}"; shift
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
if [[ $# -gt 0 ]]; then TENSORS=("$@"); else
  # the first tensor of each non-F32 type, in file order
  TENSORS=()
  while IFS= read -r t; do TENSORS+=("$t"); done < <(python3 - "$MODEL" "$ROOT" <<'PY'
import sys
sys.path.insert(0, sys.argv[2])
from redlite.expert_map import read_tensor_directory
seen = {}
for t in read_tensor_directory(sys.argv[1])[2]:
    if t.ggml_type != 0 and t.ggml_type not in seen and t.shape[0] % 256 == 0:
        seen[t.ggml_type] = t.name
print("\n".join(seen.values()))
PY
)
fi
OK=1
for T in "${TENSORS[@]}"; do
  for R in 0 777; do
    if "$BIN/redlite-engine" dequant "$MODEL" --tensor "$T" --row-first "$R" --rows 8 --out "$TMP/n.bin" >"$TMP/n.log" 2>&1 &&
       "$BIN/redlite-ref-llama" "$MODEL" dequant --tensor "$T" --row-first "$R" --rows 8 --out "$TMP/r.bin" >"$TMP/r.log" 2>&1 &&
       cmp -s "$TMP/n.bin" "$TMP/r.bin"; then
      echo "PASS  $T rows $R..$((R + 7)) $(grep -o '([A-Z0-9_]*)' "$TMP/r.log" | head -1) bit-identical"
    else
      echo "FAIL  $T rows $R..$((R + 7)): $(tail -1 "$TMP/n.log") | $(tail -1 "$TMP/r.log")"; OK=0
    fi
  done
done
[[ $OK == 1 ]] && echo "DEQUANT PARITY: YES" || { echo "DEQUANT PARITY: NO"; exit 1; }
