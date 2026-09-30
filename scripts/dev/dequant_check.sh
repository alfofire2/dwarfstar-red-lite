#!/usr/bin/env bash
# dev31: the native CPU dequantization of real tensors vs ggml's own to_float, bit for bit.
#
#   scripts/dev/dequant_check.sh MODEL [TENSOR ...]
#
# For each tensor (default: one tensor of every quant type the file uses except F32 and the IQ2_XS / IQ1_M
# experts of the reference GGUF, whose CPU oracle is a row dot, not a dequantizer), rows 0..7 and
# 777..784 are dequantized by `redlite-engine dequant` and by `redlite-ref-llama MODEL dequant` (pinned
# llama.cpp, oracle only) and compared with cmp. Prints "DEQUANT PARITY: YES" when every row is identical.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="$ROOT/.deps/redmetal"
MODEL="${1:?usage: $0 MODEL [TENSOR ...]}"; shift
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
if [[ $# -gt 0 ]]; then TENSORS=(); for t in "$@"; do TENSORS+=("$t 800"); done; else
  # the first tensor of each non-F32 type, in file order
  TENSORS=()
  while IFS= read -r t; do TENSORS+=("$t"); done < <(python3 - "$MODEL" "$ROOT" <<'PY'
import sys
sys.path.insert(0, sys.argv[2])
from redlite.expert_map import read_tensor_directory
seen = {}
# 17 IQ2_XS and 29 IQ1_M (routed experts of the reference GGUF) have row-dot CPU oracles, not dequantizers
for t in read_tensor_directory(sys.argv[1])[2]:
    if t.ggml_type not in (0, 17, 29) and t.ggml_type not in seen and t.shape[0] % 256 == 0:
        rows = 1
        for d in t.shape[1:]:
            rows *= d
        seen[t.ggml_type] = f"{t.name} {rows}"
print("\n".join(seen.values()))
PY
)
fi
OK=1
for TR in "${TENSORS[@]}"; do
  T="${TR% *}"; ROWS="${TR##* }"
  LAST=$((ROWS - 8)); [[ $LAST -gt 777 ]] && LAST=777; [[ $LAST -lt 0 ]] && LAST=0
  for R in 0 "$LAST"; do
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
