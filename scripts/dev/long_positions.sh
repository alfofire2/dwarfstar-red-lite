#!/usr/bin/env bash
# dev26: logits parity with the pinned llama.cpp, and throughput, at long positions.
#
#   scripts/dev/long_positions.sh MODEL [--positions 4096,8192] [--batch N] [--cache-mib N] [--no-bench | --bench-only]
#
# Prompt: the token ids of the frozen tests/fixtures/long_context_prompt.txt (1397 ids), repeated
# and cut to P+100 ids. That is deterministic, and it needs no new fixture text.
# For each position P:
#   parity : native batched prefill of the first P ids (chunks of --batch, default 512, --cache-mib
#            default 4096), then 100 decode steps; logits of positions P..P+99 compared with
#            redlite-ref-llama on the same ids. Argmax agreement and KL <= 2e-2 as in
#            logits.long_context_vs_llama; the max-logit bound is 4.0 instead of 2.0: on this prompt
#            native token-by-token vs native batched prefill (no llama.cpp) already differ by 2.62 at
#            position 4096 (dev26 record), so 2.0 cannot separate a defect from rounding growth there.
#            At P >= 8192 the bound is 5.0: the 0.3.0 runtime's own token-by-token path differs by 4.29 from
#            llama.cpp and by 4.45 from its own batched prefill there (22 GiB cache, dev30 record), so the
#            native self-consistency floor is above 4.0. Argmax and KL gates are the same at every position.
#   bench  : redlite-generate --raw on the fixture text repeated to at least P tokens, 128 greedy
#            tokens, --json; prefill and decode tok/s at 4 GiB and at 22 GiB (22 GiB only with
#            >= 40 GiB of RAM). Single runs with a 90 s pause before each; not medians.
# Needs .deps/redmetal/redlite-ref-llama (scripts/dev/build_ref_llama.sh). The llama.cpp dump of each
# position is cached in .deps/positions/ref.P.bin.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="$ROOT/.deps/redmetal"; OUT="$ROOT/.deps/positions"
MODEL="${1:?usage: $0 MODEL [--positions 4096,8192] [--batch N] [--cache-mib N] [--no-bench]}"; shift
POSITIONS="4096,8192"; BATCH=512; CACHE=4096; BENCH=1; PARITY=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --positions) POSITIONS="$2"; shift 2;;
    --batch) BATCH="$2"; shift 2;;
    --cache-mib) CACHE="$2"; shift 2;;
    --no-bench) BENCH=0; shift;;
    --bench-only) PARITY=0; shift;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
done
mkdir -p "$OUT"
FIXTURE="$ROOT/tests/fixtures/long_context_prompt.txt"
BASE="$("$BIN/redlite-engine" tokenize "$MODEL" --text "$(cat "$FIXTURE")" --no-special | head -1)"
FAIL=0
for P in ${POSITIONS//,/ }; do
  [[ "$PARITY" == 1 ]] || break
  N=$((P + 100)); CTX=$((P + 256))
  IDS="$(python3 -c "import sys; b=sys.argv[1].split(','); n=int(sys.argv[2]); print(','.join((b*(n//len(b)+1))[:n]))" "$BASE" "$N")"
  rm -f "$OUT/native.$P.bin" "$OUT/native.$P.log"
  if [[ ! -s "$OUT/ref.$P.bin" ]]; then
    "$BIN/redlite-ref-llama" "$MODEL" logits --tokens "$IDS" --out "$OUT/ref.$P.bin" --dump-from "$P" --ctx "$CTX" >"$OUT/ref.$P.log" 2>&1 \
      || { echo "FAIL  positions.$P (llama.cpp oracle, $OUT/ref.$P.log)"; FAIL=1; continue; }
  fi
  "$BIN/redlite-engine" logits "$MODEL" --tokens "$IDS" --backend gpu --out "$OUT/native.$P.bin" --dump-from "$P" \
    --batch "$BATCH" --cache-mib "$CACHE" --context "$CTX" >"$OUT/native.$P.log" 2>&1
  grep '^prefill [0-9]' "$OUT/native.$P.log" | sed "s/^/  /"
  BOUND=4.0; [[ "$P" -ge 8192 ]] && BOUND=5.0
  if python3 "$ROOT/scripts/dev/compare_dumps.py" "$OUT/native.$P.bin" "$OUT/ref.$P.bin" --max-logit-abs "$BOUND" --max-kl 2e-2 >"$OUT/compare.$P.log" 2>&1 \
     && grep -q "ORACLE LOGITS PARITY: YES" "$OUT/compare.$P.log"; then
    echo "PASS  positions.$P.vs_llama ($(grep '^worst logits' "$OUT/compare.$P.log"))"
  else
    echo "FAIL  positions.$P.vs_llama ($OUT/compare.$P.log)"; FAIL=1
  fi
done
if [[ "$BENCH" == 1 ]]; then
  RAM_GIB=$(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1073741824 ))
  for P in ${POSITIONS//,/ }; do
    TEXT="$(python3 -c "import sys; t=open(sys.argv[1]).read(); print((t + '\n') * (int(sys.argv[2]) // 1397 + 1))" "$FIXTURE" "$P")"
    for C in 4096 22528; do
      [[ "$C" == 22528 && "$RAM_GIB" -lt 40 ]] && continue
      sleep 90
      "$BIN/redlite-generate" "$MODEL" --raw --prompt "$TEXT" --max-tokens 128 --context $((P * 2 + 1024)) --cache-mib "$C" \
        --no-stream --json >/dev/null 2>"$OUT/bench.$P.$C.err"
      python3 - "$OUT/bench.$P.$C.err" "$P" "$C" <<'EOF'
import json, sys
line = [l for l in open(sys.argv[1]) if l.startswith('{')][-1]
d = json.loads(line)
print(f"bench position~{sys.argv[2]} cache {sys.argv[3]} MiB: prompt {d.get('prompt_tokens')} tok, "
      f"prefill {d.get('prefill_tok_s')} tok/s, decode {d.get('decode_tok_s')} tok/s ({d.get('generated_tokens')} generated)")
EOF
    done
  done
fi
exit $FAIL
