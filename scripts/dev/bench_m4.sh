#!/usr/bin/env bash
# Development benchmark for the native runtime on the local Mac (not part of regress_m4.sh).
#
#   scripts/dev/bench_m4.sh MODEL [--reps N] [--only decode22|decode4|prefill] [--json FILE]
#
# decode22 : redlite-generate greedy, 256 tokens, --cache-mib 22528 (full residency, GPU-routed)
# decode4  : same prompt, --cache-mib 4096 (bounded cache); one warm-up run first so the GGUF
#            is in the page cache, as in the dev19/dev21 records
# prefill  : redlite-engine logits, first 1100 tokens of tests/fixtures/long_context_prompt.txt
#            ingested by the batched prefill in chunks of 512 (the default), --cache-mib 4096
# Prints every run and the median. Numbers are only meaningful on an otherwise idle machine.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="$ROOT/.deps/redmetal"
MODEL="${1:-}"
[[ -z "$MODEL" || ! -f "$MODEL" ]] && { echo "usage: $0 MODEL [--reps N] [--only WHAT] [--json FILE]" >&2; exit 2; }
shift
REPS=3; ONLY=""; JSON=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --reps) REPS="$2"; shift 2;;
    --only) ONLY="$2"; shift 2;;
    --json) JSON="$2"; shift 2;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
done
PROMPT="Write a detailed, step by step explanation of how a rainbow forms, covering refraction, dispersion and reflection."
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT

median() { python3 -c "import statistics,sys; v=[float(x) for x in sys.argv[1:]]; print(f'{statistics.median(v):.2f}')" "$@"; }

decode() { # cache_mib label
  local cache="$1" label="$2" vals=()
  [[ "$cache" == 4096 ]] && "$BIN/redlite-generate" "$MODEL" --prompt "$PROMPT" --max-tokens 64 --cache-mib "$cache" --no-stream >/dev/null 2>&1
  for i in $(seq 1 "$REPS"); do
    "$BIN/redlite-generate" "$MODEL" --prompt "$PROMPT" --max-tokens 256 --cache-mib "$cache" --no-stream --json >"$TMP/out" 2>"$TMP/err"
    local line; line="$(grep '^{' "$TMP/err" | tail -1)"
    local tps; tps="$(python3 -c "import json,sys; d=json.loads(sys.argv[1]); print(d['decode_tok_s'])" "$line")"
    local gen; gen="$(python3 -c "import json,sys; d=json.loads(sys.argv[1]); print(d['generated_tokens'], d['gpu_routed'], round(d['cache_hits']/max(1,d['cache_hits']+d['cache_misses']),3))" "$line")"
    echo "$label run $i: decode $tps tok/s (generated, gpu_routed, hit_rate: $gen)"
    vals+=("$tps")
  done
  local m; m="$(median "${vals[@]}")"; echo "$label median: $m tok/s"; echo "$label $m" >>"$TMP/summary"
}

prefill() {
  local ids vals=()
  ids="$("$BIN/redlite-engine" tokenize "$MODEL" --text "$(cat "$ROOT/tests/fixtures/long_context_prompt.txt")" --no-special | head -1 | cut -d',' -f1-1101)"
  for i in $(seq 1 "$REPS"); do
    local line; line="$("$BIN/redlite-engine" logits "$MODEL" --tokens "$ids" --backend gpu --dump-from 1100 --batch 512 --cache-mib 4096 --context 1536 2>&1 | grep '^prefill ')"
    local tps; tps="$(echo "$line" | sed -E 's/.*\(([0-9.]+) tok\/s\).*/\1/')"
    echo "prefill1100 run $i: $line"
    vals+=("$tps")
  done
  local m; m="$(median "${vals[@]}")"; echo "prefill1100 median: $m tok/s"; echo "prefill1100 $m" >>"$TMP/summary"
}

echo "commit $(git -C "$ROOT" rev-parse --short HEAD)$(git -C "$ROOT" diff --quiet || echo '+dirty')  $(date '+%Y-%m-%d %H:%M')"
[[ -z "$ONLY" || "$ONLY" == decode22 ]] && decode 22528 decode22
[[ -z "$ONLY" || "$ONLY" == decode4 ]] && decode 4096 decode4
[[ -z "$ONLY" || "$ONLY" == prefill ]] && prefill
if [[ -n "$JSON" ]]; then
  python3 - "$TMP/summary" "$JSON" "$(git -C "$ROOT" rev-parse --short HEAD)" <<'EOF'
import json, sys
d = {l.split()[0]: float(l.split()[1]) for l in open(sys.argv[1])}
d["commit"] = sys.argv[3]
json.dump(d, open(sys.argv[2], "w"), indent=1)
EOF
fi
