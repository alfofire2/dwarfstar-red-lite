#!/usr/bin/env bash
# Development benchmark for the native runtime on the local Mac (not part of regress_m4.sh).
#
#   scripts/dev/bench_m4.sh MODEL [--reps N] [--only decode22|decode4|prefill|prefill8192|prefill4|prefill22|llama] [--json FILE] [--cool SECONDS]
#                           [--cache-mib N]
#
# decode22 : redlite-generate greedy, 256 tokens, --cache-mib 22528 (full residency, GPU-routed)
# decode4  : same prompt, --cache-mib 4096 (bounded cache); one warm-up run first so the GGUF
#            is in the page cache, as in the dev19/dev21 records
# prefill  : redlite-engine logits, first 1100 tokens of tests/fixtures/long_context_prompt.txt
#            ingested by the batched prefill in chunks of 512 (the default), --cache-mib 4096
#            (one untimed warm-up run first, as for decode4, so the page cache state does not decide the number)
# prefill8192 : same, 8192 ids (the fixture ids repeated, as in long_positions.sh)
# prefill4  : 1100 and 8192 ids, --cache-mib 4096 with the engine's default chunk (2048 since dev34)
# prefill22 : 1100 and 8192 ids with the full-residency cache (--cache-mib, default 22528) and the engine's
#            default chunk (2048 with every expert preloaded, dev30): the configuration of a >= 40 GiB Mac
# llama    : the pinned llama.cpp (redlite-ref-llama bench, its default context parameters, fully
#            resident) on the same 1100 and 8192 prefill ids and 256 greedy tokens after the same chat
#            prompt; not part of the default set (needs scripts/dev/build_ref_llama.sh)
# --cache-mib N replaces the decode22 cache (the full-residency cache of another GGUF).
# Prints every run and the median. Numbers are only meaningful on an otherwise idle machine.
# --cool S sleeps S seconds before every run: back-to-back prefill runs on the M4 Max 48 GiB throttle the GPU
# (dense GPU time x3 after four runs, 2026-09-30), so prefill numbers are taken with --cool 90.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="$ROOT/.deps/redmetal"
MODEL="${1:-}"
[[ -z "$MODEL" || ! -f "$MODEL" ]] && { echo "usage: $0 MODEL [--reps N] [--only WHAT] [--json FILE]" >&2; exit 2; }
shift
REPS=3; ONLY=""; JSON=""; COOL=0; FULL=22528
while [[ $# -gt 0 ]]; do
  case "$1" in
    --reps) REPS="$2"; shift 2;;
    --only) ONLY="$2"; shift 2;;
    --json) JSON="$2"; shift 2;;
    --cool) COOL="$2"; shift 2;;
    --cache-mib) FULL="$2"; shift 2;;
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
    sleep "$COOL"
    "$BIN/redlite-generate" "$MODEL" --prompt "$PROMPT" --max-tokens 256 --cache-mib "$cache" --no-stream --json >"$TMP/out" 2>"$TMP/err"
    local line; line="$(grep '^{' "$TMP/err" | tail -1)"
    local tps; tps="$(python3 -c "import json,sys; d=json.loads(sys.argv[1]); print(d['decode_tok_s'])" "$line")"
    local gen; gen="$(python3 -c "import json,sys; d=json.loads(sys.argv[1]); print(d['generated_tokens'], d['gpu_routed'], round(d['cache_hits']/max(1,d['cache_hits']+d['cache_misses']),3))" "$line")"
    echo "$label run $i: decode $tps tok/s (generated, gpu_routed, hit_rate: $gen)"
    vals+=("$tps")
  done
  local m; m="$(median "${vals[@]}")"; echo "$label median: $m tok/s"; echo "$label $m" >>"$TMP/summary"
}

fixture_ids() { # N: the fixture ids repeated and cut to N
  local base; base="$("$BIN/redlite-engine" tokenize "$MODEL" --text "$(cat "$ROOT/tests/fixtures/long_context_prompt.txt")" --no-special | head -1)"
  python3 -c "import sys; b=sys.argv[1].split(','); n=int(sys.argv[2]); print(','.join((b*(n//len(b)+1))[:n]))" "$base" "$1"
}

prefill() { # N label [cache] [chunk]: chunk "" = the engine default
  local n="$1" label="$2" cache="${3:-4096}" chunk="${4-512}" ids vals=()
  ids="$(fixture_ids $((n + 1)))"
  local bargs=(); [[ -n "$chunk" ]] && bargs=(--batch "$chunk")
  [[ "$cache" == 4096 ]] && "$BIN/redlite-engine" logits "$MODEL" --tokens "$ids" --backend gpu --dump-from "$n" ${bargs[@]+"${bargs[@]}"} --cache-mib "$cache" --context $((n + 256)) >/dev/null 2>&1
  for i in $(seq 1 "$REPS"); do
    sleep "$COOL"
    local line; line="$("$BIN/redlite-engine" logits "$MODEL" --tokens "$ids" --backend gpu --dump-from "$n" ${bargs[@]+"${bargs[@]}"} --cache-mib "$cache" --context $((n + 256)) 2>&1 | grep '^prefill [0-9]')"
    local tps; tps="$(echo "$line" | sed -E 's/.*\(([0-9.]+) tok\/s\).*/\1/')"
    echo "$label run $i: $line"
    vals+=("$tps")
  done
  local m; m="$(median "${vals[@]}")"; echo "$label median: $m tok/s"; echo "$label $m" >>"$TMP/summary"
}

llama() {
  [[ -x "$BIN/redlite-ref-llama" ]] || { echo "llama: $BIN/redlite-ref-llama missing (scripts/dev/build_ref_llama.sh)" >&2; return; }
  local chat; chat="$("$BIN/redlite-engine" tokenize "$MODEL" --text "$PROMPT" --chat | head -1)"
  local what n ids vals line
  for what in 1100 8192 decode; do
    vals=()
    if [[ "$what" == decode ]]; then ids="$chat"; else ids="$(fixture_ids "$what")"; fi
    for i in $(seq 1 "$REPS"); do
      sleep "$COOL"
      if [[ "$what" == decode ]]; then
        line="$("$BIN/redlite-ref-llama" "$MODEL" bench --tokens "$ids" --max 256 2>&1 | grep '^llama decode')"
      else
        line="$("$BIN/redlite-ref-llama" "$MODEL" bench --tokens "$ids" --max 0 2>&1 | grep '^llama prefill')"
      fi
      echo "llama_$what run $i: $line"
      vals+=("$(echo "$line" | sed -E 's/.*\(([0-9.]+) tok\/s\).*/\1/')")
    done
    local m; m="$(median "${vals[@]}")"; echo "llama_$what median: $m tok/s"; echo "llama_$what $m" >>"$TMP/summary"
  done
}

echo "commit $(git -C "$ROOT" rev-parse --short HEAD)$(git -C "$ROOT" diff --quiet || echo '+dirty')  $(date '+%Y-%m-%d %H:%M')"
[[ -z "$ONLY" || "$ONLY" == decode22 ]] && decode "$FULL" decode22
[[ -z "$ONLY" || "$ONLY" == decode4 ]] && decode 4096 decode4
[[ -z "$ONLY" || "$ONLY" == prefill ]] && prefill 1100 prefill1100
[[ "$ONLY" == prefill8192 ]] && prefill 8192 prefill8192
[[ "$ONLY" == prefill4 ]] && { prefill 1100 prefill1100_4 4096 ""; prefill 8192 prefill8192_4 4096 ""; }
[[ "$ONLY" == prefill22 ]] && { prefill 1100 prefill1100_full "$FULL" ""; prefill 8192 prefill8192_full "$FULL" ""; }
[[ "$ONLY" == llama ]] && llama
if [[ -n "$JSON" ]]; then
  python3 - "$TMP/summary" "$JSON" "$(git -C "$ROOT" rev-parse --short HEAD)" <<'EOF'
import json, sys
d = {l.split()[0]: float(l.split()[1]) for l in open(sys.argv[1])}
d["commit"] = sys.argv[3]
json.dump(d, open(sys.argv[2], "w"), indent=1)
EOF
fi
