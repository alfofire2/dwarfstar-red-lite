#!/bin/bash
# dev46: bounded-cache (24 GiB Mac) A/B of the expert-cache options on the six prompts of
# tests/fixtures/cache_trace_prompts.txt, 256 tokens each, --cache-mib 4096 (or CACHE=N).
# Per configuration: expert misses per token, MiB read from the GGUF per token, and per-prompt decode tok/s
# (cooled runs, COOL seconds before each). Configurations (env of redlite-generate):
#   default            dev37 size classes, prefetch on, page cache, no routing bias
#   uniform            RL_POOL_CLASSES=0 (0.4.0 slots)
#   bias05             RL_ROUTE_CACHE_BIAS=0.5 (cache-aware routing, changes outputs; engine PPL +0.0 % on the M4 Max)
#   nocache            RL_POOL_NOCACHE=1 (expert reads bypass the page cache)
#   noprefetch         RL_ENGINE_PREFETCH=0
# usage: scripts/dev/small_mac_ab.sh MODEL [config ...]
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="${BIN:-$ROOT/.deps/redmetal}"
MODEL="${1:?usage: $0 MODEL [config ...]}"; shift
CONFIGS=("$@"); [ ${#CONFIGS[@]} -eq 0 ] && CONFIGS=(default uniform bias05 nocache noprefetch)
env_for() {
  case "$1" in
    default) echo "" ;; uniform) echo "RL_POOL_CLASSES=0" ;; bias05) echo "RL_ROUTE_CACHE_BIAS=0.5" ;;
    nocache) echo "RL_POOL_NOCACHE=1" ;; noprefetch) echo "RL_ENGINE_PREFETCH=0" ;; *) echo "BAD" ;;
  esac
}
echo "machine: $(sysctl -n machdep.cpu.brand_string) $(($(sysctl -n hw.memsize) / 1073741824)) GiB  commit $(git -C "$ROOT" rev-parse --short HEAD)"
for c in "${CONFIGS[@]}"; do
  e="$(env_for "$c")"; [ "$e" = BAD ] && { echo "unknown config $c"; continue; }
  H=0; X=0; MIB=0; TOK=0; RATES=""
  while IFS= read -r P; do
    sleep "${COOL:-30}"
    r=$(env $e "$BIN/redlite-generate" "$MODEL" --prompt "$P" --max-tokens 256 --cache-mib "${CACHE:-4096}" --no-stream --json 2>&1 >/dev/null | grep '^{' | \
        python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print(d['cache_hits'], d['cache_misses'], d['ssd_mib'], d['generated_tokens'], d['decode_tok_s'])")
    set -- $r; H=$((H+$1)); X=$((X+$2)); MIB=$(python3 -c "print($MIB+$3)"); TOK=$((TOK+$4)); RATES="$RATES $5"
  done < "$ROOT/tests/fixtures/cache_trace_prompts.txt"
  med=$(python3 -c "import statistics,sys; print(statistics.median([float(x) for x in sys.argv[1:]]))" $RATES)
  echo "$c: misses/token $(python3 -c "print(round($X/($H+$X)*480,1))")  MiB read/token $(python3 -c "print(round($MIB/$TOK,1))")  decode tok/s:$RATES (median $med)"
done
