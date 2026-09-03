#!/usr/bin/env bash
# Complete native regression suite for the target Apple Silicon machine.
#
#   scripts/regress_m4.sh MODEL.gguf [--quick]
#
# Builds every native tool, runs the model-free self-tests, every stage parity
# validator that the dev10..dev17 milestones introduced, the persistent-engine
# multi-token CPU-vs-Metal parity, the native tokenizer against llama_tokenize
# and native greedy generation against the pinned llama.cpp greedy decode
# (reference tools are built on demand from the bootstrapped .deps/llama.cpp).
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${1:-}"
QUICK=0
[[ "${2:-}" == "--quick" ]] && QUICK=1
if [[ -z "$MODEL" || ! -f "$MODEL" ]]; then
  echo "usage: $0 MODEL.gguf [--quick]" >&2
  exit 2
fi
if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "the M4 regression suite requires macOS Metal" >&2
  exit 2
fi
BIN="$ROOT/.deps/redmetal"
LOG="${REDLITE_REGRESS_LOG:-$ROOT/.deps/regress}"
mkdir -p "$LOG"
PASS=0
FAIL=0
FAILED=()

run() {
  local name="$1"; shift
  local out="$LOG/$name.log"
  if "$@" >"$out" 2>&1; then
    echo "PASS  $name"
    PASS=$((PASS + 1))
  else
    echo "FAIL  $name  (see $out)"
    FAIL=$((FAIL + 1))
    FAILED+=("$name")
  fi
}

expect_line() {
  # expect_line NAME PATTERN CMD...
  local name="$1" pattern="$2"; shift 2
  local out="$LOG/$name.log"
  if "$@" >"$out" 2>&1 && grep -q -- "$pattern" "$out"; then
    echo "PASS  $name"
    PASS=$((PASS + 1))
  else
    echo "FAIL  $name  (expected '$pattern'; see $out)"
    FAIL=$((FAIL + 1))
    FAILED+=("$name")
  fi
}

echo "== build =="
run build.native make -C "$ROOT" native

echo "== model-free =="
run selftest.native "$BIN/redlite-native" selftest
run selftest.offline "$BIN/redlite-native-offline-test"
run selftest.router "$BIN/redlite-router-offline-test"
run selftest.shared "$BIN/redlite-shared-exec-offline-test"
run selftest.engine "$BIN/redlite-engine-offline-test"
run selftest.attention "$BIN/redlite-attention" --selftest

echo "== stage parity (real GGUF) =="
expect_line topk.parity "parity match       : YES" "$BIN/redlite-native" topk-parity "$MODEL"
expect_line router.parity "router parity      : YES" "$BIN/redlite-router" router-parity "$MODEL"
expect_line routed.parity "routed parity      : YES" "$BIN/redlite-router" routed-parity "$MODEL"
expect_line ffn.layer0 "COMPLETE FFN parity: YES" "$BIN/redlite-ffn" parity "$MODEL" --layer 0 --top-k 10 --rows 8 --cache-mib 256
expect_line ffn.layer0.full "COMPLETE FFN parity: YES" "$BIN/redlite-ffn" parity "$MODEL" --layer 0 --top-k 10 --rows 2048 --cache-mib 256
expect_line ffn.layer6 "COMPLETE FFN parity: YES" "$BIN/redlite-ffn" parity "$MODEL" --layer 6 --top-k 10 --rows 8 --cache-mib 256
expect_line deltanet.proj "projection parity  : YES" "$BIN/redlite-deltanet-proj" parity "$MODEL" --layer 0 --rows 8
expect_line deltanet.prestate "prestate parity    : YES" "$BIN/redlite-deltanet-prestate" parity "$MODEL" --layer 0
expect_line deltanet.state "state parity       : YES" "$BIN/redlite-deltanet-state" parity "$MODEL" --layer 0
expect_line deltanet.tail "tail parity        : YES" "$BIN/redlite-deltanet-tail" parity "$MODEL" --layer 0
expect_line deltanet.layer "COMPLETE DELTANET   : YES" "$BIN/redlite-deltanet-layer" parity "$MODEL" --layer 0
expect_line recurrent.block "COMPLETE BLOCK       : YES" "$BIN/redlite-recurrent-block" parity "$MODEL" --layer 0 --top-k 10 --cache-mib 256
expect_line attention.ctx1 "FULL ATTENTION     : YES" "$BIN/redlite-attention" parity "$MODEL" --layer 3 --position 0
expect_line attention.ctx16 "FULL ATTENTION     : YES" "$BIN/redlite-attention" parity "$MODEL" --layer 3 --position 15
expect_line attention.block "COMPLETE FULL ATTENTION BLOCK: YES" "$BIN/redlite-attention-block" parity "$MODEL" --layer 3 --position 7 --top-k 10 --cache-mib 256
if [[ $QUICK -eq 1 ]]; then
  expect_line decoder.stack.partial "PARTIAL DECODER STACK : YES" "$BIN/redlite-decoder-stack" parity "$MODEL" --layers 4
else
  expect_line decoder.stack "COMPLETE 48-LAYER DECODER STACK: YES" "$BIN/redlite-decoder-stack" parity "$MODEL" --position 7 --top-k 10 --cache-mib 256
fi

echo "== persistent engine =="
expect_line engine.info "recurrent=36 full-attention=12" "$BIN/redlite-engine" info "$MODEL"
expect_line engine.parity "MULTI-TOKEN ENGINE PARITY: YES" "$BIN/redlite-engine" parity "$MODEL" --tokens 9707,11,1879,0,785,12884 --cache-mib 1024 --context 64

echo "== tokenizer / generation =="
PROMPT="Explain in one sentence why the sky is blue."
expect_line tokenize.chat "^151644,872,198,840,20772,304,825,11652,3170,279,12884,374,6303,13,151645,198,151644,77091,198$" \
  "$BIN/redlite-engine" tokenize "$MODEL" --text "$PROMPT" --chat
expect_line generate.greedy "Rayleigh scattering" "$BIN/redlite-generate" "$MODEL" --prompt "$PROMPT" --max-tokens 40 --cache-mib 2048 --no-stream --stats

if [[ -x "$ROOT/.deps/llama.cpp/build/bin/libllama.dylib" || -f "$ROOT/.deps/llama.cpp/build/bin/libllama.dylib" ]]; then
  echo "== pinned llama.cpp oracle =="
  run build.ref bash "$ROOT/scripts/dev/build_ref_llama.sh"
  if [[ -x "$BIN/redlite-ref-llama" ]]; then
    REF_TOK="$("$BIN/redlite-ref-llama" "$MODEL" tokenize --text "$PROMPT" 2>/dev/null | head -1)"
    NAT_TOK="$("$BIN/redlite-engine" tokenize "$MODEL" --text "$PROMPT" | head -1)"
    if [[ -n "$REF_TOK" && "$REF_TOK" == "$NAT_TOK" ]]; then echo "PASS  tokenize.vs_llama"; PASS=$((PASS + 1)); else echo "FAIL  tokenize.vs_llama ($NAT_TOK vs $REF_TOK)"; FAIL=$((FAIL + 1)); FAILED+=(tokenize.vs_llama); fi
    "$BIN/redlite-engine" logits "$MODEL" --tokens 9707,11,1879 --backend gpu --out "$LOG/native.bin" --cache-mib 1024 --context 64 >"$LOG/logits.native.log" 2>&1
    "$BIN/redlite-ref-llama" "$MODEL" logits --tokens 9707,11,1879 --out "$LOG/ref.bin" --ctx 64 >"$LOG/logits.ref.log" 2>&1
    expect_line logits.vs_llama "ARGMAX AGREEMENT: YES" python3 "$ROOT/scripts/dev/compare_dumps.py" "$LOG/native.bin" "$LOG/ref.bin"
    CHAT_IDS="$("$BIN/redlite-engine" tokenize "$MODEL" --text "$PROMPT" --chat | head -1)"
    "$BIN/redlite-generate" "$MODEL" --prompt "$PROMPT" --max-tokens 24 --cache-mib 2048 --no-stream --tokens-out "$LOG/gen.native.txt" >"$LOG/gen.native.log" 2>&1
    NAT_GEN="$(tail -n +20 "$LOG/gen.native.txt" | tr '\n' ' ' | sed 's/ *$//')"
    REF_GEN="$("$BIN/redlite-ref-llama" "$MODEL" greedy --tokens "$CHAT_IDS" --max 24 --ctx 128 2>/dev/null | grep '^generated:' | sed 's/^generated: //' | cut -d' ' -f1-24)"
    NAT_N=$(echo "$NAT_GEN" | wc -w | tr -d ' ')
    REF_HEAD="$(echo "$REF_GEN" | cut -d' ' -f1-"$NAT_N")"
    if [[ "$NAT_N" -gt 0 && "$NAT_GEN" == "$REF_HEAD" ]]; then echo "PASS  generate.vs_llama ($NAT_N tokens identical)"; PASS=$((PASS + 1)); else echo "FAIL  generate.vs_llama"; echo "  native: $NAT_GEN"; echo "  ref   : $REF_GEN"; FAIL=$((FAIL + 1)); FAILED+=(generate.vs_llama); fi
  fi
else
  echo "SKIP  pinned llama.cpp oracle (run 'redlite bootstrap' to enable)"
fi

echo
echo "regression summary: pass=$PASS fail=$FAIL"
if [[ $FAIL -ne 0 ]]; then
  printf '  failed: %s\n' "${FAILED[@]}"
  exit 3
fi
