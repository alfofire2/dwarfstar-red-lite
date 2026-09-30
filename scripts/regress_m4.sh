#!/usr/bin/env bash
# Complete native regression suite for the target Apple Silicon machine.
#
#   scripts/regress_m4.sh MODEL.gguf [--quick]
#
# Builds every native tool, runs the model-free self-tests, every stage parity
# validator that the dev10..dev17 milestones introduced, the persistent-engine
# multi-token CPU-vs-Metal parity, the native tokenizer against llama_tokenize
# (tests/fixtures/tokenizer_corpus.txt), native logits against the pinned
# llama.cpp with KL / max-abs thresholds, native greedy generation against the
# pinned llama.cpp greedy decode, the dev20 batched-prefill parity checks and,
# in the full run, a >1024-position prompt (batched ingestion of the first 1100
# positions, then token by token) that exercises both attention paths against
# llama.cpp.
# Reference tools are built on demand from the bootstrapped .deps/llama.cpp
# (or the checkout named by REDLITE_LLAMA_DIR).
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
LLAMA_DIR="${REDLITE_LLAMA_DIR:-$ROOT/.deps/llama.cpp}"
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
run selftest.gguf_fuzz "$BIN/redlite-gguf-fuzz" --iterations 20000
run selftest.sanitize bash "$ROOT/scripts/sanitize_offline.sh"
expect_line selftest.engine_kernels "ENGINE KERNEL SELFTEST: OK" "$BIN/redlite-engine" kernel-selftest

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

if [[ "$(sysctl -n hw.memsize)" -ge 42949672960 ]]; then
  echo "== GPU-routed decode (dev21, full residency; >= 40 GiB) =="
  expect_line engine.parity.gpu_routed "MULTI-TOKEN ENGINE PARITY: YES" "$BIN/redlite-engine" parity "$MODEL" --tokens 9707,11,1879,0,785,12884 --cache-mib 22528 --context 64 --repeat 2
  grep -q "GPU-routed tokens     : 12 speculative" "$LOG/engine.parity.gpu_routed.log" || { echo "FAIL  engine.parity.gpu_routed.count (expected 12 GPU-routed tokens)"; FAIL=$((FAIL + 1)); FAILED+=(engine.parity.gpu_routed.count); }
fi

echo "== batched prefill (dev20) =="
expect_line prefill.parity.chunks8 "BATCHED PREFILL PARITY: YES" "$BIN/redlite-engine" prefill "$MODEL" --tokens 151644,872,198,840,20772,304,825,11652,3170,279,12884,374,6303,13,151645,198,151644,77091,198 --batch 8 --cache-mib 1024 --context 64
PREFILL96="$("$BIN/redlite-engine" tokenize "$MODEL" --text "$(cat "$ROOT/tests/fixtures/long_context_prompt.txt")" --no-special | head -1 | cut -d',' -f1-96)"
expect_line prefill.parity.96 "BATCHED PREFILL PARITY: YES" "$BIN/redlite-engine" prefill "$MODEL" --tokens "$PREFILL96" --batch 32 --cache-mib 2048 --context 128

echo "== sanitized chat turn (dev26: ASan+UBSan redlite-generate on the real model) =="
expect_line generate.sanitize "SANITIZED CHAT TURN: OK" bash "$ROOT/scripts/dev/sanitize_chat.sh" "$MODEL"

echo "== tokenizer / generation =="
PROMPT="Explain in one sentence why the sky is blue."
expect_line tokenize.chat "^151644,872,198,840,20772,304,825,11652,3170,279,12884,374,6303,13,151645,198,151644,77091,198$" \
  "$BIN/redlite-engine" tokenize "$MODEL" --text "$PROMPT" --chat
expect_line generate.greedy "Rayleigh scattering" "$BIN/redlite-generate" "$MODEL" --prompt "$PROMPT" --max-tokens 40 --cache-mib 2048 --no-stream --stats
expect_line server.stream_greedy "SERVER CHECK: YES" python3 "$ROOT/scripts/dev/server_check.py" "$MODEL" --bin "$BIN"
expect_line generate.json '"batch":512,"finish":"' "$BIN/redlite-generate" "$MODEL" --prompt "$PROMPT" --max-tokens 8 --cache-mib 2048 --no-stream --json
# Ctrl-C mid-answer: the run must stop, report finish=interrupted and exit 130 (not be killed)
"$BIN/redlite-generate" "$MODEL" --prompt "Count from 1 to 2000, separated by commas." --max-tokens 4000 \
  --cache-mib 2048 --json >"$LOG/generate.sigint.log" 2>&1 &
SIGINT_PID=$!
for _ in $(seq 1 240); do
  [[ $(wc -c <"$LOG/generate.sigint.log") -gt 200 ]] && break
  kill -0 "$SIGINT_PID" 2>/dev/null || break
  sleep 0.5
done
kill -INT "$SIGINT_PID" 2>/dev/null
wait "$SIGINT_PID"; SIGINT_RC=$?
if [[ "$SIGINT_RC" == 130 ]] && grep -q '"finish":"interrupted"' "$LOG/generate.sigint.log"; then
  echo "PASS  generate.sigint"; PASS=$((PASS + 1))
else
  echo "FAIL  generate.sigint  (exit $SIGINT_RC; see $LOG/generate.sigint.log)"; FAIL=$((FAIL + 1)); FAILED+=(generate.sigint)
fi
# an empty prompt must be refused instead of sampling from uninitialised logits
if "$BIN/redlite-generate" "$MODEL" --prompt "" --raw --max-tokens 4 --cache-mib 256 >"$LOG/generate.empty.log" 2>&1; then
  echo "FAIL  generate.empty_prompt (exit 0; see $LOG/generate.empty.log)"; FAIL=$((FAIL + 1)); FAILED+=(generate.empty_prompt)
else
  echo "PASS  generate.empty_prompt"; PASS=$((PASS + 1))
fi

if [[ -f "$LLAMA_DIR/build/bin/libllama.dylib" ]]; then
  echo "== pinned llama.cpp oracle ($LLAMA_DIR) =="
  run build.ref env REDLITE_LLAMA_DIR="$LLAMA_DIR" bash "$ROOT/scripts/dev/build_ref_llama.sh"
  run build.ref_sampler env REDLITE_LLAMA_DIR="$LLAMA_DIR" bash "$ROOT/scripts/dev/build_ref_sampler.sh"
  expect_line sampler.vs_llama "SAMPLER PARITY: YES" python3 "$ROOT/scripts/dev/compare_sampler.py" --bin "$BIN" --draws 20000
  if [[ -x "$BIN/redlite-ref-llama" ]]; then
    REF_TOK="$("$BIN/redlite-ref-llama" "$MODEL" tokenize --text "$PROMPT" 2>/dev/null | head -1)"
    NAT_TOK="$("$BIN/redlite-engine" tokenize "$MODEL" --text "$PROMPT" | head -1)"
    if [[ -n "$REF_TOK" && "$REF_TOK" == "$NAT_TOK" ]]; then echo "PASS  tokenize.vs_llama"; PASS=$((PASS + 1)); else echo "FAIL  tokenize.vs_llama ($NAT_TOK vs $REF_TOK)"; FAIL=$((FAIL + 1)); FAILED+=(tokenize.vs_llama); fi
    # whole corpus (special tokens, whitespace runs, code, CJK, emoji, ...) in one model load per tool
    CORPUS="$ROOT/tests/fixtures/tokenizer_corpus.txt"
    "$BIN/redlite-ref-llama" "$MODEL" tokenize --file "$CORPUS" >"$LOG/tokenize.corpus.ref.txt" 2>"$LOG/tokenize.corpus.ref.log"
    "$BIN/redlite-engine" tokenize "$MODEL" --file "$CORPUS" >"$LOG/tokenize.corpus.native.txt" 2>"$LOG/tokenize.corpus.native.log"
    CORPUS_N="$(wc -l <"$CORPUS" | tr -d ' ')"
    if [[ -s "$LOG/tokenize.corpus.ref.txt" && "$(wc -l <"$LOG/tokenize.corpus.ref.txt" | tr -d ' ')" == "$CORPUS_N" ]] && cmp -s "$LOG/tokenize.corpus.ref.txt" "$LOG/tokenize.corpus.native.txt"; then
      echo "PASS  tokenize.corpus_vs_llama ($CORPUS_N inputs identical)"; PASS=$((PASS + 1))
    else
      echo "FAIL  tokenize.corpus_vs_llama (diff $LOG/tokenize.corpus.native.txt $LOG/tokenize.corpus.ref.txt)"; FAIL=$((FAIL + 1)); FAILED+=(tokenize.corpus_vs_llama)
    fi
    "$BIN/redlite-engine" logits "$MODEL" --tokens 9707,11,1879 --backend gpu --out "$LOG/native.bin" --cache-mib 1024 --context 64 >"$LOG/logits.native.log" 2>&1
    "$BIN/redlite-ref-llama" "$MODEL" logits --tokens 9707,11,1879 --out "$LOG/ref.bin" --ctx 64 >"$LOG/logits.ref.log" 2>&1
    expect_line logits.vs_llama "ORACLE LOGITS PARITY: YES" python3 "$ROOT/scripts/dev/compare_dumps.py" "$LOG/native.bin" "$LOG/ref.bin"
    CHAT_IDS="$("$BIN/redlite-engine" tokenize "$MODEL" --text "$PROMPT" --chat | head -1)"
    CHAT_N="$(echo "$CHAT_IDS" | tr ',' '\n' | wc -l | tr -d ' ')"
    "$BIN/redlite-generate" "$MODEL" --prompt "$PROMPT" --max-tokens 24 --cache-mib 2048 --no-stream --tokens-out "$LOG/gen.native.txt" >"$LOG/gen.native.log" 2>&1
    NAT_GEN="$(tail -n +$((CHAT_N + 1)) "$LOG/gen.native.txt" | tr '\n' ' ' | sed 's/ *$//')"
    REF_GEN="$("$BIN/redlite-ref-llama" "$MODEL" greedy --tokens "$CHAT_IDS" --max 24 --ctx 128 2>/dev/null | grep '^generated:' | sed 's/^generated: //' | cut -d' ' -f1-24)"
    NAT_N=$(echo "$NAT_GEN" | wc -w | tr -d ' ')
    REF_HEAD="$(echo "$REF_GEN" | cut -d' ' -f1-"$NAT_N")"
    if [[ "$NAT_N" -gt 0 && "$NAT_GEN" == "$REF_HEAD" ]]; then echo "PASS  generate.vs_llama ($NAT_N tokens identical)"; PASS=$((PASS + 1)); else echo "FAIL  generate.vs_llama"; echo "  native: $NAT_GEN"; echo "  ref   : $REF_GEN"; FAIL=$((FAIL + 1)); FAILED+=(generate.vs_llama); fi
    if [[ $QUICK -eq 0 ]]; then
      # >1024 positions: the GQA kernel switches to its multi-chunk online-softmax path past 1024 keys,
      # which the short prompts above never reach. The prompt is the frozen fixture
      # tests/fixtures/long_context_prompt.txt cut to 1200 tokens; positions 1100..1199 are compared.
      # On this fixture llama.cpp's layer-2 router has an exact probability tie at position 1035
      # (experts 403/101, both 0.0077861) that the two implementations break differently, so after
      # that position the logits legitimately differ by up to ~1.1 (KL ~6e-3) while every argmax
      # stays identical; the thresholds below are sized for that, not for the 1e-5 short-context drift.
      LONG_IDS="$("$BIN/redlite-engine" tokenize "$MODEL" --text "$(cat "$ROOT/tests/fixtures/long_context_prompt.txt")" --no-special | head -1 | cut -d',' -f1-1200)"
      LONG_N="$(echo "$LONG_IDS" | tr ',' '\n' | wc -l | tr -d ' ')"
      if [[ "$LONG_N" -gt 1024 ]]; then
        # the first 1100 positions are ingested by the batched prefill (one chunk), the compared positions token by token
        "$BIN/redlite-engine" logits "$MODEL" --tokens "$LONG_IDS" --backend gpu --out "$LOG/native.long.bin" --dump-from 1100 --batch 1100 --cache-mib 4096 --context 1536 >"$LOG/logits.long.native.log" 2>&1
        "$BIN/redlite-ref-llama" "$MODEL" logits --tokens "$LONG_IDS" --out "$LOG/ref.long.bin" --dump-from 1100 --ctx 1536 >"$LOG/logits.long.ref.log" 2>&1
        expect_line logits.long_context_vs_llama "ORACLE LOGITS PARITY: YES" python3 "$ROOT/scripts/dev/compare_dumps.py" "$LOG/native.long.bin" "$LOG/ref.long.bin" --max-logit-abs 2.0 --max-kl 2e-2
      else
        echo "FAIL  logits.long_context_vs_llama (prompt only $LONG_N tokens)"; FAIL=$((FAIL + 1)); FAILED+=(logits.long_context_vs_llama)
      fi
    fi
  fi
else
  echo "SKIP  pinned llama.cpp oracle (run 'redlite bootstrap' or set REDLITE_LLAMA_DIR to enable)"
fi

echo
echo "regression summary: pass=$PASS fail=$FAIL"
if [[ $FAIL -ne 0 ]]; then
  printf '  failed: %s\n' "${FAILED[@]}"
  exit 3
fi
