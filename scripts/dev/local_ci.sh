#!/bin/bash
# The model-free checks that GitHub CI ran until 2026-10-03 (removed: hosted runners are paid), run locally.
# Same steps: ruff (fatal errors), compileall, make native and make sanitize with compiler warnings as failures,
# make test. macOS or Linux. Prints "LOCAL CI: PASS" or the first failing step.
set -uo pipefail
cd "$(dirname "$0")/../.."
LOG="${TMPDIR:-/tmp}/redlite-local-ci"; mkdir -p "$LOG"
step() { # name command...
  local name="$1"; shift
  if ! "$@" >"$LOG/$name.log" 2>&1; then echo "FAIL  $name (see $LOG/$name.log)"; exit 1; fi
  if [[ "$name" == native || "$name" == sanitize ]] && grep -q 'warning:' "$LOG/$name.log"; then
    echo "FAIL  $name: compiler warnings"; grep -n 'warning:' "$LOG/$name.log" | head; exit 1
  fi
  echo "PASS  $name"
}
step ruff ruff check redlite tests --select E9,F63,F7,F82
step compileall python3 -m compileall -q redlite tests scripts
step native make native
step sanitize make sanitize
step test make test
echo "LOCAL CI: PASS"
