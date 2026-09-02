#!/bin/bash
set -euo pipefail
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
export PYTHONPATH="$ROOT"
python3 -m unittest discover -s "$ROOT/tests" -v
python3 -m redlite.cli models
