#!/bin/bash
set -euo pipefail
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
python3 -m pip install -e "$ROOT"
USER_BIN="$(python3 -m site --user-base)/bin"
if [[ ":${PATH}:" != *":${USER_BIN}:"* ]]; then
  echo
  echo "redlite was installed, but ${USER_BIN} is not on PATH."
  echo "For zsh, run:"
  echo "  echo 'export PATH=\"${USER_BIN}:\$PATH\"' >> ~/.zshrc && source ~/.zshrc"
  echo "Until then you can use: $ROOT/bin/redlite"
else
  echo "Installed redlite CLI. Next: redlite doctor"
fi
