#!/usr/bin/env bash
# Build a release tarball of the native Red Lite binaries for Apple Silicon (arm64 macOS).
#
#   scripts/package_release.sh [--out DIR] [--mcpu CPU] [--no-build]
#
# The binaries are rebuilt from the current checkout into a staging directory (the development
# build in .deps/redmetal is not touched) with -mcpu=apple-m1 by default, so the tarball does not
# depend on instructions newer than the first Apple Silicon generation (the development build uses
# -mcpu=native). The Metal kernels are compiled from source at run time on the target GPU.
# Output (default DIR = dist/):
#   redlite-<version>-macos-arm64.tar.gz          bin/redlite-generate, bin/redlite-server, bin/redlite-engine,
#                                                 INSTALL.md, LICENSE, NOTICE.md, VERSION, BUILDINFO
#   redlite-<version>-macos-arm64.tar.gz.sha256   `shasum -a 256` line, checked with `shasum -a 256 -c`
# The model file is never packaged.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUTDIR="$ROOT/dist"; MCPU="apple-m1"; BUILD=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) OUTDIR="$2"; shift 2;;
    --mcpu) MCPU="$2"; shift 2;;
    --no-build) BUILD=0; shift;;
    -h|--help) sed -n '2,16p' "$0"; exit 0;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
done
if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "package_release.sh builds the Metal runtime: it needs an Apple Silicon Mac" >&2
  exit 2
fi

# without it clang targets the SDK of the build machine and the binaries refuse to start on older macOS
export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-14.0}"
VERSION="$(tr -d '[:space:]' <"$ROOT/VERSION")"
NAME="redlite-$VERSION-macos-arm64"
STAGE="$ROOT/.deps/package/$NAME"
BUILD_DIR="$ROOT/.deps/package/build"
rm -rf "$STAGE"
mkdir -p "$STAGE/bin" "$BUILD_DIR" "$OUTDIR"

if [[ "$BUILD" == 1 ]]; then
  echo "== building with -mcpu=$MCPU, macOS >= $MACOSX_DEPLOYMENT_TARGET, into $BUILD_DIR =="
  REDLITE_MCPU="$MCPU" REDLITE_BUILD_OUT="$BUILD_DIR" bash "$ROOT/scripts/build_engine.sh"
  REDLITE_MCPU="$MCPU" REDLITE_BUILD_OUT="$BUILD_DIR" bash "$ROOT/scripts/build_server.sh"
fi
for b in redlite-generate redlite-server redlite-engine; do
  [[ -x "$BUILD_DIR/$b" ]] || { echo "missing $BUILD_DIR/$b (run without --no-build)" >&2; exit 1; }
  cp "$BUILD_DIR/$b" "$STAGE/bin/$b"
done
"$STAGE/bin/redlite-generate" --help >/dev/null
"$STAGE/bin/redlite-server" --help >/dev/null

cp "$ROOT/LICENSE" "$ROOT/NOTICE.md" "$ROOT/VERSION" "$STAGE/"
# dev57: the Python launcher (pure stdlib) travels with the binaries, for Homebrew and tarball users
mkdir -p "$STAGE/python" && cp -R "$ROOT/redlite" "$STAGE/python/" && find "$STAGE/python" -name __pycache__ -prune -exec rm -rf {} +
COMMIT="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
git -C "$ROOT" diff --quiet 2>/dev/null || COMMIT="$COMMIT+dirty"
{
  echo "version  $VERSION"
  echo "commit   $COMMIT"
  echo "built    $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "mcpu     $MCPU"
  echo "minos    $MACOSX_DEPLOYMENT_TARGET"
  echo "machine  $(sysctl -n machdep.cpu.brand_string 2>/dev/null || uname -m)"
  echo "macos    $(sw_vers -productVersion 2>/dev/null || echo unknown)"
  echo "clang    $(xcrun --sdk "${REDLITE_SDK:-macosx}" clang --version | head -1)"
} >"$STAGE/BUILDINFO"
cat >"$STAGE/INSTALL.md" <<EOF
# Red Lite $VERSION native runtime (macOS arm64)

Binaries of the native Red Lite runtime for Qwen3-Next-80B-A3B GGUF files on Apple Silicon
(C / Objective-C / Metal, no Python and no llama.cpp at run time). Built with
\`-mcpu=$MCPU\` for macOS $MACOSX_DEPLOYMENT_TARGET or later; the Metal kernels are compiled on your GPU at start-up. See BUILDINFO for the
commit and the build machine.

## Verify and install

    shasum -a 256 -c $NAME.tar.gz.sha256
    tar -xzf $NAME.tar.gz
    sudo install -m 0755 $NAME/bin/redlite-* /usr/local/bin/     # or add $NAME/bin to PATH

Binaries downloaded with a browser carry the quarantine attribute; clear it with
\`xattr -dr com.apple.quarantine $NAME\` if macOS refuses to run them.

## Model

The model is not included. The reference file is Bartowski's
\`Qwen_Qwen3-Next-80B-A3B-Instruct-IQ2_XXS.gguf\` from
https://huggingface.co/bartowski/Qwen_Qwen3-Next-80B-A3B-Instruct-GGUF (put it in \`models/\`).

## Run

    redlite-generate MODEL.gguf --prompt "Hello" --cache-mib 4096       # 24 GiB Macs: 4 GiB expert cache
    redlite-generate MODEL.gguf --prompt "Hello" --cache-mib 22528      # >= 40 GiB: every expert resident
    redlite-server MODEL.gguf --port 8080 --cache-mib 4096              # OpenAI /v1/chat/completions (SSE)

\`redlite-generate --help\` and \`redlite-server --help\` list every option. The Python launcher
(\`redlite chat\`, \`redlite serve --native\`, Python 3.10 or later) is in \`python/\`; with the binaries on PATH:

    PYTHONPATH=$NAME/python python3 -m redlite.cli chat     # models in ~/.redlite/models (REDLITE_MODELS)

Homebrew does all of this: \`brew tap alfofire2/redlite https://github.com/alfofire2/dwarfstar-red-lite && brew install redlite\`.
EOF

tar -C "$(dirname "$STAGE")" -czf "$OUTDIR/$NAME.tar.gz" "$NAME"
(cd "$OUTDIR" && shasum -a 256 "$NAME.tar.gz" >"$NAME.tar.gz.sha256" && shasum -a 256 -c "$NAME.tar.gz.sha256")
echo "Packaged $OUTDIR/$NAME.tar.gz ($(du -h "$OUTDIR/$NAME.tar.gz" | cut -f1))"
