#!/usr/bin/env python3
"""dev57: write Formula/redlite.rb for a release (URL and SHA-256 of the release tarball).

    python3 scripts/dev/brew_formula.py 0.5.5                  # hashes dist/redlite-0.5.5-macos-arm64.tar.gz
    python3 scripts/dev/brew_formula.py 0.5.5 --url-base file:///dir --out /tmp/redlite.rb   # local test

The repository is its own tap: brew tap alfofire2/redlite https://github.com/alfofire2/dwarfstar-red-lite
The formula installs the three native binaries and the stdlib-only `redlite` CLI from the tarball, plus `hf`
for `redlite download`.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = "alfofire2/dwarfstar-red-lite"

TEMPLATE = '''class Redlite < Formula
  desc "Native Metal runtime for Qwen3-Next-80B-A3B on Apple Silicon Macs"
  homepage "https://github.com/{repo}"
  url "{url}"
  version "{version}"
  sha256 "{sha}"
  license "MIT"

  depends_on arch: :arm64
  depends_on "hf"
  depends_on :macos
  depends_on "python@3.13"

  def install
    bin.install Dir["bin/redlite-*"]
    libexec.install "python/redlite"
    (bin/"redlite").write <<~EOS
      #!/bin/sh
      PYTHONPATH="#{{libexec}}" exec "#{{Formula["python@3.13"].opt_bin}}/python3.13" -m redlite.cli "$@"
    EOS
  end

  def caveats
    <<~EOS
      Models go to ~/.redlite/models (set REDLITE_MODELS to change it):
        redlite download 24gb        # 19.3 GB, the Red Lite F2 file (48gb: IQ3_XXS for 48 GiB Macs)
        redlite download mtp         # 2.4 GB, the MTP head (faster decode, same output)
        redlite chat
      On a 24 GiB Mac, `redlite doctor` prints the GPU limit that keeps every expert resident.
    EOS
  end

  test do
    assert_match version.to_s, shell_output("#{{bin}}/redlite --version")
    assert_match "redlite-generate", shell_output("#{{bin}}/redlite-generate --help 2>&1")
  end
end
'''


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("version")
    ap.add_argument("--tarball", help="default: dist/redlite-VERSION-macos-arm64.tar.gz")
    ap.add_argument("--url-base", help="override the URL's base (local tests: file:///dir)")
    ap.add_argument("--out", default=str(ROOT / "Formula" / "redlite.rb"))
    a = ap.parse_args()
    tarball = Path(a.tarball or ROOT / "dist" / f"redlite-{a.version}-macos-arm64.tar.gz")
    base = a.url_base or f"https://github.com/{REPO}/releases/download/v{a.version}"
    # the version is explicit: from "redlite-X-macos-arm64.tar.gz" Homebrew guesses "64" (seen on the M4 Pro, dev57c)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(TEMPLATE.format(repo=REPO, url=f"{base}/{tarball.name}", version=a.version, sha=sha256(tarball)))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
