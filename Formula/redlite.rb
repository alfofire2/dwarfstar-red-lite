class Redlite < Formula
  desc "Native Metal runtime for Qwen3-Next-80B-A3B on Apple Silicon Macs"
  homepage "https://github.com/alfofire2/dwarfstar-red-lite"
  url "https://github.com/alfofire2/dwarfstar-red-lite/releases/download/v0.5.7/redlite-0.5.7-macos-arm64.tar.gz"
  sha256 "e6eb34de9082040e804aef37d241cdce7bd6c5b6a6b5846edfacb8d89517f5ba"
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
      PYTHONPATH="#{libexec}" exec "#{Formula["python@3.13"].opt_bin}/python3.13" -m redlite.cli "$@"
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
    assert_match version.to_s, shell_output("#{bin}/redlite --version")
    assert_match "redlite-generate", shell_output("#{bin}/redlite-generate --help 2>&1")
  end
end
