# Upgrade an existing Red Lite install to 0.2.0

The in-place update archive contains only Red Lite source/config/docs/tests. It does
not contain or delete `.deps/` or `models/`, so an already-built backend and downloaded
GGUF can be retained.

From the existing `dwarfstar-red-lite` directory:

```bash
unzip -o /path/to/dwarfstar-red-lite-v0.2.0-in-place-update.zip -d .
./scripts/install.sh
redlite --version
redlite doctor
```

Expected version:

```text
redlite 0.2.0
```

No `redlite bootstrap` is required when the existing pinned v0.1.x engines are already
`READY`; v0.2 changes the Red Lite control plane and benchmarking policy, not the
engine pin used by that installation.

For the existing 24 GiB resident model:

```bash
redlite plan models/Qwen_Qwen3-Next-80B-A3B-Instruct-IQ2_XXS.gguf
redlite pressure
redlite sweep models/Qwen_Qwen3-Next-80B-A3B-Instruct-IQ2_XXS.gguf \
  --contexts 2048,4096,8192 \
  --output benchmarks/m4pro-24gb.json
```
