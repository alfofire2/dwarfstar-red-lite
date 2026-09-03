# Notices and attribution

DwarfStar Red Lite is an independent, unofficial project. It is inspired by the
narrow, model-specific design and SSD-streaming philosophy of Salvatore
Sanfilippo's DwarfStar/DS4 project. No DwarfStar source code is included in this
repository.

Runtime dependencies fetched by `scripts/bootstrap_macos.sh`:

- `ggml-org/llama.cpp` — MIT licensed. Used as the Qwen3-Next Metal inference
  engine for resident models and as the canonical reference for GGUF/quantization
  formats during Red Metal development.
- `dimitripavlov/oversized-moe-runtime` — MIT licensed. Used for validated
  CPU-only mmap + bounded expert residency when the GGUF exceeds physical RAM.

Red Lite also retains a small source-level adaptation of canonical IQ2_XS and
IQ1_S/IQ1_M quantization grid data from the pinned llama.cpp/GGML implementation.
The compact grid literals are expanded by the standalone native runtime and used
by the Red Metal kernels. These adapted data remain subject to the upstream MIT
license and copyright notices; the source file identifies the pinned upstream
commit from which they were derived.

The native tokenizer's Unicode category, whitespace and lowercase tables
(`native/redlite_native_unicode_data.h`) are generated from the pinned
llama.cpp `src/unicode-data.cpp` by `scripts/dev/gen_unicode_data.py` and remain
subject to the upstream MIT license and copyright notices.

Those projects retain their own copyright notices and license files in `.deps/`
after bootstrap. DwarfStar, llama.cpp, Qwen, Hugging Face, Apple and other names
belong to their respective owners. This project is not endorsed by them.
