# Notices and attribution

DwarfStar Red Lite is an independent, unofficial project. It is inspired by the
narrow, model-specific design and SSD-streaming philosophy of Salvatore
Sanfilippo's DwarfStar/DS4 project. No DwarfStar source code is included in this
repository.

Runtime dependencies fetched by `scripts/bootstrap_macos.sh`:

- `ggml-org/llama.cpp` — MIT licensed. Used as the Qwen3-Next Metal inference
  engine for resident models.
- `dimitripavlov/oversized-moe-runtime` — MIT licensed. Used for validated
  CPU-only mmap + bounded expert residency when the GGUF exceeds physical RAM.

Those projects retain their own copyright notices and license files in `.deps/`
after bootstrap. DwarfStar, llama.cpp, Qwen, Hugging Face, Apple and other names
belong to their respective owners. This project is not endorsed by them.
