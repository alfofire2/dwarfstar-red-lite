# Third-party runtime policy

Dependencies are intentionally not vendored into this archive. `bootstrap_macos.sh`
clones exact commits so a build is reproducible and upstream license histories remain
intact.

Pinned on 2026-09-02:

- llama.cpp: `7798007a29a90e3053e799394da48cf53a2f8e0f`
- oversized-moe-runtime: `18815840f02bd860c2cbb5c6f0ff89e97a837229`

If you update a pin, rerun the functional and memory tests on actual Apple Silicon
before claiming the configuration is validated.
