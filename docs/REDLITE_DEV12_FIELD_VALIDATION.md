# Red Lite dev12 — standalone native field validation

Target field system: Apple M4 Pro, 24 GiB unified memory.
Model: `Qwen_Qwen3-Next-80B-A3B-Instruct-IQ2_XXS.gguf`.

This record closes the dev9-dev12 migration from the Python-assisted development harness to the standalone native C + Objective-C/Metal path.

## Native GGUF / ExpertMap

`redlite-native inspect` on the real 843-tensor GGUF reported:

- GGUF v3, alignment 32;
- 843 tensors, 49 metadata entries;
- 144 routed expert tensors across 48 layers;
- 512 experts;
- slice-safe outer expert axis: YES;
- routed payload: 16.910 GiB;
- 33 IQ2_XS routed tensors and 111 IQ1_M routed tensors;
- 11 all-IQ2_XS routed layers and 37 all-IQ1_M routed layers;
- layers 0-5 and 43-47: IQ2_XS gate/up/down;
- layers 6-42: IQ1_M gate/up/down;
- max expert gate+up+down triplet: 0.867 MiB;
- 256 MiB cache: 295 aligned slots, top-10 fits YES;
- layer 0 probe: hidden 2048, expert FFN 512.

These values exactly reproduce the earlier Python quant audit / ExpertMap observations.

## Standalone IQ2_XS routed top-10 parity

Layer 0, deterministic non-contiguous top-10 fixture:

- runtime: native C + Objective-C/Metal; no Python/ctypes;
- resident slots: 10/295;
- expert loads: 10;
- cache hits/misses: 0/10;
- evictions: 0;
- SSD read: 8.672 MiB;
- positional reads: 30 (10 experts x gate/up/down);
- SSD reads during Metal top-k command: 0 bytes / 0 calls;
- GPU selected-row top-k time: 9.254 ms;
- native CPU reference: 19.896 ms;
- max absolute error: 1.95297e-08;
- max relative error: 4.72804e-06;
- parity: YES.

## Standalone IQ1_M routed top-10 parity

Layer 6, same deterministic top-10 fixture:

- runtime: native C + Objective-C/Metal; no Python/ctypes;
- resident slots: 10/295;
- expert loads: 10;
- cache hits/misses: 0/10;
- evictions: 0;
- SSD read: 6.562 MiB;
- positional reads: 30;
- SSD reads during Metal top-k command: 0 bytes / 0 calls;
- GPU selected-row top-k time: 8.658 ms;
- native CPU reference: 24.724 ms;
- max absolute error: 1.7304e-08;
- max relative error: 1.31451e-04 (dominated by a near-zero output row);
- parity: YES.

## Conclusion

The standalone native path is field-validated for the routed expert subsystem on both real routed quant formats:

`GGUF -> native parser -> native ExpertMap -> transactional native LRU -> pread -> shared Metal residency -> complete expert FFN -> GPU weighted top-k accumulation -> native CPU oracle`.

No Python interpreter or ctypes boundary is involved in this validated execution path.

The remaining top-k ids and weights are deterministic fixtures. The next correctness boundary is the actual Qwen3-Next router network.
