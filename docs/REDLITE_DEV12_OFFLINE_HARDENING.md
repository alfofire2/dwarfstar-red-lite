# Red Lite dev12 — offline native hardening

Dev12 intentionally avoids adding new Qwen graph functionality while the target M4 Pro is unavailable. Instead it hardens the standalone native runtime in areas that can be verified completely in CI without the 18 GiB model.

## Transactional cache publication

The native top-k LRU now separates selection planning from publication:

1. `rl_native_lru_prepare_many()` resolves hits and reserves unique free/victim slots without changing published cache metadata;
2. expert miss payloads are read into those reserved slots;
3. only after every load succeeds does `rl_native_lru_commit()` publish all new `(layer, expert) -> slot` mappings and update hit/miss/eviction counters;
4. if any load fails, `rl_native_lru_abort()` invalidates touched miss slots rather than restoring metadata for expert bytes that may have been partially or fully overwritten.

This prevents the native runtime from advertising a resident expert whose Metal slot contains an incomplete replacement payload.

The existing metadata-only `rl_native_lru_acquire_many()` remains as a convenience wrapper implemented as prepare + immediate commit.

## Synthetic GGUF integration test

`redlite-native-offline-test` creates a valid temporary GGUF v3 file directly in C with:

- `general.alignment = 32`;
- one routed layer;
- gate/up tensors shaped `(256, 8, 4)`;
- down tensor shaped `(8, 256, 4)`;
- four experts;
- IQ2_XS type 17;
- equal 32-byte physical slices per tensor/expert.

The test then runs the production native GGUF parser and verifies:

- GGUF version/alignment/tensor count;
- three routed tensors and one routed layer;
- safe outer expert slicing;
- total routed payload and max expert triplet sizing;
- `hidden_size = 256`, `ffn_size = 8`;
- expert-2 gate/up/down offsets and 32-byte slice lengths.

This exercises parser -> ExpertMap -> layer-info -> expert-layout without Python and without downloading a real model.

## Transaction failure-path test

The same offline binary creates a two-slot native LRU, prepares a selection containing one hit and one miss, and verifies that:

- prepare does not increment public counters;
- the uncommitted miss cannot be looked up;
- the previous victim remains visible until an overwrite is assumed;
- abort invalidates the touched victim so stale metadata cannot survive a failed load;
- a subsequent prepare + commit successfully publishes the new selection.

## CI

`make native` now builds and runs both:

```text
redlite-native selftest
redlite-native-offline-test
```

on macOS and Linux. The macOS `redlite-native` build continues to link the standalone Red Metal top-k path directly; the offline test itself is portable C.

## Validation boundary

Dev12 does not change the Metal arithmetic, router fixtures or Qwen model graph. Real-model validation is still required for dev9-dev12 when the target M4 Pro becomes available again.

Until that field validation succeeds, the next production feature — the actual Qwen3-Next router network — remains intentionally deferred.
