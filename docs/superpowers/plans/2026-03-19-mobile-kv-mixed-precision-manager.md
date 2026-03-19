# Mobile KV Mixed-Precision Manager Hardening Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `mobilekv` production-ready for mobile inference by correctly supporting per-layer and per-plane precision, ring-buffer correctness, and extensible multi-format KV storage metadata.

**Architecture:** Keep the existing `KVTemplate + KVPlane + LayerStorage` layering, but harden semantics instead of rewriting the stack. Add strict correctness checks first (allocator, view offset, type safety, ring indexing), then extend format descriptors so K/V can carry precision and auxiliary metadata consistently. All behavior changes are gated by failing tests first and verified via focused + full test runs.

**Tech Stack:** C++17, CMake, GoogleTest, libc aligned allocation APIs

---

## Scope and Boundaries

This plan is one subsystem (KV cache manager core) and can ship incrementally. It intentionally does not include kernel integration or quant/dequant math kernels, only storage/layout/metadata correctness and API contracts.

Referenced process skills during execution:
- `@test-driven-development`
- `@root-cause-and-verification`
- `@verification-before-completion`

---

## File Structure (Target Changes)

- Modify: `include/mobilekv/kv_cache_basic.h`
- Modify: `src/kv_cache.cpp`
- Modify: `include/mobilekv/kv_cache_convenience.h`
- Modify: `test/kv_cache_test.cpp`
- Create: `test/kv_cache_ring_buffer_test.cpp`
- Create: `test/kv_cache_format_test.cpp`
- Modify: `example/convenience_api_example.cpp`
- Modify: `example/packed_layout_example.cpp`
- Modify: `README.md`
- Create: `docs/format-descriptor.md`

Responsibility split:
- `kv_cache_basic.h`: core API contracts and structs.
- `kv_cache.cpp`: runtime semantics (alloc/realloc/ring/view behavior).
- `kv_cache_convenience.h`: high-level constructors and typed access helper safety.
- `test/*`: behavior lock-in and regression protection.
- `example/* + README + docs/*`: user-visible contract and usage.

---

### Task 1: Lock Current Bugs With Failing Tests

**Files:**
- Modify: `test/kv_cache_test.cpp`
- Create: `test/kv_cache_ring_buffer_test.cpp`
- Create: `test/kv_cache_format_test.cpp`

- [ ] **Step 1: Add failing test for packed view offset correctness**
Add test asserting `acquire_seq_view(seq_begin=1, seq_len=1)` starts at `locate(LogicalCoord{...,seq=1,head=0,dim=0}).byte_offset` for packed layouts when contiguous export is allowed.

- [ ] **Step 2: Run single test and confirm failure**
Run: `./kv_cache_test --gtest_filter=*Packed*View*`
Expected: FAIL due to `bytes_for_tokens(seq_begin)` offset bug.

- [ ] **Step 3: Add failing test for convenience complex BF16/UINT8 preservation**
Add test building complex storage with `{k=BF16, v=UINT8}` and assert created layer plane types remain BF16/UINT8.

- [ ] **Step 4: Run single test and confirm failure**
Run: `./kv_cache_test --gtest_filter=*Complex*BF16*`
Expected: FAIL because fallback currently builds FP32 in default branches.

- [ ] **Step 5: Add failing test for type-safe accessor mismatch**
Add test: creating `KVAccessor<float>` over FP16 plane must throw `std::invalid_argument` (or explicit error API).

- [ ] **Step 6: Run single test and confirm failure**
Run: `./kv_cache_test --gtest_filter=*Accessor*TypeMismatch*`
Expected: FAIL because no runtime type check exists.

- [ ] **Step 7: Add failing ring-buffer window test**
Add test for max seq ring mode: after wrap, logical index `0` must map to oldest live token in current window (define and assert exact contract).

- [ ] **Step 8: Run single test and confirm failure**
Run: `./kv_cache_test --gtest_filter=*Ring*Window*`
Expected: FAIL because locate/index translation is currently linear only.

- [ ] **Step 9: Commit tests**
```bash
git add test/kv_cache_test.cpp test/kv_cache_ring_buffer_test.cpp test/kv_cache_format_test.cpp
git commit -m "test: add failing coverage for packed view, ring window, format and accessor safety"
```

---

### Task 2: Fix Allocator Ownership and Alignment Contract

**Files:**
- Modify: `src/kv_cache.cpp`
- Modify: `include/mobilekv/kv_cache_basic.h`
- Test: `test/kv_cache_test.cpp`

- [ ] **Step 1: Add failing regression test for repeated reserve/resize lifecycle**
Test loop reserve/resize/clear/destruct in many iterations to catch allocator misuse under ASan-friendly flow.

- [ ] **Step 2: Run targeted test to verify failure/risk visibility**
Run: `./kv_cache_test --gtest_filter=*Reserve*Lifecycle*`
Expected: Fails or is marked unstable before fix in sanitizer builds.

- [ ] **Step 3: Implement aligned buffer RAII with proper deleter**
Replace `std::unique_ptr<Byte[]>` with:
```cpp
using AlignedBytePtr = std::unique_ptr<Byte, void(*)(void*)>;
```
allocate with `aligned_alloc`, free with `std::free`, and keep null-safe move semantics.

- [ ] **Step 4: Add alignment validation helper**
Validate alignment is non-zero power-of-two and `size % alignment == 0` before alloc; return `false` on invalid config.

- [ ] **Step 5: Run targeted tests**
Run: `./kv_cache_test --gtest_filter=*Reserve*:*Resize*`
Expected: PASS.

- [ ] **Step 6: Commit allocator fix**
```bash
git add src/kv_cache.cpp include/mobilekv/kv_cache_basic.h test/kv_cache_test.cpp
git commit -m "fix: use aligned allocation RAII with free and strict alignment checks"
```

---

### Task 3: Make Ring Buffer Semantics Correct and Explicit

**Files:**
- Modify: `include/mobilekv/kv_cache_basic.h`
- Modify: `src/kv_cache.cpp`
- Test: `test/kv_cache_ring_buffer_test.cpp`
- Modify: `README.md`

- [ ] **Step 1: Define ring-buffer logical index contract in tests**
Define: API-visible `seq` is window-local `[0, seq_length)`, where `0` is oldest live token.

- [ ] **Step 2: Run ring-buffer tests to verify failure**
Run: `./kv_cache_test --gtest_filter=*Ring*`
Expected: FAIL prior to translation changes.

- [ ] **Step 3: Extend plane runtime metadata for ring translation**
Add fields (for example): `window_start`, `write_head`, `is_ring_buffer` and helper:
```cpp
uint32_t logical_to_physical_seq(uint32_t local_seq) const;
```

- [ ] **Step 4: Route `locate()` through ring translation**
If ring mode is active, translate local seq to physical seq before calling template `locate`.

- [ ] **Step 5: Fix `acquire_seq_view()` behavior in ring mode**
When requested range crosses wrap boundary, return non-contiguous view; when not crossing and template allows, return contiguous view with translated begin offset.

- [ ] **Step 6: Run ring-focused tests**
Run: `./kv_cache_test --gtest_filter=*Ring*:*AcquireSeqView*`
Expected: PASS.

- [ ] **Step 7: Commit ring semantic fix**
```bash
git add include/mobilekv/kv_cache_basic.h src/kv_cache.cpp test/kv_cache_ring_buffer_test.cpp README.md
git commit -m "fix: implement explicit ring-buffer logical-to-physical seq mapping"
```

---

### Task 4: Correct Packed View Offset and Range Export Logic

**Files:**
- Modify: `src/kv_cache.cpp`
- Modify: `include/mobilekv/kv_cache_basic.h`
- Test: `test/kv_cache_test.cpp`
- Modify: `example/packed_layout_example.cpp`

- [ ] **Step 1: Add failing packed contiguous-range tests**
Add tests for begin offsets across different `seq_begin` values in packed mode.

- [ ] **Step 2: Run targeted tests and verify failure**
Run: `./kv_cache_test --gtest_filter=*Packed*View*`
Expected: FAIL due to current `bytes_for_tokens(seq_begin)` usage.

- [ ] **Step 3: Introduce byte-range helper contract**
Add API method on template or plane-side helper:
```cpp
bool seq_span_byte_range(uint32_t seq_begin, uint32_t seq_len, size_t& offset, size_t& bytes) const;
```
Use `locate()`-based offset derivation, not capacity-byte math.

- [ ] **Step 4: Implement correct offset/bytes in `acquire_seq_view()`**
Use helper result for contiguous case; otherwise return non-contiguous fallback.

- [ ] **Step 5: Fix example underflow output**
In packed example, print signed difference or branch on `plain_bytes >= packed_bytes` to avoid `size_t` underflow display.

- [ ] **Step 6: Run tests**
Run: `./kv_cache_test --gtest_filter=*Packed*`
Expected: PASS.

- [ ] **Step 7: Commit packed view fix**
```bash
git add src/kv_cache.cpp include/mobilekv/kv_cache_basic.h test/kv_cache_test.cpp example/packed_layout_example.cpp
git commit -m "fix: correct packed view offset/range export and example byte diff display"
```

---

### Task 5: Enforce K/V Type Safety and Complete Convenience API Precision Matrix

**Files:**
- Modify: `include/mobilekv/kv_cache_convenience.h`
- Modify: `test/kv_cache_format_test.cpp`
- Modify: `example/convenience_api_example.cpp`

- [ ] **Step 1: Add failing tests for full precision matrix**
Cover `FP32/FP16/BF16/INT8/UINT8/INT16` for K and V in convenience APIs and assert no silent fallback.

- [ ] **Step 2: Run targeted tests and confirm failure**
Run: `./kv_cache_test --gtest_filter=*ComplexStorage*:*PrecisionMatrix*`
Expected: FAIL for unsupported branches currently defaulting to FP32.

- [ ] **Step 3: Remove dead mapping path and implement deterministic template cache**
Replace current `type_pair_to_id` misuse with explicit maps:
- `std::unordered_map<ScalarType, TemplateId> k_type_to_id`
- `std::unordered_map<ScalarType, TemplateId> v_type_to_id`
Never fallback silently; unsupported types throw.

- [ ] **Step 4: Add runtime check for `KVAccessor<T>`**
Map `T` to `ScalarType` and validate against plane template scalar type in constructor; mismatch throws.

- [ ] **Step 5: Update examples to use correct accessor element types**
FP16/BF16 should use `uint16_t` (raw storage) or dedicated half wrapper, not `float`.

- [ ] **Step 6: Run tests**
Run: `./kv_cache_test --gtest_filter=*Accessor*:*ComplexStorage*`
Expected: PASS.

- [ ] **Step 7: Commit convenience/type-safety fixes**
```bash
git add include/mobilekv/kv_cache_convenience.h test/kv_cache_format_test.cpp example/convenience_api_example.cpp
git commit -m "fix: complete precision matrix and enforce KVAccessor runtime type safety"
```

---

### Task 6: Add Multi-Format Descriptor (Storage Metadata Only)

**Files:**
- Modify: `include/mobilekv/kv_cache_basic.h`
- Modify: `src/kv_cache.cpp`
- Create: `docs/format-descriptor.md`
- Modify: `README.md`
- Test: `test/kv_cache_format_test.cpp`

- [ ] **Step 1: Add failing tests for format metadata persistence**
Test that per-template format descriptor (e.g., quant block size, scale type, zero-point mode) is stored and retrievable at runtime.

- [ ] **Step 2: Run targeted tests and confirm failure**
Run: `./kv_cache_test --gtest_filter=*FormatDescriptor*`
Expected: FAIL before descriptor fields are added.

- [ ] **Step 3: Add `FormatDescriptor` and bind to `TemplateConfig`**
Add struct fields only for metadata, for example:
```cpp
enum class QuantScheme { None, PerTensorAffine, PerChannelSymmetric, PerGroupAffine };
struct FormatDescriptor {
  QuantScheme scheme;
  uint32_t group_size;
  ScalarType storage_type;
  ScalarType scale_type;
  bool has_zero_point;
};
```

- [ ] **Step 4: Ensure descriptor is available through existing template queries**
No kernel math; only contract propagation and validation.

- [ ] **Step 5: Document supported formats and unsupported semantics**
Explicitly state: storage metadata only, no quant/dequant compute in this library.

- [ ] **Step 6: Run tests**
Run: `./kv_cache_test --gtest_filter=*FormatDescriptor*:*MixedPrecision*`
Expected: PASS.

- [ ] **Step 7: Commit format descriptor**
```bash
git add include/mobilekv/kv_cache_basic.h src/kv_cache.cpp test/kv_cache_format_test.cpp README.md docs/format-descriptor.md
git commit -m "feat: add format descriptor metadata for multi-format KV storage"
```

---

### Task 7: Final Verification and Release Readiness

**Files:**
- Modify: `README.md`
- Modify: `CMakeLists.txt` (only if new test files need wiring)

- [ ] **Step 1: Run full unit tests**
Run: `./kv_cache_test`
Expected: All tests PASS.

- [ ] **Step 2: Run examples**
Run:
- `./mobilekv_demo`
- `./mobilekv_mixed_precision_demo`
- `./mobilekv_packed_layout_demo`
- `./mobilekv_convenience_demo`
Expected: All examples exit `0` with no obvious precision mismatch logs.

- [ ] **Step 3: Build benchmark target**
Run: `cmake --build . -j --target mobilekv_benchmark`
Expected: Build PASS.

- [ ] **Step 4: Verify docs consistency**
Check that README and `docs/format-descriptor.md` match runtime behavior and test coverage.

- [ ] **Step 5: Commit docs/release notes**
```bash
git add README.md docs/format-descriptor.md CMakeLists.txt
git commit -m "docs: finalize mobile mixed-precision and multi-format KV manager contract"
```

---

## Definition of Done

- Allocator lifecycle is memory-safe and deterministic.
- Ring-buffer mode has explicit and tested logical/physical mapping.
- Packed range/view offsets are correct for all tested spans.
- Convenience API preserves requested K/V precision combinations without silent downgrade.
- Accessor type mismatch is rejected at runtime.
- Multi-format descriptor metadata is part of the public template contract.
- Unit tests + examples pass locally.

---

## Execution Notes

- Keep each commit scoped to one task.
- Do not introduce kernel-coupled logic in this plan.
- Prefer minimal API surface additions; document all behavior changes.

