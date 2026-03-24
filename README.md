# MobileKV

MobileKV is a KV cache storage manager for edge/mobile LLM inference.
It focuses on memory layout, allocation, addressing, and window retention.
It does not run model kernels or quant/dequant math.

## Current Scope

MobileKV currently provides:

- Multi-layer KV storage (`LayerStorage`) with per-layer K/V template binding
- Mixed precision per plane (`K` and `V` can use different scalar types)
- Ring-buffer mode for sliding-window retention
- Plain and dim-block storage templates
- Opaque-scalar registry for dim-block template construction
- Raw-typed convenience accessors with runtime scalar-type checks
- Format metadata descriptor (`FormatDescriptor`) attached to templates

Supported scalar types in core templates:

- `FP32`, `FP16`, `BF16`, `INT8`, `UINT8`, `INT16`

## What It Does Not Do

Current implementation explicitly does not include:

- Attention kernel execution
- Quantization/dequantization compute
- Disk-backed KV runtime implementation
- Block-table/page-table scheduler APIs for large-scale continuous batching

`StorageMode::Blocked`, `MemoryDomain::DiskMapped`, and `thread_safe` flags exist in API surface for future extension, but are not fully implemented runtime features yet.

## Core Concepts

- `KVTemplate`: Defines layout mapping from logical coordinates to physical bytes.
- `KVPlane`: Concrete storage instance (one `K` or one `V`) for a layer.
- `LayerStorage`: Owns `K/V` planes for one layer.
- `KVCacheStorage`: Global container for all layers and templates.

Logical coordinate:

- `LogicalCoord{layer, seq, head, dim}`

Physical address:

- `PhysicalAddr{byte_offset, byte_size, ...}`

## Ring Buffer and Sliding Window Semantics

When ring mode is enabled (`max_seq_capacity > 0`):

- Plane keeps only the latest window up to `max_seq_capacity`.
- `locate()` interprets `seq` as a window-local logical index in `[0, seq_length)`.
- Internal logic maps logical index to physical slot after wrap-around.
- Overflow append overwrites oldest tokens.

Strict validation:

- Build fails if `initial_seq_capacity > max_seq_capacity` (no silent clamp).

## Default Ring Configuration for Edge

`KVCacheStorageConfig` includes:

- `default_max_seq_capacity`

Behavior:

- 4-arg `add_layer(layer, k_template, v_template, initial)` inherits `config.default_max_seq_capacity`.
- 5-arg `add_layer(..., initial, max)` explicitly overrides it.
- Set `default_max_seq_capacity=0` to keep non-ring growth behavior.

## Quick Start

```cpp
#include "mobilekv/kv_cache.h"

using namespace mobilekv;

KVCacheStorageBuilder builder;
builder.config({64, false, 2048});  // default ring window for 4-arg add_layer

auto k_fp16 = std::make_shared<PlainKVTemplate<ScalarType::FP16>>(8, 128, 1, "k_fp16");
auto v_int8 = std::make_shared<PlainKVTemplate<ScalarType::INT8>>(8, 128, 2, "v_int8");

builder.add_template(k_fp16);
builder.add_template(v_int8);

// Inherits default_max_seq_capacity=2048
builder.add_layer(0, 1, 2, 1024);

auto storage = builder.build();
if (!storage) {
    // Build can fail on invalid config or missing templates.
    return;
}

auto& layer0 = storage->layer(0);
auto& k_plane = layer0.plane(PlaneKind::K);

k_plane.append_seq(1);  // ring append if max_seq_capacity > 0
auto addr = k_plane.locate(LogicalCoord(0, 0, 0, 0));
```

## Typical Flow: Prefill

Prefill typically writes a prompt chunk (`N` tokens) into every layer's `K/V` planes, then commits length.

```cpp
constexpr uint32_t NUM_LAYERS = 4;
constexpr uint32_t NUM_HEADS = 8;
constexpr uint32_t HEAD_DIM = 128;
constexpr uint32_t PROMPT_LEN = 16;
constexpr size_t TOKEN_ELEMS = static_cast<size_t>(NUM_HEADS) * HEAD_DIM;

auto storage = create_fp32_storage(NUM_LAYERS, NUM_HEADS, HEAD_DIM, 2048);
if (!storage) return;

// Reserve once for prefill.
storage->reserve_all(PROMPT_LEN);

// Your runtime output buffers for one token.
float src_k[TOKEN_ELEMS];
float src_v[TOKEN_ELEMS];

for (uint32_t layer = 0; layer < NUM_LAYERS; ++layer) {
    auto& layer_storage = storage->layer(layer);
    auto& k_plane = layer_storage.plane(PlaneKind::K);
    auto& v_plane = layer_storage.plane(PlaneKind::V);

    float* k_ptr = static_cast<float*>(k_plane.data());
    float* v_ptr = static_cast<float*>(v_plane.data());

    // Write prompt token-by-token (or bulk memcpy from your runtime buffer).
    for (uint32_t t = 0; t < PROMPT_LEN; ++t) {
        size_t token_offset = static_cast<size_t>(t) * TOKEN_ELEMS;
        // Fill src_k/src_v with model output for token t before memcpy.
        std::memcpy(k_ptr + token_offset, src_k, TOKEN_ELEMS * sizeof(float));
        std::memcpy(v_ptr + token_offset, src_v, TOKEN_ELEMS * sizeof(float));
    }

    // Commit visible length for this layer.
    k_plane.resize_seq(PROMPT_LEN);
    v_plane.resize_seq(PROMPT_LEN);
}
```

## Typical Flow: Decode

Decode typically appends one token per step, writes only the newest token slot, then reads a window for attention.

```cpp
constexpr uint32_t DECODE_STEPS = 32;
constexpr uint32_t TOKEN_STRIDE = NUM_HEADS * HEAD_DIM;
float step_k[TOKEN_STRIDE];
float step_v[TOKEN_STRIDE];

for (uint32_t step = 0; step < DECODE_STEPS; ++step) {
    // 1) Grow all layers by one token.
    storage->append_all(1);

    // 2) Write newly generated K/V for the newest logical token.
    for (uint32_t layer = 0; layer < NUM_LAYERS; ++layer) {
        auto& layer_storage = storage->layer(layer);
        auto& k_plane = layer_storage.plane(PlaneKind::K);
        auto& v_plane = layer_storage.plane(PlaneKind::V);

        uint32_t newest = k_plane.stats().seq_length - 1;  // logical index in current window

        auto k_addr = k_plane.locate(LogicalCoord(layer, newest, 0, 0));
        auto v_addr = v_plane.locate(LogicalCoord(layer, newest, 0, 0));

        float* k_base = reinterpret_cast<float*>(static_cast<Byte*>(k_plane.data()) + k_addr.byte_offset);
        float* v_base = reinterpret_cast<float*>(static_cast<Byte*>(v_plane.data()) + v_addr.byte_offset);

        // Fill step_k/step_v with model output for this decode step before memcpy.
        std::memcpy(k_base, step_k, TOKEN_STRIDE * sizeof(float));
        std::memcpy(v_base, step_v, TOKEN_STRIDE * sizeof(float));
    }

    // 3) Acquire attention window view (window-local logical indices).
    auto& layer0_k = storage->layer(0).plane(PlaneKind::K);
    uint32_t len = layer0_k.stats().seq_length;
    uint32_t win = std::min<uint32_t>(len, 512);
    uint32_t begin = len - win;
    AccessView view = layer0_k.acquire_seq_view(begin, win, AccessMode::ReadOnly);
    layer0_k.release_seq_view(view);
}
```

Notes:

- In ring mode, `seq` indices are always window-local `[0, seq_length)`.
- `locate()` already resolves logical-to-physical mapping after wrap-around.
- If `acquire_seq_view(...)` returns `contiguous=false`, treat it as non-contiguous window access in your runtime path.

## Convenience APIs

`include/mobilekv/kv_cache_convenience.h` provides:

- `create_simple_storage(...)`
- `create_complex_storage(...)`
- `create_fp32_storage(...)`, `create_fp16_storage(...)`, `create_int8_storage(...)`
- `KVAccessor<T>` with runtime scalar-type validation

Accessor compatibility:

- `KVAccessor<float>` -> `FP32`
- `KVAccessor<uint16_t>` -> `FP16` / `BF16` raw storage bits
- `KVAccessor<int8_t>` -> `INT8`
- `KVAccessor<uint8_t>` -> `UINT8`
- `KVAccessor<int16_t>` -> `INT16`

## Format Descriptor

Template config contains `FormatDescriptor` metadata to describe storage format for downstream runtime/kernel decisions.

See:

- [docs/format-descriptor.md](docs/format-descriptor.md)

This is metadata only; MobileKV does not perform quant/dequant compute.

## Opaque Scalar Registry (Dim-Block)

For pre-packed D-last storage (for example `pack4` in `[H, S, D/4]` block space), use builder-level opaque scalar registration.
This is the recommended end-to-end path:

```cpp
KVCacheStorageBuilder builder;
builder.config({64, false, 2048});

OpaqueScalarId pack4 = builder.register_opaque_scalar({"int8_pack4", 4, 4});
if (pack4 == 0) {
    return;  // invalid descriptor
}

auto k_t = builder.make_dim_block_template(32, 128 / 4, pack4, 1, "k_pack4");
auto v_t = builder.make_dim_block_template(32, 128 / 4, pack4, 2, "v_pack4");
if (!k_t || !v_t) {
    return;  // unknown scalar id
}

builder.add_template(k_t);
builder.add_template(v_t);
builder.add_layer(0, 1, 2, 1024, 2048);

auto storage = builder.build();
if (!storage) {
    return;  // invalid layer/template wiring
}
```

Notes:

- `register_opaque_scalar(...)` returns `0` on invalid descriptor (`bytes==0` or `alignment==0`).
- `make_dim_block_template(...)` returns `nullptr` for unknown scalar IDs.
- `dim_blocks` is already block-space (`D/4`), not raw `D`.

## Build and Run

Configure and build:

```bash
cmake -S . -B build_mobilekv
cmake --build build_mobilekv -j
```

Run tests:

```bash
./build_mobilekv/kv_cache_test
```

Run examples:

```bash
./build_mobilekv/mobilekv_demo
./build_mobilekv/mobilekv_mixed_precision_demo
./build_mobilekv/mobilekv_dim_block_demo
./build_mobilekv/mobilekv_convenience_demo
```

Run benchmark:

```bash
./build_mobilekv/mobilekv_benchmark
```

Benchmark includes both:

- Ring steady-state cases
- Growth-stress cases (non-ring expansion pressure)

## Example Files

- `example/fp32_prefill_decode_example.cpp`
- `example/mixed_precision_example.cpp`
- `example/dim_block_example.cpp`
- `example/convenience_api_example.cpp`

## Test Coverage (Current Focus)

Unit tests currently validate:

- Template addressing and byte sizing
- Mixed precision K/V assignment
- Dim-block layout locate/view correctness
- Ring-buffer logical-to-physical mapping and wrap behavior
- Default ring configuration inheritance and explicit override
- Strict build failure for invalid `initial > max`
- Accessor scalar-type mismatch rejection

## Notes for Production Integration

For simple single-request or light batching workloads, current ring-window storage is usable.
For large-scale continuous batching runtimes, add a separate scheduling/index indirection layer above MobileKV (request-token mapping, gather plans, block tables) to avoid coupling runtime policy with storage internals.
