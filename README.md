# MobileKV

MobileKV is a focused KV cache manager for edge and mobile LLM inference.
It is designed to be a production-grade memory component, not a full model runtime:
MobileKV owns KV memory planning, token retention policy, and attention-facing metadata,
while your inference engine keeps control of scheduling and kernel execution.

## Project Vision

MobileKV targets real-world on-device deployment constraints:

- Predictable memory usage with preallocated arenas.
- Stable, lightweight C ABI for easy embedding in C/C++ runtimes.
- Fast decode-time updates with configurable window management.
- Clear metadata boundaries between KV state and attention kernel inputs.
- Portable core design that can evolve from CPU-first to heterogeneous backends.

## What Is Implemented Today

The current codebase provides a compact, test-covered baseline:

- Contiguous physical KV layout: `[layer][token][hidden]`.
- Prefix + recent-window policy with `KVPolicy`:
  - `max_prefix_tokens`
  - `recent_keep`
  - `use_ring_buffer` (compact mode vs ring-buffer recent zone)
- Append path and reserve/write/commit path for token insertion.
- Prefix sealing (`KVSealPrefix`) and sliding-window behavior.
- Read APIs:
  - token-level access (`KVKToken*`, `KVVToken*`)
  - view/span metadata (`KVGetReadView`, `KVGetLayerReadSpan`)
- Attention metadata adapter API:
  - `KVGetAttentionView`
  - `KVRecentFirstPhysicalSlot`
  - `KVRecentLogicalStart`
- Unit tests (GoogleTest), runnable examples, and benchmark comparison between ring and non-ring modes.

## Current Scope and Status

MobileKV is under active development toward production readiness.
At the current stage:

- CPU + FP16 path is implemented and validated.
- `INT8`/`GPU` related config fields exist but are not fully implemented.
- API and behavior are functional for integration experiments, and will continue to harden as features expand.

## Repository Structure

- `include/mobilekv/kv_cache.h`: public C API.
- `src/kv_cache.cpp`: core implementation.
- `test/kv_cache_test.cpp`: unit tests (lifecycle, compaction, ring-buffer, attention view).
- `example/kv_cache_basic_example.cpp`: basic integration example.
- `example/llama.cpp_adapter.cpp`: attention metadata adapter demo for llama.cpp-style kernels.
- `benchmark/kv_benchmark.cpp`: ring-buffer on/off benchmark.

## Long-Term Direction

MobileKV aims to become a robust KV cache layer for edge inference stacks, with stronger guarantees in correctness, performance, and integration ergonomics across frameworks and hardware targets.
