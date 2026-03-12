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
- Window management policy with `KVPolicy.max_prefix_tokens`, `KVPolicy.recent_keep`, and `KVPolicy.use_ring_buffer`.
- Prefix sealing via `KVSealPrefix(prefix_tokens)` during prefill -> decode transition.
- Append path and reserve/write/commit path for token insertion.
- Prefix sealing (`KVSealPrefix`) and sliding-window behavior.
- Read APIs:
  - unified token-level layer access (`KVAttentionReadLayer` with dtype arg)
- Attention metadata adapter API:
  - `KVGetAttentionView`
  - `KVRecentFirstPhysicalSlot`
  - `KVRecentLogicalStart`
- Unit tests (GoogleTest), runnable examples, and benchmark comparison between ring and non-ring modes.

## API Surface (Layered)

MobileKV now uses a layered API model so beginners and advanced users can work with different complexity levels.

| Layer | Header | Target User | Main APIs |
|---|---|---|---|
| Basic | `include/mobilekv/kv_cache_basic.h` | Beginners / quick integration | `KVRequiredBytes`, `KVInit`, `KVInitPreallocated`, `KVReset`, `KVRelease`, `KVSealPrefix`, `KVAppend`, `KVGetAttentionView` |
| Advanced | `include/mobilekv/kv_cache_advanced.h` | Performance-oriented integration | `KVReserveTokenSlot`, `KVAttentionWriteLayer`, `KVCommitToken`, `KVAttentionReadLayer` |
| Debug | `include/mobilekv/kv_cache_debug.h` | Observability / tooling | `KVGetSnapshot`, metadata getters (`KVValidTokens`, `KVBaseToken`, etc.) |

Backward compatibility is preserved through `include/mobilekv/kv_cache.h`, which aggregates all layers.

Lifecycle note:
- `KVReset` clears logical token state for reuse.
- `KVSealPrefix(prefix_tokens)` freezes actual prefix length (must be `<= max_prefix_tokens`):
  - non-ring mode uses fixed `recent_keep`
  - non-ring write path compacts at arena end (delayed compact)
  - ring mode can expand recent window to `max_seq - prefix_tokens`
- `KVRelease` fully invalidates the `KVCache` handle.
- `KVRelease` frees arena memory only for `KVInit` (owned mode); for `KVInitPreallocated`, arena ownership stays with caller.

## Current Scope and Status

MobileKV is under active development toward production readiness.
At the current stage:

- CPU + FP16 path is implemented and validated.
- `INT8`/`GPU` related config fields exist but are not fully implemented.
- API and behavior are functional for integration experiments, and will continue to harden as features expand.

## Repository Structure

- `include/mobilekv/kv_cache.h`: compatibility umbrella header (includes all API layers).
- `include/mobilekv/kv_cache_basic.h`: beginner-focused API layer.
- `include/mobilekv/kv_cache_advanced.h`: advanced API layer.
- `include/mobilekv/kv_cache_debug.h`: debug/introspection API layer.
- `src/kv_cache.cpp`: core implementation.
- `test/kv_cache_test.cpp`: unit tests (lifecycle, snapshot, compaction, ring-buffer, attention view, layer IO).
- `example/kv_cache_basic_example.cpp`: basic integration example.
- `example/llama.cpp_adapter.cpp`: attention metadata adapter demo for llama.cpp-style kernels.
- `example/attention_layer_submit_example.cpp`: per-layer submit/read attention demo.
- `benchmark/kv_benchmark.cpp`: ring-buffer on/off benchmark.

## Long-Term Direction

MobileKV aims to become a robust KV cache layer for edge inference stacks, with stronger guarantees in correctness, performance, and integration ergonomics across frameworks and hardware targets.
