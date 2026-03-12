#ifndef MOBILEKV_KV_CACHE_BASIC_H_
#define MOBILEKV_KV_CACHE_BASIC_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Return codes for all public APIs. */
typedef enum {
  KV_OK = 0,
  KV_ERR_NULL,
  KV_ERR_BAD_ARG,
  KV_ERR_BAD_STATE,
  KV_ERR_NO_SPACE,
  KV_ERR_UNSUPPORTED
} KVStatus;

/* Supported KV element data type. */
typedef enum {
  KV_DTYPE_FP32 = 0,
  KV_DTYPE_FP16 = 1,
  KV_DTYPE_INT8 = 2
} KVDataType;

/* Runtime backend selection. */
typedef enum {
  KV_BACKEND_CPU = 0,
  KV_BACKEND_GPU = 1
} KVBackendType;

/* Tensor shape for one model's KV cache. */
typedef struct {
  int layers;   /* transformer layer count */
  int heads;    /* attention heads */
  int head_dim; /* per-head hidden size */
  int hidden;   /* flattened hidden size: heads * head_dim */
  int max_seq;  /* total physical token capacity */
} KVShape;

/* Retention policy after prefix/decode split. */
typedef struct {
  int max_prefix_tokens; /* maximum allowed prefix size when calling KVSealPrefix */
  int recent_keep;       /* decode recent-window size for non-ring mode */
  bool use_ring_buffer;  /* true: ring-managed recent zone, false: compact-managed */
} KVPolicy;

/* Optional quantization metadata (reserved for future paths). */
typedef struct {
  bool enabled;
  int group_size;
  int scale_bytes;
} KVQuantConfig;

/* Full initialization config for KV cache. */
typedef struct {
  KVShape shape;
  KVPolicy policy;
  KVDataType dtype;
  KVBackendType backend;
  KVQuantConfig quant;
} KVConfig;

/* Runtime state of token progression and retention windows. */
typedef struct {
  int cursor;          /* next physical write slot when appending */
  int valid_tokens;    /* logical visible tokens */
  int base_token;      /* number of dropped recent tokens from original stream */

  bool prefix_frozen;
  int prefix_tokens;   /* actual frozen prefix length */

  /* ring metadata for recent zone */
  int recent_head;     /* physical write head within recent zone [0, recent_capacity) */
  int recent_size;     /* logical recent token count <= recent_capacity */
} KVState;

/* CPU arena pointers and their byte ranges. */
typedef struct {
  void* k_data;
  void* v_data;
  void* k_scales;
  void* v_scales;
  size_t k_bytes;
  size_t v_bytes;
  size_t k_scales_bytes;
  size_t v_scales_bytes;
} KVCPUArena;

/* Main KV cache handle. */
typedef struct KVCache {
  KVConfig config;
  KVState state;

  void* arena_base;
  size_t arena_bytes;
  KVCPUArena arena;

  bool initialized;
  bool owns_arena;
} KVCache;

/* Metadata that an attention kernel can consume directly. */
typedef struct {
  /* base storage */
  const void* k_base;
  const void* v_base;

  size_t layer_stride_bytes;
  size_t token_stride_bytes;

  /* logical info */
  int q_pos;               /* logical query position */
  int visible_tokens;      /* prefix + recent */
  int prefix_tokens;       /* frozen prefix length */

  /* recent window logical range */
  int recent_logical_start;
  int recent_size;
  int recent_capacity;

  /* ring physical mapping */
  int recent_first_slot;   /* physical slot of oldest recent token */
  bool recent_wrapped;
} KVAttentionView;

/* Return required bytes for KVInitPreallocated(config). */
size_t KVRequiredBytes(const KVConfig* config);

/* Allocate internal arena and initialize kv. Release with KVRelease. */
KVStatus KVInit(
    KVCache* kv,
    const KVConfig* config);

/* Initialize with caller-provided arena memory; caller keeps arena ownership. */
KVStatus KVInitPreallocated(
    KVCache* kv,
    const KVConfig* config,
    void* arena_base,
    size_t arena_bytes);

/* Reset logical state to empty; memory is kept for reuse. */
void KVReset(KVCache* kv);

/* Release handle and owned resources. Frees arena only when created by KVInit. */
KVStatus KVRelease(KVCache* kv);

/*
 * Seal prefill and switch to decode mode.
 * prefix_tokens must satisfy:
 * - 0 <= prefix_tokens <= current valid tokens
 * - prefix_tokens <= policy.max_prefix_tokens
 */
KVStatus KVSealPrefix(
    KVCache* kv,
    int prefix_tokens);

/* Build attention-facing metadata at query position q_pos. */
KVStatus KVGetAttentionView(
    const KVCache* kv,
    int q_pos,
    KVAttentionView* view);

#ifdef __cplusplus
}
#endif

#endif
