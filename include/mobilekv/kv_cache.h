#ifndef MOBILEKV_KV_CACHE_H_
#define MOBILEKV_KV_CACHE_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ================================================================
   Status
   ================================================================ */

typedef enum {
  KV_OK = 0,
  KV_ERR_NULL,
  KV_ERR_BAD_ARG,
  KV_ERR_BAD_STATE,
  KV_ERR_NO_SPACE,
  KV_ERR_UNSUPPORTED
} KVStatus;

/* ================================================================
   Data Type
   ================================================================ */

typedef enum {
  KV_DTYPE_FP16 = 0,
  KV_DTYPE_INT8 = 1
} KVDataType;

/* ================================================================
   Layout Type
   ================================================================ */

typedef enum {
  KV_LAYOUT_LAYER_TOKEN_HIDDEN = 0,
  KV_LAYOUT_RING = 1
} KVLayoutType;

/* ================================================================
   Backend
   ================================================================ */

typedef enum {
  KV_BACKEND_CPU = 0,
  KV_BACKEND_GPU = 1
} KVBackendType;

/* ================================================================
   Model Shape
   ================================================================ */

typedef struct {

  int layers;
  int heads;
  int head_dim;

  /* hidden = heads * head_dim */
  int hidden;

  /* maximum KV tokens stored physically */
  int max_seq;

} KVShape;

/* ================================================================
   Sliding Window Policy
   ================================================================ */

typedef struct {

  /* maximum allowed prefix tokens */
  int max_prefix_tokens;

  /* rolling window size */
  int recent_keep;

} KVPolicy;

/* ================================================================
   Quantization Config (reserved)
   ================================================================ */

typedef struct {

  int enabled;
  int group_size;
  int scale_bytes;

} KVQuantConfig;

/* ================================================================
   KV Config
   ================================================================ */

typedef struct {

  KVShape shape;
  KVPolicy policy;

  KVDataType dtype;
  KVLayoutType layout;
  KVBackendType backend;

  KVQuantConfig quant;

} KVConfig;

/* ================================================================
   Runtime State
   ================================================================ */

typedef struct {

  int cursor;
  int valid_tokens;

  /* logical base token in original stream */
  int base_token;

  int prefix_frozen;
  int prefix_tokens;

} KVState;

/* ================================================================
   Internal Arena
   ================================================================ */

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

/* ================================================================
   KV Cache Object
   ================================================================ */

typedef struct KVCache {

  KVConfig config;
  KVState state;

  void* arena_base;
  size_t arena_bytes;

  KVCPUArena arena;

  int initialized;

} KVCache;

/* ================================================================
   Memory Requirement
   ================================================================ */

size_t KVRequiredBytes(const KVConfig* config);

/* ================================================================
   Lifecycle
   ================================================================ */

KVStatus KVInitPreallocated(
    KVCache* kv,
    const KVConfig* config,
    void* arena_base,
    size_t arena_bytes);

void KVReset(KVCache* kv);

/* seal prefix after system prompt */
KVStatus KVSealPrefix(
    KVCache* kv,
    int prefix_tokens);

/* ================================================================
   Append API
   ================================================================ */

/* simple append: input layout = [layers][hidden] */

KVStatus KVAppend(
    KVCache* kv,
    const void* new_k_layers,
    const void* new_v_layers);

/* flexible append view */

typedef struct {

  const void* k;
  const void* v;

  size_t layer_stride_bytes;

} KVAppendView;

KVStatus KVAppendViewWrite(
    KVCache* kv,
    const KVAppendView* view);

/* ================================================================
   Token Reservation Mode (runtime friendly)
   ================================================================ */

KVStatus KVReserveTokenSlot(
    KVCache* kv,
    int* out_token);

KVStatus KVWriteLayerToken(
    KVCache* kv,
    int layer,
    int token,
    const void* k_data,
    const void* v_data,
    size_t bytes);

KVStatus KVCommitToken(
    KVCache* kv,
    int token);

/* ================================================================
   Compact (prefix + recent sliding window)
   ================================================================ */

KVStatus KVCompact(KVCache* kv);

/* ================================================================
   Read View
   ================================================================ */

typedef struct {

  int start_token;
  int token_count;

} KVSegment;

typedef struct {

  int segment_count;

  KVSegment segments[2];

} KVReadView;

KVStatus KVGetReadView(
    const KVCache* kv,
    KVReadView* view);

/* ================================================================
   Layer Read Span (fast decode)
   ================================================================ */

typedef struct {

  const void* k_base;
  const void* v_base;

  int token_count;

  size_t token_stride_bytes;

} KVLayerReadSpan;

KVStatus KVGetLayerReadSpan(
    const KVCache* kv,
    int layer,
    KVLayerReadSpan* span);

/* ================================================================
   Token Access
   ================================================================ */

const void* KVKToken(
    const KVCache* kv,
    int layer,
    int token);

const void* KVVToken(
    const KVCache* kv,
    int layer,
    int token);

/* typed helpers */

const uint16_t* KVKTokenFP16(
    const KVCache* kv,
    int layer,
    int token);

const uint16_t* KVVTokenFP16(
    const KVCache* kv,
    int layer,
    int token);

const int8_t* KVKTokenINT8(
    const KVCache* kv,
    int layer,
    int token);

const int8_t* KVVTokenINT8(
    const KVCache* kv,
    int layer,
    int token);

/* ================================================================
   Metadata
   ================================================================ */

int KVValidTokens(const KVCache* kv);

int KVCursor(const KVCache* kv);

int KVBaseToken(const KVCache* kv);

int KVPrefixTokens(const KVCache* kv);

int KVRecentTokens(const KVCache* kv);

KVDataType KVDType(const KVCache* kv);

KVLayoutType KVLayout(const KVCache* kv);

int KVIsInitialized(const KVCache* kv);

/* ================================================================
   Compatibility Helpers (llama.cpp style)
   ================================================================ */

typedef struct {

  const void* k;
  const void* v;

  int n_tokens;
  int hidden;

  size_t stride_bytes;

} KVCompatLlamaSpan;

KVStatus KVCompatGetLlamaSpan(
    const KVCache* kv,
    int layer,
    KVCompatLlamaSpan* span);

#ifdef __cplusplus
}
#endif

#endif