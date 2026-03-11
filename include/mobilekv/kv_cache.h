#ifndef MOBILEKV_KV_CACHE_H_
#define MOBILEKV_KV_CACHE_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  KV_OK = 0,
  KV_ERR_NULL,
  KV_ERR_BAD_ARG,
  KV_ERR_BAD_STATE,
  KV_ERR_NO_SPACE,
  KV_ERR_UNSUPPORTED
} KVStatus;

typedef enum {
  KV_DTYPE_FP16 = 0,
  KV_DTYPE_INT8 = 1
} KVDataType;

typedef enum {
  KV_BACKEND_CPU = 0,
  KV_BACKEND_GPU = 1
} KVBackendType;

typedef struct {
  int layers;
  int heads;
  int head_dim;
  int hidden;   /* heads * head_dim */
  int max_seq;  /* total physical capacity */
} KVShape;

typedef struct {
  int max_prefix_tokens;  /* reserved prefix capacity */
  int recent_keep;        /* recent logical window */
  int use_ring_buffer;    /* 0 = compact/memmove, 1 = ring for recent zone */
} KVPolicy;

typedef struct {
  int enabled;
  int group_size;
  int scale_bytes;
} KVQuantConfig;

typedef struct {
  KVShape shape;
  KVPolicy policy;
  KVDataType dtype;
  KVBackendType backend;
  KVQuantConfig quant;
} KVConfig;

typedef struct {
  int cursor;          /* logical total tokens = prefix + recent_valid */
  int valid_tokens;    /* logical visible tokens */
  int base_token;      /* number of dropped recent tokens from original stream */

  int prefix_frozen;
  int prefix_tokens;   /* actual frozen prefix length */

  /* ring metadata for recent zone */
  int recent_head;     /* physical write head within recent zone [0, recent_capacity) */
  int recent_size;     /* logical recent token count <= recent_capacity */
} KVState;

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

typedef struct KVCache {
  KVConfig config;
  KVState state;

  void* arena_base;
  size_t arena_bytes;
  KVCPUArena arena;

  int initialized;
} KVCache;

typedef struct {
  int start_token;   /* physical token slot in storage */
  int token_count;
} KVSegment;

typedef struct {
  int segment_count; /* 1 or 2 */
  KVSegment segments[2];
} KVReadView;

typedef struct {
  const void* k_base;
  const void* v_base;
  int token_count;
  size_t token_stride_bytes;
} KVLayerReadSpan;

typedef struct {
  const void* k;
  const void* v;
  size_t layer_stride_bytes;
} KVAppendView;

/* lifecycle */
size_t KVRequiredBytes(const KVConfig* config);

KVStatus KVInitPreallocated(
    KVCache* kv,
    const KVConfig* config,
    void* arena_base,
    size_t arena_bytes);

void KVReset(KVCache* kv);
KVStatus KVSealPrefix(KVCache* kv, int prefix_tokens);

/* write */
KVStatus KVAppend(
    KVCache* kv,
    const void* new_k_layers,
    const void* new_v_layers);

KVStatus KVAppendViewWrite(
    KVCache* kv,
    const KVAppendView* view);

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

KVStatus KVCompact(KVCache* kv);

/* read */
KVStatus KVGetReadView(
    const KVCache* kv,
    KVReadView* view);

KVStatus KVGetLayerReadSpan(
    const KVCache* kv,
    int layer,
    KVLayerReadSpan* span);

/* token access */
const void* KVKToken(const KVCache* kv, int layer, int token);
const void* KVVToken(const KVCache* kv, int layer, int token);

const uint16_t* KVKTokenFP16(const KVCache* kv, int layer, int token);
const uint16_t* KVVTokenFP16(const KVCache* kv, int layer, int token);

const int8_t* KVKTokenINT8(const KVCache* kv, int layer, int token);
const int8_t* KVVTokenINT8(const KVCache* kv, int layer, int token);

/* metadata */
int KVValidTokens(const KVCache* kv);
int KVCursor(const KVCache* kv);
int KVBaseToken(const KVCache* kv);
int KVPrefixTokens(const KVCache* kv);
int KVRecentTokens(const KVCache* kv);
KVDataType KVDType(const KVCache* kv);
int KVIsInitialized(const KVCache* kv);

// For attention view construction
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
  int recent_wrapped;      /* whether window crosses end of ring */

} KVAttentionView;

KVStatus KVGetAttentionView(
    const KVCache* kv,
    int q_pos,
    KVAttentionView* view);

int KVRecentFirstPhysicalSlot(const KVCache* kv);

int KVRecentLogicalStart(const KVCache* kv);


#ifdef __cplusplus
}
#endif

#endif
