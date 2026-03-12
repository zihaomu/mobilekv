#ifndef MOBILEKV_KV_CACHE_ADVANCED_H_
#define MOBILEKV_KV_CACHE_ADVANCED_H_

#include "mobilekv/kv_cache_basic.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  const void* k;            /* token K source, first layer base */
  const void* v;            /* token V source, first layer base */
  size_t layer_stride_bytes;/* bytes between adjacent layers in source */
} KVAppendView;

/* Append one token from a strided [layer][hidden] source view. */
KVStatus KVAppendViewWrite(
    KVCache* kv,
    const KVAppendView* view);

/* Reserve physical slot for one token in transaction-style write flow. */
KVStatus KVReserveTokenSlot(
    KVCache* kv,
    int* out_token);

/* Commit previously reserved token slot after all layer writes complete. */
KVStatus KVCommitToken(
    KVCache* kv,
    int token);

/* Force compaction in non-ring mode (mainly for control/debug). */
KVStatus KVCompact(KVCache* kv);

typedef struct {
  KVDataType dtype;     /* must match kv->config.dtype */
  const void* k_data;   /* write input / read output pointer */
  const void* v_data;   /* write input / read output pointer */
} KVAttentionLayerArg;

/* Write one layer's K/V for a reserved token; dispatches by dtype. */
KVStatus KVAttentionWriteLayer(
    KVCache* kv,
    int layer,
    int token,
    const KVAttentionLayerArg* arg);

/* Read one layer's K/V by logical token index; dispatches by dtype. */
KVStatus KVAttentionReadLayer(
    const KVCache* kv,
    int layer,
    int logical_token,
    KVAttentionLayerArg* arg);

/*
 * Append one token in contiguous [layer][hidden] layout.
 * Use this path for simple integration when you already have all layers packed.
 */
KVStatus KVAppend(
    KVCache* kv,
    const void* new_k_layers,
    const void* new_v_layers);

/*
 * Token-level typed access is unified through KVAttentionReadLayer(..., arg).
 * Use arg->dtype to select dtype and read arg->k_data / arg->v_data outputs.
 */

/* Physical slot of oldest visible recent token. */
int KVRecentFirstPhysicalSlot(const KVCache* kv);

/* Logical start index of visible recent window. */
int KVRecentLogicalStart(const KVCache* kv);

#ifdef __cplusplus
}
#endif

#endif
