#ifndef MOBILEKV_KV_CACHE_DEBUG_H_
#define MOBILEKV_KV_CACHE_DEBUG_H_

#include "mobilekv/kv_cache_basic.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int valid_tokens;   /* visible logical tokens (prefix + recent) */
  int cursor;         /* next physical write slot */
  int base_token;     /* dropped-token count from original stream */
  int prefix_tokens;  /* sealed prefix size */
  int recent_tokens;  /* visible recent window size */
  KVDataType dtype;   /* active cache dtype */
  bool initialized;   /* handle init state */
} KVSnapshot;

/* Fetch all frequently-used runtime metadata in one call. */
KVStatus KVGetSnapshot(const KVCache* kv, KVSnapshot* snapshot);

/* Scalar helpers for telemetry/debug UI. */
int KVValidTokens(const KVCache* kv);
int KVCursor(const KVCache* kv);
int KVBaseToken(const KVCache* kv);
int KVPrefixTokens(const KVCache* kv);
int KVRecentTokens(const KVCache* kv);
KVDataType KVDType(const KVCache* kv);
bool KVIsInitialized(const KVCache* kv);

#ifdef __cplusplus
}
#endif

#endif
