#include "mobilekv/kv_cache.h"

#include <string.h>
#include <stdlib.h>

#define KV_ALIGN 64

static size_t align_up(size_t v) {
  return (v + KV_ALIGN - 1) & ~(KV_ALIGN - 1);
}

static size_t tensor_bytes(const KVShape* s, size_t elem) {
  return (size_t)s->layers * s->max_seq * s->hidden * elem;
}

size_t KVRequiredBytes(const KVConfig* config) {
  if (!config) return 0;

  if (config->dtype != KV_DTYPE_FP16)
    return 0;

  size_t elem = sizeof(uint16_t);

  size_t t = tensor_bytes(&config->shape, elem);

  return KV_ALIGN + align_up(t) + align_up(t);
}

KVStatus KVInitPreallocated(
    KVCache* kv,
    const KVConfig* config,
    void* arena_base,
    size_t arena_bytes) {

  if (!kv || !config || !arena_base)
    return KV_ERR_NULL;

  memset(kv, 0, sizeof(*kv));

  kv->config = *config;
  kv->arena_base = arena_base;
  kv->arena_bytes = arena_bytes;

  if (config->dtype != KV_DTYPE_FP16)
    return KV_ERR_UNSUPPORTED;

  uint8_t* p = (uint8_t*)arena_base;

  p = (uint8_t*)align_up((size_t)p);

  size_t elem = sizeof(uint16_t);
  size_t t = tensor_bytes(&config->shape, elem);

  kv->arena.k_data = p;
  kv->arena.k_bytes = t;
  p += align_up(t);

  kv->arena.v_data = p;
  kv->arena.v_bytes = t;
  p += align_up(t);

  if ((size_t)(p - (uint8_t*)arena_base) > arena_bytes)
    return KV_ERR_NO_SPACE;

  kv->state.cursor = 0;
  kv->state.valid_tokens = 0;
  kv->state.base_token = 0;
  kv->state.prefix_frozen = 0;
  kv->state.prefix_tokens = 0;

  kv->initialized = 1;

  return KV_OK;
}

void KVReset(KVCache* kv) {
  if (!kv) return;

  kv->state.cursor = 0;
  kv->state.valid_tokens = 0;
  kv->state.base_token = 0;
  kv->state.prefix_frozen = 0;
  kv->state.prefix_tokens = 0;
}

KVStatus KVSealPrefix(KVCache* kv, int prefix_tokens) {

  if (!kv) return KV_ERR_NULL;

  if (prefix_tokens < 0 ||
      prefix_tokens > kv->state.valid_tokens)
    return KV_ERR_BAD_ARG;

  if (prefix_tokens > kv->config.policy.max_prefix_tokens)
    return KV_ERR_BAD_ARG;

  kv->state.prefix_tokens = prefix_tokens;
  kv->state.prefix_frozen = 1;

  return KV_OK;
}

static uint8_t* token_ptr(KVCache* kv, void* base,
                          int layer, int token, size_t elem) {

  KVShape* s = &kv->config.shape;

  size_t off =
      ((size_t)layer * s->max_seq + token) *
      s->hidden * elem;

  return (uint8_t*)base + off;
}

static const uint8_t* token_ptr_const(const KVCache* kv, const void* base,
                                      int layer, int token, size_t elem) {

  const KVShape* s = &kv->config.shape;

  size_t off =
      ((size_t)layer * s->max_seq + token) *
      s->hidden * elem;

  return (const uint8_t*)base + off;
}

KVStatus KVCompact(KVCache* kv) {

  if (!kv) return KV_ERR_NULL;

  KVShape* s = &kv->config.shape;

  int prefix = kv->state.prefix_frozen ?
               kv->state.prefix_tokens :
               kv->config.policy.max_prefix_tokens;

  int recent_keep = kv->config.policy.recent_keep;

  int valid = kv->state.valid_tokens;

  if (prefix > valid)
    prefix = valid;

  int recent = valid - prefix;
  if (recent < 0) recent = 0;

  int keep_recent = recent;
  if (keep_recent > recent_keep)
    keep_recent = recent_keep;

  int src_recent = valid - keep_recent;
  int dst_recent = prefix;

  size_t token_bytes = s->hidden * sizeof(uint16_t);

  for (int l = 0; l < s->layers; l++) {

    uint8_t* k = token_ptr(kv, kv->arena.k_data, l, 0, sizeof(uint16_t));
    uint8_t* v = token_ptr(kv, kv->arena.v_data, l, 0, sizeof(uint16_t));

    memmove(
        k + dst_recent * token_bytes,
        k + src_recent * token_bytes,
        keep_recent * token_bytes);

    memmove(
        v + dst_recent * token_bytes,
        v + src_recent * token_bytes,
        keep_recent * token_bytes);
  }

  int dropped = (valid - prefix) - keep_recent;

  kv->state.base_token += dropped;
  kv->state.valid_tokens = prefix + keep_recent;
  kv->state.cursor = kv->state.valid_tokens;

  return KV_OK;
}

KVStatus KVAppend(
    KVCache* kv,
    const void* new_k_layers,
    const void* new_v_layers) {

  if (!kv || !new_k_layers || !new_v_layers)
    return KV_ERR_NULL;

  KVShape* s = &kv->config.shape;

  if (kv->state.cursor >= s->max_seq) {

    KVStatus st = KVCompact(kv);
    if (st != KV_OK)
      return st;
  }

  int pos = kv->state.cursor;

  size_t bytes = s->hidden * sizeof(uint16_t);

  for (int l = 0; l < s->layers; l++) {

    uint8_t* dstk = token_ptr(kv, kv->arena.k_data, l, pos, sizeof(uint16_t));
    uint8_t* dstv = token_ptr(kv, kv->arena.v_data, l, pos, sizeof(uint16_t));

    const uint8_t* srck = (const uint8_t*)new_k_layers + l * bytes;
    const uint8_t* srcv = (const uint8_t*)new_v_layers + l * bytes;

    memcpy(dstk, srck, bytes);
    memcpy(dstv, srcv, bytes);
  }

  kv->state.cursor++;
  kv->state.valid_tokens++;

  return KV_OK;
}

KVStatus KVAppendViewWrite(
    KVCache* kv,
    const KVAppendView* view) {

  if (!kv || !view || !view->k || !view->v)
    return KV_ERR_NULL;

  KVShape* s = &kv->config.shape;

  if (kv->state.cursor >= s->max_seq) {
    KVStatus st = KVCompact(kv);
    if (st != KV_OK)
      return st;
  }

  int pos = kv->state.cursor;

  size_t bytes = s->hidden * sizeof(uint16_t);
  size_t stride = view->layer_stride_bytes ? view->layer_stride_bytes : bytes;
  if (stride < bytes)
    return KV_ERR_BAD_ARG;

  for (int l = 0; l < s->layers; l++) {
    uint8_t* dstk = token_ptr(kv, kv->arena.k_data, l, pos, sizeof(uint16_t));
    uint8_t* dstv = token_ptr(kv, kv->arena.v_data, l, pos, sizeof(uint16_t));

    const uint8_t* srck = (const uint8_t*)view->k + l * stride;
    const uint8_t* srcv = (const uint8_t*)view->v + l * stride;

    memcpy(dstk, srck, bytes);
    memcpy(dstv, srcv, bytes);
  }

  kv->state.cursor++;
  kv->state.valid_tokens++;

  return KV_OK;
}

KVStatus KVReserveTokenSlot(
    KVCache* kv,
    int* out_token) {

  if (!kv || !out_token)
    return KV_ERR_NULL;

  if (kv->state.cursor >= kv->config.shape.max_seq) {
    KVStatus st = KVCompact(kv);
    if (st != KV_OK)
      return st;
  }

  *out_token = kv->state.cursor;
  return KV_OK;
}

KVStatus KVWriteLayerToken(
    KVCache* kv,
    int layer,
    int token,
    const void* k_data,
    const void* v_data,
    size_t bytes) {

  if (!kv || !k_data || !v_data)
    return KV_ERR_NULL;

  if (layer < 0 || layer >= kv->config.shape.layers)
    return KV_ERR_BAD_ARG;

  if (token < 0 || token >= kv->config.shape.max_seq)
    return KV_ERR_BAD_ARG;

  size_t expect_bytes = (size_t)kv->config.shape.hidden * sizeof(uint16_t);
  if (bytes != expect_bytes)
    return KV_ERR_BAD_ARG;

  uint8_t* dstk = token_ptr(kv, kv->arena.k_data, layer, token, sizeof(uint16_t));
  uint8_t* dstv = token_ptr(kv, kv->arena.v_data, layer, token, sizeof(uint16_t));
  memcpy(dstk, k_data, bytes);
  memcpy(dstv, v_data, bytes);
  return KV_OK;
}

KVStatus KVCommitToken(
    KVCache* kv,
    int token) {

  if (!kv)
    return KV_ERR_NULL;

  if (token != kv->state.cursor)
    return KV_ERR_BAD_STATE;

  if (kv->state.cursor >= kv->config.shape.max_seq)
    return KV_ERR_NO_SPACE;

  kv->state.cursor++;
  kv->state.valid_tokens = kv->state.cursor;
  return KV_OK;
}

KVStatus KVGetReadView(
    const KVCache* kv,
    KVReadView* view) {

  if (!kv || !view)
    return KV_ERR_NULL;

  view->segment_count = 1;
  view->segments[0].start_token = 0;
  view->segments[0].token_count = kv->state.valid_tokens;
  view->segments[1].start_token = 0;
  view->segments[1].token_count = 0;
  return KV_OK;
}

KVStatus KVGetLayerReadSpan(
    const KVCache* kv,
    int layer,
    KVLayerReadSpan* span) {

  if (!kv || !span)
    return KV_ERR_NULL;

  if (layer < 0 || layer >= kv->config.shape.layers)
    return KV_ERR_BAD_ARG;

  span->k_base = token_ptr_const(kv, kv->arena.k_data, layer, 0, sizeof(uint16_t));
  span->v_base = token_ptr_const(kv, kv->arena.v_data, layer, 0, sizeof(uint16_t));
  span->token_count = kv->state.valid_tokens;
  span->token_stride_bytes = (size_t)kv->config.shape.hidden * sizeof(uint16_t);
  return KV_OK;
}

const void* KVKToken(
    const KVCache* kv,
    int layer,
    int token) {

  if (!kv)
    return 0;

  if (kv->config.dtype == KV_DTYPE_FP16)
    return KVKTokenFP16(kv, layer, token);

  if (kv->config.dtype == KV_DTYPE_INT8)
    return KVKTokenINT8(kv, layer, token);

  return 0;
}

const void* KVVToken(
    const KVCache* kv,
    int layer,
    int token) {

  if (!kv)
    return 0;

  if (kv->config.dtype == KV_DTYPE_FP16)
    return KVVTokenFP16(kv, layer, token);

  if (kv->config.dtype == KV_DTYPE_INT8)
    return KVVTokenINT8(kv, layer, token);

  return 0;
}

int KVValidTokens(const KVCache* kv) {
  return kv ? kv->state.valid_tokens : 0;
}

int KVCursor(const KVCache* kv) {
  return kv ? kv->state.cursor : 0;
}

int KVBaseToken(const KVCache* kv) {
  return kv ? kv->state.base_token : 0;
}

int KVPrefixTokens(const KVCache* kv) {
  return kv ? kv->state.prefix_tokens : 0;
}

int KVRecentTokens(const KVCache* kv) {
  if (!kv) return 0;
  return kv->state.valid_tokens - kv->state.prefix_tokens;
}

KVDataType KVDType(const KVCache* kv) {
  return kv ? kv->config.dtype : KV_DTYPE_FP16;
}

KVLayoutType KVLayout(const KVCache* kv) {
  return kv ? kv->config.layout : KV_LAYOUT_LAYER_TOKEN_HIDDEN;
}

int KVIsInitialized(const KVCache* kv) {
  return kv ? kv->initialized : 0;
}

const uint16_t* KVKTokenFP16(
    const KVCache* kv, int layer, int token) {

  if (!kv) return 0;
  if (layer < 0 || layer >= kv->config.shape.layers) return 0;
  if (token < 0 || token >= kv->config.shape.max_seq) return 0;
  return (const uint16_t*)token_ptr_const(
      kv, kv->arena.k_data, layer, token, sizeof(uint16_t));
}

const uint16_t* KVVTokenFP16(
    const KVCache* kv, int layer, int token) {

  if (!kv) return 0;
  if (layer < 0 || layer >= kv->config.shape.layers) return 0;
  if (token < 0 || token >= kv->config.shape.max_seq) return 0;
  return (const uint16_t*)token_ptr_const(
      kv, kv->arena.v_data, layer, token, sizeof(uint16_t));
}

const int8_t* KVKTokenINT8(
    const KVCache* kv, int layer, int token) {
  (void)kv;
  (void)layer;
  (void)token;
  return 0;
}

const int8_t* KVVTokenINT8(
    const KVCache* kv, int layer, int token) {
  (void)kv;
  (void)layer;
  (void)token;
  return 0;
}

KVStatus KVCompatGetLlamaSpan(
    const KVCache* kv,
    int layer,
    KVCompatLlamaSpan* span) {

  if (!kv || !span)
    return KV_ERR_NULL;

  KVLayerReadSpan read_span{};
  KVStatus st = KVGetLayerReadSpan(kv, layer, &read_span);
  if (st != KV_OK)
    return st;

  span->k = read_span.k_base;
  span->v = read_span.v_base;
  span->n_tokens = read_span.token_count;
  span->hidden = kv->config.shape.hidden;
  span->stride_bytes = read_span.token_stride_bytes;
  return KV_OK;
}
