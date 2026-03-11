#include "mobilekv/kv_cache.h"

#include <string.h>

#define KV_ALIGN 64

static size_t kv_align_up(size_t v) {
  return (v + KV_ALIGN - 1) & ~(KV_ALIGN - 1);
}

static int kv_valid_config(const KVConfig* c) {
  if (!c) return 0;
  if (c->shape.layers <= 0) return 0;
  if (c->shape.heads <= 0) return 0;
  if (c->shape.head_dim <= 0) return 0;
  if (c->shape.hidden != c->shape.heads * c->shape.head_dim) return 0;
  if (c->shape.max_seq <= 0) return 0;
  if (c->policy.max_prefix_tokens < 0) return 0;
  if (c->policy.recent_keep < 0) return 0;
  if (c->policy.max_prefix_tokens + c->policy.recent_keep > c->shape.max_seq) return 0;
  if (c->dtype != KV_DTYPE_FP16) return 0; /* v2 implements fp16 only */
  if (c->backend != KV_BACKEND_CPU) return 0;
  return 1;
}

static size_t kv_tensor_bytes(const KVShape* s, size_t elem_bytes) {
  return (size_t)s->layers * (size_t)s->max_seq * (size_t)s->hidden * elem_bytes;
}

static uint8_t* kv_token_ptr_mut(KVCache* kv, void* base, int layer, int token, size_t elem) {
  size_t off = ((size_t)layer * (size_t)kv->config.shape.max_seq + (size_t)token) *
               (size_t)kv->config.shape.hidden * elem;
  return (uint8_t*)base + off;
}

static const uint8_t* kv_token_ptr_const(const KVCache* kv, const void* base, int layer, int token, size_t elem) {
  size_t off = ((size_t)layer * (size_t)kv->config.shape.max_seq + (size_t)token) *
               (size_t)kv->config.shape.hidden * elem;
  return (const uint8_t*)base + off;
}

static int kv_prefix_capacity(const KVCache* kv) {
  return kv->config.policy.max_prefix_tokens;
}

static int kv_recent_capacity(const KVCache* kv) {
  return kv->config.shape.max_seq - kv->config.policy.max_prefix_tokens;
}

static int kv_prefix_tokens_runtime(const KVCache* kv) {
  return kv->state.prefix_frozen ? kv->state.prefix_tokens : 0;
}

static size_t kv_token_bytes(const KVCache* kv) {
  return (size_t)kv->config.shape.hidden * sizeof(uint16_t);
}

/* logical token index [0, valid_tokens) -> physical token slot [0, max_seq) */
static int kv_logical_to_physical(const KVCache* kv, int logical_token) {
  const int prefix = kv_prefix_tokens_runtime(kv);
  const int recent_cap = kv_recent_capacity(kv);
  const int recent_base = kv_prefix_capacity(kv);

  if (logical_token < 0 || logical_token >= kv->state.valid_tokens) return -1;

  if (logical_token < prefix) {
    return logical_token;
  }

  /* recent logical region */
  const int recent_logical = logical_token - prefix;

  if (!kv->config.policy.use_ring_buffer) {
    return prefix + recent_logical;
  } else {
    if (recent_cap <= 0) return -1;
    /* oldest recent starts at recent_head - recent_size */
    int oldest = kv->state.recent_head - kv->state.recent_size;
    while (oldest < 0) oldest += recent_cap;
    return recent_base + ((oldest + recent_logical) % recent_cap);
  }
}

size_t KVRequiredBytes(const KVConfig* config) {
  if (!kv_valid_config(config)) return 0;
  const size_t t = kv_tensor_bytes(&config->shape, sizeof(uint16_t));
  return KV_ALIGN + kv_align_up(t) + kv_align_up(t);
}

KVStatus KVInitPreallocated(
    KVCache* kv,
    const KVConfig* config,
    void* arena_base,
    size_t arena_bytes) {
  uint8_t* p;
  size_t t;

  if (!kv || !config || !arena_base) return KV_ERR_NULL;
  if (!kv_valid_config(config)) return KV_ERR_BAD_ARG;

  memset(kv, 0, sizeof(*kv));
  kv->config = *config;
  kv->arena_base = arena_base;
  kv->arena_bytes = arena_bytes;

  p = (uint8_t*)kv_align_up((size_t)arena_base);
  t = kv_tensor_bytes(&config->shape, sizeof(uint16_t));

  kv->arena.k_data = p;
  kv->arena.k_bytes = t;
  p += kv_align_up(t);

  kv->arena.v_data = p;
  kv->arena.v_bytes = t;
  p += kv_align_up(t);

  if ((size_t)(p - (uint8_t*)arena_base) > arena_bytes) {
    memset(kv, 0, sizeof(*kv));
    return KV_ERR_NO_SPACE;
  }

  kv->initialized = 1;
  KVReset(kv);
  return KV_OK;
}

void KVReset(KVCache* kv) {
  if (!kv) return;
  kv->state.cursor = 0;
  kv->state.valid_tokens = 0;
  kv->state.base_token = 0;
  kv->state.prefix_frozen = 0;
  kv->state.prefix_tokens = 0;
  kv->state.recent_head = 0;
  kv->state.recent_size = 0;
}

KVStatus KVSealPrefix(KVCache* kv, int prefix_tokens) {
  if (!kv) return KV_ERR_NULL;
  if (!kv->initialized) return KV_ERR_BAD_STATE;
  if (prefix_tokens < 0) return KV_ERR_BAD_ARG;
  if (prefix_tokens > kv->state.valid_tokens) return KV_ERR_BAD_ARG;
  if (prefix_tokens > kv->config.policy.max_prefix_tokens) return KV_ERR_BAD_ARG;

  kv->state.prefix_tokens = prefix_tokens;
  kv->state.prefix_frozen = 1;

  /* If prefix is sealed after some appended tokens, treat remaining as recent. */
  kv->state.recent_size = kv->state.valid_tokens - prefix_tokens;
  if (kv->state.recent_size < 0) kv->state.recent_size = 0;

  if (kv->config.policy.use_ring_buffer) {
    /* current recent already laid out contiguously at [prefix, valid_tokens) */
    kv->state.recent_head = kv->state.recent_size % (kv_recent_capacity(kv) > 0 ? kv_recent_capacity(kv) : 1);
  }

  return KV_OK;
}

/* Compact path for non-ring mode: [prefix] + [last recent_keep] */
KVStatus KVCompact(KVCache* kv) {
  const int prefix = kv_prefix_tokens_runtime(kv);
  const int valid = kv->state.valid_tokens;
  const int recent = valid - prefix;
  int keep_recent;
  int src_recent;
  int dst_recent;
  int l;
  size_t bytes;
  uint8_t* k;
  uint8_t* v;

  if (!kv) return KV_ERR_NULL;
  if (!kv->initialized) return KV_ERR_BAD_STATE;

  if (kv->config.policy.use_ring_buffer) {
    /* Ring mode does not require memmove compaction for recent zone. */
    return KV_OK;
  }

  keep_recent = recent;
  if (keep_recent > kv->config.policy.recent_keep) {
    keep_recent = kv->config.policy.recent_keep;
  }
  if (keep_recent < 0) keep_recent = 0;

  src_recent = valid - keep_recent;
  dst_recent = prefix;
  bytes = kv_token_bytes(kv);

  for (l = 0; l < kv->config.shape.layers; ++l) {
    k = kv_token_ptr_mut(kv, kv->arena.k_data, l, 0, sizeof(uint16_t));
    v = kv_token_ptr_mut(kv, kv->arena.v_data, l, 0, sizeof(uint16_t));

    if (keep_recent > 0 && src_recent != dst_recent) {
      memmove(k + (size_t)dst_recent * bytes,
              k + (size_t)src_recent * bytes,
              (size_t)keep_recent * bytes);
      memmove(v + (size_t)dst_recent * bytes,
              v + (size_t)src_recent * bytes,
              (size_t)keep_recent * bytes);
    }
  }

  kv->state.base_token += (recent - keep_recent);
  kv->state.recent_size = keep_recent;
  kv->state.valid_tokens = prefix + keep_recent;
  kv->state.cursor = kv->state.valid_tokens;
  return KV_OK;
}

static KVStatus kv_append_fp16_contiguous(
    KVCache* kv,
    const void* new_k_layers,
    const void* new_v_layers,
    size_t layer_stride_bytes) {
  const int layers = kv->config.shape.layers;
  const int prefix = kv_prefix_tokens_runtime(kv);
  const int recent_cap = kv_recent_capacity(kv);
  const size_t bytes = kv_token_bytes(kv);
  int l;
  int phys_slot;

  if (!kv->config.policy.use_ring_buffer) {
    if (kv->state.cursor >= kv->config.shape.max_seq) {
      KVStatus st = KVCompact(kv);
      if (st != KV_OK) return st;
    }
    phys_slot = kv->state.cursor;
  } else {
    /* Before prefix is sealed, we append contiguously into prefix area / early arena. */
    if (!kv->state.prefix_frozen) {
      if (kv->state.cursor >= kv->config.shape.max_seq) return KV_ERR_NO_SPACE;
      phys_slot = kv->state.cursor;
    } else {
      if (recent_cap <= 0) return KV_ERR_NO_SPACE;
      phys_slot = kv_prefix_capacity(kv) + kv->state.recent_head;
    }
  }

  for (l = 0; l < layers; ++l) {
    uint8_t* dst_k = kv_token_ptr_mut(kv, kv->arena.k_data, l, phys_slot, sizeof(uint16_t));
    uint8_t* dst_v = kv_token_ptr_mut(kv, kv->arena.v_data, l, phys_slot, sizeof(uint16_t));
    const uint8_t* src_k = (const uint8_t*)new_k_layers + (size_t)l * layer_stride_bytes;
    const uint8_t* src_v = (const uint8_t*)new_v_layers + (size_t)l * layer_stride_bytes;
    memcpy(dst_k, src_k, bytes);
    memcpy(dst_v, src_v, bytes);
  }

  if (!kv->config.policy.use_ring_buffer) {
    kv->state.cursor += 1;
    if (kv->state.valid_tokens < kv->config.shape.max_seq) kv->state.valid_tokens += 1;
    kv->state.recent_size = kv->state.valid_tokens - prefix;
  } else {
    if (!kv->state.prefix_frozen) {
      kv->state.cursor += 1;
      kv->state.valid_tokens += 1;
    } else {
      if (kv->state.recent_size < recent_cap) {
        kv->state.recent_size += 1;
        kv->state.valid_tokens = prefix + kv->state.recent_size;
        kv->state.cursor = kv->state.valid_tokens;
      } else {
        /* Overwrite oldest recent token. */
        kv->state.base_token += 1;
      }
      kv->state.recent_head += 1;
      if (kv->state.recent_head >= recent_cap) kv->state.recent_head = 0;
    }
  }

  return KV_OK;
}

KVStatus KVAppend(
    KVCache* kv,
    const void* new_k_layers,
    const void* new_v_layers) {
  if (!kv || !new_k_layers || !new_v_layers) return KV_ERR_NULL;
  if (!kv->initialized) return KV_ERR_BAD_STATE;
  if (kv->config.dtype != KV_DTYPE_FP16) return KV_ERR_UNSUPPORTED;

  return kv_append_fp16_contiguous(
      kv,
      new_k_layers,
      new_v_layers,
      (size_t)kv->config.shape.hidden * sizeof(uint16_t));
}

KVStatus KVAppendViewWrite(
    KVCache* kv,
    const KVAppendView* view) {
  if (!kv || !view) return KV_ERR_NULL;
  if (!view->k || !view->v) return KV_ERR_NULL;
  if (view->layer_stride_bytes < kv_token_bytes(kv)) return KV_ERR_BAD_ARG;

  return kv_append_fp16_contiguous(
      kv,
      view->k,
      view->v,
      view->layer_stride_bytes);
}

KVStatus KVReserveTokenSlot(KVCache* kv, int* out_token) {
  int phys_slot;
  if (!kv || !out_token) return KV_ERR_NULL;
  if (!kv->initialized) return KV_ERR_BAD_STATE;
  if (kv->config.dtype != KV_DTYPE_FP16) return KV_ERR_UNSUPPORTED;

  if (!kv->config.policy.use_ring_buffer) {
    if (kv->state.cursor >= kv->config.shape.max_seq) {
      KVStatus st = KVCompact(kv);
      if (st != KV_OK) return st;
    }
    phys_slot = kv->state.cursor;
  } else {
    if (!kv->state.prefix_frozen) {
      if (kv->state.cursor >= kv->config.shape.max_seq) return KV_ERR_NO_SPACE;
      phys_slot = kv->state.cursor;
    } else {
      if (kv_recent_capacity(kv) <= 0) return KV_ERR_NO_SPACE;
      phys_slot = kv_prefix_capacity(kv) + kv->state.recent_head;
    }
  }
  *out_token = phys_slot;
  return KV_OK;
}

KVStatus KVWriteLayerToken(
    KVCache* kv,
    int layer,
    int token,
    const void* k_data,
    const void* v_data,
    size_t bytes) {
  uint8_t* dst_k;
  uint8_t* dst_v;

  if (!kv || !k_data || !v_data) return KV_ERR_NULL;
  if (!kv->initialized) return KV_ERR_BAD_STATE;
  if (layer < 0 || layer >= kv->config.shape.layers) return KV_ERR_BAD_ARG;
  if (token < 0 || token >= kv->config.shape.max_seq) return KV_ERR_BAD_ARG;
  if (bytes != kv_token_bytes(kv)) return KV_ERR_BAD_ARG;

  dst_k = kv_token_ptr_mut(kv, kv->arena.k_data, layer, token, sizeof(uint16_t));
  dst_v = kv_token_ptr_mut(kv, kv->arena.v_data, layer, token, sizeof(uint16_t));
  memcpy(dst_k, k_data, bytes);
  memcpy(dst_v, v_data, bytes);
  return KV_OK;
}

KVStatus KVCommitToken(KVCache* kv, int token) {
  const int prefix = kv_prefix_tokens_runtime(kv);
  const int recent_cap = kv_recent_capacity(kv);

  if (!kv) return KV_ERR_NULL;
  if (!kv->initialized) return KV_ERR_BAD_STATE;

  if (!kv->config.policy.use_ring_buffer) {
    if (token != kv->state.cursor) return KV_ERR_BAD_ARG;
    kv->state.cursor += 1;
    if (kv->state.valid_tokens < kv->config.shape.max_seq) kv->state.valid_tokens += 1;
    kv->state.recent_size = kv->state.valid_tokens - prefix;
    return KV_OK;
  }

  if (!kv->state.prefix_frozen) {
    if (token != kv->state.cursor) return KV_ERR_BAD_ARG;
    kv->state.cursor += 1;
    kv->state.valid_tokens += 1;
    return KV_OK;
  }

  if (token != kv_prefix_capacity(kv) + kv->state.recent_head) return KV_ERR_BAD_ARG;

  if (kv->state.recent_size < recent_cap) {
    kv->state.recent_size += 1;
    kv->state.valid_tokens = prefix + kv->state.recent_size;
    kv->state.cursor = kv->state.valid_tokens;
  } else {
    kv->state.base_token += 1;
  }

  kv->state.recent_head += 1;
  if (kv->state.recent_head >= recent_cap) kv->state.recent_head = 0;

  return KV_OK;
}

KVStatus KVGetReadView(
    const KVCache* kv,
    KVReadView* view) {
  int prefix;
  int recent_cap;
  int oldest;
  int first;
  int second;

  if (!kv || !view) return KV_ERR_NULL;
  if (!kv->initialized) return KV_ERR_BAD_STATE;

  memset(view, 0, sizeof(*view));
  prefix = kv_prefix_tokens_runtime(kv);

  if (!kv->config.policy.use_ring_buffer || !kv->state.prefix_frozen) {
    view->segment_count = 1;
    view->segments[0].start_token = 0;
    view->segments[0].token_count = kv->state.valid_tokens;
    return KV_OK;
  }

  recent_cap = kv_recent_capacity(kv);

  if (kv->state.recent_size == 0) {
    view->segment_count = 1;
    view->segments[0].start_token = 0;
    view->segments[0].token_count = prefix;
    return KV_OK;
  }

  oldest = kv->state.recent_head - kv->state.recent_size;
  while (oldest < 0) oldest += recent_cap;

  first = kv->state.recent_size;
  if (oldest + first > recent_cap) {
    first = recent_cap - oldest;
  }
  second = kv->state.recent_size - first;

  if (prefix > 0) {
    if (second > 0) {
      view->segment_count = 2;
      /* Consumer should handle prefix separately or use token access helpers. */
      /* For generic view, segment 0 covers prefix + first recent not possible physically. */
      /* Keep view about recent zone only when ring wraps. */
      view->segments[0].start_token = kv_prefix_capacity(kv) + oldest;
      view->segments[0].token_count = first;
      view->segments[1].start_token = kv_prefix_capacity(kv);
      view->segments[1].token_count = second;
    } else {
      view->segment_count = 1;
      view->segments[0].start_token = kv_prefix_capacity(kv) + oldest;
      view->segments[0].token_count = first;
    }
  } else {
    if (second > 0) {
      view->segment_count = 2;
      view->segments[0].start_token = kv_prefix_capacity(kv) + oldest;
      view->segments[0].token_count = first;
      view->segments[1].start_token = kv_prefix_capacity(kv);
      view->segments[1].token_count = second;
    } else {
      view->segment_count = 1;
      view->segments[0].start_token = kv_prefix_capacity(kv) + oldest;
      view->segments[0].token_count = first;
    }
  }

  return KV_OK;
}

KVStatus KVGetLayerReadSpan(
    const KVCache* kv,
    int layer,
    KVLayerReadSpan* span) {
  if (!kv || !span) return KV_ERR_NULL;
  if (!kv->initialized) return KV_ERR_BAD_STATE;
  if (layer < 0 || layer >= kv->config.shape.layers) return KV_ERR_BAD_ARG;

  span->k_base = kv_token_ptr_const(kv, kv->arena.k_data, layer, 0, sizeof(uint16_t));
  span->v_base = kv_token_ptr_const(kv, kv->arena.v_data, layer, 0, sizeof(uint16_t));
  span->token_count = kv->state.valid_tokens;
  span->token_stride_bytes = kv_token_bytes(kv);
  return KV_OK;
}

const void* KVKToken(const KVCache* kv, int layer, int token) {
  int phys;
  if (!kv || !kv->initialized) return 0;
  if (layer < 0 || layer >= kv->config.shape.layers) return 0;
  phys = kv_logical_to_physical(kv, token);
  if (phys < 0) return 0;
  return kv_token_ptr_const(kv, kv->arena.k_data, layer, phys, sizeof(uint16_t));
}

const void* KVVToken(const KVCache* kv, int layer, int token) {
  int phys;
  if (!kv || !kv->initialized) return 0;
  if (layer < 0 || layer >= kv->config.shape.layers) return 0;
  phys = kv_logical_to_physical(kv, token);
  if (phys < 0) return 0;
  return kv_token_ptr_const(kv, kv->arena.v_data, layer, phys, sizeof(uint16_t));
}

const uint16_t* KVKTokenFP16(const KVCache* kv, int layer, int token) {
  return (const uint16_t*)KVKToken(kv, layer, token);
}

const uint16_t* KVVTokenFP16(const KVCache* kv, int layer, int token) {
  return (const uint16_t*)KVVToken(kv, layer, token);
}

const int8_t* KVKTokenINT8(const KVCache* kv, int layer, int token) {
  (void)kv; (void)layer; (void)token;
  return 0;
}

const int8_t* KVVTokenINT8(const KVCache* kv, int layer, int token) {
  (void)kv; (void)layer; (void)token;
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
  return kv ? kv_prefix_tokens_runtime(kv) : 0;
}

int KVRecentTokens(const KVCache* kv) {
  return kv ? kv->state.recent_size : 0;
}

KVDataType KVDType(const KVCache* kv) {
  return kv ? kv->config.dtype : KV_DTYPE_FP16;
}

int KVIsInitialized(const KVCache* kv) {
  return kv ? kv->initialized : 0;
}

int KVRecentFirstPhysicalSlot(const KVCache* kv) {
    if (!kv || !kv->initialized) return 0;

    const int recent_capacity = kv->config.policy.recent_keep;
    if (recent_capacity <= 0) return 0;

    if (kv->state.recent_size < recent_capacity) {
        return 0;
    }
    return kv->state.recent_head;
}

int KVRecentLogicalStart(const KVCache* kv) {
    if (!kv || !kv->initialized) return 0;
    return kv->state.prefix_tokens + kv->state.base_token;
}


KVStatus KVGetAttentionView(
    const KVCache* kv,
    int q_pos,
    KVAttentionView* view)
{
    if (!kv || !view) return KV_ERR_NULL;
    if (!kv->initialized) return KV_ERR_BAD_STATE;

    const KVState* s = &kv->state;
    const KVConfig* cfg = &kv->config;

    int recent_capacity = cfg->policy.recent_keep;

    int prefix = s->prefix_tokens;
    int recent = s->recent_size;

    int logical_recent_start = KVRecentLogicalStart(kv);

    view->q_pos = q_pos;

    view->visible_tokens = prefix + recent;
    view->prefix_tokens = prefix;

    view->recent_logical_start = logical_recent_start;
    view->recent_size = recent;
    view->recent_capacity = recent_capacity;

    /* compute oldest slot */

    int first_slot = KVRecentFirstPhysicalSlot(kv);

    view->recent_first_slot = first_slot;

    view->recent_wrapped =
        (first_slot + recent > recent_capacity);

    view->k_base = kv->arena.k_data;
    view->v_base = kv->arena.v_data;

    view->layer_stride_bytes =
        kv->config.shape.max_seq *
        kv->config.shape.hidden *
        (kv->config.dtype == KV_DTYPE_FP16 ? 2 : 1);

    view->token_stride_bytes =
        kv->config.shape.hidden *
        (kv->config.dtype == KV_DTYPE_FP16 ? 2 : 1);

    return KV_OK;
}
