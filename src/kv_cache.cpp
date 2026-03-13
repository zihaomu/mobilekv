#include "mobilekv/kv_cache.h"

#include <stdlib.h>
#include <string.h>

#define KV_ALIGN 64

static size_t kv_align_up(size_t v) {
  return (v + KV_ALIGN - 1) & ~(KV_ALIGN - 1);
}

static size_t kv_dtype_elem_bytes(KVDataType dtype) {
  switch (dtype) {
    case KV_DTYPE_FP16:
      return sizeof(uint16_t);
    case KV_DTYPE_INT8:
      return sizeof(int8_t);
    default:
      return 0;
  }
}

static int kv_valid_config(const KVConfig* c) {
  const size_t elem_bytes = c ? kv_dtype_elem_bytes(c->dtype) : 0;

  if (!c) return 0;
  if (c->shape.layers <= 0) return 0;
  if (c->shape.heads <= 0) return 0;
  if (c->shape.head_dim <= 0) return 0;
  if (c->shape.hidden != c->shape.heads * c->shape.head_dim) return 0;
  if (c->shape.max_seq <= 0) return 0;
  if (c->policy.max_prefix_tokens < 0) return 0;
  if (c->policy.recent_keep < 0) return 0;
  if (c->policy.max_prefix_tokens > c->shape.max_seq) return 0;
  if (c->policy.max_prefix_tokens + c->policy.recent_keep > c->shape.max_seq) return 0;
  if (elem_bytes == 0) return 0;
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
  if (!kv->state.prefix_frozen) {
    return kv->config.policy.max_prefix_tokens;
  }
  if (kv->config.policy.use_ring_buffer) {
    /* Ring can collapse reserved prefix tail when actual prefix is smaller. */
    return kv->state.prefix_tokens;
  }
  return kv->config.policy.max_prefix_tokens;
}

static int kv_recent_capacity(const KVCache* kv) {
  if (!kv->state.prefix_frozen) {
    return kv->config.policy.recent_keep;
  }
  if (kv->config.policy.use_ring_buffer) {
    /* Ring expands recent to use all physical tail after sealed prefix. */
    return kv->config.shape.max_seq - kv->state.prefix_tokens;
  }
  return kv->config.policy.recent_keep;
}

static int kv_prefix_tokens_runtime(const KVCache* kv) {
  return kv->state.prefix_frozen ? kv->state.prefix_tokens : 0;
}

static size_t kv_elem_bytes(const KVCache* kv) {
  return kv_dtype_elem_bytes(kv->config.dtype);
}

static size_t kv_token_bytes(const KVCache* kv) {
  return (size_t)kv->config.shape.hidden * kv_elem_bytes(kv);
}

/* logical token index [0, valid_tokens) -> physical token slot [0, max_seq) */
static int kv_logical_to_physical(const KVCache* kv, int logical_token) {
  const int prefix = kv_prefix_tokens_runtime(kv);
  const int recent_cap = kv_recent_capacity(kv);
  const int recent_base = kv_prefix_capacity(kv);

  if (logical_token < 0 || logical_token >= kv->state.valid_tokens) return -1;
  if (!kv->state.prefix_frozen) return logical_token;

  if (logical_token < prefix) {
    return logical_token;
  }

  /* recent logical region */
  const int recent_logical = logical_token - prefix;

  if (!kv->config.policy.use_ring_buffer) {
    int recent_start = kv->state.cursor - kv->state.recent_size;
    if (recent_start < recent_base) recent_start = recent_base;
    return recent_start + recent_logical;
  } else {
    if (recent_cap <= 0) return -1;
    /* oldest recent starts at recent_head - recent_size */
    int oldest = kv->state.recent_head - kv->state.recent_size;
    while (oldest < 0) oldest += recent_cap;
    return recent_base + ((oldest + recent_logical) % recent_cap);
  }
}

size_t KVRequiredBytes(const KVConfig* config) {
  const size_t elem_bytes = config ? kv_dtype_elem_bytes(config->dtype) : 0;
  if (!kv_valid_config(config)) return 0;
  const size_t t = kv_tensor_bytes(&config->shape, elem_bytes);
  return KV_ALIGN + kv_align_up(t) + kv_align_up(t);
}

KVStatus KVInit(
    KVCache* kv,
    const KVConfig* config) {
  const size_t bytes = KVRequiredBytes(config);
  void* arena = 0;
  KVStatus st;

  if (!kv || !config) return KV_ERR_NULL;
  if (bytes == 0) return KV_ERR_BAD_ARG;

  arena = malloc(bytes);
  if (!arena) return KV_ERR_NO_SPACE;

  st = KVInitPreallocated(kv, config, arena, bytes);
  if (st != KV_OK) {
    free(arena);
    return st;
  }
  kv->owns_arena = 1;
  return KV_OK;
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
  kv->owns_arena = 0;

  p = (uint8_t*)kv_align_up((size_t)arena_base);
  t = kv_tensor_bytes(&config->shape, kv_dtype_elem_bytes(config->dtype));

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

  kv->initialized = true;
  KVReset(kv);
  return KV_OK;
}

void KVReset(KVCache* kv) {
  if (!kv) return;
  kv->state.cursor = 0;
  kv->state.valid_tokens = 0;
  kv->state.base_token = 0;
  kv->state.prefix_frozen = false;
  kv->state.prefix_tokens = 0;
  kv->state.recent_head = 0;
  kv->state.recent_size = 0;
}

KVStatus KVRelease(KVCache* kv) {
  if (!kv) return KV_ERR_NULL;
  if (kv->owns_arena && kv->arena_base) {
    free(kv->arena_base);
  }
  memset(kv, 0, sizeof(*kv));
  return KV_OK;
}

KVStatus KVSealPrefix(
    KVCache* kv,
    int prefix_tokens) {
  int current_recent;
  int recent_capacity;
  int prefix_base_capacity;
  int keep_recent;
  int drop_recent;
  int src_recent;
  int dst_recent;
  int l;
  size_t bytes;
  uint8_t* k;
  uint8_t* v;

  if (!kv) return KV_ERR_NULL;
  if (!kv->initialized) return KV_ERR_BAD_STATE;
  if (kv->state.prefix_frozen) return KV_ERR_BAD_STATE;
  if (prefix_tokens < 0) return KV_ERR_BAD_ARG;
  if (prefix_tokens > kv->state.valid_tokens) return KV_ERR_BAD_ARG;
  prefix_base_capacity = kv->config.policy.max_prefix_tokens;
  if (prefix_tokens > prefix_base_capacity) return KV_ERR_BAD_ARG;

  if (kv->config.policy.use_ring_buffer) {
    recent_capacity = kv->config.shape.max_seq - prefix_tokens;
  } else {
    recent_capacity = kv->config.policy.recent_keep;
  }
  kv->state.prefix_tokens = prefix_tokens;
  kv->state.prefix_frozen = true;

  current_recent = kv->state.valid_tokens - prefix_tokens;
  if (current_recent < 0) current_recent = 0;
  keep_recent = current_recent;
  if (keep_recent > recent_capacity) keep_recent = recent_capacity;
  drop_recent = current_recent - keep_recent;

  src_recent = prefix_tokens + (current_recent - keep_recent);
  dst_recent = kv_prefix_capacity(kv);
  bytes = kv_token_bytes(kv);

  for (l = 0; l < kv->config.shape.layers; ++l) {
    k = kv_token_ptr_mut(kv, kv->arena.k_data, l, 0, kv_elem_bytes(kv));
    v = kv_token_ptr_mut(kv, kv->arena.v_data, l, 0, kv_elem_bytes(kv));

    if (keep_recent > 0 && src_recent != dst_recent) {
      memmove(k + (size_t)dst_recent * bytes,
              k + (size_t)src_recent * bytes,
              (size_t)keep_recent * bytes);
      memmove(v + (size_t)dst_recent * bytes,
              v + (size_t)src_recent * bytes,
              (size_t)keep_recent * bytes);
    }
  }

  kv->state.base_token += drop_recent;
  kv->state.recent_size = keep_recent;
  kv->state.valid_tokens = prefix_tokens + keep_recent;
  kv->state.cursor = kv_prefix_capacity(kv) + keep_recent;

  if (kv->config.policy.use_ring_buffer) {
    kv->state.recent_head = recent_capacity > 0 ? (keep_recent % recent_capacity) : 0;
  } else {
    kv->state.recent_head = 0;
  }

  return KV_OK;
}

/* Compact path for non-ring mode: [prefix] + [last recent_capacity] */
KVStatus KVCompact(KVCache* kv) {
  const int prefix = kv_prefix_tokens_runtime(kv);
  const int valid = kv->state.valid_tokens;
  const int cursor = kv->state.cursor;
  const int recent = valid - prefix;
  const int recent_cap = kv_recent_capacity(kv);
  int keep_recent;
  int src_recent;
  int dst_recent;
  int l;
  size_t bytes;
  uint8_t* k;
  uint8_t* v;

  if (!kv) return KV_ERR_NULL;
  if (!kv->initialized) return KV_ERR_BAD_STATE;
  if (!kv->state.prefix_frozen) return KV_ERR_BAD_STATE;

  if (kv->config.policy.use_ring_buffer) {
    /* Ring mode does not require memmove compaction for recent zone. */
    return KV_OK;
  }

  keep_recent = recent;
  if (keep_recent > recent_cap) {
    keep_recent = recent_cap;
  }
  if (keep_recent < 0) keep_recent = 0;

  src_recent = cursor - keep_recent;
  dst_recent = kv_prefix_capacity(kv);
  bytes = kv_token_bytes(kv);

  for (l = 0; l < kv->config.shape.layers; ++l) {
    k = kv_token_ptr_mut(kv, kv->arena.k_data, l, 0, kv_elem_bytes(kv));
    v = kv_token_ptr_mut(kv, kv->arena.v_data, l, 0, kv_elem_bytes(kv));

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
  kv->state.cursor = kv_prefix_capacity(kv) + keep_recent;
  return KV_OK;
}

static KVStatus kv_non_ring_make_room_for_append(KVCache* kv) {
  const int prefix = kv_prefix_tokens_runtime(kv);
  const int recent_base = kv_prefix_capacity(kv);
  const int recent_cap = kv_recent_capacity(kv);
  const int max_seq = kv->config.shape.max_seq;
  int keep_recent;
  int l;
  size_t bytes;
  uint8_t* k;
  uint8_t* v;

  if (!kv->state.prefix_frozen) {
    if (kv->state.cursor < max_seq) return KV_OK;
    return KV_ERR_NO_SPACE;
  }
  if (recent_cap <= 0) return KV_ERR_NO_SPACE;
  if (kv->state.cursor < max_seq) return KV_OK;

  {
    KVStatus st = KVCompact(kv);
    if (st != KV_OK) return st;
  }
  if (kv->state.cursor < max_seq) return KV_OK;

  if (kv->state.recent_size <= 0) return KV_ERR_NO_SPACE;

  keep_recent = kv->state.recent_size - 1;
  bytes = kv_token_bytes(kv);

  for (l = 0; l < kv->config.shape.layers; ++l) {
    k = kv_token_ptr_mut(kv, kv->arena.k_data, l, 0, kv_elem_bytes(kv));
    v = kv_token_ptr_mut(kv, kv->arena.v_data, l, 0, kv_elem_bytes(kv));
    if (keep_recent > 0) {
      memmove(k + (size_t)recent_base * bytes,
              k + (size_t)(recent_base + 1) * bytes,
              (size_t)keep_recent * bytes);
      memmove(v + (size_t)recent_base * bytes,
              v + (size_t)(recent_base + 1) * bytes,
              (size_t)keep_recent * bytes);
    }
  }

  kv->state.base_token += 1;
  kv->state.recent_size = keep_recent;
  kv->state.valid_tokens = prefix + keep_recent;
  kv->state.cursor = recent_base + keep_recent;
  return KV_OK;
}

static KVStatus kv_append_contiguous(
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
    KVStatus st;
    if (kv->state.prefix_frozen && recent_cap <= 0) return KV_ERR_NO_SPACE;
    st = kv_non_ring_make_room_for_append(kv);
    if (st != KV_OK) return st;
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
    uint8_t* dst_k = kv_token_ptr_mut(kv, kv->arena.k_data, l, phys_slot, kv_elem_bytes(kv));
    uint8_t* dst_v = kv_token_ptr_mut(kv, kv->arena.v_data, l, phys_slot, kv_elem_bytes(kv));
    const uint8_t* src_k = (const uint8_t*)new_k_layers + (size_t)l * layer_stride_bytes;
    const uint8_t* src_v = (const uint8_t*)new_v_layers + (size_t)l * layer_stride_bytes;
    memcpy(dst_k, src_k, bytes);
    memcpy(dst_v, src_v, bytes);
  }

  if (!kv->config.policy.use_ring_buffer) {
    kv->state.cursor += 1;
    if (!kv->state.prefix_frozen) {
      if (kv->state.valid_tokens < kv->config.shape.max_seq) kv->state.valid_tokens += 1;
    } else {
      if (kv->state.recent_size < recent_cap) {
        kv->state.recent_size += 1;
        kv->state.valid_tokens = prefix + kv->state.recent_size;
      } else {
        kv->state.base_token += 1;
      }
      return KV_OK;
    }
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
  const size_t elem_bytes = kv ? kv_elem_bytes(kv) : 0;
  if (!kv || !new_k_layers || !new_v_layers) return KV_ERR_NULL;
  if (!kv->initialized) return KV_ERR_BAD_STATE;
  if (elem_bytes == 0) return KV_ERR_UNSUPPORTED;

  return kv_append_contiguous(
      kv,
      new_k_layers,
      new_v_layers,
      (size_t)kv->config.shape.hidden * elem_bytes);
}

KVStatus KVAppendViewWrite(
    KVCache* kv,
    const KVAppendView* view) {
  if (!kv || !view) return KV_ERR_NULL;
  if (!view->k || !view->v) return KV_ERR_NULL;
  if (view->layer_stride_bytes < kv_token_bytes(kv)) return KV_ERR_BAD_ARG;

  return kv_append_contiguous(
      kv,
      view->k,
      view->v,
      view->layer_stride_bytes);
}

KVStatus KVReserveTokenSlot(KVCache* kv, int* out_token) {
  int phys_slot;
  if (!kv || !out_token) return KV_ERR_NULL;
  if (!kv->initialized) return KV_ERR_BAD_STATE;
  if (kv_elem_bytes(kv) == 0) return KV_ERR_UNSUPPORTED;

  if (!kv->config.policy.use_ring_buffer) {
    KVStatus st;
    if (kv->state.prefix_frozen && kv_recent_capacity(kv) <= 0) return KV_ERR_NO_SPACE;
    st = kv_non_ring_make_room_for_append(kv);
    if (st != KV_OK) return st;
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

KVStatus KVCommitToken(KVCache* kv, int token) {
  const int prefix = kv_prefix_tokens_runtime(kv);
  const int recent_cap = kv_recent_capacity(kv);

  if (!kv) return KV_ERR_NULL;
  if (!kv->initialized) return KV_ERR_BAD_STATE;

  if (!kv->config.policy.use_ring_buffer) {
    if (kv->state.prefix_frozen && recent_cap <= 0) return KV_ERR_NO_SPACE;
    if (token != kv->state.cursor) return KV_ERR_BAD_ARG;
    kv->state.cursor += 1;
    if (!kv->state.prefix_frozen) {
      if (kv->state.valid_tokens < kv->config.shape.max_seq) kv->state.valid_tokens += 1;
    } else {
      if (kv->state.recent_size < recent_cap) {
        kv->state.recent_size += 1;
        kv->state.valid_tokens = prefix + kv->state.recent_size;
      } else {
        kv->state.base_token += 1;
      }
      return KV_OK;
    }
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

static KVStatus kv_attention_write_layer_raw(
    KVCache* kv,
    int layer,
    int token,
    const void* k_data,
    const void* v_data) {
  const size_t bytes = kv_token_bytes(kv);
  const size_t elem_bytes = kv_elem_bytes(kv);
  uint8_t* dst_k;
  uint8_t* dst_v;

  if (!kv || !k_data || !v_data) return KV_ERR_NULL;
  if (!kv->initialized) return KV_ERR_BAD_STATE;
  if (elem_bytes == 0) return KV_ERR_UNSUPPORTED;
  if (layer < 0 || layer >= kv->config.shape.layers) return KV_ERR_BAD_ARG;
  if (token < 0 || token >= kv->config.shape.max_seq) return KV_ERR_BAD_ARG;

  dst_k = kv_token_ptr_mut(kv, kv->arena.k_data, layer, token, elem_bytes);
  dst_v = kv_token_ptr_mut(kv, kv->arena.v_data, layer, token, elem_bytes);
  memcpy(dst_k, k_data, bytes);
  memcpy(dst_v, v_data, bytes);
  return KV_OK;
}

static KVStatus kv_attention_read_layer_raw(
    const KVCache* kv,
    int layer,
    int logical_token,
    const void** out_k,
    const void** out_v) {
  const size_t elem_bytes = kv ? kv_elem_bytes(kv) : 0;
  int phys;
  const void* k;
  const void* v;

  if (!kv || !out_k || !out_v) return KV_ERR_NULL;
  if (!kv->initialized) return KV_ERR_BAD_STATE;
  if (elem_bytes == 0) return KV_ERR_UNSUPPORTED;
  if (layer < 0 || layer >= kv->config.shape.layers) return KV_ERR_BAD_ARG;
  if (logical_token < 0 || logical_token >= kv->state.valid_tokens) return KV_ERR_BAD_ARG;

  phys = kv_logical_to_physical(kv, logical_token);
  if (phys < 0) return KV_ERR_BAD_ARG;
  k = (const void*)kv_token_ptr_const(kv, kv->arena.k_data, layer, phys, elem_bytes);
  v = (const void*)kv_token_ptr_const(kv, kv->arena.v_data, layer, phys, elem_bytes);

  *out_k = k;
  *out_v = v;
  return KV_OK;
}

KVStatus KVAttentionWriteLayer(
    KVCache* kv,
    int layer,
    int token,
    const KVAttentionLayerArg* arg) {
  if (!kv || !arg) return KV_ERR_NULL;
  if (!kv->initialized) return KV_ERR_BAD_STATE;
  if (arg->dtype != kv->config.dtype) return KV_ERR_BAD_ARG;

  switch (arg->dtype) {
    case KV_DTYPE_FP16:
      return kv_attention_write_layer_raw(
          kv,
          layer,
          token,
          arg->k_data,
          arg->v_data);
    case KV_DTYPE_INT8:
      return kv_attention_write_layer_raw(
          kv,
          layer,
          token,
          arg->k_data,
          arg->v_data);
    default:
      return KV_ERR_BAD_ARG;
  }
}

KVStatus KVAttentionReadLayer(
    const KVCache* kv,
    int layer,
    int logical_token,
    KVAttentionLayerArg* arg) {
  const void* k;
  const void* v;
  KVStatus st;

  if (!kv || !arg) return KV_ERR_NULL;
  if (!kv->initialized) return KV_ERR_BAD_STATE;
  if (arg->dtype != kv->config.dtype) return KV_ERR_BAD_ARG;

  switch (arg->dtype) {
    case KV_DTYPE_FP16:
      k = 0;
      v = 0;
      st = kv_attention_read_layer_raw(kv, layer, logical_token, &k, &v);
      if (st != KV_OK) return st;
      arg->k_data = k;
      arg->v_data = v;
      return KV_OK;
    case KV_DTYPE_INT8:
      k = 0;
      v = 0;
      st = kv_attention_read_layer_raw(kv, layer, logical_token, &k, &v);
      if (st != KV_OK) return st;
      arg->k_data = k;
      arg->v_data = v;
      return KV_OK;
    default:
      return KV_ERR_BAD_ARG;
  }
}

int KVValidTokens(const KVCache* kv) {
  return kv ? kv->state.valid_tokens : 0;
}

KVStatus KVGetSnapshot(const KVCache* kv, KVSnapshot* snapshot) {
  if (!kv || !snapshot) return KV_ERR_NULL;

  snapshot->valid_tokens = kv->state.valid_tokens;
  snapshot->cursor = kv->state.cursor;
  snapshot->base_token = kv->state.base_token;
  snapshot->prefix_tokens = kv_prefix_tokens_runtime(kv);
  snapshot->recent_tokens = kv->state.recent_size;
  snapshot->dtype = kv->config.dtype;
  snapshot->initialized = kv->initialized;
  return KV_OK;
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

bool KVIsInitialized(const KVCache* kv) {
  return kv ? kv->initialized : false;
}

int KVRecentFirstPhysicalSlot(const KVCache* kv) {
    if (!kv || !kv->initialized) return 0;

    const int recent_capacity = kv_recent_capacity(kv);
    if (recent_capacity <= 0) return 0;

    if (!kv->config.policy.use_ring_buffer) {
        if (!kv->state.prefix_frozen || kv->state.recent_size <= 0) return 0;
        return kv->state.cursor - kv->state.recent_size;
    }

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
    const int recent_capacity = kv_recent_capacity(kv);

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

    if (kv->config.policy.use_ring_buffer) {
        view->recent_wrapped = (first_slot + recent > recent_capacity);
    } else {
        view->recent_wrapped = false;
    }

    view->k_base = kv->arena.k_data;
    view->v_base = kv->arena.v_data;

    view->layer_stride_bytes =
        kv->config.shape.max_seq *
        kv->config.shape.hidden *
        kv_elem_bytes(kv);

    view->token_stride_bytes =
        kv->config.shape.hidden *
        kv_elem_bytes(kv);

    return KV_OK;
}
