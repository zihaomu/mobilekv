#include "mobilekv/kv_cache.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/*
 * Minimal adapter-facing metadata for a llama.cpp-style attention kernel.
 * This keeps ring reconstruction information explicit instead of forcing
 * the kernel to understand KVCache internals.
 */
typedef struct {
  const uint16_t* k_layer_base;
  const uint16_t* v_layer_base;

  int q_pos;
  int visible_tokens;
  int prefix_tokens;

  int recent_logical_start;
  int recent_size;
  int recent_capacity;
  int recent_first_slot;
  int recent_wrapped;

  size_t token_stride_elems;
} LlamaAttentionMeta;

static KVConfig MakeConfig(void) {
  KVConfig cfg{};
  cfg.shape.layers = 1;
  cfg.shape.heads = 1;
  cfg.shape.head_dim = 4;
  cfg.shape.hidden = 4;
  cfg.shape.max_seq = 6;
  cfg.policy.max_prefix_tokens = 2;
  cfg.policy.recent_keep = 4;
  cfg.policy.use_ring_buffer = true;
  cfg.dtype = KV_DTYPE_FP16;
  cfg.backend = KV_BACKEND_CPU;
  return cfg;
}

static void FillToken(uint16_t* out, int hidden, uint16_t base) {
  for (int i = 0; i < hidden; ++i) {
    out[i] = (uint16_t)(base + i);
  }
}

static int BuildLlamaAttentionMeta(
    const KVCache* kv,
    int layer,
    int q_pos,
    LlamaAttentionMeta* out) {
  if (!kv || !out) return 0;
  if (layer < 0 || layer >= kv->config.shape.layers) return 0;

  KVAttentionView view{};
  if (KVGetAttentionView(kv, q_pos, &view) != KV_OK) return 0;

  const uint8_t* k_base = (const uint8_t*)view.k_base;
  const uint8_t* v_base = (const uint8_t*)view.v_base;
  const size_t layer_off = (size_t)layer * view.layer_stride_bytes;

  out->k_layer_base = (const uint16_t*)(k_base + layer_off);
  out->v_layer_base = (const uint16_t*)(v_base + layer_off);

  out->q_pos = view.q_pos;
  out->visible_tokens = view.visible_tokens;
  out->prefix_tokens = view.prefix_tokens;
  out->recent_logical_start = view.recent_logical_start;
  out->recent_size = view.recent_size;
  out->recent_capacity = view.recent_capacity;
  out->recent_first_slot = view.recent_first_slot;
  out->recent_wrapped = view.recent_wrapped;
  out->token_stride_elems = view.token_stride_bytes / sizeof(uint16_t);
  return 1;
}

static int PhysicalSlotFromMeta(const LlamaAttentionMeta* m, int logical_token) {
  if (logical_token < m->prefix_tokens) {
    return logical_token;
  }

  const int recent_idx = logical_token - m->prefix_tokens;
  const int ring_slot = (m->recent_first_slot + recent_idx) % m->recent_capacity;
  return m->prefix_tokens + ring_slot;
}

int main(void) {
  const int prefix_tokens = 2;
  KVConfig cfg = MakeConfig();
  const size_t bytes = KVRequiredBytes(&cfg);
  void* arena = malloc(bytes);
  assert(arena != NULL);

  KVCache kv{};
  assert(KVInitPreallocated(&kv, &cfg, arena, bytes) == KV_OK);

  uint16_t token[4];
  FillToken(token, cfg.shape.hidden, 100);
  assert(KVAppend(&kv, token, token) == KV_OK);
  FillToken(token, cfg.shape.hidden, 200);
  assert(KVAppend(&kv, token, token) == KV_OK);
  assert(KVSealPrefix(&kv, prefix_tokens) == KV_OK);

  for (int i = 0; i < 6; ++i) {
    FillToken(token, cfg.shape.hidden, (uint16_t)(300 + i * 10));
    assert(KVAppend(&kv, token, token) == KV_OK);
  }

  const int q_pos = KVBaseToken(&kv) + KVValidTokens(&kv);
  LlamaAttentionMeta meta{};
  assert(BuildLlamaAttentionMeta(&kv, /*layer=*/0, q_pos, &meta) == 1);

  printf("=== llama.cpp Adapter Skeleton ===\n");
  printf("q_pos=%d visible=%d prefix=%d recent=%d recent_start=%d first_slot=%d wrapped=%d\n",
         meta.q_pos,
         meta.visible_tokens,
         meta.prefix_tokens,
         meta.recent_size,
         meta.recent_logical_start,
         meta.recent_first_slot,
         meta.recent_wrapped);

  for (int t = 0; t < meta.visible_tokens; ++t) {
    const int phys = PhysicalSlotFromMeta(&meta, t);
    const uint16_t* k = meta.k_layer_base + (size_t)phys * meta.token_stride_elems;
    printf("logical=%d phys=%d k0=%u\n", t, phys, (unsigned)k[0]);
  }

  free(arena);
  return 0;
}
