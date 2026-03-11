#include "mobilekv/kv_cache.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

static KVConfig MakeConfig(int use_ring_buffer) {
  KVConfig cfg{};
  cfg.shape.layers = 1;
  cfg.shape.heads = 1;
  cfg.shape.head_dim = 4;
  cfg.shape.hidden = 4;
  cfg.shape.max_seq = 6;
  cfg.policy.max_prefix_tokens = 2;
  cfg.policy.recent_keep = 4;
  cfg.policy.use_ring_buffer = use_ring_buffer;
  cfg.dtype = KV_DTYPE_FP16;
  cfg.backend = KV_BACKEND_CPU;
  return cfg;
}

static void FillToken(uint16_t* out, int hidden, uint16_t base) {
  for (int i = 0; i < hidden; ++i) {
    out[i] = (uint16_t)(base + i);
  }
}

static void RunAttentionUsageDemo(void) {
  KVConfig cfg = MakeConfig(/*use_ring_buffer=*/1);
  cfg.policy.max_prefix_tokens = 2;
  cfg.policy.recent_keep = 4;

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
  assert(KVSealPrefix(&kv, 2) == KV_OK);

  for (int i = 0; i < 6; ++i) {
    FillToken(token, cfg.shape.hidden, (uint16_t)(300 + i * 10));
    assert(KVAppend(&kv, token, token) == KV_OK);
  }

  KVAttentionView view{};
  assert(KVGetAttentionView(&kv, /*q_pos=*/KVBaseToken(&kv) + KVValidTokens(&kv), &view) == KV_OK);

  printf("=== Attention View Demo ===\n");
  printf("q_pos=%d visible=%d prefix=%d recent=%d recent_start=%d first_slot=%d wrapped=%d\n",
         view.q_pos,
         view.visible_tokens,
         view.prefix_tokens,
         view.recent_size,
         view.recent_logical_start,
         view.recent_first_slot,
         view.recent_wrapped);

  const uint8_t* k_base = (const uint8_t*)view.k_base;
  const size_t layer0_off = 0;

  /* Prefix region is always physically contiguous at [0, prefix_tokens). */
  for (int t = 0; t < view.prefix_tokens; ++t) {
    int phys = t;
    const uint16_t* k = (const uint16_t*)(k_base + layer0_off + (size_t)phys * view.token_stride_bytes);
    printf("[prefix] logical=%d phys=%d k0=%u\n", t, phys, (unsigned)k[0]);
  }

  /*
   * Recent region uses ring slots. Attention kernel can reconstruct physical slot by:
   * phys = max_prefix_tokens + (recent_first_slot + i) % recent_capacity.
   */
  for (int i = 0; i < view.recent_size; ++i) {
    int ring_slot = (view.recent_first_slot + i) % view.recent_capacity;
    int phys = cfg.policy.max_prefix_tokens + ring_slot;
    int logical = view.recent_logical_start + i;
    const uint16_t* k = (const uint16_t*)(k_base + layer0_off + (size_t)phys * view.token_stride_bytes);
    printf("[recent] logical=%d phys=%d k0=%u\n", logical, phys, (unsigned)k[0]);
  }

  free(arena);
}

int main() {
  RunAttentionUsageDemo();
  return 0;
}
