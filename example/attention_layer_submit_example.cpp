#include "mobilekv/kv_cache.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static KVConfig MakeConfig(void) {
  KVConfig cfg{};
  cfg.shape.layers = 3;
  cfg.shape.heads = 1;
  cfg.shape.head_dim = 4;
  cfg.shape.hidden = 4;
  cfg.shape.max_seq = 4;
  cfg.policy.max_prefix_tokens = 0;
  cfg.policy.recent_keep = 4;
  cfg.policy.use_ring_buffer = true;
  cfg.dtype = KV_DTYPE_FP16;
  cfg.backend = KV_BACKEND_CPU;
  return cfg;
}

static void FillLayerKV(
    int stream_token,
    int layer,
    int hidden,
    uint16_t* out_k,
    uint16_t* out_v) {
  for (int i = 0; i < hidden; ++i) {
    out_k[i] = (uint16_t)(1000 + layer * 100 + stream_token * 10 + i);
    out_v[i] = (uint16_t)(2000 + layer * 100 + stream_token * 10 + i);
  }
}

static void AppendOneTokenByLayer(KVCache* kv, const KVConfig* cfg, int stream_token) {
  int token_slot = -1;
  uint16_t k[4];
  uint16_t v[4];
  KVAttentionLayerArg write_arg{};
  write_arg.dtype = KV_DTYPE_FP16;

  assert(KVReserveTokenSlot(kv, &token_slot) == KV_OK);
  for (int layer = 0; layer < cfg->shape.layers; ++layer) {
    FillLayerKV(stream_token, layer, cfg->shape.hidden, k, v);
    write_arg.k_data = k;
    write_arg.v_data = v;
    assert(KVAttentionWriteLayer(kv, layer, token_slot, &write_arg) == KV_OK);
  }
  assert(KVCommitToken(kv, token_slot) == KV_OK);

  printf("append stream_token=%d token_slot=%d base=%d valid=%d\n",
         stream_token,
         token_slot,
         KVBaseToken(kv),
         KVValidTokens(kv));
}

int main(void) {
  const int prefix_tokens = 0;
  KVConfig cfg = MakeConfig();
  const size_t bytes = KVRequiredBytes(&cfg);
  void* arena = malloc(bytes);
  assert(arena != NULL);

  KVCache kv{};
  assert(KVInitPreallocated(&kv, &cfg, arena, bytes) == KV_OK);
  /* Seal zero-length prefix so decode writes use ring recent zone immediately. */
  assert(KVSealPrefix(&kv, prefix_tokens) == KV_OK);

  for (int t = 0; t < 6; ++t) {
    AppendOneTokenByLayer(&kv, &cfg, t);
  }

  printf("\n=== Attention Layer Readback ===\n");
  printf("visible=%d base=%d\n", KVValidTokens(&kv), KVBaseToken(&kv));

  for (int logical_t = 0; logical_t < KVValidTokens(&kv); ++logical_t) {
    for (int layer = 0; layer < cfg.shape.layers; ++layer) {
      KVAttentionLayerArg read_arg{};
      read_arg.dtype = KV_DTYPE_FP16;
      assert(KVAttentionReadLayer(&kv, layer, logical_t, &read_arg) == KV_OK);
      const uint16_t* k = (const uint16_t*)read_arg.k_data;
      const uint16_t* v = (const uint16_t*)read_arg.v_data;
      printf("logical_t=%d layer=%d k0=%u v0=%u\n",
             logical_t,
             layer,
             (unsigned)k[0],
             (unsigned)v[0]);
    }
  }

  KVAttentionView view{};
  const int q_pos = KVBaseToken(&kv) + KVValidTokens(&kv);
  assert(KVGetAttentionView(&kv, q_pos, &view) == KV_OK);

  printf("\n=== Attention View Metadata ===\n");
  printf("q_pos=%d visible=%d recent_start=%d recent_size=%d first_slot=%d wrapped=%d\n",
         view.q_pos,
         view.visible_tokens,
         view.recent_logical_start,
         view.recent_size,
         view.recent_first_slot,
         view.recent_wrapped);

  free(arena);
  return 0;
}
