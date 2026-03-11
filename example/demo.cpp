
#include "mobilekv/kv_cache.h"
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

int main() {

  KVShape shape = {4, 8, 64, 512, 256};

  KVPolicy policy;
  policy.max_prefix_tokens = 16;
  policy.recent_keep = 200;

  KVConfig cfg;
  cfg.shape = shape;
  cfg.policy = policy;
  cfg.dtype = KV_DTYPE_FP16;
  cfg.layout = KV_LAYOUT_LAYER_TOKEN_HIDDEN;
  cfg.backend = KV_BACKEND_CPU;

  size_t bytes = KVRequiredBytes(&cfg);

  void* arena = malloc(bytes);

  KVCache kv;

  assert(KVInitPreallocated(&kv, &cfg, arena, bytes) == KV_OK);

  printf("KV init OK\n");

  free(arena);

  return 0;
}