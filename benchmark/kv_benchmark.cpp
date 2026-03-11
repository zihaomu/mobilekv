#include "mobilekv/kv_cache.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {

  KVShape shape = {32, 32, 128, 4096, 4096};

  KVPolicy policy;
  policy.max_prefix_tokens = 128;
  policy.recent_keep = 3072;

  KVConfig cfg;
  cfg.shape = shape;
  cfg.policy = policy;
  cfg.dtype = KV_DTYPE_FP16;
  cfg.layout = KV_LAYOUT_LAYER_TOKEN_HIDDEN;
  cfg.backend = KV_BACKEND_CPU;

  size_t bytes = KVRequiredBytes(&cfg);

  void* arena = malloc(bytes);

  KVCache kv;
  KVInitPreallocated(&kv, &cfg, arena, bytes);

  int layers = shape.layers;
  int hidden = shape.hidden;

  uint16_t* k = (uint16_t*)malloc(layers * hidden * 2);
  uint16_t* v = (uint16_t*)malloc(layers * hidden * 2);

  int steps = 2000;

  clock_t t0 = clock();

  for (int i = 0; i < steps; i++) {
    KVAppend(&kv, k, v);
  }

  clock_t t1 = clock();

  double sec = (double)(t1 - t0) / CLOCKS_PER_SEC;

  printf("append tokens: %d\n", steps);
  printf("time: %.3f sec\n", sec);
  printf("throughput: %.1f tok/s\n", steps / sec);

  free(k);
  free(v);
  free(arena);

  return 0;
}