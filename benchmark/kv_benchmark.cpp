#include "mobilekv/kv_cache.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef struct {
  double sec;
  double tok_per_sec;
  int valid_tokens;
  int base_token;
} BenchResult;

static const int kPrefixTokens = 128;
static const int kRecentKeep = 895;

static KVConfig MakeBenchConfig(int use_ring_buffer) {
  KVConfig cfg{};
  cfg.shape.layers = 8;
  cfg.shape.heads = 8;
  cfg.shape.head_dim = 64;
  cfg.shape.hidden = 512;
  cfg.shape.max_seq = 1024;
  cfg.policy.max_prefix_tokens = kPrefixTokens;
  cfg.policy.recent_keep = kRecentKeep;
  cfg.policy.use_ring_buffer = (use_ring_buffer != 0);
  cfg.dtype = KV_DTYPE_FP16;
  cfg.backend = KV_BACKEND_CPU;
  return cfg;
}

static void FillToken(uint16_t* data, int count, uint16_t base) {
  for (int i = 0; i < count; ++i) {
    data[i] = (uint16_t)(base + i);
  }
}

static BenchResult RunAppendBenchmark(int use_ring_buffer, int steps) {
  KVConfig cfg = MakeBenchConfig(use_ring_buffer);
  const size_t bytes = KVRequiredBytes(&cfg);
  void* arena = malloc(bytes);
  BenchResult ret{};

  if (!arena) {
    return ret;
  }

  KVCache kv{};
  if (KVInitPreallocated(&kv, &cfg, arena, bytes) != KV_OK) {
    free(arena);
    return ret;
  }

  const int one_token_elems = cfg.shape.layers * cfg.shape.hidden;
  uint16_t* k = (uint16_t*)malloc((size_t)one_token_elems * sizeof(uint16_t));
  uint16_t* v = (uint16_t*)malloc((size_t)one_token_elems * sizeof(uint16_t));
  if (!k || !v) {
    free(k);
    free(v);
    free(arena);
    return ret;
  }

  for (int i = 0; i < kPrefixTokens; ++i) {
    FillToken(k, one_token_elems, (uint16_t)(100 + i));
    FillToken(v, one_token_elems, (uint16_t)(200 + i));
    (void)KVAppend(&kv, k, v);
  }
  (void)KVSealPrefix(&kv, kPrefixTokens);

  const clock_t t0 = clock();
  for (int i = 0; i < steps; ++i) {
    FillToken(k, one_token_elems, (uint16_t)(1000 + (i & 0x7F)));
    FillToken(v, one_token_elems, (uint16_t)(2000 + (i & 0x7F)));
    (void)KVAppend(&kv, k, v);
  }
  const clock_t t1 = clock();

  ret.sec = (double)(t1 - t0) / CLOCKS_PER_SEC;
  ret.tok_per_sec = ret.sec > 0.0 ? (double)steps / ret.sec : 0.0;
  ret.valid_tokens = KVValidTokens(&kv);
  ret.base_token = KVBaseToken(&kv);

  free(k);
  free(v);
  free(arena);
  return ret;
}

int main(int argc, char** argv) {
  int steps = 50000;
  if (argc > 1) {
    steps = atoi(argv[1]);
    if (steps <= 0) steps = 50000;
  }

  BenchResult compact = RunAppendBenchmark(/*use_ring_buffer=*/0, steps);
  BenchResult ring = RunAppendBenchmark(/*use_ring_buffer=*/1, steps);

  printf("steps: %d\n", steps);
  printf("[compact] time: %.3f sec, throughput: %.1f tok/s, valid: %d, base: %d\n",
         compact.sec, compact.tok_per_sec, compact.valid_tokens, compact.base_token);
  printf("[ring]    time: %.3f sec, throughput: %.1f tok/s, valid: %d, base: %d\n",
         ring.sec, ring.tok_per_sec, ring.valid_tokens, ring.base_token);

  if (compact.tok_per_sec > 0.0) {
    printf("speedup (ring/compact): %.2fx\n", ring.tok_per_sec / compact.tok_per_sec);
  }

  return 0;
}
