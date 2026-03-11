#include "mobilekv/kv_cache.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <vector>

namespace {

KVConfig MakeConfig(int layers, int hidden, int max_seq, int max_prefix, int recent_keep) {
  KVConfig cfg{};
  cfg.shape.layers = layers;
  cfg.shape.heads = 1;
  cfg.shape.head_dim = hidden;
  cfg.shape.hidden = hidden;
  cfg.shape.max_seq = max_seq;
  cfg.policy.max_prefix_tokens = max_prefix;
  cfg.policy.recent_keep = recent_keep;
  cfg.dtype = KV_DTYPE_FP16;
  cfg.layout = KV_LAYOUT_LAYER_TOKEN_HIDDEN;
  cfg.backend = KV_BACKEND_CPU;
  return cfg;
}

struct KVFixture {
  KVConfig cfg;
  std::vector<uint8_t> arena;
  KVCache kv{};
};

KVFixture CreateFixture(const KVConfig& cfg) {
  KVFixture f{};
  f.cfg = cfg;
  const size_t bytes = KVRequiredBytes(&f.cfg);
  f.arena.resize(bytes);
  EXPECT_EQ(KVInitPreallocated(&f.kv, &f.cfg, f.arena.data(), f.arena.size()), KV_OK);
  return f;
}

}  // namespace

TEST(KVCacheLifecycleTest, RequiredBytesSanity) {
  KVConfig cfg = MakeConfig(/*layers=*/2, /*hidden=*/8, /*max_seq=*/16, /*max_prefix=*/4, /*recent_keep=*/8);

  EXPECT_GT(KVRequiredBytes(&cfg), 0u);
  EXPECT_EQ(KVRequiredBytes(nullptr), 0u);

  cfg.dtype = KV_DTYPE_INT8;
  EXPECT_EQ(KVRequiredBytes(&cfg), 0u);
}

TEST(KVCacheAppendTest, InitAndAppendRoundTrip) {
  KVFixture fixture = CreateFixture(MakeConfig(/*layers=*/2, /*hidden=*/4, /*max_seq=*/8, /*max_prefix=*/0, /*recent_keep=*/8));

  std::array<uint16_t, 8> k = {11, 12, 13, 14, 21, 22, 23, 24};
  std::array<uint16_t, 8> v = {31, 32, 33, 34, 41, 42, 43, 44};

  EXPECT_EQ(KVAppend(&fixture.kv, k.data(), v.data()), KV_OK);
  EXPECT_EQ(KVValidTokens(&fixture.kv), 1);
  EXPECT_EQ(KVCursor(&fixture.kv), 1);
  EXPECT_EQ(KVLayout(&fixture.kv), KV_LAYOUT_LAYER_TOKEN_HIDDEN);
  EXPECT_EQ(KVDType(&fixture.kv), KV_DTYPE_FP16);

  const uint16_t* k_l0_t0 = KVKTokenFP16(&fixture.kv, 0, 0);
  const uint16_t* k_l1_t0 = KVKTokenFP16(&fixture.kv, 1, 0);
  ASSERT_NE(k_l0_t0, nullptr);
  ASSERT_NE(k_l1_t0, nullptr);
  EXPECT_EQ(k_l0_t0[0], 11);
  EXPECT_EQ(k_l0_t0[3], 14);
  EXPECT_EQ(k_l1_t0[0], 21);
  EXPECT_EQ(k_l1_t0[3], 24);

  KVReadView view{};
  EXPECT_EQ(KVGetReadView(&fixture.kv, &view), KV_OK);
  EXPECT_EQ(view.segment_count, 1);
  EXPECT_EQ(view.segments[0].start_token, 0);
  EXPECT_EQ(view.segments[0].token_count, 1);
}

TEST(KVCacheCompactionTest, CompactKeepsPrefixAndRecentWindow) {
  KVFixture fixture = CreateFixture(MakeConfig(/*layers=*/1, /*hidden=*/2, /*max_seq=*/4, /*max_prefix=*/1, /*recent_keep=*/2));

  for (int token = 0; token < 4; ++token) {
    std::array<uint16_t, 2> k = {
        static_cast<uint16_t>(token * 10 + 1),
        static_cast<uint16_t>(token * 10 + 2),
    };
    std::array<uint16_t, 2> v = {
        static_cast<uint16_t>(token * 10 + 101),
        static_cast<uint16_t>(token * 10 + 102),
    };
    ASSERT_EQ(KVAppend(&fixture.kv, k.data(), v.data()), KV_OK);
  }

  ASSERT_EQ(KVSealPrefix(&fixture.kv, 1), KV_OK);

  std::array<uint16_t, 2> k5 = {41, 42};
  std::array<uint16_t, 2> v5 = {141, 142};
  ASSERT_EQ(KVAppend(&fixture.kv, k5.data(), v5.data()), KV_OK);

  EXPECT_EQ(KVValidTokens(&fixture.kv), 4);
  EXPECT_EQ(KVCursor(&fixture.kv), 4);
  EXPECT_EQ(KVPrefixTokens(&fixture.kv), 1);
  EXPECT_EQ(KVBaseToken(&fixture.kv), 1);

  const uint16_t* t0 = KVKTokenFP16(&fixture.kv, 0, 0);
  const uint16_t* t1 = KVKTokenFP16(&fixture.kv, 0, 1);
  const uint16_t* t2 = KVKTokenFP16(&fixture.kv, 0, 2);
  const uint16_t* t3 = KVKTokenFP16(&fixture.kv, 0, 3);
  ASSERT_NE(t0, nullptr);
  ASSERT_NE(t1, nullptr);
  ASSERT_NE(t2, nullptr);
  ASSERT_NE(t3, nullptr);

  EXPECT_EQ(t0[0], 1);
  EXPECT_EQ(t1[0], 21);
  EXPECT_EQ(t2[0], 31);
  EXPECT_EQ(t3[0], 41);
}

TEST(KVCacheCompactionTest, SealPrefixRejectsExceedingPolicy) {
  KVFixture fixture = CreateFixture(MakeConfig(/*layers=*/1, /*hidden=*/2, /*max_seq=*/8, /*max_prefix=*/2, /*recent_keep=*/8));

  for (int token = 0; token < 3; ++token) {
    std::array<uint16_t, 2> k = {
        static_cast<uint16_t>(token * 10 + 1),
        static_cast<uint16_t>(token * 10 + 2),
    };
    std::array<uint16_t, 2> v = {
        static_cast<uint16_t>(token * 10 + 101),
        static_cast<uint16_t>(token * 10 + 102),
    };
    ASSERT_EQ(KVAppend(&fixture.kv, k.data(), v.data()), KV_OK);
  }

  EXPECT_EQ(KVSealPrefix(&fixture.kv, 3), KV_ERR_BAD_ARG);
  EXPECT_EQ(KVSealPrefix(&fixture.kv, 2), KV_OK);
}

TEST(KVCacheAppendViewTest, AppendViewRejectsTooSmallStride) {
  KVFixture fixture = CreateFixture(MakeConfig(/*layers=*/2, /*hidden=*/4, /*max_seq=*/8, /*max_prefix=*/0, /*recent_keep=*/8));

  std::array<uint16_t, 16> k = {
      11, 12, 13, 14, 21, 22, 23, 24,
      31, 32, 33, 34, 41, 42, 43, 44,
  };
  std::array<uint16_t, 16> v = {
      111, 112, 113, 114, 121, 122, 123, 124,
      131, 132, 133, 134, 141, 142, 143, 144,
  };

  KVAppendView bad_view{};
  bad_view.k = k.data();
  bad_view.v = v.data();
  bad_view.layer_stride_bytes = sizeof(uint16_t) * 2;  // hidden=4 -> should be at least 8 bytes
  EXPECT_EQ(KVAppendViewWrite(&fixture.kv, &bad_view), KV_ERR_BAD_ARG);
  EXPECT_EQ(KVValidTokens(&fixture.kv), 0);

  KVAppendView good_view{};
  good_view.k = k.data();
  good_view.v = v.data();
  good_view.layer_stride_bytes = sizeof(uint16_t) * 4;
  EXPECT_EQ(KVAppendViewWrite(&fixture.kv, &good_view), KV_OK);
  EXPECT_EQ(KVValidTokens(&fixture.kv), 1);
}

TEST(KVCacheReservationTest, ReserveWriteCommitTokenFlow) {
  KVFixture fixture = CreateFixture(MakeConfig(/*layers=*/2, /*hidden=*/4, /*max_seq=*/8, /*max_prefix=*/0, /*recent_keep=*/8));

  int token = -1;
  ASSERT_EQ(KVReserveTokenSlot(&fixture.kv, &token), KV_OK);
  EXPECT_EQ(token, 0);
  EXPECT_EQ(KVCursor(&fixture.kv), 0);
  EXPECT_EQ(KVValidTokens(&fixture.kv), 0);

  std::array<uint16_t, 4> k_l0 = {1, 2, 3, 4};
  std::array<uint16_t, 4> v_l0 = {11, 12, 13, 14};
  std::array<uint16_t, 4> k_l1 = {21, 22, 23, 24};
  std::array<uint16_t, 4> v_l1 = {31, 32, 33, 34};

  ASSERT_EQ(KVWriteLayerToken(&fixture.kv, 0, token, k_l0.data(), v_l0.data(), sizeof(k_l0)), KV_OK);
  ASSERT_EQ(KVWriteLayerToken(&fixture.kv, 1, token, k_l1.data(), v_l1.data(), sizeof(k_l1)), KV_OK);

  EXPECT_EQ(KVCommitToken(&fixture.kv, token), KV_OK);
  EXPECT_EQ(KVCursor(&fixture.kv), 1);
  EXPECT_EQ(KVValidTokens(&fixture.kv), 1);

  const uint16_t* got_k0 = KVKTokenFP16(&fixture.kv, 0, 0);
  const uint16_t* got_k1 = KVKTokenFP16(&fixture.kv, 1, 0);
  ASSERT_NE(got_k0, nullptr);
  ASSERT_NE(got_k1, nullptr);
  EXPECT_EQ(got_k0[0], 1);
  EXPECT_EQ(got_k0[3], 4);
  EXPECT_EQ(got_k1[0], 21);
  EXPECT_EQ(got_k1[3], 24);

  EXPECT_EQ(KVCommitToken(&fixture.kv, token), KV_ERR_BAD_STATE);
}

TEST(KVCacheReservationTest, ReserveWriteCommitArgumentValidation) {
  KVFixture fixture = CreateFixture(MakeConfig(/*layers=*/1, /*hidden=*/4, /*max_seq=*/4, /*max_prefix=*/0, /*recent_keep=*/4));
  int token = -1;
  ASSERT_EQ(KVReserveTokenSlot(&fixture.kv, &token), KV_OK);
  ASSERT_EQ(token, 0);

  std::array<uint16_t, 4> k = {1, 2, 3, 4};
  std::array<uint16_t, 4> v = {11, 12, 13, 14};

  EXPECT_EQ(KVWriteLayerToken(&fixture.kv, -1, token, k.data(), v.data(), sizeof(k)), KV_ERR_BAD_ARG);
  EXPECT_EQ(KVWriteLayerToken(&fixture.kv, 0, -1, k.data(), v.data(), sizeof(k)), KV_ERR_BAD_ARG);
  EXPECT_EQ(KVWriteLayerToken(&fixture.kv, 0, token, k.data(), v.data(), sizeof(uint16_t) * 3), KV_ERR_BAD_ARG);
}

TEST(KVCacheCompatTest, ReadSpanAndCompatSpanExposeSameData) {
  KVFixture fixture = CreateFixture(MakeConfig(/*layers=*/2, /*hidden=*/4, /*max_seq=*/8, /*max_prefix=*/0, /*recent_keep=*/8));

  std::array<uint16_t, 8> k = {11, 12, 13, 14, 21, 22, 23, 24};
  std::array<uint16_t, 8> v = {31, 32, 33, 34, 41, 42, 43, 44};
  ASSERT_EQ(KVAppend(&fixture.kv, k.data(), v.data()), KV_OK);

  KVLayerReadSpan read_span{};
  ASSERT_EQ(KVGetLayerReadSpan(&fixture.kv, 1, &read_span), KV_OK);
  EXPECT_EQ(read_span.token_count, 1);
  EXPECT_EQ(read_span.token_stride_bytes, sizeof(uint16_t) * 4);

  const uint16_t* layer1_k = static_cast<const uint16_t*>(read_span.k_base);
  const uint16_t* layer1_v = static_cast<const uint16_t*>(read_span.v_base);
  ASSERT_NE(layer1_k, nullptr);
  ASSERT_NE(layer1_v, nullptr);
  EXPECT_EQ(layer1_k[0], 21);
  EXPECT_EQ(layer1_k[3], 24);
  EXPECT_EQ(layer1_v[0], 41);
  EXPECT_EQ(layer1_v[3], 44);

  KVCompatLlamaSpan compat{};
  ASSERT_EQ(KVCompatGetLlamaSpan(&fixture.kv, 1, &compat), KV_OK);
  EXPECT_EQ(compat.k, read_span.k_base);
  EXPECT_EQ(compat.v, read_span.v_base);
  EXPECT_EQ(compat.n_tokens, 1);
  EXPECT_EQ(compat.hidden, 4);
  EXPECT_EQ(compat.stride_bytes, sizeof(uint16_t) * 4);
}
