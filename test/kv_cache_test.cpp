#include "mobilekv/kv_cache.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <vector>

namespace {

KVConfig MakeConfig(
    int layers,
    int hidden,
    int max_seq,
    int max_prefix,
    int recent_keep,
    bool use_ring = false) {
  KVConfig cfg{};
  cfg.shape.layers = layers;
  cfg.shape.heads = 1;
  cfg.shape.head_dim = hidden;
  cfg.shape.hidden = hidden;
  cfg.shape.max_seq = max_seq;
  cfg.policy.max_prefix_tokens = max_prefix;
  cfg.policy.recent_keep = recent_keep;
  cfg.policy.use_ring_buffer = use_ring ? 1 : 0;
  cfg.dtype = KV_DTYPE_FP16;
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
  KVFixture fixture = CreateFixture(MakeConfig(/*layers=*/1, /*hidden=*/2, /*max_seq=*/8, /*max_prefix=*/2, /*recent_keep=*/6));

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

  EXPECT_EQ(KVCommitToken(&fixture.kv, token), KV_ERR_BAD_ARG);
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

TEST(KVCacheReadSpanTest, ReadSpanExposesLayerData) {
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
}

TEST(KVCacheRingTest, InitWithRingPolicyStartsEmptyAndAppendWorks) {
  KVFixture fixture = CreateFixture(MakeConfig(
      /*layers=*/1,
      /*hidden=*/2,
      /*max_seq=*/8,
      /*max_prefix=*/2,
      /*recent_keep=*/6,
      /*use_ring=*/true));

  EXPECT_EQ(KVValidTokens(&fixture.kv), 0);
  std::array<uint16_t, 2> tok = {7, 8};
  EXPECT_EQ(KVAppend(&fixture.kv, tok.data(), tok.data()), KV_OK);
  EXPECT_EQ(KVValidTokens(&fixture.kv), 1);
}

TEST(KVCacheRingTest, PrefixAndRecentWrapMaintainLogicalOrder) {
  KVFixture fixture = CreateFixture(MakeConfig(
      /*layers=*/1,
      /*hidden=*/2,
      /*max_seq=*/5,
      /*max_prefix=*/2,
      /*recent_keep=*/3,
      /*use_ring=*/true));

  std::array<uint16_t, 2> p0 = {101, 102};
  std::array<uint16_t, 2> p1 = {201, 202};
  ASSERT_EQ(KVAppend(&fixture.kv, p0.data(), p0.data()), KV_OK);
  ASSERT_EQ(KVAppend(&fixture.kv, p1.data(), p1.data()), KV_OK);
  ASSERT_EQ(KVSealPrefix(&fixture.kv, 2), KV_OK);

  for (int i = 0; i < 5; ++i) {
    const uint16_t v = static_cast<uint16_t>(300 + i * 10 + 1);
    std::array<uint16_t, 2> tok = {v, static_cast<uint16_t>(v + 1)};
    ASSERT_EQ(KVAppend(&fixture.kv, tok.data(), tok.data()), KV_OK);
  }

  EXPECT_EQ(KVPrefixTokens(&fixture.kv), 2);
  EXPECT_EQ(KVRecentTokens(&fixture.kv), 3);
  EXPECT_EQ(KVValidTokens(&fixture.kv), 5);
  EXPECT_EQ(KVBaseToken(&fixture.kv), 2);

  const uint16_t* t0 = KVKTokenFP16(&fixture.kv, 0, 0);
  const uint16_t* t1 = KVKTokenFP16(&fixture.kv, 0, 1);
  const uint16_t* t2 = KVKTokenFP16(&fixture.kv, 0, 2);
  const uint16_t* t3 = KVKTokenFP16(&fixture.kv, 0, 3);
  const uint16_t* t4 = KVKTokenFP16(&fixture.kv, 0, 4);
  ASSERT_NE(t0, nullptr);
  ASSERT_NE(t1, nullptr);
  ASSERT_NE(t2, nullptr);
  ASSERT_NE(t3, nullptr);
  ASSERT_NE(t4, nullptr);

  EXPECT_EQ(t0[0], 101);
  EXPECT_EQ(t1[0], 201);
  EXPECT_EQ(t2[0], 321);
  EXPECT_EQ(t3[0], 331);
  EXPECT_EQ(t4[0], 341);

  KVReadView view{};
  ASSERT_EQ(KVGetReadView(&fixture.kv, &view), KV_OK);
  EXPECT_EQ(view.segment_count, 2);
  EXPECT_EQ(view.segments[0].token_count + view.segments[1].token_count, 3);
}

TEST(KVCacheAttentionViewTest, BuildsMetadataForNonRingPrefixAndRecent) {
  KVFixture fixture = CreateFixture(MakeConfig(
      /*layers=*/2,
      /*hidden=*/4,
      /*max_seq=*/8,
      /*max_prefix=*/2,
      /*recent_keep=*/6,
      /*use_ring=*/false));

  std::array<uint16_t, 8> p = {11, 12, 13, 14, 21, 22, 23, 24};
  std::array<uint16_t, 8> r = {31, 32, 33, 34, 41, 42, 43, 44};
  ASSERT_EQ(KVAppend(&fixture.kv, p.data(), p.data()), KV_OK);
  ASSERT_EQ(KVSealPrefix(&fixture.kv, 1), KV_OK);
  ASSERT_EQ(KVAppend(&fixture.kv, r.data(), r.data()), KV_OK);
  ASSERT_EQ(KVAppend(&fixture.kv, r.data(), r.data()), KV_OK);

  KVAttentionView view{};
  ASSERT_EQ(KVGetAttentionView(&fixture.kv, /*q_pos=*/2, &view), KV_OK);
  EXPECT_EQ(view.q_pos, 2);
  EXPECT_EQ(view.visible_tokens, 3);
  EXPECT_EQ(view.prefix_tokens, 1);
  EXPECT_EQ(view.recent_logical_start, 1);
  EXPECT_EQ(view.recent_size, 2);
  EXPECT_EQ(view.recent_capacity, 6);
  EXPECT_EQ(view.recent_first_slot, 0);
  EXPECT_EQ(view.recent_wrapped, 0);
  EXPECT_EQ(view.layer_stride_bytes, (size_t)8 * 4 * sizeof(uint16_t));
  EXPECT_EQ(view.token_stride_bytes, (size_t)4 * sizeof(uint16_t));
  EXPECT_EQ(view.k_base, fixture.kv.arena.k_data);
  EXPECT_EQ(view.v_base, fixture.kv.arena.v_data);

  EXPECT_EQ(KVRecentFirstPhysicalSlot(&fixture.kv), 0);
  EXPECT_EQ(KVRecentLogicalStart(&fixture.kv), 1);
}

TEST(KVCacheAttentionViewTest, BuildsMetadataForRingWrappedWindow) {
  KVFixture fixture = CreateFixture(MakeConfig(
      /*layers=*/1,
      /*hidden=*/2,
      /*max_seq=*/5,
      /*max_prefix=*/2,
      /*recent_keep=*/3,
      /*use_ring=*/true));

  std::array<uint16_t, 2> p0 = {101, 102};
  std::array<uint16_t, 2> p1 = {201, 202};
  ASSERT_EQ(KVAppend(&fixture.kv, p0.data(), p0.data()), KV_OK);
  ASSERT_EQ(KVAppend(&fixture.kv, p1.data(), p1.data()), KV_OK);
  ASSERT_EQ(KVSealPrefix(&fixture.kv, 2), KV_OK);

  for (int i = 0; i < 5; ++i) {
    std::array<uint16_t, 2> tok = {
        static_cast<uint16_t>(300 + i),
        static_cast<uint16_t>(400 + i),
    };
    ASSERT_EQ(KVAppend(&fixture.kv, tok.data(), tok.data()), KV_OK);
  }

  KVAttentionView view{};
  ASSERT_EQ(KVGetAttentionView(&fixture.kv, /*q_pos=*/12, &view), KV_OK);
  EXPECT_EQ(view.q_pos, 12);
  EXPECT_EQ(view.visible_tokens, 5);
  EXPECT_EQ(view.prefix_tokens, 2);
  EXPECT_EQ(view.recent_logical_start, 4);
  EXPECT_EQ(view.recent_size, 3);
  EXPECT_EQ(view.recent_capacity, 3);
  EXPECT_EQ(view.recent_first_slot, 2);
  EXPECT_EQ(view.recent_wrapped, 1);
  EXPECT_EQ(KVRecentFirstPhysicalSlot(&fixture.kv), 2);
  EXPECT_EQ(KVRecentLogicalStart(&fixture.kv), 4);
}

TEST(KVCacheAttentionViewTest, ArgumentValidation) {
  KVAttentionView view{};
  KVCache raw{};

  EXPECT_EQ(KVGetAttentionView(nullptr, 0, &view), KV_ERR_NULL);
  EXPECT_EQ(KVGetAttentionView(&raw, 0, nullptr), KV_ERR_NULL);
  EXPECT_EQ(KVGetAttentionView(&raw, 0, &view), KV_ERR_BAD_STATE);
}
