#include "mobilekv/kv_cache.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>

namespace mobilekv {
namespace test {

TEST(KVCacheRingBufferTest, LogicalToPhysicalMappingAfterWrap) {
    KVCacheStorageBuilder builder;
    auto templ = std::make_shared<PlainKVTemplate<ScalarType::INT8>>(1, 1, 1, "int8");
    builder.add_template(templ);
    builder.add_layer(0, 1, 1, 4, 4);

    auto storage = builder.build();
    ASSERT_NE(storage, nullptr);

    auto& k_plane = storage->layer(0).plane(PlaneKind::K);
    ASSERT_TRUE(k_plane.append_seq(4));

    int8_t* data = static_cast<int8_t*>(k_plane.data());
    data[0] = 10;
    data[1] = 11;
    data[2] = 12;
    data[3] = 13;

    ASSERT_TRUE(k_plane.append_seq(2));
    data[0] = 20;
    data[1] = 21;

    auto addr0 = k_plane.locate(LogicalCoord(0, 0, 0, 0));
    auto addr1 = k_plane.locate(LogicalCoord(0, 1, 0, 0));
    auto addr2 = k_plane.locate(LogicalCoord(0, 2, 0, 0));
    auto addr3 = k_plane.locate(LogicalCoord(0, 3, 0, 0));

    ASSERT_TRUE(addr0.valid);
    ASSERT_TRUE(addr1.valid);
    ASSERT_TRUE(addr2.valid);
    ASSERT_TRUE(addr3.valid);

    EXPECT_EQ(addr0.byte_offset, 2u);
    EXPECT_EQ(addr1.byte_offset, 3u);
    EXPECT_EQ(addr2.byte_offset, 0u);
    EXPECT_EQ(addr3.byte_offset, 1u);

    EXPECT_EQ(data[addr0.byte_offset], 12);
    EXPECT_EQ(data[addr1.byte_offset], 13);
    EXPECT_EQ(data[addr2.byte_offset], 20);
    EXPECT_EQ(data[addr3.byte_offset], 21);
}

TEST(KVCacheRingBufferTest, AcquireSeqViewHandlesWrap) {
    KVCacheStorageBuilder builder;
    auto templ = std::make_shared<PlainKVTemplate<ScalarType::INT8>>(1, 1, 1, "int8");
    builder.add_template(templ);
    builder.add_layer(0, 1, 1, 4, 4);

    auto storage = builder.build();
    ASSERT_NE(storage, nullptr);

    auto& k_plane = storage->layer(0).plane(PlaneKind::K);
    ASSERT_TRUE(k_plane.append_seq(4));
    ASSERT_TRUE(k_plane.append_seq(2));  // write_head=2, oldest=2

    auto contiguous = k_plane.acquire_seq_view(0, 2, AccessMode::ReadOnly);
    EXPECT_TRUE(contiguous.contiguous);
    EXPECT_EQ(static_cast<size_t>(contiguous.base - static_cast<Byte*>(k_plane.data())), 2u);
    k_plane.release_seq_view(contiguous);

    auto wrapped = k_plane.acquire_seq_view(1, 3, AccessMode::ReadOnly);
    EXPECT_FALSE(wrapped.contiguous);
    EXPECT_EQ(wrapped.base, static_cast<Byte*>(k_plane.data()));
    k_plane.release_seq_view(wrapped);
}

TEST(KVCacheRingBufferTest, LocateRejectsOutOfRangeCoordinates) {
    KVCacheStorageBuilder builder;
    auto templ = std::make_shared<PlainKVTemplate<ScalarType::INT8>>(2, 4, 1, "int8");
    builder.add_template(templ);
    builder.add_layer(0, 1, 1, 8);

    auto storage = builder.build();
    ASSERT_NE(storage, nullptr);

    auto& k_plane = storage->layer(0).plane(PlaneKind::K);
    ASSERT_TRUE(k_plane.resize_seq(3));

    EXPECT_FALSE(k_plane.locate(LogicalCoord(0, 3, 0, 0)).valid);
    EXPECT_FALSE(k_plane.locate(LogicalCoord(0, 0, 2, 0)).valid);
    EXPECT_FALSE(k_plane.locate(LogicalCoord(0, 0, 0, 4)).valid);
    EXPECT_TRUE(k_plane.locate(LogicalCoord(0, 2, 1, 3)).valid);
}

TEST(KVCacheRingBufferTest, DimBlockViewOffsetUsesLocateBegin) {
    KVCacheStorageBuilder builder;
    auto templ = std::make_shared<DimBlockKVTemplate>(2, 1, 4, 1, "dimblock_pack4");
    builder.add_template(templ);
    builder.add_layer(0, 1, 1, 16);

    auto storage = builder.build();
    ASSERT_NE(storage, nullptr);

    auto& k_plane = storage->layer(0).plane(PlaneKind::K);
    ASSERT_TRUE(k_plane.resize_seq(8));

    auto begin_addr = k_plane.locate(LogicalCoord(0, 1, 0, 0));
    ASSERT_TRUE(begin_addr.valid);

    auto view = k_plane.acquire_seq_view(1, 2, AccessMode::ReadOnly);
    ASSERT_TRUE(view.contiguous);

    auto base_offset = static_cast<size_t>(view.base - static_cast<Byte*>(k_plane.data()));
    EXPECT_EQ(base_offset, begin_addr.byte_offset);

    k_plane.release_seq_view(view);
}

TEST(KVCacheRingBufferTest, DefaultMaxSeqCapacityEnablesRingForFourArgAddLayer) {
    KVCacheStorageBuilder builder;
    builder.config({64, false, 8});
    auto templ = std::make_shared<PlainKVTemplate<ScalarType::INT8>>(1, 1, 1, "int8");
    builder.add_template(templ);
    builder.add_layer(0, 1, 1, 4);  // 4参数接口，继承config默认max=8

    auto storage = builder.build();
    ASSERT_NE(storage, nullptr);

    const auto& stats = storage->layer(0).plane(PlaneKind::K).stats();
    EXPECT_TRUE(stats.is_ring_buffer);
    EXPECT_EQ(stats.max_seq_capacity, 8u);
}

TEST(KVCacheRingBufferTest, ExplicitMaxOverridesDefaultMaxSeqCapacity) {
    KVCacheStorageBuilder builder;
    builder.config({64, false, 8});
    auto templ = std::make_shared<PlainKVTemplate<ScalarType::INT8>>(1, 1, 1, "int8");
    builder.add_template(templ);
    builder.add_layer(0, 1, 1, 4, 0);  // 显式关闭ring

    auto storage = builder.build();
    ASSERT_NE(storage, nullptr);

    const auto& stats = storage->layer(0).plane(PlaneKind::K).stats();
    EXPECT_FALSE(stats.is_ring_buffer);
    EXPECT_EQ(stats.max_seq_capacity, 0u);
}

TEST(KVCacheRingBufferTest, BuildFailsWhenInitialGreaterThanMaxCapacity) {
    {
        KVCacheStorageBuilder builder;
        builder.config({64, false, 8});
        auto templ = std::make_shared<PlainKVTemplate<ScalarType::INT8>>(1, 1, 1, "int8");
        builder.add_template(templ);
        builder.add_layer(0, 1, 1, 16);  // default max=8, initial=16

        auto storage = builder.build();
        EXPECT_EQ(storage, nullptr);
    }

    {
        KVCacheStorageBuilder builder;
        auto templ = std::make_shared<PlainKVTemplate<ScalarType::INT8>>(1, 1, 1, "int8");
        builder.add_template(templ);
        builder.add_layer(0, 1, 1, 16, 8);  // 显式max=8, initial=16

        auto storage = builder.build();
        EXPECT_EQ(storage, nullptr);
    }
}

}  // namespace test
}  // namespace mobilekv
