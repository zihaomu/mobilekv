#include "mobilekv/kv_cache.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <vector>
#include <cmath>

namespace mobilekv {
namespace test {

// ============================================================================
// 测试 Fixture
// ============================================================================

class KVCacheBasicTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// ============================================================================
// 模板测试
// ============================================================================

TEST_F(KVCacheBasicTest, PlainFP32Template) {
    auto templ = std::make_shared<PlainKVTemplate<ScalarType::FP32>>(8, 128, 1, "plain_fp32");

    EXPECT_EQ(templ->config().id, 1u);
    EXPECT_EQ(templ->config().name, "plain_fp32");
    EXPECT_EQ(templ->config().scalar_type, ScalarType::FP32);
    EXPECT_EQ(templ->shape().num_heads, 8u);
    EXPECT_EQ(templ->shape().head_dim, 128u);
    EXPECT_EQ(templ->element_size(), 4u);

    // 测试字节计算
    size_t bytes = templ->bytes_for_tokens(10);
    EXPECT_EQ(bytes, 10u * 8u * 128u * 4u);

    // 测试定位
    LogicalCoord coord(0, 5, 3, 64);
    PhysicalAddr addr = templ->locate(coord);
    EXPECT_TRUE(addr.valid);
    EXPECT_EQ(addr.byte_size, 4u);
}

TEST_F(KVCacheBasicTest, PlainFP16Template) {
    auto templ = std::make_shared<PlainKVTemplate<ScalarType::FP16>>(4, 64, 2, "plain_fp16");

    EXPECT_EQ(templ->element_size(), 2u);
    EXPECT_EQ(templ->bytes_for_tokens(100), 100u * 4u * 64u * 2u);
}

TEST_F(KVCacheBasicTest, PlainINT8Template) {
    auto templ = std::make_shared<PlainKVTemplate<ScalarType::INT8>>(2, 32, 3, "plain_int8");

    EXPECT_EQ(templ->element_size(), 1u);
    EXPECT_EQ(templ->bytes_for_tokens(50), 50u * 2u * 32u * 1u);
}

TEST_F(KVCacheBasicTest, DimBlockKVTemplate) {
    auto templ = std::make_shared<DimBlockKVTemplate>(8, 32, 16, 10, "dimblock_u128");

    EXPECT_EQ(templ->config().name, "dimblock_u128");
    EXPECT_EQ(templ->shape().num_heads, 8u);
    EXPECT_EQ(templ->shape().head_dim, 32u);
    EXPECT_EQ(templ->element_size(), 16u);

    // 测试dim block模式下的定位
    LogicalCoord coord(0, 10, 2, 16);
    PhysicalAddr addr = templ->locate(coord);
    EXPECT_TRUE(addr.valid);

    // dim block布局按seq连续导出
    EXPECT_TRUE(templ->can_export_contiguous_span(0, 16));
    EXPECT_TRUE(templ->can_export_contiguous_span(7, 4));
}

// ============================================================================
// KVCacheStorage Builder 测试
// ============================================================================

TEST_F(KVCacheBasicTest, BuilderBasic) {
    KVCacheStorageBuilder builder;
    builder.config({64, false});

    // 添加模板
    auto k_templ = std::make_shared<PlainKVTemplate<ScalarType::FP32>>(8, 128, 1, "k_fp32");
    auto v_templ = std::make_shared<PlainKVTemplate<ScalarType::FP32>>(8, 128, 2, "v_fp32");
    builder.add_template(k_templ);
    builder.add_template(v_templ);

    // 添加层 - 初始容量为0
    builder.add_layer(0, 1, 2, 0);
    builder.add_layer(1, 1, 2, 0);

    auto storage = builder.build();
    ASSERT_NE(storage, nullptr);

    EXPECT_EQ(storage->total_bytes(), 0u);  // 还没分配内存

    // 验证层存在
    EXPECT_TRUE(storage->has_layer(0));
    EXPECT_TRUE(storage->has_layer(1));
    EXPECT_FALSE(storage->has_layer(2));

    // 验证模板查找
    EXPECT_NE(storage->find_template(1), nullptr);
    EXPECT_NE(storage->find_template(2), nullptr);
}

TEST_F(KVCacheBasicTest, BuilderWithReserve) {
    KVCacheStorageBuilder builder;
    builder.config({64, false});

    auto templ = std::make_shared<PlainKVTemplate<ScalarType::FP32>>(8, 128, 1, "fp32");
    builder.add_template(templ);
    builder.add_layer(0, 1, 1, 512);

    auto storage = builder.build();
    ASSERT_NE(storage, nullptr);

    // 预分配
    EXPECT_TRUE(storage->reserve_all(512));

    size_t expected_bytes = 2 * 512 * 8 * 128 * 4;  // K + V
    EXPECT_EQ(storage->total_bytes(), expected_bytes);
}

// ============================================================================
// Plane 操作测试
// ============================================================================

TEST_F(KVCacheBasicTest, PlaneAppendAndResize) {
    KVCacheStorageBuilder builder;
    auto templ = std::make_shared<PlainKVTemplate<ScalarType::FP32>>(4, 64, 1, "fp32");
    builder.add_template(templ);
    builder.add_layer(0, 1, 1, 100);

    auto storage = builder.build();
    ASSERT_NE(storage, nullptr);

    auto& layer = storage->layer(0);
    auto& k_plane = layer.plane(PlaneKind::K);

    // 初始状态
    EXPECT_EQ(k_plane.stats().seq_length, 0u);
    EXPECT_EQ(k_plane.stats().seq_capacity, 100u);

    // 追加token
    EXPECT_TRUE(layer.append_seq(10));
    EXPECT_EQ(k_plane.stats().seq_length, 10u);

    EXPECT_TRUE(layer.append_seq(5));
    EXPECT_EQ(k_plane.stats().seq_length, 15u);
}

TEST_F(KVCacheBasicTest, PlaneResize) {
    KVCacheStorageBuilder builder;
    auto templ = std::make_shared<PlainKVTemplate<ScalarType::FP16>>(2, 32, 1, "fp16");
    builder.add_template(templ);
    builder.add_layer(0, 1, 1, 50);

    auto storage = builder.build();
    auto& layer = storage->layer(0);
    auto& k_plane = layer.plane(PlaneKind::K);

    // 直接resize
    EXPECT_TRUE(k_plane.resize_seq(30));
    EXPECT_EQ(k_plane.stats().seq_length, 30u);

    // 扩容
    EXPECT_TRUE(k_plane.resize_seq(60));
    EXPECT_EQ(k_plane.stats().seq_length, 60u);
    EXPECT_GE(k_plane.stats().seq_capacity, 60u);
}

// ============================================================================
// 数据读写测试
// ============================================================================

TEST_F(KVCacheBasicTest, WriteAndReadData) {
    KVCacheStorageBuilder builder;
    auto templ = std::make_shared<PlainKVTemplate<ScalarType::FP32>>(2, 4, 1, "fp32");
    builder.add_template(templ);
    builder.add_layer(0, 1, 1, 10);

    auto storage = builder.build();
    auto& layer = storage->layer(0);
    auto& k_plane = layer.plane(PlaneKind::K);

    // 追加数据
    k_plane.resize_seq(3);

    // 写入数据
    float* data = static_cast<float*>(k_plane.data());
    data[0] = 1.0f;
    data[1] = 2.0f;
    data[2] = 3.0f;

    // 验证读取
    EXPECT_FLOAT_EQ(data[0], 1.0f);
    EXPECT_FLOAT_EQ(data[1], 2.0f);
    EXPECT_FLOAT_EQ(data[2], 3.0f);
}

TEST_F(KVCacheBasicTest, CoordinateLocate) {
    auto templ = std::make_shared<PlainKVTemplate<ScalarType::FP32>>(4, 8, 1, "fp32");
    PhysicalAddr addr = templ->locate(LogicalCoord(0, 2, 1, 4));

    EXPECT_TRUE(addr.valid);

    // 验证偏移计算: seq=2, head=1, dim=4
    // stride = num_heads * head_dim * 4 = 4 * 8 * 4 = 128
    // offset = 2 * 128 + 1 * 8 * 4 + 4 * 4 = 256 + 32 + 16 = 304
    size_t expected_offset = 2 * 4 * 8 * 4 + 1 * 8 * 4 + 4 * 4;
    EXPECT_EQ(addr.byte_offset, expected_offset);
}

// ============================================================================
// AccessView 测试
// ============================================================================

TEST_F(KVCacheBasicTest, AcquireSeqView) {
    KVCacheStorageBuilder builder;
    auto templ = std::make_shared<PlainKVTemplate<ScalarType::FP32>>(2, 4, 1, "fp32");
    builder.add_template(templ);
    builder.add_layer(0, 1, 1, 10);

    auto storage = builder.build();
    auto& layer = storage->layer(0);
    auto& k_plane = layer.plane(PlaneKind::K);

    k_plane.resize_seq(5);

    // 获取视图
    auto view = k_plane.acquire_seq_view(1, 3, AccessMode::ReadWrite);
    EXPECT_TRUE(view.contiguous);
    EXPECT_EQ(view.seq_begin, 1u);
    EXPECT_EQ(view.seq_len, 3u);

    // 写入数据 - 每个token 2*4*4=32 bytes
    // view.base指向seq=1的起始位置
    float* data = reinterpret_cast<float*>(view.base);
    data[0] = 100.0f;  // seq=1, head=0, dim=0
    data[1] = 200.0f;  // seq=1, head=0, dim=1

    // 释放视图
    k_plane.release_seq_view(view);

    // 验证数据 - 2*4=8 floats per token
    // seq=1 starts at float index 8 (32/4)
    float* full_data = static_cast<float*>(k_plane.data());
    EXPECT_FLOAT_EQ(full_data[8], 100.0f);   // seq=1, head=0, dim=0
    EXPECT_FLOAT_EQ(full_data[9], 200.0f);   // seq=1, head=0, dim=1
}

// ============================================================================
// 多层存储测试
// ============================================================================

TEST_F(KVCacheBasicTest, MultiLayer) {
    KVCacheStorageBuilder builder;
    auto templ = std::make_shared<PlainKVTemplate<ScalarType::FP32>>(4, 64, 1, "fp32");
    builder.add_template(templ);

    // 创建10层
    for (uint32_t i = 0; i < 10; ++i) {
        builder.add_layer(i, 1, 1, 100);
    }

    auto storage = builder.build();
    ASSERT_NE(storage, nullptr);

    // 追加所有层
    EXPECT_TRUE(storage->append_all(50));

    // 验证每层
    for (uint32_t i = 0; i < 10; ++i) {
        const auto& layer = storage->layer(i);
        EXPECT_EQ(layer.plane(PlaneKind::K).stats().seq_length, 50u);
        EXPECT_EQ(layer.plane(PlaneKind::V).stats().seq_length, 50u);
    }

    // 清空
    storage->clear_all();
    for (uint32_t i = 0; i < 10; ++i) {
        const auto& layer = storage->layer(i);
        EXPECT_EQ(layer.plane(PlaneKind::K).stats().seq_length, 0u);
    }
}

// ============================================================================
// 混合精度测试
// ============================================================================

TEST_F(KVCacheBasicTest, MixedPrecision) {
    KVCacheStorageBuilder builder;

    // K用FP32, V用FP16
    auto k_templ = std::make_shared<PlainKVTemplate<ScalarType::FP32>>(4, 64, 1, "k_fp32");
    auto v_templ = std::make_shared<PlainKVTemplate<ScalarType::FP16>>(4, 64, 2, "v_fp16");

    builder.add_template(k_templ);
    builder.add_template(v_templ);
    builder.add_layer(0, 1, 2, 0);  // 初始容量为0

    auto storage = builder.build();
    ASSERT_NE(storage, nullptr);

    auto& layer = storage->layer(0);

    // 追加
    layer.append_seq(10);

    // 验证K和V的字节数
    const auto& k_plane = layer.plane(PlaneKind::K);
    const auto& v_plane = layer.plane(PlaneKind::V);

    EXPECT_EQ(k_plane.stats().bytes_allocated, 10u * 4u * 64u * 4u);  // FP32
    EXPECT_EQ(v_plane.stats().bytes_allocated, 10u * 4u * 64u * 2u);  // FP16
}

// ============================================================================
// Dim Block Layout 测试
// ============================================================================

TEST_F(KVCacheBasicTest, DimBlockLayoutReadWrite) {
    KVCacheStorageBuilder builder;

    // 每个dim block是4 bytes, dim_blocks=4 => 等价原始D=16 pack4
    auto templ = std::make_shared<DimBlockKVTemplate>(4, 4, 4, 1, "dimblock_pack4");
    builder.add_template(templ);
    builder.add_layer(0, 1, 1, 16);

    auto storage = builder.build();
    ASSERT_NE(storage, nullptr);

    auto& layer = storage->layer(0);
    auto& k_plane = layer.plane(PlaneKind::K);

    k_plane.resize_seq(8);

    // 写入数据 - 在dim block layout中使用locate定位
    auto addr = k_plane.locate(LogicalCoord(0, 5, 1, 2));
    ASSERT_TRUE(addr.valid);

    uint8_t* data = static_cast<uint8_t*>(k_plane.data());
    data[addr.byte_offset] = 42;

    // 验证
    auto verify_addr = k_plane.locate(LogicalCoord(0, 5, 1, 2));
    EXPECT_EQ(data[verify_addr.byte_offset], 42u);
}

// ============================================================================
// 边界测试
// ============================================================================

TEST_F(KVCacheBasicTest, EdgeCases) {
    // 空构建
    KVCacheStorageBuilder builder_empty;
    auto storage_empty = builder_empty.build();
    EXPECT_EQ(storage_empty, nullptr);

    // 容量为0
    KVCacheStorageBuilder builder;
    auto templ = std::make_shared<PlainKVTemplate<ScalarType::FP32>>(2, 4, 1, "fp32");
    builder.add_template(templ);
    builder.add_layer(0, 1, 1, 0);  // 初始容量为0

    auto storage = builder.build();
    ASSERT_NE(storage, nullptr);

    auto& layer = storage->layer(0);
    EXPECT_EQ(layer.plane(PlaneKind::K).stats().seq_capacity, 0u);

    // 追加后会自动扩容
    layer.append_seq(10);
    EXPECT_GE(layer.plane(PlaneKind::K).stats().seq_capacity, 10u);
}

}  // namespace test
}  // namespace mobilekv

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
