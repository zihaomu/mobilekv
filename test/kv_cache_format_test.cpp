#include "mobilekv/kv_cache.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

namespace mobilekv {
namespace test {

TEST(KVCacheFormatTest, ComplexStoragePreservesBF16AndUINT8) {
    std::vector<ComplexLayerConfig> layers = {
        {0, ScalarType::BF16, ScalarType::UINT8},
        {1, ScalarType::UINT8, ScalarType::BF16},
    };

    auto storage = create_complex_storage(layers, 2, 8, 64);
    ASSERT_NE(storage, nullptr);

    const auto& layer0 = storage->layer(0);
    EXPECT_EQ(layer0.plane(PlaneKind::K).templ().config().scalar_type, ScalarType::BF16);
    EXPECT_EQ(layer0.plane(PlaneKind::V).templ().config().scalar_type, ScalarType::UINT8);

    const auto& layer1 = storage->layer(1);
    EXPECT_EQ(layer1.plane(PlaneKind::K).templ().config().scalar_type, ScalarType::UINT8);
    EXPECT_EQ(layer1.plane(PlaneKind::V).templ().config().scalar_type, ScalarType::BF16);
}

TEST(KVCacheFormatTest, SimpleStorageSupportsUINT8) {
    auto storage = create_simple_storage(2, 2, 8, ScalarType::UINT8, 64);
    ASSERT_NE(storage, nullptr);

    const auto& layer0 = storage->layer(0);
    EXPECT_EQ(layer0.plane(PlaneKind::K).templ().config().scalar_type, ScalarType::UINT8);
    EXPECT_EQ(layer0.plane(PlaneKind::V).templ().config().scalar_type, ScalarType::UINT8);
}

TEST(KVCacheFormatTest, AccessorRejectsScalarTypeMismatch) {
    auto storage = create_fp16_storage(1, 2, 8, 64);
    ASSERT_NE(storage, nullptr);

    auto& k_plane = storage->layer(0).plane(PlaneKind::K);
    ASSERT_TRUE(k_plane.resize_seq(4));

    EXPECT_THROW((KVAccessor<float>(k_plane)), std::invalid_argument);
    EXPECT_NO_THROW((KVAccessor<uint16_t>(k_plane)));
}

TEST(KVCacheFormatTest, FormatDescriptorDefaultMatchesTemplateScalarType) {
    auto plain = std::make_shared<PlainKVTemplate<ScalarType::INT8>>(2, 8, 1, "plain");
    auto packed = std::make_shared<PackedKVTemplate<ScalarType::BF16>>(2, 8, 4, 2, "packed");

    EXPECT_EQ(plain->config().format.quant_scheme, QuantScheme::None);
    EXPECT_EQ(plain->config().format.storage_type, ScalarType::INT8);

    EXPECT_EQ(packed->config().format.quant_scheme, QuantScheme::None);
    EXPECT_EQ(packed->config().format.storage_type, ScalarType::BF16);
}

}  // namespace test
}  // namespace mobilekv

