#include "mobilekv/kv_cache.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
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
    auto dimblock = std::make_shared<DimBlockKVTemplate>(2, 2, 8, 2, "dimblock");

    EXPECT_EQ(plain->config().format.quant_scheme, QuantScheme::None);
    EXPECT_EQ(plain->config().format.storage_type, ScalarType::INT8);

    EXPECT_EQ(dimblock->config().format.quant_scheme, QuantScheme::None);
    EXPECT_EQ(dimblock->config().format.storage_type, ScalarType::CUSTOM);
}

TEST(KVCacheFormatTest, BuilderSupportsIndependentKVCapacityConfig) {
    KVCacheStorageBuilder builder;

    auto k_templ = std::make_shared<PlainKVTemplate<ScalarType::FP16>>(1, 1, 1, "k_fp16");
    auto v_templ = std::make_shared<PlainKVTemplate<ScalarType::INT8>>(1, 1, 2, "v_int8");
    builder.add_template(k_templ);
    builder.add_template(v_templ);

    builder.add_layer(
        0,   // layer
        1,   // k template
        2,   // v template
        2,   // k initial
        2,   // v initial
        4,   // k max
        8    // v max
    );

    auto storage = builder.build();
    ASSERT_NE(storage, nullptr);

    auto& layer0 = storage->layer(0);
    EXPECT_TRUE(layer0.append_seq(10));

    const auto& k_stats = layer0.plane(PlaneKind::K).stats();
    const auto& v_stats = layer0.plane(PlaneKind::V).stats();

    EXPECT_TRUE(k_stats.is_ring_buffer);
    EXPECT_TRUE(v_stats.is_ring_buffer);
    EXPECT_EQ(k_stats.max_seq_capacity, 4u);
    EXPECT_EQ(v_stats.max_seq_capacity, 8u);
    EXPECT_EQ(k_stats.seq_length, 4u);
    EXPECT_EQ(v_stats.seq_length, 8u);
}

TEST(KVCacheFormatTest, CreateStorageFromInitConfigSupportsIndependentKV) {
    StorageInitConfig cfg;
    cfg.num_heads = 1;
    cfg.head_dim = 1;
    cfg.storage_config = {64, false, 0};

    LayerInitConfig layer0;
    layer0.layer_id = 0;
    layer0.k.scalar_type = ScalarType::INT8;
    layer0.k.initial_seq_capacity = 2;
    layer0.k.max_seq_capacity = 4;
    layer0.v.scalar_type = ScalarType::FP16;
    layer0.v.initial_seq_capacity = 2;
    layer0.v.max_seq_capacity = 8;

    cfg.layers = {layer0};

    auto storage = create_storage_from_init_config(cfg);
    ASSERT_NE(storage, nullptr);

    auto& layer = storage->layer(0);
    EXPECT_EQ(layer.plane(PlaneKind::K).templ().config().scalar_type, ScalarType::INT8);
    EXPECT_EQ(layer.plane(PlaneKind::V).templ().config().scalar_type, ScalarType::FP16);

    EXPECT_TRUE(layer.append_seq(6));
    EXPECT_EQ(layer.plane(PlaneKind::K).stats().seq_length, 4u);
    EXPECT_EQ(layer.plane(PlaneKind::V).stats().seq_length, 6u);
}

TEST(KVCacheFormatTest, CreateStorageFromInitConfigSupportsNamedCustomType) {
    StorageInitConfig cfg;
    cfg.num_heads = 2;
    cfg.head_dim = 8;
    cfg.storage_config = {64, false, 0};

    LayerInitConfig layer0;
    layer0.layer_id = 0;
    layer0.k.set_named_type("int8_pack4");
    layer0.k.initial_seq_capacity = 2;
    layer0.k.max_seq_capacity = 8;
    layer0.v.set_builtin_type(ScalarType::FP16);
    layer0.v.initial_seq_capacity = 2;
    layer0.v.max_seq_capacity = 8;

    cfg.layers = {layer0};

    ConfigTypeRegistry type_registry;
    ASSERT_TRUE(type_registry.register_type({"int8_pack4", 4, 4, 4}));

    auto storage = create_storage_from_init_config(cfg, &type_registry, nullptr);
    ASSERT_NE(storage, nullptr);

    const auto& k_templ = storage->layer(0).plane(PlaneKind::K).templ();
    const auto& v_templ = storage->layer(0).plane(PlaneKind::V).templ();

    EXPECT_EQ(k_templ.config().scalar_type, ScalarType::CUSTOM);
    EXPECT_EQ(v_templ.config().scalar_type, ScalarType::FP16);
    EXPECT_EQ(k_templ.element_size(), 4u);
    EXPECT_EQ(k_templ.shape().head_dim, 2u);
    EXPECT_EQ(v_templ.element_size(), 2u);
}

TEST(KVCacheFormatTest, LoadStorageInitConfigFromFileAndBuild) {
    const std::filesystem::path cfg_path =
        std::filesystem::temp_directory_path() / "mobilekv_init_config_test.cfg";

    {
        std::ofstream out(cfg_path);
        ASSERT_TRUE(out.is_open());
        out << "model num_heads=2 head_dim=4\n";
        out << "storage default_alignment=64 thread_safe=false default_max_seq_capacity=0\n";
        out << "defaults k_type=fp16 v_type=fp16 initial=2 max=16\n";
        out << "layer 0-1 k_type=int8 v_type=fp16 max_k=8 max_v=16\n";
        out << "layer 1 v_type=uint8 initial_v=4 max_v=12\n";
    }

    StorageInitConfig parsed;
    std::string error;
    ASSERT_TRUE(load_storage_init_config_from_file(cfg_path.string(), parsed, &error)) << error;

    ASSERT_EQ(parsed.layers.size(), 2u);
    EXPECT_EQ(parsed.num_heads, 2u);
    EXPECT_EQ(parsed.head_dim, 4u);

    EXPECT_EQ(parsed.layers[0].layer_id, 0u);
    EXPECT_EQ(parsed.layers[0].k.scalar_type, ScalarType::INT8);
    EXPECT_EQ(parsed.layers[0].v.scalar_type, ScalarType::FP16);
    EXPECT_EQ(parsed.layers[0].k.max_seq_capacity, 8u);
    EXPECT_EQ(parsed.layers[0].v.max_seq_capacity, 16u);

    EXPECT_EQ(parsed.layers[1].layer_id, 1u);
    EXPECT_EQ(parsed.layers[1].k.scalar_type, ScalarType::INT8);
    EXPECT_EQ(parsed.layers[1].v.scalar_type, ScalarType::UINT8);
    EXPECT_EQ(parsed.layers[1].k.initial_seq_capacity, 2u);
    EXPECT_EQ(parsed.layers[1].v.initial_seq_capacity, 4u);
    EXPECT_EQ(parsed.layers[1].k.max_seq_capacity, 8u);
    EXPECT_EQ(parsed.layers[1].v.max_seq_capacity, 12u);

    auto storage = create_storage_from_init_config(parsed);
    ASSERT_NE(storage, nullptr);
    EXPECT_EQ(storage->layer(1).plane(PlaneKind::V).templ().config().scalar_type, ScalarType::UINT8);

    std::filesystem::remove(cfg_path);
}

TEST(KVCacheFormatTest, LoadStorageInitConfigFromFileSupportsNamedCustomType) {
    const std::filesystem::path cfg_path =
        std::filesystem::temp_directory_path() / "mobilekv_init_config_custom.cfg";

    {
        std::ofstream out(cfg_path);
        ASSERT_TRUE(out.is_open());
        out << "model num_heads=2 head_dim=8\n";
        out << "defaults k_type=fp16 v_type=fp16 initial=2 max=16\n";
        out << "layer 0 k_type=int8_pack4 v_type=fp16\n";
        out << "layer 1 k_type=fp16 v_type=int8_pack4\n";
    }

    StorageInitConfig parsed;
    std::string error;
    ASSERT_TRUE(load_storage_init_config_from_file(cfg_path.string(), parsed, &error)) << error;
    ASSERT_EQ(parsed.layers.size(), 2u);

    ConfigTypeRegistry type_registry;
    ASSERT_TRUE(type_registry.register_type({"int8_pack4", 4, 4, 4}));

    auto storage = create_storage_from_init_config(parsed, &type_registry, &error);
    ASSERT_NE(storage, nullptr);

    const auto& layer0 = storage->layer(0);
    const auto& layer1 = storage->layer(1);
    EXPECT_EQ(layer0.plane(PlaneKind::K).templ().config().scalar_type, ScalarType::CUSTOM);
    EXPECT_EQ(layer0.plane(PlaneKind::V).templ().config().scalar_type, ScalarType::FP16);
    EXPECT_EQ(layer1.plane(PlaneKind::K).templ().config().scalar_type, ScalarType::FP16);
    EXPECT_EQ(layer1.plane(PlaneKind::V).templ().config().scalar_type, ScalarType::CUSTOM);

    std::filesystem::remove(cfg_path);
}

TEST(KVCacheFormatTest, CreateStorageFromInitConfigNamedCustomTypeNeedsRegistry) {
    StorageInitConfig cfg;
    cfg.num_heads = 2;
    cfg.head_dim = 8;
    cfg.storage_config = {64, false, 0};

    LayerInitConfig layer0;
    layer0.layer_id = 0;
    layer0.k.set_named_type("int8_pack4");
    layer0.k.initial_seq_capacity = 2;
    layer0.k.max_seq_capacity = 8;
    layer0.v.set_builtin_type(ScalarType::FP16);
    layer0.v.initial_seq_capacity = 2;
    layer0.v.max_seq_capacity = 8;
    cfg.layers = {layer0};

    std::string error;
    auto storage = create_storage_from_init_config(cfg, nullptr, &error);
    EXPECT_EQ(storage, nullptr);
    EXPECT_NE(error.find("no ConfigTypeRegistry"), std::string::npos);
}

TEST(KVCacheFormatTest, LoadStorageInitConfigRejectsInvalidDirective) {
    const std::filesystem::path cfg_path =
        std::filesystem::temp_directory_path() / "mobilekv_init_config_invalid.cfg";

    {
        std::ofstream out(cfg_path);
        ASSERT_TRUE(out.is_open());
        out << "model num_heads=2 head_dim=4\n";
        out << "bad_directive foo=bar\n";
    }

    StorageInitConfig parsed;
    std::string error;
    EXPECT_FALSE(load_storage_init_config_from_file(cfg_path.string(), parsed, &error));
    EXPECT_NE(error.find("unknown directive"), std::string::npos);

    std::filesystem::remove(cfg_path);
}

TEST(KVCacheFormatTest, LoadStorageInitConfigRejectsGenericCustomTypeToken) {
    const std::filesystem::path cfg_path =
        std::filesystem::temp_directory_path() / "mobilekv_init_config_generic_custom.cfg";

    {
        std::ofstream out(cfg_path);
        ASSERT_TRUE(out.is_open());
        out << "model num_heads=2 head_dim=8\n";
        out << "defaults k_type=custom v_type=fp16 initial=2 max=16\n";
        out << "layer 0 k_type=fp16 v_type=fp16\n";
    }

    StorageInitConfig parsed;
    std::string error;
    EXPECT_FALSE(load_storage_init_config_from_file(cfg_path.string(), parsed, &error));
    EXPECT_NE(error.find("generic 'custom' is not allowed"), std::string::npos);

    std::filesystem::remove(cfg_path);
}

TEST(KVCacheFormatTest, LoadStorageInitConfigSupportsGroupsAndOverrides) {
    const std::filesystem::path cfg_path =
        std::filesystem::temp_directory_path() / "mobilekv_rule_groups_overrides.cfg";

    {
        std::ofstream out(cfg_path);
        ASSERT_TRUE(out.is_open());
        out << "model num_heads=2 head_dim=4\n";
        out << "storage default_alignment=64 thread_safe=false default_max_seq_capacity=6\n";
        out << "defaults k_type=fp16 v_type=fp16 initial=2\n";
        out << "group 0-3 k_type=int8 max_k=8\n";
        out << "override 1 v_type=uint8 max_v=3\n";
        out << "override 2 v_type=uint8 max_v=5\n";
    }

    StorageInitConfig parsed;
    std::string error;
    ASSERT_TRUE(load_storage_init_config_from_file(cfg_path.string(), parsed, &error)) << error;
    ASSERT_EQ(parsed.layers.size(), 4u);

    EXPECT_EQ(parsed.layers[0].layer_id, 0u);
    EXPECT_EQ(parsed.layers[0].k.scalar_type, ScalarType::INT8);
    EXPECT_EQ(parsed.layers[0].v.scalar_type, ScalarType::FP16);
    EXPECT_EQ(parsed.layers[0].k.max_seq_capacity, 8u);
    EXPECT_EQ(parsed.layers[0].v.max_seq_capacity, 6u);

    EXPECT_EQ(parsed.layers[1].layer_id, 1u);
    EXPECT_EQ(parsed.layers[1].k.scalar_type, ScalarType::INT8);
    EXPECT_EQ(parsed.layers[1].v.scalar_type, ScalarType::UINT8);
    EXPECT_EQ(parsed.layers[1].k.max_seq_capacity, 8u);
    EXPECT_EQ(parsed.layers[1].v.max_seq_capacity, 3u);

    EXPECT_EQ(parsed.layers[2].layer_id, 2u);
    EXPECT_EQ(parsed.layers[2].k.scalar_type, ScalarType::INT8);
    EXPECT_EQ(parsed.layers[2].v.scalar_type, ScalarType::UINT8);
    EXPECT_EQ(parsed.layers[2].k.max_seq_capacity, 8u);
    EXPECT_EQ(parsed.layers[2].v.max_seq_capacity, 5u);

    EXPECT_EQ(parsed.layers[3].layer_id, 3u);
    EXPECT_EQ(parsed.layers[3].k.scalar_type, ScalarType::INT8);
    EXPECT_EQ(parsed.layers[3].v.scalar_type, ScalarType::FP16);
    EXPECT_EQ(parsed.layers[3].k.max_seq_capacity, 8u);
    EXPECT_EQ(parsed.layers[3].v.max_seq_capacity, 6u);

    std::filesystem::remove(cfg_path);
}

TEST(KVCacheFormatTest, ConfigFileRejectsProfileDirectives) {
    const std::filesystem::path cfg_path =
        std::filesystem::temp_directory_path() / "mobilekv_profile_reject.cfg";

    {
        std::ofstream out(cfg_path);
        ASSERT_TRUE(out.is_open());
        out << "model num_heads=1 head_dim=1\n";
        out << "defaults k_type=fp16 v_type=fp16 initial=1 max=8\n";
        out << "profile low override 0 max_k=4 max_v=4\n";
    }

    std::string error;
    auto storage = create_storage_from_config_file(cfg_path.string(), &error);
    EXPECT_EQ(storage, nullptr);
    EXPECT_NE(error.find("unknown directive"), std::string::npos);

    std::filesystem::remove(cfg_path);
}

}  // namespace test
}  // namespace mobilekv
