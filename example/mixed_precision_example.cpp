// ============================================================================
// Example: Mixed Precision Customized Layers
//
// This example demonstrates:
// - 10-layer model
// - Layers 0-1: FP32 for both K and V
// - Layers 2-7: K uses INT16, V uses INT8
// - Layers 8-9: FP32 for both K and V
// ============================================================================

#include "mobilekv/kv_cache.h"
#include "mobilekv/kv_cache_debug.h"
#include <iostream>
#include <vector>
#include <cstring>
#include <cstdint>

using namespace mobilekv;

// 配置参数
constexpr uint32_t NUM_LAYERS = 10;
constexpr uint32_t NUM_HEADS = 8;
constexpr uint32_t HEAD_DIM = 128;
constexpr uint32_t MAX_SEQ_LEN = 1024;

int main() {
    std::cout << "=== Mixed Precision Customized Layers Example ===" << std::endl;
    std::cout << "Config: " << NUM_LAYERS << " layers, "
              << NUM_HEADS << " heads, "
              << HEAD_DIM << " dim" << std::endl;

    // 创建Builder
    KVCacheStorageBuilder builder;
    builder.config({64, false});

    // 创建不同精度的模板
    // Template ID 1: FP32 K/V
    auto fp32_templ = std::make_shared<PlainKVTemplate<ScalarType::FP32>>(
        NUM_HEADS, HEAD_DIM, 1, "fp32");
    // Template ID 2: INT16 K
    auto int16_templ = std::make_shared<PlainKVTemplate<ScalarType::INT16>>(
        NUM_HEADS, HEAD_DIM, 2, "int16_k");
    // Template ID 3: INT8 V
    auto int8_templ = std::make_shared<PlainKVTemplate<ScalarType::INT8>>(
        NUM_HEADS, HEAD_DIM, 3, "int8_v");

    builder.add_template(fp32_templ);
    builder.add_template(int16_templ);
    builder.add_template(int8_templ);

    // 创建10层，每层配置不同精度
    // Layers 0-1: FP32 (template 1 for both K and V)
    builder.add_layer(0, 1, 1, MAX_SEQ_LEN);
    builder.add_layer(1, 1, 1, MAX_SEQ_LEN);

    // Layers 2-7: K=INT16 (template 2), V=INT8 (template 3)
    for (uint32_t layer = 2; layer <= 7; ++layer) {
        builder.add_layer(layer, 2, 3, MAX_SEQ_LEN);
    }

    // Layers 8-9: FP32 (template 1 for both K and V)
    builder.add_layer(8, 1, 1, MAX_SEQ_LEN);
    builder.add_layer(9, 1, 1, MAX_SEQ_LEN);

    // 构建存储
    auto storage = builder.build();
    if (!storage) {
        std::cerr << "Failed to build storage" << std::endl;
        return 1;
    }

    std::cout << "Storage created with " << storage->total_bytes() << " bytes pre-allocated" << std::endl;

    // 打印每层的配置
    std::cout << "\nLayer configurations:" << std::endl;
    for (uint32_t layer = 0; layer < NUM_LAYERS; ++layer) {
        const auto& layer_storage = storage->layer(layer);
        const auto& k_plane = layer_storage.plane(PlaneKind::K);
        const auto& v_plane = layer_storage.plane(PlaneKind::V);

        std::cout << "Layer " << layer << ": "
                  << scalar_type_to_string(k_plane.templ().config().scalar_type) << " K, "
                  << scalar_type_to_string(v_plane.templ().config().scalar_type) << " V"
                  << " | K bytes: " << k_plane.stats().bytes_allocated
                  << ", V bytes: " << v_plane.stats().bytes_allocated
                  << std::endl;
    }

    // =========================================================================
    // Write and Read Test
    // =========================================================================
    std::cout << "\n--- Write and Read Test ---" << std::endl;

    const uint32_t test_seq_len = 16;
    storage->reserve_all(test_seq_len);

    // 测试每种精度的写入和读取
    // Layer 0: FP32
    {
        auto& layer = storage->layer(0);
        auto& k_plane = layer.plane(PlaneKind::K);
        auto& v_plane = layer.plane(PlaneKind::V);

        k_plane.resize_seq(test_seq_len);
        v_plane.resize_seq(test_seq_len);

        // 写入FP32数据
        float* k_ptr = static_cast<float*>(k_plane.data());
        float* v_ptr = static_cast<float*>(v_plane.data());

        for (uint32_t i = 0; i < test_seq_len * NUM_HEADS * HEAD_DIM; ++i) {
            k_ptr[i] = static_cast<float>(i);
            v_ptr[i] = static_cast<float>(i * 2);
        }

        // 验证
        float expected = 100.0f;
        size_t offset = 100;  // arbitrary position
        if (k_ptr[offset] == expected) {
            std::cout << "Layer 0 (FP32): Write/Read verification PASSED" << std::endl;
        } else {
            std::cout << "Layer 0 (FP32): Write/Read verification FAILED" << std::endl;
        }
    }

    // Layer 2: INT16
    {
        auto& layer = storage->layer(2);
        auto& k_plane = layer.plane(PlaneKind::K);

        k_plane.resize_seq(test_seq_len);

        // 写入INT16数据
        int16_t* k_ptr = static_cast<int16_t*>(k_plane.data());

        for (uint32_t i = 0; i < test_seq_len * NUM_HEADS * HEAD_DIM; ++i) {
            k_ptr[i] = static_cast<int16_t>(i % 1000);
        }

        // 验证
        int16_t expected = 500;
        size_t offset = 50;
        if (k_ptr[offset] == expected) {
            std::cout << "Layer 2 (INT16 K): Write/Read verification PASSED" << std::endl;
        } else {
            std::cout << "Layer 2 (INT16 K): Write/Read verification FAILED" << std::endl;
        }
    }

    // Layer 2: INT8
    {
        auto& layer = storage->layer(2);
        auto& v_plane = layer.plane(PlaneKind::V);

        v_plane.resize_seq(test_seq_len);

        // 写入INT8数据
        int8_t* v_ptr = static_cast<int8_t*>(v_plane.data());

        for (uint32_t i = 0; i < test_seq_len * NUM_HEADS * HEAD_DIM; ++i) {
            v_ptr[i] = static_cast<int8_t>(i % 128);
        }

        // 验证
        int8_t expected = 64;
        size_t offset = 64;
        if (v_ptr[offset] == expected) {
            std::cout << "Layer 2 (INT8 V): Write/Read verification PASSED" << std::endl;
        } else {
            std::cout << "Layer 2 (INT8 V): Write/Read verification FAILED (got "
                      << static_cast<int>(v_ptr[offset]) << ")" << std::endl;
        }
    }

    // =========================================================================
    // Memory Usage Analysis
    // =========================================================================
    std::cout << "\n--- Memory Usage Analysis ---" << std::endl;

    size_t total_bytes = 0;
    size_t fp32_bytes = 0;
    size_t int16_bytes = 0;
    size_t int8_bytes = 0;

    for (uint32_t layer = 0; layer < NUM_LAYERS; ++layer) {
        const auto& layer_storage = storage->layer(layer);
        const auto& k_plane = layer_storage.plane(PlaneKind::K);
        const auto& v_plane = layer_storage.plane(PlaneKind::V);

        size_t layer_k_bytes = k_plane.stats().bytes_allocated;
        size_t layer_v_bytes = v_plane.stats().bytes_allocated;
        total_bytes += layer_k_bytes + layer_v_bytes;

        auto k_type = k_plane.templ().config().scalar_type;
        auto v_type = v_plane.templ().config().scalar_type;

        if (k_type == ScalarType::FP32) fp32_bytes += layer_k_bytes;
        else if (k_type == ScalarType::INT16) int16_bytes += layer_k_bytes;
        else int8_bytes += layer_k_bytes;

        if (v_type == ScalarType::FP32) fp32_bytes += layer_v_bytes;
        else if (v_type == ScalarType::INT16) int16_bytes += layer_v_bytes;
        else int8_bytes += layer_v_bytes;
    }

    // 计算如果全部用FP32需要多少内存
    size_t all_fp32_bytes = NUM_LAYERS * 2 * test_seq_len * NUM_HEADS * HEAD_DIM * 4;  // 4 bytes per float

    std::cout << "Memory breakdown:" << std::endl;
    std::cout << "  FP32:  " << fp32_bytes << " bytes ("
              << (fp32_bytes * 100 / all_fp32_bytes) << "% of all-FP32)" << std::endl;
    std::cout << "  INT16: " << int16_bytes << " bytes ("
              << (int16_bytes * 100 / all_fp32_bytes) << "% of all-FP32)" << std::endl;
    std::cout << "  INT8:  " << int8_bytes << " bytes ("
              << (int8_bytes * 100 / all_fp32_bytes) << "% of all-FP32)" << std::endl;
    std::cout << "  Total:  " << total_bytes << " bytes ("
              << (total_bytes * 100 / all_fp32_bytes) << "% of all-FP32)" << std::endl;

    std::cout << "\n=== Example Completed ===" << std::endl;
    return 0;
}
