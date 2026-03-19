// ============================================================================
// Example: Using Convenience API
//
// This example demonstrates:
// - Simple storage creation (all layers same precision)
// - Complex storage creation (different K/V precision per layer)
// - Using KVAccessor for type-safe data access
// ============================================================================

#include "mobilekv/kv_cache.h"
#include "mobilekv/kv_cache_convenience.h"
#include <iostream>
#include <vector>
#include <cstring>

using namespace mobilekv;

int main() {
    std::cout << "=== Convenience API Example ===" << std::endl;

    // =========================================================================
    // Example 1: Simple Storage - All FP16
    // =========================================================================
    std::cout << "\n--- Example 1: Simple FP16 Storage ---" << std::endl;

    {
        // 12层模型，全部使用FP16
        auto storage = create_fp16_storage(12, 32, 128, 2048);
        if (!storage) {
            std::cerr << "Failed to create storage" << std::endl;
            return 1;
        }

        std::cout << "Created FP16 storage: " << storage->total_bytes() << " bytes" << std::endl;

        // 写入数据
        auto& layer0 = storage->layer(0);
        auto& k_plane = layer0.plane(PlaneKind::K);

        // 使用便捷访问器
        KVAccessor<float> k_accessor(k_plane);
        k_plane.resize_seq(10);

        // 写入数据
        std::vector<float> k_data(32 * 128, 1.5f);
        k_accessor.write_token(5, k_data.data());

        // 读取验证
        std::vector<float> read_data(32 * 128, 0.0f);
        k_accessor.read_token(5, read_data.data());

        if (read_data[0] == 1.5f) {
            std::cout << "Write/Read verification PASSED" << std::endl;
        }
    }

    // =========================================================================
    // Example 2: Simple Storage with Vector Config
    // =========================================================================
    std::cout << "\n--- Example 2: Simple Storage with Different Layers ---" << std::endl;

    {
        // 每层可以使用不同精度
        std::vector<SimpleLayerConfig> layers = {
            {0, ScalarType::FP32},
            {1, ScalarType::FP32},
            {2, ScalarType::FP16},
            {3, ScalarType::FP16},
            {4, ScalarType::INT8},
        };

        auto storage = create_simple_storage(layers, 8, 64, 1024);
        if (!storage) {
            std::cerr << "Failed to create storage" << std::endl;
            return 1;
        }

        std::cout << "Created multi-precision storage: " << storage->total_bytes() << " bytes" << std::endl;

        // 验证各层精度
        for (uint32_t i = 0; i < 5; ++i) {
            const auto& layer = storage->layer(i);
            const auto& k_plane = layer.plane(PlaneKind::K);
            std::cout << "  Layer " << i << ": "
                      << scalar_type_to_string(k_plane.templ().config().scalar_type) << std::endl;
        }
    }

    // =========================================================================
    // Example 3: Complex Storage - K/V Different Precision
    // =========================================================================
    std::cout << "\n--- Example 3: Complex Storage (K/V Different) ---" << std::endl;

    {
        // 10层模型：前2层FP32，中间6层K=INT16/V=INT8，后2层FP32
        std::vector<ComplexLayerConfig> layers = {
            {0, ScalarType::FP32, ScalarType::FP32},
            {1, ScalarType::FP32, ScalarType::FP32},
            {2, ScalarType::INT16, ScalarType::INT8},
            {3, ScalarType::INT16, ScalarType::INT8},
            {4, ScalarType::INT16, ScalarType::INT8},
            {5, ScalarType::INT16, ScalarType::INT8},
            {6, ScalarType::INT16, ScalarType::INT8},
            {7, ScalarType::INT16, ScalarType::INT8},
            {8, ScalarType::FP32, ScalarType::FP32},
            {9, ScalarType::FP32, ScalarType::FP32},
        };

        auto storage = create_complex_storage(layers, 8, 128, 1024);
        if (!storage) {
            std::cerr << "Failed to create storage" << std::endl;
            return 1;
        }

        std::cout << "Created complex storage: " << storage->total_bytes() << " bytes" << std::endl;

        // 验证各层配置
        for (uint32_t i = 0; i < 10; ++i) {
            const auto& layer = storage->layer(i);
            const auto& k_plane = layer.plane(PlaneKind::K);
            const auto& v_plane = layer.plane(PlaneKind::V);

            std::cout << "  Layer " << i << ": K="
                      << scalar_type_to_string(k_plane.templ().config().scalar_type)
                      << ", V=" << scalar_type_to_string(v_plane.templ().config().scalar_type)
                      << std::endl;
        }
    }

    // =========================================================================
    // Example 4: Using Batch Operations
    // =========================================================================
    std::cout << "\n--- Example 4: Batch Operations ---" << std::endl;

    {
        auto storage = create_fp32_storage(1, 16, 64, 512);
        auto& layer = storage->layer(0);
        auto& k_plane = layer.plane(PlaneKind::K);

        k_plane.resize_seq(100);

        // 批量写入10个tokens
        std::vector<float> batch_data(10 * 16 * 64, 3.14f);
        batch_write_tokens(k_plane, 0, 10, batch_data.data());

        // 批量读取
        std::vector<float> read_batch(10 * 16 * 64, 0.0f);
        batch_read_tokens(k_plane, 0, 10, read_batch.data());

        if (read_batch[0] == 3.14f) {
            std::cout << "Batch write/read verification PASSED" << std::endl;
        }
    }

    // =========================================================================
    // Example 5: Using Accessor for Per-Head Access
    // =========================================================================
    std::cout << "\n--- Example 5: Per-Head Access ---" << std::endl;

    {
        auto storage = create_fp16_storage(1, 4, 32, 256);
        auto& layer = storage->layer(0);
        auto& k_plane = layer.plane(PlaneKind::K);
        auto& v_plane = layer.plane(PlaneKind::V);

        k_plane.resize_seq(10);
        v_plane.resize_seq(10);

        // 使用访问器
        KVAccessor<float> k_acc(k_plane);
        KVAccessor<float> v_acc(v_plane);

        // 写入head 2的数据
        std::vector<float> head2_k(32, 100.0f);
        std::vector<float> head2_v(32, 200.0f);

        k_acc.write_token_head(5, 2, head2_k.data());
        v_acc.write_token_head(5, 2, head2_v.data());

        // 读取验证
        std::vector<float> verify_k(32, 0.0f);
        std::vector<float> verify_v(32, 0.0f);

        k_acc.read_token_head(5, 2, verify_k.data());
        v_acc.read_token_head(5, 2, verify_v.data());

        if (verify_k[0] == 100.0f && verify_v[0] == 200.0f) {
            std::cout << "Per-head access verification PASSED" << std::endl;
        }
    }

    std::cout << "\n=== All Examples Completed ===" << std::endl;
    return 0;
}
