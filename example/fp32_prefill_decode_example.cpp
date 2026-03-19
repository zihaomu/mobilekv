// ============================================================================
// Example: FP32 Prefill and Decode Mode
//
// This example demonstrates using KVCacheStorage for:
// 1. Prefill phase: Write KV cache for a batch of tokens
// 2. Decode phase: Read and update KV cache token by token
// Layout: [layer][token][head_num][head_dim]
// ============================================================================

#include "mobilekv/kv_cache.h"
#include "mobilekv/kv_cache_debug.h"
#include <iostream>
#include <vector>
#include <cstring>

using namespace mobilekv;

// 配置参数
constexpr uint32_t NUM_LAYERS = 4;
constexpr uint32_t NUM_HEADS = 8;
constexpr uint32_t HEAD_DIM = 128;
constexpr uint32_t MAX_SEQ_LEN = 2048;

int main() {
    std::cout << "=== FP32 Prefill/Decode Example ===" << std::endl;
    std::cout << "Config: " << NUM_LAYERS << " layers, "
              << NUM_HEADS << " heads, "
              << HEAD_DIM << " dim" << std::endl;

    // 创建Builder
    KVCacheStorageBuilder builder;
    builder.config({64, false});

    // 创建FP32模板 - K和V使用相同模板
    auto fp32_templ = std::make_shared<PlainKVTemplate<ScalarType::FP32>>(
        NUM_HEADS, HEAD_DIM, 1, "fp32_kv");
    builder.add_template(fp32_templ);

    // 创建多层存储，每层K和V用同一个模板
    for (uint32_t layer = 0; layer < NUM_LAYERS; ++layer) {
        builder.add_layer(layer, 1, 1, MAX_SEQ_LEN);
    }

    // 构建存储
    auto storage = builder.build();
    if (!storage) {
        std::cerr << "Failed to build storage" << std::endl;
        return 1;
    }

    std::cout << "Storage created with " << storage->total_bytes() << " bytes pre-allocated" << std::endl;

    // =========================================================================
    // Prefill Phase: Write KV cache for a prompt
    // =========================================================================
    std::cout << "\n--- Prefill Phase ---" << std::endl;

    const uint32_t prompt_length = 16;
    std::vector<float> k_data(NUM_HEADS * HEAD_DIM, 0.0f);
    std::vector<float> v_data(NUM_HEADS * HEAD_DIM, 0.0f);

    // 为每个token填充数据
    for (uint32_t token_idx = 0; token_idx < prompt_length; ++token_idx) {
        // 模拟为每个token生成不同的KV值
        for (uint32_t head = 0; head < NUM_HEADS; ++head) {
            for (uint32_t dim = 0; dim < HEAD_DIM; ++dim) {
                k_data[head * HEAD_DIM + dim] = static_cast<float>(token_idx * 1000 + head * 100 + dim);
                v_data[head * HEAD_DIM + dim] = static_cast<float>(token_idx * 2000 + head * 200 + dim);
            }
        }

        // 将数据写入每一层
        for (uint32_t layer = 0; layer < NUM_LAYERS; ++layer) {
            auto& layer_storage = storage->layer(layer);
            auto& k_plane = layer_storage.plane(PlaneKind::K);
            auto& v_plane = layer_storage.plane(PlaneKind::V);

            // 直接访问底层数据指针
            float* k_ptr = static_cast<float*>(k_plane.data());
            float* v_ptr = static_cast<float*>(v_plane.data());

            // 计算当前token的位置
            // layout: [token][head][dim] = token * (num_heads * head_dim) + head * head_dim + dim
            size_t token_offset = token_idx * NUM_HEADS * HEAD_DIM;

            // 写入KV数据
            std::memcpy(k_ptr + token_offset, k_data.data(), k_data.size() * sizeof(float));
            std::memcpy(v_ptr + token_offset, v_data.data(), v_data.size() * sizeof(float));
        }
    }

    // 更新所有层的长度
    storage->append_all(prompt_length);

    std::cout << "Prefill completed: " << prompt_length << " tokens processed" << std::endl;
    std::cout << "Total storage used: " << storage->total_bytes() << " bytes" << std::endl;

    // 验证Prefill结果
    {
        auto& layer0 = storage->layer(0);
        auto& k_plane = layer0.plane(PlaneKind::K);
        float* k_ptr = static_cast<float*>(k_plane.data());

        // 验证token_idx=5的数据
        size_t verify_offset = 5 * NUM_HEADS * HEAD_DIM;  // token 5
        float expected = static_cast<float>(5 * 1000 + 0 * 100 + 0);
        std::cout << "Verify token 5, head 0, dim 0: expected=" << expected
                  << ", actual=" << k_ptr[verify_offset] << std::endl;
    }

    // =========================================================================
    // Decode Phase: Generate tokens one by one
    // =========================================================================
    std::cout << "\n--- Decode Phase ---" << std::endl;

    const uint32_t num_decode_tokens = 4;

    for (uint32_t decode_step = 0; decode_step < num_decode_tokens; ++decode_step) {
        uint32_t new_token_idx = prompt_length + decode_step;

        // 模拟decode: 为新token生成KV
        std::fill(k_data.begin(), k_data.end(), static_cast<float>(new_token_idx * 100));
        std::fill(v_data.begin(), v_data.end(), static_cast<float>(new_token_idx * 200));

        // 追加新token
        storage->append_all(1);

        // 写入新token的KV到每一层
        for (uint32_t layer = 0; layer < NUM_LAYERS; ++layer) {
            auto& layer_storage = storage->layer(layer);
            auto& k_plane = layer_storage.plane(PlaneKind::K);
            auto& v_plane = layer_storage.plane(PlaneKind::V);

            float* k_ptr = static_cast<float*>(k_plane.data());
            float* v_ptr = static_cast<float*>(v_plane.data());

            size_t token_offset = new_token_idx * NUM_HEADS * HEAD_DIM;
            std::memcpy(k_ptr + token_offset, k_data.data(), k_data.size() * sizeof(float));
            std::memcpy(v_ptr + token_offset, v_data.data(), v_data.size() * sizeof(float));
        }

        std::cout << "Decoded token " << new_token_idx << std::endl;
    }

    // 验证最终序列长度
    const auto& final_layer = storage->layer(0);
    const auto& final_k_plane = final_layer.plane(PlaneKind::K);
    std::cout << "\nFinal sequence length: " << final_k_plane.stats().seq_length << std::endl;

    // 验证decode结果
    {
        auto& layer0 = storage->layer(0);
        auto& k_plane = layer0.plane(PlaneKind::K);
        float* k_ptr = static_cast<float*>(k_plane.data());

        // 验证最后一个token
        uint32_t last_token = prompt_length + num_decode_tokens - 1;
        size_t verify_offset = last_token * NUM_HEADS * HEAD_DIM;
        float expected = static_cast<float>(last_token * 100);
        std::cout << "Verify last token " << last_token << ": expected=" << expected
                  << ", actual=" << k_ptr[verify_offset] << std::endl;
    }

    // =========================================================================
    // Random Access: Read specific position
    // =========================================================================
    std::cout << "\n--- Random Access ---" << std::endl;

    for (uint32_t layer = 0; layer < NUM_LAYERS; ++layer) {
        auto& layer_storage = storage->layer(layer);
        auto& k_plane = layer_storage.plane(PlaneKind::K);

        // 使用locate进行随机访问
        LogicalCoord coord(layer, 10, 2, 50);  // seq=10, head=2, dim=50
        PhysicalAddr addr = k_plane.locate(coord);

        if (addr.valid) {
            const uint8_t* data = static_cast<const uint8_t*>(k_plane.data());
            float value;
            std::memcpy(&value, data + addr.byte_offset, sizeof(float));
            std::cout << "Layer " << layer << ", seq=10, head=2, dim=50: " << value << std::endl;
        }
    }

    std::cout << "\n=== Example Completed ===" << std::endl;
    return 0;
}
