// ============================================================================
// Example: Packed Layout for Efficient Memory Usage
//
// This example demonstrates:
// - Using PackedKVTemplate to pack multiple tokens together
// - Different pack sizes (4, 8, 16)
// - Use cases for memory-constrained environments
// ============================================================================

#include "mobilekv/kv_cache.h"
#include "mobilekv/kv_cache_debug.h"
#include <iostream>
#include <vector>
#include <cstring>
#include <cstdint>

using namespace mobilekv;

// 配置参数
constexpr uint32_t NUM_LAYERS = 2;
constexpr uint32_t NUM_HEADS = 4;
constexpr uint32_t HEAD_DIM = 64;
constexpr uint32_t MAX_SEQ_LEN = 1024;

int main() {
    std::cout << "=== Packed Layout Example ===" << std::endl;
    std::cout << "Config: " << NUM_LAYERS << " layers, "
              << NUM_HEADS << " heads, "
              << HEAD_DIM << " dim" << std::endl;

    // =========================================================================
    // Example 1: Pack size = 4 (pack 4 tokens together)
    // =========================================================================
    std::cout << "\n--- Pack Size 4 ---" << std::endl;

    {
        KVCacheStorageBuilder builder;
        builder.config({64, false});

        // 创建 pack_size=4 的INT8模板
        auto packed_templ = std::make_shared<PackedKVTemplate<ScalarType::INT8>>(
            NUM_HEADS, HEAD_DIM, 4, 1, "packed_int8_x4");
        builder.add_template(packed_templ);

        for (uint32_t layer = 0; layer < NUM_LAYERS; ++layer) {
            builder.add_layer(layer, 1, 1, MAX_SEQ_LEN);
        }

        auto storage = builder.build();

        // 测试定位
        auto& layer0 = storage->layer(0);
        auto& k_plane = layer0.plane(PlaneKind::K);

        k_plane.resize_seq(8);  // 8 tokens = 2 packs

        // 定位 seq=5 在 pack 1 (since 5/4=1), offset inside pack = 1
        LogicalCoord coord(0, 5, 1, 16);  // layer 0, seq 5, head 1, dim 16
        PhysicalAddr addr = k_plane.locate(coord);

        std::cout << "Location for seq=5, head=1, dim=16:" << std::endl;
        std::cout << "  " << addr.to_string() << std::endl;
        std::cout << "  Valid: " << (addr.valid ? "yes" : "no") << std::endl;

        // 测试连续性
        std::cout << "Contiguous check:" << std::endl;
        std::cout << "  seq 0-3 (same pack): "
                  << (packed_templ->can_export_contiguous_span(0, 4) ? "yes" : "no") << std::endl;
        std::cout << "  seq 0-5 (cross pack): "
                  << (packed_templ->can_export_contiguous_span(0, 6) ? "yes" : "no") << std::endl;
    }

    // =========================================================================
    // Example 2: Compare memory usage between plain and packed
    // =========================================================================
    std::cout << "\n--- Memory Comparison: Plain vs Packed ---" << std::endl;

    {
        const uint32_t seq_len = 100;

        // Plain FP16
        auto plain_templ = std::make_shared<PlainKVTemplate<ScalarType::FP16>>(
            NUM_HEADS, HEAD_DIM, 1, "plain_fp16");
        size_t plain_bytes = plain_templ->bytes_for_tokens(seq_len);

        // Packed FP16 with pack_size=8
        auto packed_templ = std::make_shared<PackedKVTemplate<ScalarType::FP16>>(
            NUM_HEADS, HEAD_DIM, 8, 2, "packed_fp16_x8");
        size_t packed_bytes = packed_templ->bytes_for_tokens(seq_len);

        std::cout << "For " << seq_len << " tokens:" << std::endl;
        std::cout << "  Plain FP16:  " << plain_bytes << " bytes" << std::endl;
        std::cout << "  Packed FP16:  " << packed_bytes << " bytes" << std::endl;
        if (plain_bytes >= packed_bytes) {
            std::cout << "  Difference:    " << (plain_bytes - packed_bytes) << " bytes" << std::endl;
        } else {
            std::cout << "  Difference:   -" << (packed_bytes - plain_bytes) << " bytes" << std::endl;
        }
    }

    // =========================================================================
    // Example 3: Real use case - Hybrid approach
    // Use packed layout for K (decode) and plain for V
    // =========================================================================
    std::cout << "\n--- Hybrid: Packed K + Plain V ---" << std::endl;

    {
        KVCacheStorageBuilder builder;
        builder.config({64, false});

        // K: Packed INT8 with pack_size=8 (good for decode phase)
        auto packed_k_templ = std::make_shared<PackedKVTemplate<ScalarType::INT8>>(
            NUM_HEADS, HEAD_DIM, 8, 1, "packed_k_int8");

        // V: Plain FP16 (better for attention computation)
        auto plain_v_templ = std::make_shared<PlainKVTemplate<ScalarType::FP16>>(
            NUM_HEADS, HEAD_DIM, 2, "plain_v_fp16");

        builder.add_template(packed_k_templ);
        builder.add_template(plain_v_templ);

        // Layer 0: Packed K, Plain V
        builder.add_layer(0, 1, 2, MAX_SEQ_LEN);

        auto storage = builder.build();

        auto& layer0 = storage->layer(0);
        auto& k_plane = layer0.plane(PlaneKind::K);
        auto& v_plane = layer0.plane(PlaneKind::V);

        // 写入测试
        const uint32_t test_seq = 32;
        k_plane.resize_seq(test_seq);
        v_plane.resize_seq(test_seq);

        // 写入K (packed)
        int8_t* k_ptr = static_cast<int8_t*>(k_plane.data());
        for (size_t i = 0; i < k_plane.stats().bytes_allocated; ++i) {
            k_ptr[i] = static_cast<int8_t>(i % 128);
        }

        // 写入V (plain)
        uint16_t* v_ptr = static_cast<uint16_t*>(v_plane.data());
        for (size_t i = 0; i < v_plane.stats().bytes_allocated / 2; ++i) {
            v_ptr[i] = static_cast<uint16_t>(i);
        }

        // 验证
        std::cout << "K plane (Packed INT8):" << std::endl;
        std::cout << "  Bytes allocated: " << k_plane.stats().bytes_allocated << std::endl;
        std::cout << "  Element size: " << k_plane.templ().element_size() << " bytes" << std::endl;

        std::cout << "V plane (Plain FP16):" << std::endl;
        std::cout << "  Bytes allocated: " << v_plane.stats().bytes_allocated << std::endl;
        std::cout << "  Element size: " << v_plane.templ().element_size() << " bytes" << std::endl;

        // 定位测试
        auto addr = k_plane.locate(LogicalCoord(0, 10, 2, 32));
        std::cout << "\nK location (seq=10, head=2, dim=32): " << addr.to_string() << std::endl;

        auto v_addr = v_plane.locate(LogicalCoord(0, 10, 2, 32));
        std::cout << "V location (seq=10, head=2, dim=32): " << v_addr.to_string() << std::endl;
    }

    // =========================================================================
    // Example 4: Extreme packing - FP8 simulation
    // =========================================================================
    std::cout << "\n--- Extreme Packing: Simulated FP8 with INT8 ---" << std::endl;

    {
        KVCacheStorageBuilder builder;
        builder.config({64, false});

        // 使用 pack_size=32 模拟更激进的压缩
        auto extreme_packed = std::make_shared<PackedKVTemplate<ScalarType::INT8>>(
            NUM_HEADS, HEAD_DIM, 32, 1, "extreme_packed_int8");
        builder.add_template(extreme_packed);
        builder.add_layer(0, 1, 1, MAX_SEQ_LEN);

        auto storage = builder.build();

        auto& layer0 = storage->layer(0);
        auto& k_plane = layer0.plane(PlaneKind::K);

        const uint32_t seq_len = 64;  // 64 tokens = 2 packs
        k_plane.resize_seq(seq_len);

        // 计算理论内存使用
        size_t packed_bytes = extreme_packed->bytes_for_tokens(seq_len);
        size_t plain_bytes = NUM_HEADS * HEAD_DIM * seq_len * 1;  // 1 byte per INT8

        std::cout << "For " << seq_len << " tokens with pack_size=32:" << std::endl;
        std::cout << "  Theoretical packed bytes: " << packed_bytes << std::endl;
        std::cout << "  Plain INT8 bytes: " << plain_bytes << std::endl;
        std::cout << "  Actual allocated: " << k_plane.stats().bytes_allocated << std::endl;

        // 连续性检查
        std::cout << "\nContiguous access:" << std::endl;
        std::cout << "  32 tokens (1 pack): "
                  << (extreme_packed->can_export_contiguous_span(0, 32) ? "yes" : "no") << std::endl;
        std::cout << "  33 tokens (cross pack): "
                  << (extreme_packed->can_export_contiguous_span(0, 33) ? "yes" : "no") << std::endl;
    }

    std::cout << "\n=== Example Completed ===" << std::endl;
    return 0;
}
