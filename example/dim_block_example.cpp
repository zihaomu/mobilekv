// ============================================================================
// Example: Dim Block Layout (D-last pack style)
//
// This example demonstrates:
// - Using DimBlockKVTemplate for pre-packed last-dimension storage
// - Shape in block-space: [H, S, D_block]
// - Ring-buffer friendly locate/view usage
// ============================================================================

#include "mobilekv/kv_cache.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

using namespace mobilekv;

constexpr uint32_t NUM_LAYERS = 2;
constexpr uint32_t NUM_HEADS = 4;
constexpr uint32_t HEAD_DIM = 64;
constexpr uint32_t PACK_ELEMS = 4;                // int8 pack4
constexpr uint32_t BLOCK_BYTES = 4;               // 4 packed int8 values
constexpr uint32_t DIM_BLOCKS = HEAD_DIM / PACK_ELEMS;
constexpr uint32_t MAX_SEQ_LEN = 1024;

int main() {
    std::cout << "=== Dim Block Layout Example ===" << std::endl;
    std::cout << "Config: layers=" << NUM_LAYERS
              << ", heads=" << NUM_HEADS
              << ", head_dim=" << HEAD_DIM
              << ", dim_blocks=" << DIM_BLOCKS
              << ", block_bytes=" << BLOCK_BYTES << std::endl;

    KVCacheStorageBuilder builder;
    builder.config({64, false, MAX_SEQ_LEN});

    auto k_dimblock = std::make_shared<DimBlockKVTemplate>(
        NUM_HEADS, DIM_BLOCKS, BLOCK_BYTES, 1, "k_dimblock_pack4");
    auto v_dimblock = std::make_shared<DimBlockKVTemplate>(
        NUM_HEADS, DIM_BLOCKS, BLOCK_BYTES, 2, "v_dimblock_pack4");

    builder.add_template(k_dimblock);
    builder.add_template(v_dimblock);

    for (uint32_t layer = 0; layer < NUM_LAYERS; ++layer) {
        builder.add_layer(layer, 1, 2, 256);
    }

    auto storage = builder.build();
    if (!storage) {
        std::cerr << "build failed" << std::endl;
        return 1;
    }

    auto& layer0 = storage->layer(0);
    auto& k_plane = layer0.plane(PlaneKind::K);
    auto& v_plane = layer0.plane(PlaneKind::V);

    // Decode-style append.
    for (uint32_t step = 0; step < 8; ++step) {
        storage->append_all(1);
        uint32_t newest = k_plane.stats().seq_length - 1;

        // Write one block at [seq=newest, head=1, dim_block=2].
        auto k_addr = k_plane.locate(LogicalCoord(0, newest, 1, 2));
        auto v_addr = v_plane.locate(LogicalCoord(0, newest, 1, 2));
        if (!k_addr.valid || !v_addr.valid) {
            std::cerr << "locate failed at step " << step << std::endl;
            return 1;
        }

        auto* k_bytes = static_cast<uint8_t*>(k_plane.data()) + k_addr.byte_offset;
        auto* v_bytes = static_cast<uint8_t*>(v_plane.data()) + v_addr.byte_offset;
        std::fill(k_bytes, k_bytes + BLOCK_BYTES, static_cast<uint8_t>(10 + step));
        std::fill(v_bytes, v_bytes + BLOCK_BYTES, static_cast<uint8_t>(20 + step));
    }

    // Read a contiguous window view.
    uint32_t len = k_plane.stats().seq_length;
    uint32_t win = std::min<uint32_t>(len, 4);
    auto view = k_plane.acquire_seq_view(len - win, win, AccessMode::ReadOnly);
    std::cout << "Window contiguous: " << (view.contiguous ? "yes" : "no")
              << ", bytes=" << view.bytes << std::endl;
    k_plane.release_seq_view(view);

    // Verify one address roundtrip.
    auto addr = k_plane.locate(LogicalCoord(0, len - 1, 1, 2));
    auto* ptr = static_cast<uint8_t*>(k_plane.data()) + addr.byte_offset;
    std::cout << "Last block bytes: "
              << static_cast<int>(ptr[0]) << " "
              << static_cast<int>(ptr[1]) << " "
              << static_cast<int>(ptr[2]) << " "
              << static_cast<int>(ptr[3]) << std::endl;

    std::cout << "=== Example Completed ===" << std::endl;
    return 0;
}
