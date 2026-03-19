#ifndef MOBILEKV_KV_CACHE_DEBUG_H_
#define MOBILEKV_KV_CACHE_DEBUG_H_

#include <iostream>
#include <iomanip>
#include <sstream>
#include "mobilekv/kv_cache_basic.h"

namespace mobilekv {

// ============================================================================
// 调试辅助函数
// ============================================================================

inline std::string scalar_type_to_string(ScalarType t) {
    switch (t) {
        case ScalarType::FP32: return "FP32";
        case ScalarType::FP16: return "FP16";
        case ScalarType::BF16: return "BF16";
        case ScalarType::INT8: return "INT8";
        case ScalarType::UINT8: return "UINT8";
        case ScalarType::INT16: return "INT16";
        case ScalarType::CUSTOM: return "CUSTOM";
        default: return "UNKNOWN";
    }
}

inline std::string plane_kind_to_string(PlaneKind k) {
    return k == PlaneKind::K ? "K" : "V";
}

inline std::string storage_mode_to_string(StorageMode m) {
    switch (m) {
        case StorageMode::Contiguous: return "Contiguous";
        case StorageMode::Blocked: return "Blocked";
        default: return "UNKNOWN";
    }
}

inline std::string access_mode_to_string(AccessMode m) {
    switch (m) {
        case AccessMode::ReadOnly: return "ReadOnly";
        case AccessMode::WriteOnly: return "WriteOnly";
        case AccessMode::ReadWrite: return "ReadWrite";
        default: return "UNKNOWN";
    }
}

inline void print_template_info(const KVTemplate& templ) {
    const auto& cfg = templ.config();
    const auto& shape = templ.shape();

    std::cout << "Template: " << cfg.name << " (id=" << cfg.id << ")\n"
              << "  ScalarType: " << scalar_type_to_string(cfg.scalar_type) << "\n"
              << "  StorageMode: " << storage_mode_to_string(cfg.storage_mode) << "\n"
              << "  Shape: " << shape.num_heads << " heads, " << shape.head_dim << " dim\n"
              << "  Element size: " << templ.element_size() << " bytes\n"
              << "  Alignment: " << cfg.alignment << " bytes\n";
}

inline void print_plane_stats(const KVPlane& plane) {
    const auto& stats = plane.stats();
    std::cout << "Plane " << plane_kind_to_string(plane.kind())
              << " (layer=" << plane.layer_id() << "):\n"
              << "  " << stats.to_string() << "\n";
}

inline void print_layer_info(const LayerStorage& layer) {
    std::cout << "Layer " << layer.layer_id() << ":\n";
    print_plane_stats(layer.plane(PlaneKind::K));
    print_plane_stats(layer.plane(PlaneKind::V));
    std::cout << "  Total bytes: " << layer.total_bytes() << "\n";
}

inline void print_storage_info(const KVCacheStorage& storage) {
    std::cout << "KVCacheStorage:\n"
              << "  Total bytes: " << storage.total_bytes() << "\n";

    // 打印所有层信息
    for (uint32_t i = 0; i < 1000; ++i) {  // 简单遍历，实际应记录层数
        if (storage.has_layer(i)) {
            print_layer_info(storage.layer(i));
        }
    }
}

// 内存打印辅助函数
template<typename T>
void print_memory_dump(const void* ptr, size_t count, const std::string& label = "") {
    const T* data = static_cast<const T*>(ptr);
    std::cout << "Memory dump" << (label.empty() ? "" : ": " + label) << " (" << count << " elements)\n";
    std::cout << std::hex << std::setfill('0');
    for (size_t i = 0; i < count && i < 32; ++i) {
        if (i % 16 == 0) std::cout << "  ";
        std::cout << std::setw(sizeof(T) * 2) << static_cast<uint64_t>(data[i]) << " ";
        if ((i + 1) % 16 == 0) std::cout << "\n";
    }
    std::cout << std::dec << "\n";
}

// 验证数据完整性
inline bool verify_coordinate_access(
    const KVPlane& plane,
    uint32_t seq_idx,
    uint32_t head_idx,
    uint32_t dim_idx
) {
    const auto& stats = plane.stats();
    if (seq_idx >= stats.seq_length) {
        std::cerr << "Error: seq index " << seq_idx << " >= length " << stats.seq_length << "\n";
        return false;
    }

    const auto& shape = plane.templ().shape();
    if (head_idx >= shape.num_heads) {
        std::cerr << "Error: head index " << head_idx << " >= num_heads " << shape.num_heads << "\n";
        return false;
    }

    if (dim_idx >= shape.head_dim) {
        std::cerr << "Error: dim index " << dim_idx << " >= head_dim " << shape.head_dim << "\n";
        return false;
    }

    return true;
}

}  // namespace mobilekv

#endif  // MOBILEKV_KV_CACHE_DEBUG_H_
