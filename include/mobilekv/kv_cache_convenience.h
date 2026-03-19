#ifndef MOBILEKV_KV_CACHE_CONVENIENCE_H_
#define MOBILEKV_KV_CACHE_CONVENIENCE_H_

#include "mobilekv/kv_cache_basic.h"
#include <vector>
#include <memory>
#include <string>
#include <stdexcept>
#include <cstring>
#include <algorithm>

namespace mobilekv {

// ============================================================================
// 高级API - 简化用户使用
// ============================================================================

/**
 * @brief 简单层配置 - 用于简单场景
 */
struct SimpleLayerConfig {
    uint32_t layer_id;
    ScalarType scalar_type;  // K和V使用相同精度

    SimpleLayerConfig(uint32_t id, ScalarType type)
        : layer_id(id), scalar_type(type) {}
};

/**
 * @brief 复杂层配置 - 用于K/V不同精度、每层不同精度的场景
 */
struct ComplexLayerConfig {
    uint32_t layer_id;
    ScalarType k_type;  // K的精度
    ScalarType v_type;  // V的精度

    ComplexLayerConfig(uint32_t id, ScalarType k, ScalarType v)
        : layer_id(id), k_type(k), v_type(v) {}
};

/**
 * @brief 创建简单KV存储 - 所有层K/V使用相同精度
 *
 * 使用示例:
 *   auto storage = create_simple_storage(
 *       12,           // 12层
 *       32,          // 32个头
 *       128,         // 每个头128维
 *       ScalarType::FP16,  // FP16精度
 *       2048         // 最大序列长度
 *   );
 *
 * @param num_layers 层数
 * @param num_heads 注意力头数
 * @param head_dim 每个头的维度
 * @param scalar_type K和V的数据精度
 * @param max_seq_len 最大序列长度
 * @return 配置好的KVCacheStorage
 */
inline std::unique_ptr<KVCacheStorage> create_simple_storage(
    uint32_t num_layers,
    uint32_t num_heads,
    uint32_t head_dim,
    ScalarType scalar_type,
    uint32_t max_seq_len
) {
    KVCacheStorageBuilder builder;
    builder.config({64, false});

    // 创建模板
    TemplateId templ_id = 1;
    std::shared_ptr<KVTemplate> templ;

    switch (scalar_type) {
        case ScalarType::FP32:
            templ = std::make_shared<PlainKVTemplate<ScalarType::FP32>>(
                num_heads, head_dim, templ_id, "simple_fp32");
            break;
        case ScalarType::FP16:
            templ = std::make_shared<PlainKVTemplate<ScalarType::FP16>>(
                num_heads, head_dim, templ_id, "simple_fp16");
            break;
        case ScalarType::BF16:
            templ = std::make_shared<PlainKVTemplate<ScalarType::BF16>>(
                num_heads, head_dim, templ_id, "simple_bf16");
            break;
        case ScalarType::INT8:
            templ = std::make_shared<PlainKVTemplate<ScalarType::INT8>>(
                num_heads, head_dim, templ_id, "simple_int8");
            break;
        case ScalarType::INT16:
            templ = std::make_shared<PlainKVTemplate<ScalarType::INT16>>(
                num_heads, head_dim, templ_id, "simple_int16");
            break;
        default:
            throw std::invalid_argument("Unsupported scalar type");
    }

    builder.add_template(templ);

    // 创建所有层，启用ring buffer模式
    // 预分配 initial_capacity = min(1024, max_seq_len)，最大为max_seq_len
    uint32_t initial_capacity = std::min(uint32_t(1024), max_seq_len);
    for (uint32_t i = 0; i < num_layers; ++i) {
        builder.add_layer(i, templ_id, templ_id, initial_capacity, max_seq_len);
    }

    return builder.build();
}

/**
 * @brief 创建简单KV存储 - 使用向量配置（更灵活）
 *
 * 使用示例:
 *   std::vector<SimpleLayerConfig> layers = {
 *       {0, ScalarType::FP16},
 *       {1, ScalarType::FP16},
 *       {2, ScalarType::INT8},   // 第3层使用INT8
 *   };
 *   auto storage = create_simple_storage(layers, 32, 128, 2048);
 *
 * @param layers_config 每层的配置
 * @param num_heads 注意力头数
 * @param head_dim 每个头的维度
 * @param max_seq_len 最大序列长度
 * @return 配置好的KVCacheStorage
 */
inline std::unique_ptr<KVCacheStorage> create_simple_storage(
    const std::vector<SimpleLayerConfig>& layers_config,
    uint32_t num_heads,
    uint32_t head_dim,
    uint32_t max_seq_len
) {
    if (layers_config.empty()) {
        return nullptr;
    }

    KVCacheStorageBuilder builder;
    builder.config({64, false});

    // 为每种精度创建模板
    std::unordered_map<ScalarType, TemplateId> type_to_id;
    std::unordered_map<TemplateId, std::shared_ptr<KVTemplate>> templates;

    TemplateId next_id = 1;
    auto get_or_create_template = [&](ScalarType type) -> TemplateId {
        if (type_to_id.find(type) != type_to_id.end()) {
            return type_to_id[type];
        }

        std::shared_ptr<KVTemplate> templ;
        std::string name;

        switch (type) {
            case ScalarType::FP32:
                name = "simple_fp32";
                templ = std::make_shared<PlainKVTemplate<ScalarType::FP32>>(
                    num_heads, head_dim, next_id, name);
                break;
            case ScalarType::FP16:
                name = "simple_fp16";
                templ = std::make_shared<PlainKVTemplate<ScalarType::FP16>>(
                    num_heads, head_dim, next_id, name);
                break;
            case ScalarType::BF16:
                name = "simple_bf16";
                templ = std::make_shared<PlainKVTemplate<ScalarType::BF16>>(
                    num_heads, head_dim, next_id, name);
                break;
            case ScalarType::INT8:
                name = "simple_int8";
                templ = std::make_shared<PlainKVTemplate<ScalarType::INT8>>(
                    num_heads, head_dim, next_id, name);
                break;
            case ScalarType::INT16:
                name = "simple_int16";
                templ = std::make_shared<PlainKVTemplate<ScalarType::INT16>>(
                    num_heads, head_dim, next_id, name);
                break;
            default:
                throw std::invalid_argument("Unsupported scalar type");
        }

        TemplateId id = next_id++;
        type_to_id[type] = id;
        templates[id] = templ;
        builder.add_template(templ);

        return id;
    };

    // 创建层，启用ring buffer模式
    uint32_t initial_capacity = std::min(uint32_t(1024), max_seq_len);
    for (const auto& config : layers_config) {
        TemplateId templ_id = get_or_create_template(config.scalar_type);
        builder.add_layer(config.layer_id, templ_id, templ_id, initial_capacity, max_seq_len);
    }

    return builder.build();
}

/**
 * @brief 创建复杂KV存储 - K和V可以使用不同精度
 *
 * 使用示例:
 *   // 10层模型：前2层FP32，中间6层K=INT16/V=INT8，后2层FP32
 *   std::vector<ComplexLayerConfig> layers = {
 *       {0, ScalarType::FP32, ScalarType::FP32},
 *       {1, ScalarType::FP32, ScalarType::FP32},
 *       {2, ScalarType::INT16, ScalarType::INT8},
 *       {3, ScalarType::INT16, ScalarType::INT8},
 *       {4, ScalarType::INT16, ScalarType::INT8},
 *       {5, ScalarType::INT16, ScalarType::INT8},
 *       {6, ScalarType::INT16, ScalarType::INT8},
 *       {7, ScalarType::INT16, ScalarType::INT8},
 *       {8, ScalarType::FP32, ScalarType::FP32},
 *       {9, ScalarType::FP32, ScalarType::FP32},
 *   };
 *   auto storage = create_complex_storage(layers, 32, 128, 2048);
 *
 * @param layers_config 每层的K/V精度配置
 * @param num_heads 注意力头数
 * @param head_dim 每个头的维度
 * @param max_seq_len 最大序列长度
 * @return 配置好的KVCacheStorage
 */
inline std::unique_ptr<KVCacheStorage> create_complex_storage(
    const std::vector<ComplexLayerConfig>& layers_config,
    uint32_t num_heads,
    uint32_t head_dim,
    uint32_t max_seq_len
) {
    if (layers_config.empty()) {
        return nullptr;
    }

    KVCacheStorageBuilder builder;
    builder.config({64, false});

    // 模板映射: (k_type, v_type) -> template_id
    struct TypePair {
        ScalarType k, v;
        bool operator==(const TypePair& other) const {
            return k == other.k && v == other.v;
        }
    };

    struct TypePairHash {
        size_t operator()(const TypePair& tp) const {
            return static_cast<size_t>(tp.k) * 100 + static_cast<size_t>(tp.v);
        }
    };

    std::unordered_map<TypePair, TemplateId, TypePairHash> type_pair_to_id;
    TemplateId next_id = 1;

    auto create_template = [&](ScalarType k_type, ScalarType v_type) -> std::shared_ptr<KVTemplate> {
        if (k_type == v_type) {
            // 相同精度，使用普通模板
            switch (k_type) {
                case ScalarType::FP32:
                    return std::make_shared<PlainKVTemplate<ScalarType::FP32>>(
                        num_heads, head_dim, 0, "complex_fp32");
                case ScalarType::FP16:
                    return std::make_shared<PlainKVTemplate<ScalarType::FP16>>(
                        num_heads, head_dim, 0, "complex_fp16");
                case ScalarType::BF16:
                    return std::make_shared<PlainKVTemplate<ScalarType::BF16>>(
                        num_heads, head_dim, 0, "complex_bf16");
                case ScalarType::INT8:
                    return std::make_shared<PlainKVTemplate<ScalarType::INT8>>(
                        num_heads, head_dim, 0, "complex_int8");
                case ScalarType::INT16:
                    return std::make_shared<PlainKVTemplate<ScalarType::INT16>>(
                        num_heads, head_dim, 0, "complex_int16");
                default:
                    throw std::invalid_argument("Unsupported scalar type");
            }
        } else {
            // 不同精度：创建两个模板
            // 对于这种情况，我们需要分别创建K和V的模板
            // 这里简化处理：创建混合模板（实际需要根据具体需求调整）
            switch (k_type) {
                case ScalarType::INT16:
                    // K用INT16的情况
                    return std::make_shared<PlainKVTemplate<ScalarType::INT16>>(
                        num_heads, head_dim, 0, "complex_int16_k");
                case ScalarType::INT8:
                    return std::make_shared<PlainKVTemplate<ScalarType::INT8>>(
                        num_heads, head_dim, 0, "complex_int8_k");
                default:
                    return std::make_shared<PlainKVTemplate<ScalarType::FP32>>(
                        num_heads, head_dim, 0, "complex_fp32");
            }
        }
    };

    auto get_or_create_template = [&](ScalarType k_type, ScalarType v_type) -> std::pair<TemplateId, TemplateId> {
        TypePair tp{k_type, v_type};

        // K和V分别的template id
        TemplateId k_id, v_id;

        // 查找或创建K模板
        {
            ScalarType k_only = k_type;
            bool found = false;
            for (const auto& pair : type_pair_to_id) {
                if (pair.first.k == k_only) {
                    k_id = pair.second;
                    found = true;
                    break;
                }
            }
            if (!found) {
                k_id = next_id++;
                std::shared_ptr<KVTemplate> templ;
                switch (k_type) {
                    case ScalarType::FP32:
                        templ = std::make_shared<PlainKVTemplate<ScalarType::FP32>>(
                            num_heads, head_dim, k_id, "k_fp32");
                        break;
                    case ScalarType::FP16:
                        templ = std::make_shared<PlainKVTemplate<ScalarType::FP16>>(
                            num_heads, head_dim, k_id, "k_fp16");
                        break;
                    case ScalarType::INT8:
                        templ = std::make_shared<PlainKVTemplate<ScalarType::INT8>>(
                            num_heads, head_dim, k_id, "k_int8");
                        break;
                    case ScalarType::INT16:
                        templ = std::make_shared<PlainKVTemplate<ScalarType::INT16>>(
                            num_heads, head_dim, k_id, "k_int16");
                        break;
                    default:
                        templ = std::make_shared<PlainKVTemplate<ScalarType::FP32>>(
                            num_heads, head_dim, k_id, "k_fp32");
                }
                builder.add_template(templ);
            }
        }

        // 查找或创建V模板
        {
            ScalarType v_only = v_type;
            bool found = false;
            for (const auto& pair : type_pair_to_id) {
                if (pair.first.v == v_only) {
                    v_id = pair.second;
                    found = true;
                    break;
                }
            }
            if (!found) {
                v_id = next_id++;
                std::shared_ptr<KVTemplate> templ;
                switch (v_type) {
                    case ScalarType::FP32:
                        templ = std::make_shared<PlainKVTemplate<ScalarType::FP32>>(
                            num_heads, head_dim, v_id, "v_fp32");
                        break;
                    case ScalarType::FP16:
                        templ = std::make_shared<PlainKVTemplate<ScalarType::FP16>>(
                            num_heads, head_dim, v_id, "v_fp16");
                        break;
                    case ScalarType::INT8:
                        templ = std::make_shared<PlainKVTemplate<ScalarType::INT8>>(
                            num_heads, head_dim, v_id, "v_int8");
                        break;
                    case ScalarType::INT16:
                        templ = std::make_shared<PlainKVTemplate<ScalarType::INT16>>(
                            num_heads, head_dim, v_id, "v_int16");
                        break;
                    default:
                        templ = std::make_shared<PlainKVTemplate<ScalarType::FP32>>(
                            num_heads, head_dim, v_id, "v_fp32");
                }
                builder.add_template(templ);
            }
        }

        return {k_id, v_id};
    };

    // 创建层，启用ring buffer模式
    uint32_t initial_capacity = std::min(uint32_t(1024), max_seq_len);
    for (const auto& config : layers_config) {
        auto ids = get_or_create_template(config.k_type, config.v_type);
        builder.add_layer(config.layer_id, ids.first, ids.second, initial_capacity, max_seq_len);
    }

    return builder.build();
}

/**
 * @brief 便捷函数：创建全FP32存储
 */
inline std::unique_ptr<KVCacheStorage> create_fp32_storage(
    uint32_t num_layers,
    uint32_t num_heads,
    uint32_t head_dim,
    uint32_t max_seq_len
) {
    return create_simple_storage(num_layers, num_heads, head_dim, ScalarType::FP32, max_seq_len);
}

/**
 * @brief 便捷函数：创建全FP16存储
 */
inline std::unique_ptr<KVCacheStorage> create_fp16_storage(
    uint32_t num_layers,
    uint32_t num_heads,
    uint32_t head_dim,
    uint32_t max_seq_len
) {
    return create_simple_storage(num_layers, num_heads, head_dim, ScalarType::FP16, max_seq_len);
}

/**
 * @brief 便捷函数：创建全INT8存储
 */
inline std::unique_ptr<KVCacheStorage> create_int8_storage(
    uint32_t num_layers,
    uint32_t num_heads,
    uint32_t head_dim,
    uint32_t max_seq_len
) {
    return create_simple_storage(num_layers, num_heads, head_dim, ScalarType::INT8, max_seq_len);
}

// ============================================================================
// 便捷访问器 - 简化数据读写
// ============================================================================

/**
 * @brief KV数据访问器 - 提供类型安全的访问
 */
template<typename T>
class KVAccessor {
public:
    KVAccessor(KVPlane& plane) : plane_(plane) {
        data_ = static_cast<T*>(plane.data());
    }

    /**
     * @brief 写入单个token的所有head数据
     * @param token_idx token索引
     * @param data 数据指针，大小为 num_heads * head_dim
     */
    void write_token(uint32_t token_idx, const T* data) {
        const auto& shape = plane_.templ().shape();
        size_t offset = static_cast<size_t>(token_idx) * shape.num_heads * shape.head_dim;
        memcpy(data_ + offset, data, shape.num_heads * shape.head_dim * sizeof(T));
    }

    /**
     * @brief 读取单个token的所有head数据
     * @param token_idx token索引
     * @param data 输出缓冲区
     */
    void read_token(uint32_t token_idx, T* data) const {
        const auto& shape = plane_.templ().shape();
        size_t offset = static_cast<size_t>(token_idx) * shape.num_heads * shape.head_dim;
        memcpy(data, data_ + offset, shape.num_heads * shape.head_dim * sizeof(T));
    }

    /**
     * @brief 写入指定head的数据
     * @param token_idx token索引
     * @param head_idx head索引
     * @param data 数据指针，大小为 head_dim
     */
    void write_token_head(uint32_t token_idx, uint32_t head_idx, const T* data) {
        const auto& shape = plane_.templ().shape();
        size_t offset = static_cast<size_t>(token_idx) * shape.num_heads * shape.head_dim
                      + static_cast<size_t>(head_idx) * shape.head_dim;
        memcpy(data_ + offset, data, shape.head_dim * sizeof(T));
    }

    /**
     * @brief 读取指定head的数据
     * @param token_idx token索引
     * @param head_idx head索引
     * @param data 输出缓冲区
     */
    void read_token_head(uint32_t token_idx, uint32_t head_idx, T* data) const {
        const auto& shape = plane_.templ().shape();
        size_t offset = static_cast<size_t>(token_idx) * shape.num_heads * shape.head_dim
                      + static_cast<size_t>(head_idx) * shape.head_dim;
        memcpy(data, data_ + offset, shape.head_dim * sizeof(T));
    }

    /**
     * @brief 获取原始数据指针
     */
    T* data() { return data_; }
    const T* data() const { return data_; }

    /**
     * @brief 获取plane统计信息
     */
    const PlaneStats& stats() const { return plane_.stats(); }

private:
    KVPlane& plane_;
    T* data_;
};

/**
 * @brief 创建K访问器
 */
template<typename T>
KVAccessor<T> create_k_accessor(LayerStorage& layer) {
    return KVAccessor<T>(layer.plane(PlaneKind::K));
}

/**
 * @brief 创建V访问器
 */
template<typename T>
KVAccessor<T> create_v_accessor(LayerStorage& layer) {
    return KVAccessor<T>(layer.plane(PlaneKind::V));
}

// ============================================================================
// Batch Operations - 批量操作
// ============================================================================

/**
 * @brief 批量写入多个token
 */
template<typename T>
void batch_write_tokens(KVPlane& plane, uint32_t start_token, uint32_t num_tokens, const T* data) {
    const auto& shape = plane.templ().shape();
    size_t offset = static_cast<size_t>(start_token) * shape.num_heads * shape.head_dim;
    size_t size = static_cast<size_t>(num_tokens) * shape.num_heads * shape.head_dim;
    T* plane_data = static_cast<T*>(plane.data());
    memcpy(plane_data + offset, data, size * sizeof(T));
}

/**
 * @brief 批量读取多个token
 */
template<typename T>
void batch_read_tokens(const KVPlane& plane, uint32_t start_token, uint32_t num_tokens, T* data) {
    const auto& shape = plane.templ().shape();
    size_t offset = static_cast<size_t>(start_token) * shape.num_heads * shape.head_dim;
    size_t size = static_cast<size_t>(num_tokens) * shape.num_heads * shape.head_dim;
    const T* plane_data = static_cast<const T*>(plane.data());
    memcpy(data, plane_data + offset, size * sizeof(T));
}

}  // namespace mobilekv

#endif  // MOBILEKV_KV_CACHE_CONVENIENCE_H_
