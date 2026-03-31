#ifndef MOBILEKV_KV_CACHE_CONVENIENCE_H_
#define MOBILEKV_KV_CACHE_CONVENIENCE_H_

#include "mobilekv/kv_cache_basic.h"
#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <type_traits>
#include <cctype>

namespace mobilekv {

namespace detail {

inline const char* scalar_type_name(ScalarType t) {
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

inline std::shared_ptr<KVTemplate> make_plain_template_for_type(
    ScalarType type, uint32_t num_heads, uint32_t head_dim,
    TemplateId id, const std::string& name) {
    switch (type) {
        case ScalarType::FP32:
            return std::make_shared<PlainKVTemplate<ScalarType::FP32>>(num_heads, head_dim, id, name);
        case ScalarType::FP16:
            return std::make_shared<PlainKVTemplate<ScalarType::FP16>>(num_heads, head_dim, id, name);
        case ScalarType::BF16:
            return std::make_shared<PlainKVTemplate<ScalarType::BF16>>(num_heads, head_dim, id, name);
        case ScalarType::INT8:
            return std::make_shared<PlainKVTemplate<ScalarType::INT8>>(num_heads, head_dim, id, name);
        case ScalarType::UINT8:
            return std::make_shared<PlainKVTemplate<ScalarType::UINT8>>(num_heads, head_dim, id, name);
        case ScalarType::INT16:
            return std::make_shared<PlainKVTemplate<ScalarType::INT16>>(num_heads, head_dim, id, name);
        case ScalarType::CUSTOM:
            return std::make_shared<PlainKVTemplate<ScalarType::CUSTOM>>(num_heads, head_dim, id, name);
        default:
            throw std::invalid_argument("Unsupported scalar type");
    }
}

template<typename T>
inline bool is_accessor_type_compatible(ScalarType type) {
    if constexpr (std::is_same<T, float>::value) {
        return type == ScalarType::FP32;
    } else if constexpr (std::is_same<T, uint16_t>::value) {
        return type == ScalarType::FP16 || type == ScalarType::BF16;
    } else if constexpr (std::is_same<T, int8_t>::value) {
        return type == ScalarType::INT8;
    } else if constexpr (std::is_same<T, uint8_t>::value) {
        return type == ScalarType::UINT8;
    } else if constexpr (std::is_same<T, int16_t>::value) {
        return type == ScalarType::INT16;
    } else {
        return false;
    }
}

}  // namespace detail

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
        case ScalarType::UINT8:
            templ = std::make_shared<PlainKVTemplate<ScalarType::UINT8>>(
                num_heads, head_dim, templ_id, "simple_uint8");
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
            case ScalarType::UINT8:
                name = "simple_uint8";
                templ = std::make_shared<PlainKVTemplate<ScalarType::UINT8>>(
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

    // 模板缓存：每种精度只创建一次，K/V共享模板池。
    std::unordered_map<ScalarType, TemplateId> type_to_id;
    TemplateId next_id = 1;

    auto get_or_create_template_id = [&](ScalarType type, const char* prefix) -> TemplateId {
        auto it = type_to_id.find(type);
        if (it != type_to_id.end()) {
            return it->second;
        }
        TemplateId id = next_id++;
        std::string name = std::string(prefix) + "_" + detail::scalar_type_name(type);
        builder.add_template(detail::make_plain_template_for_type(type, num_heads, head_dim, id, name));
        type_to_id[type] = id;
        return id;
    };

    // 创建层，启用ring buffer模式
    uint32_t initial_capacity = std::min(uint32_t(1024), max_seq_len);
    for (const auto& config : layers_config) {
        TemplateId k_id = get_or_create_template_id(config.k_type, "k");
        TemplateId v_id = get_or_create_template_id(config.v_type, "v");
        builder.add_layer(config.layer_id, k_id, v_id, initial_capacity, max_seq_len);
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

/**
 * @brief 单个plane初始化配置（用于声明式配置构建）
 */
struct PlaneInitConfig {
    ScalarType scalar_type = ScalarType::FP16;
    bool use_named_type = false;     // true: use named custom type from registry
    std::string named_type;          // valid when use_named_type=true
    uint32_t initial_seq_capacity = 0;
    uint32_t max_seq_capacity = 0;  // 0 means unlimited/non-ring

    void set_builtin_type(ScalarType t) {
        scalar_type = t;
        use_named_type = false;
        named_type.clear();
    }

    void set_named_type(const std::string& type_name) {
        scalar_type = ScalarType::CUSTOM;
        use_named_type = true;
        named_type = type_name;
    }
};

/**
 * @brief 单层初始化配置（K/V可独立设置）
 */
struct LayerInitConfig {
    LayerId layer_id = 0;
    PlaneInitConfig k;
    PlaneInitConfig v;
};

/**
 * @brief 存储初始化配置（支持来自配置文件）
 */
struct StorageInitConfig {
    uint32_t num_heads = 0;
    uint32_t head_dim = 0;
    KVCacheStorageConfig storage_config;
    std::vector<LayerInitConfig> layers;
};

/**
 * @brief 命名自定义类型描述（用于cfg中的k_type/v_type）
 *
 * 例子：
 *   type_name = "int8_pack4"
 *   block_bytes = 4
 *   alignment = 4
 *   dim_pack_factor = 4   // raw head_dim每4个元素映射为1个存储块
 */
struct NamedCustomTypeDesc {
    std::string type_name;
    uint32_t block_bytes = 0;
    uint32_t alignment = 1;
    uint32_t dim_pack_factor = 1;
};

/**
 * @brief cfg初始化路径的自定义类型注册表
 */
class ConfigTypeRegistry {
public:
    bool register_type(const NamedCustomTypeDesc& desc, std::string* error_message = nullptr) {
        auto set_error = [&](const std::string& message) {
            if (error_message) {
                *error_message = message;
            }
        };

        std::string normalized = normalize_name(desc.type_name);
        if (normalized.empty()) {
            set_error("type_name must not be empty");
            return false;
        }
        if (desc.block_bytes == 0) {
            set_error("block_bytes must be > 0 for type '" + normalized + "'");
            return false;
        }
        if (desc.alignment == 0) {
            set_error("alignment must be > 0 for type '" + normalized + "'");
            return false;
        }
        if (desc.dim_pack_factor == 0) {
            set_error("dim_pack_factor must be > 0 for type '" + normalized + "'");
            return false;
        }

        NamedCustomTypeDesc stored = desc;
        stored.type_name = normalized;
        types_[normalized] = std::move(stored);
        return true;
    }

    const NamedCustomTypeDesc* find_type(const std::string& type_name) const {
        std::string normalized = normalize_name(type_name);
        auto it = types_.find(normalized);
        if (it == types_.end()) {
            return nullptr;
        }
        return &it->second;
    }

private:
    static std::string normalize_name(const std::string& input) {
        std::string out = input;
        auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
        out.erase(out.begin(), std::find_if(out.begin(), out.end(), not_space));
        out.erase(std::find_if(out.rbegin(), out.rend(), not_space).base(), out.end());
        std::transform(out.begin(), out.end(), out.begin(),
                       [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
        return out;
    }

    std::unordered_map<std::string, NamedCustomTypeDesc> types_;
};

/**
 * @brief 从声明式配置创建KV存储
 */
std::unique_ptr<KVCacheStorage> create_storage_from_init_config(
    const StorageInitConfig& config
);

/**
 * @brief 从声明式配置创建KV存储（支持命名自定义类型）
 */
std::unique_ptr<KVCacheStorage> create_storage_from_init_config(
    const StorageInitConfig& config,
    const ConfigTypeRegistry* type_registry,
    std::string* error_message
);

/**
 * @brief 从文本配置文件加载StorageInitConfig
 *
 * 支持的行格式（大小写不敏感）：
 *   model num_heads=32 head_dim=128
 *   storage default_alignment=64 thread_safe=false default_max_seq_capacity=0
 *   defaults k_type=fp16 v_type=fp16 initial=512 max=2048
 *   group 2-7 k_type=int8 max_k=1024
 *   override 3 v_type=uint8 max_v=512
 *
 * 类型说明：
 * - 内置类型: fp32/fp16/bf16/int8/uint8/int16
 * - 命名自定义类型: 先在ConfigTypeRegistry注册，再在cfg里直接写类型名
 * - generic 'custom' token 不允许
 */
bool load_storage_init_config_from_file(
    const std::string& path,
    StorageInitConfig& out_config,
    std::string* error_message = nullptr
);

/**
 * @brief 从文本配置字符串加载StorageInitConfig
 */
bool load_storage_init_config_from_string(
    const std::string& text,
    StorageInitConfig& out_config,
    std::string* error_message = nullptr
);

/**
 * @brief 从配置文件直接创建KV存储
 */
std::unique_ptr<KVCacheStorage> create_storage_from_config_file(
    const std::string& path,
    std::string* error_message = nullptr
);

/**
 * @brief 从配置文件直接创建KV存储（支持命名自定义类型）
 */
std::unique_ptr<KVCacheStorage> create_storage_from_config_file(
    const std::string& path,
    const ConfigTypeRegistry* type_registry,
    std::string* error_message = nullptr
);

/**
 * @brief 从配置字符串直接创建KV存储
 */
std::unique_ptr<KVCacheStorage> create_storage_from_config_string(
    const std::string& text,
    std::string* error_message = nullptr
);

/**
 * @brief 从配置字符串直接创建KV存储（支持命名自定义类型）
 */
std::unique_ptr<KVCacheStorage> create_storage_from_config_string(
    const std::string& text,
    const ConfigTypeRegistry* type_registry,
    std::string* error_message = nullptr
);

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
        ScalarType plane_type = plane_.templ().config().scalar_type;
        if (!detail::is_accessor_type_compatible<T>(plane_type)) {
            throw std::invalid_argument(
                std::string("KVAccessor type mismatch with plane scalar type: ") +
                detail::scalar_type_name(plane_type));
        }
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
