#ifndef MOBILEKV_KV_CACHE_BASIC_H_
#define MOBILEKV_KV_CACHE_BASIC_H_

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <algorithm>

namespace mobilekv {

class KVTemplate;

// ============================================================================
// 基础类型定义
// ============================================================================

enum class ScalarType {
    FP32,
    FP16,
    BF16,
    INT8,
    UINT8,
    INT16,
    CUSTOM
};

enum class PlaneKind {
    K,
    V
};

enum class StorageMode {
    Contiguous,   // 连续存储
    Blocked       // 分块存储
};

enum class AccessMode {
    ReadOnly,
    WriteOnly,
    ReadWrite
};

enum class MemoryDomain {
    HostRAM,
    DiskMapped
};

enum class QuantScheme {
    None,
    PerTensorAffine,
    PerChannelSymmetric,
    PerGroupAffine
};

inline size_t scalar_type_size(ScalarType t) {
    switch (t) {
        case ScalarType::FP32: return 4;
        case ScalarType::FP16: return 2;
        case ScalarType::BF16: return 2;
        case ScalarType::INT8: return 1;
        case ScalarType::UINT8: return 1;
        case ScalarType::INT16: return 2;
        default: return 1;
    }
}

struct FormatDescriptor {
    QuantScheme quant_scheme = QuantScheme::None;
    uint32_t group_size = 0;
    ScalarType storage_type = ScalarType::FP16;
    ScalarType scale_type = ScalarType::FP16;
    bool has_zero_point = false;
    ScalarType zero_point_type = ScalarType::INT8;
};

// 基础类型别名
using LayerId = uint32_t;
using TemplateId = uint32_t;
using TokenIndex = uint32_t;
using HeadIndex = uint32_t;
using BlockIndex = uint32_t;
using Byte = uint8_t;

// ============================================================================
// 逻辑坐标与物理地址
// ============================================================================

struct LogicalCoord {
    uint32_t layer = 0;
    uint32_t seq = 0;
    uint32_t head = 0;
    uint32_t dim = 0;

    LogicalCoord() = default;
    LogicalCoord(uint32_t l, uint32_t s, uint32_t h, uint32_t d)
        : layer(l), seq(s), head(h), dim(d) {}

    std::string to_string() const {
        std::ostringstream oss;
        oss << "LogicalCoord(layer=" << layer
            << ", seq=" << seq
            << ", head=" << head
            << ", dim=" << dim << ")";
        return oss.str();
    }
};

struct PhysicalAddr {
    size_t byte_offset = 0;
    size_t byte_size = 0;
    uint32_t loop0 = 0;
    uint32_t loop1 = 0;
    uint32_t inner_offset = 0;
    bool valid = false;

    PhysicalAddr() = default;

    std::string to_string() const {
        std::ostringstream oss;
        oss << "PhysicalAddr(offset=" << byte_offset
            << ", size=" << byte_size
            << ", loop0=" << loop0
            << ", loop1=" << loop1
            << ", inner=" << inner_offset
            << ", valid=" << (valid ? "true" : "false") << ")";
        return oss.str();
    }
};

// ============================================================================
// 模板形状与配置
// ============================================================================

struct TemplateShape {
    uint32_t num_heads = 0;
    uint32_t head_dim = 0;

    // 可选：用户自己的两个外层循环解释
    uint32_t loop0_extent = 0;
    uint32_t loop1_extent = 0;

    // inner 单元大小（字节）
    uint32_t inner_bytes = 0;

    TemplateShape() = default;
    TemplateShape(uint32_t heads, uint32_t dim)
        : num_heads(heads), head_dim(dim), inner_bytes(dim) {}
};

struct TemplateConfig {
    TemplateId id = 0;
    std::string name;

    ScalarType scalar_type = ScalarType::FP16;
    StorageMode storage_mode = StorageMode::Contiguous;

    size_t alignment = 64;
    size_t block_bytes = 0;      // Blocked 模式有效
    size_t grow_bytes = 0;       // 扩容粒度

    bool append_by_seq = true;   // 是否按 seq 追加
    bool supports_random_write = false;
    bool supports_random_read = true;
    FormatDescriptor format;
};

struct AllocationInfo {
    MemoryDomain domain = MemoryDomain::HostRAM;
    size_t bytes = 0;
    size_t alignment = 0;
};

// ============================================================================
// Span / View 访问对象
// ============================================================================

struct ByteSpan {
    Byte* ptr = nullptr;
    size_t size = 0;

    ByteSpan() = default;
    ByteSpan(Byte* p, size_t s) : ptr(p), size(s) {}

    bool empty() const { return size == 0 || ptr == nullptr; }
};

struct ConstByteSpan {
    const Byte* ptr = nullptr;
    size_t size = 0;

    ConstByteSpan() = default;
    ConstByteSpan(const Byte* p, size_t s) : ptr(p), size(s) {}

    bool empty() const { return size == 0 || ptr == nullptr; }
};

struct AccessView {
    bool contiguous = false;
    Byte* base = nullptr;
    size_t bytes = 0;

    uint32_t seq_begin = 0;
    uint32_t seq_len = 0;

    const KVTemplate* templ = nullptr;

    AccessView() = default;

    bool empty() const { return bytes == 0 || base == nullptr; }
};

// ============================================================================
// Plane 规格与统计
// ============================================================================

struct PlaneSpec {
    PlaneKind kind;
    TemplateId template_id;
    uint32_t initial_seq_capacity = 0;
    uint32_t max_seq_capacity = 0;  // 0 means unlimited
};

struct PlaneStats {
    size_t bytes_allocated = 0;
    uint32_t seq_capacity = 0;      // 预分配的容量
    uint32_t seq_length = 0;         // 当前实际长度
    uint32_t max_seq_capacity = 0;   // 最大允许长度（ring buffer模式）
    bool is_ring_buffer = false;     // 是否启用ring buffer模式
    uint32_t write_head = 0;         // 写头位置（ring buffer）

    std::string to_string() const {
        std::ostringstream oss;
        oss << "PlaneStats(bytes=" << bytes_allocated
            << ", capacity=" << seq_capacity
            << ", length=" << seq_length;
        if (is_ring_buffer) {
            oss << ", max=" << max_seq_capacity
                << ", head=" << write_head;
        }
        oss << ")";
        return oss.str();
    }
};

struct LayerSpec {
    LayerId layer_id = 0;
    PlaneSpec k_spec;
    PlaneSpec v_spec;

    LayerSpec() = default;
    LayerSpec(LayerId id, TemplateId k_templ, TemplateId v_templ, uint32_t capacity = 0)
        : layer_id(id) {
        k_spec.kind = PlaneKind::K;
        k_spec.template_id = k_templ;
        k_spec.initial_seq_capacity = capacity;

        v_spec.kind = PlaneKind::V;
        v_spec.template_id = v_templ;
        v_spec.initial_seq_capacity = capacity;
    }

    // 带max_seq_capacity的构造函数
    LayerSpec(LayerId id, TemplateId k_templ, TemplateId v_templ,
              uint32_t capacity, uint32_t max_capacity)
        : layer_id(id) {
        k_spec.kind = PlaneKind::K;
        k_spec.template_id = k_templ;
        k_spec.initial_seq_capacity = capacity;
        k_spec.max_seq_capacity = max_capacity;

        v_spec.kind = PlaneKind::V;
        v_spec.template_id = v_templ;
        v_spec.initial_seq_capacity = capacity;
        v_spec.max_seq_capacity = max_capacity;
    }
};

// ============================================================================
// KVTemplate 抽象基类
// ============================================================================

class KVTemplate : public std::enable_shared_from_this<KVTemplate> {
public:
    virtual ~KVTemplate() = default;

    virtual const TemplateConfig& config() const = 0;
    virtual const TemplateShape& shape() const = 0;

    // 给定逻辑坐标，映射到物理地址
    virtual PhysicalAddr locate(const LogicalCoord& coord) const = 0;

    // 给定 seq 范围，返回建议的存储跨度
    virtual size_t bytes_for_tokens(uint32_t token_count) const = 0;

    // 给定目标 seq 容量，计算最小所需容量
    virtual size_t bytes_for_capacity(uint32_t seq_capacity) const = 0;

    // 是否要求某个 token 范围连续
    virtual bool can_export_contiguous_span(
        uint32_t seq_begin,
        uint32_t seq_len
    ) const = 0;

    // 获取元素大小
    virtual size_t element_size() const = 0;

    // 内存域信息
    virtual AllocationInfo allocation_info() const = 0;
};

// ============================================================================
// Plain Template 实现 - [head][seq][dim] 布局
// ============================================================================

template<ScalarType ST>
class PlainKVTemplate : public KVTemplate {
public:
    PlainKVTemplate(uint32_t num_heads, uint32_t head_dim, TemplateId id = 0, const std::string& name = "plain") {
        cfg_.id = id;
        cfg_.name = name;
        cfg_.scalar_type = ST;
        cfg_.storage_mode = StorageMode::Contiguous;
        cfg_.alignment = 64;
        cfg_.supports_random_read = true;
        cfg_.supports_random_write = true;
        cfg_.format.storage_type = ST;

        shape_.num_heads = num_heads;
        shape_.head_dim = head_dim;
        shape_.inner_bytes = head_dim * element_size();
    }

    const TemplateConfig& config() const override { return cfg_; }
    const TemplateShape& shape() const override { return shape_; }

    PhysicalAddr locate(const LogicalCoord& coord) const override {
        PhysicalAddr p;
        size_t elem_size = element_size();
        size_t dim_stride = elem_size;
        size_t head_stride = static_cast<size_t>(shape_.head_dim) * elem_size;
        size_t token_stride = static_cast<size_t>(shape_.num_heads) * head_stride;

        p.byte_offset = coord.seq * token_stride +
                        coord.head * head_stride +
                        coord.dim * dim_stride;
        p.byte_size = elem_size;
        p.loop0 = coord.seq;
        p.loop1 = coord.head;
        p.inner_offset = coord.dim * elem_size;
        p.valid = true;
        return p;
    }

    size_t bytes_for_tokens(uint32_t token_count) const override {
        return static_cast<size_t>(token_count) *
               static_cast<size_t>(shape_.num_heads) *
               static_cast<size_t>(shape_.head_dim) *
               element_size();
    }

    size_t bytes_for_capacity(uint32_t seq_capacity) const override {
        return bytes_for_tokens(seq_capacity);
    }

    bool can_export_contiguous_span(uint32_t, uint32_t) const override {
        return true;
    }

    size_t element_size() const override {
        return scalar_type_size(ST);
    }

    AllocationInfo allocation_info() const override {
        AllocationInfo info;
        info.domain = MemoryDomain::HostRAM;
        info.alignment = cfg_.alignment;
        return info;
    }

private:
    TemplateConfig cfg_;
    TemplateShape shape_;
};

// ============================================================================
// Dim Block Template 实现 - [seq][head][dim_block] 布局
// 语义：dim 表示已经pack后的block索引，单个元素大小为block_bytes。
// ============================================================================

class DimBlockKVTemplate : public KVTemplate {
public:
    DimBlockKVTemplate(uint32_t num_heads, uint32_t dim_blocks, uint32_t block_bytes,
                       TemplateId id = 0, const std::string& name = "dimblock")
        : block_bytes_(block_bytes) {
        if (block_bytes_ == 0) {
            throw std::invalid_argument("DimBlockKVTemplate block_bytes must be > 0");
        }
        cfg_.id = id;
        cfg_.name = name;
        cfg_.scalar_type = ScalarType::CUSTOM;
        cfg_.storage_mode = StorageMode::Contiguous;
        cfg_.alignment = 64;
        cfg_.supports_random_read = true;
        cfg_.supports_random_write = true;
        cfg_.format.storage_type = ScalarType::CUSTOM;

        shape_.num_heads = num_heads;
        shape_.head_dim = dim_blocks;
        shape_.inner_bytes = block_bytes_;
    }

    const TemplateConfig& config() const override { return cfg_; }
    const TemplateShape& shape() const override { return shape_; }

    PhysicalAddr locate(const LogicalCoord& coord) const override {
        PhysicalAddr p;
        size_t elem_size = block_bytes_;
        size_t dim_stride = elem_size;
        size_t head_stride = static_cast<size_t>(shape_.head_dim) * dim_stride;
        size_t token_stride = static_cast<size_t>(shape_.num_heads) * head_stride;

        p.byte_offset = coord.seq * token_stride +
                        coord.head * head_stride +
                        coord.dim * dim_stride;
        p.byte_size = elem_size;
        p.loop0 = coord.seq;
        p.loop1 = coord.head;
        p.inner_offset = coord.dim * elem_size;
        p.valid = true;
        return p;
    }

    size_t bytes_for_tokens(uint32_t token_count) const override {
        return static_cast<size_t>(token_count) *
               static_cast<size_t>(shape_.num_heads) *
               static_cast<size_t>(shape_.head_dim) *
               block_bytes_;
    }

    size_t bytes_for_capacity(uint32_t seq_capacity) const override {
        return bytes_for_tokens(seq_capacity);
    }

    bool can_export_contiguous_span(uint32_t, uint32_t) const override {
        return true;
    }

    size_t element_size() const override {
        return block_bytes_;
    }

    AllocationInfo allocation_info() const override {
        AllocationInfo info;
        info.domain = MemoryDomain::HostRAM;
        info.alignment = cfg_.alignment;
        return info;
    }

private:
    TemplateConfig cfg_;
    TemplateShape shape_;
    uint32_t block_bytes_;
};

// ============================================================================
// KVPlane 接口
// ============================================================================

class KVPlane {
public:
    virtual ~KVPlane() = default;

    virtual PlaneKind kind() const = 0;
    virtual LayerId layer_id() const = 0;
    virtual const KVTemplate& templ() const = 0;

    virtual const PlaneStats& stats() const = 0;

    virtual bool reserve_seq(uint32_t target_seq_capacity) = 0;
    virtual bool resize_seq(uint32_t target_seq_length) = 0;
    virtual bool append_seq(uint32_t token_count) = 0;
    virtual void clear() = 0;

    // 原始访问
    virtual void* data() = 0;
    virtual const void* data() const = 0;

    // 单点定位
    virtual PhysicalAddr locate(const LogicalCoord& coord) const = 0;

    // 范围访问
    virtual AccessView acquire_seq_view(
        uint32_t seq_begin,
        uint32_t seq_len,
        AccessMode mode
    ) = 0;

    virtual void release_seq_view(AccessView& view) = 0;
};

// ============================================================================
// LayerStorage 接口
// ============================================================================

class LayerStorage {
public:
    virtual ~LayerStorage() = default;

    virtual LayerId layer_id() const = 0;

    virtual KVPlane& plane(PlaneKind kind) = 0;
    virtual const KVPlane& plane(PlaneKind kind) const = 0;

    virtual bool reserve_seq(uint32_t target_seq_capacity) = 0;
    virtual bool append_seq(uint32_t token_count) = 0;
    virtual void clear() = 0;

    virtual size_t total_bytes() const = 0;
};

// ============================================================================
// KVCacheStorage 接口
// ============================================================================

struct KVCacheStorageConfig {
    size_t default_alignment = 64;
    bool thread_safe = false;
    // 端侧默认ring窗口。为0时不启用默认ring。
    uint32_t default_max_seq_capacity = 0;
};

class KVCacheStorage {
public:
    virtual ~KVCacheStorage() = default;

    virtual bool register_template(
        std::shared_ptr<KVTemplate> templ
    ) = 0;

    virtual const KVTemplate* find_template(TemplateId id) const = 0;

    virtual bool create_layer(const LayerSpec& spec) = 0;
    virtual bool has_layer(LayerId layer) const = 0;

    virtual LayerStorage& layer(LayerId layer) = 0;
    virtual const LayerStorage& layer(LayerId layer) const = 0;

    virtual bool reserve_all(uint32_t target_seq_capacity) = 0;
    virtual bool append_all(uint32_t token_count) = 0;
    virtual void clear_all() = 0;

    virtual size_t total_bytes() const = 0;
};

// ============================================================================
// KVCacheStorageBuilder
// ============================================================================

class KVCacheStorageBuilder {
public:
    KVCacheStorageBuilder& config(const KVCacheStorageConfig& cfg);

    KVCacheStorageBuilder& add_template(std::shared_ptr<KVTemplate> templ);

    // 简单版本：max_seq_capacity 继承 config.default_max_seq_capacity
    KVCacheStorageBuilder& add_layer(
        LayerId layer,
        TemplateId k_template,
        TemplateId v_template,
        uint32_t initial_seq_capacity
    );

    // 完整版本：支持ring buffer模式
    KVCacheStorageBuilder& add_layer(
        LayerId layer,
        TemplateId k_template,
        TemplateId v_template,
        uint32_t initial_seq_capacity,
        uint32_t max_seq_capacity  // ring buffer模式下最大序列长度
    );

    std::unique_ptr<KVCacheStorage> build();

private:
    KVCacheStorageConfig config_;
    std::vector<std::shared_ptr<KVTemplate>> templates_;
    std::vector<LayerSpec> layers_;
};

// ============================================================================
// 便捷类型别名
// ============================================================================

using PlainFP32K = PlainKVTemplate<ScalarType::FP32>;
using PlainFP32V = PlainKVTemplate<ScalarType::FP32>;
using PlainFP16K = PlainKVTemplate<ScalarType::FP16>;
using PlainFP16V = PlainKVTemplate<ScalarType::FP16>;
using PlainINT8K = PlainKVTemplate<ScalarType::INT8>;
using PlainINT8V = PlainKVTemplate<ScalarType::INT8>;
using PlainINT16K = PlainKVTemplate<ScalarType::INT16>;
using PlainINT16V = PlainKVTemplate<ScalarType::INT16>;

using DimBlockK = DimBlockKVTemplate;
using DimBlockV = DimBlockKVTemplate;

}  // namespace mobilekv

#endif  // MOBILEKV_KV_CACHE_BASIC_H_
