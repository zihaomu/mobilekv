# MobileKV

MobileKV is a focused KV cache manager for edge and mobile LLM inference.
It is designed to be a production-grade memory component, not a full model runtime:
MobileKV owns KV memory planning, token retention policy, and attention-facing metadata,
while your inference engine keeps control of scheduling and kernel execution.

一个只做层次化存储与寻址、不做量化语义、不绑定具体 kernel、不管 disk 的 RAM KV Cache Storage API。

我建议你把第一版定位成：

“逻辑坐标 + 模板描述 + 存储后端”三层解耦。

这样用户能自定义很多内容，但你的框架本身仍然可控，不会退化成“裸指针池”。

当前实现要点（2026-03）：

- 支持按 layer 配置不同模板，且 K/V 可独立精度（`FP32/FP16/BF16/INT8/UINT8/INT16`）
- 支持 `KVCacheStorageConfig::default_max_seq_capacity` 作为端侧默认ring窗口（4参数 `add_layer` 自动继承）
- ring buffer 模式下，`locate` 使用窗口逻辑索引（`[0, seq_length)`）映射到物理槽位
- 严格校验：当 `initial_seq_capacity > max_seq_capacity` 时构建失败（不做静默截断）
- `KVAccessor<T>` 会做运行时类型检查，避免错误标量类型直接读写
- `TemplateConfig` 包含 `FormatDescriptor`（仅存储格式元数据，不做量化计算）

更多格式描述见：[docs/format-descriptor.md](docs/format-descriptor.md)

设计目标

这版 API 重点满足这些约束：

只做 RAM 存储

不关心量化/反量化逻辑

允许不同 layer 用不同模板

允许 K 和 V 分别用不同模板

模板由用户预先定义

尽量满足用户的读写/追加/定位需求

后续可平滑扩展到 disk

不把 layout 写死成某种固定维度

所以最核心的思路是：

你不直接管理 [layer][seq][head][dim]

你管理的是 Layer 下的若干 Plane

每个 Plane 绑定一个 Template

Template 负责描述逻辑坐标如何映射到物理存储

Storage 只负责分配、扩容、访问、返回 span/block

一、建议的核心抽象

我建议 API 分成 6 个核心对象：

ScalarType

KVTemplate

KVPlane

LayerStorage

KVCacheStorage

AccessView

其中最关键的是：

KVTemplate：用户定义布局

KVPlane：K 或 V 的一个具体实例

KVCacheStorage：全局管理多层

二、建议的核心类型定义

下面我先给你一版偏 C++ 的接口草图，再解释为什么这样设计。

1. 基础类型
enum class ScalarType {
    FP32,
    FP16,
    BF16,
    INT8,
    UINT8,
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

using LayerId = uint32_t;
using TemplateId = uint32_t;
using TokenIndex = uint32_t;
using HeadIndex = uint32_t;
using BlockIndex = uint32_t;
using Byte = uint8_t;
2. 逻辑坐标

你的统一抽象里不要暴露过多固定维度，但需要一个标准逻辑坐标表达，供模板解释。

struct LogicalCoord {
    uint32_t layer = 0;
    uint32_t seq = 0;
    uint32_t head = 0;
    uint32_t dim = 0;
};

这个 LogicalCoord 不是要求底层必须按这四维存，而是为了让模板有统一输入。

3. 模板描述

这是整个系统最重要的对象。

struct TemplateShape {
    uint32_t num_heads = 0;
    uint32_t head_dim = 0;

    // 可选：用户自己的两个外层循环解释
    uint32_t loop0_extent = 0;
    uint32_t loop1_extent = 0;

    // inner 单元大小（字节）
    uint32_t inner_bytes = 0;
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
};

然后模板本身不只是一堆参数，还要带映射规则。

struct PhysicalAddr {
    size_t byte_offset = 0;
    size_t byte_size = 0;
    uint32_t loop0 = 0;
    uint32_t loop1 = 0;
    uint32_t inner_offset = 0;
    bool valid = false;
};

class KVTemplate {
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
};
三、Plane：K 和 V 的具体存储实例

每个 layer 至少有两个 plane：

K plane

V plane

而且它们可以绑定不同模板。

struct PlaneSpec {
    PlaneKind kind;
    TemplateId template_id;
    uint32_t initial_seq_capacity = 0;
};

struct PlaneStats {
    size_t bytes_allocated = 0;
    uint32_t seq_capacity = 0;
    uint32_t seq_length = 0;
};

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
};
四、Span / View 访问对象

你不能只返回裸指针，否则后面扩展会很难。建议返回一个轻量 view。

struct ByteSpan {
    Byte* ptr = nullptr;
    size_t size = 0;
};

struct ConstByteSpan {
    const Byte* ptr = nullptr;
    size_t size = 0;
};

struct AccessView {
    bool contiguous = false;
    Byte* base = nullptr;
    size_t bytes = 0;

    uint32_t seq_begin = 0;
    uint32_t seq_len = 0;

    const KVTemplate* templ = nullptr;
};

然后给 Plane 提供范围访问：

class KVPlaneRangeAccess {
public:
    virtual ~KVPlaneRangeAccess() = default;

    virtual AccessView acquire_seq_view(
        uint32_t seq_begin,
        uint32_t seq_len,
        AccessMode mode
    ) = 0;

    virtual void release_seq_view(AccessView& view) = 0;
};
五、LayerStorage

每层有 K/V 两个 plane，但不要写死只能两个；后面你可能会挂别的辅助 plane。

struct LayerSpec {
    LayerId layer_id = 0;
    PlaneSpec k_spec;
    PlaneSpec v_spec;
};

class LayerStorage {
public:
    virtual ~LayerStorage() = default;

    virtual LayerId layer_id() const = 0;

    virtual KVPlane& plane(PlaneKind kind) = 0;
    virtual const KVPlane& plane(PlaneKind kind) const = 0;

    virtual bool reserve_seq(uint32_t target_seq_capacity) = 0;
    virtual bool append_seq(uint32_t token_count) = 0;
    virtual void clear() = 0;
};
六、全局 KVCacheStorage

这个对象负责：

注册模板

创建各 layer

提供统一查询与访问

struct KVCacheStorageConfig {
    size_t default_alignment = 64;
    bool thread_safe = false;
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
七、推荐再加一层 Builder API

纯构造函数会很难用，建议给用户 builder。

class KVCacheStorageBuilder {
public:
    KVCacheStorageBuilder& config(const KVCacheStorageConfig& cfg);

    KVCacheStorageBuilder& add_template(std::shared_ptr<KVTemplate> templ);

    KVCacheStorageBuilder& add_layer(
        LayerId layer,
        TemplateId k_template,
        TemplateId v_template,
        uint32_t initial_seq_capacity
    );

    std::unique_ptr<KVCacheStorage> build();
};
八、为什么要这样设计

下面我说每一层为什么需要。

1. 为什么必须有 KVTemplate

因为你想让用户自定义更多内容，但又不能让系统退化成“自己 malloc 自己算偏移”。

KVTemplate 的价值在于：

用户定义布局

你统一调度存储

Kernel 端和存储端通过模板契约对齐

也就是说，模板是用户自由度和框架可控性之间的边界。

如果没有模板，后面很快会变成：

每个用户都自己写一套指针管理

你的框架只剩下一个字节池

没法做统一 reserve / append / range access

2. 为什么 K 和 V 必须分开绑定模板

因为很多实现里：

K 的 physical layout 和 V 不同

K 的精度和 V 的精度不同

K 的追加方式和 V 的追加方式也可能不同

所以必须允许：

layer 0:
  K -> template A
  V -> template B

而不是一个 layer 一个模板。

3. 为什么要有 LogicalCoord

因为哪怕底层完全 packed，用户仍然会从逻辑上思考：

第几个 token

第几个 head

第几个 dim

你需要一个统一的逻辑输入，交给模板去解释。
否则外部用户得直接面向 loop0/loop1/inner_offset 编程，这会非常难用。

4. 为什么要有 reserve_seq / append_seq / resize_seq

因为 KV cache 的最主要访问模式不是 random write，而是：

预留长度

按 token 追加

查询当前位置

如果只给 locate，上层就得自己维护长度和扩容，存储器价值会大幅下降。

5. 为什么要返回 AccessView 而不是直接 void*

因为你现在只做 RAM，但以后要扩展 disk。
如果你现在 API 到处裸指针，后面会很难平滑升级。

AccessView 的好处是：

RAM 下可以直接是连续指针

以后 disk 下可以变成映射视图

再以后 blocked backend 可以返回一段块视图

这就是为将来留口。

九、一个用户如何自定义模板

你现在希望用户提前定义模板。可以这样。

例 1：普通 [head][seq][dim] FP16 K
class PlainKTemplate : public KVTemplate {
public:
    PlainKTemplate(uint32_t num_heads, uint32_t head_dim) {
        cfg_.id = 1;
        cfg_.name = "plain_k_fp16";
        cfg_.scalar_type = ScalarType::FP16;
        cfg_.storage_mode = StorageMode::Contiguous;
        cfg_.alignment = 64;

        shape_.num_heads = num_heads;
        shape_.head_dim = head_dim;
        shape_.inner_bytes = head_dim * sizeof(uint16_t);
    }

    const TemplateConfig& config() const override { return cfg_; }
    const TemplateShape& shape() const override { return shape_; }

    PhysicalAddr locate(const LogicalCoord& c) const override {
        PhysicalAddr p;
        size_t token_stride = shape_.num_heads * shape_.head_dim * sizeof(uint16_t);
        size_t head_stride = shape_.head_dim * sizeof(uint16_t);
        p.byte_offset = c.seq * token_stride + c.head * head_stride + c.dim * sizeof(uint16_t);
        p.byte_size = sizeof(uint16_t);
        p.valid = true;
        return p;
    }

    size_t bytes_for_tokens(uint32_t token_count) const override {
        return static_cast<size_t>(token_count) * shape_.num_heads * shape_.head_dim * sizeof(uint16_t);
    }

    size_t bytes_for_capacity(uint32_t seq_capacity) const override {
        return bytes_for_tokens(seq_capacity);
    }

    bool can_export_contiguous_span(uint32_t, uint32_t) const override {
        return true;
    }

private:
    TemplateConfig cfg_;
    TemplateShape shape_;
};
例 2：packed [head][seq/hp][dim][hp] 的 K
class PackedKTemplate : public KVTemplate {
public:
    PackedKTemplate(uint32_t num_heads, uint32_t head_dim, uint32_t hp, ScalarType st) : hp_(hp) {
        cfg_.id = 2;
        cfg_.name = "packed_k";
        cfg_.scalar_type = st;
        cfg_.storage_mode = StorageMode::Contiguous;
        cfg_.alignment = 64;

        shape_.num_heads = num_heads;
        shape_.head_dim = head_dim;
        shape_.inner_bytes = hp_ * element_size();
    }

    const TemplateConfig& config() const override { return cfg_; }
    const TemplateShape& shape() const override { return shape_; }

    PhysicalAddr locate(const LogicalCoord& c) const override {
        PhysicalAddr p;
        uint32_t seq_block = c.seq / hp_;
        uint32_t seq_inner = c.seq % hp_;

        size_t elem = element_size();
        size_t dim_rounded = shape_.head_dim;
        size_t head_bytes = 0; // 需要用户预先定义 capacity 后才可精确
        (void)head_bytes;

        // 这里只示意映射规则
        p.loop0 = seq_block;
        p.loop1 = c.head;
        p.inner_offset = (c.dim * hp_ + seq_inner) * elem;
        p.valid = true;
        return p;
    }

    size_t bytes_for_tokens(uint32_t token_count) const override {
        uint32_t blocks = (token_count + hp_ - 1) / hp_;
        return static_cast<size_t>(shape_.num_heads) * blocks * shape_.head_dim * hp_ * element_size();
    }

    size_t bytes_for_capacity(uint32_t seq_capacity) const override {
        return bytes_for_tokens(seq_capacity);
    }

    bool can_export_contiguous_span(uint32_t, uint32_t) const override {
        return false;
    }

private:
    size_t element_size() const {
        switch (cfg_.scalar_type) {
            case ScalarType::FP32: return 4;
            case ScalarType::FP16: return 2;
            case ScalarType::BF16: return 2;
            case ScalarType::INT8: return 1;
            default: return 1;
        }
    }

    TemplateConfig cfg_;
    TemplateShape shape_;
    uint32_t hp_ = 1;
};
十、推荐的第一版用户使用方式
1. 注册模板
builder.add_template(std::make_shared<PlainKTemplate>(8, 128));
builder.add_template(std::make_shared<PackedKTemplate>(8, 128, 8, ScalarType::INT8));
2. 为不同 layer 指定不同模板
builder.add_layer(0, /*K*/1, /*V*/1, 1024);
builder.add_layer(1, /*K*/2, /*V*/1, 1024);
builder.add_layer(2, /*K*/2, /*V*/2, 2048);

这就实现了：

layer 0 用 plain fp16

layer 1 的 K 用 packed int8，V 用 plain fp16

layer 2 的 K/V 都 packed

十一、你应该显式限制第一版不做什么

为了 API 清晰，我建议你在第一版明确不支持：

disk / mmap

自动量化/反量化

跨设备存储

稀疏 page table

跨 layer 共享物理块

并发写冲突解决

copy-on-write prefix fork

这样 API 会干净很多。

十二、为了第二版扩展到 disk，第一版要预留什么

虽然这版不做 disk，但建议你现在就预留这些接口，不一定实现：

enum class MemoryDomain {
    HostRAM,
    DiskMapped
};

struct AllocationInfo {
    MemoryDomain domain = MemoryDomain::HostRAM;
    size_t bytes = 0;
    size_t alignment = 0;
};

以及：

virtual AllocationInfo allocation_info() const = 0;

和前面说的 AccessView acquire/release。
这样以后加 disk 时，很多上层代码不用改。

十三、我建议的最小可落地 API 版本

如果你想尽快实现，我建议先做这一组最小接口：

class KVTemplate;
class KVPlane;
class LayerStorage;
class KVCacheStorage;

struct LogicalCoord;
struct PhysicalAddr;
struct AccessView;
struct LayerSpec;

然后最少支持这些方法：

register_template()
create_layer()
layer()
reserve_seq()
append_seq()
locate()
acquire_seq_view()
release_seq_view()
clear()
total_bytes()

这是最小但不简陋的一版。

十四、为什么这版比“只做 [layer][loop0][loop1][inner] 数组”更好

因为单纯的 [layer][loop0][loop1][inner] 只解决了“存储看起来统一”，没有解决：

用户如何表达逻辑坐标

K/V 如何不同模板

如何追加 seq

如何做范围访问

如何为后续 disk 扩展留口

而我上面这版把“统一存储器”拆成了：

模板定义布局

Plane 绑定模板

Layer 聚合 Plane

Storage 管理全局

View 管理访问

这样职责清楚，后续演进空间很大。

核心feature：
- ring-buffer存储
