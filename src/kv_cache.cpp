#include "mobilekv/kv_cache.h"
#include <cstdlib>
#include <cstring>
#include <memory>
#include <algorithm>

namespace mobilekv {

namespace {

bool is_valid_alignment(size_t alignment) {
    if (alignment == 0) {
        return false;
    }
    if ((alignment & (alignment - 1)) != 0) {
        return false;
    }
    return alignment >= alignof(void*);
}

size_t align_up(size_t size, size_t alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

}  // namespace

// ============================================================================
// KVPlane 实现
// ============================================================================

class KVPlaneImpl : public KVPlane {
public:
    KVPlaneImpl(LayerId layer_id, PlaneKind kind, std::shared_ptr<KVTemplate> templ,
                uint32_t max_seq_capacity = 0)
        : layer_id_(layer_id)
        , kind_(kind)
        , templ_(std::move(templ))
        , data_(nullptr, &std::free) {
        stats_.bytes_allocated = 0;
        stats_.seq_capacity = 0;
        stats_.seq_length = 0;
        stats_.max_seq_capacity = max_seq_capacity;
        stats_.is_ring_buffer = (max_seq_capacity > 0);
        stats_.write_head = 0;
        oldest_seq_ = 0;
    }

    PlaneKind kind() const override { return kind_; }
    LayerId layer_id() const override { return layer_id_; }
    const KVTemplate& templ() const override { return *templ_; }

    const PlaneStats& stats() const override { return stats_; }

    bool reserve_seq(uint32_t target_seq_capacity) override {
        // Ring buffer模式下，预分配max_seq_capacity或target_seq_capacity
        uint32_t actual_capacity = target_seq_capacity;
        if (stats_.is_ring_buffer && stats_.max_seq_capacity > 0) {
            actual_capacity = std::max(target_seq_capacity, stats_.max_seq_capacity);
        }

        if (actual_capacity <= stats_.seq_capacity) {
            return true;
        }

        size_t new_bytes = templ_->bytes_for_capacity(actual_capacity);

        // 对齐到模板要求的alignment
        size_t alignment = templ_->config().alignment;
        if (!is_valid_alignment(alignment)) {
            return false;
        }
        new_bytes = align_up(new_bytes, alignment);
        if (new_bytes == 0) {
            return false;
        }

        // 分配新内存
        Byte* new_data = static_cast<Byte*>(aligned_alloc(alignment, new_bytes));
        if (!new_data) {
            return false;
        }

        // 如果有旧数据，复制过来
        if (data_ && stats_.bytes_allocated > 0) {
            std::memcpy(new_data, data_.get(), stats_.bytes_allocated);
        }

        data_.reset(new_data);
        stats_.bytes_allocated = new_bytes;
        stats_.seq_capacity = actual_capacity;

        return true;
    }

    bool resize_seq(uint32_t target_seq_length) override {
        // Ring buffer模式下，长度不能超过max_seq_capacity
        if (stats_.is_ring_buffer && stats_.max_seq_capacity > 0) {
            if (target_seq_length > stats_.max_seq_capacity) {
                target_seq_length = stats_.max_seq_capacity;
            }
        }

        if (target_seq_length > stats_.seq_capacity) {
            if (!reserve_seq(target_seq_length)) {
                return false;
            }
        }
        stats_.seq_length = target_seq_length;
        if (stats_.is_ring_buffer) {
            oldest_seq_ = 0;
            if (stats_.max_seq_capacity > 0) {
                stats_.write_head = target_seq_length % stats_.max_seq_capacity;
            } else {
                stats_.write_head = 0;
            }
        }
        return true;
    }

    bool append_seq(uint32_t token_count) override {
        if (stats_.is_ring_buffer) {
            // Ring buffer模式：覆盖最旧的token
            return append_seq_ring_buffer(token_count);
        } else {
            // 普通模式：直接增长
            uint32_t new_length = stats_.seq_length + token_count;
            return resize_seq(new_length);
        }
    }

    // Ring buffer模式的追加实现
    bool append_seq_ring_buffer(uint32_t token_count) {
        uint32_t max_len = stats_.max_seq_capacity;
        if (max_len == 0) {
            return false;
        }
        if (token_count == 0) {
            return true;
        }

        if (stats_.seq_capacity < max_len) {
            if (!reserve_seq(max_len)) {
                return false;
            }
        }

        if (token_count > max_len) {
            // 追加的token比buffer还大，直接从新token开始
            stats_.write_head = 0;
            stats_.seq_length = max_len;
            oldest_seq_ = 0;
            return true;
        }

        uint32_t current_len = stats_.seq_length;
        uint32_t overflow = 0;

        if (current_len < max_len) {
            // 还没填满buffer，正常追加
            uint32_t new_length = std::min(current_len + token_count, max_len);
            if (current_len + token_count > max_len) {
                overflow = current_len + token_count - max_len;
            }
            stats_.seq_length = new_length;
            // 更新write_head
            stats_.write_head = (stats_.write_head + token_count) % max_len;
        } else {
            // Buffer已满，覆盖最旧的token
            overflow = token_count;
            stats_.write_head = (stats_.write_head + token_count) % max_len;
            // seq_length保持为max_len
        }
        oldest_seq_ = (oldest_seq_ + overflow) % max_len;

        return true;
    }

    void clear() override {
        stats_.seq_length = 0;
        oldest_seq_ = 0;
        if (stats_.is_ring_buffer) {
            stats_.write_head = 0;
        }
    }

    void* data() override {
        return data_.get();
    }

    const void* data() const override {
        return data_.get();
    }

    PhysicalAddr locate(const LogicalCoord& coord) const override {
        const auto& shape = templ_->shape();
        if (coord.head >= shape.num_heads || coord.dim >= shape.head_dim) {
            return PhysicalAddr();
        }
        if (coord.seq >= stats_.seq_length) {
            return PhysicalAddr();
        }

        LogicalCoord adjusted_coord = coord;
        adjusted_coord.layer = layer_id_;
        adjusted_coord.seq = logical_to_physical_seq(coord.seq);
        return templ_->locate(adjusted_coord);
    }

    AccessView acquire_seq_view(
        uint32_t seq_begin,
        uint32_t seq_len,
        AccessMode mode
    ) override {
        AccessView view;
        view.seq_begin = seq_begin;
        view.seq_len = seq_len;
        view.templ = templ_.get();

        if (seq_len == 0) {
            view.contiguous = true;
            view.base = data_.get();
            view.bytes = 0;
            return view;
        }

        if (seq_begin > stats_.seq_length || seq_len > (stats_.seq_length - seq_begin)) {
            return view;  // 返回空的view
        }

        uint32_t physical_begin_seq = logical_to_physical_seq(seq_begin);
        bool wraps_ring = range_wraps_ring(seq_begin, seq_len);

        // 检查是否可以连续访问。
        bool can_contiguous = !wraps_ring &&
                              templ_->can_export_contiguous_span(physical_begin_seq, seq_len);

        if (can_contiguous) {
            size_t begin_offset = 0;
            size_t view_bytes = 0;
            if (!compute_span_byte_range(seq_begin, seq_len, begin_offset, view_bytes)) {
                return AccessView();
            }
            view.contiguous = true;
            view.base = data_.get() + begin_offset;
            view.bytes = view_bytes;
        } else {
            view.contiguous = false;
            view.base = data_.get();
            view.bytes = stats_.bytes_allocated;
        }

        return view;
    }

    void release_seq_view(AccessView& view) override {
        // 当前实现不需要做什么，view是轻量引用
        // 后续如果加入disk-backed storage，在这里处理释放
        view = AccessView();
    }

private:
    uint32_t logical_to_physical_seq(uint32_t seq) const {
        if (!stats_.is_ring_buffer || stats_.max_seq_capacity == 0) {
            return seq;
        }
        if (stats_.seq_length < stats_.max_seq_capacity) {
            return seq;
        }
        return (oldest_seq_ + seq) % stats_.max_seq_capacity;
    }

    bool range_wraps_ring(uint32_t seq_begin, uint32_t seq_len) const {
        if (!stats_.is_ring_buffer || stats_.max_seq_capacity == 0) {
            return false;
        }
        if (stats_.seq_length < stats_.max_seq_capacity || seq_len == 0) {
            return false;
        }
        uint32_t physical_begin = logical_to_physical_seq(seq_begin);
        uint32_t physical_last = logical_to_physical_seq(seq_begin + seq_len - 1);
        return physical_begin > physical_last;
    }

    bool compute_span_byte_range(uint32_t seq_begin, uint32_t seq_len,
                                 size_t& begin_offset, size_t& span_bytes) const {
        if (seq_len == 0) {
            begin_offset = 0;
            span_bytes = 0;
            return true;
        }
        if (!data_) {
            return false;
        }

        const auto& shape = templ_->shape();
        if (shape.num_heads == 0 || shape.head_dim == 0) {
            return false;
        }

        uint32_t physical_begin = logical_to_physical_seq(seq_begin);
        uint32_t physical_last = logical_to_physical_seq(seq_begin + seq_len - 1);

        LogicalCoord begin_coord(layer_id_, physical_begin, 0, 0);
        LogicalCoord end_coord(layer_id_, physical_last, shape.num_heads - 1, shape.head_dim - 1);

        PhysicalAddr begin = templ_->locate(begin_coord);
        PhysicalAddr end = templ_->locate(end_coord);
        if (!begin.valid || !end.valid) {
            return false;
        }

        size_t end_exclusive = end.byte_offset + end.byte_size;
        if (end_exclusive < begin.byte_offset) {
            return false;
        }

        begin_offset = begin.byte_offset;
        span_bytes = end_exclusive - begin.byte_offset;
        return true;
    }

    LayerId layer_id_;
    PlaneKind kind_;
    std::shared_ptr<KVTemplate> templ_;
    std::unique_ptr<Byte, void(*)(void*)> data_;
    PlaneStats stats_;
    uint32_t oldest_seq_ = 0;
};

// ============================================================================
// LayerStorage 实现
// ============================================================================

class LayerStorageImpl : public LayerStorage {
public:
    LayerStorageImpl(LayerId layer_id,
                    std::shared_ptr<KVTemplate> k_templ,
                    std::shared_ptr<KVTemplate> v_templ,
                    uint32_t initial_capacity,
                    uint32_t max_seq_capacity = 0)
        : layer_id_(layer_id)
        , k_plane_(layer_id, PlaneKind::K, std::move(k_templ), max_seq_capacity)
        , v_plane_(layer_id, PlaneKind::V, std::move(v_templ), max_seq_capacity) {
        if (initial_capacity > 0) {
            k_plane_.reserve_seq(initial_capacity);
            v_plane_.reserve_seq(initial_capacity);
        }
    }

    LayerId layer_id() const override { return layer_id_; }

    KVPlane& plane(PlaneKind kind) override {
        return kind == PlaneKind::K ? static_cast<KVPlane&>(k_plane_) :
                                       static_cast<KVPlane&>(v_plane_);
    }

    const KVPlane& plane(PlaneKind kind) const override {
        return kind == PlaneKind::K ? static_cast<const KVPlane&>(k_plane_) :
                                       static_cast<const KVPlane&>(v_plane_);
    }

    bool reserve_seq(uint32_t target_seq_capacity) override {
        bool ok = k_plane_.reserve_seq(target_seq_capacity);
        if (!ok) return false;
        ok = v_plane_.reserve_seq(target_seq_capacity);
        return ok;
    }

    bool append_seq(uint32_t token_count) override {
        bool ok = k_plane_.append_seq(token_count);
        if (!ok) return false;
        ok = v_plane_.append_seq(token_count);
        return ok;
    }

    void clear() override {
        k_plane_.clear();
        v_plane_.clear();
    }

    size_t total_bytes() const override {
        return k_plane_.stats().bytes_allocated +
               v_plane_.stats().bytes_allocated;
    }

private:
    LayerId layer_id_;
    KVPlaneImpl k_plane_;
    KVPlaneImpl v_plane_;
};

// ============================================================================
// KVCacheStorage 实现
// ============================================================================

class KVCacheStorageImpl : public KVCacheStorage {
public:
    KVCacheStorageImpl(const KVCacheStorageConfig& config)
        : config_(config) {}

    bool register_template(std::shared_ptr<KVTemplate> templ) override {
        if (!templ) return false;
        TemplateId id = templ->config().id;
        if (id == 0) {
            // 自动分配ID
            id = next_template_id_++;
        }
        templates_[id] = std::move(templ);
        return true;
    }

    const KVTemplate* find_template(TemplateId id) const override {
        auto it = templates_.find(id);
        if (it == templates_.end()) {
            return nullptr;
        }
        return it->second.get();
    }

    bool create_layer(const LayerSpec& spec) override {
        // 检查模板是否存在
        auto k_it = templates_.find(spec.k_spec.template_id);
        auto v_it = templates_.find(spec.v_spec.template_id);

        if (k_it == templates_.end() || v_it == templates_.end()) {
            return false;
        }

        // 如果层已存在，返回false
        if (layers_.find(spec.layer_id) != layers_.end()) {
            return false;
        }

        if (spec.k_spec.max_seq_capacity > 0 &&
            spec.k_spec.initial_seq_capacity > spec.k_spec.max_seq_capacity) {
            return false;
        }
        if (spec.v_spec.max_seq_capacity > 0 &&
            spec.v_spec.initial_seq_capacity > spec.v_spec.max_seq_capacity) {
            return false;
        }

        // 获取max_seq_capacity（K和V使用相同的值）
        uint32_t max_seq_capacity = std::max(spec.k_spec.max_seq_capacity, spec.v_spec.max_seq_capacity);
        uint32_t initial_seq_capacity =
            std::max(spec.k_spec.initial_seq_capacity, spec.v_spec.initial_seq_capacity);

        // 创建新的layer storage
        auto layer = std::make_unique<LayerStorageImpl>(
            spec.layer_id,
            k_it->second,
            v_it->second,
            initial_seq_capacity,
            max_seq_capacity
        );

        layers_[spec.layer_id] = std::move(layer);
        return true;
    }

    bool has_layer(LayerId layer) const override {
        return layers_.find(layer) != layers_.end();
    }

    LayerStorage& layer(LayerId layer) override {
        return *layers_.at(layer);
    }

    const LayerStorage& layer(LayerId layer) const override {
        return *layers_.at(layer);
    }

    bool reserve_all(uint32_t target_seq_capacity) override {
        for (auto& pair : layers_) {
            if (!pair.second->reserve_seq(target_seq_capacity)) {
                return false;
            }
        }
        return true;
    }

    bool append_all(uint32_t token_count) override {
        for (auto& pair : layers_) {
            if (!pair.second->append_seq(token_count)) {
                return false;
            }
        }
        return true;
    }

    void clear_all() override {
        for (auto& pair : layers_) {
            pair.second->clear();
        }
    }

    size_t total_bytes() const override {
        size_t total = 0;
        for (const auto& pair : layers_) {
            total += pair.second->total_bytes();
        }
        return total;
    }

    // 便捷方法
    void set_num_layers(size_t n) { num_layers_ = n; }
    size_t num_layers() const { return num_layers_; }

private:
    KVCacheStorageConfig config_;
    std::unordered_map<TemplateId, std::shared_ptr<KVTemplate>> templates_;
    std::unordered_map<LayerId, std::unique_ptr<LayerStorage>> layers_;
    TemplateId next_template_id_ = 1;
    size_t num_layers_ = 0;
};

// 让KVTemplate支持shared_from_this
// 由于KVTemplate是抽象基类，我们需要修改它的定义来支持shared_from_this
// 但为了简化，我们在KVCacheStorageImpl中用const_cast，
// 实际上更好的做法是让KVTemplate继承enable_shared_from_this

// ============================================================================
// KVCacheStorageBuilder 实现
// ============================================================================

KVCacheStorageBuilder& KVCacheStorageBuilder::config(const KVCacheStorageConfig& cfg) {
    config_ = cfg;
    return *this;
}

KVCacheStorageBuilder& KVCacheStorageBuilder::add_template(std::shared_ptr<KVTemplate> templ) {
    templates_.push_back(std::move(templ));
    return *this;
}

KVCacheStorageBuilder& KVCacheStorageBuilder::add_layer(
    LayerId layer,
    TemplateId k_template,
    TemplateId v_template,
    uint32_t initial_seq_capacity
) {
    return add_layer(layer, k_template, v_template, initial_seq_capacity,
                     config_.default_max_seq_capacity);
}

KVCacheStorageBuilder& KVCacheStorageBuilder::add_layer(
    LayerId layer,
    TemplateId k_template,
    TemplateId v_template,
    uint32_t initial_seq_capacity,
    uint32_t max_seq_capacity
) {
    LayerSpec spec;
    spec.layer_id = layer;
    spec.k_spec.kind = PlaneKind::K;
    spec.k_spec.template_id = k_template;
    spec.k_spec.initial_seq_capacity = initial_seq_capacity;
    spec.k_spec.max_seq_capacity = max_seq_capacity;
    spec.v_spec.kind = PlaneKind::V;
    spec.v_spec.template_id = v_template;
    spec.v_spec.initial_seq_capacity = initial_seq_capacity;
    spec.v_spec.max_seq_capacity = max_seq_capacity;
    layers_.push_back(spec);
    return *this;
}

std::unique_ptr<KVCacheStorage> KVCacheStorageBuilder::build() {
    // 没有层的情况返回nullptr
    if (layers_.empty()) {
        return nullptr;
    }

    auto storage = std::make_unique<KVCacheStorageImpl>(config_);

    // 先注册所有模板
    for (auto& templ : templates_) {
        storage->register_template(std::move(templ));
    }

    // 然后创建所有层
    for (auto& layer_spec : layers_) {
        if (!storage->create_layer(layer_spec)) {
            return nullptr;
        }
    }

    storage->set_num_layers(layers_.size());

    return std::move(storage);
}

}  // namespace mobilekv
