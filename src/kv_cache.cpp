#include "mobilekv/kv_cache.h"
#include <cstdlib>
#include <cstring>
#include <memory>
#include <algorithm>
#include <cctype>
#include <fstream>
#include <map>
#include <unordered_map>
#include <sstream>
#include <set>

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

std::string trim_copy(const std::string& s) {
    size_t begin = 0;
    size_t end = s.size();
    while (begin < end && std::isspace(static_cast<unsigned char>(s[begin]))) {
        ++begin;
    }
    while (end > begin && std::isspace(static_cast<unsigned char>(s[end - 1]))) {
        --end;
    }
    return s.substr(begin, end - begin);
}

std::string to_lower_copy(std::string s) {
    for (char& ch : s) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return s;
}

std::vector<std::string> split_whitespace(const std::string& line) {
    std::vector<std::string> out;
    std::istringstream iss(line);
    std::string token;
    while (iss >> token) {
        out.push_back(token);
    }
    return out;
}

void set_error(std::string* error_message, const std::string& message) {
    if (error_message) {
        *error_message = message;
    }
}

bool parse_uint32_value(const std::string& text, uint32_t& out) {
    try {
        size_t pos = 0;
        unsigned long value = std::stoul(text, &pos, 10);
        if (pos != text.size() || value > UINT32_MAX) {
            return false;
        }
        out = static_cast<uint32_t>(value);
        return true;
    } catch (...) {
        return false;
    }
}

bool parse_size_t_value(const std::string& text, size_t& out) {
    try {
        size_t pos = 0;
        unsigned long long value = std::stoull(text, &pos, 10);
        if (pos != text.size()) {
            return false;
        }
        out = static_cast<size_t>(value);
        return true;
    } catch (...) {
        return false;
    }
}

bool parse_bool_value(const std::string& text, bool& out) {
    std::string lower = to_lower_copy(text);
    if (lower == "true" || lower == "1" || lower == "yes") {
        out = true;
        return true;
    }
    if (lower == "false" || lower == "0" || lower == "no") {
        out = false;
        return true;
    }
    return false;
}

bool parse_scalar_type_value(const std::string& text, ScalarType& out) {
    std::string lower = to_lower_copy(text);
    if (lower == "fp32") { out = ScalarType::FP32; return true; }
    if (lower == "fp16") { out = ScalarType::FP16; return true; }
    if (lower == "bf16") { out = ScalarType::BF16; return true; }
    if (lower == "int8") { out = ScalarType::INT8; return true; }
    if (lower == "uint8") { out = ScalarType::UINT8; return true; }
    if (lower == "int16") { out = ScalarType::INT16; return true; }
    if (lower == "custom") { out = ScalarType::CUSTOM; return true; }
    return false;
}

bool parse_plane_type_token(
    const std::string& value,
    PlaneInitConfig& cfg,
    std::string* error_message
) {
    ScalarType scalar;
    if (parse_scalar_type_value(value, scalar)) {
        if (scalar == ScalarType::CUSTOM) {
            set_error(error_message,
                      "generic 'custom' is not allowed; use a registered named type "
                      "(for example int8_pack4)");
            return false;
        }
        cfg.set_builtin_type(scalar);
        return true;
    }

    const std::string named = to_lower_copy(trim_copy(value));
    if (named.empty()) {
        set_error(error_message, "invalid empty type token");
        return false;
    }
    cfg.set_named_type(named);
    return true;
}

bool parse_layer_selector(const std::string& selector, uint32_t& begin, uint32_t& end) {
    const size_t dash = selector.find('-');
    if (dash == std::string::npos) {
        if (!parse_uint32_value(selector, begin)) {
            return false;
        }
        end = begin;
        return true;
    }
    if (!parse_uint32_value(selector.substr(0, dash), begin)) {
        return false;
    }
    if (!parse_uint32_value(selector.substr(dash + 1), end)) {
        return false;
    }
    return begin <= end;
}

bool parse_key_value_tokens(
    const std::vector<std::string>& tokens,
    size_t begin_index,
    std::unordered_map<std::string, std::string>& out,
    std::string* error_message
) {
    for (size_t i = begin_index; i < tokens.size(); ++i) {
        const std::string& token = tokens[i];
        const size_t eq = token.find('=');
        if (eq == std::string::npos) {
            set_error(error_message, "expected key=value token but got '" + token + "'");
            return false;
        }
        std::string key = to_lower_copy(trim_copy(token.substr(0, eq)));
        std::string value = trim_copy(token.substr(eq + 1));
        if (key.empty() || value.empty()) {
            set_error(error_message, "invalid key=value token '" + token + "'");
            return false;
        }
        out[key] = value;
    }
    return true;
}

bool apply_plane_overrides(
    const std::unordered_map<std::string, std::string>& kv,
    PlaneInitConfig& k_cfg,
    PlaneInitConfig& v_cfg,
    std::string* error_message
) {
    for (const auto& item : kv) {
        const std::string& key = item.first;
        const std::string& value = item.second;

        if (key == "k_type" || key == "k_dtype") {
            if (!parse_plane_type_token(value, k_cfg, error_message)) {
                if (error_message && error_message->empty()) {
                    set_error(error_message, "invalid k_type value '" + value + "'");
                }
                return false;
            }
            continue;
        }
        if (key == "v_type" || key == "v_dtype") {
            if (!parse_plane_type_token(value, v_cfg, error_message)) {
                if (error_message && error_message->empty()) {
                    set_error(error_message, "invalid v_type value '" + value + "'");
                }
                return false;
            }
            continue;
        }
        if (key == "type" || key == "dtype") {
            if (!parse_plane_type_token(value, k_cfg, error_message)) {
                if (error_message && error_message->empty()) {
                    set_error(error_message, "invalid dtype value '" + value + "'");
                }
                return false;
            }
            if (!parse_plane_type_token(value, v_cfg, error_message)) {
                if (error_message && error_message->empty()) {
                    set_error(error_message, "invalid dtype value '" + value + "'");
                }
                return false;
            }
            continue;
        }

        if (key == "initial" || key == "initial_seq_capacity" || key == "initial_seq_len") {
            uint32_t parsed = 0;
            if (!parse_uint32_value(value, parsed)) {
                set_error(error_message, "invalid initial value '" + value + "'");
                return false;
            }
            k_cfg.initial_seq_capacity = parsed;
            v_cfg.initial_seq_capacity = parsed;
            continue;
        }
        if (key == "initial_k" || key == "k_initial" || key == "k_init" ||
            key == "k_initial_seq_capacity" || key == "k_initial_seq_len") {
            uint32_t parsed = 0;
            if (!parse_uint32_value(value, parsed)) {
                set_error(error_message, "invalid k initial value '" + value + "'");
                return false;
            }
            k_cfg.initial_seq_capacity = parsed;
            continue;
        }
        if (key == "initial_v" || key == "v_initial" || key == "v_init" ||
            key == "v_initial_seq_capacity" || key == "v_initial_seq_len") {
            uint32_t parsed = 0;
            if (!parse_uint32_value(value, parsed)) {
                set_error(error_message, "invalid v initial value '" + value + "'");
                return false;
            }
            v_cfg.initial_seq_capacity = parsed;
            continue;
        }

        if (key == "max" || key == "max_seq_capacity" || key == "max_seq_len") {
            uint32_t parsed = 0;
            if (!parse_uint32_value(value, parsed)) {
                set_error(error_message, "invalid max value '" + value + "'");
                return false;
            }
            k_cfg.max_seq_capacity = parsed;
            v_cfg.max_seq_capacity = parsed;
            continue;
        }
        if (key == "max_k" || key == "k_max" || key == "k_max_seq_capacity" || key == "k_max_seq_len") {
            uint32_t parsed = 0;
            if (!parse_uint32_value(value, parsed)) {
                set_error(error_message, "invalid k max value '" + value + "'");
                return false;
            }
            k_cfg.max_seq_capacity = parsed;
            continue;
        }
        if (key == "max_v" || key == "v_max" || key == "v_max_seq_capacity" || key == "v_max_seq_len") {
            uint32_t parsed = 0;
            if (!parse_uint32_value(value, parsed)) {
                set_error(error_message, "invalid v max value '" + value + "'");
                return false;
            }
            v_cfg.max_seq_capacity = parsed;
            continue;
        }

        set_error(error_message, "unknown layer/defaults key '" + key + "'");
        return false;
    }

    return true;
}

struct LayerRule {
    uint32_t begin = 0;
    uint32_t end = 0;
    std::unordered_map<std::string, std::string> kv;
    size_t line_no = 0;
};

struct PlaneRule {
    std::unordered_map<std::string, std::string> kv;
    size_t line_no = 0;
};

bool append_selector_rule(
    const std::vector<std::string>& tokens,
    size_t selector_index,
    size_t kv_begin_index,
    size_t line_no,
    std::vector<LayerRule>& out_rules,
    std::string* error_message
) {
    if (tokens.size() <= selector_index) {
        set_error(error_message, "line " + std::to_string(line_no) + ": missing layer selector");
        return false;
    }

    LayerRule rule;
    if (!parse_layer_selector(tokens[selector_index], rule.begin, rule.end)) {
        set_error(error_message, "line " + std::to_string(line_no) +
            ": invalid layer selector '" + tokens[selector_index] + "'");
        return false;
    }
    rule.line_no = line_no;

    std::string local_error;
    if (!parse_key_value_tokens(tokens, kv_begin_index, rule.kv, &local_error)) {
        set_error(error_message, "line " + std::to_string(line_no) + ": " + local_error);
        return false;
    }

    out_rules.push_back(std::move(rule));
    return true;
}

void collect_layer_ids_from_rules(
    const std::vector<LayerRule>& rules,
    std::set<LayerId>& layer_ids
) {
    for (const LayerRule& rule : rules) {
        uint32_t layer = rule.begin;
        while (true) {
            layer_ids.insert(layer);
            if (layer == rule.end) {
                break;
            }
            ++layer;
        }
    }
}

bool apply_rules_for_layer(
    LayerId layer_id,
    const std::vector<LayerRule>& rules,
    PlaneInitConfig& k_cfg,
    PlaneInitConfig& v_cfg,
    std::string* error_message
) {
    for (const LayerRule& rule : rules) {
        if (layer_id < rule.begin || layer_id > rule.end) {
            continue;
        }
        std::string local_error;
        if (!apply_plane_overrides(rule.kv, k_cfg, v_cfg, &local_error)) {
            set_error(error_message, "line " + std::to_string(rule.line_no) + ": " + local_error);
            return false;
        }
    }
    return true;
}

bool apply_plane_rules(
    const std::vector<PlaneRule>& rules,
    PlaneInitConfig& k_cfg,
    PlaneInitConfig& v_cfg,
    std::string* error_message
) {
    for (const PlaneRule& rule : rules) {
        std::string local_error;
        if (!apply_plane_overrides(rule.kv, k_cfg, v_cfg, &local_error)) {
            set_error(error_message, "line " + std::to_string(rule.line_no) + ": " + local_error);
            return false;
        }
    }
    return true;
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
                    uint32_t k_initial_capacity,
                    uint32_t v_initial_capacity,
                    uint32_t k_max_seq_capacity = 0,
                    uint32_t v_max_seq_capacity = 0)
        : layer_id_(layer_id)
        , k_plane_(layer_id, PlaneKind::K, std::move(k_templ), k_max_seq_capacity)
        , v_plane_(layer_id, PlaneKind::V, std::move(v_templ), v_max_seq_capacity) {
        if (k_initial_capacity > 0) {
            k_plane_.reserve_seq(k_initial_capacity);
        }
        if (v_initial_capacity > 0) {
            v_plane_.reserve_seq(v_initial_capacity);
        }
    }

    LayerStorageImpl(LayerId layer_id,
                    std::shared_ptr<KVTemplate> k_templ,
                    std::shared_ptr<KVTemplate> v_templ,
                    uint32_t initial_capacity,
                    uint32_t max_seq_capacity = 0)
        : LayerStorageImpl(layer_id,
                           std::move(k_templ),
                           std::move(v_templ),
                           initial_capacity,
                           initial_capacity,
                           max_seq_capacity,
                           max_seq_capacity) {}

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

        // 创建新的layer storage（K/V capacity独立生效）
        auto layer = std::make_unique<LayerStorageImpl>(
            spec.layer_id,
            k_it->second,
            v_it->second,
            spec.k_spec.initial_seq_capacity,
            spec.v_spec.initial_seq_capacity,
            spec.k_spec.max_seq_capacity,
            spec.v_spec.max_seq_capacity
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

OpaqueScalarId KVCacheStorageBuilder::register_opaque_scalar(const OpaqueScalarDesc& desc) {
    if (desc.bytes == 0 || desc.alignment == 0) {
        return 0;
    }
    OpaqueScalarId id = next_opaque_scalar_id_++;
    opaque_scalars_[id] = desc;
    return id;
}

const OpaqueScalarDesc* KVCacheStorageBuilder::find_opaque_scalar(OpaqueScalarId id) const {
    auto it = opaque_scalars_.find(id);
    if (it == opaque_scalars_.end()) {
        return nullptr;
    }
    return &it->second;
}

std::shared_ptr<KVTemplate> KVCacheStorageBuilder::make_dim_block_template(
    uint32_t num_heads,
    uint32_t dim_blocks,
    OpaqueScalarId scalar_id,
    TemplateId id,
    const std::string& name
) const {
    const OpaqueScalarDesc* desc = find_opaque_scalar(scalar_id);
    if (!desc) {
        return nullptr;
    }
    return std::make_shared<DimBlockKVTemplate>(
        num_heads, dim_blocks, desc->bytes, id, name, desc->alignment);
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
    return add_layer(layer, k_template, v_template,
                     initial_seq_capacity, initial_seq_capacity,
                     max_seq_capacity, max_seq_capacity);
}

KVCacheStorageBuilder& KVCacheStorageBuilder::add_layer(
    LayerId layer,
    TemplateId k_template,
    TemplateId v_template,
    uint32_t k_initial_seq_capacity,
    uint32_t v_initial_seq_capacity,
    uint32_t k_max_seq_capacity,
    uint32_t v_max_seq_capacity
) {
    layers_.emplace_back(
        layer,
        k_template,
        v_template,
        k_initial_seq_capacity,
        v_initial_seq_capacity,
        k_max_seq_capacity,
        v_max_seq_capacity
    );
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

std::unique_ptr<KVCacheStorage> create_storage_from_init_config(
    const StorageInitConfig& config
) {
    return create_storage_from_init_config(config, nullptr, nullptr);
}

std::unique_ptr<KVCacheStorage> create_storage_from_init_config(
    const StorageInitConfig& config,
    const ConfigTypeRegistry* type_registry,
    std::string* error_message
) {
    if (config.num_heads == 0 || config.head_dim == 0 || config.layers.empty()) {
        set_error(error_message, "invalid init config: model/layers are empty");
        return nullptr;
    }

    KVCacheStorageBuilder builder;
    builder.config(config.storage_config);

    std::unordered_map<ScalarType, TemplateId> type_to_id;
    std::unordered_map<std::string, TemplateId> named_type_to_id;
    TemplateId next_id = 1;

    auto get_or_create_plain_template_id = [&](ScalarType type, const char* prefix) -> TemplateId {
        if (type == ScalarType::CUSTOM) {
            set_error(error_message,
                      "generic ScalarType::CUSTOM is not supported in cfg/init path; "
                      "use PlaneInitConfig::set_named_type(...) with a registered type");
            return 0;
        }
        auto it = type_to_id.find(type);
        if (it != type_to_id.end()) {
            return it->second;
        }
        try {
            const TemplateId id = next_id++;
            std::string name = std::string(prefix) + "_" + detail::scalar_type_name(type);
            builder.add_template(detail::make_plain_template_for_type(
                type, config.num_heads, config.head_dim, id, name));
            type_to_id[type] = id;
            return id;
        } catch (const std::exception& e) {
            set_error(error_message, std::string("failed to create plain template: ") + e.what());
            return 0;
        }
    };

    auto get_or_create_named_template_id =
        [&](const std::string& raw_named_type, const char* prefix) -> TemplateId {
        const std::string named_type = to_lower_copy(trim_copy(raw_named_type));
        if (named_type.empty()) {
            set_error(error_message, "named custom type is empty");
            return 0;
        }
        auto hit = named_type_to_id.find(named_type);
        if (hit != named_type_to_id.end()) {
            return hit->second;
        }

        if (!type_registry) {
            set_error(error_message, "named custom type '" + named_type +
                "' requested but no ConfigTypeRegistry provided");
            return 0;
        }

        const NamedCustomTypeDesc* desc = type_registry->find_type(named_type);
        if (!desc) {
            set_error(error_message, "named custom type '" + named_type +
                "' is not registered in ConfigTypeRegistry");
            return 0;
        }
        if (desc->block_bytes == 0 || desc->alignment == 0 || desc->dim_pack_factor == 0) {
            set_error(error_message, "invalid registered custom type '" + named_type + "'");
            return 0;
        }
        if (config.head_dim % desc->dim_pack_factor != 0) {
            set_error(error_message, "head_dim " + std::to_string(config.head_dim) +
                " is not divisible by dim_pack_factor " + std::to_string(desc->dim_pack_factor) +
                " for type '" + named_type + "'");
            return 0;
        }

        OpaqueScalarId scalar_id = builder.register_opaque_scalar(
            {desc->type_name, desc->block_bytes, desc->alignment});
        if (scalar_id == 0) {
            set_error(error_message, "failed to register opaque scalar for type '" + named_type + "'");
            return 0;
        }

        const uint32_t dim_blocks = config.head_dim / desc->dim_pack_factor;
        const TemplateId id = next_id++;
        auto templ = builder.make_dim_block_template(
            config.num_heads,
            dim_blocks,
            scalar_id,
            id,
            std::string(prefix) + "_" + named_type
        );
        if (!templ) {
            set_error(error_message, "failed to create dim-block template for type '" + named_type + "'");
            return 0;
        }
        builder.add_template(templ);
        named_type_to_id[named_type] = id;
        return id;
    };

    for (const LayerInitConfig& layer : config.layers) {
        TemplateId k_template = 0;
        TemplateId v_template = 0;

        if (layer.k.use_named_type) {
            k_template = get_or_create_named_template_id(layer.k.named_type, "k");
        } else {
            k_template = get_or_create_plain_template_id(layer.k.scalar_type, "k");
        }
        if (k_template == 0) {
            return nullptr;
        }

        if (layer.v.use_named_type) {
            v_template = get_or_create_named_template_id(layer.v.named_type, "v");
        } else {
            v_template = get_or_create_plain_template_id(layer.v.scalar_type, "v");
        }
        if (v_template == 0) {
            return nullptr;
        }

        builder.add_layer(
            layer.layer_id,
            k_template,
            v_template,
            layer.k.initial_seq_capacity,
            layer.v.initial_seq_capacity,
            layer.k.max_seq_capacity,
            layer.v.max_seq_capacity
        );
    }

    auto storage = builder.build();
    if (!storage) {
        set_error(error_message, "failed to build storage from init config");
    }
    return storage;
}

bool load_storage_init_config_from_file(
    const std::string& path,
    StorageInitConfig& out_config,
    std::string* error_message
) {
    std::ifstream in(path);
    if (!in.is_open()) {
        set_error(error_message, "failed to open config file: " + path);
        return false;
    }

    StorageInitConfig parsed;
    parsed.num_heads = out_config.num_heads;
    parsed.head_dim = out_config.head_dim;
    parsed.storage_config = out_config.storage_config;
    parsed.layers.clear();

    std::vector<PlaneRule> global_defaults;
    std::vector<LayerRule> global_groups;
    std::vector<LayerRule> global_overrides;

    std::string raw_line;
    size_t line_no = 0;

    while (std::getline(in, raw_line)) {
        ++line_no;

        const size_t comment_pos = raw_line.find('#');
        const std::string no_comment =
            comment_pos == std::string::npos ? raw_line : raw_line.substr(0, comment_pos);
        const std::string line = trim_copy(no_comment);
        if (line.empty()) {
            continue;
        }

        const std::vector<std::string> tokens = split_whitespace(line);
        if (tokens.empty()) {
            continue;
        }

        const std::string directive = to_lower_copy(tokens[0]);
        std::unordered_map<std::string, std::string> kv;
        std::string local_error;

        if (directive == "model") {
            if (!parse_key_value_tokens(tokens, 1, kv, &local_error)) {
                set_error(error_message, "line " + std::to_string(line_no) + ": " + local_error);
                return false;
            }
            for (const auto& item : kv) {
                if (item.first == "num_heads") {
                    if (!parse_uint32_value(item.second, parsed.num_heads)) {
                        set_error(error_message, "line " + std::to_string(line_no) +
                            ": invalid num_heads value '" + item.second + "'");
                        return false;
                    }
                } else if (item.first == "head_dim") {
                    if (!parse_uint32_value(item.second, parsed.head_dim)) {
                        set_error(error_message, "line " + std::to_string(line_no) +
                            ": invalid head_dim value '" + item.second + "'");
                        return false;
                    }
                } else {
                    set_error(error_message, "line " + std::to_string(line_no) +
                        ": unknown model key '" + item.first + "'");
                    return false;
                }
            }
            continue;
        }

        if (directive == "storage") {
            if (!parse_key_value_tokens(tokens, 1, kv, &local_error)) {
                set_error(error_message, "line " + std::to_string(line_no) + ": " + local_error);
                return false;
            }
            for (const auto& item : kv) {
                if (item.first == "default_alignment") {
                    if (!parse_size_t_value(item.second, parsed.storage_config.default_alignment)) {
                        set_error(error_message, "line " + std::to_string(line_no) +
                            ": invalid default_alignment value '" + item.second + "'");
                        return false;
                    }
                } else if (item.first == "thread_safe") {
                    if (!parse_bool_value(item.second, parsed.storage_config.thread_safe)) {
                        set_error(error_message, "line " + std::to_string(line_no) +
                            ": invalid thread_safe value '" + item.second + "'");
                        return false;
                    }
                } else if (item.first == "default_max_seq_capacity") {
                    if (!parse_uint32_value(item.second, parsed.storage_config.default_max_seq_capacity)) {
                        set_error(error_message, "line " + std::to_string(line_no) +
                            ": invalid default_max_seq_capacity value '" + item.second + "'");
                        return false;
                    }
                } else {
                    set_error(error_message, "line " + std::to_string(line_no) +
                        ": unknown storage key '" + item.first + "'");
                    return false;
                }
            }
            continue;
        }

        if (directive == "defaults") {
            if (!parse_key_value_tokens(tokens, 1, kv, &local_error)) {
                set_error(error_message, "line " + std::to_string(line_no) + ": " + local_error);
                return false;
            }
            PlaneRule rule;
            rule.kv = std::move(kv);
            rule.line_no = line_no;
            global_defaults.push_back(std::move(rule));
            continue;
        }

        if (directive == "group") {
            if (!append_selector_rule(tokens, 1, 2, line_no, global_groups, error_message)) {
                return false;
            }
            continue;
        }

        if (directive == "override" || directive == "layer") {
            if (!append_selector_rule(tokens, 1, 2, line_no, global_overrides, error_message)) {
                return false;
            }
            continue;
        }

        set_error(error_message, "line " + std::to_string(line_no) +
            ": unknown directive '" + directive + "'");
        return false;
    }

    if (parsed.num_heads == 0 || parsed.head_dim == 0) {
        set_error(error_message, "config missing valid model num_heads/head_dim");
        return false;
    }

    std::set<LayerId> layer_ids;
    collect_layer_ids_from_rules(global_groups, layer_ids);
    collect_layer_ids_from_rules(global_overrides, layer_ids);

    if (layer_ids.empty()) {
        set_error(error_message, "config resolved no layers (add group/override/layer rules)");
        return false;
    }

    for (LayerId layer_id : layer_ids) {
        LayerInitConfig layer;
        layer.layer_id = layer_id;
        layer.k = PlaneInitConfig();
        layer.v = PlaneInitConfig();

        if (!apply_plane_rules(global_defaults, layer.k, layer.v, error_message)) {
            return false;
        }

        if (!apply_rules_for_layer(layer_id, global_groups, layer.k, layer.v, error_message)) {
            return false;
        }

        if (!apply_rules_for_layer(layer_id, global_overrides, layer.k, layer.v, error_message)) {
            return false;
        }

        // 如果未显式设置plane max，继承storage默认max
        if (layer.k.max_seq_capacity == 0 && parsed.storage_config.default_max_seq_capacity > 0) {
            layer.k.max_seq_capacity = parsed.storage_config.default_max_seq_capacity;
        }
        if (layer.v.max_seq_capacity == 0 && parsed.storage_config.default_max_seq_capacity > 0) {
            layer.v.max_seq_capacity = parsed.storage_config.default_max_seq_capacity;
        }

        if (layer.k.max_seq_capacity > 0 &&
            layer.k.initial_seq_capacity > layer.k.max_seq_capacity) {
            set_error(error_message, "layer " + std::to_string(layer_id) +
                " has invalid k initial/max capacity");
            return false;
        }
        if (layer.v.max_seq_capacity > 0 &&
            layer.v.initial_seq_capacity > layer.v.max_seq_capacity) {
            set_error(error_message, "layer " + std::to_string(layer_id) +
                " has invalid v initial/max capacity");
            return false;
        }

        parsed.layers.push_back(layer);
    }

    out_config = parsed;
    return true;
}

std::unique_ptr<KVCacheStorage> create_storage_from_config_file(
    const std::string& path,
    std::string* error_message
) {
    return create_storage_from_config_file(path, nullptr, error_message);
}

std::unique_ptr<KVCacheStorage> create_storage_from_config_file(
    const std::string& path,
    const ConfigTypeRegistry* type_registry,
    std::string* error_message
) {
    StorageInitConfig config;
    if (!load_storage_init_config_from_file(path, config, error_message)) {
        return nullptr;
    }
    auto storage = create_storage_from_init_config(config, type_registry, error_message);
    if (!storage && error_message) {
        if (error_message->empty()) {
            *error_message = "failed to build storage from parsed config";
        }
    }
    return storage;
}

}  // namespace mobilekv
