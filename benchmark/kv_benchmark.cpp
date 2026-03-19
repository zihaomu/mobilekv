// ============================================================================
// Benchmark: KV Cache Storage Performance Tests
//
// This benchmark tests:
// 1. Template locate() performance
// 2. Sequential read/write performance
// 3. Random access performance
// 4. Memory allocation performance
// 5. Multi-layer operations
// ============================================================================

#include "mobilekv/kv_cache.h"
#include "mobilekv/kv_cache_debug.h"

#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <iomanip>
#include <cstring>
#include <algorithm>

using namespace mobilekv;

// ============================================================================
// Benchmark Utilities
// ============================================================================

class BenchmarkTimer {
public:
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }

    double stop() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        return duration.count() / 1000.0;  // return milliseconds
    }

private:
    std::chrono::high_resolution_clock::time_point start_time;
};

struct BenchmarkResult {
    std::string name;
    double time_ms;
    size_t operations;
    double ops_per_sec;
    double ns_per_op;
};

// ============================================================================
// Benchmark Configurations
// ============================================================================

constexpr uint32_t BENCH_NUM_LAYERS = 12;
constexpr uint32_t BENCH_NUM_HEADS = 32;
constexpr uint32_t BENCH_HEAD_DIM = 128;
constexpr uint32_t BENCH_MAX_SEQ = 2048;

// ============================================================================
// Benchmark Tests
// ============================================================================

void benchmark_template_locate(BenchmarkResult& result) {
    auto templ = std::make_shared<PlainKVTemplate<ScalarType::FP16>>(
        BENCH_NUM_HEADS, BENCH_HEAD_DIM, 1, "bench_fp16");

    const size_t num_ops = 1000000;
    LogicalCoord coord;

    BenchmarkTimer timer;
    timer.start();

    for (size_t i = 0; i < num_ops; ++i) {
        coord.seq = i % BENCH_MAX_SEQ;
        coord.head = i % BENCH_NUM_HEADS;
        coord.dim = i % BENCH_HEAD_DIM;
        auto addr = templ->locate(coord);
        (void)addr;
    }

    result.time_ms = timer.stop();
    result.operations = num_ops;
    result.ops_per_sec = result.operations / (result.time_ms / 1000.0);
    result.ns_per_op = (result.time_ms * 1000000.0) / result.operations;
}

void benchmark_sequential_write(BenchmarkResult& result) {
    KVCacheStorageBuilder builder;
    auto templ = std::make_shared<PlainKVTemplate<ScalarType::FP32>>(
        BENCH_NUM_HEADS, BENCH_HEAD_DIM, 1, "bench_fp32");
    builder.add_template(templ);
    builder.add_layer(0, 1, 1, BENCH_MAX_SEQ);

    auto storage = builder.build();
    auto& layer = storage->layer(0);
    auto& k_plane = layer.plane(PlaneKind::K);

    const size_t num_tokens = 1024;
    k_plane.resize_seq(num_tokens);

    // 预分配数据
    std::vector<float> write_data(BENCH_NUM_HEADS * BENCH_HEAD_DIM, 1.0f);

    BenchmarkTimer timer;
    timer.start();

    float* k_ptr = static_cast<float*>(k_plane.data());
    for (uint32_t token = 0; token < num_tokens; ++token) {
        size_t offset = token * BENCH_NUM_HEADS * BENCH_HEAD_DIM;
        std::memcpy(k_ptr + offset, write_data.data(), write_data.size() * sizeof(float));
    }

    result.time_ms = timer.stop();
    result.operations = num_tokens * BENCH_NUM_HEADS * BENCH_HEAD_DIM;
    result.ops_per_sec = result.operations / (result.time_ms / 1000.0);
    result.ns_per_op = (result.time_ms * 1000000.0) / result.operations;
}

void benchmark_sequential_read(BenchmarkResult& result) {
    KVCacheStorageBuilder builder;
    auto templ = std::make_shared<PlainKVTemplate<ScalarType::FP32>>(
        BENCH_NUM_HEADS, BENCH_HEAD_DIM, 1, "bench_fp32");
    builder.add_template(templ);
    builder.add_layer(0, 1, 1, BENCH_MAX_SEQ);

    auto storage = builder.build();
    auto& layer = storage->layer(0);
    auto& k_plane = layer.plane(PlaneKind::K);

    const size_t num_tokens = 1024;
    k_plane.resize_seq(num_tokens);

    // 预先写入数据
    float* k_ptr = static_cast<float*>(k_plane.data());
    for (size_t i = 0; i < k_plane.stats().bytes_allocated / sizeof(float); ++i) {
        k_ptr[i] = static_cast<float>(i);
    }

    volatile float sum = 0;  // prevent optimization

    BenchmarkTimer timer;
    timer.start();

    for (uint32_t token = 0; token < num_tokens; ++token) {
        size_t offset = token * BENCH_NUM_HEADS * BENCH_HEAD_DIM;
        for (uint32_t i = 0; i < BENCH_NUM_HEADS * BENCH_HEAD_DIM; ++i) {
            sum += k_ptr[offset + i];
        }
    }

    result.time_ms = timer.stop();
    result.operations = num_tokens * BENCH_NUM_HEADS * BENCH_HEAD_DIM;
    result.ops_per_sec = result.operations / (result.time_ms / 1000.0);
    result.ns_per_op = (result.time_ms * 1000000.0) / result.operations;

    (void)sum;  // use the sum
}

void benchmark_random_access(BenchmarkResult& result) {
    KVCacheStorageBuilder builder;
    auto templ = std::make_shared<PlainKVTemplate<ScalarType::FP32>>(
        BENCH_NUM_HEADS, BENCH_HEAD_DIM, 1, "bench_fp32");
    builder.add_template(templ);
    builder.add_layer(0, 1, 1, BENCH_MAX_SEQ);

    auto storage = builder.build();
    auto& layer = storage->layer(0);
    auto& k_plane = layer.plane(PlaneKind::K);

    const size_t num_accesses = 100000;
    k_plane.resize_seq(BENCH_MAX_SEQ);

    // 预先写入数据
    float* k_ptr = static_cast<float*>(k_plane.data());
    for (size_t i = 0; i < k_plane.stats().bytes_allocated / sizeof(float); ++i) {
        k_ptr[i] = static_cast<float>(i);
    }

    // 生成随机坐标
    std::vector<LogicalCoord> coords(num_accesses);
    std::mt19937 rng(42);
    std::uniform_int_distribution<uint32_t> seq_dist(0, BENCH_MAX_SEQ - 1);
    std::uniform_int_distribution<uint32_t> head_dist(0, BENCH_NUM_HEADS - 1);
    std::uniform_int_distribution<uint32_t> dim_dist(0, BENCH_HEAD_DIM - 1);

    for (size_t i = 0; i < num_accesses; ++i) {
        coords[i] = LogicalCoord(0, seq_dist(rng), head_dist(rng), dim_dist(rng));
    }

    volatile float sum = 0;

    BenchmarkTimer timer;
    timer.start();

    for (size_t i = 0; i < num_accesses; ++i) {
        auto addr = k_plane.locate(coords[i]);
        if (addr.valid) {
            sum += k_ptr[addr.byte_offset / sizeof(float)];
        }
    }

    result.time_ms = timer.stop();
    result.operations = num_accesses;
    result.ops_per_sec = result.operations / (result.time_ms / 1000.0);
    result.ns_per_op = (result.time_ms * 1000000.0) / result.operations;

    (void)sum;
}

void benchmark_multi_layer_append_ring(BenchmarkResult& result) {
    KVCacheStorageBuilder builder;
    // 端侧默认：4参数add_layer走固定窗口ring，避免decode扩容抖动。
    builder.config({64, false, BENCH_MAX_SEQ});
    auto templ = std::make_shared<PlainKVTemplate<ScalarType::FP16>>(
        BENCH_NUM_HEADS, BENCH_HEAD_DIM, 1, "bench_fp16");
    builder.add_template(templ);

    for (uint32_t layer = 0; layer < BENCH_NUM_LAYERS; ++layer) {
        builder.add_layer(layer, 1, 1, BENCH_MAX_SEQ);
    }

    auto storage = builder.build();

    const uint32_t num_tokens = 512;
    const uint32_t num_iterations = 100;

    BenchmarkTimer timer;
    timer.start();

    for (uint32_t iter = 0; iter < num_iterations; ++iter) {
        storage->append_all(num_tokens);
    }

    result.time_ms = timer.stop();
    result.operations = num_iterations * num_tokens * BENCH_NUM_LAYERS * 2;  // K and V
    result.ops_per_sec = result.operations / (result.time_ms / 1000.0);
    result.ns_per_op = (result.time_ms * 1000000.0) / result.operations;
}

void benchmark_multi_layer_append_growth(BenchmarkResult& result) {
    KVCacheStorageBuilder builder;
    auto templ = std::make_shared<PlainKVTemplate<ScalarType::FP16>>(
        BENCH_NUM_HEADS, BENCH_HEAD_DIM, 1, "bench_fp16_growth");
    builder.add_template(templ);

    for (uint32_t layer = 0; layer < BENCH_NUM_LAYERS; ++layer) {
        // 4参数接口 + default_max_seq_capacity=0 => 非ring增长模式
        builder.add_layer(layer, 1, 1, BENCH_MAX_SEQ);
    }

    auto storage = builder.build();

    // 为了可观测扩容而不让benchmark耗时过长，使用小规模压力参数
    const uint32_t num_tokens = 128;
    const uint32_t num_iterations = 24;  // final length = 3072 (> 2048)

    BenchmarkTimer timer;
    timer.start();

    for (uint32_t iter = 0; iter < num_iterations; ++iter) {
        storage->append_all(num_tokens);
    }

    result.time_ms = timer.stop();
    result.operations = num_iterations * num_tokens * BENCH_NUM_LAYERS * 2;  // K and V
    result.ops_per_sec = result.operations / (result.time_ms / 1000.0);
    result.ns_per_op = (result.time_ms * 1000000.0) / result.operations;
}

void benchmark_memory_allocation(BenchmarkResult& result) {
    auto templ = std::make_shared<PlainKVTemplate<ScalarType::FP32>>(
        BENCH_NUM_HEADS, BENCH_HEAD_DIM, 1, "bench_fp32");

    const size_t num_allocations = 1000;
    std::vector<std::unique_ptr<float[]>> allocations;

    BenchmarkTimer timer;
    timer.start();

    for (size_t i = 0; i < num_allocations; ++i) {
        size_t bytes = templ->bytes_for_capacity(1024);
        auto ptr = std::unique_ptr<float[]>(static_cast<float*>(
            aligned_alloc(64, bytes)));
        allocations.push_back(std::move(ptr));
    }

    result.time_ms = timer.stop();

    // 清理
    allocations.clear();

    result.operations = num_allocations;
    result.ops_per_sec = result.operations / (result.time_ms / 1000.0);
    result.ns_per_op = (result.time_ms * 1000000.0) / result.operations;
}

void benchmark_resize_operations(BenchmarkResult& result) {
    KVCacheStorageBuilder builder;
    auto templ = std::make_shared<PlainKVTemplate<ScalarType::FP16>>(
        BENCH_NUM_HEADS, BENCH_HEAD_DIM, 1, "bench_fp16");
    builder.add_template(templ);
    builder.add_layer(0, 1, 1, 0);  // Start with 0 capacity

    auto storage = builder.build();
    auto& layer = storage->layer(0);
    auto& k_plane = layer.plane(PlaneKind::K);

    const uint32_t num_resizes = 1000;

    BenchmarkTimer timer;
    timer.start();

    for (uint32_t i = 1; i <= num_resizes; ++i) {
        k_plane.resize_seq(i);
    }

    result.time_ms = timer.stop();
    result.operations = num_resizes;
    result.ops_per_sec = result.operations / (result.time_ms / 1000.0);
    result.ns_per_op = (result.time_ms * 1000000.0) / result.operations;
}

void benchmark_acquire_view(BenchmarkResult& result) {
    KVCacheStorageBuilder builder;
    auto templ = std::make_shared<PlainKVTemplate<ScalarType::FP32>>(
        8, 32, 1, "bench_fp32");
    builder.add_template(templ);
    builder.add_layer(0, 1, 1, 1024);

    auto storage = builder.build();
    auto& layer = storage->layer(0);
    auto& k_plane = layer.plane(PlaneKind::K);

    k_plane.resize_seq(512);

    const size_t num_views = 100000;

    BenchmarkTimer timer;
    timer.start();

    for (size_t i = 0; i < num_views; ++i) {
        uint32_t seq = i % 256;
        auto view = k_plane.acquire_seq_view(seq, 32, AccessMode::ReadWrite);
        k_plane.release_seq_view(view);
    }

    result.time_ms = timer.stop();
    result.operations = num_views;
    result.ops_per_sec = result.operations / (result.time_ms / 1000.0);
    result.ns_per_op = (result.time_ms * 1000000.0) / result.operations;
}

void benchmark_mixed_precision_ring(BenchmarkResult& result) {
    KVCacheStorageBuilder builder;
    builder.config({64, false, BENCH_MAX_SEQ});

    auto fp32_templ = std::make_shared<PlainKVTemplate<ScalarType::FP32>>(16, 64, 1, "fp32");
    auto fp16_templ = std::make_shared<PlainKVTemplate<ScalarType::FP16>>(16, 64, 2, "fp16");
    auto int8_templ = std::make_shared<PlainKVTemplate<ScalarType::INT8>>(16, 64, 3, "int8");

    builder.add_template(fp32_templ);
    builder.add_template(fp16_templ);
    builder.add_template(int8_templ);

    builder.add_layer(0, 1, 1, 1024);
    builder.add_layer(1, 2, 2, 1024);
    builder.add_layer(2, 3, 3, 1024);

    auto storage = builder.build();

    const uint32_t num_ops = 10000;

    BenchmarkTimer timer;
    timer.start();

    for (uint32_t i = 0; i < num_ops; ++i) {
        storage->append_all(1);
    }

    result.time_ms = timer.stop();
    result.operations = num_ops * 3 * 2;  // 3 layers, K and V each
    result.ops_per_sec = result.operations / (result.time_ms / 1000.0);
    result.ns_per_op = (result.time_ms * 1000000.0) / result.operations;
}

void benchmark_mixed_precision_growth(BenchmarkResult& result) {
    KVCacheStorageBuilder builder;

    auto fp32_templ = std::make_shared<PlainKVTemplate<ScalarType::FP32>>(16, 64, 1, "fp32");
    auto fp16_templ = std::make_shared<PlainKVTemplate<ScalarType::FP16>>(16, 64, 2, "fp16");
    auto int8_templ = std::make_shared<PlainKVTemplate<ScalarType::INT8>>(16, 64, 3, "int8");

    builder.add_template(fp32_templ);
    builder.add_template(fp16_templ);
    builder.add_template(int8_templ);

    // 非ring增长模式：默认max_seq_capacity=0
    builder.add_layer(0, 1, 1, 1024);
    builder.add_layer(1, 2, 2, 1024);
    builder.add_layer(2, 3, 3, 1024);

    auto storage = builder.build();

    const uint32_t num_ops = 1500;  // 触发扩容但保持可接受运行时间

    BenchmarkTimer timer;
    timer.start();

    for (uint32_t i = 0; i < num_ops; ++i) {
        storage->append_all(1);
    }

    result.time_ms = timer.stop();
    result.operations = num_ops * 3 * 2;  // 3 layers, K and V each
    result.ops_per_sec = result.operations / (result.time_ms / 1000.0);
    result.ns_per_op = (result.time_ms * 1000000.0) / result.operations;
}

// ============================================================================
// Main
// ============================================================================

void print_result(const BenchmarkResult& r) {
    std::cout << std::left << std::setw(30) << r.name
              << std::right << std::setw(12) << std::fixed << std::setprecision(2)
              << r.time_ms << " ms"
              << std::setw(15) << r.operations
              << std::setw(15) << static_cast<uint64_t>(r.ops_per_sec)
              << std::setw(15) << std::fixed << std::setprecision(1)
              << r.ns_per_op << " ns/op" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "   MobileKV Benchmark Suite" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Layers: " << BENCH_NUM_LAYERS << std::endl;
    std::cout << "  Heads: " << BENCH_NUM_HEADS << std::endl;
    std::cout << "  Head Dim: " << BENCH_HEAD_DIM << std::endl;
    std::cout << "  Max Seq: " << BENCH_MAX_SEQ << std::endl;
    std::cout << std::endl;

    std::cout << std::left << std::setw(30) << "Benchmark"
              << std::right << std::setw(15) << "Time"
              << std::setw(18) << "Operations"
              << std::setw(18) << "Ops/sec"
              << std::setw(18) << "ns/op" << std::endl;
    std::cout << std::string(100, '-') << std::endl;

    // Run benchmarks
    {
        BenchmarkResult r;
        r.name = "Template Locate";
        benchmark_template_locate(r);
        print_result(r);
    }

    {
        BenchmarkResult r;
        r.name = "Sequential Write";
        benchmark_sequential_write(r);
        print_result(r);
    }

    {
        BenchmarkResult r;
        r.name = "Sequential Read";
        benchmark_sequential_read(r);
        print_result(r);
    }

    {
        BenchmarkResult r;
        r.name = "Random Access";
        benchmark_random_access(r);
        print_result(r);
    }

    {
        BenchmarkResult r;
        r.name = "Multi-layer Append (Ring)";
        benchmark_multi_layer_append_ring(r);
        print_result(r);
    }

    {
        BenchmarkResult r;
        r.name = "Multi-layer Append (Growth)";
        benchmark_multi_layer_append_growth(r);
        print_result(r);
    }

    {
        BenchmarkResult r;
        r.name = "Memory Allocation";
        benchmark_memory_allocation(r);
        print_result(r);
    }

    {
        BenchmarkResult r;
        r.name = "Resize Operations";
        benchmark_resize_operations(r);
        print_result(r);
    }

    {
        BenchmarkResult r;
        r.name = "Acquire/Release View";
        benchmark_acquire_view(r);
        print_result(r);
    }

    {
        BenchmarkResult r;
        r.name = "Mixed Precision (Ring)";
        benchmark_mixed_precision_ring(r);
        print_result(r);
    }

    {
        BenchmarkResult r;
        r.name = "Mixed Precision (Growth)";
        benchmark_mixed_precision_growth(r);
        print_result(r);
    }

    std::cout << std::string(100, '-') << std::endl;
    std::cout << "Benchmark completed." << std::endl;

    return 0;
}
