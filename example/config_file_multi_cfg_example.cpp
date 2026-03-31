// ============================================================================
// Example: Switch Config By File Path
//
// This example demonstrates:
// - MobileKV only reads one cfg and executes it
// - "low/high" is user-side file selection, not MobileKV-internal semantics
// ============================================================================

#include "mobilekv/kv_cache.h"

#include <iostream>
#include <string>

using namespace mobilekv;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <cfg_a> <cfg_b>\n"
                  << "Example: " << argv[0]
                  << " ./example/configs/low.cfg ./example/configs/high.cfg\n";
        return 1;
    }

    const std::string cfg_a = argv[1];
    const std::string cfg_b = argv[2];
    std::string error;

    auto storage_a = create_storage_from_config_file(cfg_a, &error);
    if (!storage_a) {
        std::cerr << "create_storage_from_config_file(cfg_a) failed: " << error << std::endl;
        return 1;
    }
    std::cout << "[cfg_a] layer0: K="
              << scalar_type_to_string(storage_a->layer(0).plane(PlaneKind::K).templ().config().scalar_type)
              << ", V=" << scalar_type_to_string(storage_a->layer(0).plane(PlaneKind::V).templ().config().scalar_type)
              << std::endl;

    auto storage_b = create_storage_from_config_file(cfg_b, &error);
    if (!storage_b) {
        std::cerr << "create_storage_from_config_file(cfg_b) failed: " << error << std::endl;
        return 1;
    }
    std::cout << "[cfg_b] layer0: K="
              << scalar_type_to_string(storage_b->layer(0).plane(PlaneKind::K).templ().config().scalar_type)
              << ", V=" << scalar_type_to_string(storage_b->layer(0).plane(PlaneKind::V).templ().config().scalar_type)
              << std::endl;

    StorageInitConfig parsed_a;
    if (!load_storage_init_config_from_file(cfg_a, parsed_a, &error)) {
        std::cerr << "load_storage_init_config_from_file(cfg_a) failed: " << error << std::endl;
        return 1;
    }
    std::cout << "Parsed(cfg_a) layer count: " << parsed_a.layers.size() << std::endl;

    return 0;
}
