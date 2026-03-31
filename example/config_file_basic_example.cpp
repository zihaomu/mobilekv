// ============================================================================
// Example: Build Storage From Config File (Basic)
//
// This example demonstrates:
// - Reading an existing local cfg file
// - Parsing cfg into StorageInitConfig
// - Building storage from parsed config
// - Direct file-to-storage creation
// ============================================================================

#include "mobilekv/kv_cache.h"

#include <iostream>
#include <string>

using namespace mobilekv;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <config.cfg>\n"
                  << "Example: " << argv[0] << " ./example/configs/basic.cfg\n"
                  << "Example: " << argv[0] << " ./example/configs/low.cfg\n";
        return 1;
    }

    const std::string cfg_path = argv[1];

    std::string error;
    StorageInitConfig parsed;
    if (!load_storage_init_config_from_file(cfg_path, parsed, &error)) {
        std::cerr << "load_storage_init_config_from_file failed: " << error << std::endl;
        return 1;
    }

    std::cout << "Parsed " << parsed.layers.size()
              << " layers from cfg: " << cfg_path
              << std::endl;

    auto storage_from_cfg = create_storage_from_init_config(parsed);
    if (!storage_from_cfg) {
        std::cerr << "create_storage_from_init_config failed" << std::endl;
        return 1;
    }

    for (const auto& layer : parsed.layers) {
        std::cout << "Layer " << layer.layer_id
                  << ": K=" << scalar_type_to_string(layer.k.scalar_type)
                  << " (max=" << layer.k.max_seq_capacity << ")"
                  << ", V=" << scalar_type_to_string(layer.v.scalar_type)
                  << " (max=" << layer.v.max_seq_capacity << ")"
                  << std::endl;
    }

    auto storage_direct = create_storage_from_config_file(cfg_path, &error);
    if (!storage_direct) {
        std::cerr << "create_storage_from_config_file failed: " << error << std::endl;
        return 1;
    }

    std::cout << "Direct create_storage_from_config_file succeeded" << std::endl;
    return 0;
}
