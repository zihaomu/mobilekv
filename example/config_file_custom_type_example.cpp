// ============================================================================
// Example: Build Storage With custom k_type/v_type From Config File
//
// Notes:
// - User registers named custom type in code (for example int8_pack4)
// - cfg references that name directly via k_type/v_type
// - storage builds dim-block templates based on registry metadata
// ============================================================================

#include "mobilekv/kv_cache.h"

#include <iostream>
#include <string>

using namespace mobilekv;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <config.cfg>\n"
                  << "Example: " << argv[0] << " ./example/configs/custom_type.cfg\n";
        return 1;
    }

    const std::string cfg_path = argv[1];
    std::string error;

    ConfigTypeRegistry type_registry;
    if (!type_registry.register_type({"int8_pack4", 4, 4, 4}, &error)) {
        std::cerr << "register_type(int8_pack4) failed: " << error << std::endl;
        return 1;
    }

    auto storage = create_storage_from_config_file(cfg_path, &type_registry, &error);
    if (!storage) {
        std::cerr << "create_storage_from_config_file failed: " << error << std::endl;
        return 1;
    }

    StorageInitConfig parsed;
    if (!load_storage_init_config_from_file(cfg_path, parsed, &error)) {
        std::cerr << "load_storage_init_config_from_file failed: " << error << std::endl;
        return 1;
    }

    for (const auto& layer_cfg : parsed.layers) {
        const uint32_t layer_id = layer_cfg.layer_id;
        const auto& layer = storage->layer(layer_id);
        const auto& k_templ = layer.plane(PlaneKind::K).templ();
        const auto& v_templ = layer.plane(PlaneKind::V).templ();

        std::cout << "Layer " << layer_id
                  << ": K=" << scalar_type_to_string(k_templ.config().scalar_type)
                  << " (element_size=" << k_templ.element_size()
                  << ", dim=" << k_templ.shape().head_dim << ")"
                  << ", V=" << scalar_type_to_string(v_templ.config().scalar_type)
                  << " (element_size=" << v_templ.element_size()
                  << ", dim=" << v_templ.shape().head_dim << ")"
                  << std::endl;
    }

    return 0;
}
