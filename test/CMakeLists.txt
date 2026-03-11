add_subdirectory(googletest EXCLUDE_FROM_ALL)

add_executable(kv_cache_test
  kv_cache_test.cpp
)

set_target_properties(kv_cache_test PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

target_link_libraries(kv_cache_test
  PRIVATE
    mobilekv::mobilekv
    gtest_main
)

target_compile_features(kv_cache_test PRIVATE cxx_std_17)

include(GoogleTest)
gtest_discover_tests(kv_cache_test)
