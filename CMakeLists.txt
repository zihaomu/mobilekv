cmake_minimum_required(VERSION 3.16)

project(mobilekv LANGUAGES CXX)

option(MOBILEKV_BUILD_TESTS "Build mobilekv unit tests" ON)
option(MOBILEKV_BUILD_EXAMPLES "Build mobilekv examples" ON)
option(MOBILEKV_BUILD_BENCHMARKS "Build mobilekv benchmarks" ON)

add_library(mobilekv STATIC
  src/kv_cache.cpp
)

add_library(mobilekv::mobilekv ALIAS mobilekv)

target_include_directories(mobilekv
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(mobilekv PUBLIC cxx_std_17)

if(MOBILEKV_BUILD_EXAMPLES)
  add_executable(mobilekv_demo
    example/fp32_prefill_decode_example.cpp
  )
  target_link_libraries(mobilekv_demo PRIVATE mobilekv::mobilekv)

  add_executable(mobilekv_mixed_precision_demo
    example/mixed_precision_example.cpp
  )
  target_link_libraries(mobilekv_mixed_precision_demo PRIVATE mobilekv::mobilekv)

  add_executable(mobilekv_packed_layout_demo
    example/packed_layout_example.cpp
  )
  target_link_libraries(mobilekv_packed_layout_demo PRIVATE mobilekv::mobilekv)

  add_executable(mobilekv_convenience_demo
    example/convenience_api_example.cpp
  )
  target_link_libraries(mobilekv_convenience_demo PRIVATE mobilekv::mobilekv)
endif()

if(MOBILEKV_BUILD_BENCHMARKS)
  add_executable(mobilekv_benchmark
    benchmark/kv_benchmark.cpp
  )
  target_link_libraries(mobilekv_benchmark PRIVATE mobilekv::mobilekv)
endif()

include(CTest)
if(BUILD_TESTING AND MOBILEKV_BUILD_TESTS)
  add_subdirectory(test)
endif()
