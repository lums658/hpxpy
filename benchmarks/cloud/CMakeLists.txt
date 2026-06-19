# benchmarks/cloud/CMakeLists.txt
# Build the medusa-as-cloud distributed HPX benchmark.
# Follows the same pattern as cpp_baseline/CMakeLists.txt.
# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.18...3.29)
project(hpxpy_cloud_bench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(HPX REQUIRED)
message(STATUS "cloud_bench: found HPX ${HPX_VERSION} at ${HPX_DIR}")

add_executable(cloud_bench cloud_bench.cpp)
# src/timing.hpp path (two levels up from this CMakeLists.txt)
target_include_directories(cloud_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src)
# -O3 explicitly (NOT -Os — see HPXPy penalty-method note on nanobind -Os gotcha)
target_compile_options(cloud_bench PRIVATE -O3)
target_link_libraries(cloud_bench PRIVATE HPX::hpx)
target_compile_features(cloud_bench PRIVATE cxx_std_20)
