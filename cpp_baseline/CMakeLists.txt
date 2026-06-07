# Standalone build for the C++ HPX baseline kernels (the perf ceiling). Built
# separately from the Python extension; benchmarks.runner invokes the binary via
# `sweep --baseline <path-to-bench>`. SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.18...3.29)
project(hpxpy_cpp_baseline LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)            # match the HPX install (C++20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(HPX REQUIRED)
message(STATUS "cpp_baseline: found HPX ${HPX_VERSION} at ${HPX_DIR}")

# We call hpx::init() explicitly from our own main(), so do NOT link HPX::wrap_main
# (which would hijack main). compute_local lives in HPX::hpx core.
add_executable(bench bench.cpp)
target_link_libraries(bench PRIVATE HPX::hpx)
target_compile_features(bench PRIVATE cxx_std_20)
