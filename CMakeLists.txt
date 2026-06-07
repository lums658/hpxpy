# hpxpy — CMake build for the nanobind _core extension, linked against installed HPX.
# Driven by scikit-build-core (see pyproject.toml). SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.18...3.29)
project(hpxpy LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)            # HPX here is built with C++20
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# --- Python + nanobind ------------------------------------------------------
if(NOT DEV_MODULE)
  set(DEV_MODULE Development.Module)
endif()
find_package(Python 3.13 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

# --- HPX (installed; pinned) -------------------------------------------------
find_package(HPX REQUIRED)
message(STATUS "hpxpy: found HPX ${HPX_VERSION} at ${HPX_DIR}")

# --- The extension ----------------------------------------------------------
# nanobind size-optimizes extensions with -Os by default. But the HPX reduce/
# compute templates inline into THIS translation unit, and -Os de-optimizes that
# numeric kernel (~2x slower than the -O3 C++ baseline, worst single-threaded; the
# cpp_baseline/diag ladder shows the wrapper itself adds 0 penalty). NOMINSIZE
# drops the -Os entirely so the Release -O3 stands and the penalty is ~1.0.
nanobind_add_module(_core NOMINSIZE src/_core.cpp)
target_link_libraries(_core PRIVATE
  HPX::hpx
  HPX::wrap_main
  HPX::iostreams_component)
target_compile_features(_core PRIVATE cxx_std_20)

install(TARGETS _core LIBRARY DESTINATION hpxpy)
