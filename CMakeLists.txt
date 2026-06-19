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
# compute templates inline into THESE translation units, and -Os de-optimizes that
# numeric kernel (~2x slower than the -O3 C++ baseline, worst single-threaded; the
# cpp_baseline/diag ladder shows the wrapper itself adds 0 penalty). NOMINSIZE
# drops the -Os entirely so the Release -O3 stands and the penalty is ~1.0. NOMINSIZE
# applies to the whole target, so EVERY bind_*.cpp below is compiled with -O3.
#
# BUILD VELOCITY: the bindings were one giant TU (src/_core.cpp #including
# array.hpp's inline templated kernels), so a single cc1plus instantiated every
# dispatch_dtype lambda x3 dtypes single-threaded (~11-14 min) while 23 cores idled.
# The .def(...) registrations now live in per-feature bind_*.cpp TUs (each
# instantiating only its kernel subset); ninja -j compiles them in parallel. Pure
# build refactor — the registered bindings are byte-for-byte identical.
nanobind_add_module(_core NOMINSIZE
  src/_core.cpp
  src/bind_runtime.cpp
  src/bind_reductions.cpp
  src/bind_elementwise.cpp
  src/bind_ufuncs.cpp
  src/bind_wave1_reductions.cpp
  src/bind_indexing.cpp
  src/bind_views.cpp
  src/bind_construction.cpp
  src/bind_bridge.cpp
  src/bind_sparse.cpp
  src/bind_bench.cpp
  src/bind_random.cpp)
target_link_libraries(_core PRIVATE
  HPX::hpx
  HPX::wrap_main
  HPX::iostreams_component)
target_compile_features(_core PRIVATE cxx_std_20)

# Precompile the heavy HPX + nanobind headers ONCE and reuse the result across every
# bind_*.cpp. The HPX headers dominate each TU's front-end cost; building the PCH once
# (instead of re-parsing hpx/hpx.hpp etc. per TU) is a large additional win on top of
# the parallelism. The PCH is compiled with the same target flags (NOMINSIZE/-O3).
target_precompile_headers(_core PRIVATE
  <hpx/hpx.hpp>
  <hpx/algorithm.hpp>
  <hpx/numeric.hpp>
  <hpx/execution.hpp>
  <hpx/hpx_start.hpp>
  <nanobind/nanobind.h>
  <nanobind/ndarray.h>
  <nanobind/stl/string.h>
  <nanobind/stl/vector.h>
  <nanobind/stl/pair.h>
  <nanobind/stl/tuple.h>)

install(TARGETS _core LIBRARY DESTINATION hpxpy)
