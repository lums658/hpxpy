// hpxpy._core — forward declarations of the per-TU registration functions.
//
// Each register_*() is DEFINED in its own bind_*.cpp translation unit and appends its
// .def(...) bindings to the module `m` and/or the shared Array class `cls`. _core.cpp
// (the thin orchestrator) includes this header and calls each one from NB_MODULE, so
// the bindings are identical to the old single-TU module — only the template-
// instantiation cost is now spread across parallel compiles.
//
// SPDX-License-Identifier: MIT
#pragma once

#include <nanobind/nanobind.h>

// Forward-declare Array so the register-fn signatures that take the shared class don't
// need array.hpp here (the orchestrator stays light).
namespace hpxpy { class Array; }

namespace nb = nanobind;

// Module-level free functions (no Array methods): runtime, construction, bridge, etc.
void register_runtime(nb::module_& m);
void register_construction(nb::module_& m);
void register_bridge(nb::module_& m);
void register_sparse(nb::module_& m);
void register_bench(nb::module_& m);
void register_random(nb::module_& m);

// Array-method registrations: append .def(...) to the shared nb::class_<Array> cls.
void register_reductions(nb::module_& m, nb::class_<hpxpy::Array>& cls);
void register_elementwise(nb::module_& m, nb::class_<hpxpy::Array>& cls);
void register_ufuncs(nb::module_& m, nb::class_<hpxpy::Array>& cls);
void register_wave1_reductions(nb::module_& m, nb::class_<hpxpy::Array>& cls);
void register_indexing(nb::module_& m, nb::class_<hpxpy::Array>& cls);
void register_views(nb::module_& m, nb::class_<hpxpy::Array>& cls);
