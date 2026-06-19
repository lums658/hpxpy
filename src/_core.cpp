// hpxpy._core — module orchestrator (multi-TU split).
//
// Proves the full toolchain end to end: scikit-build-core -> CMake -> nanobind,
// linked against an installed HPX, with (a) a managed HPX runtime started from
// Python and (b) a ZERO-COPY nb::ndarray fed straight into an HPX parallel
// algorithm. Everything real (the Array type, fused ops, etc.) is built on top
// of this in later milestones.
//
// BUILD VELOCITY: the bindings were one giant translation unit (_core.cpp
// #including array.hpp's inline templated kernels) so a single cc1plus instantiated
// every dispatch_dtype lambda x3 dtypes single-threaded. The .def(...) registrations
// now live in per-feature bind_*.cpp TUs (each instantiating only its kernels), and
// ninja -j compiles them in parallel. This file is the thin orchestrator: it builds
// the module + the shared Array class, then calls each register_*(); the registered
// bindings are byte-for-byte the same as before — a pure build refactor.
//
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>

#include "array.hpp"      // hpxpy::Array (the type the shared nb::class_ wraps)
#include "bind_fwd.hpp"   // register_*() forward declarations

namespace nb = nanobind;

using hpxpy::Array;

NB_MODULE(_core, m)
{
    m.doc() = "hpxpy._core — managed HPX runtime + the Array type and HPX reductions";

    // Runtime + distributed introspection / collectives (module-level functions).
    register_runtime(m);

    // The shared Array class. Each register_*(m, cls) below appends its .def(...)
    // calls to THIS object, so the resulting type is identical to the old monolith.
    nb::class_<Array> cls(m, "Array");
    register_reductions(m, cls);        // sum/min/max(+axis), dot, matmul, __matmul__
    register_elementwise(m, cls);       // add/sub/mul/div + operator forms (Array/scalar)
    register_ufuncs(m, cls);            // unary + binary math ufuncs (negative..clip)
    register_wave1_reductions(m, cls);  // mean/prod/any/all/count_nonzero + copy/sort/...
    register_indexing(m, cls);          // __getitem__/__setitem__/__len__/__repr__ + props
    register_views(m, cls);             // transpose/.T/reshape/ravel/squeeze/expand_dims

    // Module-level free functions that construct or bridge Arrays.
    register_construction(m);           // zeros/ones/full/arange/linspace/eye/empty
    register_bridge(m);                 // to_numpy/from_numpy
    register_sparse(m);                 // CsrMatrix/DenseMatrix + bench_spmv/bench_spmm
    register_bench(m);                  // bench/bench_dot/bench_binary
    register_random(m);                 // random submodule: seed/rand/randn/uniform/randint
}
