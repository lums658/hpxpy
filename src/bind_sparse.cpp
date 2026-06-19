// hpxpy._core — register_sparse: CsrMatrix / DenseMatrix + the sparse factories and
// the C++-timed SpMV/SpMM benchmarks.
//
// CSR matrix + SpMV (M5a) and the dense 2-D matrix (M5b, for SpMM operands/results),
// plus csr_from/laplacian_1d/dense_zeros/dense_from and the bench_spmv/bench_spmm
// kernels (timed in C++ with the GIL released, result pre-allocated and reused).
//
// SPDX-License-Identifier: MIT

#include "array.hpp"
#include "sparse.hpp"
#include "timing.hpp"
#include "bind_fwd.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include <string>
#include <utility>

namespace nb = nanobind;
using namespace nb::literals;

using hpxpy::Array;

void register_sparse(nb::module_& m)
{
    // --- Sparse (CSR) matrix + SpMV (M5a) ---------------------------------
    nb::class_<hpxpy::CsrMatrix>(m, "CsrMatrix")
        .def_prop_ro("rows", &hpxpy::CsrMatrix::rows)
        .def_prop_ro("cols", &hpxpy::CsrMatrix::cols)
        .def_prop_ro("nnz", &hpxpy::CsrMatrix::nnz)
        .def("spmv", [](hpxpy::CsrMatrix const& a, Array const& x) {
            nb::gil_scoped_release r; return a.spmv(x);
        }, "x"_a, "Sparse matrix-vector product y = A @ x (row-parallel).")
        .def("spmm", [](hpxpy::CsrMatrix const& a, hpxpy::DenseMatrix const& b) {
            nb::gil_scoped_release r; return a.spmm(b);
        }, "b"_a, "Sparse x dense product C = A @ B (row-parallel).")
        .def("__matmul__", [](hpxpy::CsrMatrix const& a, Array const& x) {
            nb::gil_scoped_release r; return a.spmv(x);
        })
        .def("__matmul__", [](hpxpy::CsrMatrix const& a, hpxpy::DenseMatrix const& b) {
            nb::gil_scoped_release r; return a.spmm(b);
        })
        .def("__repr__", [](hpxpy::CsrMatrix const& a) {
            return "CsrMatrix(rows=" + std::to_string(a.rows()) + ", cols=" +
                   std::to_string(a.cols()) + ", nnz=" + std::to_string(a.nnz()) + ")";
        });

    // --- Dense 2-D matrix (M5b, for SpMM operands/results) ----------------
    nb::class_<hpxpy::DenseMatrix>(m, "DenseMatrix")
        .def_prop_ro("rows", &hpxpy::DenseMatrix::rows)
        .def_prop_ro("cols", &hpxpy::DenseMatrix::cols)
        .def_prop_ro("size", &hpxpy::DenseMatrix::size)
        .def("at", &hpxpy::DenseMatrix::at, "i"_a, "j"_a, "Element [i, j].")
        .def("set", &hpxpy::DenseMatrix::set, "i"_a, "j"_a, "value"_a,
             "Set element [i, j].")
        .def("__repr__", [](hpxpy::DenseMatrix const& d) {
            return "DenseMatrix(rows=" + std::to_string(d.rows()) + ", cols=" +
                   std::to_string(d.cols()) + ")";
        });
    m.def("dense_zeros", &hpxpy::dense_zeros, "rows"_a, "cols"_a,
          "Create a rows x cols dense matrix of zeros (NUMA-aware).");
    m.def("dense_from", &hpxpy::dense_from, "rows"_a, "cols"_a, "values"_a,
          "Create a rows x cols dense matrix from a row-major flat list.");

    m.def("csr_from", &hpxpy::CsrMatrix::from_csr, "rows"_a, "cols"_a,
          "row_ptr"_a, "col_idx"_a, "values"_a,
          "Build a CsrMatrix from explicit CSR arrays (row_ptr, col_idx, values).");
    m.def("laplacian_1d", &hpxpy::laplacian_1d, "n"_a,
          "1-D Laplacian CSR matrix (tridiagonal [-1, 2, -1], n x n).");

    m.def("bench_spmv", [](hpxpy::CsrMatrix const& a, Array const& x, double budget,
                           int min_reps, int max_reps) {
        nb::gil_scoped_release release;
        // Time the KERNEL (y pre-allocated, reused) so the penalty reflects the
        // wrapper, not the result allocation's per-process first-touch variance
        // (which dominates this memory-bound op at small sizes).
        Array y(a.rows(), 0.0);
        hpxpy::timing::result r = hpxpy::timing::measure(
            [&]() -> double { a.spmv_into(x, y); return y.size() ? y.data()[0] : 0.0; },
            budget, min_reps, max_reps);
        return std::make_pair(r.median_s, r.reps);
    }, "a"_a, "x"_a, "budget"_a = 0.5, "min_reps"_a = 5, "max_reps"_a = 200,
       "C++-timed median-of-times (s) for the SpMV kernel spmv_into(A, x, y).");

    m.def("bench_spmm", [](hpxpy::CsrMatrix const& a, hpxpy::DenseMatrix const& b,
                           double budget, int min_reps, int max_reps) {
        nb::gil_scoped_release release;
        hpxpy::DenseMatrix c(a.rows(), b.cols(), 0.0);    // pre-allocated, reused
        hpxpy::timing::result r = hpxpy::timing::measure(
            [&]() -> double { a.spmm_into(b, c); return c.size() ? c.at(0, 0) : 0.0; },
            budget, min_reps, max_reps);
        return std::make_pair(r.median_s, r.reps);
    }, "a"_a, "b"_a, "budget"_a = 0.5, "min_reps"_a = 5, "max_reps"_a = 200,
       "C++-timed median-of-times (s) for the SpMM kernel spmm_into(A, B, C).");
}
