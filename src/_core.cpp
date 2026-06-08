// hpxpy._core — M0 substrate spike.
//
// Proves the full toolchain end to end: scikit-build-core -> CMake -> nanobind,
// linked against an installed HPX, with (a) a managed HPX runtime started from
// Python and (b) a ZERO-COPY nb::ndarray fed straight into an HPX parallel
// algorithm. Everything real (the Array type, fused ops, etc.) is built on top
// of this in later milestones.
//
// SPDX-License-Identifier: MIT

#include <hpx/hpx.hpp>
#include <hpx/hpx_start.hpp>
#include <hpx/algorithm.hpp>
#include <hpx/numeric.hpp>
#include <hpx/execution.hpp>
#include <hpx/version.hpp>

#include "array.hpp"
#include "sparse.hpp"
#include "timing.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

// ---------------------------------------------------------------------------
// Managed HPX runtime (started on a background thread; the Python main thread
// is not an HPX thread). Pattern adapted from HPyX's global_runtime_manager so
// the two projects stay runtime-compatible for the eventual merge.
//
// Thread count is passed via the command line "--hpx:threads=N" — the cfg key
// "hpx.os_threads" was found NOT to limit the pool in practice (M0 lesson from
// the prototype benchmarking), whereas the CLI option does.
// ---------------------------------------------------------------------------
namespace {

struct runtime_manager
{
    explicit runtime_manager(int num_threads) : running_(false), rts_(nullptr)
    {
        argv_storage_.emplace_back("hpxpy");
        if (num_threads > 0)
            argv_storage_.emplace_back("--hpx:threads=" + std::to_string(num_threads));
        // Avoid mmap thread-stack exhaustion (max_map_count) at high thread counts.
        argv_storage_.emplace_back("--hpx:ini=hpx.stacks.use_guard_pages=0");
        for (auto& s : argv_storage_) argv_.push_back(s.data());

        hpx::init_params params;
        hpx::function<int(int, char**)> start_fn =
            hpx::bind_front(&runtime_manager::hpx_main, this);

        if (!hpx::start(start_fn, static_cast<int>(argv_.size()), argv_.data(), params))
            std::abort();  // runtime failed to start

        std::unique_lock<std::mutex> lk(startup_mtx_);
        while (!running_) startup_cond_.wait(lk);
    }

    ~runtime_manager()
    {
        { std::lock_guard<hpx::spinlock> lk(mtx_); rts_ = nullptr; }
        cond_.notify_one();
        hpx::stop();
    }

    int hpx_main(int, char**)
    {
        rts_ = hpx::get_runtime_ptr();
        { std::lock_guard<std::mutex> lk(startup_mtx_); running_ = true; }
        startup_cond_.notify_one();
        { std::unique_lock<hpx::spinlock> lk(mtx_); if (rts_ != nullptr) cond_.wait(lk); }
        return hpx::finalize();
    }

private:
    hpx::spinlock mtx_;
    hpx::condition_variable_any cond_;
    std::mutex startup_mtx_;
    std::condition_variable startup_cond_;
    bool running_;
    hpx::runtime* rts_;
    std::vector<std::string> argv_storage_;
    std::vector<char*> argv_;
};

runtime_manager* g_rts = nullptr;

void init_runtime(int num_threads)
{
    if (g_rts == nullptr)
    {
        nb::gil_scoped_acquire acquire;
        g_rts = new runtime_manager(num_threads);
    }
}

void finalize_runtime()
{
    runtime_manager* r = g_rts;
    g_rts = nullptr;
    if (r != nullptr)
    {
        nb::gil_scoped_release release;
        delete r;
    }
}

// ---------------------------------------------------------------------------
// Array — the core data type. Defined in array.hpp as a binding-agnostic C++/HPX
// wrapper so the bindings AND the C++ abstraction-penalty diagnostic share the
// exact same code. GIL handling lives HERE, at the binding layer (see the call
// sites below), not inside the wrapper.
// ---------------------------------------------------------------------------
using hpxpy::Array;

// NumPy bridge types: a 1-D float64 C-contiguous input (writable, so a borrow can
// share mutations both ways), and a 1-D float64 output view.
using np_rw = nb::ndarray<nb::numpy, double, nb::c_contig, nb::ndim<1>>;
using np_out = nb::ndarray<nb::numpy, double, nb::ndim<1>>;

// Zero-copy NumPy view of an Array. `obj` (the Array Python object) is the ndarray's
// owner, so the HPX buffer outlives the view. Writable; works for slice views too.
np_out to_numpy_view(nb::object obj)
{
    Array& a = nb::cast<Array&>(obj);
    return np_out(a.mutable_data(), {a.size()}, obj);
}

}  // namespace

NB_MODULE(_core, m)
{
    m.doc() = "hpxpy._core — managed HPX runtime + the Array type and HPX reductions";
    m.def("init_runtime", &init_runtime, "num_threads"_a = 0,
          "Start the HPX runtime (num_threads<=0 => all cores).");
    m.def("finalize_runtime", &finalize_runtime, "Stop the HPX runtime.");
    m.def("num_worker_threads", []() { return hpx::get_num_worker_threads(); });
    m.def("hpx_version", []() { return hpx::complete_version(); });

    // Reductions release the GIL around the HPX work (the wrapper itself is
    // GIL-agnostic). std::invalid_argument from min()/max() maps to ValueError.
    nb::class_<Array>(m, "Array")
        .def_prop_ro("size", &Array::size)
        .def_prop_ro("ndim", &Array::ndim)
        .def("sum", [](Array const& a) {
            nb::gil_scoped_release release;
            return a.sum();
        }, "Parallel sum (hpx::reduce).")
        .def("min", [](Array const& a) {
            nb::gil_scoped_release release;
            return a.min();
        }, "Parallel minimum (empty -> ValueError).")
        .def("max", [](Array const& a) {
            nb::gil_scoped_release release;
            return a.max();
        }, "Parallel maximum (empty -> ValueError).")
        .def("dot", [](Array const& a, Array const& b) {
            nb::gil_scoped_release release;
            return a.dot(b);
        }, "b"_a, "Fused dot product (single-pass transform_reduce).")
        .def("copy", [](Array const& a) {
            nb::gil_scoped_release r; return a.copy();
        }, "Deep copy to a new Array (numpy a.copy()).")
        .def("sort", [](Array& a) {
            nb::gil_scoped_release r; a.sort();
        }, "Sort ascending IN PLACE (numpy a.sort(); returns None).")
        .def("is_sorted", [](Array const& a) {
            nb::gil_scoped_release r; return a.is_sorted();
        }, "True if ascending (hpx::is_sorted).")
        .def("cumsum", [](Array const& a) {
            nb::gil_scoped_release r; return a.cumsum();
        }, "Inclusive prefix sum -> new Array (numpy a.cumsum()).")
        // Element-wise binary ops -> new Array (hpx::transform). Operators and the
        // NumPy-style named methods share one kernel; GIL released around the work.
        .def("add", [](Array const& a, Array const& b) {
            nb::gil_scoped_release r; return a.add(b);
        }, "b"_a, "Element-wise a + b.")
        .def("sub", [](Array const& a, Array const& b) {
            nb::gil_scoped_release r; return a.sub(b);
        }, "b"_a, "Element-wise a - b.")
        .def("mul", [](Array const& a, Array const& b) {
            nb::gil_scoped_release r; return a.mul(b);
        }, "b"_a, "Element-wise a * b.")
        .def("div", [](Array const& a, Array const& b) {
            nb::gil_scoped_release r; return a.div(b);
        }, "b"_a, "Element-wise a / b.")
        .def("__add__", [](Array const& a, Array const& b) {
            nb::gil_scoped_release r; return a.add(b);
        })
        .def("__sub__", [](Array const& a, Array const& b) {
            nb::gil_scoped_release r; return a.sub(b);
        })
        .def("__mul__", [](Array const& a, Array const& b) {
            nb::gil_scoped_release r; return a.mul(b);
        })
        .def("__truediv__", [](Array const& a, Array const& b) {
            nb::gil_scoped_release r; return a.div(b);
        })
        // Scalar broadcast: a ⊙ s and the reflected s ⊙ a (one unary transform).
        // nanobind tries the Array×Array overloads above first, then these.
        .def("__add__", [](Array const& a, double s) {
            nb::gil_scoped_release r; return a.add_scalar(s);
        })
        .def("__radd__", [](Array const& a, double s) {
            nb::gil_scoped_release r; return a.add_scalar(s);
        })
        .def("__sub__", [](Array const& a, double s) {
            nb::gil_scoped_release r; return a.sub_scalar(s);
        })
        .def("__rsub__", [](Array const& a, double s) {
            nb::gil_scoped_release r; return a.rsub_scalar(s);    // s - a
        })
        .def("__mul__", [](Array const& a, double s) {
            nb::gil_scoped_release r; return a.mul_scalar(s);
        })
        .def("__rmul__", [](Array const& a, double s) {
            nb::gil_scoped_release r; return a.mul_scalar(s);
        })
        .def("__truediv__", [](Array const& a, double s) {
            nb::gil_scoped_release r; return a.div_scalar(s);
        })
        .def("__rtruediv__", [](Array const& a, double s) {
            nb::gil_scoped_release r; return a.rdiv_scalar(s);    // s / a
        })
        // Indexing: a[i] -> float, a[i:j] -> contiguous VIEW (shares memory).
        // Index/bounds normalization (numpy semantics) lives here; the wrapper is raw.
        .def("__getitem__", [](Array const& a, nb::object key) -> nb::object {
            if (PySlice_Check(key.ptr())) {
                Py_ssize_t start, stop, step;
                if (PySlice_Unpack(key.ptr(), &start, &stop, &step) < 0)
                    throw nb::python_error();
                Py_ssize_t const n =
                    PySlice_AdjustIndices((Py_ssize_t) a.size(), &start, &stop, step);
                if (step != 1)
                    throw nb::value_error(
                        "hpxpy.Array supports only contiguous slices (step == 1)");
                return nb::cast(a.view((std::size_t) start, (std::size_t) n));
            }
            Py_ssize_t i = PyNumber_AsSsize_t(key.ptr(), PyExc_IndexError);
            if (i == -1 && PyErr_Occurred())
                throw nb::python_error();
            Py_ssize_t const n = (Py_ssize_t) a.size();
            if (i < 0)
                i += n;
            if (i < 0 || i >= n)
                throw nb::index_error("Array index out of range");
            return nb::cast(a.getitem((std::size_t) i));
        }, "a[i] -> float; a[i:j] -> a contiguous view sharing memory (step must be 1).")
        .def("__setitem__", [](Array& a, Py_ssize_t i, double v) {
            Py_ssize_t const n = (Py_ssize_t) a.size();
            if (i < 0)
                i += n;
            if (i < 0 || i >= n)
                throw nb::index_error("Array index out of range");
            a.setitem((std::size_t) i, v);
        }, "a[i] = value (write a single element).")
        .def("to_numpy", &to_numpy_view,
             "Zero-copy NumPy view (writable; shares memory with the Array).")
        .def("__array__", [](nb::object self, nb::args, nb::kwargs) {
            return to_numpy_view(self);    // np.asarray(a) views; np.array(a) copies
        })
        .def("__len__", &Array::size)
        .def("__repr__", [](Array const& a) {
            return "Array(size=" + std::to_string(a.size()) + ")";
        });
    // C++-timed benchmark entry point: times a reduction in C++ (monotonic clock,
    // adaptive repeats) with the GIL released, so the perf harness never times
    // across the Python boundary. Returns (median_seconds, reps).
    m.def("bench", [](Array const& a, std::string const& op, double budget,
                      int min_reps, int max_reps) {
        nb::gil_scoped_release release;
        auto run = [&]() -> double {
            if (op == "sum") return a.sum();
            if (op == "min") return a.min();
            if (op == "max") return a.max();
            throw std::invalid_argument("unknown op: " + op);
        };
        hpxpy::timing::result r =
            hpxpy::timing::measure(run, budget, min_reps, max_reps);
        return std::make_pair(r.median_s, r.reps);
    }, "a"_a, "op"_a, "budget"_a = 0.5, "min_reps"_a = 5, "max_reps"_a = 200,
       "C++-timed median-of-times (s) and rep count for a reduction.");

    // Two-operand variant for dot (fused transform_reduce), same C++ timing.
    m.def("bench_dot", [](Array const& a, Array const& b, double budget,
                          int min_reps, int max_reps) {
        nb::gil_scoped_release release;
        hpxpy::timing::result r = hpxpy::timing::measure(
            [&]() -> double { return a.dot(b); }, budget, min_reps, max_reps);
        return std::make_pair(r.median_s, r.reps);
    }, "a"_a, "b"_a, "budget"_a = 0.5, "min_reps"_a = 5, "max_reps"_a = 200,
       "C++-timed median-of-times (s) and rep count for dot(a, b).");

    // Element-wise binary op benchmark (result is a new Array; the timing harness
    // keeps it observable so the transform is not elided).
    m.def("bench_binary", [](Array const& a, Array const& b, std::string const& op,
                             double budget, int min_reps, int max_reps) {
        nb::gil_scoped_release release;
        auto run = [&]() -> Array {
            if (op == "add") return a.add(b);
            if (op == "sub") return a.sub(b);
            if (op == "mul") return a.mul(b);
            if (op == "div") return a.div(b);
            throw std::invalid_argument("unknown op: " + op);
        };
        hpxpy::timing::result r =
            hpxpy::timing::measure(run, budget, min_reps, max_reps);
        return std::make_pair(r.median_s, r.reps);
    }, "a"_a, "b"_a, "op"_a, "budget"_a = 0.5, "min_reps"_a = 5, "max_reps"_a = 200,
       "C++-timed median-of-times (s) and rep count for an element-wise op.");

    m.def("zeros", &hpxpy::zeros, "n"_a, "Create an Array of n zeros (NUMA-aware).");
    m.def("full", &hpxpy::full, "n"_a, "value"_a,
          "Create an Array of n elements set to value (NUMA-aware).");
    m.def("arange", &hpxpy::arange, "n"_a,
          "Create an Array [0, 1, ..., n-1] (NUMA-aware parallel first-touch).");

    // --- NumPy bridge (Phase 2) -------------------------------------------
    m.def("to_numpy", &to_numpy_view, "a"_a,
          "Zero-copy NumPy view of an Array (writable; shares memory).");
    m.def("from_numpy", [](np_rw arr, bool copy) -> Array {
        std::size_t const n = arr.shape(0);
        double* p = arr.data();
        if (copy) {
            Array a(n, 0.0);    // NUMA-aware; correct first-touch for HPX ops
            if (n) {
                nb::gil_scoped_release release;
                hpx::copy(hpx::execution::par, p, p + n, a.mutable_data());
            }
            return a;
        }
        // Zero-copy borrow: keep the NumPy buffer alive via a GIL-aware deleter.
        // numa-naive (numpy placement); use copy=True for HPX compute.
        auto* hold = new np_rw(arr);
        std::shared_ptr<void> keep(hold, [](void* q) {
            nb::gil_scoped_acquire g;
            delete static_cast<np_rw*>(q);
        });
        return Array::borrow(p, n, std::move(keep));
    }, nb::arg("a").noconvert(), "copy"_a = true,
       "Bring a 1-D float64 C-contiguous NumPy array into hpxpy. copy=True (default) "
       "copies into a NUMA-aware Array; copy=False borrows it (zero-copy, shares "
       "memory both ways, but numa-naive). Non-float64/non-contiguous input raises "
       "(never a silent copy/cast).");

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
