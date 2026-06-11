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
#include <hpx/runtime_distributed.hpp>        // get_num_localities / get_locality_id
#include <hpx/collectives/all_reduce.hpp>     // distributed_sum (cross-locality collective)

#include "array.hpp"
#include "sparse.hpp"
#include "timing.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

// ---------------------------------------------------------------------------
// Managed HPX runtime (started on a background thread; the Python main thread
// is not an HPX thread). hpx::start launches the runtime and returns; Python
// drives the work from the foreign main thread.
//
// Multi-locality: HPX runs the entry function (hpx_main) only on the CONSOLE
// locality. The console uses it as its readiness handshake; a WORKER locality
// (launched with --hpx:worker) never runs hpx_main, so it instead gates on HPX's
// own hpx::is_running() signal. Shutdown stays collective — the console's
// finalize broadcasts to all localities, and a worker's hpx::stop() (in the dtor)
// returns once that broadcast arrives.
//
// Thread count is passed via the command line "--hpx:threads=N" — the cfg key
// "hpx.os_threads" was found NOT to limit the pool in practice (M0 lesson from
// the prototype benchmarking), whereas the CLI option does.
// ---------------------------------------------------------------------------
namespace {

struct runtime_manager
{
    explicit runtime_manager(int num_threads, std::vector<std::string> const& hpx_args)
      : running_(false), rts_(nullptr), is_worker_(false)
    {
        argv_storage_.emplace_back("hpxpy");
        if (num_threads > 0)
            argv_storage_.emplace_back("--hpx:threads=" + std::to_string(num_threads));
        // Avoid mmap thread-stack exhaustion (max_map_count) at high thread counts.
        argv_storage_.emplace_back("--hpx:ini=hpx.stacks.use_guard_pages=0");
        // Extra HPX flags (e.g. distributed: --hpx:localities/--hpx:agas/--hpx:hpx/--hpx:worker).
        for (auto const& a : hpx_args)
        {
            argv_storage_.emplace_back(a);
            if (a == "--hpx:worker")
                is_worker_ = true;
        }
        for (auto& s : argv_storage_) argv_.push_back(s.data());

        hpx::init_params params;
        hpx::function<int(int, char**)> start_fn =
            hpx::bind_front(&runtime_manager::hpx_main, this);

        if (!hpx::start(start_fn, static_cast<int>(argv_.size()), argv_.data(), params))
            std::abort();  // runtime failed to start

        if (is_worker_)
        {
            // hpx_main runs only on the console; gate on HPX's own readiness signal.
            for (int i = 0; !hpx::is_running(); ++i)
            {
                if (i > 30000)
                    throw std::runtime_error(
                        "HPX worker runtime did not reach running state");
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        else
        {
            std::unique_lock<std::mutex> lk(startup_mtx_);
            while (!running_) startup_cond_.wait(lk);
        }
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

    bool is_worker() const { return is_worker_; }

private:
    hpx::spinlock mtx_;
    hpx::condition_variable_any cond_;
    std::mutex startup_mtx_;
    std::condition_variable startup_cond_;
    bool running_;
    bool is_worker_;
    hpx::runtime* rts_;
    std::vector<std::string> argv_storage_;
    std::vector<char*> argv_;
};

runtime_manager* g_rts = nullptr;

void init_runtime(int num_threads, std::vector<std::string> hpx_args)
{
    if (g_rts == nullptr)
    {
        nb::gil_scoped_acquire acquire;
        g_rts = new runtime_manager(num_threads, hpx_args);
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
using np_rw = nb::ndarray<nb::numpy, double, nb::c_contig>;   // any rank, C-contiguous in
using np_out = nb::ndarray<nb::numpy, double>;                // any rank out (dynamic)

// Zero-copy NumPy view of an Array. `obj` (the Array Python object) is the ndarray's
// owner, so the HPX buffer outlives the view. Writable; works for slice views too.
// For strided views (stride != 1) the DLPack element stride is passed so numpy sees
// the correct view without a copy. Negative strides (reverse views) are also passed
// through — numpy supports them as long as base_ points to the first logical element.
// Zero-copy NumPy view of an Array, any rank. `obj` (the Array Python object) owns the
// ndarray, so the HPX buffer outlives the view. Contiguous arrays pass nullptr strides
// (nanobind infers row-major); non-contiguous ones (a[::2], a[::-1], transposed views)
// pass explicit strides. NOTE: nanobind's typed ndarray takes strides in ELEMENTS (it
// multiplies by sizeof(T) internally) — passing bytes segfaults on negative strides.
// Parse an axis argument (int, or tuple/list of ints) into normalized axes:
// negatives wrapped (+ndim), validated in range, deduplicated, and sorted ascending.
// `None` is handled by the caller (full reduction -> scalar) and never reaches here.
// Bad axis -> IndexError; duplicate axis -> ValueError (numpy semantics).
std::vector<std::size_t> parse_axes(nb::object axis, std::size_t ndim)
{
    auto norm_one = [ndim](Py_ssize_t ax) -> std::size_t {
        Py_ssize_t nd = static_cast<Py_ssize_t>(ndim);
        if (ax < 0) ax += nd;
        if (ax < 0 || ax >= nd)
            throw nb::index_error("axis out of range for array dimensions");
        return static_cast<std::size_t>(ax);
    };

    std::vector<std::size_t> raw;
    if (PyTuple_Check(axis.ptr()) || PyList_Check(axis.ptr())) {
        nb::sequence seq = nb::cast<nb::sequence>(axis);
        for (nb::handle item : seq) {
            Py_ssize_t ax = PyNumber_AsSsize_t(item.ptr(), PyExc_IndexError);
            if (ax == -1 && PyErr_Occurred()) throw nb::python_error();
            raw.push_back(norm_one(ax));
        }
    } else {
        Py_ssize_t ax = PyNumber_AsSsize_t(axis.ptr(), PyExc_IndexError);
        if (ax == -1 && PyErr_Occurred()) throw nb::python_error();
        raw.push_back(norm_one(ax));
    }

    // Detect duplicates (after normalization), then sort + unique.
    std::vector<std::size_t> sorted = raw;
    std::sort(sorted.begin(), sorted.end());
    for (std::size_t k = 1; k < sorted.size(); ++k)
        if (sorted[k] == sorted[k - 1])
            throw nb::value_error("duplicate value in axis");
    return sorted;
}

np_out to_numpy_view(nb::object obj)
{
    Array& a = nb::cast<Array&>(obj);
    std::size_t nd = a.ndim();
    std::vector<std::size_t> shape(nd);
    std::vector<int64_t> strides(nd);
    for (std::size_t k = 0; k < nd; ++k) {
        shape[k] = a.shape()[k];
        strides[k] = static_cast<int64_t>(a.strides()[k]);   // ELEMENTS, not bytes
    }
    return np_out(a.mutable_data(), nd, shape.data(), obj,
                  a.is_contiguous() ? nullptr : strides.data());
}

}  // namespace

NB_MODULE(_core, m)
{
    m.doc() = "hpxpy._core — managed HPX runtime + the Array type and HPX reductions";
    m.def("init_runtime", &init_runtime, "num_threads"_a = 0,
          "hpx_args"_a = std::vector<std::string>{},
          "Start the HPX runtime (num_threads<=0 => all cores). hpx_args are raw HPX "
          "CLI flags appended to argv (e.g. distributed --hpx:localities/agas/hpx/worker).");
    m.def("finalize_runtime", &finalize_runtime, "Stop the HPX runtime.");
    m.def("num_worker_threads", []() { return hpx::get_num_worker_threads(); });
    m.def("hpx_version", []() { return hpx::complete_version(); });

    // Distributed introspection + a cross-locality collective (M4). num_localities/
    // locality_id are quick runtime queries; distributed_sum blocks on a collective so
    // it releases the GIL and runs on an HPX thread (collectives suspend).
    m.def("num_localities", []() {
        return static_cast<int>(hpx::get_num_localities(hpx::launch::sync));
    }, "Number of localities in the running HPX runtime.");
    m.def("locality_id", []() {
        return static_cast<int>(hpx::get_locality_id());
    }, "This locality's id (0 = console).");
    m.def("is_worker", []() {
        return g_rts != nullptr && g_rts->is_worker();
    }, "True if this process was started as a --hpx:worker locality.");
    m.def("is_console", []() {
        return !(g_rts != nullptr && g_rts->is_worker());
    }, "True on the console locality (the one that runs the user program).");
    m.def("distributed_sum", [](double local) -> double {
        nb::gil_scoped_release release;
        std::uint32_t n = hpx::get_num_localities(hpx::launch::sync);
        if (n <= 1)
            return local;    // single locality: collective is the identity
        std::uint32_t site = hpx::get_locality_id();
        // Collectives suspend, so run on an HPX thread; every site must participate.
        return hpx::async([=]() {
            return hpx::collectives::all_reduce(
                "hpxpy_distributed_sum", local, std::plus<double>{},
                hpx::collectives::num_sites_arg(n),
                hpx::collectives::this_site_arg(site)).get();
        }).get();
    }, "local"_a, "All-reduce(sum) of a scalar across localities (every site must call).");

    // Reductions release the GIL around the HPX work (the wrapper itself is
    // GIL-agnostic). std::invalid_argument from min()/max() maps to ValueError.
    nb::class_<Array>(m, "Array")
        .def_prop_ro("size", &Array::size)
        .def_prop_ro("ndim", &Array::ndim)
        .def_prop_ro("stride", &Array::stride)
        .def_prop_ro("shape", [](Array const& a) {
            nb::tuple t = nb::steal<nb::tuple>(PyTuple_New((Py_ssize_t) a.shape().size()));
            for (std::size_t k = 0; k < a.shape().size(); ++k)
                PyTuple_SET_ITEM(t.ptr(), (Py_ssize_t) k,
                                 PyLong_FromSize_t(a.shape()[k]));
            return t;
        }, "Shape of the array as a tuple of ints (N-D).")
        // sum/min/max: axis=None (default) -> scalar fast path (zero-penalty,
        // contiguous hpx::reduce); axis given (int or tuple) -> N-D axis reduction
        // returning a new Array. keepdims retains the reduced axes as size 1.
        .def("sum", [](Array const& a, nb::object axis, bool keepdims) -> nb::object {
            // GIL released around the C++ work; cast back to Python AFTER it is
            // re-acquired (the release scope ends before the nb::cast).
            if (axis.is_none()) {
                double s;
                { nb::gil_scoped_release release; s = a.sum(); }
                return nb::cast(s);
            }
            auto ax = parse_axes(axis, a.ndim());
            Array r;
            { nb::gil_scoped_release release; r = a.sum_axis(ax, keepdims); }
            return nb::cast(std::move(r));
        }, "axis"_a = nb::none(), "keepdims"_a = false,
           "Parallel sum. axis=None -> scalar (hpx::reduce); axis -> reduced Array.")
        .def("min", [](Array const& a, nb::object axis, bool keepdims) -> nb::object {
            if (axis.is_none()) {
                double s;
                { nb::gil_scoped_release release; s = a.min(); }
                return nb::cast(s);
            }
            auto ax = parse_axes(axis, a.ndim());
            Array r;
            { nb::gil_scoped_release release; r = a.min_axis(ax, keepdims); }
            return nb::cast(std::move(r));
        }, "axis"_a = nb::none(), "keepdims"_a = false,
           "Parallel minimum. axis=None -> scalar (empty -> ValueError); axis -> Array.")
        .def("max", [](Array const& a, nb::object axis, bool keepdims) -> nb::object {
            if (axis.is_none()) {
                double s;
                { nb::gil_scoped_release release; s = a.max(); }
                return nb::cast(s);
            }
            auto ax = parse_axes(axis, a.ndim());
            Array r;
            { nb::gil_scoped_release release; r = a.max_axis(ax, keepdims); }
            return nb::cast(std::move(r));
        }, "axis"_a = nb::none(), "keepdims"_a = false,
           "Parallel maximum. axis=None -> scalar (empty -> ValueError); axis -> Array.")
        // dot: 1-D . 1-D -> scalar (fused transform_reduce); 2-D . 2-D -> matmul Array.
        .def("dot", [](Array const& a, Array const& b) -> nb::object {
            if (a.ndim() == 1 && b.ndim() == 1) {
                double s;
                { nb::gil_scoped_release release; s = a.dot(b); }
                return nb::cast(s);
            }
            if (a.ndim() == 2 && b.ndim() == 2) {
                Array r;
                { nb::gil_scoped_release release; r = a.matmul(b); }
                return nb::cast(std::move(r));
            }
            throw nb::value_error(
                "dot: only 1-D . 1-D (scalar) and 2-D . 2-D (matmul) are supported");
        }, "b"_a, "1-D.1-D -> scalar (fused); 2-D.2-D -> matrix product (Array).")
        .def("matmul", [](Array const& a, Array const& b) {
            nb::gil_scoped_release release;
            return a.matmul(b);
        }, "b"_a, "2-D matrix product A @ B (naive O(m*n*k)).")
        .def("__matmul__", [](Array const& a, Array const& b) -> nb::object {
            if (a.ndim() == 1 && b.ndim() == 1) {
                double s;
                { nb::gil_scoped_release release; s = a.dot(b); }
                return nb::cast(s);
            }
            if (a.ndim() == 2 && b.ndim() == 2) {
                Array r;
                { nb::gil_scoped_release release; r = a.matmul(b); }
                return nb::cast(std::move(r));
            }
            throw nb::value_error(
                "@: only 1-D @ 1-D (scalar) and 2-D @ 2-D (matmul) are supported");
        })
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
        // a[i:j:k] with k != 1 -> strided VIEW (shares memory, no copy).
        // a[i,j,...] (tuple of ints, len==ndim) -> multi-index scalar get (N-D).
        // a[i:j,...] (tuple with a slice) -> NotImplementedError (stage 6).
        // Index/bounds normalization (numpy semantics) lives here; the wrapper is raw.
        .def("__getitem__", [](Array const& a, nb::object key) -> nb::object {
            // Tuple key: multi-index (all ints) or N-D slice (deferred).
            if (PyTuple_Check(key.ptr())) {
                Py_ssize_t tlen = PyTuple_GET_SIZE(key.ptr());
                // Check if any element is a slice -> deferred stage 6
                for (Py_ssize_t k = 0; k < tlen; ++k) {
                    if (PySlice_Check(PyTuple_GET_ITEM(key.ptr(), k)))
                        throw nb::type_error(
                            "N-D slice indexing is not yet implemented (stage 6); "
                            "use integer multi-index a[i, j, ...] for scalar access.");
                }
                // All-integer tuple: multi-index scalar get
                if (tlen != (Py_ssize_t) a.ndim())
                    throw nb::index_error("index tuple length must equal ndim");
                std::vector<std::size_t> idx(tlen);
                for (Py_ssize_t k = 0; k < tlen; ++k) {
                    Py_ssize_t ix = PyNumber_AsSsize_t(
                        PyTuple_GET_ITEM(key.ptr(), k), PyExc_IndexError);
                    if (ix == -1 && PyErr_Occurred()) throw nb::python_error();
                    Py_ssize_t dim = (Py_ssize_t) a.shape()[k];
                    if (ix < 0) ix += dim;
                    if (ix < 0 || ix >= dim)
                        throw nb::index_error("Array index out of range");
                    idx[k] = (std::size_t) ix;
                }
                return nb::cast(a.getitem(idx));
            }
            if (PySlice_Check(key.ptr())) {
                Py_ssize_t start, stop, step;
                if (PySlice_Unpack(key.ptr(), &start, &stop, &step) < 0)
                    throw nb::python_error();
                Py_ssize_t const n =
                    PySlice_AdjustIndices((Py_ssize_t) a.size(), &start, &stop, step);
                if (step == 1)
                    return nb::cast(a.view((std::size_t) start, (std::size_t) n));
                // Strided slice (including step==-1 for reverse): zero-copy view.
                return nb::cast(a.view_strided((std::ptrdiff_t) start,
                                               (std::size_t) n,
                                               (std::ptrdiff_t) step));
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
        }, "a[i] -> float; a[i:j] -> contiguous view; a[i:j:k] -> strided view (zero-copy); "
           "a[i,j,...] (tuple of ints, len==ndim) -> scalar (N-D multi-index).")
        .def("__setitem__", [](Array& a, nb::object key, nb::object value) {
            // Tuple key: multi-index set (all ints) or N-D slice (deferred).
            if (PyTuple_Check(key.ptr())) {
                Py_ssize_t tlen = PyTuple_GET_SIZE(key.ptr());
                for (Py_ssize_t k = 0; k < tlen; ++k) {
                    if (PySlice_Check(PyTuple_GET_ITEM(key.ptr(), k)))
                        throw nb::type_error(
                            "N-D slice assignment is not yet implemented (stage 6).");
                }
                if (tlen != (Py_ssize_t) a.ndim())
                    throw nb::index_error("index tuple length must equal ndim");
                std::vector<std::size_t> idx(tlen);
                for (Py_ssize_t k = 0; k < tlen; ++k) {
                    Py_ssize_t ix = PyNumber_AsSsize_t(
                        PyTuple_GET_ITEM(key.ptr(), k), PyExc_IndexError);
                    if (ix == -1 && PyErr_Occurred()) throw nb::python_error();
                    Py_ssize_t dim = (Py_ssize_t) a.shape()[k];
                    if (ix < 0) ix += dim;
                    if (ix < 0 || ix >= dim)
                        throw nb::index_error("Array index out of range");
                    idx[k] = (std::size_t) ix;
                }
                a.setitem(idx, nb::cast<double>(value));
                return;
            }
            if (PySlice_Check(key.ptr())) {
                Py_ssize_t start, stop, step;
                if (PySlice_Unpack(key.ptr(), &start, &stop, &step) < 0)
                    throw nb::python_error();
                Py_ssize_t const n =
                    PySlice_AdjustIndices((Py_ssize_t) a.size(), &start, &stop, step);
                if (step != 1)
                    throw nb::value_error(
                        "hpxpy.Array supports only contiguous slice assignment (step == 1)");
                if (nb::isinstance<Array>(value)) {          // a[i:j] = Array (copy in)
                    Array const& rhs = nb::cast<Array&>(value);
                    if (rhs.size() != (std::size_t) n)
                        throw nb::value_error("slice assignment size mismatch");
                    nb::gil_scoped_release r;
                    a.assign_range((std::size_t) start, rhs);
                } else {                                     // a[i:j] = scalar (fill)
                    double v = nb::cast<double>(value);
                    nb::gil_scoped_release r;
                    a.fill_range((std::size_t) start, (std::size_t) n, v);
                }
                return;
            }
            Py_ssize_t i = PyNumber_AsSsize_t(key.ptr(), PyExc_IndexError);
            if (i == -1 && PyErr_Occurred())
                throw nb::python_error();
            Py_ssize_t const n = (Py_ssize_t) a.size();
            if (i < 0)
                i += n;
            if (i < 0 || i >= n)
                throw nb::index_error("Array index out of range");
            a.setitem((std::size_t) i, nb::cast<double>(value));
        }, "a[i] = value; a[i:j] = scalar (fill) or Array (copy, contiguous step 1); "
           "a[i,j,...] = scalar (N-D multi-index set).")
        .def("to_numpy", &to_numpy_view,
             "Zero-copy NumPy view (writable; shares memory with the Array).")
        .def("__array__", [](nb::object self, nb::args, nb::kwargs) {
            return to_numpy_view(self);    // np.asarray(a) views; np.array(a) copies
        })
        .def("__len__", &Array::size)
        .def("__repr__", [](Array const& a) {
            return "Array(size=" + std::to_string(a.size()) + ")";
        })
        // --- N-D view ops (stage 3) -----------------------------------------
        // transpose(axes=None): permute axes; empty/None => reverse all.
        .def("transpose", [](Array const& a, nb::object axes_arg) -> Array {
            if (axes_arg.is_none()) {
                return a.transpose();
            }
            // Accept a tuple or list of ints.
            std::vector<std::size_t> axes;
            nb::sequence seq = nb::cast<nb::sequence>(axes_arg);
            for (nb::handle item : seq)
                axes.push_back(nb::cast<std::size_t>(item));
            return a.transpose(std::move(axes));
        }, "axes"_a = nb::none(),
           "Permute axes (None/empty => reverse). Zero-copy view sharing memory.")
        .def_prop_ro("T", [](Array const& a) { return a.transpose(); },
            "Shorthand for .transpose() with no args (reverse all axes).")
        // reshape(new_shape): product must equal size. Contiguous => view; else copy.
        // Accepts an int (1-D), a tuple, or a list. Supports a single -1 for inference.
        .def("reshape", [](Array const& a, nb::object shape_arg) -> Array {
            std::vector<std::ptrdiff_t> new_shape;
            if (PyLong_Check(shape_arg.ptr())) {
                new_shape.push_back(nb::cast<std::ptrdiff_t>(shape_arg));
            } else {
                nb::sequence seq = nb::cast<nb::sequence>(shape_arg);
                for (nb::handle item : seq)
                    new_shape.push_back(nb::cast<std::ptrdiff_t>(item));
            }
            return a.reshape(std::move(new_shape));
        }, "shape"_a,
           "Reshape to new_shape (int or tuple). Contiguous => zero-copy view; "
           "non-contiguous => copy. One -1 is inferred.")
        .def("ravel", [](Array const& a) { return a.ravel(); },
            "Flatten to 1-D (reshape to (size,)). Contiguous => zero-copy view.")
        // squeeze(axis=None): drop size-1 dims. axis may be None, int, or tuple.
        .def("squeeze", [](Array const& a, nb::object axis_arg) -> Array {
            if (axis_arg.is_none())
                return a.squeeze();
            std::vector<std::size_t> axes;
            if (PyLong_Check(axis_arg.ptr())) {
                axes.push_back(nb::cast<std::size_t>(axis_arg));
            } else {
                nb::sequence seq = nb::cast<nb::sequence>(axis_arg);
                for (nb::handle item : seq)
                    axes.push_back(nb::cast<std::size_t>(item));
            }
            return a.squeeze(std::move(axes));
        }, "axis"_a = nb::none(),
           "Remove size-1 dims (all if axis=None, else the named axis/axes).")
        // expand_dims(axis): insert a size-1 dim.
        .def("expand_dims", [](Array const& a, std::size_t axis) {
            return a.expand_dims(axis);
        }, "axis"_a, "Insert a size-1 dimension at position axis.");
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

    // zeros/full/ones accept either an int (1-D) or a tuple/list of ints (N-D).
    m.def("zeros", [](nb::object shape_arg) -> Array {
        if (PyTuple_Check(shape_arg.ptr()) || PyList_Check(shape_arg.ptr())) {
            nb::sequence seq = nb::cast<nb::sequence>(shape_arg);
            std::vector<std::size_t> shape;
            for (nb::handle item : seq)
                shape.push_back(nb::cast<std::size_t>(item));
            return hpxpy::zeros_nd(std::move(shape));
        }
        return hpxpy::zeros(nb::cast<std::size_t>(shape_arg));
    }, "shape"_a, "Create an Array of zeros. shape may be an int (1-D) or tuple/list (N-D).");
    m.def("ones", [](nb::object shape_arg) -> Array {
        if (PyTuple_Check(shape_arg.ptr()) || PyList_Check(shape_arg.ptr())) {
            nb::sequence seq = nb::cast<nb::sequence>(shape_arg);
            std::vector<std::size_t> shape;
            for (nb::handle item : seq)
                shape.push_back(nb::cast<std::size_t>(item));
            return hpxpy::ones_nd(std::move(shape));
        }
        return hpxpy::full(nb::cast<std::size_t>(shape_arg), 1.0);
    }, "shape"_a, "Create an Array of ones. shape may be an int (1-D) or tuple/list (N-D).");
    m.def("full", [](nb::object shape_arg, double value) -> Array {
        if (PyTuple_Check(shape_arg.ptr()) || PyList_Check(shape_arg.ptr())) {
            nb::sequence seq = nb::cast<nb::sequence>(shape_arg);
            std::vector<std::size_t> shape;
            for (nb::handle item : seq)
                shape.push_back(nb::cast<std::size_t>(item));
            return hpxpy::full_nd(std::move(shape), value);
        }
        return hpxpy::full(nb::cast<std::size_t>(shape_arg), value);
    }, "shape"_a, "value"_a,
       "Create an Array filled with value. shape may be an int (1-D) or tuple/list (N-D).");
    m.def("arange", &hpxpy::arange, "n"_a,
          "Create an Array [0, 1, ..., n-1] (NUMA-aware parallel first-touch).");

    // --- NumPy bridge (Phase 2) -------------------------------------------
    m.def("to_numpy", &to_numpy_view, "a"_a,
          "Zero-copy NumPy view of an Array (writable; shares memory).");
    m.def("from_numpy", [](np_rw arr, bool copy) -> Array {
        std::size_t const nd = arr.ndim();
        std::vector<std::size_t> shape(nd);
        for (std::size_t k = 0; k < nd; ++k)
            shape[k] = arr.shape(k);
        std::size_t const total = arr.size();    // product of all axes
        double* p = arr.data();
        if (copy) {
            Array a(shape, 0.0);    // NUMA-aware N-D; correct first-touch for HPX ops
            if (total) {
                nb::gil_scoped_release release;
                hpx::copy(hpx::execution::par, p, p + total, a.mutable_data());
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
        return Array::borrow_nd(p, std::move(shape), std::move(keep));
    }, nb::arg("a").noconvert(), "copy"_a = true,
       "Bring a float64 C-contiguous NumPy array (any rank) into hpxpy. copy=True "
       "(default) copies into a NUMA-aware Array; copy=False borrows it (zero-copy, "
       "shares memory both ways, but numa-naive). Non-float64/non-contiguous input "
       "raises (never a silent copy/cast).");

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
