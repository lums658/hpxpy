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
#include "timing.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>

#include <condition_variable>
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
}
