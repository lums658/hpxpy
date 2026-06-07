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
#include <hpx/modules/compute_local.hpp>
#include <hpx/runtime_local/run_as_hpx_thread.hpp>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <condition_variable>
#include <cstdlib>
#include <mutex>
#include <string>
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

// Zero-copy spike: sum a NumPy array via an HPX parallel reduction. The buffer
// is borrowed directly from NumPy (no copy); the GIL is released while HPX runs.
double array_sum(nb::ndarray<nb::numpy, const double, nb::c_contig> a)
{
    const double* data = a.data();
    std::size_t n = a.size();
    nb::gil_scoped_release release;
    return hpx::reduce(hpx::execution::par, data, data + n, 0.0);
}

// ---------------------------------------------------------------------------
// Array — the core data type: a 1-D float64 array backed by an HPX compute::vector
// with a NUMA-aware host block_allocator (parallel first-touch). A THIN wrapper:
// HPX owns the storage. We allocate ONLY for genuinely new arrays — never a
// between-layers/transport/gather buffer. Construction runs on an HPX thread
// (run_as_hpx_thread) so the allocator's parallel first-touch executes on the HPX
// thread pool. The buffer is contiguous, so NumPy interop (Phase 2) is a zero-copy
// borrow — never a copy. Multi-locality distribution is a later collectives
// layer (M4), over per-locality arrays; never a gather.
// ---------------------------------------------------------------------------
using numa_alloc = hpx::compute::host::block_allocator<double>;
using dvec = hpx::compute::vector<double, numa_alloc>;

class Array {
public:
    Array() = default;
    Array(std::size_t n, double fill) : size_(n) {
        hpx::run_as_hpx_thread([&] {
            data_ = std::make_shared<dvec>(n, fill);  // NUMA-aware first-touch
        });
    }

    std::size_t size() const { return size_; }
    std::size_t ndim() const { return 1; }

    // 0, 1, 2, ..., n-1. The block_allocator first-touches at construction; the
    // parallel for_loop writes the ramp on the same HPX workers (stays NUMA-local).
    static Array iota(std::size_t n) {
        Array a;
        a.size_ = n;
        hpx::run_as_hpx_thread([&] {
            a.data_ = std::make_shared<dvec>(n);
            double* p = a.data_->data();
            hpx::experimental::for_loop(hpx::execution::par, std::size_t(0), n,
                [p](std::size_t i) { p[i] = static_cast<double>(i); });
        });
        return a;
    }

private:
    std::shared_ptr<dvec> data_;
    std::size_t size_ = 0;
};

Array zeros(std::size_t n) { return Array(n, 0.0); }
Array full(std::size_t n, double value) { return Array(n, value); }
Array arange(std::size_t n) { return Array::iota(n); }

}  // namespace

NB_MODULE(_core, m)
{
    m.doc() = "hpxpy._core — M0 substrate (managed HPX runtime + zero-copy spike)";
    m.def("init_runtime", &init_runtime, "num_threads"_a = 0,
          "Start the HPX runtime (num_threads<=0 => all cores).");
    m.def("finalize_runtime", &finalize_runtime, "Stop the HPX runtime.");
    m.def("array_sum", &array_sum, "a"_a,
          "Zero-copy parallel sum of a 1-D float64 array via hpx::reduce.");
    m.def("num_worker_threads", []() { return hpx::get_num_worker_threads(); });
    m.def("hpx_version", []() { return hpx::complete_version(); });

    nb::class_<Array>(m, "Array")
        .def_prop_ro("size", &Array::size)
        .def_prop_ro("ndim", &Array::ndim)
        .def("__len__", &Array::size)
        .def("__repr__", [](Array const& a) {
            return "Array(size=" + std::to_string(a.size()) + ")";
        });
    m.def("zeros", &zeros, "n"_a, "Create an Array of n zeros (NUMA-aware).");
    m.def("full", &full, "n"_a, "value"_a,
          "Create an Array of n elements set to value (NUMA-aware).");
    m.def("arange", &arange, "n"_a,
          "Create an Array [0, 1, ..., n-1] (NUMA-aware parallel first-touch).");
}
