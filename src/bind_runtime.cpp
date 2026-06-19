// hpxpy._core — register_runtime: the managed HPX runtime + distributed introspection.
//
// Managed HPX runtime (started on a background thread; the Python main thread is not an
// HPX thread). hpx::start launches the runtime and returns; Python drives the work from
// the foreign main thread.
//
// Multi-locality: HPX runs the entry function (hpx_main) only on the CONSOLE locality.
// The console uses it as its readiness handshake; a WORKER locality (launched with
// --hpx:worker) never runs hpx_main, so it instead gates on HPX's own hpx::is_running()
// signal. Shutdown stays collective — the console's finalize broadcasts to all
// localities, and a worker's hpx::stop() (in the dtor) returns once that broadcast
// arrives.
//
// Thread count is passed via the command line "--hpx:threads=N" — the cfg key
// "hpx.os_threads" was found NOT to limit the pool in practice (M0 lesson from the
// prototype benchmarking), whereas the CLI option does.
//
// This TU does NOT include array.hpp: it has no Array kernels, so it stays a light,
// fast compile.
//
// SPDX-License-Identifier: MIT

#include <hpx/hpx.hpp>
#include <hpx/hpx_start.hpp>
#include <hpx/version.hpp>
#include <hpx/runtime_distributed.hpp>        // get_num_localities / get_locality_id
#include <hpx/collectives/all_reduce.hpp>     // distributed_sum (cross-locality collective)

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "bind_fwd.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

struct runtime_manager
{
    explicit runtime_manager(int num_threads, std::vector<std::string> const& hpx_args)
      : running_(false), is_worker_(false), rts_(nullptr)
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

}  // namespace

void register_runtime(nb::module_& m)
{
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
}
