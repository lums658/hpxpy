// benchmarks/cloud/cloud_bench.cpp
// Distributed HPX benchmark for "medusa-as-cloud" harness.
//
// Two operations, both weak-scaling (each locality owns N elements):
//
//   all_reduce_scalar — each locality computes a local sum of its NUMA-first-
//     touched buffer then all_reduce(sum, plus) across all localities.
//     Measures compute + comms together; the fraction that is comms scales with
//     locality count, making fabric latency/bandwidth visible.
//
//   all_reduce_vec — each locality all_reduce's a vector of K doubles (default
//     K=1024 => 8 KB).  This is almost purely comms-bound: total transferred
//     per locality = K*8*2*(L-1)/L bytes (ring-like reduction), so at 2 nodes
//     it is K*8 = 8 KB each way.  The 1 GbE vs IPoIB bandwidth difference is
//     clearly measurable at this message size.
//
// Each op outputs one JSON line per run:
//   {"op":"...","localities":L,"n_per_locality":N,"threads":T,
//    "median_s":s,"gbytes_s":G,"gelem_s":E}
//
// Build: -O3, link HPX::hpx (same as cpp_baseline).
// SPDX-License-Identifier: MIT

#include "../../src/timing.hpp"

#include <hpx/collectives/all_reduce.hpp>
#include <hpx/collectives/create_communicator.hpp>
#include <hpx/collectives/argument_types.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/modules/runtime_distributed.hpp>
#include <hpx/algorithm.hpp>
#include <hpx/execution.hpp>
#include <hpx/modules/compute_local.hpp>
#include <hpx/numeric.hpp>

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>
#include <numeric>

// ---------------------------------------------------------------------------
// NUMA-first-touched allocation (same substrate as cpp_baseline)
using numa_alloc = hpx::compute::host::block_allocator<double>;
using dvec       = hpx::compute::vector<double, numa_alloc>;

static dvec make_iota(std::size_t n)
{
    dvec v(n);
    double* p = v.data();
    hpx::experimental::for_loop(hpx::execution::par, std::size_t(0), n,
        [p](std::size_t i) { p[i] = static_cast<double>(i); });
    return v;
}

// ---------------------------------------------------------------------------
namespace {

struct config {
    std::string op       = "all_reduce_scalar";
    std::size_t n        = 1'000'000;   // elements per locality
    std::size_t vec_k    = 1024;        // doubles in comms-microbench vector
    double      budget   = 2.0;         // adaptive timing budget (s)
    int         min_reps = 5;
    int         max_reps = 100;
};

config g_cfg;

// ---------------------------------------------------------------------------
// Scalar all_reduce: local sum -> all_reduce(+)
// The communicator is created ONCE outside the timed loop; the generation
// counter increments per call (mandatory for repeated use of same communicator).
double run_all_reduce_scalar(dvec const& buf, std::size_t n,
    hpx::collectives::communicator comm,
    std::size_t gen)
{
    double const* p = buf.data();
    double local_sum = hpx::reduce(hpx::execution::par, p, p + n, 0.0);

    // all_reduce: every locality sends its local_sum, receives the global sum.
    // Pass the generation explicitly so HPX's collective server knows which
    // round this call belongs to.
    double global_sum = hpx::collectives::all_reduce(
        comm, local_sum, std::plus<double>{},
        hpx::collectives::generation_arg(gen)).get();
    return global_sum;
}

// ---------------------------------------------------------------------------
// Vector all_reduce: comms-bound microbench.
// Each locality all_reduce's a std::vector<double> of K elements (all ones).
// The reduction is element-wise addition. Almost all wall time is network.
std::vector<double> run_all_reduce_vec(std::size_t k,
    hpx::collectives::communicator comm,
    std::size_t gen)
{
    std::vector<double> local_vec(k, 1.0);

    auto result = hpx::collectives::all_reduce(
        comm, std::move(local_vec),
        [](std::vector<double> a, std::vector<double> const& b) {
            for (std::size_t i = 0; i < a.size(); ++i) a[i] += b[i];
            return a;
        },
        hpx::collectives::generation_arg(gen)).get();
    return result;
}

// ---------------------------------------------------------------------------
int hpx_main(int, char**)
{
    int const threads      = static_cast<int>(hpx::get_num_worker_threads());
    std::size_t const loc  = hpx::get_num_localities(hpx::launch::sync);
    std::size_t const me   = hpx::get_locality_id();
    std::size_t const n    = g_cfg.n;
    std::size_t const k    = g_cfg.vec_k;
    std::string const& op  = g_cfg.op;

    // Pre-allocate data ONCE (first-touch in parallel on this locality's NUMA).
    dvec buf;
    if (op == "all_reduce_scalar" || op == "both") {
        buf = make_iota(n);
    }

    // ---------------------------------------------------------------------------
    // CORRECT communicator reuse pattern for HPX collectives:
    // 1. Create ONE communicator (without generation_arg — defaults to auto).
    // 2. Pass a strictly-increasing generation_arg to EACH call.
    //    Generation 0 is invalid; start at 1 and increment each rep.
    //
    // Creating a new communicator per rep (with different generation_arg) does
    // NOT work — the internal AGAS server state for the basename gets out of sync.
    // ---------------------------------------------------------------------------

    if (op == "all_reduce_scalar" || op == "both")
    {
        // Create communicator once — no generation_arg means default/auto.
        // Both localities must create communicators with the SAME basename and
        // num_sites_arg at the SAME time (collective create).
        auto comm = hpx::collectives::create_communicator(
            "/cloud_bench/scalar",
            hpx::collectives::num_sites_arg(loc),
            hpx::collectives::this_site_arg(me));

        std::size_t gen = 1;    // starts at 1; 0 is reserved/invalid
        auto run_fn = [&]() -> double {
            return run_all_reduce_scalar(buf, n, comm, gen++);
        };

        // FIXED rep count — NOT adaptive. A collective (all_reduce) requires every
        // locality to issue the SAME number of calls: each rep advances the
        // generation counter (gen++ above), so if localities ran different rep
        // counts — which an adaptive, wall-clock-budgeted loop guarantees via
        // timing jitter — the last call would block forever waiting for a
        // generation a peer never reaches, deadlocking at finalize() (one rank in
        // the collective, the other in shutdown). min==max forces exactly max_reps
        // rounds on ALL localities (measure breaks on reps>=max_reps before the
        // budget check), keeping the collective lockstep. budget is unused here.
        int const reps = g_cfg.max_reps;
        hpxpy::timing::result r =
            hpxpy::timing::measure(run_fn, /*budget_s=*/0.0, reps, reps);

        // Throughput: N doubles per locality = N*8 bytes per locality.
        double gelem  = (double)n / r.median_s / 1e9;
        double gbytes = (double)n * 8.0 / r.median_s / 1e9;

        // Only locality 0 prints to avoid duplicate output lines.
        if (me == 0) {
            std::printf(
                "{\"op\":\"all_reduce_scalar\",\"localities\":%zu,"
                "\"n_per_locality\":%zu,\"threads\":%d,"
                "\"median_s\":%.9g,\"gelem_s\":%.6g,\"gbytes_s\":%.6g}\n",
                loc, n, threads, r.median_s, gelem, gbytes);
            std::fflush(stdout);
        }
    }

    // ---------------------------------------------------------------------------
    if (op == "all_reduce_vec" || op == "both")
    {
        auto comm = hpx::collectives::create_communicator(
            "/cloud_bench/vec",
            hpx::collectives::num_sites_arg(loc),
            hpx::collectives::this_site_arg(me));

        std::size_t gen = 1;
        auto run_fn = [&]() -> std::vector<double> {
            return run_all_reduce_vec(k, comm, gen++);
        };

        // Fixed rep count — lockstep collectives (see note in the scalar branch).
        int const reps = g_cfg.max_reps;
        hpxpy::timing::result r =
            hpxpy::timing::measure(run_fn, /*budget_s=*/0.0, reps, reps);

        // For comms microbench: bytes sent per locality per round ~= k*8 bytes
        // (send + receive).  Report "effective bandwidth" = k*8*2 / median_s.
        double msg_bytes  = (double)k * 8.0;
        double eff_gbytes = msg_bytes * 2.0 / r.median_s / 1e9;
        double gelem      = (double)k / r.median_s / 1e9;

        if (me == 0) {
            std::printf(
                "{\"op\":\"all_reduce_vec\",\"localities\":%zu,"
                "\"n_per_locality\":%zu,\"vec_k\":%zu,\"threads\":%d,"
                "\"median_s\":%.9g,\"gelem_s\":%.6g,\"gbytes_s\":%.6g}\n",
                loc, n, k, threads, r.median_s, gelem, eff_gbytes);
            std::fflush(stdout);
        }
    }

    return hpx::finalize();
}

}  // namespace

// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    // Parse our custom flags before handing the rest to hpx::init.
    std::vector<std::string> hargs;
    hargs.emplace_back(argv[0]);

    int threads = 0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            return (i + 1 < argc) ? std::string(argv[++i]) : std::string();
        };
        if      (a == "--op")       g_cfg.op       = next();
        else if (a == "--n")        g_cfg.n        = static_cast<std::size_t>(std::stod(next()));
        else if (a == "--vec-k")    g_cfg.vec_k    = static_cast<std::size_t>(std::stod(next()));
        else if (a == "--budget")   g_cfg.budget   = std::stod(next());
        else if (a == "--min-reps") g_cfg.min_reps = std::stoi(next());
        else if (a == "--max-reps") g_cfg.max_reps = std::stoi(next());
        else if (a == "--threads")  threads = std::stoi(next());
        else    hargs.emplace_back(a);  // pass through to HPX
    }

    if (threads > 0)
        hargs.emplace_back("--hpx:threads=" + std::to_string(threads));
    // Avoid mmap exhaustion at high thread counts (same as cpp_baseline).
    hargs.emplace_back("--hpx:ini=hpx.stacks.use_guard_pages=0");

    std::vector<char*> hargv;
    for (auto& s : hargs) hargv.push_back(s.data());

    // CRITICAL for distributed benchmarks: hpx_main must run on ALL localities,
    // not just locality 0. Without "hpx.run_hpx_main!=1", collectives hang
    // because non-root localities never enter the benchmark and never call
    // all_reduce — locality 0 waits forever. See 1d_stencil_8.cpp for the
    // canonical example of this pattern.
    hpx::init_params params;
    params.cfg = {"hpx.run_hpx_main!=1"};
    return hpx::init(
        &hpx_main, static_cast<int>(hargv.size()), hargv.data(), params);
}
