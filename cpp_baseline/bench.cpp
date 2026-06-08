// cpp_baseline/bench.cpp — hand-written C++ HPX reference kernels: the performance
// CEILING the hpxpy bindings are measured against (abstraction penalty = hpxpy / this).
//
// Same NUMA-aware substrate as hpxpy's Array — hpx::compute::vector<double,
// block_allocator<double>> with parallel first-touch — and the same execution policy,
// so the comparison isolates binding overhead, not data layout. Obeys the "point
// contract" in cpp_baseline/README.md so benchmarks.runner can drive it:
//
//   bench --op sum --sizes 1e6,1e7 --threads 8 --repeats 7 --warmup 1
//
// emits one JSON line per size on stdout:
//   {"op":"sum","n":10000000,"threads":8,"impl":"cpp","median_s":0.0123,"value":...}
// ("value" is the computed reduction result — an extra key the runner ignores, used
// by M2 to cross-check the kernel against analytic values.)
//
// SPDX-License-Identifier: MIT

#include "sparse.hpp"
#include "timing.hpp"

#include <hpx/algorithm.hpp>
#include <hpx/execution.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/modules/compute_local.hpp>
#include <hpx/numeric.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <string>
#include <vector>

namespace {

using numa_alloc = hpx::compute::host::block_allocator<double>;
using dvec = hpx::compute::vector<double, numa_alloc>;

struct config
{
    std::string op = "sum";
    std::vector<std::size_t> sizes;
    double budget = 0.5;    // adaptive: time reps until this many seconds elapse
    int min_reps = 5;
    int max_reps = 200;
};

config g_cfg;

std::vector<std::size_t> parse_sizes(std::string const& spec)
{
    std::vector<std::size_t> out;
    std::size_t start = 0;
    while (start <= spec.size())
    {
        std::size_t comma = spec.find(',', start);
        std::string tok = spec.substr(
            start, comma == std::string::npos ? std::string::npos : comma - start);
        if (!tok.empty())
        {
            double v = std::stod(tok);    // accepts "1e6"
            if (v <= 0.0)
            {
                std::fprintf(stderr, "size must be positive: %s\n", tok.c_str());
                std::exit(2);
            }
            out.push_back(static_cast<std::size_t>(v));
        }
        if (comma == std::string::npos)
            break;
        start = comma + 1;
    }
    return out;
}

// NUMA-first-touched vector of n elements set to iota 0..n-1, matching what an
// hpxpy Array constructor produces. The block_allocator first-touches at
// construction; the parallel for_loop writes the values on the same workers.
dvec make_iota(std::size_t n)
{
    dvec v(n);
    double* p = v.data();
    hpx::experimental::for_loop(hpx::execution::par, std::size_t(0), n,
        [p](std::size_t i) { p[i] = static_cast<double>(i); });
    return v;
}

int hpx_main(int, char**)
{
    int const threads = static_cast<int>(hpx::get_num_worker_threads());
    for (std::size_t n : g_cfg.sizes)
    {
        if (g_cfg.op == "spmv")    // direct C++ HPX CSR SpMV baseline (perf ceiling)
        {
            hpxpy::CsrMatrix A = hpxpy::laplacian_1d(n);
            hpxpy::Array x = hpxpy::arange(n);
            const std::int64_t* rp = A.row_ptr_data();
            const std::int64_t* ci = A.col_idx_data();
            const double* vp = A.values_data();
            const double* xp = x.data();
            hpxpy::Array y(n, 0.0);    // pre-allocated, reused — time the KERNEL only
            double* yp = y.mutable_data();
            auto run = [&]() -> double {
                hpx::experimental::for_loop(hpx::execution::par, std::size_t(0), n,
                    [rp, ci, vp, xp, yp](std::size_t i) {
                        double acc = 0.0;
                        for (std::int64_t k = rp[i]; k < rp[i + 1]; ++k)
                            acc += vp[k] * xp[ci[k]];
                        yp[i] = acc;
                    });
                return n ? yp[0] : 0.0;
            };
            hpxpy::timing::result r =
                hpxpy::timing::measure(run, g_cfg.budget, g_cfg.min_reps, g_cfg.max_reps);
            double value = run();
            std::printf("{\"op\": \"spmv\", \"n\": %zu, \"threads\": %d, \"impl\": \"cpp\", "
                        "\"median_s\": %.12g, \"reps\": %d, \"value\": %.12g}\n",
                n, threads, r.median_s, r.reps, value);
            std::fflush(stdout);
            continue;
        }

        if (g_cfg.op == "spmm")    // direct C++ HPX sparse x dense baseline (kernel)
        {
            std::size_t const K = 16;
            hpxpy::CsrMatrix A = hpxpy::laplacian_1d(n);
            hpxpy::DenseMatrix B(n, K, 1.0);
            hpxpy::DenseMatrix C(n, K, 0.0);
            const std::int64_t* rp = A.row_ptr_data();
            const std::int64_t* ci = A.col_idx_data();
            const double* vp = A.values_data();
            const double* bp = B.data();
            double* cp = C.mutable_data();
            auto run = [&]() -> double {
                hpx::experimental::for_loop(hpx::execution::par, std::size_t(0), n,
                    [rp, ci, vp, bp, cp, K](std::size_t i) {
                        double* crow = cp + i * K;
                        for (std::size_t c = 0; c < K; ++c) crow[c] = 0.0;
                        for (std::int64_t k = rp[i]; k < rp[i + 1]; ++k) {
                            double const v = vp[k];
                            const double* brow = bp + static_cast<std::size_t>(ci[k]) * K;
                            for (std::size_t c = 0; c < K; ++c) crow[c] += v * brow[c];
                        }
                    });
                return cp[0];
            };
            hpxpy::timing::result r =
                hpxpy::timing::measure(run, g_cfg.budget, g_cfg.min_reps, g_cfg.max_reps);
            double value = run();
            std::printf("{\"op\": \"spmm\", \"n\": %zu, \"threads\": %d, \"impl\": \"cpp\", "
                        "\"median_s\": %.12g, \"reps\": %d, \"value\": %.12g}\n",
                n, threads, r.median_s, r.reps, value);
            std::fflush(stdout);
            continue;
        }

        std::string const& op = g_cfg.op;
        bool const needs_b =
            (op == "dot" || op == "add" || op == "sub" || op == "mul" || op == "div");

        // Operand data is built (and first-touched) ONCE, outside the timed region.
        // Reductions time only the reduce; element-wise mirrors the wrapper a⊙b by
        // allocating a fresh result each call (so the alloc cost is in both sides).
        dvec a = make_iota(n);
        dvec b;
        if (needs_b)
            b = make_iota(n);
        double const* pa = a.data();
        double const* pb = needs_b ? b.data() : nullptr;

        auto run = [&]() -> double {
            if (op == "sum")
                return hpx::reduce(hpx::execution::par, pa, pa + n, 0.0);
            if (op == "min")
                return hpx::reduce(hpx::execution::par, pa, pa + n,
                    std::numeric_limits<double>::infinity(),
                    [](double x, double y) { return x < y ? x : y; });
            if (op == "max")
                return hpx::reduce(hpx::execution::par, pa, pa + n,
                    -std::numeric_limits<double>::infinity(),
                    [](double x, double y) { return x > y ? x : y; });
            if (op == "dot")
                return hpx::transform_reduce(hpx::execution::par, pa, pa + n, pb, 0.0);
            // element-wise: new result buffer + one transform pass (matches a⊙b)
            dvec out(n);
            double* o = out.data();
            if (op == "add")
                hpx::transform(hpx::execution::par, pa, pa + n, pb, o, std::plus<double>{});
            else if (op == "sub")
                hpx::transform(hpx::execution::par, pa, pa + n, pb, o, std::minus<double>{});
            else if (op == "mul")
                hpx::transform(hpx::execution::par, pa, pa + n, pb, o, std::multiplies<double>{});
            else if (op == "div")
                hpx::transform(hpx::execution::par, pa, pa + n, pb, o, std::divides<double>{});
            else { std::fprintf(stderr, "unknown op: %s\n", op.c_str()); std::exit(2); }
            return n ? o[0] : 0.0;
        };

        hpxpy::timing::result r =
            hpxpy::timing::measure(run, g_cfg.budget, g_cfg.min_reps, g_cfg.max_reps);
        double value = run();    // deterministic; for analytic cross-check

        std::printf("{\"op\": \"%s\", \"n\": %zu, \"threads\": %d, \"impl\": \"cpp\", "
                    "\"median_s\": %.12g, \"reps\": %d, \"value\": %.12g}\n",
            op.c_str(), n, threads, r.median_s, r.reps, value);
        std::fflush(stdout);
    }
    return hpx::finalize();
}

}    // namespace

int main(int argc, char** argv)
{
    int threads = 0;    // 0 => all cores (point contract)
    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            return (i + 1 < argc) ? std::string(argv[++i]) : std::string();
        };
        if (a == "--op")
            g_cfg.op = next();
        else if (a == "--sizes")
            g_cfg.sizes = parse_sizes(next());
        else if (a == "--threads")
            threads = std::stoi(next());
        else if (a == "--budget")
            g_cfg.budget = std::stod(next());
        else if (a == "--min-reps")
            g_cfg.min_reps = std::stoi(next());
        else if (a == "--max-reps")
            g_cfg.max_reps = std::stoi(next());
        // unknown args ignored
    }
    if (g_cfg.sizes.empty())
    {
        std::fprintf(stderr, "no --sizes given\n");
        return 2;
    }

    // Set the HPX worker count the same documented way as hpxpy: --hpx:threads=N.
    std::vector<std::string> hargs;
    hargs.emplace_back(argv[0]);
    if (threads > 0)
        hargs.emplace_back("--hpx:threads=" + std::to_string(threads));
    // Avoid mmap thread-stack exhaustion (max_map_count) at high thread counts.
    hargs.emplace_back("--hpx:ini=hpx.stacks.use_guard_pages=0");
    std::vector<char*> hargv;
    for (auto& s : hargs)
        hargv.push_back(s.data());

    hpx::init_params params;
    return hpx::init(
        &hpx_main, static_cast<int>(hargv.size()), hargv.data(), params);
}
