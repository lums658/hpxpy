// cpp_baseline/diag.cpp — the abstraction-penalty LADDER, in one binary / one TU.
//
// Measures, over the SAME compute::vector buffer, same threads, median-of-times:
//   L0  direct C++:  hpx::reduce(par, p, p+n, ...)         (the bare HPX call)
//   L1  wrapped C++: hpxpy::Array::sum()/min()/max()       (the exact wrapper code)
//
// Because both rungs live in ONE translation unit, compiler flags, template
// instantiation, process, HPX runtime, and dataset are constant by construction —
// so L1/L0 is the TRUE C++ wrapping penalty, isolated from Python and from build
// flags. If L1/L0 ≈ 1, the wrapper is provably thin and any cross-binary 2× is a
// comparison artifact, not a penalty.
//
// SPDX-License-Identifier: MIT

#include "array.hpp"
#include "timing.hpp"

#include <hpx/algorithm.hpp>
#include <hpx/execution.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/numeric.hpp>

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

namespace {

struct config
{
    std::string op = "sum";
    std::vector<std::size_t> sizes;
    double budget = 0.5;
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
            double v = std::stod(tok);
            if (v <= 0.0)
                std::exit(2);
            out.push_back(static_cast<std::size_t>(v));
        }
        if (comma == std::string::npos)
            break;
        start = comma + 1;
    }
    return out;
}

// L0 — direct HPX reduce over a raw pointer (mirrors hpxpy::Array's reduce bodies).
double l0(std::string const& op, double const* p, std::size_t n)
{
    if (op == "sum")
        return hpx::reduce(hpx::execution::par, p, p + n, 0.0);
    if (op == "min")
        return hpx::reduce(hpx::execution::par, p, p + n,
            std::numeric_limits<double>::infinity(),
            [](double x, double y) { return x < y ? x : y; });
    if (op == "max")
        return hpx::reduce(hpx::execution::par, p, p + n,
            -std::numeric_limits<double>::infinity(),
            [](double x, double y) { return x > y ? x : y; });
    std::exit(2);
}

// L1 — the exact wrapper method.
double l1(std::string const& op, hpxpy::Array const& a)
{
    if (op == "sum")
        return a.sum();
    if (op == "min")
        return a.min();
    if (op == "max")
        return a.max();
    std::exit(2);
}

int hpx_main(int, char**)
{
    int const threads = static_cast<int>(hpx::get_num_worker_threads());
    for (std::size_t n : g_cfg.sizes)
    {
        // ONE Array, built once; both rungs reduce its SAME buffer. Timed by the
        // shared C++ harness (hpxpy::timing) — identical to how the extension and
        // the cross-process baseline are timed.
        hpxpy::Array a = hpxpy::arange(n);
        double const* p = a.data();

        hpxpy::timing::result r0 = hpxpy::timing::measure(
            [&] { return l0(g_cfg.op, p, n); },
            g_cfg.budget, g_cfg.min_reps, g_cfg.max_reps);
        hpxpy::timing::result r1 = hpxpy::timing::measure(
            [&] { return l1(g_cfg.op, a); },
            g_cfg.budget, g_cfg.min_reps, g_cfg.max_reps);
        double v0 = l0(g_cfg.op, p, n);
        double v1 = l1(g_cfg.op, a);

        double t0 = r0.median_s, t1 = r1.median_s;
        double g0 = t0 > 0 ? n / t0 / 1e9 : 0.0;
        double g1 = t1 > 0 ? n / t1 / 1e9 : 0.0;
        double penalty = t0 > 0 ? t1 / t0 : 0.0;

        std::printf("op=%s n=%zu threads=%d | L0 %.6gs %.2f GEl/s (%dx) | "
                    "L1 %.6gs %.2f GEl/s (%dx) | L1/L0=%.3f%s\n",
            g_cfg.op.c_str(), n, threads, t0, g0, r0.reps, t1, g1, r1.reps,
            penalty, (v0 == v1) ? "" : "  [VALUE MISMATCH]");
        std::fflush(stdout);
    }
    return hpx::finalize();
}

}    // namespace

int main(int argc, char** argv)
{
    int threads = 0;
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
    }
    if (g_cfg.sizes.empty())
    {
        std::fprintf(stderr, "no --sizes given\n");
        return 2;
    }

    std::vector<std::string> hargs;
    hargs.emplace_back(argv[0]);
    if (threads > 0)
        hargs.emplace_back("--hpx:threads=" + std::to_string(threads));
    std::vector<char*> hargv;
    for (auto& s : hargs)
        hargv.push_back(s.data());

    hpx::init_params params;
    return hpx::init(
        &hpx_main, static_cast<int>(hargv.size()), hargv.data(), params);
}
