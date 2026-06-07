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
#include <functional>
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
    double scalar = 2.0;    // runtime (set via --scalar) so neither L0 nor L1 can
                            // constant-fold it — a FAIR scalar-op ladder.
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

// L0 — direct HPX algorithm over raw pointers (mirrors hpxpy::Array's bodies).
// q is the second operand for dot (nullptr otherwise).
double l0(std::string const& op, double const* p, double const* q, std::size_t n)
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
    if (op == "dot")
        return hpx::transform_reduce(hpx::execution::par, p, p + n, q, 0.0);
    // element-wise / scalar: new result buffer + one transform pass, return out[0].
    // Allocate the result the SAME way the wrapper does (make_shared<dvec>) so the
    // ladder isolates the wrapper call, not the allocation path.
    auto outp = std::make_shared<hpxpy::dvec>(n);
    double* o = outp->data();
    if (op == "muls")    // scalar broadcast (unary transform): x * s (runtime s)
    {
        double const s = g_cfg.scalar;
        hpx::transform(hpx::execution::par, p, p + n, o,
            [s](double x) { return x * s; });
    }
    else if (op == "add")
        hpx::transform(hpx::execution::par, p, p + n, q, o, std::plus<double>{});
    else if (op == "sub")
        hpx::transform(hpx::execution::par, p, p + n, q, o, std::minus<double>{});
    else if (op == "mul")
        hpx::transform(hpx::execution::par, p, p + n, q, o, std::multiplies<double>{});
    else if (op == "div")
        hpx::transform(hpx::execution::par, p, p + n, q, o, std::divides<double>{});
    else
        std::exit(2);
    return n ? o[0] : 0.0;
}

// L1 — the exact wrapper method.
double l1(std::string const& op, hpxpy::Array const& a, hpxpy::Array const& b)
{
    if (op == "sum")
        return a.sum();
    if (op == "min")
        return a.min();
    if (op == "max")
        return a.max();
    if (op == "dot")
        return a.dot(b);
    hpxpy::Array res;
    if (op == "muls")    // scalar broadcast (the exact wrapper method)
        res = a.mul_scalar(g_cfg.scalar);
    else if (op == "add")
        res = a.add(b);
    else if (op == "sub")
        res = a.sub(b);
    else if (op == "mul")
        res = a.mul(b);
    else if (op == "div")
        res = a.div(b);
    else
        std::exit(2);
    return res.size() ? res.data()[0] : 0.0;
}

int hpx_main(int, char**)
{
    int const threads = static_cast<int>(hpx::get_num_worker_threads());
    for (std::size_t n : g_cfg.sizes)
    {
        // Arrays built once; both rungs operate on the SAME buffers. Timed by the
        // shared C++ harness (hpxpy::timing) — identical to how the extension and
        // the cross-process baseline are timed. dot/element-wise need a 2nd operand.
        std::string const& op = g_cfg.op;
        bool const needs_b = (op == "dot" || op == "add" || op == "sub" ||
                              op == "mul" || op == "div");
        hpxpy::Array a = hpxpy::arange(n);
        hpxpy::Array b = needs_b ? hpxpy::arange(n) : hpxpy::Array();
        double const* p = a.data();
        double const* q = b.data();

        hpxpy::timing::result r0 = hpxpy::timing::measure(
            [&] { return l0(g_cfg.op, p, q, n); },
            g_cfg.budget, g_cfg.min_reps, g_cfg.max_reps);
        hpxpy::timing::result r1 = hpxpy::timing::measure(
            [&] { return l1(g_cfg.op, a, b); },
            g_cfg.budget, g_cfg.min_reps, g_cfg.max_reps);
        double v0 = l0(g_cfg.op, p, q, n);
        double v1 = l1(g_cfg.op, a, b);

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
        else if (a == "--scalar")
            g_cfg.scalar = std::stod(next());
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
