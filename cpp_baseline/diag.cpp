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
// BR1 extension (June 2026):
//   * matmul L0/L1 (2-D, raw for_loop triple vs Array::matmul)
//   * axis_reduce L0/L1 (2-D sum over axis=0 and axis=1, raw vs Array::sum_axis)
//   * dtype variants: float32 and int64 versions of sum, add, muls
//     (dispatched via --dtype f32|i64|f64, default f64)
//
// SPDX-License-Identifier: MIT

#include "array.hpp"
#include "sparse.hpp"
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
    std::string dtype = "f64";  // "f64", "f32", "i64"
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

// Map dtype string to DType enum.
hpxpy::DType parse_dtype(std::string const& s)
{
    if (s == "f32") return hpxpy::DType::F32;
    if (s == "i64") return hpxpy::DType::I64;
    return hpxpy::DType::F64;  // default: f64
}

// L0 — direct HPX algorithm over raw pointers (mirrors hpxpy::Array's bodies).
// q is the second operand for dot (nullptr otherwise).
// T is the element type.
template <typename T>
double l0_typed(std::string const& op, T const* p, T const* q, std::size_t n, double scalar)
{
    if (op == "sum")
        return static_cast<double>(hpx::reduce(hpx::execution::par, p, p + n, T(0)));
    if (op == "min")
        return static_cast<double>(hpx::reduce(hpx::execution::par, p, p + n,
            std::numeric_limits<T>::max(),
            [](T x, T y) { return x < y ? x : y; }));
    if (op == "max")
        return static_cast<double>(hpx::reduce(hpx::execution::par, p, p + n,
            std::numeric_limits<T>::lowest(),
            [](T x, T y) { return x > y ? x : y; }));
    if (op == "dot")
        return static_cast<double>(hpx::transform_reduce(hpx::execution::par, p, p + n, q, T(0)));

    // element-wise / scalar: new result buffer + one transform pass, return out[0].
    using Vec = hpx::compute::vector<T, hpxpy::block_allocator<T>>;
    auto outp = std::make_shared<Vec>(n);
    T* o = outp->data();
    if (op == "sort")    // copy + sort (repeatable: re-copies the input each call)
    {
        hpx::copy(hpx::execution::par, p, p + n, o);
        hpx::sort(hpx::execution::par, o, o + n);
        return n ? static_cast<double>(o[0]) : 0.0;
    }
    if (op == "scan")    // inclusive prefix sum
    {
        hpx::inclusive_scan(hpx::execution::par, p, p + n, o);
        return n ? static_cast<double>(o[n - 1]) : 0.0;
    }
    T const s = static_cast<T>(scalar);
    if (op == "muls")    // scalar broadcast (unary transform): x * s (runtime s)
    {
        hpx::transform(hpx::execution::par, p, p + n, o,
            [s](T x) { return x * s; });
    }
    else if (op == "add")
        hpx::transform(hpx::execution::par, p, p + n, q, o, std::plus<T>{});
    else if (op == "sub")
        hpx::transform(hpx::execution::par, p, p + n, q, o, std::minus<T>{});
    else if (op == "mul")
        hpx::transform(hpx::execution::par, p, p + n, q, o, std::multiplies<T>{});
    else if (op == "div")
        hpx::transform(hpx::execution::par, p, p + n, q, o, std::divides<T>{});
    else
        std::exit(2);
    return n ? static_cast<double>(o[0]) : 0.0;
}

// Legacy f64 wrapper (used by spmv/spmm/strided/matmul/axis_reduce paths).
double l0(std::string const& op, double const* p, double const* q, std::size_t n)
{
    return l0_typed<double>(op, p, q, n, g_cfg.scalar);
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
    if (op == "sort")    // copy + in-place sort (the exact wrapper path of hpx.sort)
    {
        hpxpy::Array c = a.copy();
        c.sort();
        return c.size() ? c.data()[0] : 0.0;
    }
    if (op == "scan")
    {
        hpxpy::Array c = a.cumsum();
        return c.size() ? c.data()[c.size() - 1] : 0.0;
    }
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

// ---- dtype-generic L0/L1 helpers for typed benchmark ops ----

// Run a dtype-generic scalar/elementwise L0/L1 benchmark at the given DType.
// Used for: sum, add, muls with --dtype f32|i64|f64.
void run_typed_ladder(std::string const& op, std::size_t n, hpxpy::DType dt, int threads)
{
    // Build typed Array using iota (parallel first-touch, NUMA-local).
    hpxpy::Array a = hpxpy::arange(n, dt);
    bool const needs_b = (op == "dot" || op == "add" || op == "sub" ||
                          op == "mul" || op == "div");
    hpxpy::Array b = needs_b ? hpxpy::arange(n, dt) : hpxpy::Array();

    // dtype name for output
    const char* dname = (dt == hpxpy::DType::F32) ? "f32" :
                        (dt == hpxpy::DType::I64) ? "i64" : "f64";

    // L0: dispatch to typed raw-HPX kernel at dt.
    auto l0run = [&]() -> double {
        return hpxpy::dispatch_dtype(dt, [&](auto tag) -> double {
            using T = decltype(tag);
            T const* p = a.data_as<T>();
            T const* q = needs_b ? b.data_as<T>() : nullptr;
            return l0_typed<T>(op, p, q, n, g_cfg.scalar);
        });
    };
    // L1: the wrapper method (already typed internally via dispatch_dtype).
    auto l1run = [&]() -> double { return l1(op, a, b); };

    hpxpy::timing::result r0 = hpxpy::timing::measure(
        l0run, g_cfg.budget, g_cfg.min_reps, g_cfg.max_reps);
    hpxpy::timing::result r1 = hpxpy::timing::measure(
        l1run, g_cfg.budget, g_cfg.min_reps, g_cfg.max_reps);

    double t0 = r0.median_s, t1 = r1.median_s;
    double g0 = t0 > 0 ? n / t0 / 1e9 : 0.0;
    double g1 = t1 > 0 ? n / t1 / 1e9 : 0.0;
    double penalty = t0 > 0 ? t1 / t0 : 0.0;

    std::printf("op=%s dtype=%s n=%zu threads=%d | L0 %.6gs %.2f GEl/s (%dx) | "
                "L1 %.6gs %.2f GEl/s (%dx) | L1/L0=%.3f\n",
        op.c_str(), dname, n, threads, t0, g0, r0.reps, t1, g1, r1.reps, penalty);
    std::fflush(stdout);
}

// ---- matmul L0/L1 ladder ----
// L0: raw for_loop over output cells (i,j), serial k inner loop — same algorithm
//     as Array::matmul, but called directly on typed buffers without the wrapper.
// L1: Array::matmul (the exact wrapper, pre-allocated C outside the timed region).
void run_matmul(std::size_t m, int threads)
{
    std::size_t const k = m;  // square matrix
    std::size_t const nn = m;

    // Build operands once (NUMA first-touch via alloc_nd_ -> parallel iota).
    hpxpy::Array A({m, k}, 1.0, hpxpy::DType::F64);
    hpxpy::Array B({k, nn}, 1.0, hpxpy::DType::F64);
    // Pre-allocate C (output) once — BOTH L0 and L1 write into this.
    // For L0 we zero it before each rep; L1 (matmul) always allocates a fresh C.
    // To make them COMPARABLE: L0 times alloc+kernel (same as L1), OR we test both
    // kernel-only. We do KERNEL-ONLY here: L0 writes into a pre-allocated array
    // (zeroed once), L1 allocates C each call — to isolate the dispatch cost,
    // use kernel-only L0 (write into pre-alloc) vs L1 (alloc+write).
    // NOTE: L1 allocates a new C each call (matmul returns Array). So "kernel-only"
    // for L1 is not easily separated. We time the SAME thing: alloc + triple loop.
    // Both L0 and L1 include the alloc cost to be symmetric.

    std::ptrdiff_t sta0 = A.strides()[0], sta1 = A.strides()[1];
    std::ptrdiff_t stb0 = B.strides()[0], stb1 = B.strides()[1];
    double const* a = A.data_as<double>();
    double const* b = B.data_as<double>();

    // L0: allocate C each call (same as L1), do the raw triple-loop.
    auto l0run = [&]() -> double {
        auto cvec = std::make_shared<hpxpy::dvec>(m * nn, 0.0);
        double* c = cvec->data();
        hpx::experimental::for_loop(hpx::execution::par,
            std::size_t(0), m * nn,
            [a, b, c, nn, k, sta0, sta1, stb0, stb1](std::size_t out) {
                std::size_t i = out / nn;
                std::size_t j = out % nn;
                double acc = 0.0;
                for (std::size_t p = 0; p < k; ++p)
                    acc += a[static_cast<std::ptrdiff_t>(i) * sta0 +
                             static_cast<std::ptrdiff_t>(p) * sta1] *
                           b[static_cast<std::ptrdiff_t>(p) * stb0 +
                             static_cast<std::ptrdiff_t>(j) * stb1];
                c[out] = acc;
            });
        return c[0];
    };
    // L1: Array::matmul (the exact wrapper — allocates + runs the same loop).
    auto l1run = [&]() -> double {
        hpxpy::Array C = A.matmul(B);
        return C.size() ? C.data_as<double>()[0] : 0.0;
    };

    hpxpy::timing::result r0 = hpxpy::timing::measure(
        l0run, g_cfg.budget, g_cfg.min_reps, g_cfg.max_reps);
    hpxpy::timing::result r1 = hpxpy::timing::measure(
        l1run, g_cfg.budget, g_cfg.min_reps, g_cfg.max_reps);
    double t0 = r0.median_s, t1 = r1.median_s;
    // GFLOP/s: 2*m*k*n flops (multiply-add)
    double flops = 2.0 * static_cast<double>(m) * static_cast<double>(k) * static_cast<double>(nn);
    double gf0 = t0 > 0 ? flops / t0 / 1e9 : 0.0;
    double gf1 = t1 > 0 ? flops / t1 / 1e9 : 0.0;
    std::printf("op=matmul m=%zu k=%zu n=%zu threads=%d | L0 %.6gs %.2f GFLOP/s (%dx) | "
                "L1 %.6gs %.2f GFLOP/s (%dx) | L1/L0=%.3f\n",
        m, k, nn, threads, t0, gf0, r0.reps, t1, gf1, r1.reps, t0 > 0 ? t1 / t0 : 0.0);
    std::fflush(stdout);
}

// ---- axis_reduce L0/L1 ladder ----
// 2-D m x n array; reduce over axis=0 (output length n) or axis=1 (output length m).
// L0: raw for_loop over output cells, serial inner accumulator.
// L1: Array::sum_axis({axis}, false).
void run_axis_reduce(std::size_t rows, std::size_t cols, int axis, int threads)
{
    // Build a 2-D array (NUMA-first-touch via alloc_nd_).
    hpxpy::Array A({rows, cols}, 1.0, hpxpy::DType::F64);
    std::ptrdiff_t st0 = A.strides()[0];  // == cols (row stride in elements)
    std::ptrdiff_t st1 = A.strides()[1];  // == 1 (col stride in elements)
    double const* a = A.data_as<double>();

    std::size_t out_size = (axis == 0) ? cols : rows;

    // L0: raw for_loop over output elements; serial inner loop.
    auto l0run = [&]() -> double {
        auto cvec = std::make_shared<hpxpy::dvec>(out_size, 0.0);
        double* c = cvec->data();
        if (axis == 0) {
            // reduce over rows -> output[j] = sum_i a[i,j]
            hpx::experimental::for_loop(hpx::execution::par,
                std::size_t(0), cols,
                [a, c, rows, st0, st1](std::size_t j) {
                    double acc = 0.0;
                    for (std::size_t i = 0; i < rows; ++i)
                        acc += a[static_cast<std::ptrdiff_t>(i) * st0 +
                                 static_cast<std::ptrdiff_t>(j) * st1];
                    c[j] = acc;
                });
        } else {
            // reduce over cols -> output[i] = sum_j a[i,j]
            hpx::experimental::for_loop(hpx::execution::par,
                std::size_t(0), rows,
                [a, c, cols, st0, st1](std::size_t i) {
                    double acc = 0.0;
                    for (std::size_t j = 0; j < cols; ++j)
                        acc += a[static_cast<std::ptrdiff_t>(i) * st0 +
                                 static_cast<std::ptrdiff_t>(j) * st1];
                    c[i] = acc;
                });
        }
        return c[0];
    };
    // L1: Array::sum_axis (the exact wrapper).
    auto l1run = [&]() -> double {
        hpxpy::Array out = A.sum_axis({static_cast<std::size_t>(axis)}, false);
        return out.size() ? out.data_as<double>()[0] : 0.0;
    };

    hpxpy::timing::result r0 = hpxpy::timing::measure(
        l0run, g_cfg.budget, g_cfg.min_reps, g_cfg.max_reps);
    hpxpy::timing::result r1 = hpxpy::timing::measure(
        l1run, g_cfg.budget, g_cfg.min_reps, g_cfg.max_reps);
    double t0 = r0.median_s, t1 = r1.median_s;
    std::size_t total = rows * cols;
    double g0 = t0 > 0 ? total / t0 / 1e9 : 0.0;
    double g1 = t1 > 0 ? total / t1 / 1e9 : 0.0;
    std::printf("op=axis_reduce axis=%d rows=%zu cols=%zu n=%zu threads=%d | "
                "L0 %.6gs %.2f GEl/s (%dx) | L1 %.6gs %.2f GEl/s (%dx) | L1/L0=%.3f\n",
        axis, rows, cols, total, threads, t0, g0, r0.reps, t1, g1, r1.reps,
        t0 > 0 ? t1 / t0 : 0.0);
    std::fflush(stdout);
}

int hpx_main(int, char**)
{
    int const threads = static_cast<int>(hpx::get_num_worker_threads());

    // ---- New BR1 ops: matmul, axis_reduce, dtype variants ----

    if (g_cfg.op == "matmul")
    {
        for (std::size_t n : g_cfg.sizes)
            run_matmul(n, threads);
        return hpx::finalize();
    }

    if (g_cfg.op == "axis_reduce_0" || g_cfg.op == "axis_reduce_1")
    {
        int axis = (g_cfg.op == "axis_reduce_0") ? 0 : 1;
        for (std::size_t n : g_cfg.sizes)
        {
            // Interpret n as the TOTAL elements; use a square-ish 2-D array.
            // rows = cols = sqrt(n), rounded to nearest perfect square.
            std::size_t rows = static_cast<std::size_t>(std::sqrt(static_cast<double>(n)));
            if (rows == 0) rows = 1;
            std::size_t cols = rows;  // square
            run_axis_reduce(rows, cols, axis, threads);
        }
        return hpx::finalize();
    }

    // dtype-generic ops (sum, add, muls with --dtype f32|i64|f64).
    // These run the same ladder but dispatch to the typed raw path.
    hpxpy::DType dt = parse_dtype(g_cfg.dtype);
    if (g_cfg.dtype != "f64" &&
        (g_cfg.op == "sum" || g_cfg.op == "add" || g_cfg.op == "muls" ||
         g_cfg.op == "min" || g_cfg.op == "max" || g_cfg.op == "mul" ||
         g_cfg.op == "sub" || g_cfg.op == "div"))
    {
        for (std::size_t n : g_cfg.sizes)
            run_typed_ladder(g_cfg.op, n, dt, threads);
        return hpx::finalize();
    }

    // ---- Original ops (f64 only) ----

    for (std::size_t n : g_cfg.sizes)
    {
        if (g_cfg.op == "spmv" || g_cfg.op == "spmvk")
        {
            // spmv  = full op (allocate y each call + kernel) — what the user calls.
            // spmvk = KERNEL ONLY (y pre-allocated once, reused) — isolates the
            //         wrapper kernel from the result allocation's first-touch/variance.
            bool const kernel_only = (g_cfg.op == "spmvk");
            hpxpy::CsrMatrix A = hpxpy::laplacian_1d(n);
            hpxpy::Array x = hpxpy::arange(n);
            hpxpy::Array y(n, 0.0);    // reused buffer (spmvk) / reference for alloc cost
            const std::int64_t* rp = A.row_ptr_data();
            const std::int64_t* ci = A.col_idx_data();
            const double* vp = A.values_data();
            const double* xp = x.data();
            double* yp = y.mutable_data();
            auto kernel = [&] {
                hpx::experimental::for_loop(hpx::execution::par, std::size_t(0), n,
                    [rp, ci, vp, xp, yp](std::size_t i) {
                        double acc = 0.0;
                        for (std::int64_t k = rp[i]; k < rp[i + 1]; ++k)
                            acc += vp[k] * xp[ci[k]];
                        yp[i] = acc;
                    });
            };
            auto l0run = [&]() -> double {
                if (kernel_only) { kernel(); return yp[0]; }
                hpxpy::Array yy(n, 0.0);            // full: alloc + kernel
                double* o = yy.mutable_data();
                hpx::experimental::for_loop(hpx::execution::par, std::size_t(0), n,
                    [rp, ci, vp, xp, o](std::size_t i) {
                        double acc = 0.0;
                        for (std::int64_t k = rp[i]; k < rp[i + 1]; ++k)
                            acc += vp[k] * xp[ci[k]];
                        o[i] = acc;
                    });
                return n ? o[0] : 0.0;
            };
            auto l1run = [&]() -> double {
                if (kernel_only) { A.spmv_into(x, y); return yp[0]; }
                hpxpy::Array yy = A.spmv(x);        // full: alloc + kernel (wrapper)
                return yy.size() ? yy.data()[0] : 0.0;
            };
            auto allocrun = [&]() -> double {        // allocation alone (first-touch)
                hpxpy::Array yy(n, 0.0);
                return yy.size() ? yy.data()[0] : 0.0;
            };
            hpxpy::timing::result r0 = hpxpy::timing::measure(
                l0run, g_cfg.budget, g_cfg.min_reps, g_cfg.max_reps);
            hpxpy::timing::result r1 = hpxpy::timing::measure(
                l1run, g_cfg.budget, g_cfg.min_reps, g_cfg.max_reps);
            double ta = kernel_only ? 0.0 :
                hpxpy::timing::measure(allocrun, g_cfg.budget, g_cfg.min_reps,
                    g_cfg.max_reps).median_s;
            double t0 = r0.median_s, t1 = r1.median_s;
            std::printf("op=%s n=%zu threads=%d | L0 %.6gs (%dx) | L1 %.6gs (%dx) | "
                        "L1/L0=%.3f | alloc=%.6gs\n",
                g_cfg.op.c_str(), n, threads, t0, r0.reps, t1, r1.reps,
                t0 > 0 ? t1 / t0 : 0.0, ta);
            std::fflush(stdout);
            continue;
        }

        // Strided sum: n is the length of the BACKING buffer; stride=2, view_len=n/2.
        // L0 and L1 BOTH time the KERNEL ONLY — the buffer and the strided view are
        // built ONCE outside the timed region, just like the contiguous cases above.
        if (g_cfg.op == "strided_sum" || g_cfg.op == "strided_muls")
        {
            std::size_t const stride = 2;
            // Back buffer: 2× the requested size so the strided view covers n elements.
            std::size_t const full_n = n * stride;
            std::size_t const view_n = n;    // logical length of the strided view

            // Build the backing Array once (allocates the NUMA buffer, first-touch).
            hpxpy::Array backing = hpxpy::arange(full_n);
            // Take the strided view once — zero-copy, same buffer.
            hpxpy::Array sv = backing.view_strided(0, view_n,
                static_cast<std::ptrdiff_t>(stride));

            double const* base_ptr = backing.data();    // raw pointer into the buffer
            double const sc = g_cfg.scalar;             // runtime scalar (no const-fold)

            if (g_cfg.op == "strided_sum")
            {
                // L0: direct for_loop+reduction over base_ptr[i*stride].
                auto l0run = [&]() -> double {
                    double r = 0.0;
                    hpx::experimental::for_loop(hpx::execution::par,
                        std::size_t(0), view_n,
                        hpx::experimental::reduction(r, 0.0, std::plus<double>{}),
                        [base_ptr, stride](std::size_t i, double& acc) {
                            acc += base_ptr[i * stride];
                        });
                    return r;
                };
                // L1: the exact wrapper method on the strided Array view.
                auto l1run = [&]() -> double { return sv.sum(); };

                hpxpy::timing::result r0 = hpxpy::timing::measure(
                    l0run, g_cfg.budget, g_cfg.min_reps, g_cfg.max_reps);
                hpxpy::timing::result r1 = hpxpy::timing::measure(
                    l1run, g_cfg.budget, g_cfg.min_reps, g_cfg.max_reps);
                double v0 = l0run(), v1 = l1run();
                double t0 = r0.median_s, t1 = r1.median_s;
                double g0 = t0 > 0 ? view_n / t0 / 1e9 : 0.0;
                double g1 = t1 > 0 ? view_n / t1 / 1e9 : 0.0;
                std::printf(
                    "op=strided_sum n=%zu stride=%zu view_n=%zu threads=%d | "
                    "L0 %.6gs %.2f GEl/s (%dx) | L1 %.6gs %.2f GEl/s (%dx) | "
                    "L1/L0=%.3f%s\n",
                    full_n, stride, view_n, threads,
                    t0, g0, r0.reps, t1, g1, r1.reps,
                    t0 > 0 ? t1 / t0 : 0.0,
                    (v0 == v1) ? "" : "  [VALUE MISMATCH]");
                std::fflush(stdout);
            }
            else    // strided_muls: scalar multiply a[::2] * scalar -> new buffer
            {
                // L0: allocate a fresh result buffer + for_loop (mirrors what L1 does
                // via mul_scalar: alloc + transform). Both sides allocate in the timed
                // region so the comparison is symmetric (same as contiguous muls).
                auto l0run = [&]() -> double {
                    auto outp = std::make_shared<hpxpy::dvec>(view_n);
                    double* o = outp->data();
                    hpx::experimental::for_loop(hpx::execution::par,
                        std::size_t(0), view_n,
                        [base_ptr, stride, sc, o](std::size_t i) {
                            o[i] = base_ptr[i * stride] * sc;
                        });
                    return view_n ? o[0] : 0.0;
                };
                // L1: the exact wrapper method (unary path with stride!=1).
                auto l1run = [&]() -> double {
                    hpxpy::Array res = sv.mul_scalar(sc);
                    return res.size() ? res.data()[0] : 0.0;
                };

                hpxpy::timing::result r0 = hpxpy::timing::measure(
                    l0run, g_cfg.budget, g_cfg.min_reps, g_cfg.max_reps);
                hpxpy::timing::result r1 = hpxpy::timing::measure(
                    l1run, g_cfg.budget, g_cfg.min_reps, g_cfg.max_reps);
                double v0 = l0run(), v1 = l1run();
                double t0 = r0.median_s, t1 = r1.median_s;
                double g0 = t0 > 0 ? view_n / t0 / 1e9 : 0.0;
                double g1 = t1 > 0 ? view_n / t1 / 1e9 : 0.0;
                std::printf(
                    "op=strided_muls n=%zu stride=%zu view_n=%zu scalar=%.1f threads=%d | "
                    "L0 %.6gs %.2f GEl/s (%dx) | L1 %.6gs %.2f GEl/s (%dx) | "
                    "L1/L0=%.3f%s\n",
                    full_n, stride, view_n, sc, threads,
                    t0, g0, r0.reps, t1, g1, r1.reps,
                    t0 > 0 ? t1 / t0 : 0.0,
                    (v0 == v1) ? "" : "  [VALUE MISMATCH]");
                std::fflush(stdout);
            }
            continue;
        }

        if (g_cfg.op == "spmm")    // sparse x dense, kernel-only (C pre-allocated)
        {
            std::size_t const K = 16;
            hpxpy::CsrMatrix A = hpxpy::laplacian_1d(n);
            hpxpy::DenseMatrix B(n, K, 1.0);
            hpxpy::DenseMatrix C(n, K, 0.0);    // reused
            const std::int64_t* rp = A.row_ptr_data();
            const std::int64_t* ci = A.col_idx_data();
            const double* vp = A.values_data();
            const double* bp = B.data();
            double* cp = C.mutable_data();
            auto l0run = [&]() -> double {
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
            auto l1run = [&]() -> double { A.spmm_into(B, C); return cp[0]; };
            hpxpy::timing::result r0 = hpxpy::timing::measure(
                l0run, g_cfg.budget, g_cfg.min_reps, g_cfg.max_reps);
            hpxpy::timing::result r1 = hpxpy::timing::measure(
                l1run, g_cfg.budget, g_cfg.min_reps, g_cfg.max_reps);
            double t0 = r0.median_s, t1 = r1.median_s;
            std::printf("op=spmm n=%zu K=%zu threads=%d | L0 %.6gs (%dx) | "
                        "L1 %.6gs (%dx) | L1/L0=%.3f\n",
                n, K, threads, t0, r0.reps, t1, r1.reps, t0 > 0 ? t1 / t0 : 0.0);
            std::fflush(stdout);
            continue;
        }

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
        else if (a == "--dtype")
            g_cfg.dtype = next();
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
    // Avoid mmap thread-stack exhaustion (max_map_count) at high thread counts.
    hargs.emplace_back("--hpx:ini=hpx.stacks.use_guard_pages=0");
    std::vector<char*> hargv;
    for (auto& s : hargs)
        hargv.push_back(s.data());

    hpx::init_params params;
    return hpx::init(
        &hpx_main, static_cast<int>(hargv.size()), hargv.data(), params);
}
