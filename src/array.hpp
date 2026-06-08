// hpxpy::Array — the core data type, as a binding-agnostic C++/HPX header.
//
// A 1-D float64 array backed by an HPX compute::vector with a NUMA-aware host
// block_allocator (parallel first-touch). A THIN wrapper: HPX owns the storage,
// operations ARE HPX algorithms run in place over the contiguous buffer — no
// copies, no reassembly, no between-layers buffers. Nothing here depends on
// nanobind/Python, so the same code is used by the bindings (_core.cpp) AND by the
// C++ abstraction-penalty diagnostic (cpp_baseline/diag.cpp): the penalty is a
// C++ question (wrapper-C++ vs direct-C++), measured with this exact wrapper.
//
// SPDX-License-Identifier: MIT
#pragma once

#include <hpx/algorithm.hpp>
#include <hpx/execution.hpp>
#include <hpx/modules/compute_local.hpp>
#include <hpx/numeric.hpp>
#include <hpx/runtime_local/run_as_hpx_thread.hpp>
#include <hpx/threading_base/threading_base_fwd.hpp>

#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

namespace hpxpy {

using numa_alloc = hpx::compute::host::block_allocator<double>;
using dvec = hpx::compute::vector<double, numa_alloc>;

// Run f on an HPX thread, where the allocator's parallel first-touch and the
// parallel algorithms must execute. If we are already on an HPX thread (e.g. the
// diagnostic's hpx_main) run it directly — run_as_hpx_thread asserts it is NOT
// called from an HPX thread. If we are on a foreign thread (e.g. the Python main
// thread) hop onto the pool. This keeps the wrapper identical in both contexts.
template <typename F>
inline void on_hpx_thread(F&& f)
{
    if (hpx::threads::get_self_ptr() != nullptr)
        std::forward<F>(f)();
    else
        hpx::run_as_hpx_thread(std::forward<F>(f));
}

class Array
{
public:
    Array() = default;
    Array(std::size_t n, double fill) : size_(n)
    {
        on_hpx_thread([&] { data_ = std::make_shared<dvec>(n, fill); });
    }

    std::size_t size() const { return size_; }
    std::size_t ndim() const { return 1; }

    // Raw contiguous buffer (nullptr if empty). Used by the diagnostic to run the
    // direct (L0) reduce over the exact same memory the wrapper (L1) reduces.
    const double* data() const { return data_ ? data_->data() : nullptr; }

    // Reductions — thin wrappers over HPX algorithms, run in place on the buffer.
    // The result is a scalar; the penalty of these vs a direct C++ call must be ~1.
    double sum() const
    {
        if (size_ == 0)
            return 0.0;
        const double* p = data_->data();
        return hpx::reduce(hpx::execution::par, p, p + size_, 0.0);
    }
    double min() const
    {
        if (size_ == 0)
            throw std::invalid_argument("min() of an empty Array");
        const double* p = data_->data();
        return hpx::reduce(hpx::execution::par, p, p + size_,
            std::numeric_limits<double>::infinity(),
            [](double x, double y) { return x < y ? x : y; });
    }
    double max() const
    {
        if (size_ == 0)
            throw std::invalid_argument("max() of an empty Array");
        const double* p = data_->data();
        return hpx::reduce(hpx::execution::par, p, p + size_,
            -std::numeric_limits<double>::infinity(),
            [](double x, double y) { return x > y ? x : y; });
    }
    // Fused dot product: a SINGLE pass (multiply+accumulate) via transform_reduce —
    // never a materialized product array then a sum.
    double dot(Array const& other) const
    {
        if (size_ != other.size_)
            throw std::invalid_argument("dot(): size mismatch");
        if (size_ == 0)
            return 0.0;
        const double* p = data_->data();
        const double* q = other.data_->data();
        return hpx::transform_reduce(hpx::execution::par, p, p + size_, q, 0.0);
    }

    // Element-wise binary ops -> a NEW Array (allocating a new top-level array is
    // allowed; never a between-layers buffer). One fused hpx::transform pass.
    Array add(Array const& o) const { return binary(o, std::plus<double>{}); }
    Array sub(Array const& o) const { return binary(o, std::minus<double>{}); }
    Array mul(Array const& o) const { return binary(o, std::multiplies<double>{}); }
    Array div(Array const& o) const { return binary(o, std::divides<double>{}); }

    // Scalar broadcast -> new Array (one unary hpx::transform pass). The r* forms
    // are the reflected ops (scalar on the left: s - a, s / a).
    Array add_scalar(double s) const { return unary([s](double x) { return x + s; }); }
    Array sub_scalar(double s) const { return unary([s](double x) { return x - s; }); }
    Array rsub_scalar(double s) const { return unary([s](double x) { return s - x; }); }
    Array mul_scalar(double s) const { return unary([s](double x) { return x * s; }); }
    Array div_scalar(double s) const { return unary([s](double x) { return x / s; }); }
    Array rdiv_scalar(double s) const { return unary([s](double x) { return s / x; }); }

    // 0, 1, 2, ..., n-1. The block_allocator first-touches at construction; the
    // parallel for_loop writes the ramp on the same HPX workers (stays NUMA-local).
    static Array iota(std::size_t n)
    {
        Array a;
        a.size_ = n;
        on_hpx_thread([&] {
            a.data_ = std::make_shared<dvec>(n);
            double* p = a.data_->data();
            hpx::experimental::for_loop(hpx::execution::par, std::size_t(0), n,
                [p](std::size_t i) { p[i] = static_cast<double>(i); });
        });
        return a;
    }

private:
    // Only the allocation (block_allocator first-touch) is wrapped in
    // on_hpx_thread; the transform runs DIRECTLY — wrapping it in the lambda
    // prevents the optimizer from inlining `op` into the inner loop, which costs
    // ~2x on compute-light ops at low thread counts (caught by the diag ladder).
    // The reductions already call their hpx:: algorithm directly for the same reason.
    template <typename Op>
    Array unary(Op op) const
    {
        Array r;
        r.size_ = size_;
        std::size_t const n = size_;
        const double* p = n ? data_->data() : nullptr;
        on_hpx_thread([&] { r.data_ = std::make_shared<dvec>(n); });
        double* out = n ? r.data_->data() : nullptr;
        hpx::transform(hpx::execution::par, p, p + n, out, op);
        return r;
    }

    template <typename Op>
    Array binary(Array const& o, Op op) const
    {
        if (size_ != o.size_)
            throw std::invalid_argument("element-wise op: size mismatch");
        Array r;
        r.size_ = size_;
        std::size_t const n = size_;
        const double* p = n ? data_->data() : nullptr;
        const double* q = n ? o.data_->data() : nullptr;
        on_hpx_thread([&] { r.data_ = std::make_shared<dvec>(n); });
        double* out = n ? r.data_->data() : nullptr;
        hpx::transform(hpx::execution::par, p, p + n, q, out, op);
        return r;
    }

    std::shared_ptr<dvec> data_;
    std::size_t size_ = 0;
};

inline Array zeros(std::size_t n) { return Array(n, 0.0); }
inline Array full(std::size_t n, double value) { return Array(n, value); }
inline Array arange(std::size_t n) { return Array::iota(n); }

}    // namespace hpxpy
