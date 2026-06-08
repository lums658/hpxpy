// hpxpy::Array — the core data type, as a binding-agnostic C++/HPX header.
//
// A 1-D float64 array backed by an HPX compute::vector with a NUMA-aware host
// block_allocator (parallel first-touch). A THIN wrapper: HPX owns the storage,
// operations ARE HPX algorithms run in place over the contiguous buffer — no
// copies, no reassembly, no between-layers buffers. Nothing here depends on
// nanobind/Python, so the same code is used by the bindings (_core.cpp) AND by the
// C++ abstraction-penalty diagnostic (cpp_baseline/diag.cpp).
//
// An Array may be a VIEW: it shares its parent's shared_ptr<dvec> (keeping the
// buffer alive) plus an offset_; the base pointer is data_->data() + offset_. A
// slice a[i:j] is a contiguous (step-1) view — no copy, numpy semantics. New
// (owning) arrays have offset_ == 0, so they stay contiguous from index 0.
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

    // Base of this (possibly offset) view; nullptr if empty. The public const form
    // is also used by the diagnostic to run the direct (L0) reduce over the exact
    // same memory the wrapper (L1) reduces.
    const double* data() const { return data_ ? data_->data() + offset_ : nullptr; }

    // Mutable base — for kernels in sibling wrappers (e.g. CsrMatrix::spmv) that
    // write into a freshly-allocated result Array. Same pointer as data().
    double* mutable_data() { return data_ ? data_->data() + offset_ : nullptr; }

    // Element access / scalar write — direct host-memory access (no parallel work,
    // no allocation). Indices are assumed already normalized by the binding layer.
    double getitem(std::size_t i) const { return (data_->data() + offset_)[i]; }
    void setitem(std::size_t i, double v) { (data_->data() + offset_)[i] = v; }

    // Contiguous view [start, start+n) sharing this array's buffer (numpy a[i:j]).
    Array view(std::size_t start, std::size_t n) const
    {
        return Array(data_, offset_ + start, n);
    }

    // Reductions — thin wrappers over HPX algorithms, run in place on the buffer.
    double sum() const
    {
        if (size_ == 0)
            return 0.0;
        const double* p = cbase();
        return hpx::reduce(hpx::execution::par, p, p + size_, 0.0);
    }
    double min() const
    {
        if (size_ == 0)
            throw std::invalid_argument("min() of an empty Array");
        const double* p = cbase();
        return hpx::reduce(hpx::execution::par, p, p + size_,
            std::numeric_limits<double>::infinity(),
            [](double x, double y) { return x < y ? x : y; });
    }
    double max() const
    {
        if (size_ == 0)
            throw std::invalid_argument("max() of an empty Array");
        const double* p = cbase();
        return hpx::reduce(hpx::execution::par, p, p + size_,
            -std::numeric_limits<double>::infinity(),
            [](double x, double y) { return x > y ? x : y; });
    }
    // Fused dot product: a SINGLE pass (multiply+accumulate) via transform_reduce.
    double dot(Array const& other) const
    {
        if (size_ != other.size_)
            throw std::invalid_argument("dot(): size mismatch");
        if (size_ == 0)
            return 0.0;
        const double* p = cbase();
        const double* q = other.cbase();
        return hpx::transform_reduce(hpx::execution::par, p, p + size_, q, 0.0);
    }

    // Element-wise binary ops -> a NEW (owning, offset-0) Array. One hpx::transform.
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

    // Deep copy -> a new independent (owning, offset-0) Array (numpy a.copy()).
    Array copy() const
    {
        Array r;
        r.size_ = size_;
        std::size_t const n = size_;
        const double* p = cbase();
        on_hpx_thread([&] { r.data_ = std::make_shared<dvec>(n); });
        double* out = n ? r.data_->data() : nullptr;
        hpx::copy(hpx::execution::par, p, p + n, out);
        return r;
    }

    // Sort ascending IN PLACE (numpy a.sort(); mutates this view's range, no alloc).
    void sort()
    {
        if (size_ < 2)
            return;
        double* p = base();
        hpx::sort(hpx::execution::par, p, p + size_);
    }
    bool is_sorted() const
    {
        if (size_ < 2)
            return true;
        const double* p = cbase();
        return hpx::is_sorted(hpx::execution::par, p, p + size_);
    }

    // Inclusive prefix sum -> a NEW (owning, offset-0) Array (numpy a.cumsum()).
    Array cumsum() const
    {
        Array r;
        r.size_ = size_;
        std::size_t const n = size_;
        const double* p = cbase();
        on_hpx_thread([&] { r.data_ = std::make_shared<dvec>(n); });
        double* out = n ? r.data_->data() : nullptr;
        hpx::inclusive_scan(hpx::execution::par, p, p + n, out);
        return r;
    }

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
    // View constructor: share an existing buffer at an offset (no allocation).
    Array(std::shared_ptr<dvec> d, std::size_t off, std::size_t n)
      : data_(std::move(d)), offset_(off), size_(n)
    {
    }

    // Base of this view (data + offset); nullptr if empty.
    const double* cbase() const { return data_ ? data_->data() + offset_ : nullptr; }
    double* base() { return data_ ? data_->data() + offset_ : nullptr; }

    // Only the allocation (block_allocator first-touch) is wrapped in on_hpx_thread;
    // the transform runs DIRECTLY — wrapping it in the lambda prevents the optimizer
    // from inlining `op` into the inner loop, ~2x on compute-light ops at low thread
    // counts (caught by the diag ladder). Reductions call their algorithm directly
    // for the same reason. The result is a fresh owning array (offset 0).
    template <typename Op>
    Array unary(Op op) const
    {
        Array r;
        r.size_ = size_;
        std::size_t const n = size_;
        const double* p = cbase();
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
        const double* p = cbase();
        const double* q = o.cbase();
        on_hpx_thread([&] { r.data_ = std::make_shared<dvec>(n); });
        double* out = n ? r.data_->data() : nullptr;
        hpx::transform(hpx::execution::par, p, p + n, q, out, op);
        return r;
    }

    std::shared_ptr<dvec> data_;
    std::size_t offset_ = 0;
    std::size_t size_ = 0;
};

inline Array zeros(std::size_t n) { return Array(n, 0.0); }
inline Array full(std::size_t n, double value) { return Array(n, value); }
inline Array arange(std::size_t n) { return Array::iota(n); }

}    // namespace hpxpy
