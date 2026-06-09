// hpxpy::Array — the core data type, as a binding-agnostic C++/HPX header.
//
// A 1-D float64 array. Backing is uniform: `owner_` (a type-erased shared_ptr that
// keeps the buffer alive) + `base_` (the start of THIS array's data) + `size_`. Three
// flavours, all the same machine:
//   - owning:   owner_ holds an HPX compute::vector (NUMA block_allocator,
//               parallel first-touch); base_ = its data().
//   - view:     a[i:j] shares the parent's owner_; base_ = parent.base_ + start
//               (contiguous, step-1; numpy view semantics).
//   - borrow:   owner_ keeps an external buffer alive (e.g. a NumPy array via a
//               GIL-aware keep-alive); base_ points into it (zero-copy from_numpy).
// Operations ARE HPX algorithms run in place over base_ — no copies, no reassembly,
// no between-layers buffers. Nothing here depends on nanobind/Python.
//
// Strided views (stride_ != 1): element i lives at base_[i*stride_]. A signed
// stride_ supports reverse views (negative step). When stride_==1 the original
// contiguous fast-path (raw pointer range) is taken unchanged, so contiguous
// performance is unaffected.
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
#include <vector>

namespace hpxpy {

using numa_alloc = hpx::compute::host::block_allocator<double>;
using dvec = hpx::compute::vector<double, numa_alloc>;

// Run f on an HPX thread, where the allocator's parallel first-touch must execute. If
// already on an HPX thread (e.g. the diagnostic's hpx_main) run it directly —
// run_as_hpx_thread asserts it is NOT called from an HPX thread. From a foreign thread
// (e.g. the Python main thread) hop onto the pool. Identical in both contexts.
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
    Array(std::size_t n, double fill) { alloc_(n, fill); }

    std::size_t size() const { return size_; }
    std::size_t ndim() const { return 1; }
    std::ptrdiff_t stride() const { return stride_; }

    // Start of this array's data (nullptr if empty). The const form is also used by
    // the diagnostic and by to_numpy (zero-copy export).
    const double* data() const { return base_; }
    double* mutable_data() { return base_; }

    // Element access / scalar write — direct host-memory access (no parallel work).
    // Indices are assumed already normalized by the binding layer.
    double getitem(std::size_t i) const { return base_[static_cast<std::ptrdiff_t>(i) * stride_]; }
    void setitem(std::size_t i, double v) { base_[static_cast<std::ptrdiff_t>(i) * stride_] = v; }

    // Slice assignment a[start:start+n] = ... (contiguous; binding computes start/n).
    // fill_range broadcasts a scalar; assign_range copies an Array of length n in.
    void fill_range(std::size_t start, std::size_t n, double v)
    {
        if (n)
            hpx::fill(hpx::execution::par, base_ + start, base_ + start + n, v);
    }
    void assign_range(std::size_t start, Array const& rhs)
    {
        if (rhs.size_)
            hpx::copy(hpx::execution::par, rhs.base_, rhs.base_ + rhs.size_,
                base_ + start);
    }

    // Contiguous view [start, start+n) sharing this array's buffer (numpy a[i:j]).
    Array view(std::size_t start, std::size_t n) const
    {
        Array r;
        r.owner_ = owner_;    // share the keep-alive (owned or borrowed parent)
        r.base_ = base_ ? base_ + static_cast<std::ptrdiff_t>(start) * stride_ : nullptr;
        r.size_ = n;
        r.stride_ = stride_;
        return r;
    }

    // Strided view: element i is at base_[i*result.stride_] in the result.
    // start and step are in THIS array's logical index space; step may be negative.
    Array view_strided(std::ptrdiff_t start, std::size_t n, std::ptrdiff_t step) const
    {
        Array r;
        r.owner_ = owner_;
        r.base_ = base_ ? base_ + start * stride_ : nullptr;
        r.stride_ = stride_ * step;
        r.size_ = n;
        return r;
    }

    // Wrap an external buffer (non-owning); `keepalive` keeps it alive. Used by the
    // zero-copy from_numpy borrow; the keep-alive is type-erased so this header stays
    // nanobind-free (the binding supplies a GIL-aware deleter).
    static Array borrow(double* p, std::size_t n, std::shared_ptr<void> keepalive)
    {
        Array r;
        r.owner_ = std::move(keepalive);
        r.base_ = p;
        r.size_ = n;
        return r;
    }

    // Reductions — thin wrappers over HPX algorithms, run in place on the buffer.
    // Contiguous path (stride==1): raw pointer range passed to hpx::reduce.
    // Strided path: hpx::experimental::for_loop with reduction object, zero-copy.
    double sum() const
    {
        if (size_ == 0)
            return 0.0;
        if (stride_ == 1)
            return hpx::reduce(hpx::execution::par, base_, base_ + size_, 0.0);
        double r = 0.0;
        double* b = base_;
        std::ptrdiff_t s = stride_;
        hpx::experimental::for_loop(hpx::execution::par, std::size_t(0), size_,
            hpx::experimental::reduction(r, 0.0, std::plus<double>{}),
            [b, s](std::size_t i, double& acc) { acc += b[static_cast<std::ptrdiff_t>(i) * s]; });
        return r;
    }
    double min() const
    {
        if (size_ == 0)
            throw std::invalid_argument("min() of an empty Array");
        if (stride_ == 1)
            return hpx::reduce(hpx::execution::par, base_, base_ + size_,
                std::numeric_limits<double>::infinity(),
                [](double x, double y) { return x < y ? x : y; });
        double r = std::numeric_limits<double>::infinity();
        double* b = base_;
        std::ptrdiff_t s = stride_;
        hpx::experimental::for_loop(hpx::execution::par, std::size_t(0), size_,
            hpx::experimental::reduction(r, std::numeric_limits<double>::infinity(),
                [](double x, double y) { return x < y ? x : y; }),
            [b, s](std::size_t i, double& acc) {
                double v = b[static_cast<std::ptrdiff_t>(i) * s];
                if (v < acc) acc = v;
            });
        return r;
    }
    double max() const
    {
        if (size_ == 0)
            throw std::invalid_argument("max() of an empty Array");
        if (stride_ == 1)
            return hpx::reduce(hpx::execution::par, base_, base_ + size_,
                -std::numeric_limits<double>::infinity(),
                [](double x, double y) { return x > y ? x : y; });
        double r = -std::numeric_limits<double>::infinity();
        double* b = base_;
        std::ptrdiff_t s = stride_;
        hpx::experimental::for_loop(hpx::execution::par, std::size_t(0), size_,
            hpx::experimental::reduction(r, -std::numeric_limits<double>::infinity(),
                [](double x, double y) { return x > y ? x : y; }),
            [b, s](std::size_t i, double& acc) {
                double v = b[static_cast<std::ptrdiff_t>(i) * s];
                if (v > acc) acc = v;
            });
        return r;
    }
    // Fused dot product: a SINGLE pass (multiply+accumulate) via transform_reduce
    // (contiguous) or for_loop reduction (strided).
    double dot(Array const& other) const
    {
        if (size_ != other.size_)
            throw std::invalid_argument("dot(): size mismatch");
        if (size_ == 0)
            return 0.0;
        if (stride_ == 1 && other.stride_ == 1)
            return hpx::transform_reduce(
                hpx::execution::par, base_, base_ + size_, other.base_, 0.0);
        double r = 0.0;
        double* ba = base_;
        double* bb = other.base_;
        std::ptrdiff_t sa = stride_;
        std::ptrdiff_t sb = other.stride_;
        hpx::experimental::for_loop(hpx::execution::par, std::size_t(0), size_,
            hpx::experimental::reduction(r, 0.0, std::plus<double>{}),
            [ba, bb, sa, sb](std::size_t i, double& acc) {
                acc += ba[static_cast<std::ptrdiff_t>(i) * sa] *
                       bb[static_cast<std::ptrdiff_t>(i) * sb];
            });
        return r;
    }

    // Element-wise binary ops -> a NEW (owning) Array. One hpx::transform pass
    // (contiguous) or for_loop element-wise (strided; result is always contiguous).
    Array add(Array const& o) const { return binary(o, std::plus<double>{}); }
    Array sub(Array const& o) const { return binary(o, std::minus<double>{}); }
    Array mul(Array const& o) const { return binary(o, std::multiplies<double>{}); }
    Array div(Array const& o) const { return binary(o, std::divides<double>{}); }

    // Scalar broadcast -> new Array. The r* forms are reflected (s - a, s / a).
    Array add_scalar(double s) const { return unary([s](double x) { return x + s; }); }
    Array sub_scalar(double s) const { return unary([s](double x) { return x - s; }); }
    Array rsub_scalar(double s) const { return unary([s](double x) { return s - x; }); }
    Array mul_scalar(double s) const { return unary([s](double x) { return x * s; }); }
    Array div_scalar(double s) const { return unary([s](double x) { return x / s; }); }
    Array rdiv_scalar(double s) const { return unary([s](double x) { return s / x; }); }

    // Deep copy -> a new independent (owning) Array (numpy a.copy()).
    Array copy() const
    {
        Array r;
        r.alloc_(size_);
        if (size_) {
            if (stride_ == 1) {
                hpx::copy(hpx::execution::par, base_, base_ + size_, r.base_);
            } else {
                double* src = base_;
                double* dst = r.base_;
                std::ptrdiff_t s = stride_;
                hpx::experimental::for_loop(hpx::execution::par,
                    std::size_t(0), size_,
                    [src, dst, s](std::size_t i) {
                        dst[i] = src[static_cast<std::ptrdiff_t>(i) * s];
                    });
            }
        }
        return r;
    }

    // Sort ascending IN PLACE (numpy a.sort(); mutates this view's range, no alloc).
    // Strided: gather into a temporary contiguous buffer, sort, scatter back.
    void sort()
    {
        if (size_ < 2)
            return;
        if (stride_ == 1) {
            hpx::sort(hpx::execution::par, base_, base_ + size_);
        } else {
            // Gather strided elements into a temporary contiguous dvec, sort, scatter.
            double* src = base_;
            std::ptrdiff_t s = stride_;
            std::vector<double> tmp(size_);
            for (std::size_t i = 0; i < size_; ++i)
                tmp[i] = src[static_cast<std::ptrdiff_t>(i) * s];
            hpx::sort(hpx::execution::par, tmp.begin(), tmp.end());
            for (std::size_t i = 0; i < size_; ++i)
                src[static_cast<std::ptrdiff_t>(i) * s] = tmp[i];
        }
    }
    bool is_sorted() const
    {
        if (size_ < 2)
            return true;
        if (stride_ == 1)
            return hpx::is_sorted(hpx::execution::par, base_, base_ + size_);
        // Strided: one zero-copy pass — AND-reduce "each logical element <= its
        // successor" over the n-1 adjacent pairs (no gather).
        bool ok = true;
        double* b = base_;
        std::ptrdiff_t s = stride_;
        hpx::experimental::for_loop(hpx::execution::par, std::size_t(0), size_ - 1,
            hpx::experimental::reduction(ok, true, std::logical_and<bool>{}),
            [b, s](std::size_t i, bool& acc) {
                acc = acc && (b[static_cast<std::ptrdiff_t>(i) * s] <=
                              b[(static_cast<std::ptrdiff_t>(i) + 1) * s]);
            });
        return ok;
    }

    // Inclusive prefix sum -> a NEW (owning) Array (numpy a.cumsum()).
    // Strided: gather into the result buffer then do the scan in place.
    Array cumsum() const
    {
        Array r;
        r.alloc_(size_);
        if (size_) {
            if (stride_ == 1) {
                hpx::inclusive_scan(hpx::execution::par, base_, base_ + size_, r.base_);
            } else {
                double* src = base_;
                double* dst = r.base_;
                std::ptrdiff_t s = stride_;
                // Gather strided into contiguous result, then scan in place.
                hpx::experimental::for_loop(hpx::execution::par,
                    std::size_t(0), size_,
                    [src, dst, s](std::size_t i) {
                        dst[i] = src[static_cast<std::ptrdiff_t>(i) * s];
                    });
                hpx::inclusive_scan(hpx::execution::par, dst, dst + size_, dst);
            }
        }
        return r;
    }

    // 0, 1, 2, ..., n-1. block_allocator first-touches at allocation; the parallel
    // for_loop (run directly) writes the ramp on the same HPX workers (NUMA-local).
    static Array iota(std::size_t n)
    {
        Array a;
        a.alloc_(n);
        double* p = a.base_;
        if (n)
            hpx::experimental::for_loop(hpx::execution::par, std::size_t(0), n,
                [p](std::size_t i) { p[i] = static_cast<double>(i); });
        return a;
    }

private:
    // Allocate an owning NUMA buffer of n elements (value-initialized to fill) on an
    // HPX thread, and point base_/owner_ at it. Only the allocation is wrapped in
    // on_hpx_thread; kernels run DIRECTLY at the top level of each method so the op
    // inlines into the algorithm's inner loop (the op-inlining lesson).
    void alloc_(std::size_t n, double fill = 0.0)
    {
        size_ = n;
        on_hpx_thread([&] {
            auto d = std::make_shared<dvec>(n, fill);
            base_ = n ? d->data() : nullptr;
            owner_ = std::move(d);
        });
    }

    template <typename Op>
    Array unary(Op op) const
    {
        Array r;
        r.alloc_(size_);
        if (size_) {
            if (stride_ == 1) {
                hpx::transform(hpx::execution::par, base_, base_ + size_, r.base_, op);
            } else {
                double* src = base_;
                double* dst = r.base_;
                std::ptrdiff_t s = stride_;
                hpx::experimental::for_loop(hpx::execution::par,
                    std::size_t(0), size_,
                    [src, dst, s, op](std::size_t i) {
                        dst[i] = op(src[static_cast<std::ptrdiff_t>(i) * s]);
                    });
            }
        }
        return r;
    }

    template <typename Op>
    Array binary(Array const& o, Op op) const
    {
        if (size_ != o.size_)
            throw std::invalid_argument("element-wise op: size mismatch");
        Array r;
        r.alloc_(size_);
        if (size_) {
            if (stride_ == 1 && o.stride_ == 1) {
                hpx::transform(
                    hpx::execution::par, base_, base_ + size_, o.base_, r.base_, op);
            } else {
                double* ba = base_;
                double* bb = o.base_;
                double* dst = r.base_;
                std::ptrdiff_t sa = stride_;
                std::ptrdiff_t sb = o.stride_;
                hpx::experimental::for_loop(hpx::execution::par,
                    std::size_t(0), size_,
                    [ba, bb, dst, sa, sb, op](std::size_t i) {
                        dst[i] = op(ba[static_cast<std::ptrdiff_t>(i) * sa],
                                    bb[static_cast<std::ptrdiff_t>(i) * sb]);
                    });
            }
        }
        return r;
    }

    std::shared_ptr<void> owner_;    // keeps the backing alive (dvec, or numpy ref)
    double* base_ = nullptr;         // start of this array's data (offset folded in)
    std::size_t size_ = 0;
    std::ptrdiff_t stride_ = 1;      // element stride (1 = contiguous, negative = reverse)
};

inline Array zeros(std::size_t n) { return Array(n, 0.0); }
inline Array full(std::size_t n, double value) { return Array(n, value); }
inline Array arange(std::size_t n) { return Array::iota(n); }

}    // namespace hpxpy
