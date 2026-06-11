// hpxpy::Array — the core data type, as a binding-agnostic C++/HPX header.
//
// An N-D float64 array. Backing is uniform: `owner_` (a type-erased shared_ptr that
// keeps the buffer alive) + `base_` (the start of THIS array's data) + `shape_`/
// `strides_` (N-D shape and element strides, row-major / C-order). Three flavours,
// all the same machine:
//   - owning:   owner_ holds an HPX compute::vector (NUMA block_allocator,
//               parallel first-touch); base_ = its data().
//   - view:     a[i:j] shares the parent's owner_; base_ = parent.base_ + start
//               (contiguous step-1 or strided; numpy view semantics).
//   - borrow:   owner_ keeps an external buffer alive (e.g. a NumPy array via a
//               GIL-aware keep-alive); base_ points into it (zero-copy from_numpy).
// Operations ARE HPX algorithms run in place over base_ — no copies, no reassembly,
// no between-layers buffers. Nothing here depends on nanobind/Python.
//
// N-D layout (stage 1): owning N-D arrays are always contiguous row-major.
// 1-D views (stride != 1): element i lives at base_[i*strides_[0]]. A signed
// stride supports reverse views (negative step). When is_contiguous() the original
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
#include <numeric>
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

    // 1-D constructor (backward-compatible): shape={n}, strides={1}.
    Array(std::size_t n, double fill) { alloc_(n, fill); }

    // N-D constructor: shape may have any number of dims; row-major strides computed.
    Array(std::vector<std::size_t> shape, double fill)
    {
        std::size_t total = 1;
        for (auto d : shape) total *= d;
        alloc_nd_(std::move(shape), total, fill);
    }

    std::size_t size() const { return size_; }
    std::size_t ndim() const { return shape_.size(); }

    // shape/strides accessors (N-D).
    std::vector<std::size_t> const& shape() const { return shape_; }
    std::vector<std::ptrdiff_t> const& strides() const { return strides_; }

    // 1-D backward-compat: stride() returns strides_[0] (1 for contiguous, !=1 for strided views).
    std::ptrdiff_t stride() const { return strides_.empty() ? 1 : strides_[0]; }

    // True iff strides are the row-major strides for shape_ (fast-path predicate).
    bool is_contiguous() const
    {
        if (strides_.empty()) return true;
        // For a contiguous C-order array: strides_[ndim-1]==1, strides_[k] == shape_[k+1]*strides_[k+1]
        if (strides_.back() != 1) return false;
        for (std::size_t k = strides_.size() - 1; k-- > 0; ) {
            if (strides_[k] != static_cast<std::ptrdiff_t>(shape_[k + 1]) * strides_[k + 1])
                return false;
        }
        return true;
    }

    // Start of this array's data (nullptr if empty). The const form is also used by
    // the diagnostic and by to_numpy (zero-copy export).
    const double* data() const { return base_; }
    double* mutable_data() { return base_; }

    // 1-D element access: index in logical index space; strides_[0] applied.
    // (Indices are assumed already normalized by the binding layer.)
    double getitem(std::size_t i) const
    {
        return base_[static_cast<std::ptrdiff_t>(i) * strides_[0]];
    }
    void setitem(std::size_t i, double v)
    {
        base_[static_cast<std::ptrdiff_t>(i) * strides_[0]] = v;
    }

    // N-D multi-index access: element at idx[] using strides_.
    std::size_t linear_offset(std::vector<std::size_t> const& idx) const
    {
        std::ptrdiff_t off = 0;
        for (std::size_t k = 0; k < idx.size(); ++k)
            off += static_cast<std::ptrdiff_t>(idx[k]) * strides_[k];
        return static_cast<std::size_t>(off);
    }
    double getitem(std::vector<std::size_t> const& idx) const
    {
        return base_[linear_offset(idx)];
    }
    void setitem(std::vector<std::size_t> const& idx, double v)
    {
        base_[linear_offset(idx)] = v;
    }

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
    // Returns a 1-D view: shape_={n}, strides_={parent_stride}.
    Array view(std::size_t start, std::size_t n) const
    {
        Array r;
        r.owner_ = owner_;    // share the keep-alive (owned or borrowed parent)
        r.base_ = base_ ? base_ + static_cast<std::ptrdiff_t>(start) * strides_[0] : nullptr;
        r.size_ = n;
        r.shape_ = {n};
        r.strides_ = {strides_[0]};
        return r;
    }

    // Strided view: element i is at base_[i*result.strides_[0]] in the result.
    // start and step are in THIS array's logical index space; step may be negative.
    // Returns a 1-D view: shape_={n}, strides_={parent_stride*step}.
    Array view_strided(std::ptrdiff_t start, std::size_t n, std::ptrdiff_t step) const
    {
        Array r;
        r.owner_ = owner_;
        r.base_ = base_ ? base_ + start * strides_[0] : nullptr;
        r.strides_ = {strides_[0] * step};
        r.shape_ = {n};
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
        r.shape_ = {n};
        r.strides_ = {1};
        return r;
    }

    // N-D borrow: wrap an external C-contiguous N-D buffer (row-major). Computes
    // row-major strides from shape; `keepalive` keeps the buffer alive (from_numpy).
    static Array borrow_nd(double* p, std::vector<std::size_t> shape,
        std::shared_ptr<void> keepalive)
    {
        Array r;
        r.owner_ = std::move(keepalive);
        r.base_ = p;
        r.shape_ = std::move(shape);
        std::size_t total = 1;
        for (std::size_t d : r.shape_)
            total *= d;
        r.size_ = total;
        r.strides_.assign(r.shape_.size(), 1);
        for (std::size_t k = r.shape_.size(); k-- > 1;)
            r.strides_[k - 1] = static_cast<std::ptrdiff_t>(r.shape_[k]) * r.strides_[k];
        return r;
    }

    // Reductions — thin wrappers over HPX algorithms, run in place on the buffer.
    // Contiguous path (is_contiguous()): raw pointer range passed to hpx::reduce.
    // Strided path: hpx::experimental::for_loop with reduction object, zero-copy.
    double sum() const
    {
        if (size_ == 0)
            return 0.0;
        if (is_contiguous())
            return hpx::reduce(hpx::execution::par, base_, base_ + size_, 0.0);
        double r = 0.0;
        double* b = base_;
        std::ptrdiff_t s = strides_[0];
        hpx::experimental::for_loop(hpx::execution::par, std::size_t(0), size_,
            hpx::experimental::reduction(r, 0.0, std::plus<double>{}),
            [b, s](std::size_t i, double& acc) { acc += b[static_cast<std::ptrdiff_t>(i) * s]; });
        return r;
    }
    double min() const
    {
        if (size_ == 0)
            throw std::invalid_argument("min() of an empty Array");
        if (is_contiguous())
            return hpx::reduce(hpx::execution::par, base_, base_ + size_,
                std::numeric_limits<double>::infinity(),
                [](double x, double y) { return x < y ? x : y; });
        double r = std::numeric_limits<double>::infinity();
        double* b = base_;
        std::ptrdiff_t s = strides_[0];
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
        if (is_contiguous())
            return hpx::reduce(hpx::execution::par, base_, base_ + size_,
                -std::numeric_limits<double>::infinity(),
                [](double x, double y) { return x > y ? x : y; });
        double r = -std::numeric_limits<double>::infinity();
        double* b = base_;
        std::ptrdiff_t s = strides_[0];
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
        if (is_contiguous() && other.is_contiguous())
            return hpx::transform_reduce(
                hpx::execution::par, base_, base_ + size_, other.base_, 0.0);
        double r = 0.0;
        double* ba = base_;
        double* bb = other.base_;
        std::ptrdiff_t sa = strides_[0];
        std::ptrdiff_t sb = other.strides_[0];
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
            if (is_contiguous()) {
                hpx::copy(hpx::execution::par, base_, base_ + size_, r.base_);
            } else {
                double* src = base_;
                double* dst = r.base_;
                std::ptrdiff_t s = strides_[0];
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
        if (is_contiguous()) {
            hpx::sort(hpx::execution::par, base_, base_ + size_);
        } else {
            // Gather strided elements into a temporary contiguous dvec, sort, scatter.
            double* src = base_;
            std::ptrdiff_t s = strides_[0];
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
        if (is_contiguous())
            return hpx::is_sorted(hpx::execution::par, base_, base_ + size_);
        // Strided: one zero-copy pass — AND-reduce "each logical element <= its
        // successor" over the n-1 adjacent pairs (no gather).
        bool ok = true;
        double* b = base_;
        std::ptrdiff_t s = strides_[0];
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
            if (is_contiguous()) {
                hpx::inclusive_scan(hpx::execution::par, base_, base_ + size_, r.base_);
            } else {
                double* src = base_;
                double* dst = r.base_;
                std::ptrdiff_t s = strides_[0];
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
    // Allocate a 1-D owning NUMA buffer of n elements (value-initialized to fill) on an
    // HPX thread, and point base_/owner_/shape_/strides_/size_ at it.
    void alloc_(std::size_t n, double fill = 0.0)
    {
        size_ = n;
        shape_ = {n};
        strides_ = {1};
        on_hpx_thread([&] {
            auto d = std::make_shared<dvec>(n, fill);
            base_ = n ? d->data() : nullptr;
            owner_ = std::move(d);
        });
    }

    // Allocate an N-D owning NUMA buffer: shape already provided; total = product(shape).
    void alloc_nd_(std::vector<std::size_t> shape, std::size_t total, double fill)
    {
        shape_ = std::move(shape);
        size_ = total;
        // Compute row-major strides: strides_[ndim-1]=1, strides_[k]=shape_[k+1]*strides_[k+1]
        std::size_t nd = shape_.size();
        strides_.resize(nd);
        if (nd > 0) {
            strides_[nd - 1] = 1;
            for (std::size_t k = nd - 1; k-- > 0; )
                strides_[k] = static_cast<std::ptrdiff_t>(shape_[k + 1]) * strides_[k + 1];
        }
        std::size_t n = total;
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
            if (is_contiguous()) {
                hpx::transform(hpx::execution::par, base_, base_ + size_, r.base_, op);
            } else {
                double* src = base_;
                double* dst = r.base_;
                std::ptrdiff_t s = strides_[0];
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
            if (is_contiguous() && o.is_contiguous()) {
                hpx::transform(
                    hpx::execution::par, base_, base_ + size_, o.base_, r.base_, op);
            } else {
                double* ba = base_;
                double* bb = o.base_;
                double* dst = r.base_;
                std::ptrdiff_t sa = strides_[0];
                std::ptrdiff_t sb = o.strides_[0];
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
    std::size_t size_ = 0;           // total element count = product(shape_)
    std::vector<std::size_t> shape_; // N-D shape (1 entry for 1-D)
    std::vector<std::ptrdiff_t> strides_; // element strides (row-major C-order for owning)
};

inline Array zeros(std::size_t n) { return Array(n, 0.0); }
inline Array full(std::size_t n, double value) { return Array(n, value); }
inline Array arange(std::size_t n) { return Array::iota(n); }

// N-D overloads
inline Array zeros_nd(std::vector<std::size_t> shape) { return Array(std::move(shape), 0.0); }
inline Array ones_nd(std::vector<std::size_t> shape) { return Array(std::move(shape), 1.0); }
inline Array full_nd(std::vector<std::size_t> shape, double value) { return Array(std::move(shape), value); }

}    // namespace hpxpy
