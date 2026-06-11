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
// N-D view ops (stage 3): transpose/reshape/squeeze/expand_dims — zero-copy when
// the source is contiguous; non-contiguous reshape copies.
// N-D compute (stage 3): non-contiguous else-branches use flat_to_offset (an N-D
// unravel via pre-computed inner_volumes) so every op works on any non-contiguous
// N-D array. The contiguous fast-path is byte-for-byte unchanged.
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

// ---------------------------------------------------------------------------
// N-D unravel helpers (used by the non-contiguous compute else-branches).
//
// inner_volumes(shape) -> aux[k] = product(shape[k+1..nd-1]), aux[nd-1] = 1.
// Computed ONCE per operation, before any loop.
//
// flat_to_offset(i, shape, strides, aux) -> element offset from base_.
//   coord_k = (i / aux[k]) % shape[k]  (integer div/mod with the inner volume)
//   off += coord_k * strides[k]
// The 1-D degenerate case (aux[0]=1) gives i*strides[0], matching the old branch.
//
// BcastView: a lightweight broadcast view of an operand used in binary().
// strides may contain zeros (stride-0 trick: repeats the element along that axis).
// ---------------------------------------------------------------------------

struct BcastView {
    double* base;
    std::vector<std::size_t> shape;
    std::vector<std::ptrdiff_t> strides;
};

inline std::vector<std::size_t> inner_volumes(std::vector<std::size_t> const& shape)
{
    std::size_t nd = shape.size();
    std::vector<std::size_t> aux(nd, 1);
    for (std::size_t k = nd; k-- > 1; )
        aux[k - 1] = aux[k] * shape[k];
    return aux;
}

inline std::ptrdiff_t flat_to_offset(std::size_t i,
    std::vector<std::size_t> const& shape,
    std::vector<std::ptrdiff_t> const& strides,
    std::vector<std::size_t> const& aux)
{
    std::ptrdiff_t off = 0;
    std::size_t rem = i;
    std::size_t nd = shape.size();
    for (std::size_t k = 0; k < nd; ++k) {
        std::size_t coord = rem / aux[k];
        rem %= aux[k];
        off += static_cast<std::ptrdiff_t>(coord) * strides[k];
    }
    return off;
}

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

    // -----------------------------------------------------------------------
    // N-D view ops (stage 3) — all zero-copy except non-contiguous reshape.
    // -----------------------------------------------------------------------

    // transpose(axes): permute shape_/strides_ by axes. empty axes => reverse all.
    // Returns a zero-copy view sharing owner_/base_.
    Array transpose(std::vector<std::size_t> axes = {}) const
    {
        std::size_t nd = shape_.size();
        if (axes.empty()) {
            // Default: reverse all axes.
            axes.resize(nd);
            for (std::size_t k = 0; k < nd; ++k)
                axes[k] = nd - 1 - k;
        }
        if (axes.size() != nd)
            throw std::invalid_argument(
                "transpose: axes length must equal ndim");
        // Validate that axes is a permutation of 0..nd-1.
        std::vector<bool> seen(nd, false);
        for (std::size_t ax : axes) {
            if (ax >= nd)
                throw std::invalid_argument(
                    "transpose: axis out of range");
            if (seen[ax])
                throw std::invalid_argument(
                    "transpose: duplicate axis");
            seen[ax] = true;
        }
        Array r;
        r.owner_ = owner_;
        r.base_  = base_;
        r.size_  = size_;
        r.shape_.resize(nd);
        r.strides_.resize(nd);
        for (std::size_t k = 0; k < nd; ++k) {
            r.shape_[k]   = shape_[axes[k]];
            r.strides_[k] = strides_[axes[k]];
        }
        return r;
    }

    // reshape(new_shape): product must equal size_. If contiguous returns a zero-copy
    // view with row-major strides for new_shape; otherwise copies first.
    // A single -1 in new_shape is inferred (must divide size_ evenly).
    Array reshape(std::vector<std::ptrdiff_t> new_shape_signed) const
    {
        // Resolve -1 (at most one).
        std::size_t infer_idx = std::size_t(-1);
        std::size_t known_prod = 1;
        for (std::size_t k = 0; k < new_shape_signed.size(); ++k) {
            if (new_shape_signed[k] == -1) {
                if (infer_idx != std::size_t(-1))
                    throw std::invalid_argument(
                        "reshape: only one -1 allowed");
                infer_idx = k;
            } else if (new_shape_signed[k] < 0) {
                throw std::invalid_argument(
                    "reshape: negative dimension other than -1");
            } else {
                known_prod *= static_cast<std::size_t>(new_shape_signed[k]);
            }
        }
        std::vector<std::size_t> new_shape(new_shape_signed.size());
        for (std::size_t k = 0; k < new_shape_signed.size(); ++k) {
            if (new_shape_signed[k] != -1)
                new_shape[k] = static_cast<std::size_t>(new_shape_signed[k]);
        }
        if (infer_idx != std::size_t(-1)) {
            if (known_prod == 0 || size_ % known_prod != 0)
                throw std::invalid_argument(
                    "reshape: cannot infer dimension (size not divisible)");
            new_shape[infer_idx] = size_ / known_prod;
        }
        // Validate product.
        std::size_t prod = 1;
        for (std::size_t d : new_shape) prod *= d;
        if (prod != size_)
            throw std::invalid_argument(
                "reshape: new shape product does not equal size");

        // Compute row-major strides for new_shape.
        std::size_t nd2 = new_shape.size();
        std::vector<std::ptrdiff_t> new_strides(nd2);
        if (nd2 > 0) {
            new_strides[nd2 - 1] = 1;
            for (std::size_t k = nd2 - 1; k-- > 0; )
                new_strides[k] = static_cast<std::ptrdiff_t>(new_shape[k + 1]) * new_strides[k + 1];
        }

        if (is_contiguous()) {
            // Zero-copy view.
            Array r;
            r.owner_   = owner_;
            r.base_    = base_;
            r.size_    = size_;
            r.shape_   = new_shape;
            r.strides_ = new_strides;
            return r;
        }
        // Non-contiguous: copy to a fresh contiguous array, then set shape/strides.
        Array c = copy();
        c.shape_   = new_shape;
        c.strides_ = new_strides;
        return c;
    }

    // ravel(): flatten to 1-D (reshape to {size_}).
    Array ravel() const
    {
        return reshape({static_cast<std::ptrdiff_t>(size_)});
    }

    // squeeze(axes): drop size-1 dims. Empty axes => drop ALL size-1 dims.
    // Named axes must each have size 1.
    Array squeeze(std::vector<std::size_t> axes = {}) const
    {
        std::size_t nd = shape_.size();
        std::vector<bool> drop(nd, false);
        if (axes.empty()) {
            for (std::size_t k = 0; k < nd; ++k)
                if (shape_[k] == 1) drop[k] = true;
        } else {
            for (std::size_t ax : axes) {
                if (ax >= nd)
                    throw std::invalid_argument("squeeze: axis out of range");
                if (shape_[ax] != 1)
                    throw std::invalid_argument(
                        "squeeze: cannot squeeze axis with size > 1");
                drop[ax] = true;
            }
        }
        Array r;
        r.owner_ = owner_;
        r.base_  = base_;
        r.size_  = size_;
        for (std::size_t k = 0; k < nd; ++k) {
            if (!drop[k]) {
                r.shape_.push_back(shape_[k]);
                r.strides_.push_back(strides_[k]);
            }
        }
        // Edge case: all dims squeezed -> 0-D (treat as 1-D size-1 scalar).
        if (r.shape_.empty()) {
            r.shape_   = {1};
            r.strides_ = {1};
        }
        return r;
    }

    // expand_dims(axis): insert a size-1 dim at position axis (supports nd+1).
    Array expand_dims(std::size_t axis) const
    {
        std::size_t nd = shape_.size();
        if (axis > nd)
            throw std::invalid_argument("expand_dims: axis out of range");
        Array r;
        r.owner_ = owner_;
        r.base_  = base_;
        r.size_  = size_;
        r.shape_.reserve(nd + 1);
        r.strides_.reserve(nd + 1);
        for (std::size_t k = 0; k < axis; ++k) {
            r.shape_.push_back(shape_[k]);
            r.strides_.push_back(strides_[k]);
        }
        r.shape_.push_back(1);
        r.strides_.push_back(axis < nd ? strides_[axis] : 1);
        for (std::size_t k = axis; k < nd; ++k) {
            r.shape_.push_back(shape_[k]);
            r.strides_.push_back(strides_[k]);
        }
        return r;
    }

    // -----------------------------------------------------------------------
    // N-D multi-axis slicing (stage 6) — zero-copy view, numpy a[i:j, ::k, m].
    //
    // AxisSpec describes ONE input axis: either an INT (the axis is dropped from
    // the output, contributing int_idx*stride to the base offset) or a SLICE
    // (the axis survives with length slicelen and stride strides_[ax]*step,
    // starting at start*stride). start/step are signed (step<0 => reverse view,
    // negative output stride — same convention as view_strided).
    // -----------------------------------------------------------------------
    struct AxisSpec
    {
        enum Kind { INT, SLICE } kind;
        std::ptrdiff_t int_idx = 0;    // INT: normalized, in-range index
        std::ptrdiff_t start = 0;      // SLICE: adjusted start
        std::ptrdiff_t step = 1;       // SLICE: adjusted step (may be negative)
        std::size_t slicelen = 0;      // SLICE: number of elements
    };

    // slice_nd(specs): specs.size() must equal ndim (the binding pads with full
    // slices / expands Ellipsis). Returns a zero-copy view sharing owner_.
    Array slice_nd(std::vector<AxisSpec> const& specs) const
    {
        std::ptrdiff_t base_off = 0;
        Array r;
        for (std::size_t ax = 0; ax < specs.size(); ++ax) {
            AxisSpec const& s = specs[ax];
            if (s.kind == AxisSpec::INT) {
                base_off += s.int_idx * strides_[ax];    // drop this axis
            } else {
                base_off += s.start * strides_[ax];
                r.shape_.push_back(s.slicelen);
                r.strides_.push_back(strides_[ax] * s.step);
            }
        }
        r.owner_ = owner_;
        r.base_ = base_ ? base_ + base_off : nullptr;
        if (r.shape_.empty()) {
            // All axes were INT (out shape empty) -> a 1-element scalar array.
            // (The binding's all-int fast path means this is rarely hit.)
            r.shape_ = {1};
            r.strides_ = {1};
            r.size_ = 1;
        } else {
            std::size_t total = 1;
            for (std::size_t d : r.shape_) total *= d;
            r.size_ = total;
        }
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
        auto aux = inner_volumes(shape_);
        auto const& sh  = shape_;
        auto const& st  = strides_;
        hpx::experimental::for_loop(hpx::execution::par, std::size_t(0), size_,
            hpx::experimental::reduction(r, 0.0, std::plus<double>{}),
            [b, &sh, &st, &aux](std::size_t i, double& acc) {
                acc += b[flat_to_offset(i, sh, st, aux)];
            });
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
        auto aux = inner_volumes(shape_);
        auto const& sh  = shape_;
        auto const& st  = strides_;
        hpx::experimental::for_loop(hpx::execution::par, std::size_t(0), size_,
            hpx::experimental::reduction(r, std::numeric_limits<double>::infinity(),
                [](double x, double y) { return x < y ? x : y; }),
            [b, &sh, &st, &aux](std::size_t i, double& acc) {
                double v = b[flat_to_offset(i, sh, st, aux)];
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
        auto aux = inner_volumes(shape_);
        auto const& sh  = shape_;
        auto const& st  = strides_;
        hpx::experimental::for_loop(hpx::execution::par, std::size_t(0), size_,
            hpx::experimental::reduction(r, -std::numeric_limits<double>::infinity(),
                [](double x, double y) { return x > y ? x : y; }),
            [b, &sh, &st, &aux](std::size_t i, double& acc) {
                double v = b[flat_to_offset(i, sh, st, aux)];
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
        auto auxa = inner_volumes(shape_);
        auto auxb = inner_volumes(other.shape_);
        auto const& sha = shape_;
        auto const& sta = strides_;
        auto const& shb = other.shape_;
        auto const& stb = other.strides_;
        hpx::experimental::for_loop(hpx::execution::par, std::size_t(0), size_,
            hpx::experimental::reduction(r, 0.0, std::plus<double>{}),
            [ba, bb, &sha, &sta, &auxa, &shb, &stb, &auxb](std::size_t i, double& acc) {
                acc += ba[flat_to_offset(i, sha, sta, auxa)] *
                       bb[flat_to_offset(i, shb, stb, auxb)];
            });
        return r;
    }

    // -----------------------------------------------------------------------
    // N-D axis reductions (stage 5) — reduce over `axes` (already normalized:
    // non-negative, sorted, unique — done by the binding). Each output cell owns
    // its accumulator (no shared reduction object): the parallel for_loop runs
    // over [0, out_size) and each iteration does a serial inner reduction over
    // the reduced extents. Works on any (contiguous or not) N-D array via strides_.
    //
    //   out_shape : reduced axes dropped (keepdims=false) or set to 1 (keepdims=true)
    //   kept_axes : the surviving axes (their extents form the parallel index space)
    //   red_axes  : the reduced axes (their extents form the serial inner loop)
    // Identities: sum 0.0, min +inf, max -inf. Empty (size_==0 or a reduced extent
    // is 0): min/max throw; sum yields zeros.
    Array sum_axis(std::vector<std::size_t> axes, bool keepdims) const
    {
        return reduce_axis(std::move(axes), keepdims, 0.0,
            [](double acc, double v) { return acc + v; }, /*throw_empty=*/false);
    }
    Array min_axis(std::vector<std::size_t> axes, bool keepdims) const
    {
        return reduce_axis(std::move(axes), keepdims,
            std::numeric_limits<double>::infinity(),
            [](double acc, double v) { return v < acc ? v : acc; },
            /*throw_empty=*/true);
    }
    Array max_axis(std::vector<std::size_t> axes, bool keepdims) const
    {
        return reduce_axis(std::move(axes), keepdims,
            -std::numeric_limits<double>::infinity(),
            [](double acc, double v) { return v > acc ? v : acc; },
            /*throw_empty=*/true);
    }

    // 2-D matrix multiply: this is (m,k), B is (k2,n) with k==k2 -> C (m,n).
    // Both operands must be 2-D. Uses BOTH operands' strides_, so transposed /
    // non-contiguous operands are handled correctly (no copy needed). C is a fresh
    // contiguous owning Array. NAIVE O(m*n*k); tiled/BLAS optimization is future work.
    Array matmul(Array const& B) const
    {
        if (ndim() != 2 || B.ndim() != 2)
            throw std::invalid_argument("matmul: both operands must be 2-D");
        std::size_t m = shape_[0];
        std::size_t k = shape_[1];
        std::size_t k2 = B.shape_[0];
        std::size_t n = B.shape_[1];
        if (k != k2)
            throw std::invalid_argument(
                "matmul: inner dimensions do not match");
        Array C;
        C.alloc_nd_({m, n}, m * n, 0.0);
        if (m * n == 0)
            return C;
        double const* a = base_;
        double const* b = B.base_;
        double* c = C.base_;
        std::ptrdiff_t sta0 = strides_[0], sta1 = strides_[1];
        std::ptrdiff_t stb0 = B.strides_[0], stb1 = B.strides_[1];
        hpx::experimental::for_loop(hpx::execution::par,
            std::size_t(0), m * n,
            [a, b, c, n, k, sta0, sta1, stb0, stb1](std::size_t out) {
                std::size_t i = out / n;
                std::size_t j = out % n;
                double acc = 0.0;
                for (std::size_t p = 0; p < k; ++p)
                    acc += a[static_cast<std::ptrdiff_t>(i) * sta0 +
                             static_cast<std::ptrdiff_t>(p) * sta1] *
                           b[static_cast<std::ptrdiff_t>(p) * stb0 +
                             static_cast<std::ptrdiff_t>(j) * stb1];
                c[out] = acc;
            });
        return C;
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

    // Deep copy -> a new independent (owning) contiguous Array (numpy a.copy()).
    // The result is ALWAYS contiguous row-major with the same shape as *this.
    Array copy() const
    {
        Array r;
        r.alloc_nd_(shape_, size_, 0.0);
        if (size_) {
            if (is_contiguous()) {
                hpx::copy(hpx::execution::par, base_, base_ + size_, r.base_);
            } else {
                double* src = base_;
                double* dst = r.base_;
                auto aux = inner_volumes(shape_);
                auto const& sh = shape_;
                auto const& st = strides_;
                hpx::experimental::for_loop(hpx::execution::par,
                    std::size_t(0), size_,
                    [src, dst, &sh, &st, &aux](std::size_t i) {
                        dst[i] = src[flat_to_offset(i, sh, st, aux)];
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
            // Gather non-contiguous elements into a temp buffer, sort, scatter back.
            double* src = base_;
            auto aux = inner_volumes(shape_);
            auto const& sh = shape_;
            auto const& st = strides_;
            std::vector<double> tmp(size_);
            for (std::size_t i = 0; i < size_; ++i)
                tmp[i] = src[flat_to_offset(i, sh, st, aux)];
            hpx::sort(hpx::execution::par, tmp.begin(), tmp.end());
            for (std::size_t i = 0; i < size_; ++i)
                src[flat_to_offset(i, sh, st, aux)] = tmp[i];
        }
    }
    bool is_sorted() const
    {
        if (size_ < 2)
            return true;
        if (is_contiguous())
            return hpx::is_sorted(hpx::execution::par, base_, base_ + size_);
        // Non-contiguous N-D: AND-reduce over adjacent flat-index pairs.
        bool ok = true;
        double* b = base_;
        auto aux = inner_volumes(shape_);
        auto const& sh = shape_;
        auto const& st = strides_;
        hpx::experimental::for_loop(hpx::execution::par, std::size_t(0), size_ - 1,
            hpx::experimental::reduction(ok, true, std::logical_and<bool>{}),
            [b, &sh, &st, &aux](std::size_t i, bool& acc) {
                acc = acc && (b[flat_to_offset(i,     sh, st, aux)] <=
                              b[flat_to_offset(i + 1, sh, st, aux)]);
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
                auto aux = inner_volumes(shape_);
                auto const& sh = shape_;
                auto const& st = strides_;
                // Gather non-contiguous elements into result, then scan in place.
                hpx::experimental::for_loop(hpx::execution::par,
                    std::size_t(0), size_,
                    [src, dst, &sh, &st, &aux](std::size_t i) {
                        dst[i] = src[flat_to_offset(i, sh, st, aux)];
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
        r.alloc_nd_(shape_, size_, 0.0);
        if (size_) {
            if (is_contiguous()) {
                hpx::transform(hpx::execution::par, base_, base_ + size_, r.base_, op);
            } else {
                double* src = base_;
                double* dst = r.base_;
                auto aux = inner_volumes(shape_);
                auto const& sh = shape_;
                auto const& st = strides_;
                hpx::experimental::for_loop(hpx::execution::par,
                    std::size_t(0), size_,
                    [src, dst, &sh, &st, &aux, op](std::size_t i) {
                        dst[i] = op(src[flat_to_offset(i, sh, st, aux)]);
                    });
            }
        }
        return r;
    }

    // Generic axis-reduction kernel shared by sum_axis/min_axis/max_axis.
    // `identity` seeds each output cell; `combine(acc, v)` folds an element in.
    // `throw_empty` is true for min/max (reducing over an empty extent is undefined).
    template <typename Combine>
    Array reduce_axis(std::vector<std::size_t> axes, bool keepdims,
        double identity, Combine combine, bool throw_empty) const
    {
        std::size_t nd = shape_.size();
        // Partition axes into reduced vs kept (axes is sorted/unique/in-range).
        std::vector<bool> is_red(nd, false);
        for (std::size_t ax : axes)
            is_red[ax] = true;
        std::vector<std::size_t> kept_axes, red_axes;
        for (std::size_t k = 0; k < nd; ++k)
            (is_red[k] ? red_axes : kept_axes).push_back(k);

        // out_shape: reduced axes dropped (keepdims=false) or set to 1 (keepdims=true).
        std::vector<std::size_t> out_shape;
        for (std::size_t k = 0; k < nd; ++k) {
            if (is_red[k]) {
                if (keepdims) out_shape.push_back(1);
            } else {
                out_shape.push_back(shape_[k]);
            }
        }

        // Extents of the kept (parallel) and reduced (serial inner) index spaces.
        std::vector<std::size_t> kept_shape, red_shape;
        std::size_t out_size = 1, red_size = 1;
        for (std::size_t ax : kept_axes) { kept_shape.push_back(shape_[ax]); out_size *= shape_[ax]; }
        for (std::size_t ax : red_axes)  { red_shape.push_back(shape_[ax]);  red_size *= shape_[ax]; }

        // Empty reduced extent: min/max are undefined; sum yields the identity (0).
        if (throw_empty && red_size == 0)
            throw std::invalid_argument(
                "axis reduction over an empty axis is undefined (min/max)");

        Array out;
        out.alloc_nd_(out_shape, out_size, identity);
        if (out_size == 0 || red_size == 0)
            return out;    // sum over empty reduced extent => zeros (identity fill)

        double const* b = base_;
        double* dst = out.base_;
        auto kept_aux = inner_volumes(kept_shape);
        auto red_aux = inner_volumes(red_shape);
        auto const& st = strides_;
        std::size_t kept_nd = kept_shape.size();
        std::size_t red_nd = red_shape.size();
        // for_loop over output cells; each owns its accumulator (no reduction object).
        hpx::experimental::for_loop(hpx::execution::par,
            std::size_t(0), out_size,
            [b, dst, kept_nd, &kept_aux, &kept_axes, red_size, red_nd, &red_aux,
             &red_axes, &st, identity, combine](std::size_t out_i) {
                // Unravel out_i over the kept extents -> base offset (strides of kept axes).
                std::ptrdiff_t base_off = 0;
                {
                    std::size_t rem = out_i;
                    for (std::size_t j = 0; j < kept_nd; ++j) {
                        std::size_t coord = rem / kept_aux[j];
                        rem %= kept_aux[j];
                        base_off += static_cast<std::ptrdiff_t>(coord) * st[kept_axes[j]];
                    }
                }
                double acc = identity;
                for (std::size_t red_i = 0; red_i < red_size; ++red_i) {
                    std::ptrdiff_t red_off = 0;
                    std::size_t rem = red_i;
                    for (std::size_t r = 0; r < red_nd; ++r) {
                        std::size_t coord = rem / red_aux[r];
                        rem %= red_aux[r];
                        red_off += static_cast<std::ptrdiff_t>(coord) * st[red_axes[r]];
                    }
                    acc = combine(acc, b[base_off + red_off]);
                }
                dst[out_i] = acc;
            });
        return out;
    }

    // broadcast_shapes: right-align shapes sa and sb (treat missing leading axes as 1);
    // per axis: equal => that size; one is 1 => the other; else throw.
    static std::vector<std::size_t> broadcast_shapes(
        std::vector<std::size_t> const& sa,
        std::vector<std::size_t> const& sb)
    {
        std::size_t ra = sa.size();
        std::size_t rb = sb.size();
        std::size_t rr = ra > rb ? ra : rb;
        std::vector<std::size_t> result(rr);
        for (std::size_t k = 0; k < rr; ++k) {
            // right-align: axis k from the end is index (rr-1-k) from the back
            std::size_t ak = (k < rr - ra) ? std::size_t(0) : ra - (rr - k);
            std::size_t bk = (k < rr - rb) ? std::size_t(0) : rb - (rr - k);
            std::size_t da = (k < rr - ra) ? std::size_t(1) : sa[ak];
            std::size_t db = (k < rr - rb) ? std::size_t(1) : sb[bk];
            if (da == db) {
                result[k] = da;
            } else if (da == 1) {
                result[k] = db;
            } else if (db == 1) {
                result[k] = da;
            } else {
                // Build numpy-style error message: "(a,b,..) (c,..)"
                std::string msg = "operands could not be broadcast together with shapes (";
                for (std::size_t i = 0; i < ra; ++i) {
                    if (i) msg += ",";
                    msg += std::to_string(sa[i]);
                }
                msg += ") (";
                for (std::size_t i = 0; i < rb; ++i) {
                    if (i) msg += ",";
                    msg += std::to_string(sb[i]);
                }
                msg += ")";
                throw std::invalid_argument(msg);
            }
        }
        return result;
    }

    // Build a BcastView of operand x aligned to rshape (rank rr).
    // Axes that are absent (left-padded) or size-1 in x but >1 in rshape get stride 0.
    static BcastView make_bcast_view(Array const& x,
        std::vector<std::size_t> const& rshape)
    {
        std::size_t rr = rshape.size();
        std::size_t rx = x.shape_.size();
        BcastView v;
        v.base = x.base_;
        v.shape = rshape;
        v.strides.resize(rr, 0);
        for (std::size_t k = 0; k < rr; ++k) {
            if (k < rr - rx) {
                // leading padded axis: stride 0
                v.strides[k] = 0;
            } else {
                std::size_t xk = rx - (rr - k);   // corresponding axis in x
                if (x.shape_[xk] == 1 && rshape[k] > 1)
                    v.strides[k] = 0;              // broadcast axis: stride 0
                else
                    v.strides[k] = x.strides_[xk];
            }
        }
        return v;
    }

    template <typename Op>
    Array binary(Array const& o, Op op) const
    {
        // ---- FAST PATH (zero-penalty): same shape, both contiguous ----
        if (shape_ == o.shape_ && is_contiguous() && o.is_contiguous()) {
            Array r;
            r.alloc_nd_(shape_, size_, 0.0);
            if (size_)
                hpx::transform(
                    hpx::execution::par, base_, base_ + size_, o.base_, r.base_, op);
            return r;
        }

        // ---- Compute result shape (broadcast or same-shape) ----
        std::vector<std::size_t> rshape =
            (shape_ == o.shape_) ? shape_ : broadcast_shapes(shape_, o.shape_);
        std::size_t rsize = 1;
        for (auto d : rshape) rsize *= d;

        Array r;
        r.alloc_nd_(rshape, rsize, 0.0);
        if (rsize == 0)
            return r;

        // Build broadcast views of *this and o aligned to rshape.
        BcastView va = make_bcast_view(*this, rshape);
        BcastView vb = make_bcast_view(o, rshape);
        double* ba = va.base;
        double* bb = vb.base;
        double* dst = r.base_;
        auto aux = inner_volumes(rshape);
        auto sha = va.shape;
        auto sta = va.strides;
        auto shb = vb.shape;
        auto stb = vb.strides;
        hpx::experimental::for_loop(hpx::execution::par,
            std::size_t(0), rsize,
            [ba, bb, dst, sha, sta, shb, stb, aux, op](std::size_t i) {
                dst[i] = op(ba[flat_to_offset(i, sha, sta, aux)],
                            bb[flat_to_offset(i, shb, stb, aux)]);
            });
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
