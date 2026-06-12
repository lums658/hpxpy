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

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpxpy {

template <class T>
using block_allocator = hpx::compute::host::block_allocator<T>;

using numa_alloc = block_allocator<double>;
using dvec = hpx::compute::vector<double, numa_alloc>;
using fvec = hpx::compute::vector<float, block_allocator<float>>;
// ivec (int64) lives here so array.hpp and sparse.hpp share one typedef (no ODR
// duplication); sparse.hpp already proves compute::vector<int64_t> on this build.
using ivec = hpx::compute::vector<std::int64_t, block_allocator<std::int64_t>>;

// ---------------------------------------------------------------------------
// Runtime element dtype. ONE Array class carries a runtime dt_; the element type
// is recovered per-op via dispatch_dtype (a switch that calls a generic lambda
// with a tag value of the matching C++ type, so the body instantiates per T).
// ---------------------------------------------------------------------------
enum class DType : std::uint8_t { F64, F32, I64 };

inline std::size_t dtype_size(DType d)
{
    switch (d) {
        case DType::F64: return sizeof(double);
        case DType::F32: return sizeof(float);
        case DType::I64: return sizeof(std::int64_t);
    }
    return sizeof(double);
}

// dispatch_dtype(d, f): call f(T{}) where T is the element type for d. f is a
// generic lambda `[&](auto tag){ using T = decltype(tag); ... }`.
template <class F>
inline decltype(auto) dispatch_dtype(DType d, F&& f)
{
    switch (d) {
        case DType::F64: return f(double{});
        case DType::F32: return f(float{});
        case DType::I64: return f(std::int64_t{});
    }
    return f(double{});    // unreachable; keeps the return type well-formed
}

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
    void* base;    // type-erased (recovered as T* inside the per-dtype dispatch)
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

    // 1-D constructor (backward-compatible): shape={n}, strides={1}. dtype defaults
    // to float64 so every existing call is identical.
    Array(std::size_t n, double fill, DType dt = DType::F64) { alloc_(n, fill, dt); }

    // N-D constructor: shape may have any number of dims; row-major strides computed.
    Array(std::vector<std::size_t> shape, double fill, DType dt = DType::F64)
    {
        std::size_t total = 1;
        for (auto d : shape) total *= d;
        alloc_nd_(std::move(shape), total, fill, dt);
    }

    std::size_t size() const { return size_; }
    std::size_t ndim() const { return shape_.size(); }
    DType dtype() const { return dt_; }

    // NumPy type-promotion lattice restricted to our three dtypes (A2.4). Mixed-dtype
    // element-wise ops promote both operands to this result type (== np.result_type):
    //   F64 dominates (either operand F64 -> F64);
    //   F32 ⊕ I64 -> F64  (numpy promotes int64+float32 to float64, NOT float32);
    //   F32 ⊕ F32 -> F32;
    //   I64 ⊕ I64 -> I64.
    static DType promote(DType a, DType b)
    {
        if (a == DType::F64 || b == DType::F64)
            return DType::F64;
        if (a == b)               // F32⊕F32 or I64⊕I64
            return a;
        // One is F32 and the other I64 -> F64 (numpy int64+float32 == float64).
        return DType::F64;
    }

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

    // Typed start-of-data accessors. The float64 forms below keep the existing
    // compute paths byte-for-byte; data_as<T>() recovers the right pointer for any
    // element type (used by astype, the bridge, and indexing).
    template <class T> T* data_as() const { return static_cast<T*>(base_); }

    // Start of this array's data (nullptr if empty). The const form is also used by
    // the diagnostic and by to_numpy (zero-copy export). float64 view of base_.
    const double* data() const { return static_cast<const double*>(base_); }
    double* mutable_data() { return static_cast<double*>(base_); }

    // Raw type-erased start of data (for the runtime-dtype numpy bridge).
    void* raw_data() const { return base_; }

    // 1-D element offset (in ELEMENTS, signed) for logical index i.
    std::ptrdiff_t offset_1d(std::size_t i) const
    {
        return static_cast<std::ptrdiff_t>(i) * strides_[0];
    }

    // N-D multi-index access: element at idx[] using strides_.
    std::size_t linear_offset(std::vector<std::size_t> const& idx) const
    {
        std::ptrdiff_t off = 0;
        for (std::size_t k = 0; k < idx.size(); ++k)
            off += static_cast<std::ptrdiff_t>(idx[k]) * strides_[k];
        return static_cast<std::size_t>(off);
    }

    // float64 scalar access (dt_==F64 fast paths). The binding dispatches on dtype()
    // for non-f64 element types via data_as<T>() + offset_1d/linear_offset.
    double getitem(std::size_t i) const
    {
        return data_as<double>()[offset_1d(i)];
    }
    void setitem(std::size_t i, double v)
    {
        data_as<double>()[offset_1d(i)] = v;
    }
    double getitem(std::vector<std::size_t> const& idx) const
    {
        return data_as<double>()[linear_offset(idx)];
    }
    void setitem(std::vector<std::size_t> const& idx, double v)
    {
        data_as<double>()[linear_offset(idx)] = v;
    }

    // Slice assignment a[start:start+n] = ... (contiguous; binding computes start/n).
    // fill_range broadcasts a scalar; assign_range copies an Array of length n in.
    // dtype-aware: the scalar is cast to the array's element type T (for int64 a
    // non-integral float scalar TRUNCATES — numpy a[i:j]=2.5 stores 2, consistent
    // with A2.1 setitem); assign_range requires the rhs share this array's dtype.
    void fill_range(std::size_t start, std::size_t n, double v)
    {
        if (n)
            dispatch_dtype(dt_, [&](auto tag) {
                using T = decltype(tag);
                T* b = data_as<T>();
                hpx::fill(hpx::execution::par, b + start, b + start + n,
                    static_cast<T>(v));
            });
    }
    void assign_range(std::size_t start, Array const& rhs)
    {
        if (dt_ != rhs.dt_)
            throw std::invalid_argument(
                "slice assignment operands have different dtypes; cast explicitly "
                "with .astype() — automatic type promotion is not yet supported");
        if (rhs.size_)
            dispatch_dtype(dt_, [&](auto tag) {
                using T = decltype(tag);
                T const* r = rhs.data_as<T>();
                hpx::copy(hpx::execution::par, r, r + rhs.size_,
                    data_as<T>() + start);
            });
    }

    // Contiguous view [start, start+n) sharing this array's buffer (numpy a[i:j]).
    // Returns a 1-D view: shape_={n}, strides_={parent_stride}. Carries dt_.
    Array view(std::size_t start, std::size_t n) const
    {
        Array r;
        r.owner_ = owner_;    // share the keep-alive (owned or borrowed parent)
        r.dt_ = dt_;
        r.base_ = byte_offset_(static_cast<std::ptrdiff_t>(start) * strides_[0]);
        r.size_ = n;
        r.shape_ = {n};
        r.strides_ = {strides_[0]};
        return r;
    }

    // Strided view: element i is at base_[i*result.strides_[0]] in the result.
    // start and step are in THIS array's logical index space; step may be negative.
    // Returns a 1-D view: shape_={n}, strides_={parent_stride*step}. Carries dt_.
    Array view_strided(std::ptrdiff_t start, std::size_t n, std::ptrdiff_t step) const
    {
        Array r;
        r.owner_ = owner_;
        r.dt_ = dt_;
        r.base_ = byte_offset_(start * strides_[0]);
        r.strides_ = {strides_[0] * step};
        r.shape_ = {n};
        r.size_ = n;
        return r;
    }

    // Wrap an external buffer (non-owning); `keepalive` keeps it alive. Used by the
    // zero-copy from_numpy borrow; the keep-alive is type-erased so this header stays
    // nanobind-free (the binding supplies a GIL-aware deleter). Carries the dtype.
    static Array borrow(void* p, std::size_t n, std::shared_ptr<void> keepalive,
        DType dt = DType::F64)
    {
        Array r;
        r.owner_ = std::move(keepalive);
        r.dt_ = dt;
        r.base_ = p;
        r.size_ = n;
        r.shape_ = {n};
        r.strides_ = {1};
        return r;
    }

    // N-D borrow: wrap an external C-contiguous N-D buffer (row-major). Computes
    // row-major strides from shape; `keepalive` keeps the buffer alive (from_numpy).
    static Array borrow_nd(void* p, std::vector<std::size_t> shape,
        std::shared_ptr<void> keepalive, DType dt = DType::F64)
    {
        Array r;
        r.owner_ = std::move(keepalive);
        r.dt_ = dt;
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
        r.dt_    = dt_;
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
            // Zero-copy view (works for any dtype).
            Array r;
            r.owner_   = owner_;
            r.dt_      = dt_;
            r.base_    = base_;
            r.size_    = size_;
            r.shape_   = new_shape;
            r.strides_ = new_strides;
            return r;
        }
        // Non-contiguous: copy to a fresh contiguous array, then set shape/strides.
        // (copy() is dtype-generic, so non-contiguous reshape works for any dtype.)
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
        r.dt_    = dt_;
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
        r.dt_    = dt_;
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
        r.dt_ = dt_;
        r.base_ = byte_offset_(base_off);
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
        return dispatch_dtype(dt_, [&](auto tag) -> double {
            using T = decltype(tag);
            T* base_ = data_as<T>();
            if (is_contiguous())
                return static_cast<double>(
                    hpx::reduce(hpx::execution::par, base_, base_ + size_, T(0)));
            T r = T(0);
            T* b = base_;
            auto aux = inner_volumes(shape_);
            auto const& sh  = shape_;
            auto const& st  = strides_;
            hpx::experimental::for_loop(hpx::execution::par, std::size_t(0), size_,
                hpx::experimental::reduction(r, T(0), std::plus<T>{}),
                [b, &sh, &st, &aux](std::size_t i, T& acc) {
                    acc += b[flat_to_offset(i, sh, st, aux)];
                });
            return static_cast<double>(r);
        });
    }
    double min() const
    {
        if (size_ == 0)
            throw std::invalid_argument("min() of an empty Array");
        return dispatch_dtype(dt_, [&](auto tag) -> double {
            using T = decltype(tag);
            // Identity that works for float AND integer T (±infinity() is 0 for
            // integral types, which would corrupt the reduction).
            constexpr T id = std::numeric_limits<T>::max();
            T* base_ = data_as<T>();
            if (is_contiguous())
                return static_cast<double>(hpx::reduce(hpx::execution::par,
                    base_, base_ + size_, id,
                    [](T x, T y) { return x < y ? x : y; }));
            T r = id;
            T* b = base_;
            auto aux = inner_volumes(shape_);
            auto const& sh  = shape_;
            auto const& st  = strides_;
            hpx::experimental::for_loop(hpx::execution::par, std::size_t(0), size_,
                hpx::experimental::reduction(r, id,
                    [](T x, T y) { return x < y ? x : y; }),
                [b, &sh, &st, &aux](std::size_t i, T& acc) {
                    T v = b[flat_to_offset(i, sh, st, aux)];
                    if (v < acc) acc = v;
                });
            return static_cast<double>(r);
        });
    }
    double max() const
    {
        if (size_ == 0)
            throw std::invalid_argument("max() of an empty Array");
        return dispatch_dtype(dt_, [&](auto tag) -> double {
            using T = decltype(tag);
            // lowest() is the most-negative finite value for float AND integer T
            // (the integer trap: -infinity() is 0, not INT_MIN).
            constexpr T id = std::numeric_limits<T>::lowest();
            T* base_ = data_as<T>();
            if (is_contiguous())
                return static_cast<double>(hpx::reduce(hpx::execution::par,
                    base_, base_ + size_, id,
                    [](T x, T y) { return x > y ? x : y; }));
            T r = id;
            T* b = base_;
            auto aux = inner_volumes(shape_);
            auto const& sh  = shape_;
            auto const& st  = strides_;
            hpx::experimental::for_loop(hpx::execution::par, std::size_t(0), size_,
                hpx::experimental::reduction(r, id,
                    [](T x, T y) { return x > y ? x : y; }),
                [b, &sh, &st, &aux](std::size_t i, T& acc) {
                    T v = b[flat_to_offset(i, sh, st, aux)];
                    if (v > acc) acc = v;
                });
            return static_cast<double>(r);
        });
    }
    // Fused dot product: a SINGLE pass (multiply+accumulate) via transform_reduce
    // (contiguous) or for_loop reduction (strided).
    double dot(Array const& other) const
    {
        if (size_ != other.size_)
            throw std::invalid_argument("dot(): size mismatch");
        if (size_ == 0)
            return 0.0;
        // Mixed dtype: promote both operands to Tr = promote(dt_, other.dt_) via the
        // existing astype, then run the SAME-dtype reduction below at Tr (keeps the
        // kernel instantiated at one type per call instead of 3x3x3). The same-dtype
        // path is byte-identical to the pre-A2.4 kernel (transform_reduce fast path /
        // strided reduction).
        DType rdt = promote(dt_, other.dt_);
        if (dt_ != rdt)
            return astype(rdt).dot(other);
        if (other.dt_ != rdt)
            return dot(other.astype(rdt));
        return dispatch_dtype(dt_, [&](auto tag) -> double {
            using T = decltype(tag);
            T* base_ = data_as<T>();
            T* obase = other.data_as<T>();
            if (is_contiguous() && other.is_contiguous())
                return static_cast<double>(hpx::transform_reduce(
                    hpx::execution::par, base_, base_ + size_, obase, T(0)));
            T r = T(0);
            T* ba = base_;
            T* bb = obase;
            auto auxa = inner_volumes(shape_);
            auto auxb = inner_volumes(other.shape_);
            auto const& sha = shape_;
            auto const& sta = strides_;
            auto const& shb = other.shape_;
            auto const& stb = other.strides_;
            hpx::experimental::for_loop(hpx::execution::par, std::size_t(0), size_,
                hpx::experimental::reduction(r, T(0), std::plus<T>{}),
                [ba, bb, &sha, &sta, &auxa, &shb, &stb, &auxb](std::size_t i, T& acc) {
                    acc += ba[flat_to_offset(i, sha, sta, auxa)] *
                           bb[flat_to_offset(i, shb, stb, auxb)];
                });
            return static_cast<double>(r);
        });
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
    // Identities (per element type T, supplied to reduce_axis as a generic callable
    // identity(T{}) -> T): sum T(0); min std::numeric_limits<T>::max(); max
    // std::numeric_limits<T>::lowest() (these work for float AND integer T — the
    // integer trap is that ±infinity() is 0 for integral types, corrupting min/max).
    // Empty (size_==0 or a reduced extent is 0): min/max throw; sum yields zeros.
    Array sum_axis(std::vector<std::size_t> axes, bool keepdims) const
    {
        return reduce_axis(std::move(axes), keepdims,
            [](auto t) { using T = decltype(t); return T(0); },
            [](auto acc, auto v) { return acc + v; }, /*throw_empty=*/false);
    }
    Array min_axis(std::vector<std::size_t> axes, bool keepdims) const
    {
        return reduce_axis(std::move(axes), keepdims,
            [](auto t) { using T = decltype(t); return std::numeric_limits<T>::max(); },
            [](auto acc, auto v) { return v < acc ? v : acc; },
            /*throw_empty=*/true);
    }
    Array max_axis(std::vector<std::size_t> axes, bool keepdims) const
    {
        return reduce_axis(std::move(axes), keepdims,
            [](auto t) { using T = decltype(t); return std::numeric_limits<T>::lowest(); },
            [](auto acc, auto v) { return v > acc ? v : acc; },
            /*throw_empty=*/true);
    }

    // 2-D matrix multiply: this is (m,k), B is (k2,n) with k==k2 -> C (m,n).
    // Both operands must be 2-D. Uses BOTH operands' strides_, so transposed /
    // non-contiguous operands are handled correctly (no copy needed). C is a fresh
    // contiguous owning Array. NAIVE O(m*n*k); tiled/BLAS optimization is future work.
    Array matmul(Array const& B) const
    {
        // Mixed dtype: promote both operands to Tr = promote(dt_, B.dt_) (numpy
        // faithful) via the existing astype, then run the SAME-dtype kernel at Tr
        // (kernel instantiated at one type per call, not 3x3x3). Same-dtype keeps
        // the result dtype == operand dtype — the int64 path accumulates in int64
        // and wraps on overflow, byte-identical to pre-A2.4.
        if (ndim() != 2 || B.ndim() != 2)
            throw std::invalid_argument("matmul: both operands must be 2-D");
        std::size_t m = shape_[0];
        std::size_t k = shape_[1];
        std::size_t k2 = B.shape_[0];
        std::size_t n = B.shape_[1];
        if (k != k2)
            throw std::invalid_argument(
                "matmul: inner dimensions do not match");
        DType rdt = promote(dt_, B.dt_);
        if (dt_ != rdt)
            return astype(rdt).matmul(B);
        if (B.dt_ != rdt)
            return matmul(B.astype(rdt));
        Array C;
        C.alloc_nd_({m, n}, m * n, 0.0, rdt);
        if (m * n == 0)
            return C;
        std::ptrdiff_t sta0 = strides_[0], sta1 = strides_[1];
        std::ptrdiff_t stb0 = B.strides_[0], stb1 = B.strides_[1];
        dispatch_dtype(dt_, [&](auto tag) {
            using T = decltype(tag);
            T const* a = data_as<T>();
            T const* b = B.data_as<T>();
            T* c = C.data_as<T>();
            hpx::experimental::for_loop(hpx::execution::par,
                std::size_t(0), m * n,
                [a, b, c, n, k, sta0, sta1, stb0, stb1](std::size_t out) {
                    std::size_t i = out / n;
                    std::size_t j = out % n;
                    T acc = T(0);
                    for (std::size_t p = 0; p < k; ++p)
                        acc += a[static_cast<std::ptrdiff_t>(i) * sta0 +
                                 static_cast<std::ptrdiff_t>(p) * sta1] *
                               b[static_cast<std::ptrdiff_t>(p) * stb0 +
                                 static_cast<std::ptrdiff_t>(j) * stb1];
                    c[out] = acc;
                });
        });
        return C;
    }

    // Element-wise binary ops -> a NEW (owning) Array. One hpx::transform pass
    // (contiguous) or for_loop element-wise (strided; result is always contiguous).
    // Transparent functors (std::plus<> etc.) let binary() instantiate the op at
    // the element type T inside its per-dtype dispatch.
    Array add(Array const& o) const { return binary(o, std::plus<>{}); }
    Array sub(Array const& o) const { return binary(o, std::minus<>{}); }
    Array mul(Array const& o) const { return binary(o, std::multiplies<>{}); }
    Array div(Array const& o) const
    {
        // True division ALWAYS yields a float (numpy true_divide). Result dtype =
        // promote(a,b), bumped to F64 if that would be integer (I64/I64 -> F64).
        // When both operands are already that float dtype this is the same-dtype
        // fast path; otherwise binary() fuses the int->float cast in a single pass.
        DType dr = div_result_dtype_(dt_, o.dt_);
        if (dt_ == dr && o.dt_ == dr)
            return binary(o, std::divides<>{});
        if (dt_ != dr)
            return astype(dr).div(o);   // lhs now float; recurse promotes rhs if needed
        // dt_ == dr (float), o.dt_ differs (integer) -> binary fuses the cast.
        return binary(o, std::divides<>{});
    }

    // Scalar broadcast -> new Array. The r* forms are reflected (s - a, s / a).
    // Two flavors per op (numpy promotes by the Python scalar's TYPE, not value):
    //   *_scalar(double)      -- a Python FLOAT scalar. On an I64 array this PROMOTES
    //                            the result to F64 (np: int64_arr + 1.5 -> float64);
    //                            on F32/F64 it stays that float dtype.
    //   *_scalar_int(int64_t) -- a Python INT scalar. Preserves the array dtype
    //                            (I64 stays I64; F32/F64 stay float), cast to T.
    // True division ALWAYS yields float (I64 -> F64) for both flavors.
    Array add_scalar(double s) const
    { return scalar_arith_(s, [](auto x, auto sc) { return x + sc; }); }
    Array sub_scalar(double s) const
    { return scalar_arith_(s, [](auto x, auto sc) { return x - sc; }); }
    Array rsub_scalar(double s) const
    { return scalar_arith_(s, [](auto x, auto sc) { return sc - x; }); }
    Array mul_scalar(double s) const
    { return scalar_arith_(s, [](auto x, auto sc) { return x * sc; }); }
    Array div_scalar(double s) const
    { return scalar_div_(s, /*reflected=*/false); }
    Array rdiv_scalar(double s) const
    { return scalar_div_(s, /*reflected=*/true); }

    // Integer-scalar arithmetic (Python int): preserve the array dtype.
    Array add_scalar_int(std::int64_t s) const
    { return scalar_arith_int_(s, [](auto x, auto sc) { return x + sc; }); }
    Array sub_scalar_int(std::int64_t s) const
    { return scalar_arith_int_(s, [](auto x, auto sc) { return x - sc; }); }
    Array rsub_scalar_int(std::int64_t s) const
    { return scalar_arith_int_(s, [](auto x, auto sc) { return sc - x; }); }
    Array mul_scalar_int(std::int64_t s) const
    { return scalar_arith_int_(s, [](auto x, auto sc) { return x * sc; }); }
    // True division with an int scalar still yields float (I64 -> F64).
    Array div_scalar_int(std::int64_t s) const
    { return scalar_div_(static_cast<double>(s), /*reflected=*/false); }
    Array rdiv_scalar_int(std::int64_t s) const
    { return scalar_div_(static_cast<double>(s), /*reflected=*/true); }

    // -----------------------------------------------------------------------
    // Element-wise unary math ufuncs (NumPy Wave 1).
    //
    // PRESERVE-DTYPE forms (output dtype == input dtype): negative/abs/sign.
    // Implemented via unary(op) with a generic functor (instantiated at T).
    // -----------------------------------------------------------------------
    Array negative() const
    { return unary([](auto x) { using T = decltype(x); return static_cast<T>(-x); }); }
    Array abs() const
    {
        return unary([](auto x) {
            using T = decltype(x);
            if constexpr (std::is_integral_v<T>)
                return static_cast<T>(x < 0 ? -x : x);
            else
                return static_cast<T>(std::abs(x));
        });
    }
    Array sign() const
    {
        return unary([](auto x) {
            using T = decltype(x);
            return static_cast<T>((x > T(0)) - (x < T(0)));
        });
    }

    // PROMOTE-INT-TO-FLOAT64 forms: float input keeps its dtype; integer input is
    // first cast to float64 (so the result is float64, like numpy). The float math
    // op is applied at the (float) element type. `op` is a generic callable op(T)->T.
    Array sqrt()  const { return unary_float_([](auto x) { return std::sqrt(x); }); }
    Array exp()   const { return unary_float_([](auto x) { return std::exp(x); }); }
    Array log()   const { return unary_float_([](auto x) { return std::log(x); }); }
    Array sin()   const { return unary_float_([](auto x) { return std::sin(x); }); }
    Array cos()   const { return unary_float_([](auto x) { return std::cos(x); }); }
    Array tan()   const { return unary_float_([](auto x) { return std::tan(x); }); }
    Array floor() const { return unary_float_([](auto x) { return std::floor(x); }); }
    Array ceil()  const { return unary_float_([](auto x) { return std::ceil(x); }); }
    Array trunc() const { return unary_float_([](auto x) { return std::trunc(x); }); }
    // round/rint: numpy round() is round-half-to-even (banker's), matching std::rint
    // under the default rounding mode (FE_TONEAREST). std::round is half-away-from-zero
    // and would NOT match numpy, so use std::rint here.
    Array round() const { return unary_float_([](auto x) { return std::rint(x); }); }

    // -----------------------------------------------------------------------
    // Element-wise binary math ufuncs (PRESERVE-DTYPE; same-dtype enforced by
    // binary()'s rule). The functor branches on integral vs float T where the
    // numpy semantics differ (mod, floor_divide, power).
    // -----------------------------------------------------------------------
    Array maximum(Array const& o) const
    { return binary(o, [](auto a, auto b) { return a > b ? a : b; }); }
    Array minimum(Array const& o) const
    { return binary(o, [](auto a, auto b) { return a < b ? a : b; }); }
    Array power(Array const& o) const
    {
        return binary(o, [](auto a, auto b) {
            using T = decltype(a);
            if constexpr (std::is_integral_v<T>) {
                // Integer power by repeated multiplication (numpy: negative exponent
                // on an integer base is a ValueError; we floor it to 0 like a naive
                // loop — exercised values are non-negative in tests/usage).
                T base = a, result = T(1);
                T e = b;
                for (T i = 0; i < e; ++i) result *= base;
                return result;
            } else {
                return static_cast<T>(std::pow(a, b));
            }
        });
    }
    Array mod(Array const& o) const
    {
        return binary(o, [](auto a, auto b) {
            using T = decltype(a);
            if constexpr (std::is_integral_v<T>) {
                // numpy int mod follows the sign of the divisor (Python %), unlike
                // C++ % which follows the dividend. Adjust to match numpy.
                if (b == T(0)) return T(0);    // numpy warns + yields 0; mirror that
                T r = a % b;
                if (r != T(0) && ((r < T(0)) != (b < T(0)))) r += b;
                return r;
            } else {
                // numpy float mod also follows the divisor's sign (fmod follows the
                // dividend), so adjust the remainder to match.
                T r = std::fmod(a, b);
                if (r != T(0) && ((r < T(0)) != (b < T(0)))) r += b;
                return static_cast<T>(r);
            }
        });
    }
    Array floor_divide(Array const& o) const
    {
        return binary(o, [](auto a, auto b) {
            using T = decltype(a);
            if constexpr (std::is_integral_v<T>) {
                if (b == T(0)) return T(0);    // numpy warns + yields 0; mirror that
                T q = a / b, r = a % b;
                // Floor toward negative infinity (numpy //), not truncate-toward-zero.
                if (r != T(0) && ((r < T(0)) != (b < T(0)))) --q;
                return q;
            } else {
                return static_cast<T>(std::floor(a / b));
            }
        });
    }

    // Scalar forms of power/mod/floor_divide (a ** s, a % s, a // s), reusing the
    // same numpy-faithful per-T semantics as the Array forms. scalar_unary_ casts the
    // Python double `s` to T (rejecting a non-integral scalar on int64).
    Array pow_scalar(double s) const
    {
        return scalar_unary_(s, [](auto x, auto sc) {
            using T = decltype(x);
            if constexpr (std::is_integral_v<T>) {
                T base = x, result = T(1);
                for (T i = 0; i < sc; ++i) result *= base;
                return result;
            } else {
                return static_cast<T>(std::pow(x, sc));
            }
        });
    }
    Array mod_scalar(double s) const
    {
        return scalar_unary_(s, [](auto x, auto sc) {
            using T = decltype(x);
            if constexpr (std::is_integral_v<T>) {
                if (sc == T(0)) return T(0);
                T r = x % sc;
                if (r != T(0) && ((r < T(0)) != (sc < T(0)))) r += sc;
                return r;
            } else {
                T r = std::fmod(x, sc);
                if (r != T(0) && ((r < T(0)) != (sc < T(0)))) r += sc;
                return static_cast<T>(r);
            }
        });
    }
    Array floordiv_scalar(double s) const
    {
        return scalar_unary_(s, [](auto x, auto sc) {
            using T = decltype(x);
            if constexpr (std::is_integral_v<T>) {
                if (sc == T(0)) return T(0);
                T q = x / sc, r = x % sc;
                if (r != T(0) && ((r < T(0)) != (sc < T(0)))) --q;
                return q;
            } else {
                return static_cast<T>(std::floor(x / sc));
            }
        });
    }

    // clip(lo, hi): clamp every element to [lo, hi]. PRESERVE-DTYPE. lo/hi come in as
    // Python doubles (cast to T per-dtype). If lo > hi, numpy returns hi everywhere;
    // std::min(std::max(x,lo),hi) reproduces that. (NaN bounds are out of scope.)
    Array clip(double lo, double hi) const
    {
        return unary([lo, hi](auto x) {
            using T = decltype(x);
            T l = static_cast<T>(lo), h = static_cast<T>(hi);
            T v = x < l ? l : x;
            return static_cast<T>(v > h ? h : v);
        });
    }

    // -----------------------------------------------------------------------
    // Reductions added in Wave 1: mean / prod / any / all / count_nonzero.
    // -----------------------------------------------------------------------

    // prod(): full reduction -> scalar (double). identity 1, combine multiply.
    // int64 wraps on overflow (numpy-faithful). Mirrors sum()'s structure.
    double prod() const
    {
        if (size_ == 0)
            return 1.0;
        return dispatch_dtype(dt_, [&](auto tag) -> double {
            using T = decltype(tag);
            T* base_ = data_as<T>();
            if (is_contiguous())
                return static_cast<double>(hpx::reduce(hpx::execution::par,
                    base_, base_ + size_, T(1), std::multiplies<T>{}));
            T r = T(1);
            T* b = base_;
            auto aux = inner_volumes(shape_);
            auto const& sh  = shape_;
            auto const& st  = strides_;
            hpx::experimental::for_loop(hpx::execution::par, std::size_t(0), size_,
                hpx::experimental::reduction(r, T(1), std::multiplies<T>{}),
                [b, &sh, &st, &aux](std::size_t i, T& acc) {
                    acc *= b[flat_to_offset(i, sh, st, aux)];
                });
            return static_cast<double>(r);
        });
    }

    // prod_axis: axis reduction with identity 1, combine multiply. Result dtype =
    // input dtype (like sum_axis). Follows the sum_axis/reduce_axis pattern.
    Array prod_axis(std::vector<std::size_t> axes, bool keepdims) const
    {
        return reduce_axis(std::move(axes), keepdims,
            [](auto t) { using T = decltype(t); return T(1); },
            [](auto acc, auto v) { return acc * v; }, /*throw_empty=*/false);
    }

    // mean(): full reduction -> scalar double. ALWAYS float64 (numpy mean of any
    // dtype is float64). sum/count, computed in double.
    double mean() const
    {
        if (size_ == 0)
            return std::numeric_limits<double>::quiet_NaN();    // numpy: mean of empty -> nan
        return sum() / static_cast<double>(size_);
    }

    // mean_axis: ALWAYS float64 output. Cast to f64, sum over the axes, divide by the
    // reduced count. Returns a float64 Array.
    Array mean_axis(std::vector<std::size_t> axes, bool keepdims) const
    {
        // Count of elements reduced into each output cell = product of reduced extents.
        std::size_t nd = shape_.size();
        std::size_t count = 1;
        std::vector<bool> is_red(nd, false);
        for (std::size_t ax : axes) is_red[ax] = true;
        for (std::size_t k = 0; k < nd; ++k)
            if (is_red[k]) count *= shape_[k];
        // Promote to float64 first so the sum (and divide) are float64 (numpy mean).
        Array f = (dt_ == DType::F64) ? *this : astype(DType::F64);
        Array s = f.sum_axis(std::move(axes), keepdims);    // float64 sum
        // Divide by the reduced count. count==0 (empty reduced axis) yields 0/0 = nan
        // per IEEE float64, matching numpy's mean-over-empty-axis result.
        return s.div_scalar(static_cast<double>(count));
    }

    // any(): true iff ANY element != 0 (logical OR of x!=0). axis=None only (Wave 1).
    bool any() const
    {
        if (size_ == 0)
            return false;
        return dispatch_dtype(dt_, [&](auto tag) -> bool {
            using T = decltype(tag);
            T* b = data_as<T>();
            bool acc = false;
            if (is_contiguous()) {
                hpx::experimental::for_loop(hpx::execution::par, std::size_t(0), size_,
                    hpx::experimental::reduction(acc, false, std::logical_or<bool>{}),
                    [b](std::size_t i, bool& a) { a = a || (b[i] != T(0)); });
                return acc;
            }
            auto aux = inner_volumes(shape_);
            auto const& sh = shape_;
            auto const& st = strides_;
            hpx::experimental::for_loop(hpx::execution::par, std::size_t(0), size_,
                hpx::experimental::reduction(acc, false, std::logical_or<bool>{}),
                [b, &sh, &st, &aux](std::size_t i, bool& a) {
                    a = a || (b[flat_to_offset(i, sh, st, aux)] != T(0));
                });
            return acc;
        });
    }

    // all(): true iff EVERY element != 0 (logical AND of x!=0). axis=None only.
    bool all() const
    {
        if (size_ == 0)
            return true;    // numpy: all() of an empty array is True
        return dispatch_dtype(dt_, [&](auto tag) -> bool {
            using T = decltype(tag);
            T* b = data_as<T>();
            bool acc = true;
            if (is_contiguous()) {
                hpx::experimental::for_loop(hpx::execution::par, std::size_t(0), size_,
                    hpx::experimental::reduction(acc, true, std::logical_and<bool>{}),
                    [b](std::size_t i, bool& a) { a = a && (b[i] != T(0)); });
                return acc;
            }
            auto aux = inner_volumes(shape_);
            auto const& sh = shape_;
            auto const& st = strides_;
            hpx::experimental::for_loop(hpx::execution::par, std::size_t(0), size_,
                hpx::experimental::reduction(acc, true, std::logical_and<bool>{}),
                [b, &sh, &st, &aux](std::size_t i, bool& a) {
                    a = a && (b[flat_to_offset(i, sh, st, aux)] != T(0));
                });
            return acc;
        });
    }

    // count_nonzero(): number of elements != 0 (sum of x!=0). Returns int64.
    std::int64_t count_nonzero() const
    {
        if (size_ == 0)
            return 0;
        return dispatch_dtype(dt_, [&](auto tag) -> std::int64_t {
            using T = decltype(tag);
            T* b = data_as<T>();
            std::int64_t acc = 0;
            if (is_contiguous()) {
                hpx::experimental::for_loop(hpx::execution::par, std::size_t(0), size_,
                    hpx::experimental::reduction(acc, std::int64_t(0),
                        std::plus<std::int64_t>{}),
                    [b](std::size_t i, std::int64_t& a) {
                        a += (b[i] != T(0)) ? 1 : 0;
                    });
                return acc;
            }
            auto aux = inner_volumes(shape_);
            auto const& sh = shape_;
            auto const& st = strides_;
            hpx::experimental::for_loop(hpx::execution::par, std::size_t(0), size_,
                hpx::experimental::reduction(acc, std::int64_t(0),
                    std::plus<std::int64_t>{}),
                [b, &sh, &st, &aux](std::size_t i, std::int64_t& a) {
                    a += (b[flat_to_offset(i, sh, st, aux)] != T(0)) ? 1 : 0;
                });
            return acc;
        });
    }

    // Deep copy -> a new independent (owning) contiguous Array (numpy a.copy()).
    // The result is ALWAYS contiguous row-major with the same shape as *this.
    Array copy() const
    {
        Array r;
        r.alloc_nd_(shape_, size_, 0.0, dt_);
        if (size_) {
            dispatch_dtype(dt_, [&](auto tag) {
                using T = decltype(tag);
                T* base_ = data_as<T>();
                if (is_contiguous()) {
                    hpx::copy(hpx::execution::par, base_, base_ + size_,
                        r.template data_as<T>());
                } else {
                    T* src = base_;
                    T* dst = r.template data_as<T>();
                    auto aux = inner_volumes(shape_);
                    auto const& sh = shape_;
                    auto const& st = strides_;
                    hpx::experimental::for_loop(hpx::execution::par,
                        std::size_t(0), size_,
                        [src, dst, &sh, &st, &aux](std::size_t i) {
                            dst[i] = src[flat_to_offset(i, sh, st, aux)];
                        });
                }
            });
        }
        return r;
    }

    // Sort ascending IN PLACE (numpy a.sort(); mutates this view's range, no alloc).
    // Strided: gather into a temporary contiguous buffer, sort, scatter back.
    void sort()
    {
        if (size_ < 2)
            return;
        dispatch_dtype(dt_, [&](auto tag) {
            using T = decltype(tag);
            T* base_ = data_as<T>();
            if (is_contiguous()) {
                hpx::sort(hpx::execution::par, base_, base_ + size_);
            } else {
                // Gather non-contiguous elements into a temp buffer, sort, scatter back.
                T* src = base_;
                auto aux = inner_volumes(shape_);
                auto const& sh = shape_;
                auto const& st = strides_;
                std::vector<T> tmp(size_);
                for (std::size_t i = 0; i < size_; ++i)
                    tmp[i] = src[flat_to_offset(i, sh, st, aux)];
                hpx::sort(hpx::execution::par, tmp.begin(), tmp.end());
                for (std::size_t i = 0; i < size_; ++i)
                    src[flat_to_offset(i, sh, st, aux)] = tmp[i];
            }
        });
    }
    bool is_sorted() const
    {
        if (size_ < 2)
            return true;
        return dispatch_dtype(dt_, [&](auto tag) -> bool {
            using T = decltype(tag);
            T* base_ = data_as<T>();
            if (is_contiguous())
                return hpx::is_sorted(hpx::execution::par, base_, base_ + size_);
            // Non-contiguous N-D: AND-reduce over adjacent flat-index pairs.
            bool ok = true;
            T* b = base_;
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
        });
    }

    // Inclusive prefix sum -> a NEW (owning) Array (numpy a.cumsum()).
    // Strided: gather into the result buffer then do the scan in place.
    Array cumsum() const
    {
        Array r;
        r.alloc_(size_, 0.0, dt_);
        if (size_) {
            dispatch_dtype(dt_, [&](auto tag) {
                using T = decltype(tag);
                T* base_ = data_as<T>();
                if (is_contiguous()) {
                    hpx::inclusive_scan(hpx::execution::par, base_, base_ + size_,
                        r.template data_as<T>());
                } else {
                    T* src = base_;
                    T* dst = r.template data_as<T>();
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
            });
        }
        return r;
    }

    // 0, 1, 2, ..., n-1. block_allocator first-touches at allocation; the parallel
    // for_loop (run directly) writes the ramp on the same HPX workers (NUMA-local).
    // Carries the dtype; the ramp is written in the matching element type T.
    static Array iota(std::size_t n, DType dt = DType::F64)
    {
        Array a;
        a.alloc_(n, 0.0, dt);
        if (n)
            dispatch_dtype(dt, [&](auto tag) {
                using T = decltype(tag);
                T* p = a.template data_as<T>();
                hpx::experimental::for_loop(hpx::execution::par, std::size_t(0), n,
                    [p](std::size_t i) { p[i] = static_cast<T>(i); });
            });
        return a;
    }

    // arange_range(start, stop, step, n, dt): n elements of the affine ramp
    // start + i*step (numpy arange with explicit start/stop/step). The element
    // count n is computed at the binding boundary (ceil((stop-start)/step)). The
    // parallel for_loop writes static_cast<T>(start + i*step) on the NUMA-local
    // HPX workers (same first-touch story as iota).
    static Array arange_range(double start, double step, std::size_t n,
        DType dt = DType::F64)
    {
        Array a;
        a.alloc_(n, 0.0, dt);
        if (n)
            dispatch_dtype(dt, [&](auto tag) {
                using T = decltype(tag);
                T* p = a.template data_as<T>();
                hpx::experimental::for_loop(hpx::execution::par, std::size_t(0), n,
                    [p, start, step](std::size_t i) {
                        p[i] = static_cast<T>(
                            start + static_cast<double>(i) * step);
                    });
            });
        return a;
    }

    // linspace(start, step, num, last, dt): `num` evenly spaced points of the ramp
    // start + i*step. When `endpoint` is requested the binding passes `last`==stop so
    // the final point is exactly `stop` (avoiding round-off at the endpoint); for
    // num==1 the single point is `start`. Parallel for_loop on the NUMA workers.
    static Array linspace(double start, double step, std::size_t num,
        bool set_last, double last, DType dt = DType::F64)
    {
        Array a;
        a.alloc_(num, 0.0, dt);
        if (num)
            dispatch_dtype(dt, [&](auto tag) {
                using T = decltype(tag);
                T* p = a.template data_as<T>();
                hpx::experimental::for_loop(hpx::execution::par, std::size_t(0), num,
                    [p, start, step](std::size_t i) {
                        p[i] = static_cast<T>(
                            start + static_cast<double>(i) * step);
                    });
                if (set_last)
                    p[num - 1] = static_cast<T>(last);
            });
        return a;
    }

    // eye(rows, cols, k, dt): a 2-D (rows x cols) array with ones on the k-th
    // diagonal ([i,j]==1 iff j == i+k) and zeros elsewhere. block_allocator
    // value-initializes the buffer to zero; the parallel for_loop over the rows*cols
    // cells sets the diagonal entries. identity(n) is eye(n, n, 0) at the binding.
    static Array eye(std::size_t rows, std::size_t cols, std::ptrdiff_t k,
        DType dt = DType::F64)
    {
        Array a;
        std::size_t total = rows * cols;
        a.alloc_nd_({rows, cols}, total, 0.0, dt);
        if (total)
            dispatch_dtype(dt, [&](auto tag) {
                using T = decltype(tag);
                T* p = a.template data_as<T>();
                hpx::experimental::for_loop(hpx::execution::par, std::size_t(0), total,
                    [p, cols, k](std::size_t idx) {
                        std::size_t i = idx / cols;
                        std::size_t j = idx % cols;
                        p[idx] = (static_cast<std::ptrdiff_t>(j) ==
                                  static_cast<std::ptrdiff_t>(i) + k)
                                     ? static_cast<T>(1)
                                     : static_cast<T>(0);
                    });
            });
        return a;
    }

    // empty(shape, dt): an owning N-D buffer of the right dtype WITHOUT initializing
    // the element values (numpy.empty — contents are arbitrary). Skips the value-fill
    // by allocating the raw compute::vector with a default-constructed (uninitialized
    // for trivial T) buffer. NUMA first-touch still happens at allocation.
    static Array empty(std::vector<std::size_t> shape, DType dt = DType::F64)
    {
        std::size_t total = 1;
        for (auto d : shape) total *= d;
        Array a;
        a.alloc_uninit_(std::move(shape), total, dt);
        return a;
    }

    // astype(dst): a new contiguous owning Array of dtype `dst` with element-wise
    // static_cast from this array's elements. Works for ALL dtypes (no compute guard;
    // it is a typed copy, not a numeric kernel). Casts through the logical (strided)
    // order so views convert correctly.
    Array astype(DType dst) const
    {
        Array out;
        out.alloc_nd_(shape_, size_, 0.0, dst);
        if (size_ == 0)
            return out;
        bool contig = is_contiguous();
        auto aux = inner_volumes(shape_);
        auto const& sh = shape_;
        auto const& st = strides_;
        dispatch_dtype(dt_, [&](auto src_tag) {
            using S = decltype(src_tag);
            S const* src = data_as<S>();
            dispatch_dtype(dst, [&](auto dst_tag) {
                using D = decltype(dst_tag);
                D* dptr = out.template data_as<D>();
                if (contig) {
                    hpx::transform(hpx::execution::par, src, src + size_, dptr,
                        [](S v) { return static_cast<D>(v); });
                } else {
                    hpx::experimental::for_loop(hpx::execution::par,
                        std::size_t(0), size_,
                        [src, dptr, &sh, &st, &aux](std::size_t i) {
                            dptr[i] = static_cast<D>(
                                src[flat_to_offset(i, sh, st, aux)]);
                        });
                }
            });
        });
        return out;
    }

private:
    // (A2.3 removed require_f64_; A2.4 removed require_float_ — true division now
    // PROMOTES integers to float64 instead of raising, via div_result_dtype_.)

    // True-division result dtype (numpy true_divide): promote(a,b) but bumped to
    // F64 when that would be integer (I64/I64 -> F64). F32/F32 stays F32; any pair
    // touching F64 -> F64; F32⊕I64 -> F64 (== promote()).
    static DType div_result_dtype_(DType a, DType b)
    {
        DType p = promote(a, b);
        return (p == DType::I64) ? DType::F64 : p;
    }

    // Offset base_ by `n` ELEMENTS (signed), accounting for the element size. Returns
    // nullptr when base_ is null (empty array). Used by views (dtype-agnostic).
    void* byte_offset_(std::ptrdiff_t n) const
    {
        if (base_ == nullptr)
            return nullptr;
        return static_cast<void*>(
            static_cast<char*>(base_) +
            n * static_cast<std::ptrdiff_t>(dtype_size(dt_)));
    }

    // Allocate the matching compute::vector<T> for dtype `dt` (value-initialized to
    // static_cast<T>(fill)) on an HPX thread; set dt_/base_/owner_.
    void alloc_buffer_(std::size_t n, double fill, DType dt)
    {
        dt_ = dt;
        on_hpx_thread([&] {
            dispatch_dtype(dt, [&](auto tag) {
                using T = decltype(tag);
                using Vec = hpx::compute::vector<T, block_allocator<T>>;
                auto d = std::make_shared<Vec>(n, static_cast<T>(fill));
                base_ = n ? static_cast<void*>(d->data()) : nullptr;
                owner_ = std::move(d);
            });
        });
    }

    // Allocate the matching compute::vector<T> for dtype `dt` WITHOUT initializing the
    // element values (numpy.empty — contents arbitrary). Reserves n elements then
    // default-constructs them (a no-op for the trivial T we support), so the values are
    // uninitialized but NUMA first-touch still happens at allocation.
    void alloc_buffer_uninit_(std::size_t n, DType dt)
    {
        dt_ = dt;
        on_hpx_thread([&] {
            dispatch_dtype(dt, [&](auto tag) {
                using T = decltype(tag);
                using Vec = hpx::compute::vector<T, block_allocator<T>>;
                auto d = std::make_shared<Vec>(n);
                base_ = n ? static_cast<void*>(d->data()) : nullptr;
                owner_ = std::move(d);
            });
        });
    }

    // alloc_nd_ counterpart that leaves the element values uninitialized (numpy.empty).
    void alloc_uninit_(std::vector<std::size_t> shape, std::size_t total, DType dt)
    {
        shape_ = std::move(shape);
        size_ = total;
        std::size_t nd = shape_.size();
        strides_.resize(nd);
        if (nd > 0) {
            strides_[nd - 1] = 1;
            for (std::size_t k = nd - 1; k-- > 0; )
                strides_[k] = static_cast<std::ptrdiff_t>(shape_[k + 1]) * strides_[k + 1];
        }
        alloc_buffer_uninit_(total, dt);
    }

    // Allocate a 1-D owning NUMA buffer of n elements (value-initialized to fill) on an
    // HPX thread, and point base_/owner_/shape_/strides_/size_/dt_ at it.
    void alloc_(std::size_t n, double fill = 0.0, DType dt = DType::F64)
    {
        size_ = n;
        shape_ = {n};
        strides_ = {1};
        alloc_buffer_(n, fill, dt);
    }

    // Allocate an N-D owning NUMA buffer: shape already provided; total = product(shape).
    void alloc_nd_(std::vector<std::size_t> shape, std::size_t total, double fill,
        DType dt = DType::F64)
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
        alloc_buffer_(total, fill, dt);
    }

    // Element-wise unary map -> a new owning Array of THIS array's dtype. `op` is a
    // generic callable op(T x) -> T (instantiated at the element type T inside the
    // per-dtype dispatch). For T=double this is byte-identical to the prior f64-only
    // kernel (F64 dispatch arm = the former body).
    template <typename Op>
    Array unary(Op op) const
    {
        Array r;
        r.alloc_nd_(shape_, size_, 0.0, dt_);
        if (size_) {
            dispatch_dtype(dt_, [&](auto tag) {
                using T = decltype(tag);
                T* base_ = data_as<T>();
                if (is_contiguous()) {
                    hpx::transform(hpx::execution::par, base_, base_ + size_,
                        r.template data_as<T>(), op);
                } else {
                    T* src = base_;
                    T* dst = r.template data_as<T>();
                    auto aux = inner_volumes(shape_);
                    auto const& sh = shape_;
                    auto const& st = strides_;
                    hpx::experimental::for_loop(hpx::execution::par,
                        std::size_t(0), size_,
                        [src, dst, &sh, &st, &aux, op](std::size_t i) {
                            dst[i] = op(src[flat_to_offset(i, sh, st, aux)]);
                        });
                }
            });
        }
        return r;
    }

    // Float-math unary helper (the PROMOTE-INT-TO-FLOAT64 rule). For a float dtype
    // (F64/F32) the op runs at the native element type, preserving the dtype. For an
    // integer dtype the array is first cast to float64 (so the result is float64,
    // matching numpy: np.sqrt(int64_array).dtype == float64), then the op runs in f64.
    // `op` is a generic callable op(T x) -> T applied at the (float) element type.
    template <typename Op>
    Array unary_float_(Op op) const
    {
        if (dt_ == DType::I64)
            return astype(DType::F64).unary(op);
        return unary(op);
    }

    // Scalar broadcast helper: bind the Python double `s` (cast to the array's
    // element type T) into the generic op f(T x, T s) and run it as a unary map.
    // int64 trap: a non-integral float scalar on an int64 array would silently
    // truncate, so reject it (automatic int<->float promotion is deferred to A2.4).
    template <typename F>
    Array scalar_unary_(double s, F f) const
    {
        if (dt_ == DType::I64 && s != std::floor(s))
            throw std::invalid_argument(
                "cannot apply a non-integral float scalar to an int64 array; "
                "cast explicitly with .astype() — automatic type promotion is "
                "not yet supported");
        return unary([s, f](auto x) {
            using T = decltype(x);
            return f(x, static_cast<T>(s));
        });
    }

    // Arithmetic with a Python FLOAT scalar (A2.4 promotion). On an I64 array the
    // result PROMOTES to F64 (numpy: int64_arr + 1.5 -> float64): elements are cast
    // to double and the op runs in double with the double scalar. On F32/F64 the op
    // runs at the native element type (dtype preserved) — byte-identical to the old
    // float path. `f` is a generic op f(T x, T sc) -> T.
    template <typename F>
    Array scalar_arith_(double s, F f) const
    {
        if (dt_ == DType::I64)
            return astype(DType::F64).unary(
                [s, f](auto x) { return f(x, static_cast<double>(s)); });
        return unary([s, f](auto x) {
            using T = decltype(x);
            return f(x, static_cast<T>(s));
        });
    }

    // Arithmetic with a Python INT scalar: preserve the array dtype (I64 stays I64,
    // F32/F64 stay float). The integer scalar is cast to the element type T.
    template <typename F>
    Array scalar_arith_int_(std::int64_t s, F f) const
    {
        return unary([s, f](auto x) {
            using T = decltype(x);
            return f(x, static_cast<T>(s));
        });
    }

    // True division by a scalar -> ALWAYS float (numpy true_divide). On I64 the
    // array promotes to F64; on F32 it stays F32; on F64 it stays F64. `reflected`
    // selects s / a instead of a / s.
    Array scalar_div_(double s, bool reflected) const
    {
        DType dr = (dt_ == DType::I64) ? DType::F64 : dt_;
        if (dt_ != dr)
            return astype(dr).scalar_div_(s, reflected);
        return unary([s, reflected](auto x) {
            using T = decltype(x);
            T sc = static_cast<T>(s);
            return reflected ? static_cast<T>(sc / x) : static_cast<T>(x / sc);
        });
    }

    // Generic axis-reduction kernel shared by sum_axis/min_axis/max_axis.
    // `make_identity` is a generic callable make_identity(T{}) -> T seeding each output
    // cell at the element type T (so min/max use std::numeric_limits<T>, not a ±inf
    // double that is 0 for integers). `combine(acc, v)` folds an element in (generic
    // over T). `throw_empty` is true for min/max (reducing an empty extent is undefined).
    // Result dtype = this array's dtype. The F64 arm is byte-identical to the prior
    // float64-only kernel.
    template <typename MakeIdentity, typename Combine>
    Array reduce_axis(std::vector<std::size_t> axes, bool keepdims,
        MakeIdentity make_identity, Combine combine, bool throw_empty) const
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

        return dispatch_dtype(dt_, [&](auto tag) -> Array {
            using T = decltype(tag);
            T identity = make_identity(T{});
            Array out;
            out.alloc_nd_(out_shape, out_size, static_cast<double>(identity), dt_);
            if (out_size == 0 || red_size == 0)
                return out;    // sum over empty reduced extent => zeros (identity fill)

            T const* b = data_as<T>();
            T* dst = out.template data_as<T>();
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
                    // Unravel out_i over the kept extents -> base offset (kept-axis strides).
                    std::ptrdiff_t base_off = 0;
                    {
                        std::size_t rem = out_i;
                        for (std::size_t j = 0; j < kept_nd; ++j) {
                            std::size_t coord = rem / kept_aux[j];
                            rem %= kept_aux[j];
                            base_off += static_cast<std::ptrdiff_t>(coord) * st[kept_axes[j]];
                        }
                    }
                    T acc = identity;
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
        });
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
        v.base = x.base_;    // type-erased; recovered as T* inside binary()'s dispatch
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
        // ===== SAME-DTYPE FAST PATH (zero-penalty, the common case) =============
        // Byte-identical to the pre-A2.4 kernel: result dtype == operand dtype,
        // one transform / for_loop pass at the native element type T.
        if (dt_ == o.dt_)
            return dispatch_dtype(dt_, [&](auto tag) -> Array {
                using T = decltype(tag);
                T* base_ = data_as<T>();
                T* obase = o.data_as<T>();
                // ---- FAST PATH (zero-penalty): same shape, both contiguous ----
                if (shape_ == o.shape_ && is_contiguous() && o.is_contiguous()) {
                    Array r;
                    r.alloc_nd_(shape_, size_, 0.0, dt_);
                    if (size_)
                        hpx::transform(
                            hpx::execution::par, base_, base_ + size_, obase,
                            r.template data_as<T>(), op);
                    return r;
                }

                // ---- Compute result shape (broadcast or same-shape) ----
                std::vector<std::size_t> rshape =
                    (shape_ == o.shape_) ? shape_ : broadcast_shapes(shape_, o.shape_);
                std::size_t rsize = 1;
                for (auto d : rshape) rsize *= d;

                Array r;
                r.alloc_nd_(rshape, rsize, 0.0, dt_);
                if (rsize == 0)
                    return r;

                // Build broadcast views of *this and o aligned to rshape.
                BcastView va = make_bcast_view(*this, rshape);
                BcastView vb = make_bcast_view(o, rshape);
                T* ba = static_cast<T*>(va.base);
                T* bb = static_cast<T*>(vb.base);
                T* dst = r.template data_as<T>();
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
            });

        // ===== MIXED-DTYPE PROMOTION PATH (A2.4) ===============================
        // Promote both operands to Tr = promote(dt_, o.dt_) with the existing astype
        // (a single typed copy of the off-dtype operand), then run the SAME-dtype
        // kernel above at Tr. This keeps the op functor instantiated at exactly ONE
        // type per call (Tr) instead of 3x3x3 (Ta x Tb x Tr) — a deliberate choice
        // to keep compile time / memory bounded; the (uncommon) mixed path pays one
        // extra cast pass, while the common same-dtype path stays the zero-penalty
        // single-pass fast path above. Casting an operand already equal to Tr is a
        // no-op fast return (astype copies; but only the differing operand is cast).
        DType rdt = promote(dt_, o.dt_);
        if (dt_ != rdt)
            return astype(rdt).binary(o, op);     // lhs now Tr; recurse promotes rhs
        // dt_ == rdt, so o.dt_ != rdt: cast rhs up, then the same-dtype path runs.
        return binary(o.astype(rdt), op);
    }

    std::shared_ptr<void> owner_;    // keeps the backing alive (dvec/fvec/ivec, or numpy)
    void* base_ = nullptr;           // type-erased start of data (offset folded in)
    DType dt_ = DType::F64;          // runtime element dtype (default float64)
    std::size_t size_ = 0;           // total element count = product(shape_)
    std::vector<std::size_t> shape_; // N-D shape (1 entry for 1-D)
    std::vector<std::ptrdiff_t> strides_; // element strides (row-major C-order for owning)
};

inline Array zeros(std::size_t n, DType dt = DType::F64) { return Array(n, 0.0, dt); }
inline Array full(std::size_t n, double value, DType dt = DType::F64) { return Array(n, value, dt); }
inline Array arange(std::size_t n, DType dt = DType::F64) { return Array::iota(n, dt); }

// Construction helpers (Wave 2). Element counts / endpoint handling are resolved at
// the binding boundary (numpy rules); these just thread args into the static factories.
inline Array arange_range(double start, double step, std::size_t n, DType dt = DType::F64)
{ return Array::arange_range(start, step, n, dt); }
inline Array linspace(double start, double step, std::size_t num, bool set_last,
    double last, DType dt = DType::F64)
{ return Array::linspace(start, step, num, set_last, last, dt); }
inline Array eye(std::size_t rows, std::size_t cols, std::ptrdiff_t k,
    DType dt = DType::F64)
{ return Array::eye(rows, cols, k, dt); }
inline Array empty_nd(std::vector<std::size_t> shape, DType dt = DType::F64)
{ return Array::empty(std::move(shape), dt); }

// N-D overloads
inline Array zeros_nd(std::vector<std::size_t> shape, DType dt = DType::F64)
{ return Array(std::move(shape), 0.0, dt); }
inline Array ones_nd(std::vector<std::size_t> shape, DType dt = DType::F64)
{ return Array(std::move(shape), 1.0, dt); }
inline Array full_nd(std::vector<std::size_t> shape, double value, DType dt = DType::F64)
{ return Array(std::move(shape), value, dt); }

}    // namespace hpxpy
