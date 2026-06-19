// hpxpy._core — register_bridge: to_numpy / from_numpy (the NumPy bridge, Phase 2).
//
// This TU OWNS the single definition of to_numpy_view (declared in bind_helpers.hpp and
// also called from bind_indexing.cpp): a zero-copy NumPy view of an Array, any rank.
// from_numpy maps the runtime numpy dtype to our DType (TypeError on unsupported —
// never a silent copy/cast); copy=True copies into a NUMA-aware Array, copy=False
// borrows the buffer zero-copy with a GIL-aware keep-alive.
//
// SPDX-License-Identifier: MIT

#include "array.hpp"
#include "bind_helpers.hpp"
#include "bind_fwd.hpp"

#include <hpx/algorithm.hpp>
#include <hpx/execution.hpp>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

using hpxpy::Array;
using hpxpy::DType;
using hpxpy_bind::dtype_from_nb;
using hpxpy_bind::nb_dtype_of;
using hpxpy_bind::np_out;
using hpxpy_bind::np_rw;

namespace hpxpy_bind {

// Zero-copy NumPy view of an Array, any rank. `obj` (the Array Python object) owns the
// ndarray, so the HPX buffer outlives the view. Contiguous arrays pass nullptr strides
// (nanobind infers row-major); non-contiguous ones (a[::2], a[::-1], transposed views)
// pass explicit strides. NOTE: nanobind's typed ndarray takes strides in ELEMENTS (it
// multiplies by sizeof(T) internally) — passing bytes segfaults on negative strides.
np_out to_numpy_view(nb::object obj)
{
    Array& a = nb::cast<Array&>(obj);
    std::size_t nd = a.ndim();
    std::vector<std::size_t> shape(nd);
    std::vector<int64_t> strides(nd);
    for (std::size_t k = 0; k < nd; ++k) {
        shape[k] = a.shape()[k];
        strides[k] = static_cast<int64_t>(a.strides()[k]);   // ELEMENTS, not bytes
    }
    // Runtime dtype: pass the matching dlpack dtype so numpy sees float64/float32/
    // int64 correctly. Strides stay in ELEMENTS (nanobind scales by itemsize).
    return np_out(a.raw_data(), nd, shape.data(), obj,
                  a.is_contiguous() ? nullptr : strides.data(),
                  nb_dtype_of(a.dtype()));
}

}  // namespace hpxpy_bind

void register_bridge(nb::module_& m)
{
    using hpxpy_bind::to_numpy_view;

    m.def("to_numpy", &to_numpy_view, "a"_a,
          "Zero-copy NumPy view of an Array (writable; shares memory).");
    m.def("from_numpy", [](np_rw arr, bool copy) -> Array {
        // Map the runtime numpy dtype to our DType (TypeError on unsupported: int32,
        // float16, complex, ... — never a silent copy/cast).
        DType dt = dtype_from_nb(arr.dtype());
        std::size_t const nd = arr.ndim();
        std::vector<std::size_t> shape(nd);
        for (std::size_t k = 0; k < nd; ++k)
            shape[k] = arr.shape(k);
        std::size_t const total = arr.size();    // product of all axes
        void* p = arr.data();
        if (copy) {
            Array a(shape, 0.0, dt);    // NUMA-aware N-D; correct first-touch
            if (total) {
                nb::gil_scoped_release release;
                hpxpy::dispatch_dtype(dt, [&](auto tag) {
                    using T = decltype(tag);
                    T const* src = static_cast<T const*>(p);
                    hpx::copy(hpx::execution::par, src, src + total,
                              a.template data_as<T>());
                });
            }
            return a;
        }
        // Zero-copy borrow: keep the NumPy buffer alive via a GIL-aware deleter.
        // numa-naive (numpy placement); use copy=True for HPX compute.
        auto* hold = new np_rw(arr);
        std::shared_ptr<void> keep(hold, [](void* q) {
            nb::gil_scoped_acquire g;
            delete static_cast<np_rw*>(q);
        });
        return Array::borrow_nd(p, std::move(shape), std::move(keep), dt);
    }, nb::arg("a").noconvert(), "copy"_a = true,
       "Bring a float64/float32/int64 C-contiguous NumPy array (any rank) into hpxpy. "
       "copy=True (default) copies into a NUMA-aware Array; copy=False borrows it "
       "(zero-copy, shares memory both ways, but numa-naive). Unsupported dtype or "
       "non-contiguous input raises (never a silent copy/cast).");
}
