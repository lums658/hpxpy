// hpxpy._core — register_wave1_reductions: mean/prod/any/all/count_nonzero +
// copy/sort/is_sorted/cumsum/astype.
//
// mean is ALWAYS float64; prod preserves dtype; any/all/count_nonzero support axis=None
// only in Wave 1. copy/sort/is_sorted/cumsum/astype are the structural ops grouped here
// to balance the per-TU kernel-instantiation cost. Each releases the GIL.
//
// SPDX-License-Identifier: MIT

#include "array.hpp"
#include "bind_helpers.hpp"
#include "bind_fwd.hpp"

#include <nanobind/nanobind.h>

#include <cstdint>
#include <utility>

namespace nb = nanobind;
using namespace nb::literals;

using hpxpy::Array;
using hpxpy::DType;
using hpxpy_bind::parse_axes;
using hpxpy_bind::parse_dtype;

void register_wave1_reductions(nb::module_&, nb::class_<Array>& cls)
{
    cls
        // astype(dtype): a new Array of the target dtype (element-wise static_cast).
        .def("astype", [](Array const& a, nb::object dt) {
            DType dst = parse_dtype(dt);
            nb::gil_scoped_release r;
            return a.astype(dst);
        }, "dtype"_a, "Cast to a new Array of the given dtype (element-wise).")
        .def("copy", [](Array const& a) {
            nb::gil_scoped_release r; return a.copy();
        }, "Deep copy to a new Array (numpy a.copy()).")
        .def("sort", [](Array& a) {
            nb::gil_scoped_release r; a.sort();
        }, "Sort ascending IN PLACE (numpy a.sort(); returns None).")
        .def("is_sorted", [](Array const& a) {
            nb::gil_scoped_release r; return a.is_sorted();
        }, "True if ascending (hpx::is_sorted).")
        .def("cumsum", [](Array const& a) {
            nb::gil_scoped_release r; return a.cumsum();
        }, "Inclusive prefix sum -> new Array (numpy a.cumsum()).")
        .def("cumprod", [](Array const& a) {
            nb::gil_scoped_release r; return a.cumprod();
        }, "Inclusive prefix product -> new Array (numpy a.cumprod()).")
        .def("argsort", [](Array const& a) {
            nb::gil_scoped_release r; return a.argsort();
        }, "I64 Array of indices that would sort a ascending (numpy a.argsort()).")
        // --- Reductions added in Wave 1: mean/prod/any/all/count_nonzero -----
        // mean: ALWAYS float64. axis=None -> Python float; axis -> float64 Array.
        .def("mean", [](Array const& a, nb::object axis, bool keepdims) -> nb::object {
            if (axis.is_none()) {
                double s;
                { nb::gil_scoped_release release; s = a.mean(); }
                return nb::cast(s);
            }
            auto ax = parse_axes(axis, a.ndim());
            Array r;
            { nb::gil_scoped_release release; r = a.mean_axis(ax, keepdims); }
            return nb::cast(std::move(r));
        }, "axis"_a = nb::none(), "keepdims"_a = false,
           "Arithmetic mean (always float64). axis=None -> float; axis -> Array.")
        // prod: preserves dtype. axis=None -> Python scalar; axis -> Array.
        .def("prod", [](Array const& a, nb::object axis, bool keepdims) -> nb::object {
            if (axis.is_none()) {
                double s;
                { nb::gil_scoped_release release; s = a.prod(); }
                // Preserve dtype on the scalar: int64 prod -> Python int.
                if (a.dtype() == DType::I64)
                    return nb::cast(static_cast<int64_t>(s));
                return nb::cast(s);
            }
            auto ax = parse_axes(axis, a.ndim());
            Array r;
            { nb::gil_scoped_release release; r = a.prod_axis(ax, keepdims); }
            return nb::cast(std::move(r));
        }, "axis"_a = nb::none(), "keepdims"_a = false,
           "Product over axis (preserves dtype). axis=None -> scalar; axis -> Array.")
        // any/all: axis=None only in Wave 1 (a bool-dtype axis variant is Wave 3).
        // Return a Python bool.
        .def("any", [](Array const& a, nb::object axis) -> bool {
            if (!axis.is_none())
                throw nb::value_error(
                    "any(axis=...) is not yet supported (Wave 1 supports axis=None only)");
            nb::gil_scoped_release r; return a.any();
        }, "axis"_a = nb::none(), "True if any element is nonzero (axis=None only).")
        .def("all", [](Array const& a, nb::object axis) -> bool {
            if (!axis.is_none())
                throw nb::value_error(
                    "all(axis=...) is not yet supported (Wave 1 supports axis=None only)");
            nb::gil_scoped_release r; return a.all();
        }, "axis"_a = nb::none(), "True if all elements are nonzero (axis=None only).")
        // count_nonzero: axis=None only; returns a Python int.
        .def("count_nonzero", [](Array const& a, nb::object axis) -> int64_t {
            if (!axis.is_none())
                throw nb::value_error(
                    "count_nonzero(axis=...) is not yet supported "
                    "(Wave 1 supports axis=None only)");
            nb::gil_scoped_release r; return a.count_nonzero();
        }, "axis"_a = nb::none(), "Number of nonzero elements (axis=None only).");
}
