// hpxpy._core — register_reductions: sum/min/max (+axis), dot, matmul, __matmul__.
//
// Reductions release the GIL around the HPX work (the wrapper itself is GIL-agnostic).
// std::invalid_argument from min()/max() maps to ValueError. axis=None -> scalar fast
// path (contiguous hpx::reduce); axis given (int or tuple) -> N-D axis reduction.
//
// SPDX-License-Identifier: MIT

#include "array.hpp"
#include "bind_helpers.hpp"
#include "bind_fwd.hpp"

#include <nanobind/nanobind.h>

#include <utility>

namespace nb = nanobind;
using namespace nb::literals;

using hpxpy::Array;
using hpxpy_bind::parse_axes;

void register_reductions(nb::module_&, nb::class_<Array>& cls)
{
    cls
        // sum/min/max: axis=None (default) -> scalar fast path (zero-penalty,
        // contiguous hpx::reduce); axis given (int or tuple) -> N-D axis reduction
        // returning a new Array. keepdims retains the reduced axes as size 1.
        .def("sum", [](Array const& a, nb::object axis, bool keepdims) -> nb::object {
            // GIL released around the C++ work; cast back to Python AFTER it is
            // re-acquired (the release scope ends before the nb::cast).
            if (axis.is_none()) {
                double s;
                { nb::gil_scoped_release release; s = a.sum(); }
                return nb::cast(s);
            }
            auto ax = parse_axes(axis, a.ndim());
            Array r;
            { nb::gil_scoped_release release; r = a.sum_axis(ax, keepdims); }
            return nb::cast(std::move(r));
        }, "axis"_a = nb::none(), "keepdims"_a = false,
           "Parallel sum. axis=None -> scalar (hpx::reduce); axis -> reduced Array.")
        .def("min", [](Array const& a, nb::object axis, bool keepdims) -> nb::object {
            if (axis.is_none()) {
                double s;
                { nb::gil_scoped_release release; s = a.min(); }
                return nb::cast(s);
            }
            auto ax = parse_axes(axis, a.ndim());
            Array r;
            { nb::gil_scoped_release release; r = a.min_axis(ax, keepdims); }
            return nb::cast(std::move(r));
        }, "axis"_a = nb::none(), "keepdims"_a = false,
           "Parallel minimum. axis=None -> scalar (empty -> ValueError); axis -> Array.")
        .def("max", [](Array const& a, nb::object axis, bool keepdims) -> nb::object {
            if (axis.is_none()) {
                double s;
                { nb::gil_scoped_release release; s = a.max(); }
                return nb::cast(s);
            }
            auto ax = parse_axes(axis, a.ndim());
            Array r;
            { nb::gil_scoped_release release; r = a.max_axis(ax, keepdims); }
            return nb::cast(std::move(r));
        }, "axis"_a = nb::none(), "keepdims"_a = false,
           "Parallel maximum. axis=None -> scalar (empty -> ValueError); axis -> Array.")
        // dot: 1-D . 1-D -> scalar (fused transform_reduce); 2-D . 2-D -> matmul Array.
        .def("dot", [](Array const& a, Array const& b) -> nb::object {
            if (a.ndim() == 1 && b.ndim() == 1) {
                double s;
                { nb::gil_scoped_release release; s = a.dot(b); }
                return nb::cast(s);
            }
            if (a.ndim() == 2 && b.ndim() == 2) {
                Array r;
                { nb::gil_scoped_release release; r = a.matmul(b); }
                return nb::cast(std::move(r));
            }
            throw nb::value_error(
                "dot: only 1-D . 1-D (scalar) and 2-D . 2-D (matmul) are supported");
        }, "b"_a, "1-D.1-D -> scalar (fused); 2-D.2-D -> matrix product (Array).")
        .def("matmul", [](Array const& a, Array const& b) {
            nb::gil_scoped_release release;
            return a.matmul(b);
        }, "b"_a, "2-D matrix product A @ B (naive O(m*n*k)).")
        .def("__matmul__", [](Array const& a, Array const& b) -> nb::object {
            if (a.ndim() == 1 && b.ndim() == 1) {
                double s;
                { nb::gil_scoped_release release; s = a.dot(b); }
                return nb::cast(s);
            }
            if (a.ndim() == 2 && b.ndim() == 2) {
                Array r;
                { nb::gil_scoped_release release; r = a.matmul(b); }
                return nb::cast(std::move(r));
            }
            throw nb::value_error(
                "@: only 1-D @ 1-D (scalar) and 2-D @ 2-D (matmul) are supported");
        });
}
