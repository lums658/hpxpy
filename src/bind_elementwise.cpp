// hpxpy._core — register_elementwise: add/sub/mul/div + all operator forms.
//
// Element-wise binary ops -> new Array (hpx::transform). Operators and the NumPy-style
// named methods share one kernel; GIL released around the work. Scalar broadcast forms
// (a ⊙ s and reflected s ⊙ a) come in two overloads per op because numpy promotes by
// the Python scalar's TYPE: the int64_t overload is declared FIRST so a Python `int`
// binds to it and PRESERVES the array dtype (int64_arr + 2 -> int64); a Python `float`
// falls through to the double overload which PROMOTES int64 -> float64.
//
// SPDX-License-Identifier: MIT

#include "array.hpp"
#include "bind_fwd.hpp"

#include <nanobind/nanobind.h>

#include <cstdint>

namespace nb = nanobind;
using namespace nb::literals;

using hpxpy::Array;

void register_elementwise(nb::module_&, nb::class_<Array>& cls)
{
    cls
        .def("add", [](Array const& a, Array const& b) {
            nb::gil_scoped_release r; return a.add(b);
        }, "b"_a, "Element-wise a + b.")
        .def("sub", [](Array const& a, Array const& b) {
            nb::gil_scoped_release r; return a.sub(b);
        }, "b"_a, "Element-wise a - b.")
        .def("mul", [](Array const& a, Array const& b) {
            nb::gil_scoped_release r; return a.mul(b);
        }, "b"_a, "Element-wise a * b.")
        .def("div", [](Array const& a, Array const& b) {
            nb::gil_scoped_release r; return a.div(b);
        }, "b"_a, "Element-wise a / b.")
        .def("__add__", [](Array const& a, Array const& b) {
            nb::gil_scoped_release r; return a.add(b);
        })
        .def("__sub__", [](Array const& a, Array const& b) {
            nb::gil_scoped_release r; return a.sub(b);
        })
        .def("__mul__", [](Array const& a, Array const& b) {
            nb::gil_scoped_release r; return a.mul(b);
        })
        .def("__truediv__", [](Array const& a, Array const& b) {
            nb::gil_scoped_release r; return a.div(b);
        })
        // Scalar broadcast: a ⊙ s and the reflected s ⊙ a (one unary transform).
        // nanobind tries the Array×Array overloads above first, then these.
        //
        // Two scalar overloads per op (numpy promotes by the Python scalar's TYPE):
        // the int64_t overload is declared FIRST so a Python `int` binds to it and
        // PRESERVES the array dtype (int64_arr + 2 -> int64); a Python `float` falls
        // through to the double overload which PROMOTES int64 -> float64 (numpy:
        // int64_arr + 1.5 -> float64).
        .def("__add__", [](Array const& a, std::int64_t s) {
            nb::gil_scoped_release r; return a.add_scalar_int(s);
        })
        .def("__add__", [](Array const& a, double s) {
            nb::gil_scoped_release r; return a.add_scalar(s);
        })
        .def("__radd__", [](Array const& a, std::int64_t s) {
            nb::gil_scoped_release r; return a.add_scalar_int(s);
        })
        .def("__radd__", [](Array const& a, double s) {
            nb::gil_scoped_release r; return a.add_scalar(s);
        })
        .def("__sub__", [](Array const& a, std::int64_t s) {
            nb::gil_scoped_release r; return a.sub_scalar_int(s);
        })
        .def("__sub__", [](Array const& a, double s) {
            nb::gil_scoped_release r; return a.sub_scalar(s);
        })
        .def("__rsub__", [](Array const& a, std::int64_t s) {
            nb::gil_scoped_release r; return a.rsub_scalar_int(s);    // s - a
        })
        .def("__rsub__", [](Array const& a, double s) {
            nb::gil_scoped_release r; return a.rsub_scalar(s);    // s - a
        })
        .def("__mul__", [](Array const& a, std::int64_t s) {
            nb::gil_scoped_release r; return a.mul_scalar_int(s);
        })
        .def("__mul__", [](Array const& a, double s) {
            nb::gil_scoped_release r; return a.mul_scalar(s);
        })
        .def("__rmul__", [](Array const& a, std::int64_t s) {
            nb::gil_scoped_release r; return a.mul_scalar_int(s);
        })
        .def("__rmul__", [](Array const& a, double s) {
            nb::gil_scoped_release r; return a.mul_scalar(s);
        })
        .def("__truediv__", [](Array const& a, std::int64_t s) {
            nb::gil_scoped_release r; return a.div_scalar_int(s);
        })
        .def("__truediv__", [](Array const& a, double s) {
            nb::gil_scoped_release r; return a.div_scalar(s);
        })
        .def("__rtruediv__", [](Array const& a, std::int64_t s) {
            nb::gil_scoped_release r; return a.rdiv_scalar_int(s);    // s / a
        })
        .def("__rtruediv__", [](Array const& a, double s) {
            nb::gil_scoped_release r; return a.rdiv_scalar(s);    // s / a
        });
}
