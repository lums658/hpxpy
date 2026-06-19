// hpxpy._core — register_ufuncs: element-wise unary + binary math ufuncs (Wave 1).
//
// Unary (preserve-dtype): negative/abs/sign. Unary (promote int->f64): sqrt/exp/log/
// sin/cos/tan/floor/ceil/trunc/round. Binary (preserve-dtype): maximum/minimum/power/
// mod/floor_divide. Plus clip and the operator forms __neg__/__abs__/__pow__/__mod__/
// __floordiv__ (Array and scalar). Each releases the GIL around the kernel.
//
// SPDX-License-Identifier: MIT

#include "array.hpp"
#include "bind_fwd.hpp"

#include <nanobind/nanobind.h>

namespace nb = nanobind;
using namespace nb::literals;

using hpxpy::Array;

void register_ufuncs(nb::module_&, nb::class_<Array>& cls)
{
    cls
        // --- Element-wise unary math ufuncs (Wave 1) -------------------------
        // Preserve-dtype: negative/abs/sign. Promote-int-to-f64: sqrt/exp/log/sin/
        // cos/tan/floor/ceil/trunc/round. Each releases the GIL around the kernel.
        .def("negative", [](Array const& a) {
            nb::gil_scoped_release r; return a.negative();
        }, "Element-wise -a (preserves dtype).")
        .def("abs", [](Array const& a) {
            nb::gil_scoped_release r; return a.abs();
        }, "Element-wise absolute value (preserves dtype).")
        .def("sign", [](Array const& a) {
            nb::gil_scoped_release r; return a.sign();
        }, "Element-wise sign (-1/0/1, preserves dtype).")
        .def("sqrt", [](Array const& a) {
            nb::gil_scoped_release r; return a.sqrt();
        }, "Element-wise square root (int -> float64; float keeps dtype).")
        .def("exp", [](Array const& a) {
            nb::gil_scoped_release r; return a.exp();
        }, "Element-wise exp (int -> float64).")
        .def("log", [](Array const& a) {
            nb::gil_scoped_release r; return a.log();
        }, "Element-wise natural log (int -> float64).")
        .def("sin", [](Array const& a) {
            nb::gil_scoped_release r; return a.sin();
        }, "Element-wise sine (int -> float64).")
        .def("cos", [](Array const& a) {
            nb::gil_scoped_release r; return a.cos();
        }, "Element-wise cosine (int -> float64).")
        .def("tan", [](Array const& a) {
            nb::gil_scoped_release r; return a.tan();
        }, "Element-wise tangent (int -> float64).")
        .def("floor", [](Array const& a) {
            nb::gil_scoped_release r; return a.floor();
        }, "Element-wise floor (int -> float64).")
        .def("ceil", [](Array const& a) {
            nb::gil_scoped_release r; return a.ceil();
        }, "Element-wise ceil (int -> float64).")
        .def("trunc", [](Array const& a) {
            nb::gil_scoped_release r; return a.trunc();
        }, "Element-wise truncate toward zero (int -> float64).")
        .def("round", [](Array const& a) {
            nb::gil_scoped_release r; return a.round();
        }, "Element-wise round-half-to-even (int -> float64).")
        // Operators: __neg__ / __abs__.
        .def("__neg__", [](Array const& a) {
            nb::gil_scoped_release r; return a.negative();
        })
        .def("__abs__", [](Array const& a) {
            nb::gil_scoped_release r; return a.abs();
        })
        // --- Element-wise binary math ufuncs (Wave 1; preserve-dtype) --------
        // maximum/minimum/power/mod/floor_divide. Same-dtype enforced by binary().
        .def("maximum", [](Array const& a, Array const& b) {
            nb::gil_scoped_release r; return a.maximum(b);
        }, "b"_a, "Element-wise max(a, b) (same dtype).")
        .def("minimum", [](Array const& a, Array const& b) {
            nb::gil_scoped_release r; return a.minimum(b);
        }, "b"_a, "Element-wise min(a, b) (same dtype).")
        .def("power", [](Array const& a, Array const& b) {
            nb::gil_scoped_release r; return a.power(b);
        }, "b"_a, "Element-wise a ** b (same dtype).")
        .def("mod", [](Array const& a, Array const& b) {
            nb::gil_scoped_release r; return a.mod(b);
        }, "b"_a, "Element-wise a %% b (divisor-signed, like numpy).")
        .def("floor_divide", [](Array const& a, Array const& b) {
            nb::gil_scoped_release r; return a.floor_divide(b);
        }, "b"_a, "Element-wise a // b (floor toward -inf, like numpy).")
        .def("clip", [](Array const& a, double lo, double hi) {
            nb::gil_scoped_release r; return a.clip(lo, hi);
        }, "lo"_a, "hi"_a, "Clamp each element to [lo, hi] (preserves dtype).")
        // Operators: __pow__ / __mod__ / __floordiv__ (Array forms).
        .def("__pow__", [](Array const& a, Array const& b) {
            nb::gil_scoped_release r; return a.power(b);
        })
        .def("__mod__", [](Array const& a, Array const& b) {
            nb::gil_scoped_release r; return a.mod(b);
        })
        .def("__floordiv__", [](Array const& a, Array const& b) {
            nb::gil_scoped_release r; return a.floor_divide(b);
        })
        // Scalar operator forms (a ** s, a % s, a // s). nanobind tries the Array
        // overloads above first, then these. Reflected forms (s ** a, ...) are out
        // of scope for Wave 1 (uncommon for these ops).
        .def("__pow__", [](Array const& a, double s) {
            nb::gil_scoped_release r; return a.pow_scalar(s);
        })
        .def("__mod__", [](Array const& a, double s) {
            nb::gil_scoped_release r; return a.mod_scalar(s);
        })
        .def("__floordiv__", [](Array const& a, double s) {
            nb::gil_scoped_release r; return a.floordiv_scalar(s);
        });
}
