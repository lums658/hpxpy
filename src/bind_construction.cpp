// hpxpy._core — register_construction: zeros/ones/full/arange/arange_range/linspace/
// eye/empty.
//
// Module-level factory functions. Element counts and endpoint handling are resolved in
// Python (numpy rules); these bindings parse the dtype at the boundary and write the
// ramp / diagonal / fill / uninitialized buffer in parallel on the NUMA workers.
//
// SPDX-License-Identifier: MIT

#include "array.hpp"
#include "bind_helpers.hpp"
#include "bind_fwd.hpp"

#include <nanobind/nanobind.h>

#include <cstddef>
#include <utility>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

using hpxpy::Array;
using hpxpy::DType;
using hpxpy_bind::parse_dtype;

void register_construction(nb::module_& m)
{
    // zeros/full/ones accept either an int (1-D) or a tuple/list of ints (N-D), plus
    // a dtype (default float64). dtype is parsed at the binding boundary.
    m.def("zeros", [](nb::object shape_arg, nb::object dtype) -> Array {
        DType dt = parse_dtype(dtype);
        if (PyTuple_Check(shape_arg.ptr()) || PyList_Check(shape_arg.ptr())) {
            nb::sequence seq = nb::cast<nb::sequence>(shape_arg);
            std::vector<std::size_t> shape;
            for (nb::handle item : seq)
                shape.push_back(nb::cast<std::size_t>(item));
            return hpxpy::zeros_nd(std::move(shape), dt);
        }
        return hpxpy::zeros(nb::cast<std::size_t>(shape_arg), dt);
    }, "shape"_a, "dtype"_a = nb::none(),
       "Create an Array of zeros. shape may be an int (1-D) or tuple/list (N-D).");
    m.def("ones", [](nb::object shape_arg, nb::object dtype) -> Array {
        DType dt = parse_dtype(dtype);
        if (PyTuple_Check(shape_arg.ptr()) || PyList_Check(shape_arg.ptr())) {
            nb::sequence seq = nb::cast<nb::sequence>(shape_arg);
            std::vector<std::size_t> shape;
            for (nb::handle item : seq)
                shape.push_back(nb::cast<std::size_t>(item));
            return hpxpy::ones_nd(std::move(shape), dt);
        }
        return hpxpy::full(nb::cast<std::size_t>(shape_arg), 1.0, dt);
    }, "shape"_a, "dtype"_a = nb::none(),
       "Create an Array of ones. shape may be an int (1-D) or tuple/list (N-D).");
    m.def("full", [](nb::object shape_arg, double value, nb::object dtype) -> Array {
        DType dt = parse_dtype(dtype);
        if (PyTuple_Check(shape_arg.ptr()) || PyList_Check(shape_arg.ptr())) {
            nb::sequence seq = nb::cast<nb::sequence>(shape_arg);
            std::vector<std::size_t> shape;
            for (nb::handle item : seq)
                shape.push_back(nb::cast<std::size_t>(item));
            return hpxpy::full_nd(std::move(shape), value, dt);
        }
        return hpxpy::full(nb::cast<std::size_t>(shape_arg), value, dt);
    }, "shape"_a, "value"_a, "dtype"_a = nb::none(),
       "Create an Array filled with value. shape may be an int (1-D) or tuple/list (N-D).");
    m.def("arange", [](std::size_t n, nb::object dtype) -> Array {
        return hpxpy::arange(n, parse_dtype(dtype));
    }, "n"_a, "dtype"_a = nb::none(),
       "Create an Array [0, 1, ..., n-1] (NUMA-aware parallel first-touch).");

    // --- Construction helpers (Wave 2) ------------------------------------
    // arange_range/linspace/eye/empty: element counts and endpoint handling are
    // resolved in Python (numpy rules); these bindings parse the dtype and write the
    // ramp / diagonal / uninitialized buffer in parallel on the NUMA workers.
    m.def("arange_range", [](double start, double step, std::size_t n,
                             nb::object dtype) -> Array {
        return hpxpy::arange_range(start, step, n, parse_dtype(dtype));
    }, "start"_a, "step"_a, "n"_a, "dtype"_a = nb::none(),
       "n elements of the affine ramp start + i*step (numpy arange).");
    m.def("linspace", [](double start, double step, std::size_t num, bool set_last,
                         double last, nb::object dtype) -> Array {
        return hpxpy::linspace(start, step, num, set_last, last, parse_dtype(dtype));
    }, "start"_a, "step"_a, "num"_a, "set_last"_a, "last"_a, "dtype"_a = nb::none(),
       "num evenly spaced points start + i*step (last forced to stop if endpoint).");
    m.def("eye", [](std::size_t rows, std::size_t cols, std::ptrdiff_t k,
                    nb::object dtype) -> Array {
        return hpxpy::eye(rows, cols, k, parse_dtype(dtype));
    }, "rows"_a, "cols"_a, "k"_a, "dtype"_a = nb::none(),
       "2-D (rows x cols) array with ones on the k-th diagonal, zeros elsewhere.");
    m.def("empty", [](nb::object shape_arg, nb::object dtype) -> Array {
        DType dt = parse_dtype(dtype);
        if (PyTuple_Check(shape_arg.ptr()) || PyList_Check(shape_arg.ptr())) {
            nb::sequence seq = nb::cast<nb::sequence>(shape_arg);
            std::vector<std::size_t> shape;
            for (nb::handle item : seq)
                shape.push_back(nb::cast<std::size_t>(item));
            return hpxpy::empty_nd(std::move(shape), dt);
        }
        return hpxpy::empty_nd({nb::cast<std::size_t>(shape_arg)}, dt);
    }, "shape"_a, "dtype"_a = nb::none(),
       "Allocate an Array without initializing values (numpy.empty).");

    m.def("where", [](Array const& cond, Array const& x, Array const& y) {
        nb::gil_scoped_release r;
        return hpxpy::where(cond, x, y);
    }, "condition"_a, "x"_a, "y"_a,
       "Element-wise conditional: result[i] = x[i] if condition[i]!=0 else y[i].");

    m.def("cumprod", [](Array const& a) {
        nb::gil_scoped_release r; return a.cumprod();
    }, "a"_a, "Inclusive prefix product (numpy.cumprod).");

    m.def("argsort", [](Array const& a) {
        nb::gil_scoped_release r; return a.argsort();
    }, "a"_a, "I64 indices that would sort a ascending (numpy.argsort).");
}
