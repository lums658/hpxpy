// hpxpy._core — register_views: transpose/.T/reshape/ravel/squeeze/expand_dims.
//
// N-D view ops (stage 3): zero-copy when the source is contiguous (non-contiguous
// reshape copies). Shape/axes specifiers are parsed at the binding boundary; the
// wrapper does the actual stride math.
//
// SPDX-License-Identifier: MIT

#include "array.hpp"
#include "bind_fwd.hpp"

#include <nanobind/nanobind.h>

#include <cstddef>
#include <utility>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

using hpxpy::Array;

void register_views(nb::module_&, nb::class_<Array>& cls)
{
    cls
        // transpose(axes=None): permute axes; empty/None => reverse all.
        .def("transpose", [](Array const& a, nb::object axes_arg) -> Array {
            if (axes_arg.is_none()) {
                return a.transpose();
            }
            // Accept a tuple or list of ints.
            std::vector<std::size_t> axes;
            nb::sequence seq = nb::cast<nb::sequence>(axes_arg);
            for (nb::handle item : seq)
                axes.push_back(nb::cast<std::size_t>(item));
            return a.transpose(std::move(axes));
        }, "axes"_a = nb::none(),
           "Permute axes (None/empty => reverse). Zero-copy view sharing memory.")
        .def_prop_ro("T", [](Array const& a) { return a.transpose(); },
            "Shorthand for .transpose() with no args (reverse all axes).")
        // reshape(new_shape): product must equal size. Contiguous => view; else copy.
        // Accepts an int (1-D), a tuple, or a list. Supports a single -1 for inference.
        .def("reshape", [](Array const& a, nb::object shape_arg) -> Array {
            std::vector<std::ptrdiff_t> new_shape;
            if (PyLong_Check(shape_arg.ptr())) {
                new_shape.push_back(nb::cast<std::ptrdiff_t>(shape_arg));
            } else {
                nb::sequence seq = nb::cast<nb::sequence>(shape_arg);
                for (nb::handle item : seq)
                    new_shape.push_back(nb::cast<std::ptrdiff_t>(item));
            }
            return a.reshape(std::move(new_shape));
        }, "shape"_a,
           "Reshape to new_shape (int or tuple). Contiguous => zero-copy view; "
           "non-contiguous => copy. One -1 is inferred.")
        .def("ravel", [](Array const& a) { return a.ravel(); },
            "Flatten to 1-D (reshape to (size,)). Contiguous => zero-copy view.")
        // squeeze(axis=None): drop size-1 dims. axis may be None, int, or tuple.
        .def("squeeze", [](Array const& a, nb::object axis_arg) -> Array {
            if (axis_arg.is_none())
                return a.squeeze();
            std::vector<std::size_t> axes;
            if (PyLong_Check(axis_arg.ptr())) {
                axes.push_back(nb::cast<std::size_t>(axis_arg));
            } else {
                nb::sequence seq = nb::cast<nb::sequence>(axis_arg);
                for (nb::handle item : seq)
                    axes.push_back(nb::cast<std::size_t>(item));
            }
            return a.squeeze(std::move(axes));
        }, "axis"_a = nb::none(),
           "Remove size-1 dims (all if axis=None, else the named axis/axes).")
        // expand_dims(axis): insert a size-1 dim.
        .def("expand_dims", [](Array const& a, std::size_t axis) {
            return a.expand_dims(axis);
        }, "axis"_a, "Insert a size-1 dimension at position axis.");
}
