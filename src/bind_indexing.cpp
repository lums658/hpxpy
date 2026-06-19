// hpxpy._core — register_indexing: __getitem__/__setitem__/__len__/__repr__ + the
// scalar properties (size/ndim/stride/shape/dtype) + to_numpy/__array__.
//
// Index/bounds normalization (numpy semantics) lives HERE; the wrapper is raw. The
// per-element scalar read/write helpers (getitem_at/setitem_at) are TU-local. The
// fill_range/assign_range paths are reached through __setitem__ on a slice key.
//
// SPDX-License-Identifier: MIT

#include "array.hpp"
#include "bind_helpers.hpp"
#include "bind_fwd.hpp"

#include <nanobind/nanobind.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

using hpxpy::Array;
using hpxpy::DType;
using hpxpy_bind::numpy_dtype_object;
using hpxpy_bind::to_numpy_view;

namespace {

// Scalar element read at logical ELEMENT offset `off` (already includes strides),
// dispatched on the array's dtype: int64 -> Python int, float -> Python float.
nb::object getitem_at(Array const& a, std::ptrdiff_t off)
{
    return hpxpy::dispatch_dtype(a.dtype(), [&](auto tag) -> nb::object {
        using T = decltype(tag);
        T v = a.template data_as<T>()[off];
        return nb::cast(v);
    });
}

// Scalar element write at logical ELEMENT offset `off`, casting the Python value to
// the array's element type T (numpy-faithful: a[i] = 3.9 on an int64 array stores 3;
// a[i] = 5 on a float32 array stores 5.0). The Python value is read as a double (which
// accepts both Python int and float) and static_cast to T — matching numpy's assign
// semantics for our three real dtypes.
void setitem_at(Array& a, std::ptrdiff_t off, nb::handle value)
{
    double v = nb::cast<double>(nb::borrow(value));    // accepts int or float
    hpxpy::dispatch_dtype(a.dtype(), [&](auto tag) {
        using T = decltype(tag);
        a.template data_as<T>()[off] = static_cast<T>(v);
    });
}

}  // namespace

void register_indexing(nb::module_&, nb::class_<Array>& cls)
{
    cls
        .def_prop_ro("size", &Array::size)
        .def_prop_ro("ndim", &Array::ndim)
        .def_prop_ro("stride", &Array::stride)
        .def_prop_ro("shape", [](Array const& a) {
            nb::tuple t = nb::steal<nb::tuple>(PyTuple_New((Py_ssize_t) a.shape().size()));
            for (std::size_t k = 0; k < a.shape().size(); ++k)
                PyTuple_SET_ITEM(t.ptr(), (Py_ssize_t) k,
                                 PyLong_FromSize_t(a.shape()[k]));
            return t;
        }, "Shape of the array as a tuple of ints (N-D).")
        // dtype: the numpy dtype object (so a.dtype == np.float32 works like numpy).
        .def_prop_ro("dtype", [](Array const& a) {
            return numpy_dtype_object(a.dtype());
        }, "The element dtype as a numpy dtype object (float64/float32/int64).")
        // Indexing: a[i] -> float, a[i:j] -> contiguous VIEW (shares memory).
        // a[i:j:k] with k != 1 -> strided VIEW (shares memory, no copy).
        // a[i,j,...] (tuple of ints, len==ndim) -> multi-index scalar get (N-D).
        // a[i:j, ::k, m, ...] (tuple with a slice, Ellipsis, int) -> N-D VIEW (stage 6).
        // Index/bounds normalization (numpy semantics) lives here; the wrapper is raw.
        .def("__getitem__", [](Array const& a, nb::object key) -> nb::object {
            // Tuple key: multi-index (all ints) or N-D slice (real, stage 6).
            // Supports int / slice / Ellipsis specifiers per axis; one Ellipsis
            // expands to full slices; missing trailing axes get full slices.
            if (PyTuple_Check(key.ptr())) {
                Py_ssize_t const tlen = PyTuple_GET_SIZE(key.ptr());
                std::size_t const ndim = a.ndim();

                // Count non-Ellipsis specifiers; at most one Ellipsis allowed.
                Py_ssize_t n_non_ellipsis = 0;
                Py_ssize_t ellipsis_pos = -1;
                for (Py_ssize_t k = 0; k < tlen; ++k) {
                    PyObject* item = PyTuple_GET_ITEM(key.ptr(), k);
                    if (item == Py_Ellipsis) {
                        if (ellipsis_pos != -1)
                            throw nb::index_error(
                                "an index can only have a single ellipsis ('...')");
                        ellipsis_pos = k;
                    } else {
                        ++n_non_ellipsis;
                    }
                }
                if ((std::size_t) n_non_ellipsis > ndim)
                    throw nb::index_error(
                        "too many indices for array");

                // Build the per-axis spec list, expanding Ellipsis / padding.
                std::vector<Array::AxisSpec> specs;
                specs.reserve(ndim);
                bool all_int = true;
                std::size_t ax = 0;    // current input axis
                auto handle_specifier = [&](PyObject* item) {
                    Py_ssize_t dim = (Py_ssize_t) a.shape()[ax];
                    if (PySlice_Check(item)) {
                        all_int = false;
                        Py_ssize_t start, stop, step;
                        if (PySlice_Unpack(item, &start, &stop, &step) < 0)
                            throw nb::python_error();
                        Py_ssize_t slen =
                            PySlice_AdjustIndices(dim, &start, &stop, step);
                        Array::AxisSpec s;
                        s.kind = Array::AxisSpec::SLICE;
                        s.start = (std::ptrdiff_t) start;
                        s.step = (std::ptrdiff_t) step;
                        s.slicelen = (std::size_t) slen;
                        specs.push_back(s);
                    } else {
                        Py_ssize_t ix =
                            PyNumber_AsSsize_t(item, PyExc_IndexError);
                        if (ix == -1 && PyErr_Occurred()) throw nb::python_error();
                        if (ix < 0) ix += dim;
                        if (ix < 0 || ix >= dim)
                            throw nb::index_error("Array index out of range");
                        Array::AxisSpec s;
                        s.kind = Array::AxisSpec::INT;
                        s.int_idx = (std::ptrdiff_t) ix;
                        specs.push_back(s);
                    }
                    ++ax;
                };
                for (Py_ssize_t k = 0; k < tlen; ++k) {
                    PyObject* item = PyTuple_GET_ITEM(key.ptr(), k);
                    if (item == Py_Ellipsis) {
                        // Expand to (ndim - n_non_ellipsis) full slices.
                        std::size_t fill = ndim - (std::size_t) n_non_ellipsis;
                        for (std::size_t f = 0; f < fill; ++f) {
                            all_int = false;
                            Array::AxisSpec s;
                            s.kind = Array::AxisSpec::SLICE;
                            s.start = 0;
                            s.step = 1;
                            s.slicelen = a.shape()[ax];
                            specs.push_back(s);
                            ++ax;
                        }
                    } else {
                        handle_specifier(item);
                    }
                }
                // No ellipsis & fewer specifiers than ndim: pad trailing full slices.
                while (ax < ndim) {
                    all_int = false;
                    Array::AxisSpec s;
                    s.kind = Array::AxisSpec::SLICE;
                    s.start = 0;
                    s.step = 1;
                    s.slicelen = a.shape()[ax];
                    specs.push_back(s);
                    ++ax;
                }

                // All-int, full rank -> scalar multi-index (dtype-aware read).
                if (all_int && specs.size() == ndim) {
                    std::vector<std::size_t> idx(ndim);
                    for (std::size_t k = 0; k < ndim; ++k)
                        idx[k] = (std::size_t) specs[k].int_idx;
                    return getitem_at(a, (std::ptrdiff_t) a.linear_offset(idx));
                }
                return nb::cast(a.slice_nd(specs));
            }
            if (PySlice_Check(key.ptr())) {
                // N-D array, plain slice: a[1:3] == a[1:3, :, ...] (implicit
                // trailing full slices). Route through slice_nd.
                if (a.ndim() > 1) {
                    std::vector<Array::AxisSpec> specs;
                    specs.reserve(a.ndim());
                    Py_ssize_t start, stop, step;
                    if (PySlice_Unpack(key.ptr(), &start, &stop, &step) < 0)
                        throw nb::python_error();
                    Py_ssize_t slen = PySlice_AdjustIndices(
                        (Py_ssize_t) a.shape()[0], &start, &stop, step);
                    Array::AxisSpec s0;
                    s0.kind = Array::AxisSpec::SLICE;
                    s0.start = (std::ptrdiff_t) start;
                    s0.step = (std::ptrdiff_t) step;
                    s0.slicelen = (std::size_t) slen;
                    specs.push_back(s0);
                    for (std::size_t ax = 1; ax < a.ndim(); ++ax) {
                        Array::AxisSpec s;
                        s.kind = Array::AxisSpec::SLICE;
                        s.start = 0;
                        s.step = 1;
                        s.slicelen = a.shape()[ax];
                        specs.push_back(s);
                    }
                    return nb::cast(a.slice_nd(specs));
                }
                Py_ssize_t start, stop, step;
                if (PySlice_Unpack(key.ptr(), &start, &stop, &step) < 0)
                    throw nb::python_error();
                Py_ssize_t const n =
                    PySlice_AdjustIndices((Py_ssize_t) a.size(), &start, &stop, step);
                if (step == 1)
                    return nb::cast(a.view((std::size_t) start, (std::size_t) n));
                // Strided slice (including step==-1 for reverse): zero-copy view.
                return nb::cast(a.view_strided((std::ptrdiff_t) start,
                                               (std::size_t) n,
                                               (std::ptrdiff_t) step));
            }
            // Bare Ellipsis: a[...] -> full view of all axes (numpy semantics).
            if (key.ptr() == Py_Ellipsis) {
                std::vector<Array::AxisSpec> specs;
                specs.reserve(a.ndim());
                for (std::size_t ax = 0; ax < a.ndim(); ++ax) {
                    Array::AxisSpec s;
                    s.kind = Array::AxisSpec::SLICE;
                    s.start = 0;
                    s.step = 1;
                    s.slicelen = a.shape()[ax];
                    specs.push_back(s);
                }
                return nb::cast(a.slice_nd(specs));
            }
            Py_ssize_t i = PyNumber_AsSsize_t(key.ptr(), PyExc_IndexError);
            if (i == -1 && PyErr_Occurred())
                throw nb::python_error();
            Py_ssize_t const n = (Py_ssize_t) a.size();
            if (i < 0)
                i += n;
            if (i < 0 || i >= n)
                throw nb::index_error("Array index out of range");
            return getitem_at(a, a.offset_1d((std::size_t) i));
        }, "a[i] -> scalar (float/int by dtype); a[i:j] -> contiguous view; "
           "a[i:j:k] -> strided view (zero-copy); "
           "a[i,j,...] (tuple of ints, len==ndim) -> scalar (N-D multi-index); "
           "a[i:j, ::k, m, ...] (tuple with slice/Ellipsis) -> N-D view (zero-copy).")
        .def("__setitem__", [](Array& a, nb::object key, nb::object value) {
            // Tuple key: multi-index set (all ints) or N-D slice (deferred).
            if (PyTuple_Check(key.ptr())) {
                Py_ssize_t tlen = PyTuple_GET_SIZE(key.ptr());
                for (Py_ssize_t k = 0; k < tlen; ++k) {
                    if (PySlice_Check(PyTuple_GET_ITEM(key.ptr(), k)))
                        throw nb::type_error(
                            "N-D slice assignment is not yet implemented (stage 6).");
                }
                if (tlen != (Py_ssize_t) a.ndim())
                    throw nb::index_error("index tuple length must equal ndim");
                std::vector<std::size_t> idx(tlen);
                for (Py_ssize_t k = 0; k < tlen; ++k) {
                    Py_ssize_t ix = PyNumber_AsSsize_t(
                        PyTuple_GET_ITEM(key.ptr(), k), PyExc_IndexError);
                    if (ix == -1 && PyErr_Occurred()) throw nb::python_error();
                    Py_ssize_t dim = (Py_ssize_t) a.shape()[k];
                    if (ix < 0) ix += dim;
                    if (ix < 0 || ix >= dim)
                        throw nb::index_error("Array index out of range");
                    idx[k] = (std::size_t) ix;
                }
                setitem_at(a, (std::ptrdiff_t) a.linear_offset(idx), value);
                return;
            }
            if (PySlice_Check(key.ptr())) {
                Py_ssize_t start, stop, step;
                if (PySlice_Unpack(key.ptr(), &start, &stop, &step) < 0)
                    throw nb::python_error();
                Py_ssize_t const n =
                    PySlice_AdjustIndices((Py_ssize_t) a.size(), &start, &stop, step);
                if (step != 1)
                    throw nb::value_error(
                        "hpxpy.Array supports only contiguous slice assignment (step == 1)");
                if (nb::isinstance<Array>(value)) {          // a[i:j] = Array (copy in)
                    Array const& rhs = nb::cast<Array&>(value);
                    if (rhs.size() != (std::size_t) n)
                        throw nb::value_error("slice assignment size mismatch");
                    nb::gil_scoped_release r;
                    a.assign_range((std::size_t) start, rhs);
                } else {                                     // a[i:j] = scalar (fill)
                    double v = nb::cast<double>(value);
                    nb::gil_scoped_release r;
                    a.fill_range((std::size_t) start, (std::size_t) n, v);
                }
                return;
            }
            Py_ssize_t i = PyNumber_AsSsize_t(key.ptr(), PyExc_IndexError);
            if (i == -1 && PyErr_Occurred())
                throw nb::python_error();
            Py_ssize_t const n = (Py_ssize_t) a.size();
            if (i < 0)
                i += n;
            if (i < 0 || i >= n)
                throw nb::index_error("Array index out of range");
            setitem_at(a, a.offset_1d((std::size_t) i), value);
        }, "a[i] = value; a[i:j] = scalar (fill) or Array (copy, contiguous step 1); "
           "a[i,j,...] = scalar (N-D multi-index set).")
        .def("to_numpy", &to_numpy_view,
             "Zero-copy NumPy view (writable; shares memory with the Array).")
        .def("__array__", [](nb::object self, nb::args, nb::kwargs) {
            return to_numpy_view(self);    // np.asarray(a) views; np.array(a) copies
        })
        .def("__len__", &Array::size)
        .def("__repr__", [](Array const& a) {
            const char* dn = "float64";
            switch (a.dtype()) {
                case DType::F64: dn = "float64"; break;
                case DType::F32: dn = "float32"; break;
                case DType::I64: dn = "int64";   break;
            }
            return "Array(size=" + std::to_string(a.size()) +
                   ", dtype=" + dn + ")";
        });
}
