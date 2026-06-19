// hpxpy._core — shared binding helpers (multi-TU split).
//
// These helpers are needed by MORE THAN ONE binding TU, so they live here as inline
// functions / shared typedefs rather than being duplicated. To keep the LIGHT TUs
// (bind_runtime / bind_bench) cheap to compile, this header pulls in ONLY nanobind +
// the Python C API — NOT <hpx/...> and NOT array.hpp. The element dtype enum is
// forward-declared (it is a fixed-underlying-type enum, so a forward declaration is a
// complete-enough type for the switch-on-dtype helpers below).
//
// to_numpy_view() needs the full Array definition, so it is only DECLARED here and
// DEFINED once in bind_bridge.cpp (the TU that owns the NumPy bridge). The compute TUs
// that call it (bridge + indexing) include array.hpp themselves before using it.
//
// SPDX-License-Identifier: MIT
#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>   // numpy_dtype_object/parse_dtype cast std::string

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// Forward declarations only — no hpx / array.hpp pulled in here.
namespace hpxpy {
enum class DType : std::uint8_t;
class Array;
}  // namespace hpxpy

namespace hpxpy_bind {

namespace nb = nanobind;
using hpxpy::Array;
using hpxpy::DType;

// NumPy bridge types: a runtime-dtype C-contiguous input (writable, so a borrow can
// share mutations both ways), and a runtime-dtype output view. The element dtype is
// carried at run time so ONE bridge serves float64/float32/int64 (no-silent-cast).
using np_rw = nb::ndarray<nb::numpy, nb::c_contig>;   // any rank/dtype, C-contiguous in
using np_out = nb::ndarray<nb::numpy>;                // any rank/dtype out (dynamic)

// dlpack dtype tags for the three supported element types (runtime comparison).
inline nb::dlpack::dtype nb_dtype_of(DType d)
{
    switch (d) {
        case DType::F64: return nb::dtype<double>();
        case DType::F32: return nb::dtype<float>();
        case DType::I64: return nb::dtype<int64_t>();
    }
    return nb::dtype<double>();
}

// Map a runtime numpy/dlpack dtype to our DType, or throw (no silent cast). Only
// float64/float32/int64 are supported; anything else (int32, float16, complex, ...)
// is rejected with TypeError.
inline DType dtype_from_nb(nb::dlpack::dtype dt)
{
    if (dt == nb::dtype<double>())  return DType::F64;
    if (dt == nb::dtype<float>())   return DType::F32;
    if (dt == nb::dtype<int64_t>()) return DType::I64;
    throw nb::type_error(
        "unsupported dtype: hpxpy supports float64, float32, and int64 only "
        "(no silent cast)");
}

// parse_dtype(obj): accept numpy dtype objects, numpy scalar types, dtype strings
// ('float32','f4','int64','i8', ...), and Python float->F64 / int->I64. Normalizes
// via numpy.dtype(obj) then maps to {F64,F32,I64}; raises TypeError on anything else
// (int32, float16, complex, bool, ...). None -> F64 (the default).
inline DType parse_dtype(nb::object obj)
{
    if (obj.is_none())
        return DType::F64;
    // Python built-in scalar types map like numpy: float->float64, int->int64.
    if (obj.is(reinterpret_cast<PyObject*>(&PyFloat_Type)))
        return DType::F64;
    if (obj.is(reinterpret_cast<PyObject*>(&PyLong_Type)))
        return DType::I64;
    nb::module_ np = nb::module_::import_("numpy");
    nb::object npdt;
    try {
        npdt = np.attr("dtype")(obj);    // normalize anything numpy understands
    } catch (nb::python_error&) {
        throw nb::type_error("invalid dtype");
    }
    std::string name = nb::cast<std::string>(npdt.attr("name"));
    if (name == "float64") return DType::F64;
    if (name == "float32") return DType::F32;
    if (name == "int64")   return DType::I64;
    throw nb::type_error(
        ("unsupported dtype '" + name +
         "': hpxpy supports float64, float32, and int64 only").c_str());
}

// The numpy dtype OBJECT for a DType (so a.dtype == np.float32 works like numpy).
inline nb::object numpy_dtype_object(DType d)
{
    nb::module_ np = nb::module_::import_("numpy");
    const char* name = "float64";
    switch (d) {
        case DType::F64: name = "float64"; break;
        case DType::F32: name = "float32"; break;
        case DType::I64: name = "int64";   break;
    }
    return np.attr("dtype")(name);
}

// Parse an axis argument (int, or tuple/list of ints) into normalized axes:
// negatives wrapped (+ndim), validated in range, deduplicated, and sorted ascending.
// `None` is handled by the caller (full reduction -> scalar) and never reaches here.
// Bad axis -> IndexError; duplicate axis -> ValueError (numpy semantics).
inline std::vector<std::size_t> parse_axes(nb::object axis, std::size_t ndim)
{
    auto norm_one = [ndim](Py_ssize_t ax) -> std::size_t {
        Py_ssize_t nd = static_cast<Py_ssize_t>(ndim);
        if (ax < 0) ax += nd;
        if (ax < 0 || ax >= nd)
            throw nb::index_error("axis out of range for array dimensions");
        return static_cast<std::size_t>(ax);
    };

    std::vector<std::size_t> raw;
    if (PyTuple_Check(axis.ptr()) || PyList_Check(axis.ptr())) {
        nb::sequence seq = nb::cast<nb::sequence>(axis);
        for (nb::handle item : seq) {
            Py_ssize_t ax = PyNumber_AsSsize_t(item.ptr(), PyExc_IndexError);
            if (ax == -1 && PyErr_Occurred()) throw nb::python_error();
            raw.push_back(norm_one(ax));
        }
    } else {
        Py_ssize_t ax = PyNumber_AsSsize_t(axis.ptr(), PyExc_IndexError);
        if (ax == -1 && PyErr_Occurred()) throw nb::python_error();
        raw.push_back(norm_one(ax));
    }

    // Detect duplicates (after normalization), then sort + unique.
    std::vector<std::size_t> sorted = raw;
    std::sort(sorted.begin(), sorted.end());
    for (std::size_t k = 1; k < sorted.size(); ++k)
        if (sorted[k] == sorted[k - 1])
            throw nb::value_error("duplicate value in axis");
    return sorted;
}

// Zero-copy NumPy view of an Array (any rank). DECLARED here; DEFINED in
// bind_bridge.cpp (needs the full Array type). `obj` (the Array Python object) owns
// the ndarray, so the HPX buffer outlives the view.
np_out to_numpy_view(nb::object obj);

}  // namespace hpxpy_bind
