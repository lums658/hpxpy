// hpxpy._core — register_bench: bench / bench_dot / bench_binary.
//
// C++-timed benchmark entry points: each times a kernel in C++ (monotonic clock,
// adaptive repeats) with the GIL released, so the perf harness never times across the
// Python boundary. Returns (median_seconds, reps).
//
// SPDX-License-Identifier: MIT

#include "array.hpp"
#include "timing.hpp"
#include "bind_fwd.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>

#include <stdexcept>
#include <string>
#include <utility>

namespace nb = nanobind;
using namespace nb::literals;

using hpxpy::Array;

void register_bench(nb::module_& m)
{
    // C++-timed benchmark entry point: times a reduction in C++ (monotonic clock,
    // adaptive repeats) with the GIL released, so the perf harness never times
    // across the Python boundary. Returns (median_seconds, reps).
    m.def("bench", [](Array const& a, std::string const& op, double budget,
                      int min_reps, int max_reps) {
        nb::gil_scoped_release release;
        auto run = [&]() -> double {
            if (op == "sum") return a.sum();
            if (op == "min") return a.min();
            if (op == "max") return a.max();
            throw std::invalid_argument("unknown op: " + op);
        };
        hpxpy::timing::result r =
            hpxpy::timing::measure(run, budget, min_reps, max_reps);
        return std::make_pair(r.median_s, r.reps);
    }, "a"_a, "op"_a, "budget"_a = 0.5, "min_reps"_a = 5, "max_reps"_a = 200,
       "C++-timed median-of-times (s) and rep count for a reduction.");

    // Two-operand variant for dot (fused transform_reduce), same C++ timing.
    m.def("bench_dot", [](Array const& a, Array const& b, double budget,
                          int min_reps, int max_reps) {
        nb::gil_scoped_release release;
        hpxpy::timing::result r = hpxpy::timing::measure(
            [&]() -> double { return a.dot(b); }, budget, min_reps, max_reps);
        return std::make_pair(r.median_s, r.reps);
    }, "a"_a, "b"_a, "budget"_a = 0.5, "min_reps"_a = 5, "max_reps"_a = 200,
       "C++-timed median-of-times (s) and rep count for dot(a, b).");

    // Element-wise binary op benchmark (result is a new Array; the timing harness
    // keeps it observable so the transform is not elided).
    m.def("bench_binary", [](Array const& a, Array const& b, std::string const& op,
                             double budget, int min_reps, int max_reps) {
        nb::gil_scoped_release release;
        auto run = [&]() -> Array {
            if (op == "add") return a.add(b);
            if (op == "sub") return a.sub(b);
            if (op == "mul") return a.mul(b);
            if (op == "div") return a.div(b);
            throw std::invalid_argument("unknown op: " + op);
        };
        hpxpy::timing::result r =
            hpxpy::timing::measure(run, budget, min_reps, max_reps);
        return std::make_pair(r.median_s, r.reps);
    }, "a"_a, "b"_a, "op"_a, "budget"_a = 0.5, "min_reps"_a = 5, "max_reps"_a = 200,
       "C++-timed median-of-times (s) and rep count for an element-wise op.");
}
