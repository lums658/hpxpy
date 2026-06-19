// hpxpy._core — register_random: the hpxpy.random submodule.
//
// Default seed: std::random_device at module import — truly non-deterministic.
// Reproducible runs: seed(n) sets an explicit seed and resets the counter;
// get_seed() / get_state() / set_state() let the caller save and restore
// exactly enough state to replay any sequence.
//
// Generation: pure index-based splitmix64 hash — element i always maps to
// f(seed, counter+i) regardless of thread count or scheduler chunk boundaries,
// so reproducibility is exact even under parallel execution.
//
// SPDX-License-Identifier: MIT

#include "array.hpp"
#include "bind_fwd.hpp"

#include <hpx/execution.hpp>
#include <hpx/algorithm.hpp>

#include <nanobind/nanobind.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;
using hpxpy::Array;
using hpxpy::DType;

namespace {

static uint64_t g_seed    = 0;
static uint64_t g_counter = 0;

// splitmix64 hash: (seed, index) -> uniform double in [0, 1).
// Index-based so the same (seed, i) always yields the same value — no shared
// mutable state in the parallel loop, no data races.
inline double uniform01(uint64_t seed, uint64_t idx) noexcept
{
    uint64_t x = seed + idx * 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return static_cast<double>(x >> 11) * (1.0 / 9007199254740992.0);
}

std::vector<std::size_t> parse_shape(nb::args const& args)
{
    if (args.size() == 0)
        throw std::invalid_argument("shape must have at least one dimension");
    if (args.size() == 1) {
        nb::handle h = args[0];
        if (PyTuple_Check(h.ptr()) || PyList_Check(h.ptr())) {
            std::vector<std::size_t> shape;
            for (nb::handle item : nb::cast<nb::sequence>(h))
                shape.push_back(nb::cast<std::size_t>(item));
            return shape;
        }
        return {nb::cast<std::size_t>(h)};
    }
    std::vector<std::size_t> shape;
    for (nb::handle h : args) shape.push_back(nb::cast<std::size_t>(h));
    return shape;
}

std::vector<std::size_t> parse_size(nb::object const& size_arg)
{
    if (PyTuple_Check(size_arg.ptr()) || PyList_Check(size_arg.ptr())) {
        std::vector<std::size_t> shape;
        for (nb::handle item : nb::cast<nb::sequence>(size_arg))
            shape.push_back(nb::cast<std::size_t>(item));
        return shape;
    }
    return {nb::cast<std::size_t>(size_arg)};
}

std::size_t total(std::vector<std::size_t> const& shape)
{
    std::size_t n = 1;
    for (auto d : shape) n *= d;
    return n;
}

} // namespace

void register_random(nb::module_& m)
{
    // Seed from random_device at module import — truly non-deterministic by default.
    {
        std::random_device rd;
        g_seed = (static_cast<uint64_t>(rd()) << 32) | static_cast<uint64_t>(rd());
    }

    auto rm = m.def_submodule("random",
        "Parallel random number generation (index-based, reproducible when seeded).");

    rm.def("seed", [](uint64_t s) { g_seed = s; g_counter = 0; },
        "s"_a,
        "Set the global seed and reset the counter. "
        "All subsequent draws are fully reproducible.");

    rm.def("get_seed", []() -> uint64_t { return g_seed; },
        "Return the current seed. Save this to replay any run.");

    rm.def("get_state", []() -> std::tuple<uint64_t, uint64_t> {
        return {g_seed, g_counter};
    }, "Return (seed, counter). Pass to set_state() to resume from any mid-run point.");

    rm.def("set_state", [](uint64_t seed, uint64_t counter) {
        g_seed    = seed;
        g_counter = counter;
    }, "seed"_a, "counter"_a, "Restore full RNG state from a prior get_state() call.");

    // rand(*shape) — uniform [0, 1).
    rm.def("rand", [](nb::args args) -> Array {
        auto shape = parse_shape(args);
        std::size_t n = total(shape);
        uint64_t base = g_counter; g_counter += n;
        uint64_t s = g_seed;
        Array out = hpxpy::empty_nd(shape, DType::F64);
        double* p = out.data_as<double>();
        hpxpy::on_hpx_thread([p, n, s, base] {
            hpx::experimental::for_loop(hpx::execution::par, std::size_t(0), n,
                [p, s, base](std::size_t i) { p[i] = uniform01(s, base + i); });
        });
        return out;
    }, "Uniform [0,1). rand(n) or rand(m,n) or rand((m,n)).");

    // randn(*shape) — standard normal via Box-Muller on consecutive index pairs.
    rm.def("randn", [](nb::args args) -> Array {
        auto shape = parse_shape(args);
        std::size_t n = total(shape);
        std::size_t pairs = (n + 1) / 2;
        uint64_t base = g_counter; g_counter += pairs * 2;
        uint64_t s = g_seed;
        Array out = hpxpy::empty_nd(shape, DType::F64);
        double* p = out.data_as<double>();
        hpxpy::on_hpx_thread([p, n, pairs, s, base] {
            hpx::experimental::for_loop(hpx::execution::par, std::size_t(0), pairs,
                [p, n, s, base](std::size_t i) {
                    double u1 = uniform01(s, base + 2*i);
                    double u2 = uniform01(s, base + 2*i + 1);
                    if (u1 < 1e-300) u1 = 1e-300;
                    double r = std::sqrt(-2.0 * std::log(u1));
                    double th = 6.283185307179586 * u2;
                    if (2*i     < n) p[2*i]     = r * std::cos(th);
                    if (2*i + 1 < n) p[2*i + 1] = r * std::sin(th);
                });
        });
        return out;
    }, "Standard normal. randn(n) or randn(m,n) or randn((m,n)).");

    // uniform(low, high, size) — uniform [low, high).
    rm.def("uniform", [](double low, double high, nb::object size_arg) -> Array {
        auto shape = parse_size(size_arg);
        std::size_t n = total(shape);
        uint64_t base = g_counter; g_counter += n;
        uint64_t s = g_seed;
        double scale = high - low;
        Array out = hpxpy::empty_nd(shape, DType::F64);
        double* p = out.data_as<double>();
        hpxpy::on_hpx_thread([p, n, s, base, low, scale] {
            hpx::experimental::for_loop(hpx::execution::par, std::size_t(0), n,
                [p, s, base, low, scale](std::size_t i) {
                    p[i] = low + scale * uniform01(s, base + i);
                });
        });
        return out;
    }, "low"_a, "high"_a, "size"_a, "Uniform [low, high) with given size.");

    // randint(low, high, size) — integer uniform in [low, high), stored as I64.
    rm.def("randint", [](int64_t low, int64_t high, nb::object size_arg) -> Array {
        if (high <= low)
            throw std::invalid_argument("randint: high must be > low");
        auto shape = parse_size(size_arg);
        std::size_t n = total(shape);
        uint64_t base = g_counter; g_counter += n;
        uint64_t s = g_seed;
        uint64_t range = static_cast<uint64_t>(high - low);
        Array out = hpxpy::empty_nd(shape, DType::I64);
        int64_t* p = out.data_as<int64_t>();
        hpxpy::on_hpx_thread([p, n, s, base, low, range] {
            hpx::experimental::for_loop(hpx::execution::par, std::size_t(0), n,
                [p, s, base, low, range](std::size_t i) {
                    uint64_t x = s + (base + i) * 0x9e3779b97f4a7c15ULL;
                    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
                    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
                    x ^= x >> 31;
                    p[i] = low + static_cast<int64_t>(x % range);
                });
        });
        return out;
    }, "low"_a, "high"_a, "size"_a,
       "Integer uniform in [low, high) with given size. Returns an I64 Array.");
}
