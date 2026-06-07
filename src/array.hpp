// hpxpy::Array — the core data type, as a binding-agnostic C++/HPX header.
//
// A 1-D float64 array backed by an HPX compute::vector with a NUMA-aware host
// block_allocator (parallel first-touch). A THIN wrapper: HPX owns the storage,
// operations ARE HPX algorithms run in place over the contiguous buffer — no
// copies, no reassembly, no between-layers buffers. Nothing here depends on
// nanobind/Python, so the same code is used by the bindings (_core.cpp) AND by the
// C++ abstraction-penalty diagnostic (cpp_baseline/diag.cpp): the penalty is a
// C++ question (wrapper-C++ vs direct-C++), measured with this exact wrapper.
//
// SPDX-License-Identifier: MIT
#pragma once

#include <hpx/algorithm.hpp>
#include <hpx/execution.hpp>
#include <hpx/modules/compute_local.hpp>
#include <hpx/numeric.hpp>
#include <hpx/runtime_local/run_as_hpx_thread.hpp>
#include <hpx/threading_base/threading_base_fwd.hpp>

#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

namespace hpxpy {

using numa_alloc = hpx::compute::host::block_allocator<double>;
using dvec = hpx::compute::vector<double, numa_alloc>;

// Run f on an HPX thread, where the allocator's parallel first-touch and the
// parallel algorithms must execute. If we are already on an HPX thread (e.g. the
// diagnostic's hpx_main) run it directly — run_as_hpx_thread asserts it is NOT
// called from an HPX thread. If we are on a foreign thread (e.g. the Python main
// thread) hop onto the pool. This keeps the wrapper identical in both contexts.
template <typename F>
inline void on_hpx_thread(F&& f)
{
    if (hpx::threads::get_self_ptr() != nullptr)
        std::forward<F>(f)();
    else
        hpx::run_as_hpx_thread(std::forward<F>(f));
}

class Array
{
public:
    Array() = default;
    Array(std::size_t n, double fill) : size_(n)
    {
        on_hpx_thread([&] { data_ = std::make_shared<dvec>(n, fill); });
    }

    std::size_t size() const { return size_; }
    std::size_t ndim() const { return 1; }

    // Raw contiguous buffer (nullptr if empty). Used by the diagnostic to run the
    // direct (L0) reduce over the exact same memory the wrapper (L1) reduces.
    const double* data() const { return data_ ? data_->data() : nullptr; }

    // Reductions — thin wrappers over HPX algorithms, run in place on the buffer.
    // The result is a scalar; the penalty of these vs a direct C++ call must be ~1.
    double sum() const
    {
        if (size_ == 0)
            return 0.0;
        const double* p = data_->data();
        return hpx::reduce(hpx::execution::par, p, p + size_, 0.0);
    }
    double min() const
    {
        if (size_ == 0)
            throw std::invalid_argument("min() of an empty Array");
        const double* p = data_->data();
        return hpx::reduce(hpx::execution::par, p, p + size_,
            std::numeric_limits<double>::infinity(),
            [](double x, double y) { return x < y ? x : y; });
    }
    double max() const
    {
        if (size_ == 0)
            throw std::invalid_argument("max() of an empty Array");
        const double* p = data_->data();
        return hpx::reduce(hpx::execution::par, p, p + size_,
            -std::numeric_limits<double>::infinity(),
            [](double x, double y) { return x > y ? x : y; });
    }

    // 0, 1, 2, ..., n-1. The block_allocator first-touches at construction; the
    // parallel for_loop writes the ramp on the same HPX workers (stays NUMA-local).
    static Array iota(std::size_t n)
    {
        Array a;
        a.size_ = n;
        on_hpx_thread([&] {
            a.data_ = std::make_shared<dvec>(n);
            double* p = a.data_->data();
            hpx::experimental::for_loop(hpx::execution::par, std::size_t(0), n,
                [p](std::size_t i) { p[i] = static_cast<double>(i); });
        });
        return a;
    }

private:
    std::shared_ptr<dvec> data_;
    std::size_t size_ = 0;
};

inline Array zeros(std::size_t n) { return Array(n, 0.0); }
inline Array full(std::size_t n, double value) { return Array(n, value); }
inline Array arange(std::size_t n) { return Array::iota(n); }

}    // namespace hpxpy
