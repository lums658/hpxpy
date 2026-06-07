// hpxpy::timing — the one C++ measurement convention ("dobench"), shared by the
// extension's in-process benchmark entry point AND the cpp_baseline binaries, so
// hpxpy and the C++ baseline are timed by identical code (the abstraction-penalty
// comparison is then apples-to-apples).
//
// Timing uses HPX's own hpx::chrono::high_resolution_timer (hardware timestamp) —
// the highest-precision timer available, and the same one HPX's internals use — so
// every TU that links HPX times identically. Timing is in C++, never across the
// Python boundary, which removes interpreter/GIL/subprocess jitter. Repeats are
// ADAPTIVE: one untimed warmup, then time reps until both (reps >= min_reps) and
// (accumulated timed seconds >= budget_s), capped at max_reps. Cheap kernels get
// many reps, expensive ones few; total timed wall clock is ~budget_s (+ one rep).
//
// (HPX also ships hpx::util::perftests_report, but it runs a FIXED step count and
// only PRINTS a report; we need an adaptive budget AND the median returned as a
// value to compute the cross-process penalty, hence this small helper on top of
// the HPX timer.)
//
// SPDX-License-Identifier: MIT
#pragma once

#include <hpx/modules/timing.hpp>

#include <algorithm>
#include <cstddef>
#include <vector>

namespace hpxpy::timing {

struct result
{
    double median_s;
    int reps;
};

// Keep a value observable so the optimizer cannot elide the call that produced it
// (the "DoNotOptimize" trick). Works for any type — double from a reduction, or an
// Array from a transform — via a memory clobber on the object's storage.
template <typename T>
inline void keep(T const& x)
{
    asm volatile("" : : "m"(x) : "memory");
}

// Time any callable f(); its result type is irrelevant (kept observable outside the
// timed region). Used by reductions (return double) and transforms (return Array).
template <typename F>
inline result measure(F&& f, double budget_s, int min_reps, int max_reps)
{
    {
        auto warm = f();    // warmup (caches / first-touch), untimed
        keep(warm);
    }

    std::vector<double> ts;
    double total = 0.0;
    hpx::chrono::high_resolution_timer timer;
    while (true)
    {
        timer.restart();
        auto r = f();
        double dt = timer.elapsed();
        keep(r);    // outside timing — defeats dead-code elimination

        ts.push_back(dt);
        total += dt;

        int const reps = static_cast<int>(ts.size());
        if (reps >= max_reps)
            break;
        if (reps >= min_reps && total >= budget_s)
            break;
    }

    // Median via selection (O(n)), not a full sort. The STL has no median; the
    // tool for "the k-th element" is nth_element. For even n, the lower-middle is
    // the max of the elements left of the upper-middle after partitioning.
    std::size_t const n = ts.size();
    auto upper = ts.begin() + n / 2;
    std::nth_element(ts.begin(), upper, ts.end());
    double med = *upper;
    if (n % 2 == 0)
        med = 0.5 * (*std::max_element(ts.begin(), upper) + *upper);
    return result{med, static_cast<int>(n)};
}

}    // namespace hpxpy::timing
