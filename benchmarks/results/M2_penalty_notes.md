# M2 reductions — abstraction-penalty diagnosis & result

**Goal (Phase 1):** prove HPXPy is a thin wrapper — `hpxpy ÷ C++-HPX ≈ 1.0` for the
same algorithm, same NUMA-aware substrate, same threads.

## Result

Zero abstraction penalty for `sum`/`min`/`max` over `Array` (n=1e8, exclusive
medusa node, both sides timed by the shared C++ harness `src/timing.hpp`):

| threads | sum | min | max |
|--------:|----:|----:|----:|
| 1  | 1.000 | 1.000 | 0.999 |
| 2  | 1.001 | 0.999 | 0.998 |
| 4  | 1.001 | 1.001 | 1.002 |
| 8  | 1.000 | 1.000 | 1.000 |
| 16 | 1.003 | 1.000 | 1.008 |
| 32 | 0.980 | 1.009 | 0.979 |
| 40 | 1.027 | 1.026 | 1.029 |

(penalty = hpxpy median ÷ cpp median; ≈1.0 = none. Raw CSVs: `m2_{sum,min,max}.csv`.)
Throughput scales 0.72 → 18.4 GEl/s (1 → 40 threads, n=1e8) — memory-bandwidth bound.

## How it was found (the value of this exercise)

The first measurement showed a **~2.0× penalty**, worst at 1 thread, shrinking with
threads. Rather than guess, we built a **measurement ladder** in `cpp_baseline/diag.cpp`:
one binary / one TU times **L0** (direct `hpx::reduce`) vs **L1** (`hpxpy::Array::sum`,
the exact wrapper) over the **same buffer**, so flags/process/runtime/data are constant.

- **Ladder verdict: `L1/L0 = 1.000`** at every thread count — the C++ wrapper adds
  *nothing*. The 2× was therefore a *cross-binary* artifact, not a wrapping penalty.
- The penalty's shape (worst single-threaded, fading as cores saturate the memory
  bus) is the signature of a **de-optimized kernel**, not NUMA.

## Root cause & fix

`nanobind` size-optimizes extensions with **`-Os` by default**. But the HPX
`reduce`/`compute` templates inline into our extension TU, and `-Os` suppressed the
vectorization/unrolling that the `-O3` C++ baseline got — so the *same source* ran
~2× slower in the `.so` than in the standalone binary.

Fix: `nanobind_add_module(_core NOMINSIZE ...)` drops `-Os` so the Release `-O3`
stands. Penalty → ~1.0. (The wrapper was always thin; only the kernel's compilation
differed.)

## Measurement methodology (why these numbers are trustworthy)

- **Timed in C++, never across the Python boundary** — removes interpreter/GIL/
  subprocess jitter. One shared harness `hpxpy::timing::measure` ("dobench") on HPX's
  own `hpx::chrono::high_resolution_timer`, used by the extension (`_core.bench`) AND
  both `cpp_baseline` binaries, so hpxpy and the baseline are timed by identical code.
- **Adaptive repeats:** run until a time budget elapses, bounded by min/max reps —
  many reps for cheap points, few for slow ones. Median via `std::nth_element` (O(n)).
- Switching from Python-side timing to this C++ harness is what removed the residual
  t=40 jitter seen earlier (then ~1.3×, now ≤1.03×).
- Same NUMA-aware substrate on both sides: `compute::vector<double, block_allocator>`
  with parallel first-touch; data built once, outside the timed region.

## Reproduce

```bash
source env.sh && export LD_PRELOAD="$HPXPY_TCMALLOC"
# in-binary wrapper-penalty ladder:
srun --partition=medusa --exclusive --cpus-per-task=40 \
  ./cpp_baseline/build/diag --op sum --sizes 1e8 --threads 40
# cross-process penalty:
srun --partition=medusa --exclusive --cpus-per-task=40 \
  python -m benchmarks.runner sweep --op sum --sizes 1e8 \
    --threads 1,2,4,8,16,32,40 --baseline ./cpp_baseline/build/bench --out m2_sum.csv
```
