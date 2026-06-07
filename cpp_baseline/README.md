# C++ HPX baseline kernels

Hand-written, NUMA-correct C++ HPX implementations of each kernel — the **performance
ceiling** the Python bindings are measured against (abstraction penalty = HPXPy ÷ this).

Populated per operation as milestones land (M2+). Each baseline:
- uses parallel **first-touch** allocation (no `std::vector` sequential zero-fill) —
  the same NUMA-aware substrate (`compute::vector` + `block_allocator`) as hpxpy,
- the same execution policy as the HPXPy op being compared,
- is timed by the shared `src/timing.hpp` harness (HPX `high_resolution_timer`,
  adaptive repeats, median) — identical to how the extension times itself,
- controls threads via `--hpx:threads=N` (the cfg key does not limit the pool).

There are two binaries:
- **`bench`** — the cross-process baseline (direct HPX reduce) the runner compares
  hpxpy against (the abstraction-penalty denominator).
- **`diag`** — the in-binary ladder: it measures L0 (direct HPX reduce) vs L1 (the
  exact `hpxpy::Array` wrapper from `src/array.hpp`) over the SAME buffer, in one
  TU, so `L1/L0` is the true C++ wrapping penalty (isolated from Python and flags).

## Point contract (so the runner can compare apples to apples)

The benchmark binary must obey the same CLI/output contract as
`benchmarks.runner point`, so `runner sweep --baseline <binary>` can drive it and
compute the abstraction penalty (hpxpy median ÷ this binary's median).

- **Input flags:** `--op <name> --sizes <csv> --threads <N> --budget <s> --min-reps <r> --max-reps <r>`
  (e.g. `--op sum --sizes 1e6,1e7 --threads 8 --budget 0.5`). Set the HPX worker
  count from `--threads` (pass `--hpx:threads=N`; `0` = all cores). Timing is
  adaptive: run reps until `budget` timed seconds elapse, bounded by min/max reps.
- **Output:** one JSON object per size on stdout, one per line:
  ```json
  {"op": "sum", "n": 10000000, "threads": 8, "impl": "cpp", "median_s": 0.0123, "reps": 41}
  ```
  `threads` is the ACTUAL worker count after the runtime clamps the request;
  `impl` must be `"cpp"`. Lines not starting with `{` are ignored by the runner.
  `bench.cpp` also emits a `"value"` key (the computed result) for analytic
  cross-checking; the runner ignores extra keys.

`bench.cpp` implements `sum`/`min`/`max`/`dot` over the same NUMA-aware
`compute::vector<double, block_allocator<double>>` substrate as hpxpy's Array.

## Build & run

```bash
source ../env.sh                # gcc 15 / HPX / Boost on Rostam
unset LD_PRELOAD                # do NOT preload tcmalloc during the build
cmake -S . -B build -G Ninja \
  -DCMAKE_CXX_COMPILER=/opt/apps/gcc/15.1.0/bin/g++ \
  -DCMAKE_PREFIX_PATH="$HPX_ROOT" -DHPX_DIR="$HPX_DIR"
cmake --build build

# direct (one JSON line per size):
LD_PRELOAD="$HPXPY_TCMALLOC" ./build/bench --op sum --sizes 1e7 --threads 8

# in-binary wrapping-penalty ladder (L0 direct vs L1 the wrapper):
LD_PRELOAD="$HPXPY_TCMALLOC" ./build/diag --op sum --sizes 1e7,1e8 --threads 8

# as the abstraction-penalty denominator in a full sweep:
LD_PRELOAD="$HPXPY_TCMALLOC" python -m benchmarks.runner sweep \
  --op sum --sizes 1e6,1e7,1e8 --threads 1,2,4,8,16,32,40 \
  --baseline ./cpp_baseline/build/bench --out results.csv
```
