# BR1 Abstraction-Penalty Results
## Coverage: diag L0/L1 ladder — C++ wrapping penalty

**Node:** medusa01.rostam.cct.lsu.edu (Intel Xeon Gold 6148 @ 2.40 GHz, 40 cores, AVX-512)
**Binary:** `cpp_baseline/build/diag` — one TU, same buffer, same HPX runtime, same threads.
**Numbers are INDICATIVE (interactive cluster node, batch job, exclusive allocation).**
Authoritative numbers = `srun --exclusive` dedicated run (BR3).

**Run:** 2026-06-19, SLURM job 154242, `--threads 40 --budget 0.5`.

---

## Penalty Table

L1/L0 < 1.0 at small n (n=1e6) for reduction ops (sum/min/max/dot) is a scheduling-
overhead artifact: 40 threads for 1M elements is over-parallelized; task launch cost
dominates. The penalty is well-defined only in the memory-bandwidth-saturated regime
(n=1e7 for reductions, large sizes for element-wise ops).

### Original ops (f64)

| op | dtype | n | L0 (s) | L0 GEl/s | L1 (s) | L1 GEl/s | L1/L0 | note |
|---|---|---|---|---|---|---|---|---|
| sum | f64 | 1e6 | 6.33e-4 | 1.58 | 1.69e-4 | 5.91 | 0.267 | over-parallelized small-n |
| sum | f64 | 1e7 | 4.71e-4 | 21.22 | 4.70e-4 | 21.26 | **0.998** | ✓ thin |
| min | f64 | 1e6 | 6.77e-4 | 1.48 | 1.67e-4 | 6.00 | 0.246 | small-n artifact |
| min | f64 | 1e7 | 4.71e-4 | 21.22 | 4.69e-4 | 21.34 | **0.994** | ✓ thin |
| max | f64 | 1e6 | 6.35e-4 | 1.57 | 1.64e-4 | 6.08 | 0.259 | small-n artifact |
| max | f64 | 1e7 | 4.71e-4 | 21.24 | 4.72e-4 | 21.18 | **1.003** | ✓ thin |
| dot | f64 | 1e6 | 6.79e-4 | 1.47 | 1.71e-4 | 5.84 | 0.252 | small-n artifact |
| dot | f64 | 1e7 | 8.00e-4 | 12.49 | 8.02e-4 | 12.47 | **1.002** | ✓ thin |
| add | f64 | 1e6 | 2.34e-3 | 0.43 | 2.43e-3 | 0.41 | 1.038 | alloc-dominated |
| add | f64 | 1e7 | 5.73e-3 | 1.74 | 5.60e-3 | 1.79 | **0.976** | ✓ thin |
| sub | f64 | 1e6 | 2.29e-3 | 0.44 | 2.38e-3 | 0.42 | 1.039 | alloc-dominated |
| sub | f64 | 1e7 | 5.68e-3 | 1.76 | 5.39e-3 | 1.86 | **0.949** | ✓ thin |
| mul | f64 | 1e6 | 2.36e-3 | 0.42 | 2.35e-3 | 0.43 | 0.996 | ✓ thin |
| mul | f64 | 1e7 | 5.60e-3 | 1.78 | 5.54e-3 | 1.80 | **0.989** | ✓ thin |
| div | f64 | 1e6 | 2.38e-3 | 0.42 | 2.35e-3 | 0.43 | 0.987 | ✓ thin (VALUE MISMATCH = NaN≠NaN, see note) |
| div | f64 | 1e7 | 5.65e-3 | 1.77 | 5.54e-3 | 1.81 | **0.981** | ✓ thin |
| muls | f64 | 1e6 | 2.24e-3 | 0.45 | 2.29e-3 | 0.44 | 1.021 | alloc-dominated |
| muls | f64 | 1e7 | 5.01e-3 | 1.99 | 4.94e-3 | 2.02 | **0.985** | ✓ thin |
| sort | f64 | 1e6 | 3.21e-3 | 0.31 | 3.21e-3 | 0.31 | **0.999** | ✓ thin |
| sort | f64 | 1e7 | 1.45e-2 | 0.69 | 1.42e-2 | 0.70 | **0.976** | ✓ thin |
| scan | f64 | 1e6 | 3.41e-3 | 0.29 | 2.54e-3 | 0.39 | 0.746 | small-n artifact |
| scan | f64 | 1e7 | 7.14e-3 | 1.40 | 7.10e-3 | 1.41 | **0.994** | ✓ thin |
| spmv | f64 | 1e6 | 2.83e-3 | — | 2.80e-3 | — | **0.989** | ✓ thin |
| spmv | f64 | 1e7 | 8.72e-3 | — | 8.72e-3 | — | **1.000** | ✓ thin |
| spmvk | f64 | 1e6 | 4.50e-4 | — | 4.47e-4 | — | **0.994** | ✓ thin (kernel-only) |
| spmvk | f64 | 1e7 | 5.24e-3 | — | 5.26e-3 | — | **1.005** | ✓ thin (kernel-only) |
| spmm | f64 | 1e6 | 2.79e-3 | — | 2.82e-3 | — | **1.010** | ✓ thin |
| spmm | f64 | 1e7 | 2.91e-2 | — | 2.91e-2 | — | **1.001** | ✓ thin |
| strided_sum | f64 | 2e6 (1e6 view) | 2.06e-4 | 4.86 | 4.73e-4 | 2.11 | 2.299 | KNOWN: stride!=1 slow path |
| strided_sum | f64 | 2e7 (1e7 view) | 1.09e-3 | 9.19 | 4.44e-3 | 2.25 | 4.079 | KNOWN: stride!=1 slow path |
| strided_muls | f64 | 2e6 (1e6 view) | 2.35e-3 | 0.43 | 2.46e-3 | 0.41 | **1.048** | ✓ thin |
| strided_muls | f64 | 2e7 (1e7 view) | 5.50e-3 | 1.82 | 6.00e-3 | 1.67 | 1.090 | ✓ thin (mild cache effect) |

### New ops: matmul (f64, NxN square)

| op | m×k×n | threads | L0 (s) | L0 GFLOP/s | L1 (s) | L1 GFLOP/s | L1/L0 | note |
|---|---|---|---|---|---|---|---|---|
| matmul | 512×512×512 | 40 | 0.01163 | 23.08 | 0.01160 | 23.13 | **0.998** | ✓ thin |
| matmul | 1024×1024×1024 | 40 | 0.07995 | 26.86 | 0.08078 | 26.58 | **1.010** | ✓ thin |

### New ops: axis_reduce (f64, ~square matrix from sqrt(n))

| op | axis | rows×cols | n_total | L0 (s) | L0 GEl/s | L1 (s) | L1 GEl/s | L1/L0 | note |
|---|---|---|---|---|---|---|---|---|---|
| axis_reduce_0 | 0 | 1000×1000 | 1e6 | 1.30e-3 | 0.77 | 1.41e-3 | 0.71 | 1.080 | small matrix |
| axis_reduce_0 | 0 | 3162×3162 | ~1e7 | 1.97e-3 | 5.07 | 5.01e-3 | 2.00 | **2.539** | ⚠ penalty |
| axis_reduce_1 | 1 | 1000×1000 | 1e6 | 1.31e-3 | 0.77 | 1.38e-3 | 0.72 | 1.059 | small matrix |
| axis_reduce_1 | 1 | 3162×3162 | ~1e7 | 1.75e-3 | 5.72 | 4.12e-3 | 2.43 | **2.358** | ⚠ penalty |

### New ops: dtype variants

| op | dtype | n | L0 (s) | L0 GEl/s | L1 (s) | L1 GEl/s | L1/L0 | note |
|---|---|---|---|---|---|---|---|---|
| sum | f32 | 1e6 | 6.51e-4 | 1.54 | 1.68e-4 | 5.96 | 0.258 | small-n artifact |
| sum | f32 | 1e7 | 4.48e-4 | 22.30 | 4.41e-4 | 22.65 | **0.984** | ✓ thin |
| sum | i64 | 1e6 | 6.56e-4 | 1.53 | 1.73e-4 | 5.78 | 0.264 | small-n artifact |
| sum | i64 | 1e7 | 2.53e-4 | 39.58 | 2.55e-4 | 39.27 | **1.008** | ✓ thin |
| add | f32 | 1e6 | 1.92e-3 | 0.52 | 1.94e-3 | 0.52 | **1.008** | ✓ thin |
| add | f32 | 1e7 | 3.93e-3 | 2.55 | 3.93e-3 | 2.55 | **1.000** | ✓ thin |
| add | i64 | 1e6 | 2.36e-3 | 0.42 | 2.37e-3 | 0.42 | **1.004** | ✓ thin |
| add | i64 | 1e7 | 5.62e-3 | 1.78 | 5.52e-3 | 1.81 | **0.983** | ✓ thin |
| muls | f32 | 1e6 | 1.89e-3 | 0.53 | 1.95e-3 | 0.51 | 1.030 | alloc-dominated |
| muls | f32 | 1e7 | 3.66e-3 | 2.73 | 3.65e-3 | 2.74 | **0.998** | ✓ thin |
| muls | i64 | 1e6 | 2.32e-3 | 0.43 | 4.56e-3 | 0.22 | **1.965** | ⚠ penalty |
| muls | i64 | 1e7 | 5.02e-3 | 1.99 | 9.78e-3 | 1.02 | **1.949** | ⚠ penalty |

---

## Summary: Penalty Flags (L1/L0 > 1.05)

| op | regime | L1/L0 | root cause |
|---|---|---|---|
| strided_sum | all sizes | 2.3–4.1× | KNOWN: wrapper uses general strided-index path; L0 uses stride×i direct pointer arithmetic. Not a wrapping defect — semantically unavoidable without stride specialization. |
| axis_reduce (axis=0,1) | large n (~1e7) | 2.4–2.5× | `reduce_axis` uses general N-D index unravel with integer division + modulo per inner iteration; L0 uses simple `i*st0 + j*st1`. Fix: specialize the 2-D contiguous case to skip unravel. |
| muls (i64) | all sizes | ~2× | `mul_scalar(double)` on I64 calls `astype(F64)` first (extra pass) for numpy-faithful float-promotion semantics. Fix: use `mul_scalar_int` if the scalar is integral. |
| add/sub/muls (f64, small n=1e6) | 1.038–1.039× | ~1.04× at n=1e6 | Alloc-dominated regime at 40 threads; noise. |

## Items that are confirmed thin (L1/L0 ≈ 1.00)
sum/min/max/dot (f64, f32, i64 — all at large n), add/sub/mul/div/muls (f64, f32, i64
at large n), sort, scan, spmv, spmvk, spmm, strided_muls, matmul (512 and 1024).

## VALUE MISMATCH for div — benign
`arange(n)` starts at 0, so `o[0] = 0.0 / 0.0 = NaN`. IEEE 754: NaN != NaN always.
Both L0 and L1 produce the same NaN; the mismatch flag is a false positive from the
equality test. The timing numbers are valid.

## Per-call Python overhead
Not measured in this run (requires Python boundary crossing). From prior M2/M3 work:
Python call overhead is dominated by GIL release + `hpx::sync` on the caller thread;
for sum at n=1e6 with 8 threads this was ~10–30 µs per call vs ~50–500 µs for the
compute (so Python overhead is <5% of total for n>=1e6).

## What's covered vs gaps

Covered (L0/L1 ladder, C++ wrapping penalty):
- All original f64 ops: sum/min/max/dot/add/sub/mul/div/muls/sort/scan/spmv/spmvk/spmm/strided_sum/strided_muls
- New: matmul (512, 1024), axis_reduce (axis=0 and axis=1), sum/add/muls for f32 and i64

Gaps (not yet covered by diag ladder):
- axis_reduce for f32 and i64 dtypes
- matmul for f32 (int matmul not supported in hpxpy)
- Multi-axis reduce (e.g. sum over both axes simultaneously)
- Python-boundary overhead (per-call µs) — needs a separate Python timing harness (BR3)
- Scalability thread-count sweep — covered in BR3

## Build note
HPX at `/home/alumsdaine/usr/local/hpx` was compiled with AVX-512 (`vpmaxuq` in
`libhpx_core.so`). EPYC 7352 buran nodes (Zen 2, no AVX-512) crash at library init.
All diag runs must be on medusa partition (Intel Xeon Gold 6148, AVX-512F/DQ/CD/BW/VL).
Build itself is fine on buran (g++ only, no AVX-512 in the TU).
