# HPXPy Benchmark Revisit — Plan

Branch: `bench/benchmark-revisit` (off `main` @ b9335a0). Measures the **committed** hpxpy
surface only. Goal: a rigorous, reproducible evidence base for the three claims that the whole
project rests on — **zero abstraction penalty**, **raw performance**, and **scalability** —
plus an honest competitive picture. (Evidence for the NSF report / paper / positioning.)

## 0. What already exists (revisit, don't rebuild)
- `src/timing.hpp` — `hpxpy::timing::measure`: HPX `high_resolution_timer`, one untimed warmup,
  adaptive repeats to a time budget, median via `nth_element`, `keep<T>` to defeat DCE. Sound; keep.
- `cpp_baseline/diag.cpp` — the **in-binary L0/L1 ladder**: L0 = raw `hpx::reduce/transform/sort/...`,
  L1 = the `hpxpy::Array` wrapper method, same TU + same timer → penalty isolated from Python and
  from cross-binary artifacts. Currently covers: sum/min/max/dot, add/sub/mul/div, muls (scalar),
  sort, scan(cumsum), spmv/spmvk, spmm, strided_sum, strided_muls. **float64 only.**
- `cpp_baseline/bench.cpp` — hand-written C++-HPX kernels emitting JSON (cross-process baseline).
- `benchmarks/runner.py` — thread/size sweep (one subprocess per thread count, since HPX fixes
  threads at startup), aggregates, computes `abstraction_penalty = hpxpy/cpp` and `scaling`, CSV out.
- `.github/workflows/bench.yml` — `workflow_dispatch` + `run-bench` label; `srun --exclusive` on a
  medusa node; currently sweeps sum/min/max at 1e7,1e8 over 1..40 threads.
- `benchmarks/results/` — prior evidence: zero penalty (~1.00) shown for sum/min/max/dot,
  add/sub/mul/div, scalar broadcast, sort/cumsum, spmv/spmm, strided. **Gaps below.**

## 1. Dimensions to measure

### A. Abstraction penalty (thesis-critical)
hpxpy ÷ hand-written C++-HPX, over the op surface × dtype × size. Report **two numbers, never
conflated**:
- **Pure-kernel penalty** (L1/L0 from `diag`, C++-only) — target ≈ 1.00. This is the thin-wrapper claim.
- **End-to-end per-call overhead** — time a bare Python `a.sum()` on a small array, minus kernel time;
  report as **absolute microseconds**, not a ratio. Honest characterization of the small-array regime
  (where fixed Python↔C++ dispatch cost dominates — true of any binding).

### B. Raw performance (absolute, vs roofline)
Absolute throughput against the machine's limits, so "fast" is grounded:
- Memory-bound ops (sum/axpy/elementwise/strided): **Gelem/s and GB/s** vs measured **STREAM bandwidth**.
- Compute-bound (matmul): **GFLOP/s** vs node **peak FLOPs** (and vs BLAS as a reference ceiling).
- Identify which ops saturate the roofline vs leave performance on the table (e.g. the known
  elementwise double-write inefficiency).

### C. Scalability
- **Strong scaling**: speedup + parallel efficiency vs `HPX_NUM_THREADS` 1..ncores, for a memory-bound
  archetype (sum, expect bandwidth saturation) and a compute-bound one (matmul, expect ~linear).
- **Size scaling**: cache-resident → L3 → memory-bound sweep.
- **Weak scaling**: fixed work/thread.
- NUMA first-touch effects (the `block_allocator`); the sequential-dependency ops (cumsum) characterized honestly.

### D. Competitive (context, honestly framed)
hpxpy vs **NumPy** (single-thread baseline + MKL/OpenBLAS-threaded) vs **Dask** on representative
kernels (large reduction, elementwise expression, matmul, sort, axis-reduction). End-to-end wall time
incl. Dask `.compute()`. The story: same code, all your cores, no penalty — while disclosing where
naive matmul loses to BLAS and where Dask's scheduler overhead amortizes.

## 2. Op coverage (the committed surface)
Reductions sum/min/max/mean/prod/dot; axis-reductions (axis=, keepdims); elementwise + - * / (incl.
broadcast + mixed-dtype promotion); ufuncs (sqrt/exp/log/sin/cos/abs/clip/maximum/minimum/power/...);
matmul/`@`; sort/cumsum; strided/non-contiguous (transpose/slice); sparse spmv/spmm —
each across **float64 / float32 / int64** where applicable.

## 3. Methodology (the load-bearing rigor)
1. **L0/L1 in one TU** at identical flags — the only honest penalty number.
2. **Time the kernel, pre-allocated** (output reused across reps) as the canonical metric; also report
   full alloc+op as a UX metric. (The SpMV lesson: alloc first-touch can swamp a memory-bound kernel.)
3. **NUMA**: build inputs with the parallel first-touch path (`arange`), never numpy-placed data.
4. **Stats**: median + min, adaptive budget (≥0.5s, ≥5 reps; raise to ≥50 reps for sub-µs small arrays).
5. **Environment**: `srun --exclusive` on a dedicated node; record node, HPX config, compiler, thread pin.
6. **-O3 / NOMINSIZE invariant**: the extension's kernel TUs must compile `-O3` (not `-Os`); a CI check
   fails on penalty > 1.05 at large n so a flag regression can't slip in silently.
7. **Roofline**: measure STREAM bandwidth + peak FLOPs for the node to contextualize §B.

## 4. Results schema + reproducibility
One row per cell: `op, dtype, n, layout, threads, level{l0,l1,hpxpy,numpy,dask,cpp}, median_s, min_s,
gelem_s, gflop_s, penalty, speedup, efficiency`. Machine-readable CSV/JSON in `benchmarks/results/`
with committed golden baselines; a `report` step renders a Markdown summary; `check_regression.py`
gates penalty drift and scaling collapse.

## 5. Staging (see beads under the benchmark epic)
- **BR1 — Penalty coverage**: extend `diag.cpp`/`bench.cpp`/`runner.py` to the FULL op surface ×
  {f64,f32,i64} × size sweep; add matmul + axis-reduce + broadcast ladders; the two-number method.
- **BR2 — Raw performance + roofline**: absolute Gelem/s + GFLOP/s vs measured STREAM/peak; roofline plot.
- **BR3 — Scalability**: strong + weak scaling, parallel efficiency, NUMA, memory- vs compute-bound.
- **BR4 — Competitive**: NumPy(1T + MKL) + Dask harness; representative kernels; honest framing.
- **BR5 — Harness/CI/regression + methodology doc**: results schema, golden baselines, `bench.yml`
  coverage, regression gate, the pitfall register (NUMA, -O3/NOMINSIZE, shared-node, alloc variance,
  small-array honesty, fair-vs-MKL).

## Scope now vs later
Single-node CPU (penalty + raw + scaling + numpy/dask) is fully measurable now. Distributed
weak/strong scaling across localities slots into the same schema once Phase B lands; GPU vs
CuPy/cuPyNumeric once Phase C lands — `level`/`impl` columns already accommodate them.
