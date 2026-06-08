# M5a SpMV — penalty result + diagnosis

CSR SpMV (`y = A @ x`) over a 1-D Laplacian (≈3 nnz/row), n=1e7, exclusive medusa node.

## Result: the wrapper is thin (zero abstraction penalty)

In-binary diag ladder, L0 = direct C++ HPX row-parallel kernel, L1 = `CsrMatrix::spmv`,
same TU/process/data:

| threads | full `spmv` L1/L0 (alloc+kernel) | kernel-only `spmvk` L1/L0 |
|--------:|----:|----:|
| 1  | 1.000 | 1.000 |
| 4  | 0.998 | 0.993 |
| 8  | 0.996 | 1.005 |
| 16 | 0.998 | 1.000 |
| 40 | 0.973 | 1.003 |

Cross-process (hpxpy via Python ÷ C++ baseline, KERNEL timed — `spmv_into` into a
pre-allocated `y`): ≈1.0 for ≥2 threads (t=2..40: 0.98–1.02). The **t=1 point shows
~1.39** — see below; it is not a wrapper/kernel cost.

## Diagnosis (two false leads ruled out, one real caveat)

The first measurements showed a **1.25–1.4× spike at t=4–8** and a **crash at t=32/40**.

1. **Mid-thread spike = result-allocation variance, not the wrapper.** SpMV here is
   tiny and memory-bound; the result alloc `Array y(n,0.0)` (parallel first-touch of an
   80 MB buffer) measured ~0.005–0.019 s — *comparable to the ~0.01–0.02 s kernel*.
   With each rep re-allocating, the alloc's per-run NUMA/page-fault variance perturbed
   under-sampled medians. Isolating the kernel (`spmvk`: pre-allocated `y`, reused) and
   raising the sample budget made L1/L0 ≈ 1.0 everywhere. The kernel bodies are
   byte-identical in one TU, so a true wrapper cost was implausible — confirmed.
2. **t=32/40 crash = guard-page stacks**, not our code:
   `mmap() failed to allocate thread stack … max_map_count`. Fixed by passing
   `--hpx:ini=hpx.stacks.use_guard_pages=0` (added to diag, bench, and the managed
   runtime). Runs clean to 40 threads.
3. **t=1 cross-process ~1.39 = foreign-thread launch cost, not the wrapper.** In-binary
   `spmvk` at t=1 is 1.000 (kernel runs on an HPX worker). Cross-process, hpxpy calls
   `hpx::for_loop(par)` from the foreign (GIL-released) Python thread; at a single
   thread there is no parallelism to amortize the foreign→HPX launch, so a fixed
   overhead shows (~0.03 s). It is gone by 2 threads. (Reduce-based ops did not show
   this at t=1 — `for_loop` has a higher foreign-thread launch cost than `reduce`.) A
   minor, characterized Python-boundary effect at single-thread only — tracked as a
   possible future micro-opt (run the kernel via `run_as_hpx_thread`), not a defect.

## Takeaways for the harness

- For alloc-dominated memory-bound kernels, time the **kernel** (pre-allocated output),
  not the full allocate+compute op, or the allocation's variance masquerades as penalty.
- Always sample enough reps (budget ≥ ~0.8 s here) before trusting a per-point ratio.
- `spmvk` (kernel-only) is the canonical SpMV penalty measure; `spmv` (full) is the
  user-facing op.
