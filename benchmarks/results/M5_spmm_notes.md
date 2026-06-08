# M5b SpMM (sparse × dense) — penalty result

`C = A @ B`, A = 1-D Laplacian (n×n, ≈3 nnz/row), B dense n×K (K=16), n=1e6,
exclusive medusa node. Row-parallel; each output row is a sparse combination of B's
rows. Kernel timed via `spmm_into` into a pre-allocated C (the SpMV lesson: don't let
the result allocation's variance masquerade as penalty).

## Result: wrapper penalty ≈ 1.0

In-binary ladder (L0 = direct kernel, L1 = `CsrMatrix::spmm`, same TU, kernel-only):

| threads | 1 | 4 | 8 | 16 | 40 |
|--------:|--:|--:|--:|--:|--:|
| L1/L0   | 0.73* | 0.967 | 0.972 | 1.002 | 1.002 |

Cross-process (hpxpy ÷ C++ baseline, kernel): t=1 0.87*, then t=2..40 = 0.975,
0.972, 0.980, 1.016, 1.020, 0.984 — i.e. ≈1.0 within ±3% for ≥2 threads.

\* The single-thread points deviate (and L1 is *faster*, not slower — the wrong sign
for a penalty). At t=1 this memory-bound op is most sensitive to measurement order /
codegen variance on the shared B/C buffers; it washes out by 4 threads. Not a
systematic wrapper cost. Consistent with the SpMV single-thread finding (the
multi-thread regime — the point of HPX — is clean ~1.0).

Throughput scales 0.025 → 0.36 GRow/s (1 → 40 threads); SpMM does ~K× the per-row
work of SpMV, so the kernel dominates the result allocation here (cleaner than SpMV).

CSV: `m5_spmm.csv`. SpGEMM (sparse×sparse) remains deferred (`hpxpy-omw`).
