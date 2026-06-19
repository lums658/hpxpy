# BR1 follow-up — penalty fixes (axis-reduce + int64 scalar-mul)

Two wrapper inefficiencies surfaced by the BR1 ladder, now fixed in `src/array.hpp`.
Validated on medusa (SLURM job 154247): `check.sh` → **834 passed, 100% coverage, `== OK ==`**.
Re-benched f64/i64, n=1e7, 40 threads (medusa03).

| op | before (BR1) | after (L1/L0) | fix |
|---|---|---|---|
| `axis_reduce` axis=0 | ~2.4× | **0.955** | 2-D single-axis reduce now uses direct `i*st0 + j*st1` stride arithmetic instead of a per-element N-D div/mod unravel |
| `axis_reduce` axis=1 | ~2.4× | **0.945** | (same) |
| `muls` int64 | ~2.0× | **0.963** | int64 × float-scalar fuses the int64→f64 promotion into the multiply (one pass) instead of `astype(F64)` then `unary()` (two passes) |

L1/L0 ≈ 0.95 ⇒ the wrapper is now within noise of raw HPX for both ops. The general
N-D reduction path and float-scalar promotion semantics are unchanged; only the
common contiguous-2-D and integral/float-scalar cases get the fast path.

Beads: hpxpy-hjj, hpxpy-awv. (These array.hpp fixes should also be promoted to `main`
via a follow-up PR once the main checkout is deconflicted.)
