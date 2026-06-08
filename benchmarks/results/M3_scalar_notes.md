# M3 scalar broadcast (Array ⊙ scalar) — penalty result + a wrapper fix

Scalar ops (`a*s`, `a+s`, `s-a`, `s/a`, …) are one unary `hpx::transform` pass →
a new Array. Penalty ≈ 1.0, proven by the in-binary diag ladder (L0 = direct
`hpx::transform`, L1 = the `Array` wrapper, same buffer/flags/process, n=1e8):

| threads | muls L1/L0 (after fix) | add L1/L0 |
|--------:|----:|----:|
| 1  | 1.008 | 0.998 |
| 8  | 1.006 | — |
| 40 | 1.032 | 1.076 |

## A real wrapper inefficiency the ladder caught (and the fix)

First measurement showed **L1/L0 ≈ 1.9 at 1 thread** for the scalar (unary) op —
fading to ~1.0 by 40 threads. Diagnosis (in one TU, so not flags/process):

- It was NOT constant-folding (made the ladder's scalar runtime via `--scalar`).
- It was NOT the allocation path (made L0 use the same `make_shared<dvec>`).
- Root cause: the wrapper ran `hpx::transform` **inside** the `on_hpx_thread([&]{…})`
  lambda. That extra layer stopped the optimizer from inlining the element op into
  the transform's inner loop → an indirect call per element. For a compute-light
  unary op at low thread count that ~doubled the time; for memory-bound binary ops
  it was hidden (so element-wise had looked fine at ~1.0).

**Fix** (`src/array.hpp`): wrap only the *allocation* (block_allocator first-touch)
in `on_hpx_thread`; run the `transform` **directly** — the same way the reductions
already call `hpx::reduce` directly. Penalty → ~1.0 across all thread counts for
both unary (scalar) and binary (element-wise) ops; correctness unchanged (76 tests).

Lesson: keep the parallel algorithm call at the top level of the wrapper method, not
nested in a helper lambda, so `op` inlines. Compute-light ops at low concurrency are
the sensitive case — measure them, not just the memory-bound ones.
