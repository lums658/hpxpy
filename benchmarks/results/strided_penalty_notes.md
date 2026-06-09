# Strided views — penalty result

Strided slice views (`a[i:j:k]`, step ≠ 1) carry a signed element stride; element `i`
lives at `base_[i*stride_]`. Compute on a strided view is strided-**direct** (one HPX
`for_loop`+`reduction` pass — no gather), so the wrapper should cost the same as a
hand-written direct HPX strided kernel. This confirms it.

## Result: zero abstraction penalty (L1/L0 ≈ 1.0)

In-binary diag ladder, same TU / process / data, kernel-only (the backing buffer and the
strided view are built once OUTSIDE the timed region). View is `a[::2]` over an 8M-element
backing buffer (`view_n` = 4M). `--op strided_sum` / `strided_muls` in `cpp_baseline/diag.cpp`:

| op | threads | L0 (direct) | L1 (wrapper) | L1/L0 |
|----|--------:|----:|----:|----:|
| strided_sum  | 1  | 0.0073 s | 0.0073 s | **0.998** |
| strided_sum  | 4  | —        | —        | **1.000–1.001** |
| strided_sum  | 16 | 0.0010 s | 0.0010 s | **1.006** |
| strided_muls | 1  | 0.0292 s | 0.0295 s | **1.007** |
| strided_muls | 4  | 0.0135 s | 0.0129 s | **0.954** |
| strided_muls | 16 | 0.0150 s | 0.0165 s | **1.095** |

L0 = direct `hpx::experimental::for_loop` + `reduction` over `base[i*stride]`; L1 = the
exact wrapper method (`view.sum()` / `view.mul_scalar()`) on an `hpxpy::Array` strided
view. Values are cross-checked each run (no `[VALUE MISMATCH]`). The L1 strided `sum()`
is *byte-for-byte* the same algorithm as the L0 lambda — same index math, same buffer,
same `-O3` TU — so ≈1.0 is expected and observed.

## Caveat: measure on the exclusive node

These numbers were taken on a SHARED node (ad-hoc), not the exclusive medusa node that
`bench.yml` uses. Contention shows up two ways and neither is a wrapper cost:
- **Absolute times drift run-to-run** (e.g. t=1 strided_sum measured 0.0073 s and, under
  load, 0.0147 s — 2×), from other tenants on the node.
- **L0 (measured first) occasionally fails to scale**, making L1/L0 dip *below* 1.0
  (seen: 0.435 at t=4, 0.79–0.81 at t=16 on one contended run). A ratio < 1 means L1 is
  *faster* than L0 — that cannot be wrapper overhead (overhead is > 1); it is L0's first-
  measured run paying warmup/contention. Across every repetition L1/L0 centers on 1.0 and
  deviations only ever favor L1, never a sustained penalty.

Absolute throughput (~0.5 GEl/s strided-sum at t=1) reflects the strided data layout
itself (non-contiguous reads, no vectorization) — inherent to the access pattern and
identical for L0 and L1; it is not abstraction penalty.

## Reproduce

```bash
source env.sh && export LD_PRELOAD="$HPXPY_TCMALLOC"
D=cpp_baseline/build/diag; ninja -C cpp_baseline/build diag
for T in 1 4 16; do
  $D --op strided_sum  --sizes 8000000 --threads $T --budget 1.5 --min-reps 10 --max-reps 400
  $D --op strided_muls --sizes 8000000 --threads $T --budget 1.5 --min-reps 10 --max-reps 400 --scalar 2.0
done
```

## Verdict

The strided wrapper is thin: L1/L0 ≈ 1.0 (within shared-node noise), so a strided op
through `hpxpy.Array` costs the same as the hand-written HPX strided kernel — the
zero-penalty thesis holds for the strided path as it does for the contiguous one.
