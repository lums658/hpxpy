# M3 element-wise (add/sub/mul/div) — abstraction-penalty result

Each op is one `hpx::transform` pass producing a NEW Array. Penalty ≈ 1.0 (n=1e8,
exclusive medusa node, both sides timed by the shared C++ harness):

| threads | add | mul | (in-binary ladder L1/L0, add) |
|--------:|----:|----:|----:|
| 1  | 1.002 | 1.000 | 0.996 |
| 8  | 0.994 | 0.993 | 0.999 |
| 16 | 0.979 | 0.979 | — |
| 40 | 0.941 | 0.949 | 1.009 |

(<1.0 = hpxpy marginally faster = noise. Raw CSVs: `m3_{add,mul}.csv`.) The wrapper
is thin (ladder ≈1.0); the C++ baseline allocates a fresh result each call too, so
the comparison is fair.

**Known inefficiency (same on both sides, not a penalty):** the result
`compute::vector(n)` value-initializes (zero-fills 800 MB) and the transform then
overwrites it — a double write. So element-wise throughput (~0.27→2.5 GEl/s, add)
and scaling are well below the reductions (sum 0.72→18 GEl/s). A future optimization
is an UNINITIALIZED result allocation (first-touch fused into the transform write);
it would lift absolute throughput for both hpxpy and the baseline without changing
the ~1.0 penalty. Tracked for later, not an M3 blocker.
