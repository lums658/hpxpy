# M3 sort + prefix scan — penalty result

NumPy semantics: `a.sort()` sorts IN PLACE (no allocation), `hpx.sort(a)` returns a
sorted copy (`a.copy()` then `.sort()`), `a.cumsum()` returns a new Array (inclusive
`hpx::inclusive_scan`). `a.is_sorted()` (wraps `hpx::is_sorted`) lets tests verify
sortedness without element access (no slicing yet — that's a deferred milestone).

Penalty ≈ 1.0, in-binary diag ladder (L0 = direct HPX algorithm, L1 = the wrapper,
same buffer/flags/process):

| threads | sort L1/L0 (copy+sort, n=1e7) | scan L1/L0 (n=1e8) |
|--------:|----:|----:|
| 1  | 0.999 | 1.000 |
| 8  | 0.996 | 1.007 |
| 40 | 0.971 | 0.961 |

The sort ladder measures the **copy+sort** path (`hpx.sort`) so each repeat re-copies
the input and sorts fresh data (in-place sort would be idempotent after the first
rep — not a valid repeated benchmark). copy/sort/scan call their `hpx::` algorithm
directly (only the allocation is wrapped in `on_hpx_thread`), so the op-inlining fix
from the scalar work applies here too. Throughput: sort ~0.25→0.68 GEl/s (compute
bound, O(n log n)); scan ~0.25→1.2 GEl/s.
