# C++ HPX baseline kernels

Hand-written, NUMA-correct C++ HPX implementations of each kernel — the **performance
ceiling** the Python bindings are measured against (abstraction penalty = HPXPy ÷ this).

Populated per operation as milestones land (M2+). Each baseline:
- uses parallel **first-touch** allocation (no `std::vector` sequential zero-fill) —
  the same NUMA-aware substrate (`compute::vector` + `block_allocator`) as hpxpy,
- the same execution policy as the HPXPy op being compared,
- reports median-of-times throughput,
- controls threads via `--hpx:threads=N` (the cfg key does not limit the pool).

(The validated `cpp_benchmark_worker.cpp` from the prototype is the reference to port.)

## Point contract (so the runner can compare apples to apples)

The benchmark binary must obey the same CLI/output contract as
`benchmarks.runner point`, so `runner sweep --baseline <binary>` can drive it and
compute the abstraction penalty (hpxpy median ÷ this binary's median).

- **Input flags:** `--op <name> --sizes <csv> --threads <N> --repeats <R> --warmup <W>`
  (e.g. `--op sum --sizes 1e6,1e7 --threads 8 --repeats 7 --warmup 1`). Set the HPX
  worker count from `--threads` (pass `--hpx:threads=N`; `0` = all cores).
- **Output:** one JSON object per size on stdout, one per line:
  ```json
  {"op": "sum", "n": 10000000, "threads": 8, "impl": "cpp", "median_s": 0.0123}
  ```
  `threads` is the ACTUAL worker count after the runtime clamps the request;
  `impl` must be `"cpp"`. Lines not starting with `{` are ignored by the runner.
