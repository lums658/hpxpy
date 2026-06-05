# C++ HPX baseline kernels

Hand-written, NUMA-correct C++ HPX implementations of each kernel — the **performance
ceiling** the Python bindings are measured against (abstraction penalty = HPXPy ÷ this).

Populated per operation as milestones land (M2+). Each baseline:
- uses parallel **first-touch** allocation (no `std::vector` sequential zero-fill),
- the same execution policy as the HPXPy op being compared,
- reports median-of-times throughput,
- controls threads via `--hpx:threads=N` (the cfg key does not limit the pool).

(The validated `cpp_benchmark_worker.cpp` from the prototype is the reference to port.)
