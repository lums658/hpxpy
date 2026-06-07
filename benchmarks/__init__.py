"""hpxpy microbenchmarks — throughput, thread scaling, and abstraction penalty.

The runner (:mod:`benchmarks.runner`) is the single measurement tool; per-op
kernels are registered there once and reused by every milestone's perf gate.
"""
