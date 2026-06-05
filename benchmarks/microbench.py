"""Microbenchmark seed (M0): median-of-times throughput for the zero-copy sum,
HPXPy vs NumPy. The full 3-way (vs C++ baseline) + thread/locality sweeps land
with the per-op milestones; this establishes the measurement convention.

Run (on an exclusive node for clean numbers):
    LD_PRELOAD=$HPXPY_TCMALLOC python benchmarks/microbench.py
"""
from __future__ import annotations

import statistics
import time

import numpy as np

import hpxpy as hpx


def median_time(fn, warmup: int = 1, repeats: int = 5) -> float:
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def main() -> None:
    hpx.init()
    print(f"HPX worker threads: {hpx.num_worker_threads()}")
    print(f"{'N':>12} {'HPXPy GEl/s':>12} {'NumPy GEl/s':>12}")
    for n in (10**6, 10**7, 10**8):
        a = np.arange(n, dtype=np.float64)
        t_hpx = median_time(lambda a=a: hpx.sum(a))
        t_np = median_time(lambda a=a: a.sum())
        print(f"{n:>12} {n/t_hpx/1e9:>12.2f} {n/t_np/1e9:>12.2f}")
    hpx.finalize()


if __name__ == "__main__":
    main()
