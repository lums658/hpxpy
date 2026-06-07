"""Smoke tests: the HPX runtime starts, and an Array reaches an HPX parallel
algorithm with a correct result. This is the substrate the rest builds on.
"""
import hpxpy as hpx


def test_runtime_up():
    assert hpx.num_worker_threads() >= 1
    assert isinstance(hpx.hpx_version(), str) and hpx.hpx_version()


def test_array_sum_reaches_hpx():
    # sum(0..n-1) = n(n-1)/2 — proves the Array buffer reaches hpx::reduce.
    n = 1_000_000
    assert hpx.arange(n).sum() == n * (n - 1) / 2


def test_free_function_is_method_alias():
    a = hpx.arange(1000)
    assert hpx.sum(a) == a.sum()
