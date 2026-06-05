"""M0 smoke tests: the HPX runtime starts, and a zero-copy array reaches an HPX
parallel algorithm with a correct result. This is the substrate the rest builds on.
"""
import numpy as np

import hpxpy as hpx


def test_runtime_up():
    assert hpx.num_worker_threads() >= 1
    assert isinstance(hpx.hpx_version(), str) and hpx.hpx_version()


def test_zero_copy_sum_matches_numpy():
    a = np.arange(1_000_000, dtype=np.float64)
    got = hpx.sum(a)
    assert abs(got - a.sum()) / a.sum() < 1e-12


def test_sum_empty_and_single():
    assert hpx.sum(np.array([], dtype=np.float64)) == 0.0
    assert hpx.sum(np.array([42.0])) == 42.0
