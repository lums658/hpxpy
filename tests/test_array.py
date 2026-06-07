"""M1 #1: the Array core — a thin wrapper over a NUMA-aware HPX compute::vector.

Phase 1 has no NumPy bridge, so these cover structure/lifecycle only; content
correctness arrives with M2 reductions (checked against analytic values).
"""
import hpxpy as hpx


def test_zeros_basic():
    a = hpx.zeros(1000)
    assert isinstance(a, hpx.Array)
    assert a.size == 1000
    assert a.ndim == 1
    assert len(a) == 1000


def test_zeros_empty():
    a = hpx.zeros(0)
    assert a.size == 0


def test_full_basic():
    a = hpx.full(500, 3.5)
    assert isinstance(a, hpx.Array)
    assert a.size == 500
    assert a.ndim == 1
    assert len(a) == 500


def test_arange_basic():
    a = hpx.arange(2000)
    assert isinstance(a, hpx.Array)
    assert a.size == 2000
    assert a.ndim == 1
    assert len(a) == 2000


def test_arange_empty():
    a = hpx.arange(0)
    assert a.size == 0


# Content correctness of full/arange is validated in M2 via analytic reductions
# (e.g. sum(arange(n)) == n*(n-1)/2; min/max/sum of full(n, c)).


def test_repr():
    assert "Array(size=10" in repr(hpx.zeros(10))
