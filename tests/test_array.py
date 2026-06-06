"""M1 #1: the Array core — a thin wrapper over hpx::partitioned_vector.

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
    assert a.num_partitions >= 1


def test_zeros_empty():
    a = hpx.zeros(0)
    assert a.size == 0


def test_repr():
    assert "Array(size=10" in repr(hpx.zeros(10))
