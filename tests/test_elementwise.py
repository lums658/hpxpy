"""M3: element-wise binary ops over Array, checked against ANALYTIC values via a
reduction of the result (no NumPy in the data path). Each op returns a new Array.
"""
import pytest

import hpxpy as hpx


@pytest.mark.parametrize("n", [1, 2, 1000, 100_000])
def test_add(n):
    # (i + i) summed = 2 * sum i = n(n-1)
    r = hpx.arange(n) + hpx.arange(n)
    assert isinstance(r, hpx.Array)
    assert r.size == n
    assert r.sum() == pytest.approx(n * (n - 1), rel=1e-12)


def test_sub_is_zero():
    a = hpx.arange(10_000)
    assert (a - a).sum() == 0.0


def test_mul_is_sum_of_squares():
    n = 10_000
    # (i * i) summed = sum i^2 = dot(arange, arange)
    a = hpx.arange(n)
    assert (a * a).sum() == pytest.approx(a.dot(a), rel=1e-12)


def test_div_constant():
    a = hpx.full(1000, 6.0)
    b = hpx.full(1000, 2.0)
    assert (a / b).sum() == pytest.approx(1000 * 3.0, rel=1e-12)


def test_operator_matches_named_method():
    a, b = hpx.arange(5000, dtype="float64"), hpx.full(5000, 2.0)
    assert (a + b).sum() == a.add(b).sum()
    assert (a * b).sum() == a.mul(b).sum()


def test_empty():
    a = hpx.arange(0)
    assert (a + a).size == 0
    assert (a + a).sum() == 0.0


@pytest.mark.parametrize("op", ["add", "sub", "mul", "div"])
def test_size_mismatch_raises(op):
    a, b = hpx.arange(10, dtype="float64"), hpx.arange(11, dtype="float64")
    with pytest.raises(ValueError):
        getattr(a, op)(b)
