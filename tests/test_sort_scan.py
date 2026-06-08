"""M3: sort + inclusive prefix scan over the Array, with NumPy semantics:
``a.sort()`` sorts IN PLACE, ``hpx.sort(a)`` returns a sorted copy (``a`` unchanged),
``a.cumsum()`` returns a new Array. Sortedness is checked with ``is_sorted`` (no
element access needed); content via analytic reductions.
"""
import pytest

import hpxpy as hpx


def test_arange_is_sorted():
    assert hpx.arange(1000).is_sorted()
    assert hpx.arange(0).is_sorted()
    assert hpx.arange(1).is_sorted()


def test_in_place_sort_mutates():
    # arange * -1 = [0, -1, -2, ...] is descending (not sorted) for n > 1.
    a = hpx.arange(1000) * -1.0
    assert not a.is_sorted()
    total = a.sum()
    assert a.sort() is None       # in-place, numpy-style
    assert a.is_sorted()
    assert a.sum() == pytest.approx(total, rel=1e-12)   # a permutation


@pytest.mark.parametrize("n", [1, 2, 1000, 100_000])
def test_in_place_sort_is_a_permutation(n):
    a = hpx.arange(n) * -1.0
    before = a.sum()
    a.sort()
    assert a.is_sorted()
    assert a.sum() == pytest.approx(before, rel=1e-12)


def test_free_sort_returns_copy_and_leaves_input(a_n=1000):
    a = hpx.arange(a_n) * -1.0
    s = hpx.sort(a)
    assert s.is_sorted()
    assert not a.is_sorted()       # numpy.sort does NOT modify the input
    assert s.size == a_n


def test_copy_is_independent():
    a = hpx.arange(1000) * -1.0
    c = a.copy()
    assert c.sum() == pytest.approx(a.sum(), rel=1e-12)
    c.sort()                       # mutating the copy...
    assert c.is_sorted()
    assert not a.is_sorted()       # ...must not touch the original


# --- cumsum (inclusive scan) ------------------------------------------------

def test_cumsum_of_ones():
    n = 1000
    # cumsum([1,1,...]) = [1,2,...,n]; sum = n(n+1)/2
    c = hpx.full(n, 1.0).cumsum()
    assert c.size == n
    assert c.sum() == pytest.approx(n * (n + 1) / 2, rel=1e-12)


def test_cumsum_scales_constant():
    n = 1000
    assert hpx.full(n, 3.0).cumsum().sum() == pytest.approx(3.0 * n * (n + 1) / 2, rel=1e-12)


def test_cumsum_empty():
    assert hpx.arange(0).cumsum().size == 0


def test_free_function_cumsum():
    assert hpx.cumsum(hpx.full(100, 1.0)).sum() == pytest.approx(100 * 101 / 2, rel=1e-12)
