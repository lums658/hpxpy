"""Slice/view model: a[i] scalar get/set and a[i:j] contiguous views (numpy
semantics — slices share memory, not copies). step != 1 is not yet supported.
"""
import pytest

import hpxpy as hpx


# --- element access ---------------------------------------------------------

def test_getitem_value_and_negative():
    a = hpx.arange(10)
    assert a[0] == 0.0
    assert a[5] == 5.0
    assert a[-1] == 9.0
    assert a[-10] == 0.0
    assert isinstance(a[3], float)


@pytest.mark.parametrize("i", [10, 11, -11, -100])
def test_getitem_out_of_range(i):
    a = hpx.arange(10)
    with pytest.raises(IndexError):
        a[i]


def test_setitem_scalar():
    a = hpx.arange(10)
    a[3] = 99.0
    assert a[3] == 99.0
    a[-1] = 7.0
    assert a[9] == 7.0
    with pytest.raises(IndexError):
        a[10] = 1.0


# --- views share memory -----------------------------------------------------

def test_slice_is_a_view_sharing_memory():
    a = hpx.arange(10)
    v = a[2:5]                 # elements 2,3,4
    assert v.size == 3
    assert v[0] == 2.0
    a[3] = 99.0                # mutate parent...
    assert v[1] == 99.0        # ...seen through the view
    v[2] = 7.0                 # mutate view...
    assert a[4] == 7.0         # ...seen through the parent


def test_empty_slice():
    assert hpx.arange(10)[5:5].size == 0


def test_step_not_supported():
    a = hpx.arange(10)
    with pytest.raises(ValueError):
        a[::2]


# --- views compose with the existing ops ------------------------------------

def test_view_reduction_matches_analytic():
    n, i, j = 1000, 100, 400
    # sum of arange[i:j] = sum(0..j-1) - sum(0..i-1)
    expected = (j - 1) * j / 2 - (i - 1) * i / 2
    assert hpx.arange(n)[i:j].sum() == pytest.approx(expected, rel=1e-12)


def test_view_elementwise():
    a = hpx.arange(1000)
    # (a[0:100] + a[0:100]).sum() = 2 * sum(0..99)
    r = a[0:100] + a[0:100]
    assert r.size == 100
    assert r.sum() == pytest.approx(2 * (99 * 100 / 2), rel=1e-12)


def test_view_sort_mutates_parent_region():
    a = hpx.arange(20) * -1.0          # descending [0,-1,...,-19]
    v = a[5:15]
    assert not v.is_sorted()
    v.sort()                            # sorts the parent's [5:15) in place
    assert v.is_sorted()
    assert a[5:15].is_sorted()          # a fresh view of the same region agrees


def test_view_copy_is_independent():
    a = hpx.arange(20) * -1.0
    c = a[5:15].copy()                  # owning copy
    c.sort()
    assert c.is_sorted()
    assert not a[5:15].is_sorted()      # parent untouched
