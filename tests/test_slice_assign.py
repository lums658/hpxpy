"""Slice assignment a[i:j] = scalar (fill) / Array (copy) — NumPy semantics,
contiguous (step 1). Mutates in place (the Array, or its parent if a is a view).
"""
import numpy as np
import pytest

import hpxpy as hpx


def test_assign_scalar_fills_range():
    a = hpx.arange(10)
    a[2:5] = 7.0
    assert a[2] == 7.0 and a[3] == 7.0 and a[4] == 7.0
    assert a[1] == 1.0 and a[5] == 5.0        # neighbours untouched
    assert a.sum() == 7.0 * 3 + (0 + 1 + 5 + 6 + 7 + 8 + 9)


def test_assign_scalar_negative_and_open():
    a = hpx.arange(10)
    a[-3:] = 0.0
    assert a[7] == 0.0 and a[8] == 0.0 and a[9] == 0.0
    a[:2] = 1.0
    assert a[0] == 1.0 and a[1] == 1.0


def test_assign_array_copies_in():
    a = hpx.zeros(10)
    a[3:7] = hpx.full(4, 5.0)
    assert a.sum() == pytest.approx(4 * 5.0)
    assert a[3] == 5.0 and a[6] == 5.0 and a[2] == 0.0 and a[7] == 0.0


def test_assign_array_from_view_source():
    a = hpx.arange(10)
    a[0:3] = hpx.arange(10)[7:10]            # rhs is itself a slice view [7,8,9]
    assert a[0] == 7.0 and a[1] == 8.0 and a[2] == 9.0


def test_assign_through_view_mutates_parent():
    a = hpx.arange(10)
    v = a[2:8]
    v[1:3] = 99.0                            # a[3:5] = 99
    assert a[3] == 99.0 and a[4] == 99.0


def test_assign_size_mismatch_raises():
    a = hpx.arange(10)
    with pytest.raises(ValueError):
        a[2:5] = hpx.full(2, 1.0)            # 3 vs 2


def test_assign_step_not_supported():
    a = hpx.arange(10)
    with pytest.raises(ValueError):
        a[::2] = 0.0


def test_parity_with_numpy():
    n = 100
    npx = np.arange(n, dtype=np.float64)
    a = hpx.from_numpy(npx.copy())
    npx[10:20] = 3.0
    a[10:20] = 3.0
    npx[50:60] = np.arange(10, dtype=np.float64)
    a[50:60] = hpx.arange(10, dtype="float64")
    np.testing.assert_array_equal(a.to_numpy(), npx)
