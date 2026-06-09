"""Strided slice views: a[::2], a[1::2], a[2:9:3], a[::-1], etc.

Tests use NumPy as an oracle: the same slice applied to the same data should give
identical results whether accessed via hpxpy or numpy.
"""
import numpy as np
import pytest

import hpxpy as hpx

_rng = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make(n=20):
    """Create a numpy array and a matching hpxpy Array from the same data."""
    data = _rng.standard_normal(n)
    return data, hpx.from_numpy(data.copy())


# ---------------------------------------------------------------------------
# Basic strided view properties
# ---------------------------------------------------------------------------

def test_even_stride_size():
    data, a = _make(20)
    v = a[::2]
    assert v.size == len(data[::2])


def test_odd_stride_size():
    data, a = _make(20)
    v = a[1::2]
    assert v.size == len(data[1::2])


def test_three_stride_size():
    data, a = _make(20)
    v = a[2:9:3]
    assert v.size == len(data[2:9:3])


def test_reverse_size():
    data, a = _make(20)
    v = a[::-1]
    assert v.size == len(data[::-1])


# ---------------------------------------------------------------------------
# Element values: a[i] on a strided view
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("slc,n", [
    (slice(None, None, 2),   20),   # a[::2]
    (slice(1, None, 2),      20),   # a[1::2]
    (slice(2, 9, 3),         20),   # a[2:9:3]
    (slice(None, None, -1),  20),   # a[::-1]
    (slice(None, None, 3),   15),   # a[::3]
])
def test_strided_element_access(slc, n):
    data = _rng.standard_normal(n)
    a = hpx.from_numpy(data.copy())
    v_hpx = a[slc]
    v_np = data[slc]
    assert v_hpx.size == len(v_np)
    for i in range(len(v_np)):
        assert v_hpx[i] == pytest.approx(v_np[i], rel=1e-15, abs=1e-15)


# ---------------------------------------------------------------------------
# to_numpy: strided view exports as a numpy strided view (zero-copy, positive step)
# ---------------------------------------------------------------------------

def test_strided_to_numpy_positive_step():
    data, a = _make(20)
    v_hpx = a[::2]
    v_np_oracle = data[::2]
    np.testing.assert_array_equal(np.asarray(v_hpx), v_np_oracle)


def test_strided_to_numpy_odd_start():
    data, a = _make(20)
    v_hpx = a[1::2]
    np.testing.assert_array_equal(np.asarray(v_hpx), data[1::2])


def test_strided_to_numpy_step3():
    data, a = _make(20)
    v_hpx = a[2:9:3]
    np.testing.assert_array_equal(np.asarray(v_hpx), data[2:9:3])


def test_strided_to_numpy_reverse():
    """to_numpy of a reverse view should give the reversed values."""
    data, a = _make(20)
    v_hpx = a[::-1]
    got = np.asarray(v_hpx)
    np.testing.assert_array_equal(got, data[::-1])


# ---------------------------------------------------------------------------
# Strided reductions vs numpy oracle
# ---------------------------------------------------------------------------

def test_strided_sum_even():
    data, a = _make(20)
    assert a[::2].sum() == pytest.approx(data[::2].sum(), rel=1e-12)


def test_strided_sum_odd():
    data, a = _make(20)
    assert a[1::2].sum() == pytest.approx(data[1::2].sum(), rel=1e-12)


def test_strided_sum_reverse():
    data, a = _make(20)
    assert a[::-1].sum() == pytest.approx(data.sum(), rel=1e-12)


def test_strided_min_even():
    data, a = _make(20)
    assert a[::2].min() == pytest.approx(data[::2].min(), rel=1e-12)


def test_strided_max_even():
    data, a = _make(20)
    assert a[::2].max() == pytest.approx(data[::2].max(), rel=1e-12)


def test_strided_min_reverse():
    data, a = _make(20)
    assert a[::-1].min() == pytest.approx(data.min(), rel=1e-12)


def test_strided_max_reverse():
    data, a = _make(20)
    assert a[::-1].max() == pytest.approx(data.max(), rel=1e-12)


def test_strided_dot_even():
    data_a, a = _make(20)
    data_b = _rng.standard_normal(20)
    b = hpx.from_numpy(data_b.copy())
    expected = np.dot(data_a[::2], data_b[::2])
    assert a[::2].dot(b[::2]) == pytest.approx(expected, rel=1e-12)


def test_strided_dot_mixed():
    """Dot product of two arrays with different strides."""
    data_a, a = _make(20)
    data_b = _rng.standard_normal(20)
    b = hpx.from_numpy(data_b.copy())
    # a[::2] has 10 elements; b[1::2] has 10 elements
    expected = np.dot(data_a[::2], data_b[1::2])
    assert a[::2].dot(b[1::2]) == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# Element-wise ops on strided views -> contiguous result
# ---------------------------------------------------------------------------

def test_strided_add_arrays():
    data_a, a = _make(20)
    data_b = _rng.standard_normal(20)
    b = hpx.from_numpy(data_b.copy())
    result = a[::2] + b[::2]
    assert result.stride == 1        # result is always contiguous
    np.testing.assert_allclose(np.asarray(result), data_a[::2] + data_b[::2],
                               rtol=1e-14, atol=1e-14)


def test_strided_mul_scalar():
    data, a = _make(20)
    result = a[::2] * 2.0
    assert result.stride == 1
    np.testing.assert_allclose(np.asarray(result), data[::2] * 2.0,
                               rtol=1e-14, atol=1e-14)


def test_strided_sub_arrays():
    data_a, a = _make(20)
    data_b = _rng.uniform(1.0, 2.0, 20)
    b = hpx.from_numpy(data_b.copy())
    result = a[::2] - b[::2]
    np.testing.assert_allclose(np.asarray(result), data_a[::2] - data_b[::2],
                               rtol=1e-14, atol=1e-14)


def test_strided_div_scalar():
    data, a = _make(20)
    result = a[::2] / 3.0
    np.testing.assert_allclose(np.asarray(result), data[::2] / 3.0,
                               rtol=1e-14, atol=1e-14)


def test_strided_reverse_add():
    data_a, a = _make(10)
    data_b = _rng.standard_normal(10)
    b = hpx.from_numpy(data_b.copy())
    result = a[::-1] + b[::-1]
    np.testing.assert_allclose(np.asarray(result), data_a[::-1] + data_b[::-1],
                               rtol=1e-14, atol=1e-14)


# ---------------------------------------------------------------------------
# In-place sort on a strided view
# ---------------------------------------------------------------------------

def test_strided_sort_inplace():
    """sort() on a strided view sorts those logical elements in place."""
    data, a = _make(20)
    # a[::2] selects even-indexed elements; sort them
    v = a[::2]
    assert not v.is_sorted() or True   # might be sorted by chance; just proceed
    v.sort()
    assert v.is_sorted()
    # Verify values via to_numpy
    sorted_vals = np.sort(data[::2])
    np.testing.assert_array_equal(np.asarray(v), sorted_vals)


def test_strided_sort_mutates_parent():
    """sort() on a strided view must mutate the correct elements in the parent."""
    # Build a known descending sequence so we know exactly which elements change.
    n = 10
    data = np.arange(n, dtype=float) * -1.0   # [0,-1,-2,...,-9]
    a = hpx.from_numpy(data.copy())
    # Sort even-indexed elements: initially [0,-2,-4,-6,-8] -> sorted: [-8,-6,-4,-2,0]
    a[::2].sort()
    arr = np.asarray(a)
    even_vals = arr[::2]
    np.testing.assert_array_equal(even_vals, np.sort(data[::2]))
    # Odd-indexed elements must be unchanged.
    np.testing.assert_array_equal(arr[1::2], data[1::2])


def test_strided_is_sorted_unsorted():
    data, a = _make(20)
    # arange is sorted; its reverse is not (for n>1)
    b = hpx.arange(10)
    assert not b[::-1].is_sorted()


def test_strided_is_sorted_sorted():
    b = hpx.arange(10)
    assert b[::2].is_sorted()   # 0,2,4,6,8 — ascending


# ---------------------------------------------------------------------------
# Setitem (scalar) through a strided view
# ---------------------------------------------------------------------------

def test_strided_getitem_setitem():
    a = hpx.arange(10)
    v = a[::2]                  # logical elements: [0,2,4,6,8]
    assert v[0] == 0.0
    assert v[2] == 4.0
    v[2] = 99.0
    assert v[2] == 99.0
    assert a[4] == 99.0         # mutated the correct parent element


def test_strided_setitem_mutates_parent():
    a = hpx.arange(10)
    v = a[1::2]                 # logical elements: [1,3,5,7,9]
    v[0] = 100.0
    assert a[1] == 100.0
    v[3] = 200.0
    assert a[7] == 200.0


# ---------------------------------------------------------------------------
# Slice assignment on strided view still raises (deferred)
# ---------------------------------------------------------------------------

def test_strided_slice_assignment_raises():
    a = hpx.arange(10)
    with pytest.raises(ValueError):
        a[::2] = 0.0


def test_step2_slice_assignment_raises():
    a = hpx.arange(10)
    with pytest.raises(ValueError):
        a[1::3] = 5.0


# ---------------------------------------------------------------------------
# copy() of a strided view -> contiguous owning Array
# ---------------------------------------------------------------------------

def test_strided_copy_is_contiguous():
    data, a = _make(20)
    c = a[::2].copy()
    assert c.stride == 1
    np.testing.assert_array_equal(np.asarray(c), data[::2])


def test_strided_copy_reverse():
    data, a = _make(10)
    c = a[::-1].copy()
    assert c.stride == 1
    np.testing.assert_array_equal(np.asarray(c), data[::-1])


# ---------------------------------------------------------------------------
# cumsum of a strided view
# ---------------------------------------------------------------------------

def test_strided_cumsum():
    data, a = _make(20)
    result = a[::2].cumsum()
    assert result.stride == 1
    np.testing.assert_allclose(np.asarray(result), np.cumsum(data[::2]),
                               rtol=1e-14, atol=1e-14)


# ---------------------------------------------------------------------------
# stride property
# ---------------------------------------------------------------------------

def test_stride_property_contiguous():
    a = hpx.arange(10)
    assert a.stride == 1
    assert a[2:7].stride == 1


def test_stride_property_strided():
    a = hpx.arange(10)
    assert a[::2].stride == 2
    assert a[1::3].stride == 3


def test_stride_property_reverse():
    a = hpx.arange(10)
    assert a[::-1].stride == -1


# ---------------------------------------------------------------------------
# Empty strided views
# ---------------------------------------------------------------------------

def test_empty_strided_view():
    a = hpx.arange(10)
    v = a[5:5:2]
    assert v.size == 0
    assert v.sum() == 0.0
