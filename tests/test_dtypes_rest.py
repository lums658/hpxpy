"""Phase A.2 stage A2.3: the REMAINING compute kernels are now dtype-generic.

After A2.2 templated the core element-wise kernels (sum/min/max/dot, the binary
and scalar ops), this stage templates the rest over the element type T:

  - axis reductions: sum/min/max(axis=..., keepdims=...)
  - matmul (a @ b / a.dot(b))
  - copy, sort, is_sorted, cumsum
  - 1-D slice assignment (a[i:j] = scalar / a[i:j] = other_array)

All of these now work for float32 and int64, not just float64, with NumPy as the
oracle. The min/max axis identities use std::numeric_limits<T>::max()/lowest()
(NOT a +/-inf double, which is 0 for integers — a negative int64 array exercises
this). Result dtype is preserved; no automatic promotion (matmul / slice-assign
on mismatched dtypes raise). The ONLY thing still deferred is N-D slice
ASSIGNMENT (a[i:j, :] = ...), which still raises (see test_nd_slicing.py).
"""
import numpy as np
import pytest

import hpxpy as hpx


def hpx_arr(np_arr):
    """Import a numpy array into hpxpy (copy=True for NUMA-aware allocation)."""
    return hpx.from_numpy(np.ascontiguousarray(np_arr), copy=True)


def np_out(a):
    """Export an hpxpy Array to a contiguous numpy array."""
    return np.asarray(a).copy()


# ===========================================================================
# Axis reductions: sum / min / max (axis int + keepdims) on 2-D and 3-D.
# int64 cases include NEGATIVE values so min/max exercise the lowest()/max()
# identities (a wrong +/-inf->0 seed would give 0 for an all-negative axis).
# ===========================================================================

@pytest.mark.parametrize("dt", [np.float32, np.int64])
@pytest.mark.parametrize("op", ["sum", "min", "max"])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("keepdims", [False, True])
def test_axis_reduce_2d(dt, op, axis, keepdims):
    data = np.arange(20, dtype=dt).reshape(4, 5)
    a = hpx_arr(data)
    got = getattr(a, op)(axis=axis, keepdims=keepdims)
    exp = getattr(np, op)(data, axis=axis, keepdims=keepdims)
    assert got.shape == exp.shape
    assert got.dtype == np.dtype(dt)
    np.testing.assert_array_equal(np_out(got), exp)


@pytest.mark.parametrize("dt", [np.float32, np.int64])
@pytest.mark.parametrize("op", ["sum", "min", "max"])
@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("keepdims", [False, True])
def test_axis_reduce_3d(dt, op, axis, keepdims):
    data = np.arange(24, dtype=dt).reshape(2, 3, 4)
    a = hpx_arr(data)
    got = getattr(a, op)(axis=axis, keepdims=keepdims)
    exp = getattr(np, op)(data, axis=axis, keepdims=keepdims)
    assert got.shape == exp.shape
    assert got.dtype == np.dtype(dt)
    np.testing.assert_array_equal(np_out(got), exp)


@pytest.mark.parametrize("dt", [np.float32, np.int64])
@pytest.mark.parametrize("op", ["min", "max"])
@pytest.mark.parametrize("axis", [0, 1])
def test_axis_minmax_all_negative_not_zero(dt, op, axis):
    """The int identity trap: an all-negative axis must NOT reduce to 0.

    With a +/-inf->0 seed (wrong for integers) an all-negative int64 column/row
    would yield 0; the lowest()/max() seeds give the true extreme.
    """
    data = -(np.arange(1, 13, dtype=dt).reshape(3, 4))   # all strictly negative
    a = hpx_arr(data)
    got = getattr(a, op)(axis=axis)
    exp = getattr(np, op)(data, axis=axis)
    np.testing.assert_array_equal(np_out(got), exp)
    assert np.all(np_out(got) < 0)                       # never the 0 seed


def test_axis_multi_axis_3d():
    """axis=(0, 2) on a 3-D int64 array, sum/min/max vs numpy."""
    data = (np.arange(24, dtype=np.int64).reshape(2, 3, 4) - 10)
    a = hpx_arr(data)
    for op in ("sum", "min", "max"):
        got = getattr(a, op)(axis=(0, 2))
        exp = getattr(np, op)(data, axis=(0, 2))
        assert got.shape == exp.shape
        np.testing.assert_array_equal(np_out(got), exp)


# ===========================================================================
# matmul (a @ b): square, rectangular, transposed operand, vs numpy.
# ===========================================================================

@pytest.mark.parametrize("dt", [np.float32, np.int64])
def test_matmul_square(dt):
    A = np.arange(1, 10, dtype=dt).reshape(3, 3)
    B = (np.arange(9, dtype=dt).reshape(3, 3) + 2)
    r = hpx_arr(A) @ hpx_arr(B)
    assert r.dtype == np.dtype(dt)
    np.testing.assert_array_equal(np_out(r), A @ B)


@pytest.mark.parametrize("dt", [np.float32, np.int64])
def test_matmul_rectangular(dt):
    A = np.arange(12, dtype=dt).reshape(3, 4)     # (3,4)
    B = np.arange(8, dtype=dt).reshape(4, 2)      # (4,2)
    r = hpx_arr(A) @ hpx_arr(B)
    assert r.shape == (3, 2)
    assert r.dtype == np.dtype(dt)
    np.testing.assert_array_equal(np_out(r), A @ B)


@pytest.mark.parametrize("dt", [np.float32, np.int64])
def test_matmul_transposed_operand(dt):
    """A @ B.T via strides (B.T is a non-contiguous view; no copy)."""
    A = np.arange(12, dtype=dt).reshape(3, 4)
    B = np.arange(8, dtype=dt).reshape(2, 4)      # B.T is (4,2)
    r = hpx_arr(A) @ hpx_arr(B).T
    assert r.shape == (3, 2)
    np.testing.assert_array_equal(np_out(r), A @ B.T)


def test_matmul_dtype_mismatch_promotes():
    """A2.4: mixed-dtype matmul promotes to the numpy result dtype (no longer raises)."""
    A = np.arange(4, dtype=np.float32).reshape(2, 2)
    B = np.arange(4, dtype=np.float64).reshape(2, 2)
    r = hpx_arr(A) @ hpx_arr(B)
    assert r.dtype == np.dtype(np.float64) == (A @ B).dtype
    np.testing.assert_allclose(np_out(r), A @ B, rtol=1e-6)


def test_matmul_dtype_preserved_f64():
    """The f64 path is unchanged (zero-penalty regression)."""
    A = np.arange(1, 5, dtype=np.float64).reshape(2, 2)
    r = hpx_arr(A) @ hpx_arr(A)
    assert r.dtype == np.float64
    np.testing.assert_allclose(np_out(r), A @ A)


# ===========================================================================
# copy / sort / cumsum / is_sorted: dtype preserved + match numpy.
# ===========================================================================

@pytest.mark.parametrize("dt", [np.float32, np.int64])
def test_copy_preserves_dtype(dt):
    data = np.array([5, -3, 8, 1, 0, -7], dtype=dt)
    a = hpx_arr(data)
    c = a.copy()
    assert c.dtype == np.dtype(dt)
    np.testing.assert_array_equal(np_out(c), data)
    # Independent buffer: mutating the copy does not touch the source.
    c[0] = 99
    assert a[0] == data[0]


@pytest.mark.parametrize("dt", [np.float32, np.int64])
def test_sort_preserves_dtype(dt):
    data = np.array([5, -3, 8, 1, 0, -7, 4], dtype=dt)
    a = hpx_arr(data)
    a.sort()
    assert a.dtype == np.dtype(dt)
    np.testing.assert_array_equal(np_out(a), np.sort(data))


@pytest.mark.parametrize("dt", [np.float32, np.int64])
def test_is_sorted(dt):
    asc = hpx_arr(np.array([1, 2, 3, 4, 5], dtype=dt))
    assert asc.is_sorted() is True
    unsorted = hpx_arr(np.array([1, 9, 2, 8, 3], dtype=dt))
    assert unsorted.is_sorted() is False


@pytest.mark.parametrize("dt", [np.float32, np.int64])
def test_cumsum_preserves_dtype(dt):
    data = np.array([1, 2, 3, 4, 5], dtype=dt)
    a = hpx_arr(data)
    c = a.cumsum()
    assert c.dtype == np.dtype(dt)
    np.testing.assert_array_equal(np_out(c), np.cumsum(data))


# ===========================================================================
# 1-D slice assignment: a[2:5] = scalar / a[2:5] = other (f32 + i64).
# ===========================================================================

@pytest.mark.parametrize("dt", [np.float32, np.int64])
def test_slice_assign_scalar(dt):
    data = np.arange(8, dtype=dt)
    a = hpx_arr(data)
    a[2:5] = 7
    data[2:5] = 7
    np.testing.assert_array_equal(np_out(a), data)
    assert a.dtype == np.dtype(dt)


@pytest.mark.parametrize("dt", [np.float32, np.int64])
def test_slice_assign_array(dt):
    data = np.arange(8, dtype=dt)
    other = np.array([100, 200, 300], dtype=dt)
    a = hpx_arr(data)
    a[2:5] = hpx_arr(other)
    data[2:5] = other
    np.testing.assert_array_equal(np_out(a), data)
    assert a.dtype == np.dtype(dt)


def test_int64_slice_assign_float_scalar_truncates():
    """numpy a[i:j] = 2.5 stores 2 on an int array (truncation); match it."""
    data = np.arange(8, dtype=np.int64)
    a = hpx_arr(data)
    a[2:5] = 2.5
    data[2:5] = 2.5                # numpy truncates -> 2
    np.testing.assert_array_equal(np_out(a), data)
    assert np_out(a)[2] == 2


def test_slice_assign_dtype_mismatch_raises():
    a = hpx.from_numpy(np.arange(8, dtype=np.float32))
    other = hpx.from_numpy(np.arange(3, dtype=np.float64))
    with pytest.raises((TypeError, ValueError)):
        a[2:5] = other


# ===========================================================================
# Strided (non-contiguous) inputs go through the flat_to_offset else-branches.
# ===========================================================================

@pytest.mark.parametrize("dt", [np.float32, np.int64])
def test_strided_copy_sort_cumsum_axis(dt):
    x = (np.arange(12, dtype=dt).reshape(3, 4) - 5)
    a = hpx_arr(x)
    at = a.T                                      # non-contiguous view
    xt = x.T
    # copy of a transposed view is contiguous + equal
    c = at.copy()
    assert c.dtype == np.dtype(dt)
    np.testing.assert_array_equal(np_out(c), xt)
    # axis reduction over a non-contiguous array
    np.testing.assert_array_equal(np_out(at.sum(axis=0)), xt.sum(axis=0))
    np.testing.assert_array_equal(np_out(at.min(axis=1)), xt.min(axis=1))
    np.testing.assert_array_equal(np_out(at.max(axis=0)), xt.max(axis=0))


@pytest.mark.parametrize("dt", [np.float32, np.int64])
def test_reverse_view_sort_cumsum(dt):
    x = np.array([5, 1, 4, 2, 3], dtype=dt)
    a = hpx_arr(x)
    rev = a[::-1]                                  # strided reverse view
    np.testing.assert_array_equal(np_out(rev.cumsum()), np.cumsum(x[::-1]))
    rev.sort()                                     # sorts the underlying range
    np.testing.assert_array_equal(np_out(rev), np.sort(x[::-1]))
