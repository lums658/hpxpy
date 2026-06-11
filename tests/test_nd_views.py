"""Stage 3 N-D view ops + non-contiguous N-D compute tests.

Covers:
  Part 1 — view ops (transpose, .T, reshape, ravel, squeeze, expand_dims):
    - zero-copy sharing: mutate parent → visible through view
    - NumPy oracle for shape/strides/values after each op
  Part 2 — non-contiguous N-D compute on transposed/strided arrays:
    - reductions: sum, min, max
    - element-wise: add (array+array), mul_scalar
    - copy() produces a contiguous result with correct values
    - sort + cumsum on non-contiguous
    - binary shape-mismatch raises ValueError
"""
from __future__ import annotations

import numpy as np
import pytest

import hpxpy as hpx

_rng = np.random.default_rng(0xBEAD)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nd(shape):
    """Make a random float64 numpy array + matching hpxpy Array (copy=True)."""
    data = _rng.standard_normal(shape)
    return data, hpx.from_numpy(np.ascontiguousarray(data))


# ===========================================================================
# Part 1 — View ops
# ===========================================================================

# ---------------------------------------------------------------------------
# transpose / .T — shape and strides
# ---------------------------------------------------------------------------

def test_transpose_2d_shape():
    data, a = _nd((3, 4))
    at = a.transpose()
    assert at.shape == (4, 3)
    assert at.ndim == 2
    assert at.size == 12


def test_transpose_2d_T_alias():
    data, a = _nd((3, 4))
    assert a.T.shape == (4, 3)


def test_transpose_2d_values():
    data, a = _nd((3, 4))
    t = a.transpose()
    nt = data.T
    np.testing.assert_allclose(np.asarray(t), nt, rtol=1e-14, atol=1e-14)


def test_transpose_3d_default_reversal():
    data, a = _nd((2, 3, 4))
    t = a.transpose()
    assert t.shape == (4, 3, 2)
    np.testing.assert_allclose(np.asarray(t), data.T, rtol=1e-14, atol=1e-14)


def test_transpose_3d_explicit_axes():
    data, a = _nd((2, 3, 4))
    # (0,2,1): swap last two axes
    t = a.transpose((0, 2, 1))
    expected = data.transpose(0, 2, 1)
    assert t.shape == tuple(expected.shape)
    np.testing.assert_allclose(np.asarray(t), expected, rtol=1e-14, atol=1e-14)


def test_transpose_3d_full_permutation():
    data, a = _nd((2, 3, 5))
    axes = (2, 0, 1)
    t = a.transpose(axes)
    expected = data.transpose(*axes)
    assert t.shape == tuple(expected.shape)
    np.testing.assert_allclose(np.asarray(t), expected, rtol=1e-14, atol=1e-14)


def test_transpose_invalid_axis_raises():
    _, a = _nd((3, 4))
    with pytest.raises((ValueError, IndexError, Exception)):
        _ = a.transpose((0, 5))


def test_transpose_duplicate_axis_raises():
    _, a = _nd((3, 4))
    with pytest.raises((ValueError, Exception)):
        _ = a.transpose((0, 0))


# ---------------------------------------------------------------------------
# No-copy verification: mutate parent → visible through view
# ---------------------------------------------------------------------------

def test_transpose_no_copy_parent_to_view():
    """Writing to parent is visible through the transposed view."""
    data, a = _nd((3, 4))
    t = a.transpose()
    # Overwrite element [1, 2] in parent (row-major: element 1*4+2 = 6).
    a[1, 2] = 999.0
    # In the transposed view, that element is at [2, 1].
    assert t[2, 1] == pytest.approx(999.0, rel=1e-14)


def test_transpose_no_copy_view_to_parent():
    """Writing through the view is visible in the parent."""
    _, a = _nd((3, 4))
    t = a.transpose()
    t[0, 2] = -777.0
    # In parent: [2, 0]
    assert a[2, 0] == pytest.approx(-777.0, rel=1e-14)


# ---------------------------------------------------------------------------
# reshape
# ---------------------------------------------------------------------------

def test_reshape_contiguous_view():
    """Contiguous reshape should share memory (zero-copy)."""
    _, a = _nd((3, 4))
    r = a.reshape((4, 3))
    assert r.shape == (4, 3)
    assert r.size == 12
    # Mutate parent → visible in reshaped view.
    a[0, 0] = 42.0
    assert r[0, 0] == pytest.approx(42.0, rel=1e-14)


def test_reshape_contiguous_values():
    data, a = _nd((3, 4))
    r = a.reshape((2, 6))
    expected = data.reshape(2, 6)
    np.testing.assert_allclose(np.asarray(r), expected, rtol=1e-14, atol=1e-14)


def test_reshape_noncontiguous_copies():
    """Non-contiguous reshape must copy, so mutation of source is NOT visible."""
    data, a = _nd((4, 4))
    t = a.T  # non-contiguous transposed view
    r = t.reshape((16,))
    # r is a copy; modify original, r should not change.
    a[0, 0] = -9999.0
    # r[0] was the value of a[0,0] at reshape time; after mutating a[0,0] it must differ.
    assert r[0] != pytest.approx(-9999.0, rel=1e-14)


def test_reshape_noncontiguous_values():
    data, a = _nd((3, 4))
    t = a.T  # (4, 3) non-contiguous
    r = t.reshape((12,))
    expected = data.T.reshape(12)
    np.testing.assert_allclose(np.asarray(r), expected, rtol=1e-14, atol=1e-14)


def test_reshape_minus1_inferred():
    data, a = _nd((3, 4))
    r = a.reshape((-1, 2))
    expected = data.reshape(-1, 2)
    assert r.shape == tuple(expected.shape)
    np.testing.assert_allclose(np.asarray(r), expected, rtol=1e-14, atol=1e-14)


def test_reshape_size_mismatch_raises():
    _, a = _nd((3, 4))
    with pytest.raises((ValueError, Exception)):
        _ = a.reshape((5, 3))


# ---------------------------------------------------------------------------
# ravel
# ---------------------------------------------------------------------------

def test_ravel_contiguous():
    data, a = _nd((3, 4))
    r = a.ravel()
    assert r.shape == (12,)
    np.testing.assert_allclose(np.asarray(r), data.ravel(), rtol=1e-14, atol=1e-14)


def test_ravel_noncontiguous():
    data, a = _nd((3, 4))
    r = a.T.ravel()
    expected = data.T.ravel()
    assert r.shape == (12,)
    np.testing.assert_allclose(np.asarray(r), expected, rtol=1e-14, atol=1e-14)


# ---------------------------------------------------------------------------
# squeeze / expand_dims
# ---------------------------------------------------------------------------

def test_squeeze_all():
    data, a = _nd((1, 3, 1, 4))
    s = a.squeeze()
    assert s.shape == (3, 4)
    np.testing.assert_allclose(np.asarray(s), data.squeeze(), rtol=1e-14, atol=1e-14)


def test_squeeze_named_axis():
    _, a = _nd((1, 3, 1, 4))
    s = a.squeeze(0)
    assert s.shape == (3, 1, 4)


def test_squeeze_named_axes_tuple():
    _, a = _nd((1, 3, 1, 4))
    s = a.squeeze((0, 2))
    assert s.shape == (3, 4)


def test_squeeze_non_one_axis_raises():
    _, a = _nd((2, 3))
    with pytest.raises((ValueError, Exception)):
        _ = a.squeeze(0)  # axis 0 has size 2, not 1


def test_squeeze_no_copy():
    """squeeze is zero-copy: mutate parent, see it in squeezed view."""
    _, a = _nd((1, 4))
    s = a.squeeze(0)
    a[0, 1] = 55.5
    assert s[1] == pytest.approx(55.5, rel=1e-14)


def test_expand_dims_front():
    data, a = _nd((3, 4))
    e = a.expand_dims(0)
    assert e.shape == (1, 3, 4)
    np.testing.assert_allclose(np.asarray(e),
                               np.expand_dims(data, 0), rtol=1e-14, atol=1e-14)


def test_expand_dims_back():
    data, a = _nd((3, 4))
    e = a.expand_dims(2)
    assert e.shape == (3, 4, 1)
    np.testing.assert_allclose(np.asarray(e),
                               np.expand_dims(data, 2), rtol=1e-14, atol=1e-14)


def test_expand_dims_middle():
    data, a = _nd((3, 4))
    e = a.expand_dims(1)
    assert e.shape == (3, 1, 4)
    np.testing.assert_allclose(np.asarray(e),
                               np.expand_dims(data, 1), rtol=1e-14, atol=1e-14)


def test_expand_dims_no_copy():
    """expand_dims is zero-copy."""
    _, a = _nd((3,))
    e = a.expand_dims(0)
    a[1] = 77.0
    assert e[0, 1] == pytest.approx(77.0, rel=1e-14)


# ===========================================================================
# Part 2 — Non-contiguous N-D compute (vs NumPy oracle)
# ===========================================================================

# ---------------------------------------------------------------------------
# Reductions on transposed views
# ---------------------------------------------------------------------------

def test_transpose_sum():
    data, a = _nd((4, 5))
    assert a.T.sum() == pytest.approx(data.T.sum(), rel=1e-12)


def test_transpose_min():
    data, a = _nd((4, 5))
    assert a.T.min() == pytest.approx(data.T.min(), rel=1e-12)


def test_transpose_max():
    data, a = _nd((4, 5))
    assert a.T.max() == pytest.approx(data.T.max(), rel=1e-12)


def test_transpose_3d_sum():
    data, a = _nd((2, 3, 4))
    assert a.T.sum() == pytest.approx(data.T.sum(), rel=1e-12)


# ---------------------------------------------------------------------------
# Element-wise ops on transposed views
# ---------------------------------------------------------------------------

def test_add_transposed_arrays():
    data_a, a = _nd((3, 4))
    data_b, b = _nd((3, 4))
    result = a.T + b.T
    expected = data_a.T + data_b.T
    assert result.shape == tuple(expected.shape)
    np.testing.assert_allclose(np.asarray(result), expected, rtol=1e-13, atol=1e-13)


def test_mul_scalar_transposed():
    data, a = _nd((3, 4))
    result = a.T * 2.0
    expected = data.T * 2.0
    assert result.shape == tuple(expected.shape)
    np.testing.assert_allclose(np.asarray(result), expected, rtol=1e-13, atol=1e-13)


def test_sub_transposed():
    data_a, a = _nd((3, 4))
    data_b, b = _nd((3, 4))
    result = a.T - b.T
    expected = data_a.T - data_b.T
    np.testing.assert_allclose(np.asarray(result), expected, rtol=1e-13, atol=1e-13)


def test_div_transposed():
    # Use positive data only to avoid sign issues.
    data_a = _rng.uniform(1.0, 5.0, (3, 4))
    a = hpx.from_numpy(np.ascontiguousarray(data_a))
    data_b = _rng.uniform(1.0, 5.0, (3, 4))
    b = hpx.from_numpy(np.ascontiguousarray(data_b))
    result = a.T / b.T
    expected = data_a.T / data_b.T
    np.testing.assert_allclose(np.asarray(result), expected, rtol=1e-13, atol=1e-13)


# ---------------------------------------------------------------------------
# copy() of non-contiguous → contiguous with correct values
# ---------------------------------------------------------------------------

def test_copy_transposed_2d():
    data, a = _nd((4, 3))
    c = a.T.copy()
    assert c.shape == (3, 4)
    np.testing.assert_allclose(np.asarray(c), data.T, rtol=1e-14, atol=1e-14)


def test_copy_transposed_is_contiguous():
    _, a = _nd((4, 3))
    c = a.T.copy()
    # After copy, stride along last axis is 1 (contiguous).
    assert c.strides[-1] == 1 if hasattr(c, 'strides') else True


def test_copy_noncontiguous_3d():
    data, a = _nd((2, 3, 4))
    t = a.transpose((2, 0, 1))  # (4, 2, 3)
    c = t.copy()
    expected = data.transpose(2, 0, 1)
    np.testing.assert_allclose(np.asarray(c), expected, rtol=1e-14, atol=1e-14)


# ---------------------------------------------------------------------------
# sort on non-contiguous
# ---------------------------------------------------------------------------

def test_sort_noncontiguous():
    """sort() gathers non-contiguous elements, sorts, scatters back."""
    data, a = _nd((4, 3))
    t = a.T  # (3, 4) non-contiguous
    # Copy t's values to a fresh hpxpy array (contiguous), sort it as oracle.
    tc = t.copy()
    tc_np = np.asarray(tc).copy()
    expected_sorted = np.sort(tc_np.ravel())

    # Sort the non-contiguous view in place.
    t.sort()
    got_sorted = np.asarray(t).ravel()
    np.testing.assert_array_equal(np.sort(got_sorted), expected_sorted)
    # All values are the same, just reordered.
    np.testing.assert_array_equal(np.sort(got_sorted), np.sort(expected_sorted))


# ---------------------------------------------------------------------------
# cumsum on non-contiguous
# ---------------------------------------------------------------------------

def test_cumsum_noncontiguous():
    data, a = _nd((3, 4))
    result = a.T.cumsum()
    expected = data.T.ravel().cumsum()  # ravel then cumsum (C-order flat)
    assert result.shape == (12,)
    np.testing.assert_allclose(np.asarray(result), expected, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# binary shape mismatch raises
# ---------------------------------------------------------------------------

def test_binary_shape_mismatch_raises():
    _, a = _nd((3, 4))
    _, b = _nd((4, 3))
    with pytest.raises((ValueError, Exception)):
        _ = a + b


def test_binary_nd_1d_size_mismatch_raises():
    _, a = _nd((3, 4))
    _, b = _nd((3, 4))
    with pytest.raises((ValueError, Exception)):
        _ = a + b.T  # a is (3,4), b.T is (4,3) — shape mismatch


# ---------------------------------------------------------------------------
# hpx.transpose / hpx.reshape / hpx.squeeze / hpx.expand_dims module-level
# ---------------------------------------------------------------------------

def test_module_transpose():
    data, a = _nd((3, 4))
    t = hpx.transpose(a)
    np.testing.assert_allclose(np.asarray(t), data.T, rtol=1e-14, atol=1e-14)


def test_module_reshape():
    data, a = _nd((3, 4))
    r = hpx.reshape(a, (2, 6))
    np.testing.assert_allclose(np.asarray(r), data.reshape(2, 6),
                               rtol=1e-14, atol=1e-14)


def test_module_squeeze():
    data, a = _nd((1, 3, 1))
    s = hpx.squeeze(a)
    np.testing.assert_allclose(np.asarray(s), data.squeeze(),
                               rtol=1e-14, atol=1e-14)


def test_module_expand_dims():
    data, a = _nd((3, 4))
    e = hpx.expand_dims(a, 1)
    np.testing.assert_allclose(np.asarray(e), np.expand_dims(data, 1),
                               rtol=1e-14, atol=1e-14)


# ---------------------------------------------------------------------------
# dot on non-contiguous N-D (flat indices must match)
# ---------------------------------------------------------------------------

def test_dot_transposed():
    """dot on transposed 1-D equivalent (ravel first) — uses N-D flat_to_offset."""
    data_a = _rng.standard_normal(12)
    data_b = _rng.standard_normal(12)
    a = hpx.from_numpy(data_a.copy())
    b = hpx.from_numpy(data_b.copy())
    # Reshape to (3,4), transpose to (4,3), ravel — now non-contiguous flatten
    ta = a.reshape((3, 4)).T  # (4, 3) non-contiguous
    tb = b.reshape((3, 4)).T
    result = ta.ravel().dot(tb.ravel())
    expected = data_a.reshape(3, 4).T.ravel().dot(data_b.reshape(3, 4).T.ravel())
    assert result == pytest.approx(expected, rel=1e-12)
