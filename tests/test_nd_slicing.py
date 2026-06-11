"""Stage 6 N-D multi-axis slicing tests (numpy oracle).

Covers a[i:j, ::k], mixed int+slice (axis-drop), Ellipsis, negative/reverse
slices, implicit trailing axes, empty/singleton slices, zero-copy memory
sharing, compute on sliced (non-contiguous) views, and error cases. N-D slice
ASSIGNMENT remains deferred (still raises).
"""
from __future__ import annotations

import numpy as np
import pytest

import hpxpy as hpx

_rng = np.random.default_rng(0x51CE)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nd(shape):
    """Random float64 numpy array + matching contiguous hpxpy Array."""
    data = _rng.standard_normal(shape)
    return data, hpx.from_numpy(np.ascontiguousarray(data))


def _check(got, expected):
    """Assert an hpxpy view matches a numpy subarray in shape and values."""
    exp = np.asarray(expected)
    assert tuple(got.shape) == tuple(exp.shape)
    np.testing.assert_allclose(np.asarray(got), exp, rtol=1e-14, atol=1e-14)


# ===========================================================================
# View shapes / values vs numpy (2-D 4x6)
# ===========================================================================

def test_slice_both_axes():
    data, a = _nd((4, 6))
    _check(a[1:3, ::2], data[1:3, ::2])


def test_slice_strided_both_axes():
    data, a = _nd((4, 6))
    _check(a[0:4:2, 1:5:2], data[0:4:2, 1:5:2])


def test_slice_axis0_full_axis1():
    data, a = _nd((4, 6))
    _check(a[1:3, :], data[1:3, :])


def test_slice_full_axis0_axis1():
    data, a = _nd((4, 6))
    _check(a[:, 1:5], data[:, 1:5])


# ===========================================================================
# Mixed int + slice (axis-drop)
# ===========================================================================

def test_int_row():
    data, a = _nd((4, 6))
    for i in (0, 2, -1, -4):
        _check(a[i, :], data[i, :])


def test_int_col():
    data, a = _nd((4, 6))
    for j in (0, 3, -1, -6):
        _check(a[:, j], data[:, j])


def test_int_plus_slice():
    data, a = _nd((4, 6))
    _check(a[1, 2:5], data[1, 2:5])


def test_int_plus_strided_slice():
    data, a = _nd((4, 6))
    _check(a[1, 2:5:2], data[1, 2:5:2])


def test_int_plus_slice_negative_index():
    data, a = _nd((4, 6))
    _check(a[-2, 2:5], data[-2, 2:5])
    _check(a[1, -4:-1], data[1, -4:-1])


# ===========================================================================
# Ellipsis (3-D 3x4x5)
# ===========================================================================

def test_ellipsis_trailing_int():
    data, a = _nd((3, 4, 5))
    _check(a[..., 0], data[..., 0])


def test_ellipsis_leading_int():
    data, a = _nd((3, 4, 5))
    _check(a[0, ...], data[0, ...])


def test_ellipsis_middle():
    data, a = _nd((3, 4, 5))
    _check(a[1:2, ..., 3:5], data[1:2, ..., 3:5])


def test_ellipsis_only():
    data, a = _nd((3, 4, 5))
    _check(a[...], data[...])


# ===========================================================================
# Negative / reverse slices
# ===========================================================================

def test_negative_and_reverse():
    data, a = _nd((4, 6))
    _check(a[-2:, ::-1], data[-2:, ::-1])


def test_reverse_axis0_strided_axis1():
    data, a = _nd((4, 6))
    _check(a[::-1, ::2], data[::-1, ::2])


# ===========================================================================
# Implicit trailing axes
# ===========================================================================

def test_implicit_trailing():
    data, a = _nd((4, 6))
    _check(a[1:3], data[1:3, :])
    # And explicitly equal to the padded form.
    np.testing.assert_array_equal(np.asarray(a[1:3]), np.asarray(a[1:3, :]))


def test_implicit_trailing_3d():
    data, a = _nd((3, 4, 5))
    _check(a[1:3], data[1:3])
    _check(a[1:3, 0:2], data[1:3, 0:2])


# ===========================================================================
# Empty / singleton slices
# ===========================================================================

def test_empty_slice():
    data, a = _nd((4, 6))
    v = a[2:2, :]
    assert v.size == 0
    _check(v, data[2:2, :])


def test_singleton_row():
    data, a = _nd((4, 6))
    _check(a[1:2, :], data[1:2, :])


def test_singleton_col():
    data, a = _nd((4, 6))
    _check(a[:, 3:4], data[:, 3:4])


# ===========================================================================
# Zero-copy: view shares memory with parent (both directions)
# ===========================================================================

def test_view_shares_memory_parent_to_view():
    data, a = _nd((4, 6))
    v = a[1:3, ::2]
    parent = np.asarray(a)
    parent[1, 0] = -321.0          # element at view[0, 0]
    assert np.asarray(v)[0, 0] == pytest.approx(-321.0, rel=1e-14)


def test_view_shares_memory_view_to_parent():
    data, a = _nd((4, 6))
    v = a[1:3, ::2]
    np.asarray(v)[1, 2] = 654.0    # parent element [2, 4]
    assert np.asarray(a)[2, 4] == pytest.approx(654.0, rel=1e-14)


def test_int_drop_view_shares_memory():
    data, a = _nd((4, 6))
    row = a[2, :]
    np.asarray(a)[2, 3] = 111.0
    assert np.asarray(row)[3] == pytest.approx(111.0, rel=1e-14)


# ===========================================================================
# Compute on sliced (non-contiguous) views vs numpy
# ===========================================================================

def test_sum_on_sliced_view():
    data, a = _nd((4, 6))
    assert a[1:3, ::2].sum() == pytest.approx(data[1:3, ::2].sum(), rel=1e-12)


def test_min_on_reversed_view():
    data, a = _nd((4, 6))
    assert a[::-1, :].min() == pytest.approx(data[::-1, :].min(), rel=1e-12)


def test_add_two_sliced_views():
    data, a = _nd((4, 6))
    result = a[1:3, 1:3] + a[0:2, 0:2]
    expected = data[1:3, 1:3] + data[0:2, 0:2]
    _check(result, expected)


def test_copy_of_sliced_view():
    data, a = _nd((4, 6))
    c = a[1:3, ::2].copy()
    _check(c, data[1:3, ::2])


def test_asarray_of_sliced_view():
    data, a = _nd((4, 6))
    np.testing.assert_allclose(
        np.asarray(a[1:3, ::2]), data[1:3, ::2], rtol=1e-14, atol=1e-14)


# ===========================================================================
# Errors
# ===========================================================================

def test_two_ellipsis_raises():
    _, a = _nd((3, 4, 5))
    with pytest.raises(IndexError):
        _ = a[..., ...]


def test_too_many_specifiers_raises():
    _, a = _nd((4, 6))
    with pytest.raises(IndexError):
        _ = a[1:2, 1:2, 0]


def test_out_of_range_int_raises():
    _, a = _nd((4, 6))
    with pytest.raises(IndexError):
        _ = a[10, :]
    with pytest.raises(IndexError):
        _ = a[:, 99]


def test_nd_slice_assignment_still_raises():
    _, a = _nd((4, 6))
    with pytest.raises((TypeError, ValueError, Exception)):
        a[1:3, :] = 0.0
