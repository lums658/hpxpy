"""Stage 5 N-D axis reductions + matmul tests (numpy oracle).

Covers:
  Part 1 — axis reductions (sum/min/max over axis=int or axis=tuple):
    - 2-D 4x5 and 3-D 2x3x4 along each axis
    - axis=(0,2) on the 3-D array
    - keepdims=True shape checks
    - axis=None scalar regression (unchanged scalar fast path)
    - non-contiguous (transposed) reduction
    - error cases: out-of-range axis, duplicate axis, min/max over an empty axis
  Part 2 — matmul (a@b / a.dot(b) / hpx.matmul):
    - square and rectangular products vs numpy A@B
    - transposed operands (A.T@B, A@B.T) via strides
    - 1-D.1-D dot regression (scalar)
    - inner-dim mismatch and non-2-D operands raise
"""
from __future__ import annotations

import numpy as np
import pytest

import hpxpy as hpx

_rng = np.random.default_rng(0x5A_5E)


def _nd(shape):
    """Random float64 numpy array + matching contiguous hpxpy Array."""
    data = _rng.standard_normal(shape)
    return data, hpx.from_numpy(np.ascontiguousarray(data))


def _close(a_hpx, a_np):
    np.testing.assert_allclose(np.asarray(a_hpx), a_np, rtol=1e-13, atol=1e-13)


# ===========================================================================
# Part 1 — axis reductions
# ===========================================================================

@pytest.mark.parametrize("op", ["sum", "min", "max"])
@pytest.mark.parametrize("axis", [0, 1])
def test_axis_reduce_2d(op, axis):
    data, a = _nd((4, 5))
    got = getattr(a, op)(axis=axis)
    exp = getattr(np, op)(data, axis=axis)
    assert got.shape == exp.shape
    _close(got, exp)


@pytest.mark.parametrize("op", ["sum", "min", "max"])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_axis_reduce_3d_single(op, axis):
    data, a = _nd((2, 3, 4))
    got = getattr(a, op)(axis=axis)
    exp = getattr(np, op)(data, axis=axis)
    assert got.shape == exp.shape
    _close(got, exp)


@pytest.mark.parametrize("op", ["sum", "min", "max"])
def test_axis_reduce_3d_tuple(op):
    data, a = _nd((2, 3, 4))
    got = getattr(a, op)(axis=(0, 2))
    exp = getattr(np, op)(data, axis=(0, 2))
    assert got.shape == exp.shape
    _close(got, exp)


def test_axis_reduce_negative_axis():
    data, a = _nd((2, 3, 4))
    got = a.sum(axis=-1)
    exp = data.sum(axis=-1)
    assert got.shape == exp.shape
    _close(got, exp)


@pytest.mark.parametrize("op", ["sum", "min", "max"])
def test_axis_reduce_keepdims_shape(op):
    data, a = _nd((2, 3, 4))
    got = getattr(a, op)(axis=1, keepdims=True)
    exp = getattr(np, op)(data, axis=1, keepdims=True)
    assert got.shape == exp.shape == (2, 1, 4)
    _close(got, exp)

    got2 = getattr(a, op)(axis=(0, 2), keepdims=True)
    exp2 = getattr(np, op)(data, axis=(0, 2), keepdims=True)
    assert got2.shape == exp2.shape == (1, 3, 1)
    _close(got2, exp2)


@pytest.mark.parametrize("op", ["sum", "min", "max"])
def test_axis_none_scalar_regression(op):
    data, a = _nd((4, 5))
    got = getattr(a, op)()             # axis=None default
    exp = float(getattr(np, op)(data))
    assert isinstance(got, float)
    assert got == pytest.approx(exp, rel=1e-13, abs=1e-13)


def test_axis_reduce_noncontiguous():
    # a.T is a non-contiguous view; reduce over axis 0 must match numpy.
    data, a = _nd((4, 5))
    got = a.T.sum(axis=0)
    exp = data.T.sum(axis=0)
    assert got.shape == exp.shape
    _close(got, exp)

    gmin = a.T.min(axis=0)
    _close(gmin, data.T.min(axis=0))
    gmax = a.T.max(axis=0)
    _close(gmax, data.T.max(axis=0))


def test_free_functions_axis():
    data, a = _nd((3, 4))
    _close(hpx.sum(a, axis=0), data.sum(axis=0))
    _close(hpx.min(a, axis=1), data.min(axis=1))
    _close(hpx.max(a, axis=0), data.max(axis=0))
    _close(hpx.sum(a, axis=1, keepdims=True), data.sum(axis=1, keepdims=True))
    # axis=None scalar path through the free functions.
    assert hpx.sum(a) == pytest.approx(data.sum(), rel=1e-13)
    assert hpx.min(a) == pytest.approx(data.min(), rel=1e-13)
    assert hpx.max(a) == pytest.approx(data.max(), rel=1e-13)


def test_axis_out_of_range_raises():
    _, a = _nd((2, 3))
    with pytest.raises(IndexError):
        a.sum(axis=5)
    with pytest.raises(IndexError):
        a.min(axis=-4)


def test_axis_duplicate_raises():
    _, a = _nd((2, 3, 4))
    with pytest.raises(ValueError):
        a.sum(axis=(0, 0))
    with pytest.raises(ValueError):
        a.max(axis=(1, -2))   # -2 normalizes to 1 -> duplicate


def test_axis_minmax_empty_raises():
    # A zero-length reduced axis is undefined for min/max.
    z = hpx.zeros((0, 3))
    with pytest.raises(ValueError):
        z.min(axis=0)
    with pytest.raises(ValueError):
        z.max(axis=0)
    # sum over an empty axis -> zeros (shape (3,)).
    s = z.sum(axis=0)
    assert s.shape == (3,)
    _close(s, np.zeros(3))


# ===========================================================================
# Part 2 — matmul
# ===========================================================================

def test_matmul_square():
    a_np, a = _nd((4, 4))
    b_np, b = _nd((4, 4))
    exp = a_np @ b_np
    _close(a @ b, exp)
    _close(a.dot(b), exp)
    _close(a.matmul(b), exp)
    _close(hpx.matmul(a, b), exp)
    _close(hpx.dot(a, b), exp)


def test_matmul_rectangular():
    a_np, a = _nd((3, 5))
    b_np, b = _nd((5, 2))
    exp = a_np @ b_np
    assert (a @ b).shape == (3, 2)
    _close(a @ b, exp)
    _close(a.dot(b), exp)
    _close(hpx.matmul(a, b), exp)


def test_matmul_transposed_operands():
    a_np, a = _nd((5, 3))
    b_np, b = _nd((5, 2))
    # A.T (3x5) @ B (5x2)
    _close(a.T @ b, a_np.T @ b_np)
    _close(a.T.matmul(b), a_np.T @ b_np)

    c_np, c = _nd((3, 4))
    d_np, d = _nd((2, 4))
    # C (3x4) @ D.T (4x2)
    _close(c @ d.T, c_np @ d_np.T)
    _close(c.matmul(d.T), c_np @ d_np.T)


def test_dot_1d_scalar_regression():
    a_np = _rng.standard_normal(7)
    b_np = _rng.standard_normal(7)
    a = hpx.from_numpy(np.ascontiguousarray(a_np))
    b = hpx.from_numpy(np.ascontiguousarray(b_np))
    exp = float(a_np @ b_np)
    assert isinstance(a.dot(b), float)
    assert a.dot(b) == pytest.approx(exp, rel=1e-13)
    assert hpx.dot(a, b) == pytest.approx(exp, rel=1e-13)
    assert isinstance(a @ b, float)
    assert (a @ b) == pytest.approx(exp, rel=1e-13)


def test_matmul_inner_mismatch_raises():
    _, a = _nd((3, 5))
    _, b = _nd((4, 2))   # inner 5 != 4
    with pytest.raises(ValueError):
        a.matmul(b)
    with pytest.raises(ValueError):
        a @ b


def test_matmul_non_2d_raises():
    _, a = _nd((3, 4, 2))   # 3-D
    _, b = _nd((2, 3))
    with pytest.raises(ValueError):
        a.matmul(b)
    # dot / @ with a 3-D operand -> ValueError (unsupported rank).
    with pytest.raises(ValueError):
        a.dot(b)
    with pytest.raises(ValueError):
        a @ b
