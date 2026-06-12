"""NumPy breadth Wave 2: construction helpers.

NumPy is the oracle throughout. The new constructors (arange with start/stop/step,
linspace, eye, identity, empty) and the ``*_like`` helpers (zeros_like, ones_like,
empty_like, full_like) are each checked against ``numpy``: values (assert_allclose /
assert_array_equal) AND output dtype. ``empty``/``empty_like`` only assert shape +
dtype (their values are arbitrary by design).
"""
from __future__ import annotations

import numpy as np
import pytest

import hpxpy as hpx


def np_out(a):
    """Export an hpxpy Array to a contiguous numpy array."""
    return np.asarray(a).copy()


# ===========================================================================
# A. arange(start, stop, step) — values + dtype vs numpy; back-compat.
# ===========================================================================

def test_arange_start_stop_step():
    r = hpx.arange(2, 10, 2)
    e = np.arange(2, 10, 2)
    np.testing.assert_array_equal(np_out(r), e)
    assert r.dtype == e.dtype == np.dtype(np.int64)


def test_arange_float_step():
    r = hpx.arange(0, 1, 0.1)
    e = np.arange(0, 1, 0.1)
    np.testing.assert_allclose(np_out(r), e)
    # All-integer args would be int64, but a float step => float64 (numpy rule).
    assert r.dtype == e.dtype == np.dtype(np.float64)
    assert r.shape == e.shape  # same element count (ceil((1-0)/0.1) == 10)


def test_arange_negative_step():
    r = hpx.arange(10, 0, -1)
    e = np.arange(10, 0, -1)
    np.testing.assert_array_equal(np_out(r), e)
    assert r.dtype == e.dtype == np.dtype(np.int64)


def test_arange_one_arg_back_compat():
    r = hpx.arange(5)
    e = np.arange(5)
    np.testing.assert_array_equal(np_out(r), e)
    assert r.dtype == e.dtype == np.dtype(np.int64)


def test_arange_dtype_override():
    r = hpx.arange(3, dtype="float32")
    e = np.arange(3, dtype="float32")
    np.testing.assert_array_equal(np_out(r), e)
    assert r.dtype == e.dtype == np.dtype(np.float32)


def test_arange_empty_when_step_overshoots():
    r = hpx.arange(5, 5)
    e = np.arange(5, 5)
    assert r.shape == e.shape == (0,)
    np.testing.assert_array_equal(np_out(r), e)


def test_arange_zero_step_raises():
    with pytest.raises(ValueError):
        hpx.arange(0, 10, 0)


# ===========================================================================
# B. linspace — values (allclose) + float64 dtype + exact endpoint.
# ===========================================================================

def test_linspace_basic():
    r = hpx.linspace(0, 1, 11)
    e = np.linspace(0, 1, 11)
    np.testing.assert_allclose(np_out(r), e)
    assert r.dtype == e.dtype == np.dtype(np.float64)


def test_linspace_endpoint_false():
    r = hpx.linspace(0, 1, 5, endpoint=False)
    e = np.linspace(0, 1, 5, endpoint=False)
    np.testing.assert_allclose(np_out(r), e)
    assert r.dtype == np.dtype(np.float64)


def test_linspace_single_point():
    r = hpx.linspace(2, 3, 1)
    e = np.linspace(2, 3, 1)
    np.testing.assert_allclose(np_out(r), e)
    assert r.shape == e.shape == (1,)


def test_linspace_last_point_exact_when_endpoint():
    r = hpx.linspace(0, 1, 11)
    assert np_out(r)[-1] == 1.0  # exactly stop, no round-off


def test_linspace_dtype_override():
    r = hpx.linspace(0, 1, 5, dtype="float32")
    e = np.linspace(0, 1, 5, dtype="float32")
    np.testing.assert_allclose(np_out(r), e)
    assert r.dtype == np.dtype(np.float32)


def test_linspace_zero_num_is_empty():
    r = hpx.linspace(0, 5, 0)
    e = np.linspace(0, 5, 0)
    assert r.shape == e.shape == (0,)
    assert r.dtype == np.dtype(np.float64)


def test_linspace_negative_num_raises():
    with pytest.raises(ValueError):
        hpx.linspace(0, 1, -3)


# ===========================================================================
# C. eye / identity — values + dtype vs numpy.
# ===========================================================================

def test_eye_square():
    r = hpx.eye(3)
    e = np.eye(3)
    np.testing.assert_array_equal(np_out(r), e)
    assert r.dtype == e.dtype == np.dtype(np.float64)
    assert r.shape == (3, 3)


def test_eye_rectangular():
    r = hpx.eye(3, 4)
    e = np.eye(3, 4)
    np.testing.assert_array_equal(np_out(r), e)
    assert r.shape == (3, 4)


def test_eye_upper_diagonal():
    r = hpx.eye(3, 3, k=1)
    e = np.eye(3, 3, k=1)
    np.testing.assert_array_equal(np_out(r), e)


def test_eye_lower_diagonal():
    r = hpx.eye(3, 3, k=-1)
    e = np.eye(3, 3, k=-1)
    np.testing.assert_array_equal(np_out(r), e)


def test_eye_dtype_override():
    r = hpx.eye(3, dtype="int64")
    e = np.eye(3, dtype="int64")
    np.testing.assert_array_equal(np_out(r), e)
    assert r.dtype == np.dtype(np.int64)


def test_identity():
    r = hpx.identity(4)
    e = np.identity(4)
    np.testing.assert_array_equal(np_out(r), e)
    assert r.dtype == e.dtype == np.dtype(np.float64)
    assert r.shape == (4, 4)


# ===========================================================================
# D. empty — shape + dtype only (values arbitrary).
# ===========================================================================

def test_empty_shape_dtype():
    r = hpx.empty((2, 3), dtype="float32")
    assert r.shape == (2, 3)
    assert r.dtype == np.dtype(np.float32)


def test_empty_1d_default_dtype():
    r = hpx.empty(5)
    assert r.shape == (5,)
    assert r.dtype == np.dtype(np.float64)


# ===========================================================================
# E. *_like helpers — shape + dtype match (and values where defined).
# ===========================================================================

@pytest.mark.parametrize("dt", [np.float64, np.float32, np.int64])
def test_zeros_like(dt):
    a = hpx.zeros((2, 3), dtype=dt)
    r = hpx.zeros_like(a)
    e = np.zeros((2, 3), dtype=dt)
    np.testing.assert_array_equal(np_out(r), e)
    assert r.shape == (2, 3)
    assert r.dtype == np.dtype(dt)


def test_zeros_like_dtype_override():
    a = hpx.zeros((2, 3), dtype=np.int64)
    r = hpx.zeros_like(a, dtype="float32")
    assert r.shape == (2, 3)
    assert r.dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(np_out(r), np.zeros((2, 3), dtype="float32"))


@pytest.mark.parametrize("dt", [np.float64, np.float32, np.int64])
def test_ones_like(dt):
    a = hpx.zeros((4,), dtype=dt)
    r = hpx.ones_like(a)
    e = np.ones((4,), dtype=dt)
    np.testing.assert_array_equal(np_out(r), e)
    assert r.shape == (4,)
    assert r.dtype == np.dtype(dt)


def test_ones_like_dtype_override():
    a = hpx.zeros((4,), dtype=np.float32)
    r = hpx.ones_like(a, dtype="int64")
    assert r.dtype == np.dtype(np.int64)
    np.testing.assert_array_equal(np_out(r), np.ones((4,), dtype="int64"))


@pytest.mark.parametrize("dt", [np.float64, np.float32, np.int64])
def test_full_like(dt):
    a = hpx.zeros((2, 2), dtype=dt)
    r = hpx.full_like(a, 7)
    e = np.full((2, 2), 7, dtype=dt)
    np.testing.assert_array_equal(np_out(r), e)
    assert r.shape == (2, 2)
    assert r.dtype == np.dtype(dt)


def test_full_like_dtype_override():
    a = hpx.zeros((3,), dtype=np.int64)
    r = hpx.full_like(a, 2.0, dtype="float64")
    assert r.dtype == np.dtype(np.float64)
    np.testing.assert_array_equal(np_out(r), np.full((3,), 2.0, dtype="float64"))


@pytest.mark.parametrize("dt", [np.float64, np.float32, np.int64])
def test_empty_like(dt):
    a = hpx.zeros((2, 3), dtype=dt)
    r = hpx.empty_like(a)
    # empty_like: only shape + dtype are defined (values arbitrary).
    assert r.shape == (2, 3)
    assert r.dtype == np.dtype(dt)


def test_empty_like_dtype_override():
    a = hpx.zeros((5,), dtype=np.float64)
    r = hpx.empty_like(a, dtype="int64")
    assert r.shape == (5,)
    assert r.dtype == np.dtype(np.int64)
