"""NumPy breadth Wave 1: element-wise math ufuncs + mean/prod/any/all/count_nonzero.

NumPy is the oracle throughout. Every ufunc is checked vs ``np.<fn>`` for float64,
float32, and int64 (where applicable), asserting BOTH the values (assert_allclose /
assert_array_equal) AND the output dtype (the dtype-resolution rules: preserve-dtype
for negative/abs/sign/maximum/minimum/power/mod/floor_divide/clip; promote-int-to-
float64 for sqrt/exp/log/sin/cos/tan/floor/ceil/trunc/round; mean -> always float64;
prod -> preserves dtype). Operators (-a, abs(a), a**2, a%b, a//b) and a couple of
N-D / strided (transposed) inputs are exercised too.
"""
from __future__ import annotations

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
# A. Preserve-dtype unary ufuncs: negative / abs / sign.
# ===========================================================================

@pytest.mark.parametrize("dt", [np.float64, np.float32, np.int64])
def test_negative(dt):
    x = np.array([3, -7, 0, 5, -1], dtype=dt)
    r = hpx.negative(hpx_arr(x))
    np.testing.assert_array_equal(np_out(r), np.negative(x))
    assert r.dtype == np.dtype(dt)


@pytest.mark.parametrize("dt", [np.float64, np.float32, np.int64])
def test_abs(dt):
    x = np.array([3, -7, 0, 5, -1], dtype=dt)
    r = hpx.abs(hpx_arr(x))
    np.testing.assert_array_equal(np_out(r), np.abs(x))
    assert r.dtype == np.dtype(dt)


@pytest.mark.parametrize("dt", [np.float64, np.float32, np.int64])
def test_sign(dt):
    x = np.array([3, -7, 0, 5, -1], dtype=dt)
    r = hpx.sign(hpx_arr(x))
    np.testing.assert_array_equal(np_out(r), np.sign(x))
    assert r.dtype == np.dtype(dt)


# ===========================================================================
# B. Promote-int-to-float64 unary ufuncs: sqrt/exp/log/sin/cos/tan/
#    floor/ceil/trunc/round. float input keeps dtype; int64 -> float64.
# ===========================================================================

# Positive-only base so sqrt/log are valid for every dtype.
_POS = [1, 2, 3, 4, 9, 16]
# General base (with negatives) for the ops valid everywhere.
_GEN = [-2.5, -1.0, 0.0, 1.5, 2.5, 3.7]

_PROMOTE_OPS = [
    ("sqrt", np.sqrt, _POS),
    ("exp", np.exp, [-1, 0, 1, 2]),
    ("log", np.log, _POS),
    ("sin", np.sin, _GEN),
    ("cos", np.cos, _GEN),
    ("tan", np.tan, _GEN),
    ("floor", np.floor, _GEN),
    ("ceil", np.ceil, _GEN),
    ("trunc", np.trunc, _GEN),
    ("round", np.rint, _GEN),
]


@pytest.mark.parametrize("name,npfn,vals", _PROMOTE_OPS)
@pytest.mark.parametrize("dt", [np.float64, np.float32])
def test_promote_float_input_keeps_dtype(name, npfn, vals, dt):
    x = np.array(vals, dtype=dt)
    r = getattr(hpx, name)(hpx_arr(x))
    np.testing.assert_allclose(np_out(r), npfn(x), rtol=1e-5, atol=1e-6)
    assert r.dtype == np.dtype(dt)    # float input preserves its float dtype


@pytest.mark.parametrize("name,npfn,vals", _PROMOTE_OPS)
def test_promote_int_input_becomes_float64(name, npfn, vals):
    # int64 -> float64 output (numpy: np.sqrt(int64_array).dtype == float64).
    xi = np.array([int(v) for v in vals], dtype=np.int64)
    r = getattr(hpx, name)(hpx_arr(xi))
    assert r.dtype == np.float64
    np.testing.assert_allclose(np_out(r), npfn(xi.astype(np.float64)),
                               rtol=1e-12, atol=1e-12)


def test_rint_alias():
    x = np.array([0.5, 1.5, 2.5, -0.5, -1.5], dtype=np.float64)
    # round-half-to-even (banker's): matches np.rint, NOT np.round-away-from-zero.
    np.testing.assert_array_equal(np_out(hpx.rint(hpx_arr(x))), np.rint(x))
    np.testing.assert_array_equal(np_out(hpx.round(hpx_arr(x))), np.rint(x))


# ===========================================================================
# C. Preserve-dtype binary ufuncs: maximum/minimum/power/mod/floor_divide/clip.
# ===========================================================================

@pytest.mark.parametrize("dt", [np.float64, np.float32, np.int64])
def test_maximum_minimum(dt):
    a = np.array([1, 8, 3, 5, -2], dtype=dt)
    b = np.array([4, 2, 9, 5, -7], dtype=dt)
    ha, hb = hpx_arr(a), hpx_arr(b)
    rmax = hpx.maximum(ha, hb)
    rmin = hpx.minimum(ha, hb)
    np.testing.assert_array_equal(np_out(rmax), np.maximum(a, b))
    np.testing.assert_array_equal(np_out(rmin), np.minimum(a, b))
    assert rmax.dtype == np.dtype(dt)
    assert rmin.dtype == np.dtype(dt)


@pytest.mark.parametrize("dt", [np.float64, np.float32, np.int64])
def test_power(dt):
    a = np.array([1, 2, 3, 4], dtype=dt)
    b = np.array([3, 2, 2, 1], dtype=dt)
    r = hpx.power(hpx_arr(a), hpx_arr(b))
    np.testing.assert_allclose(np_out(r), np.power(a, b), rtol=1e-5)
    assert r.dtype == np.dtype(dt)


@pytest.mark.parametrize("dt", [np.float64, np.float32, np.int64])
def test_mod(dt):
    # Include negative dividends/divisors so the divisor-signed semantics are checked.
    a = np.array([7, -7, 7, -7, 10], dtype=dt)
    b = np.array([3, 3, -3, -3, 4], dtype=dt)
    r = hpx.mod(hpx_arr(a), hpx_arr(b))
    np.testing.assert_allclose(np_out(r), np.mod(a, b), rtol=1e-5)
    assert r.dtype == np.dtype(dt)


@pytest.mark.parametrize("dt", [np.float64, np.float32, np.int64])
def test_floor_divide(dt):
    a = np.array([7, -7, 7, -7, 10], dtype=dt)
    b = np.array([3, 3, -3, -3, 4], dtype=dt)
    r = hpx.floor_divide(hpx_arr(a), hpx_arr(b))
    np.testing.assert_allclose(np_out(r), np.floor_divide(a, b), rtol=1e-5)
    assert r.dtype == np.dtype(dt)


@pytest.mark.parametrize("dt", [np.float64, np.float32, np.int64])
def test_clip(dt):
    x = np.array([-5, -1, 0, 3, 7, 12], dtype=dt)
    r = hpx.clip(hpx_arr(x), 0, 8)
    np.testing.assert_array_equal(np_out(r), np.clip(x, 0, 8))
    assert r.dtype == np.dtype(dt)


def test_binary_dtype_mismatch_raises():
    a = hpx.ones(3, dtype=np.float32)
    b = hpx.ones(3, dtype=np.float64)
    for fn in (hpx.maximum, hpx.minimum, hpx.power, hpx.mod, hpx.floor_divide):
        with pytest.raises((TypeError, ValueError)):
            fn(a, b)


# ===========================================================================
# D. Operators: -a, abs(a), a**2, a%b, a//b.
# ===========================================================================

@pytest.mark.parametrize("dt", [np.float64, np.float32, np.int64])
def test_operator_neg_abs(dt):
    x = np.array([3, -7, 0, 5, -1], dtype=dt)
    a = hpx_arr(x)
    np.testing.assert_array_equal(np_out(-a), -x)
    assert (-a).dtype == np.dtype(dt)
    np.testing.assert_array_equal(np_out(abs(a)), np.abs(x))
    assert abs(a).dtype == np.dtype(dt)


@pytest.mark.parametrize("dt", [np.float64, np.float32, np.int64])
def test_operator_pow_scalar(dt):
    x = np.array([1, 2, 3, 4], dtype=dt)
    a = hpx_arr(x)
    np.testing.assert_allclose(np_out(a ** 2), x ** 2, rtol=1e-5)
    assert (a ** 2).dtype == np.dtype(dt)


@pytest.mark.parametrize("dt", [np.float64, np.float32, np.int64])
def test_operator_mod_floordiv(dt):
    x = np.array([7, -7, 10, 13, -13], dtype=dt)
    y = np.array([3, 3, 4, -5, 5], dtype=dt)
    a, b = hpx_arr(x), hpx_arr(y)
    np.testing.assert_allclose(np_out(a % b), x % y, rtol=1e-5)
    np.testing.assert_allclose(np_out(a // b), x // y, rtol=1e-5)
    assert (a % b).dtype == np.dtype(dt)
    assert (a // b).dtype == np.dtype(dt)
    # Scalar operator forms too.
    np.testing.assert_allclose(np_out(a % 3), x % 3, rtol=1e-5)
    np.testing.assert_allclose(np_out(a // 3), x // 3, rtol=1e-5)


def test_power_operator_array():
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    y = np.array([2.0, 3.0, 2.0], dtype=np.float64)
    a, b = hpx_arr(x), hpx_arr(y)
    np.testing.assert_allclose(np_out(a ** b), x ** y, rtol=1e-12)


# ===========================================================================
# E. N-D + strided (transposed) inputs reuse the unary/binary strided paths.
# ===========================================================================

def test_unary_nd_and_strided():
    x = np.arange(1, 13, dtype=np.float64).reshape(3, 4)
    a = hpx_arr(x)
    # N-D contiguous.
    np.testing.assert_allclose(np_out(hpx.sqrt(a)), np.sqrt(x), rtol=1e-12)
    assert hpx.sqrt(a).shape == (3, 4)
    # Transposed (non-contiguous) view -> strided unary path.
    at = a.T
    np.testing.assert_allclose(np_out(hpx.negative(at)), -x.T)
    np.testing.assert_allclose(np_out(hpx.sqrt(at)), np.sqrt(x.T), rtol=1e-12)


def test_binary_nd_and_strided():
    x = np.arange(12, dtype=np.float64).reshape(3, 4) + 1
    y = (np.arange(12, dtype=np.float64).reshape(3, 4) % 5) + 1
    a, b = hpx_arr(x), hpx_arr(y)
    np.testing.assert_array_equal(np_out(hpx.maximum(a, b)), np.maximum(x, y))
    # Transposed operands (non-contiguous) go through the strided/broadcast path.
    at = a.T
    bc = hpx_arr(np.ascontiguousarray(x.T))
    np.testing.assert_array_equal(np_out(hpx.maximum(at, bc)), np.maximum(x.T, x.T))


# ===========================================================================
# F. Reductions: mean / prod / any / all / count_nonzero.
# ===========================================================================

@pytest.mark.parametrize("dt", [np.float64, np.float32, np.int64])
def test_mean_axis_none(dt):
    x = np.array([2, 4, 6, 8, 10], dtype=dt)
    m = hpx.mean(hpx_arr(x))
    assert isinstance(m, float)
    assert m == pytest.approx(float(np.mean(x)), rel=1e-6)


@pytest.mark.parametrize("dt", [np.float64, np.float32, np.int64])
def test_mean_axis_and_keepdims(dt):
    x = (np.arange(12, dtype=dt)).reshape(3, 4)
    a = hpx_arr(x)
    for axis in (0, 1):
        r = hpx.mean(a, axis=axis)
        assert r.dtype == np.float64            # mean is ALWAYS float64
        np.testing.assert_allclose(np_out(r), np.mean(x, axis=axis), rtol=1e-12)
    rk = hpx.mean(a, axis=1, keepdims=True)
    assert rk.shape == (3, 1)
    np.testing.assert_allclose(np_out(rk), np.mean(x, axis=1, keepdims=True),
                               rtol=1e-12)
    assert rk.dtype == np.float64


@pytest.mark.parametrize("dt", [np.float64, np.float32, np.int64])
def test_prod_axis_none(dt):
    x = np.array([1, 2, 3, 4, 5], dtype=dt)
    p = hpx.prod(hpx_arr(x))
    assert p == pytest.approx(float(np.prod(x)), rel=1e-6)
    if dt is np.int64:
        assert isinstance(p, int)


@pytest.mark.parametrize("dt", [np.float64, np.float32, np.int64])
def test_prod_axis_and_keepdims(dt):
    x = (np.arange(1, 13, dtype=dt)).reshape(3, 4)
    a = hpx_arr(x)
    for axis in (0, 1):
        r = hpx.prod(a, axis=axis)
        assert r.dtype == np.dtype(dt)          # prod PRESERVES dtype
        np.testing.assert_allclose(np_out(r), np.prod(x, axis=axis), rtol=1e-6)
    rk = hpx.prod(a, axis=0, keepdims=True)
    assert rk.shape == (1, 4)
    np.testing.assert_allclose(np_out(rk), np.prod(x, axis=0, keepdims=True),
                               rtol=1e-6)


@pytest.mark.parametrize("dt", [np.float64, np.float32, np.int64])
def test_any_all(dt):
    has_zero = np.array([0, 1, 2, 3], dtype=dt)
    all_nonzero = np.array([1, 2, 3, 4], dtype=dt)
    all_zero = np.array([0, 0, 0], dtype=dt)
    # any
    assert hpx.any(hpx_arr(has_zero)) is bool(np.any(has_zero))
    assert hpx.any(hpx_arr(all_zero)) is bool(np.any(all_zero))
    assert hpx.any(hpx_arr(all_nonzero)) is bool(np.any(all_nonzero))
    # all
    assert hpx.all(hpx_arr(has_zero)) is bool(np.all(has_zero))
    assert hpx.all(hpx_arr(all_nonzero)) is bool(np.all(all_nonzero))
    assert hpx.all(hpx_arr(all_zero)) is bool(np.all(all_zero))


def test_any_all_return_python_bool():
    a = hpx_arr(np.array([1, 2, 3], dtype=np.float64))
    assert type(hpx.any(a)) is bool
    assert type(hpx.all(a)) is bool


def test_any_all_axis_not_supported():
    a = hpx_arr(np.arange(6, dtype=np.float64).reshape(2, 3))
    with pytest.raises((ValueError, NotImplementedError)):
        hpx.any(a, axis=0)
    with pytest.raises((ValueError, NotImplementedError)):
        hpx.all(a, axis=1)


@pytest.mark.parametrize("dt", [np.float64, np.float32, np.int64])
def test_count_nonzero(dt):
    x = np.array([0, 1, 0, 2, 3, 0, -4], dtype=dt)
    c = hpx.count_nonzero(hpx_arr(x))
    assert c == int(np.count_nonzero(x))
    assert type(c) is int


def test_count_nonzero_nd():
    x = np.array([[0, 1, 2], [3, 0, 0]], dtype=np.float64)
    assert hpx.count_nonzero(hpx_arr(x)) == int(np.count_nonzero(x))


def test_count_nonzero_axis_not_supported():
    a = hpx_arr(np.arange(6, dtype=np.float64).reshape(2, 3))
    with pytest.raises((ValueError, NotImplementedError)):
        hpx.count_nonzero(a, axis=0)


# ===========================================================================
# G. Method forms (a.sqrt(), a.mean(), ...) match the free functions.
# ===========================================================================

def test_method_forms():
    x = np.array([1.0, 4.0, 9.0], dtype=np.float64)
    a = hpx_arr(x)
    np.testing.assert_allclose(np_out(a.sqrt()), np.sqrt(x), rtol=1e-12)
    np.testing.assert_allclose(np_out(a.negative()), -x)
    assert a.mean() == pytest.approx(float(np.mean(x)))
    assert a.prod() == pytest.approx(float(np.prod(x)))
    assert a.any() is True
    assert a.all() is True
    assert a.count_nonzero() == 3
    b = hpx_arr(np.array([2.0, 2.0, 2.0], dtype=np.float64))
    np.testing.assert_allclose(np_out(a.maximum(b)), np.maximum(x, [2, 2, 2]))
