"""Phase A.2 stage A2.4: NumPy type promotion for mixed-dtype ops.

Through A2.3 a dtype mismatch (or an int64 array with a float scalar, or any int64
true-division) RAISED. A2.4 relaxes those to numpy-faithful promotion: mixed-dtype
element-wise ops, mixed matmul/dot, a float scalar on an int64 array, and integer
true-division all PROMOTE to the numpy result dtype instead of raising.

NumPy is the oracle throughout: every promoted result is checked for BOTH values
(assert_array_equal / assert_allclose) AND dtype (== ``np.result_type(...)``). The
key non-obvious rule is int64 ⊕ float32 -> float64 (NOT float32).

The same-dtype fast path is unchanged (zero-penalty) and is re-checked here so a
regression in the common case is caught alongside the promotion behavior.
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
# A. promote() lattice — the exact numpy result dtype for our three types.
#    The load-bearing case is int64 ⊕ float32 -> float64 (NOT float32).
# ===========================================================================

_DTYPES = [np.float64, np.float32, np.int64]


@pytest.mark.parametrize("da", _DTYPES)
@pytest.mark.parametrize("db", _DTYPES)
def test_promote_matches_numpy_result_type(da, db):
    x = np.array([1, 2, 3, 4], dtype=da)
    y = np.array([5, 6, 7, 8], dtype=db)
    r = hpx_arr(x) + hpx_arr(y)
    assert r.dtype == np.result_type(da, db)
    np.testing.assert_array_equal(np_out(r), x + y)


def test_int64_plus_float32_is_float64():
    """The headline rule: int64 ⊕ float32 promotes to float64, NOT float32."""
    x = np.array([1, 2, 3], dtype=np.int64)
    y = np.array([0.5, 1.5, 2.5], dtype=np.float32)
    r = hpx_arr(x) + hpx_arr(y)
    assert r.dtype == np.dtype(np.float64) == np.result_type(np.int64, np.float32)
    np.testing.assert_array_equal(np_out(r), x + y)


# ===========================================================================
# B. Mixed-dtype element-wise ops — values + result dtype vs numpy.
# ===========================================================================

@pytest.mark.parametrize(
    "da,db",
    [
        (np.int64, np.float64),
        (np.float32, np.float64),
        (np.int64, np.float32),   # -> float64
        (np.float64, np.int64),
        (np.float64, np.float32),
        (np.float32, np.int64),   # -> float64
    ],
)
@pytest.mark.parametrize("op", ["add", "sub", "mul"])
def test_mixed_binary(da, db, op):
    x = np.array([10, 20, 30, 40], dtype=da)
    y = np.array([1, 2, 3, 4], dtype=db)
    a = hpx_arr(x)
    b = hpx_arr(y)
    r = {"add": a + b, "sub": a - b, "mul": a * b}[op]
    e = {"add": x + y, "sub": x - y, "mul": x * y}[op]
    assert r.dtype == e.dtype == np.result_type(da, db)
    np.testing.assert_array_equal(np_out(r), e)


def test_mixed_i64_times_f32_dtype_and_values():
    x = np.array([1, 2, 3, 4], dtype=np.int64)
    y = np.array([1.5, 2.5, 0.5, 4.0], dtype=np.float32)
    r = hpx_arr(x) * hpx_arr(y)
    assert r.dtype == np.dtype(np.float64)
    np.testing.assert_array_equal(np_out(r), x * y)


# ===========================================================================
# C. Mixed-dtype WITH broadcasting (the flat_to_offset path).
# ===========================================================================

def test_mixed_broadcast_3x4_plus_4():
    x = np.arange(12, dtype=np.int64).reshape(3, 4)
    y = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
    r = hpx_arr(x) + hpx_arr(y)
    assert r.shape == (3, 4)
    assert r.dtype == np.dtype(np.float64) == np.result_type(x, y)
    np.testing.assert_array_equal(np_out(r), x + y)


def test_mixed_broadcast_i64_plus_f32_is_f64():
    x = np.arange(12, dtype=np.int64).reshape(3, 4)
    y = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32)
    r = hpx_arr(x) + hpx_arr(y)
    assert r.dtype == np.dtype(np.float64)
    np.testing.assert_array_equal(np_out(r), x + y)


def test_mixed_broadcast_strided_operand():
    """A transposed (non-contiguous) operand of a different dtype promotes too."""
    x = np.arange(12, dtype=np.int64).reshape(3, 4)
    y = np.arange(12, dtype=np.float64).reshape(3, 4)
    at = hpx_arr(x).T                  # non-contiguous int64 view
    r = at + hpx_arr(np.ascontiguousarray(y.T))
    e = x.T + y.T
    assert r.dtype == np.dtype(np.float64)
    np.testing.assert_array_equal(np_out(r), e)


# ===========================================================================
# D. Scalar promotion (numpy promotes by the Python scalar's TYPE, not value).
#    A Python float scalar on int64 -> float64; an int scalar keeps the dtype.
# ===========================================================================

def test_arange_times_float_scalar_is_float64():
    a = hpx.arange(5)                  # int64 (Wave 2)
    e = np.arange(5)
    r = a * 2.0
    assert r.dtype == np.dtype(np.float64) == (e * 2.0).dtype
    np.testing.assert_array_equal(np_out(r), e * 2.0)


def test_arange_plus_float_scalar_is_float64():
    a = hpx.arange(5)
    e = np.arange(5)
    r = a + 1.5
    assert r.dtype == np.dtype(np.float64) == (e + 1.5).dtype
    np.testing.assert_array_equal(np_out(r), e + 1.5)


def test_int64_plus_int_scalar_stays_int64():
    x = np.arange(5, dtype=np.int64)
    a = hpx_arr(x)
    r = a + 2                          # Python int scalar -> stays int64
    assert r.dtype == np.dtype(np.int64) == (x + 2).dtype
    np.testing.assert_array_equal(np_out(r), x + 2)


def test_float32_times_float_scalar_stays_float32():
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    a = hpx_arr(x)
    r = a * 2.0
    assert r.dtype == np.dtype(np.float32)
    np.testing.assert_allclose(np_out(r), x * np.float32(2.0), rtol=1e-6)


def test_scalar_sub_div_promote_and_reflected():
    a = hpx.arange(5)                  # int64
    e = np.arange(5)
    # int64 - float scalar -> float64
    assert (a - 1.5).dtype == np.dtype(np.float64)
    np.testing.assert_array_equal(np_out(a - 1.5), e - 1.5)
    # reflected: float scalar - int64 -> float64
    np.testing.assert_array_equal(np_out(1.5 - a), 1.5 - e)
    assert np_out(1.5 - a).dtype == np.dtype(np.float64)


# ===========================================================================
# E. True division — ALWAYS float (numpy true_divide). int64 / ... -> float64.
# ===========================================================================

def test_arange_div_int_scalar_is_float64():
    a = hpx.arange(6)                  # int64
    e = np.arange(6)
    r = a / 2                          # numpy: int64 / 2 -> float64
    assert r.dtype == np.dtype(np.float64) == (e / 2).dtype
    np.testing.assert_allclose(np_out(r), e / 2, rtol=1e-12)


def test_int64_div_int64_array_is_float64():
    x = np.array([1, 2, 3, 7], dtype=np.int64)
    y = np.array([2, 4, 2, 2], dtype=np.int64)
    r = hpx_arr(x) / hpx_arr(y)
    assert r.dtype == np.dtype(np.float64) == (x / y).dtype
    np.testing.assert_allclose(np_out(r), x / y, rtol=1e-12)


def test_int64_div_float_array_is_float64():
    x = np.array([1, 2, 3, 4], dtype=np.int64)
    y = np.array([2.0, 4.0, 8.0, 16.0], dtype=np.float64)
    r = hpx_arr(x) / hpx_arr(y)
    assert r.dtype == np.dtype(np.float64)
    np.testing.assert_allclose(np_out(r), x / y, rtol=1e-12)


def test_float32_div_float32_stays_float32():
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    y = np.array([2.0, 4.0, 8.0, 16.0], dtype=np.float32)
    r = hpx_arr(x) / hpx_arr(y)
    assert r.dtype == np.dtype(np.float32) == (x / y).dtype
    np.testing.assert_allclose(np_out(r), x / y, rtol=1e-6)


def test_float64_div_float64_unchanged():
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    y = np.array([2.0, 4.0, 8.0, 16.0], dtype=np.float64)
    r = hpx_arr(x) / hpx_arr(y)
    assert r.dtype == np.dtype(np.float64)
    np.testing.assert_allclose(np_out(r), x / y, rtol=1e-12)


def test_reflected_div_promotes():
    a = hpx.arange(1, 6)               # int64 [1..5]
    e = np.arange(1, 6)
    r = 2 / a                          # numpy: 2 / int64 -> float64
    assert r.dtype == np.dtype(np.float64)
    np.testing.assert_allclose(np_out(r), 2 / e, rtol=1e-12)


# ===========================================================================
# F. Same-dtype fast path is UNCHANGED (zero-penalty regression guard).
# ===========================================================================

@pytest.mark.parametrize("dt", [np.float64, np.float32, np.int64])
def test_same_dtype_add_mul_unchanged(dt):
    x = np.array([10, 20, 30, 40], dtype=dt)
    y = np.array([1, 2, 3, 4], dtype=dt)
    a = hpx_arr(x)
    b = hpx_arr(y)
    assert (a + b).dtype == np.dtype(dt)
    assert (a * b).dtype == np.dtype(dt)
    np.testing.assert_array_equal(np_out(a + b), x + y)
    np.testing.assert_array_equal(np_out(a * b), x * y)


def test_same_dtype_int_div_stays_int():
    """i64 // i64 (floor division) is the Wave 1 integer op — UNCHANGED, stays int."""
    x = np.array([7, 8, 9, 10], dtype=np.int64)
    y = np.array([2, 3, 2, 4], dtype=np.int64)
    r = hpx_arr(x) // hpx_arr(y)
    assert r.dtype == np.dtype(np.int64) == (x // y).dtype
    np.testing.assert_array_equal(np_out(r), x // y)


# ===========================================================================
# G. Mixed matmul / dot — promote to the numpy result dtype.
# ===========================================================================

def test_mixed_matmul_i64_f64_is_float64():
    A = np.arange(1, 7, dtype=np.int64).reshape(2, 3)
    B = np.arange(1, 7, dtype=np.float64).reshape(3, 2)
    r = hpx_arr(A) @ hpx_arr(B)
    assert r.dtype == np.dtype(np.float64) == (A @ B).dtype
    np.testing.assert_allclose(np_out(r), A @ B, rtol=1e-12)


def test_mixed_matmul_i64_f32_is_float64():
    A = np.arange(1, 7, dtype=np.int64).reshape(2, 3)
    B = np.arange(1, 7, dtype=np.float32).reshape(3, 2)
    r = hpx_arr(A) @ hpx_arr(B)
    assert r.dtype == np.dtype(np.float64)
    np.testing.assert_allclose(np_out(r), A @ B, rtol=1e-6)


def test_mixed_dot_i64_f64_is_float():
    x = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    y = np.array([6.0, -7.0, 8.0, -9.0, 10.0], dtype=np.float64)
    s = hpx_arr(x).dot(hpx_arr(y))
    assert s == pytest.approx(float(x.dot(y)), rel=1e-12)


def test_same_dtype_matmul_f64_unchanged():
    A = np.arange(1, 5, dtype=np.float64).reshape(2, 2)
    r = hpx_arr(A) @ hpx_arr(A)
    assert r.dtype == np.dtype(np.float64)
    np.testing.assert_allclose(np_out(r), A @ A, rtol=1e-12)
