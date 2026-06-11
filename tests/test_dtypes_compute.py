"""Phase A.2 stage A2.2: dtype-dispatched core compute kernels.

The core element-wise kernels — sum / min / max / dot, add / sub / mul / div,
the scalar ops, and the broadcast path — now work for float32 and int64 (not
just float64), via a per-dtype dispatch over the element type T. NumPy is the
oracle throughout.

Still guarded to non-float64 (deferred to later A2.x stages): the axis
reductions, matmul, copy, sort/is_sorted, cumsum. int64 true-division is
guarded (would silently floor); a non-integral float scalar on an int64 array
raises (no automatic int<->float promotion yet).
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


# ---------------------------------------------------------------------------
# Reductions: sum / min / max / dot vs numpy (f32 AND i64).
# The int64 cases include negative values so min/max exercise the
# lowest()/max() identities — a wrong ±inf->0 seed would give 0.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dt", [np.float32, np.int64])
def test_sum(dt):
    n = np.array([3, -7, 5, -1, 9, 2], dtype=dt)
    a = hpx_arr(n)
    assert a.sum() == n.sum()


@pytest.mark.parametrize("dt", [np.float32, np.int64])
def test_min_max_with_negatives(dt):
    # All-positive: a wrong max seed (lowest) still works, but min seed matters.
    p = np.array([3, 7, 5, 1, 9, 2], dtype=dt)
    ap = hpx_arr(p)
    assert ap.min() == p.min()
    assert ap.max() == p.max()
    # All-negative: a wrong ±inf->0 seed for integers would give 0 here.
    neg = np.array([-3, -7, -5, -1, -9, -2], dtype=dt)
    an = hpx_arr(neg)
    assert an.min() == neg.min()    # -9, NOT 0
    assert an.max() == neg.max()    # -1, NOT 0


def test_int64_min_max_all_negative_not_zero():
    """Regression: the int identity trap. lowest()/max() seeds, not ±inf->0."""
    neg = np.array([-5, -100, -3], dtype=np.int64)
    a = hpx_arr(neg)
    assert a.min() == -100
    assert a.max() == -3


@pytest.mark.parametrize("dt", [np.float32, np.int64])
def test_dot(dt):
    x = np.array([1, 2, 3, 4, 5], dtype=dt)
    y = np.array([6, -7, 8, -9, 10], dtype=dt)
    a = hpx_arr(x)
    b = hpx_arr(y)
    assert a.dot(b) == x.dot(y)


# ---------------------------------------------------------------------------
# Element-wise binary ops + scalar ops vs numpy (f32 AND i64).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dt", [np.float32, np.int64])
@pytest.mark.parametrize("op", ["add", "sub", "mul"])
def test_binary_same_shape(dt, op):
    x = np.array([10, 20, 30, 40], dtype=dt)
    y = np.array([1, 2, 3, 4], dtype=dt)
    a = hpx_arr(x)
    b = hpx_arr(y)
    r = {"add": a + b, "sub": a - b, "mul": a * b}[op]
    e = {"add": x + y, "sub": x - y, "mul": x * y}[op]
    np.testing.assert_array_equal(np_out(r), e)


@pytest.mark.parametrize("dt", [np.float32, np.int64])
def test_scalar_add_mul(dt):
    x = np.array([1, 2, 3, 4], dtype=dt)
    a = hpx_arr(x)
    np.testing.assert_array_equal(np_out(a + 2), x + 2)
    np.testing.assert_array_equal(np_out(a * 3), x * 3)


@pytest.mark.parametrize("dt", [np.float32, np.int64])
def test_broadcast_3x4_plus_4(dt):
    x = np.arange(12, dtype=dt).reshape(3, 4)
    y = np.array([10, 20, 30, 40], dtype=dt)
    r = hpx_arr(x) + hpx_arr(y)
    assert r.shape == (3, 4)
    np.testing.assert_array_equal(np_out(r), x + y)


# ---------------------------------------------------------------------------
# Result dtype is preserved (operand dtype; no promotion).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dt", [np.float32, np.int64, np.float64])
def test_result_dtype_preserved(dt):
    a = hpx_arr(np.array([1, 2, 3], dtype=dt))
    b = hpx_arr(np.array([4, 5, 6], dtype=dt))
    assert (a + b).dtype == np.dtype(dt)
    assert (a * b).dtype == np.dtype(dt)
    assert (a + 2).dtype == np.dtype(dt)


# ---------------------------------------------------------------------------
# Same-dtype enforcement: no automatic type promotion yet.
# ---------------------------------------------------------------------------

def test_f32_plus_f64_raises():
    a = hpx.ones(3, dtype=np.float32)
    b = hpx.ones(3, dtype=np.float64)
    with pytest.raises((TypeError, ValueError)):
        a + b


def test_i64_plus_f64_raises():
    a = hpx.from_numpy(np.arange(3, dtype=np.int64))
    b = hpx.from_numpy(np.arange(3, dtype=np.float64))
    with pytest.raises((TypeError, ValueError)):
        a + b


def test_dot_mismatched_dtype_raises():
    a = hpx.from_numpy(np.arange(3, dtype=np.float32))
    b = hpx.from_numpy(np.arange(3, dtype=np.float64))
    with pytest.raises((TypeError, ValueError)):
        a.dot(b)


# ---------------------------------------------------------------------------
# int64 + float scalar must NOT silently truncate -> raises. Integer scalar OK.
# ---------------------------------------------------------------------------

def test_int64_plus_float_scalar_raises():
    a = hpx.from_numpy(np.arange(4, dtype=np.int64))
    with pytest.raises((TypeError, ValueError)):
        a + 2.5
    with pytest.raises((TypeError, ValueError)):
        a * 1.5


def test_int64_plus_int_scalar_works():
    x = np.arange(4, dtype=np.int64)
    a = hpx_arr(x)
    np.testing.assert_array_equal(np_out(a + 2), x + 2)
    np.testing.assert_array_equal(np_out(a * 3), x * 3)
    # A float-valued-but-integral scalar (2.0) is fine on int64.
    np.testing.assert_array_equal(np_out(a + 2.0), x + 2)


def test_float_array_plus_float_scalar_works():
    # Non-integral scalars are fine on float dtypes (only int64 rejects them).
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    a = hpx_arr(x)
    np.testing.assert_allclose(np_out(a + 2.5), x + np.float32(2.5))


# ---------------------------------------------------------------------------
# Division: float dtypes divide vs numpy; int64 true-division is guarded.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dt", [np.float32, np.float64])
def test_div_float(dt):
    x = np.array([1, 2, 3, 4], dtype=dt)
    y = np.array([2, 4, 8, 16], dtype=dt)
    a = hpx_arr(x)
    b = hpx_arr(y)
    np.testing.assert_allclose(np_out(a / b), x / y, rtol=1e-6)
    np.testing.assert_allclose(np_out(a / 2), x / 2, rtol=1e-6)


def test_int64_div_guarded():
    a = hpx.from_numpy(np.arange(1, 5, dtype=np.int64))
    b = hpx.from_numpy(np.arange(1, 5, dtype=np.int64))
    with pytest.raises(RuntimeError):
        a / b
    with pytest.raises(RuntimeError):
        a / 2


# ---------------------------------------------------------------------------
# Strided (non-contiguous) inputs go through the flat_to_offset else-branches.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dt", [np.float32, np.int64])
def test_strided_reduction_and_binary(dt):
    x = np.arange(12, dtype=dt).reshape(3, 4)
    a = hpx_arr(x)
    at = a.T                                  # non-contiguous view
    xt = x.T
    assert at.sum() == xt.sum()
    assert at.min() == xt.min()
    assert at.max() == xt.max()
    # binary on a transposed (non-contiguous) operand uses the broadcast/strided path
    r = at + hpx_arr(np.ascontiguousarray(xt))
    np.testing.assert_array_equal(np_out(r), xt + xt)
