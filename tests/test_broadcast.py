"""Stage 4 N-D broadcasting: NumPy-compatible element-wise binary ops.

Tests:
  - Core broadcast shapes: (3,4)+(4,), (3,1)+(1,4), etc.
  - All four operators via parametrize
  - Regression: same-shape contiguous fast path + non-contiguous (transposed)
  - Error cases: incompatible shapes raise ValueError
  - Edge cases: (3,4)+(1,1), empty (0,4)+(4,)
"""
import pytest
import numpy as np
import hpxpy as hpx


def hpx_arr(np_arr):
    """Import a numpy array into hpxpy (copy=True for NUMA-aware allocation)."""
    return hpx.from_numpy(np.ascontiguousarray(np_arr, dtype=np.float64), copy=True)


def np_from_hpx(a):
    """Export an hpxpy Array to a contiguous numpy array."""
    return np.asarray(a).copy()


# ---------------------------------------------------------------------------
# Broadcast shape tests
# ---------------------------------------------------------------------------

def test_broadcast_3x4_plus_4():
    """(3,4) + (4,) -> (3,4)"""
    a = hpx_arr(np.arange(12.0).reshape(3, 4))
    b = hpx_arr(np.array([1.0, 2.0, 3.0, 4.0]))
    result = a + b
    expected = np.arange(12.0).reshape(3, 4) + np.array([1.0, 2.0, 3.0, 4.0])
    assert result.shape == (3, 4)
    np.testing.assert_allclose(np_from_hpx(result), expected)


def test_broadcast_3x1_plus_1x4():
    """(3,1) + (1,4) -> (3,4)"""
    a = hpx_arr(np.arange(3.0).reshape(3, 1))
    b = hpx_arr(np.arange(4.0).reshape(1, 4))
    result = a + b
    expected = np.arange(3.0).reshape(3, 1) + np.arange(4.0).reshape(1, 4)
    assert result.shape == (3, 4)
    np.testing.assert_allclose(np_from_hpx(result), expected)


def test_broadcast_3x4_plus_1():
    """(3,4) + (1,) -> (3,4)"""
    a = hpx_arr(np.arange(12.0).reshape(3, 4))
    b = hpx_arr(np.array([10.0]))
    result = a + b
    expected = np.arange(12.0).reshape(3, 4) + 10.0
    assert result.shape == (3, 4)
    np.testing.assert_allclose(np_from_hpx(result), expected)


def test_broadcast_commutative_4_plus_3x4():
    """(4,) + (3,4) -> (3,4)  (commutative)"""
    a = hpx_arr(np.array([1.0, 2.0, 3.0, 4.0]))
    b = hpx_arr(np.arange(12.0).reshape(3, 4))
    result = a + b
    expected = np.array([1.0, 2.0, 3.0, 4.0]) + np.arange(12.0).reshape(3, 4)
    assert result.shape == (3, 4)
    np.testing.assert_allclose(np_from_hpx(result), expected)


def test_broadcast_2x3x4_plus_4():
    """(2,3,4) + (4,) -> (2,3,4)"""
    a = hpx_arr(np.arange(24.0).reshape(2, 3, 4))
    b = hpx_arr(np.array([1.0, 2.0, 3.0, 4.0]))
    result = a + b
    expected = np.arange(24.0).reshape(2, 3, 4) + np.array([1.0, 2.0, 3.0, 4.0])
    assert result.shape == (2, 3, 4)
    np.testing.assert_allclose(np_from_hpx(result), expected)


def test_broadcast_2x3x1_mul_2x1x4():
    """(2,3,1) * (2,1,4) -> (2,3,4)"""
    a = hpx_arr(np.arange(6.0).reshape(2, 3, 1))
    b = hpx_arr(np.arange(8.0).reshape(2, 1, 4))
    result = a * b
    expected = np.arange(6.0).reshape(2, 3, 1) * np.arange(8.0).reshape(2, 1, 4)
    assert result.shape == (2, 3, 4)
    np.testing.assert_allclose(np_from_hpx(result), expected)


def test_broadcast_1x4_sub_3x4():
    """(1,4) - (3,4) -> (3,4)"""
    a = hpx_arr(np.array([[10.0, 20.0, 30.0, 40.0]]))
    b = hpx_arr(np.arange(12.0).reshape(3, 4))
    result = a - b
    expected = np.array([[10.0, 20.0, 30.0, 40.0]]) - np.arange(12.0).reshape(3, 4)
    assert result.shape == (3, 4)
    np.testing.assert_allclose(np_from_hpx(result), expected)


# ---------------------------------------------------------------------------
# All four operators (parametrized)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("op", ["add", "sub", "mul", "div"])
def test_all_ops_broadcast(op):
    """All four operators broadcast (3,1) op (1,4) -> (3,4). div uses positive divisor."""
    a_np = np.arange(1.0, 4.0).reshape(3, 1)   # [[1],[2],[3]]
    b_np = np.arange(1.0, 5.0).reshape(1, 4)   # [[1,2,3,4]]  (positive for div)
    a = hpx_arr(a_np)
    b = hpx_arr(b_np)
    result = getattr(a, op)(b)
    ops_np = {
        "add": np.add,
        "sub": np.subtract,
        "mul": np.multiply,
        "div": np.true_divide,
    }
    expected = ops_np[op](a_np, b_np)
    assert result.shape == (3, 4)
    np.testing.assert_allclose(np_from_hpx(result), expected)


# ---------------------------------------------------------------------------
# Regression: same-shape paths
# ---------------------------------------------------------------------------

def test_regression_same_shape_contiguous():
    """Same-shape contiguous (3,4)+(3,4) uses fast path."""
    a_np = np.arange(12.0).reshape(3, 4)
    b_np = np.arange(12.0, 24.0).reshape(3, 4)
    a = hpx_arr(a_np)
    b = hpx_arr(b_np)
    result = a + b
    expected = a_np + b_np
    assert result.shape == (3, 4)
    np.testing.assert_allclose(np_from_hpx(result), expected)


def test_regression_transposed_same_shape():
    """Non-contiguous same-shape: a.T + b.T (shapes equal)."""
    a_np = np.arange(12.0).reshape(3, 4)
    b_np = np.arange(12.0, 24.0).reshape(3, 4)
    a = hpx_arr(a_np)
    b = hpx_arr(b_np)
    at = a.T
    bt = b.T
    result = at + bt
    expected = a_np.T + b_np.T
    assert result.shape == (4, 3)
    np.testing.assert_allclose(np_from_hpx(result), expected)


# ---------------------------------------------------------------------------
# Error cases: incompatible shapes raise ValueError
# ---------------------------------------------------------------------------

def test_error_incompatible_3x4_plus_2():
    """(3,4) + (2,) raises ValueError."""
    a = hpx_arr(np.zeros((3, 4)))
    b = hpx_arr(np.zeros(2))
    with pytest.raises(ValueError):
        _ = a + b


def test_error_incompatible_3x4_plus_3():
    """(3,4) + (3,) raises ValueError."""
    a = hpx_arr(np.zeros((3, 4)))
    b = hpx_arr(np.zeros(3))
    with pytest.raises(ValueError):
        _ = a + b


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_broadcast_3x4_plus_1x1():
    """(3,4) + (1,1) -> (3,4)"""
    a = hpx_arr(np.arange(12.0).reshape(3, 4))
    b = hpx_arr(np.array([[5.0]]))
    result = a + b
    expected = np.arange(12.0).reshape(3, 4) + 5.0
    assert result.shape == (3, 4)
    np.testing.assert_allclose(np_from_hpx(result), expected)


def test_broadcast_empty_0x4_plus_4():
    """Empty (0,4) + (4,) -> (0,4)"""
    a = hpx_arr(np.zeros((0, 4)))
    b = hpx_arr(np.array([1.0, 2.0, 3.0, 4.0]))
    result = a + b
    assert result.shape == (0, 4)
    assert result.size == 0
