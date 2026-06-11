"""Stage 1 N-D foundation tests: N-D constructors, multi-index get/set, shape/ndim/size.

Design constraints:
  - NO numpy round-trip for N-D arrays (that is stage 2).
  - All 1-D backward-compat paths are exercised here too.
  - 100% branch coverage of the new Python wrappers (zeros/ones/full with tuples)
    and the C++ multi-index get/set paths.
"""
import pytest

import hpxpy as hpx


# ---------------------------------------------------------------------------
# N-D constructors: zeros / full / ones
# ---------------------------------------------------------------------------

def test_zeros_2d_shape():
    a = hpx.zeros((3, 4))
    assert a.shape == (3, 4)
    assert a.ndim == 2
    assert a.size == 12


def test_zeros_2d_values():
    a = hpx.zeros((2, 5))
    for i in range(2):
        for j in range(5):
            assert a[i, j] == 0.0


def test_zeros_3d_shape():
    a = hpx.zeros((2, 3, 4))
    assert a.shape == (2, 3, 4)
    assert a.ndim == 3
    assert a.size == 24


def test_zeros_list_shape():
    # list is also accepted
    a = hpx.zeros([2, 3])
    assert a.shape == (2, 3)
    assert a.ndim == 2
    assert a.size == 6


def test_full_2d():
    a = hpx.full((2, 3), 7.0)
    assert a.shape == (2, 3)
    assert a.ndim == 2
    assert a.size == 6
    for i in range(2):
        for j in range(3):
            assert a[i, j] == 7.0


def test_full_3d():
    a = hpx.full((2, 3, 4), 5.0)
    assert a.shape == (2, 3, 4)
    assert a.ndim == 3
    assert a.size == 24
    assert a[0, 0, 0] == 5.0
    assert a[1, 2, 3] == 5.0


def test_full_list_shape():
    a = hpx.full([3, 3], 2.5)
    assert a.shape == (3, 3)
    assert a[2, 2] == 2.5


def test_ones_2d():
    a = hpx.ones((4, 5))
    assert a.shape == (4, 5)
    assert a.ndim == 2
    assert a.size == 20
    for i in range(4):
        for j in range(5):
            assert a[i, j] == 1.0


def test_ones_3d():
    a = hpx.ones((2, 2, 2))
    assert a.shape == (2, 2, 2)
    assert a.size == 8
    assert a[1, 1, 1] == 1.0


def test_ones_list_shape():
    a = hpx.ones([2, 4])
    assert a.shape == (2, 4)
    assert a[0, 0] == 1.0


# ---------------------------------------------------------------------------
# Multi-index get/set: 2-D
# ---------------------------------------------------------------------------

def test_2d_setitem_getitem():
    a = hpx.zeros((3, 4))
    a[1, 2] = 42.0
    assert a[1, 2] == 42.0


def test_2d_negative_index():
    a = hpx.zeros((3, 4))
    a[-1, -1] = 99.0
    assert a[2, 3] == 99.0


def test_2d_all_elements():
    a = hpx.zeros((3, 4))
    for i in range(3):
        for j in range(4):
            a[i, j] = float(i * 4 + j)
    for i in range(3):
        for j in range(4):
            assert a[i, j] == float(i * 4 + j)


# ---------------------------------------------------------------------------
# Row-major layout verification: set via multi-index, confirm via flat 1-D view
# ---------------------------------------------------------------------------

def test_row_major_layout_2d():
    """Row-major: element [i,j] lives at offset i*ncols + j in the flat buffer."""
    nrows, ncols = 3, 4
    a = hpx.zeros((nrows, ncols))
    # Write known values via multi-index: a[i,j] = i*ncols + j
    for i in range(nrows):
        for j in range(ncols):
            a[i, j] = float(i * ncols + j)
    # Verify the total flat sum: 0+1+...+11 = 66
    assert a.sum() == pytest.approx(sum(range(nrows * ncols)), rel=1e-14)
    # Spot-check row-major ordering via multi-index reads:
    # a[r, c] should equal r*ncols + c
    assert a[0, 0] == 0.0
    assert a[0, 3] == 3.0   # last element of row 0
    assert a[1, 0] == 4.0   # first element of row 1
    assert a[2, 3] == 11.0  # last element (2*4+3 = 11)
    # Verify that a[i,j] and a[i, j+1] differ by 1 (contiguous within a row)
    for i in range(nrows):
        for j in range(ncols - 1):
            assert a[i, j + 1] - a[i, j] == pytest.approx(1.0, rel=1e-14)
    # Verify that a[i+1,j] - a[i,j] == ncols (stride between rows)
    for i in range(nrows - 1):
        for j in range(ncols):
            assert a[i + 1, j] - a[i, j] == pytest.approx(float(ncols), rel=1e-14)


def test_row_major_layout_3d():
    """3-D row-major: element [i,j,k] at offset i*d1*d2 + j*d2 + k."""
    d0, d1, d2 = 2, 3, 4
    a = hpx.zeros((d0, d1, d2))
    for i in range(d0):
        for j in range(d1):
            for k in range(d2):
                a[i, j, k] = float(i * d1 * d2 + j * d2 + k)
    # Verify sum
    n = d0 * d1 * d2
    assert a.sum() == pytest.approx(sum(range(n)), rel=1e-14)
    # Spot-checks
    assert a[0, 0, 0] == 0.0
    assert a[1, 2, 3] == float(1 * 3 * 4 + 2 * 4 + 3)  # = 23
    assert a[0, 1, 2] == float(0 * 3 * 4 + 1 * 4 + 2)  # = 6


# ---------------------------------------------------------------------------
# Multi-index get/set: 3-D
# ---------------------------------------------------------------------------

def test_3d_setitem_getitem():
    a = hpx.zeros((2, 3, 4))
    a[1, 2, 3] = 77.0
    assert a[1, 2, 3] == 77.0


def test_3d_negative_indices():
    a = hpx.zeros((2, 3, 4))
    a[-1, -1, -1] = 55.0
    assert a[1, 2, 3] == 55.0
    assert a[-1, -1, -1] == 55.0


# ---------------------------------------------------------------------------
# Out-of-range raises IndexError
# ---------------------------------------------------------------------------

def test_2d_oob_row():
    a = hpx.zeros((3, 4))
    with pytest.raises(IndexError):
        _ = a[3, 0]


def test_2d_oob_col():
    a = hpx.zeros((3, 4))
    with pytest.raises(IndexError):
        _ = a[0, 4]


def test_2d_oob_negative():
    a = hpx.zeros((3, 4))
    with pytest.raises(IndexError):
        _ = a[-4, 0]


def test_3d_oob():
    a = hpx.zeros((2, 3, 4))
    with pytest.raises(IndexError):
        _ = a[2, 0, 0]


def test_2d_setitem_oob():
    a = hpx.zeros((3, 4))
    with pytest.raises(IndexError):
        a[3, 0] = 1.0


# ---------------------------------------------------------------------------
# Tuple with a slice raises NotImplementedError (stage 6 boundary)
# ---------------------------------------------------------------------------

def test_nd_slice_key_raises_getitem():
    a = hpx.zeros((3, 4))
    with pytest.raises((NotImplementedError, TypeError)):
        _ = a[0:2, 1]


def test_nd_slice_key_raises_setitem():
    a = hpx.zeros((3, 4))
    with pytest.raises((NotImplementedError, TypeError)):
        a[0:2, 1] = 0.0


def test_nd_slice_all_slices_raises():
    a = hpx.zeros((3, 4))
    with pytest.raises((NotImplementedError, TypeError)):
        _ = a[0:2, 0:3]


# ---------------------------------------------------------------------------
# Wrong-length tuple raises IndexError
# ---------------------------------------------------------------------------

def test_tuple_wrong_ndim_get():
    a = hpx.zeros((3, 4))
    with pytest.raises(IndexError):
        _ = a[0, 1, 2]  # 3 indices for ndim=2


def test_tuple_wrong_ndim_set():
    a = hpx.zeros((3, 4))
    with pytest.raises(IndexError):
        a[0,] = 1.0  # 1 index for ndim=2


# ---------------------------------------------------------------------------
# 1-D backward-compat: zeros/full/arange/indexing/ndim/shape still work
# ---------------------------------------------------------------------------

def test_1d_zeros_backward():
    a = hpx.zeros(10)
    assert a.ndim == 1
    assert a.shape == (10,)
    assert a.size == 10
    assert a[0] == 0.0


def test_1d_full_backward():
    a = hpx.full(5, 3.0)
    assert a.ndim == 1
    assert a.shape == (5,)
    assert a[4] == 3.0


def test_1d_arange_backward():
    a = hpx.arange(6)
    assert a.ndim == 1
    assert a.shape == (6,)
    assert a[5] == 5.0


def test_1d_ones_backward():
    a = hpx.ones(7)
    assert a.ndim == 1
    assert a.shape == (7,)
    assert a.size == 7
    assert a[0] == 1.0
    assert a[6] == 1.0


def test_1d_shape_is_tuple():
    a = hpx.zeros(5)
    assert isinstance(a.shape, tuple)
    assert a.shape == (5,)


def test_1d_stride_backward():
    a = hpx.arange(10)
    assert a.stride == 1
    v = a[::2]
    assert v.stride == 2


def test_1d_sum_backward():
    a = hpx.arange(5)
    assert a.sum() == pytest.approx(10.0, rel=1e-14)


# ---------------------------------------------------------------------------
# N-D reductions work flat (the contiguous fast-path is unchanged)
# ---------------------------------------------------------------------------

def test_2d_sum():
    a = hpx.full((3, 4), 2.0)
    assert a.sum() == pytest.approx(24.0, rel=1e-14)


def test_2d_min():
    a = hpx.zeros((2, 3))
    a[0, 0] = -5.0
    assert a.min() == pytest.approx(-5.0, rel=1e-14)


def test_2d_max():
    a = hpx.zeros((2, 3))
    a[1, 2] = 10.0
    assert a.max() == pytest.approx(10.0, rel=1e-14)


def test_3d_sum():
    a = hpx.ones((2, 3, 4))
    assert a.sum() == pytest.approx(24.0, rel=1e-14)


def test_to_numpy_nd_raises():
    """N-D to_numpy/__array__ refuses (no silent flat view) until the bridge lands (stage 2)."""
    a = hpx.zeros((2, 3))
    with pytest.raises(TypeError):
        a.to_numpy()
    with pytest.raises(TypeError):
        hpx.to_numpy(a)
