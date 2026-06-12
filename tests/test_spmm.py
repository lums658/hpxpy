"""M5b: SpMM (sparse x dense), C = A @ B, checked against analytic values and against
per-column consistency with SpMV.
"""
import pytest

import hpxpy as hpx


def _identity(n):
    return hpx.csr_from(n, n, list(range(n + 1)), list(range(n)), [1.0] * n)


def test_identity_spmm_is_b():
    # I @ B == B
    m, k = 50, 4
    flat = [float(i) for i in range(m * k)]
    b = hpx.dense_from(m, k, flat)
    c = _identity(m) @ b
    assert isinstance(c, hpx.DenseMatrix)
    assert c.rows == m and c.cols == k
    for i in (0, 1, m // 2, m - 1):
        for j in range(k):
            assert c.at(i, j) == pytest.approx(b.at(i, j))


def test_laplacian_times_ones_matrix():
    # Each column of (A @ ones) equals A @ ones-vector = [1, 0, ..., 0, 1].
    n, k = 100, 5
    b = hpx.dense_from(n, k, [1.0] * (n * k))
    c = hpx.laplacian_1d(n) @ b
    assert c.rows == n and c.cols == k
    for j in range(k):
        assert c.at(0, j) == pytest.approx(1.0)
        assert c.at(n - 1, j) == pytest.approx(1.0)
        assert c.at(n // 2, j) == pytest.approx(0.0)


def test_spmm_matches_per_column_spmv():
    # Column j of A @ B must equal A @ (column j of B).
    n, k = 200, 3
    a = hpx.laplacian_1d(n)
    # B column j = j+1 times arange  -> build row-major flat
    flat = []
    for i in range(n):
        for j in range(k):
            flat.append((j + 1) * float(i))
    b = hpx.dense_from(n, k, flat)
    c = a @ b
    for j in range(k):
        xj = hpx.arange(n, dtype="float64") * float(j + 1)      # column j of B as a vector
        yj = a @ xj                            # SpMV
        for i in (0, 1, n // 2, n - 1):
            assert c.at(i, j) == pytest.approx(yj[i], abs=1e-9)


def test_hand_built_spmm():
    # A = [[2,0,1],[0,3,0],[4,0,5]], B = [[1,10],[2,20],[3,30]]
    # C = A@B = [[2*1+1*3, 2*10+1*30],[3*2,3*20],[4*1+5*3,4*10+5*30]]
    #         = [[5, 50], [6, 60], [19, 190]]
    a = hpx.csr_from(3, 3, [0, 2, 3, 5], [0, 2, 1, 0, 2], [2.0, 1.0, 3.0, 4.0, 5.0])
    b = hpx.dense_from(3, 2, [1.0, 10.0, 2.0, 20.0, 3.0, 30.0])
    c = a @ b
    assert (c.at(0, 0), c.at(0, 1)) == pytest.approx((5.0, 50.0))
    assert (c.at(1, 0), c.at(1, 1)) == pytest.approx((6.0, 60.0))
    assert (c.at(2, 0), c.at(2, 1)) == pytest.approx((19.0, 190.0))


def test_spmm_size_mismatch_raises():
    with pytest.raises(ValueError):
        hpx.laplacian_1d(10) @ hpx.dense_zeros(11, 3)   # B.rows != A.cols


def test_dense_from_bad_length_raises():
    with pytest.raises(ValueError):
        hpx.dense_from(2, 3, [1.0, 2.0])                # need 6 values


def test_free_function_spmm():
    c = hpx.spmm(_identity(10), hpx.dense_from(10, 2, [1.0] * 20))
    assert c.rows == 10 and c.cols == 2
    assert c.at(3, 1) == pytest.approx(1.0)
