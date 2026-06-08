"""M5a: CSR sparse matrix + SpMV (y = A @ x), checked against analytic values.
Correctness is verified via reductions of the result Array (no element access needed
beyond the existing a[i]).
"""
import pytest

import hpxpy as hpx


def _identity(n):
    # CSR identity: row_ptr = 0..n, col_idx = 0..n-1, values = 1.
    return hpx.csr_from(n, n, list(range(n + 1)), list(range(n)), [1.0] * n)


def test_csr_shape_and_nnz():
    a = hpx.laplacian_1d(1000)
    assert a.rows == 1000 and a.cols == 1000
    assert a.nnz == 3 * 1000 - 2          # tridiagonal: 3/row minus 2 missing ends


def test_identity_spmv_is_x():
    n = 1000
    x = hpx.arange(n)
    y = _identity(n) @ x                  # I @ x == x
    assert isinstance(y, hpx.Array)
    assert y.size == n
    # y - x == 0  (reuse element-wise + reduction)
    assert y.sub(x).dot(y.sub(x)) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("n", [3, 100, 10000])
def test_laplacian_times_ones(n):
    # A @ ones: interior rows sum (-1+2-1)=0; first/last row (2-1)=1. Total = 2.
    ones = hpx.full(n, 1.0)
    y = hpx.laplacian_1d(n) @ ones
    assert y.sum() == pytest.approx(2.0, abs=1e-9)


def test_laplacian_times_arange_interior_is_zero():
    n = 1000
    # interior row i: -(i-1) + 2i - (i+1) = 0. Endpoints are nonzero, so total sum is
    # the boundary contribution: row 0 = 2*0 - 1 = -1; row n-1 = 2(n-1) - (n-2) = n.
    y = hpx.laplacian_1d(n) @ hpx.arange(n)
    assert y.sum() == pytest.approx(-1.0 + n, abs=1e-9)


def test_hand_built_3x3():
    # A = [[2,0,1],[0,3,0],[4,0,5]], x = [1,2,3] -> y = [2*1+1*3, 3*2, 4*1+5*3] = [5,6,19]
    a = hpx.csr_from(3, 3, [0, 2, 3, 5], [0, 2, 1, 0, 2], [2.0, 1.0, 3.0, 4.0, 5.0])
    xv = hpx.full(3, 0.0)
    xv[0], xv[1], xv[2] = 1.0, 2.0, 3.0
    y = a @ xv
    assert y[0] == pytest.approx(5.0)
    assert y[1] == pytest.approx(6.0)
    assert y[2] == pytest.approx(19.0)


def test_spmv_size_mismatch_raises():
    with pytest.raises(ValueError):
        hpx.laplacian_1d(10) @ hpx.arange(11)


def test_free_function_spmv():
    n = 500
    assert hpx.spmv(_identity(n), hpx.arange(n)).sum() == pytest.approx(n * (n - 1) / 2)
