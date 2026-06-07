"""M2: reductions over the Array, checked against ANALYTIC values (no NumPy in the
data path). These also retroactively validate that arange/full fill correctly —
the content-correctness deferred from M1/the constructors PR.
"""
import pytest

import hpxpy as hpx


# --- sum --------------------------------------------------------------------

@pytest.mark.parametrize("n", [1, 2, 3, 1000, 1001, 1_000_000])
def test_sum_arange_is_triangular(n):
    # sum(0..n-1) = n(n-1)/2  (exact in float64 for these n)
    assert hpx.arange(n).sum() == pytest.approx(n * (n - 1) / 2, rel=1e-12)


@pytest.mark.parametrize("n,c", [(1000, 3.5), (1, 42.0), (123, -2.0)])
def test_sum_full_is_n_times_c(n, c):
    assert hpx.full(n, c).sum() == pytest.approx(n * c, rel=1e-12)


def test_sum_zeros_is_zero():
    assert hpx.zeros(1000).sum() == 0.0


def test_sum_empty_is_zero():
    assert hpx.arange(0).sum() == 0.0


def test_method_and_free_function_agree():
    a = hpx.arange(10_000)
    assert a.sum() == hpx.sum(a)


# --- min / max --------------------------------------------------------------

@pytest.mark.parametrize("n", [1, 2, 1000, 1_000_000])
def test_min_max_arange(n):
    a = hpx.arange(n)
    assert a.min() == 0.0
    assert a.max() == pytest.approx(float(n - 1))
    assert hpx.min(a) == a.min()
    assert hpx.max(a) == a.max()


def test_min_max_full_is_constant():
    a = hpx.full(500, 7.25)
    assert a.min() == pytest.approx(7.25)
    assert a.max() == pytest.approx(7.25)


def test_min_max_empty_raises():
    a = hpx.arange(0)
    with pytest.raises(ValueError):
        a.min()
    with pytest.raises(ValueError):
        a.max()


# --- dot --------------------------------------------------------------------

@pytest.mark.parametrize("n", [1, 2, 1000, 1001, 1_000_000])
def test_dot_arange_is_sum_of_squares(n):
    # dot(0..n-1, 0..n-1) = sum i^2 = (n-1)n(2n-1)/6  (exact in float64 for these n)
    expected = (n - 1) * n * (2 * n - 1) / 6
    assert hpx.arange(n).dot(hpx.arange(n)) == pytest.approx(expected, rel=1e-12)


def test_dot_full_is_n_times_cd():
    a = hpx.full(1000, 2.0)
    b = hpx.full(1000, 3.0)
    assert a.dot(b) == pytest.approx(1000 * 2.0 * 3.0, rel=1e-12)


def test_dot_empty_is_zero():
    assert hpx.arange(0).dot(hpx.arange(0)) == 0.0


def test_dot_method_and_free_function_agree():
    a, b = hpx.arange(5000), hpx.arange(5000)
    assert hpx.dot(a, b) == a.dot(b)


def test_dot_size_mismatch_raises():
    with pytest.raises(ValueError):
        hpx.arange(10).dot(hpx.arange(11))
