"""M3: scalar broadcast ops (Array ⊙ scalar, and reflected scalar ⊙ Array), checked
against ANALYTIC values via a reduction of the result. Each op returns a new Array.
"""
import pytest

import hpxpy as hpx


@pytest.mark.parametrize("n", [1, 2, 1000, 100_000])
def test_mul_scalar(n):
    # (i * 2) summed = 2 * sum i = n(n-1)
    assert (hpx.arange(n) * 2.0).sum() == pytest.approx(n * (n - 1), rel=1e-12)


def test_add_scalar():
    n = 1000
    # sum(i + 1) = n(n-1)/2 + n
    assert (hpx.arange(n) + 1.0).sum() == pytest.approx(n * (n - 1) / 2 + n, rel=1e-12)


def test_sub_scalar():
    n = 1000
    # sum(i - 1) = n(n-1)/2 - n
    assert (hpx.arange(n) - 1.0).sum() == pytest.approx(n * (n - 1) / 2 - n, rel=1e-12)


def test_reflected_mul_and_add_commute():
    a = hpx.arange(5000)
    assert (3.0 * a).sum() == (a * 3.0).sum()
    assert (1.0 + a).sum() == (a + 1.0).sum()


def test_reflected_sub():
    n = 1000
    # sum(10 - i) = 10n - n(n-1)/2
    assert (10.0 - hpx.arange(n)).sum() == pytest.approx(10 * n - n * (n - 1) / 2, rel=1e-12)


def test_div_and_rdiv():
    a = hpx.full(1000, 6.0)
    assert (a / 2.0).sum() == pytest.approx(1000 * 3.0, rel=1e-12)       # 6/2
    assert (6.0 / hpx.full(1000, 2.0)).sum() == pytest.approx(1000 * 3.0, rel=1e-12)


def test_int_operand_accepted():
    # A Python int scalar binds to the int overload (preserves the int64 dtype after
    # the A2.4 promotion split); the summed value is unchanged either way.
    assert (hpx.arange(100) * 2).sum() == pytest.approx(100 * 99, rel=1e-12)


def test_empty():
    a = hpx.arange(0)
    assert (a * 2.0).size == 0
    assert (a * 2.0).sum() == 0.0
