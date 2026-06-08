"""NumPy drop-in parity: run the SAME snippet under numpy and under hpxpy on identical
data and assert the results agree. Demonstrates the 'import hpxpy as np'-replaceable
goal and cross-checks every op against numpy as oracle (via the from_numpy/to_numpy
bridge). numpy is a test-only oracle.
"""
import numpy as np
import pytest

import hpxpy as hpx

_rng = np.random.default_rng(0)


def _to_np(x):
    """hpxpy result -> numpy. Array -> zero-copy view (__array__); scalars pass through."""
    return np.asarray(x)


def _check(snippet, *data):
    """Run `snippet(xp, *arrays)` under numpy and hpxpy on identical data; compare."""
    np_args = list(data)
    hpx_args = [hpx.from_numpy(d.copy()) for d in data]      # same data, NUMA copy
    np_res = snippet(np, *np_args)
    hpx_res = snippet(hpx, *hpx_args)
    np.testing.assert_allclose(_to_np(hpx_res), np.asarray(np_res),
                               rtol=1e-12, atol=1e-12)


# --- snippet catalogue (shared numpy/hpxpy subset) --------------------------
# Each entry: (id, arity, fn). arity = number of input arrays the snippet takes.

_NULLARY = [
    ("arange", lambda xp, n: xp.arange(n)),
    ("zeros", lambda xp, n: xp.zeros(n)),
    ("full", lambda xp, n: xp.full(n, 3.0)),
]

_UNARY = [
    ("sum", lambda xp, a: xp.sum(a)),
    ("min", lambda xp, a: xp.min(a)),
    ("max", lambda xp, a: xp.max(a)),
    ("method_sum", lambda xp, a: a.sum()),
    ("sort", lambda xp, a: xp.sort(a)),
    ("cumsum", lambda xp, a: xp.cumsum(a)),
    ("method_cumsum", lambda xp, a: a.cumsum()),
    ("mul_scalar", lambda xp, a: a * 2.0),
    ("rmul_scalar", lambda xp, a: 2.0 * a),
    ("add_scalar", lambda xp, a: a + 1.0),
    ("rsub_scalar", lambda xp, a: 5.0 - a),
    ("div_scalar", lambda xp, a: a / 2.0),
    ("slice_mid", lambda xp, a: a[2:5]),
    ("slice_tail", lambda xp, a: a[1:]),
    ("slice_head", lambda xp, a: a[:3]),
    ("slice_neg", lambda xp, a: a[-3:]),
]

_BINARY = [
    ("dot", lambda xp, a, b: xp.dot(a, b)),
    ("add", lambda xp, a, b: a + b),
    ("sub", lambda xp, a, b: a - b),
    ("mul", lambda xp, a, b: a * b),
    ("div", lambda xp, a, b: a / b),
    ("stream_triad", lambda xp, a, b: b + 3.0 * a),    # the bead's "support stream"
]

_SIZES = [1, 2, 7, 257, 4096]
# Snippets that index [2:5], [:3], [-3:] need a few elements; cap those sizes' set.
_SLICE_IDS = {"slice_mid", "slice_tail", "slice_head", "slice_neg"}


@pytest.mark.parametrize("n", _SIZES)
@pytest.mark.parametrize("name,fn", _NULLARY, ids=[t[0] for t in _NULLARY])
def test_parity_constructors(name, fn, n):
    _check(lambda xp, _n=n: fn(xp, _n))


@pytest.mark.parametrize("n", _SIZES)
@pytest.mark.parametrize("name,fn", _UNARY, ids=[t[0] for t in _UNARY])
def test_parity_unary(name, fn, n):
    if name in _SLICE_IDS and n < 5:
        pytest.skip("slice snippet needs a few elements")
    a = _rng.standard_normal(n)
    _check(fn, a)


@pytest.mark.parametrize("n", _SIZES)
@pytest.mark.parametrize("name,fn", _BINARY, ids=[t[0] for t in _BINARY])
def test_parity_binary(name, fn, n):
    a = _rng.standard_normal(n)
    b = _rng.uniform(1.0, 2.0, n)        # positive divisor (safe for a / b)
    _check(fn, a, b)


# --- sparse parity vs a dense NumPy equivalent (small n; dense oracle is O(n^2)) ---

def _dense_laplacian(n):
    return 2.0 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)


def test_spmv_parity_vs_dense():
    n = 200
    x = _rng.standard_normal(n)
    y_hpx = hpx.laplacian_1d(n) @ hpx.from_numpy(x.copy())
    y_np = _dense_laplacian(n) @ x
    np.testing.assert_allclose(_to_np(y_hpx), y_np, rtol=1e-10, atol=1e-10)


def test_spmm_parity_vs_dense():
    n, k = 200, 4
    bmat = _rng.standard_normal((n, k))
    c_hpx = hpx.laplacian_1d(n) @ hpx.dense_from(n, k, bmat.ravel().tolist())
    c_np = _dense_laplacian(n) @ bmat
    # DenseMatrix has no to_numpy yet (deferred); read it out via at(i, j).
    got = np.array([[c_hpx.at(i, j) for j in range(k)] for i in range(n)])
    np.testing.assert_allclose(got, c_np, rtol=1e-10, atol=1e-10)
