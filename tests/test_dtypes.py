"""Phase A.2 stage A2.1: dtype foundation — float32 + int64 as runtime dtypes on
the (formerly float64-only) Array, NumPy-faithfully.

This stage = create / inspect / index / astype / numpy round-trip for all three
dtypes. Compute (sum/add/...) on non-float64 is GUARDED to raise (the templated
kernels land in the next stage, A2.2). NumPy is the oracle throughout.
"""
import numpy as np
import pytest

import hpxpy as hpx


# --- construction + dtype inspection ----------------------------------------

@pytest.mark.parametrize("dt", [np.float32, np.int64, np.float64])
def test_zeros_dtype(dt):
    a = hpx.zeros(5, dtype=dt)
    assert a.dtype == np.dtype(dt)
    assert np.asarray(a).dtype == np.dtype(dt)
    np.testing.assert_array_equal(np.asarray(a), np.zeros(5, dtype=dt))


@pytest.mark.parametrize("dt", [np.float32, np.int64])
def test_ones_full_arange_dtype(dt):
    o = hpx.ones(4, dtype=dt)
    assert o.dtype == np.dtype(dt)
    np.testing.assert_array_equal(np.asarray(o), np.ones(4, dtype=dt))

    f = hpx.full(4, 7, dtype=dt)
    assert f.dtype == np.dtype(dt)
    np.testing.assert_array_equal(np.asarray(f), np.full(4, 7, dtype=dt))

    r = hpx.arange(6, dtype=dt)
    assert r.dtype == np.dtype(dt)
    np.testing.assert_array_equal(np.asarray(r), np.arange(6, dtype=dt))


def test_dtype_string_and_python_types():
    assert hpx.zeros(3, dtype="float32").dtype == np.float32
    assert hpx.zeros(3, dtype="f4").dtype == np.float32
    assert hpx.zeros(3, dtype="int64").dtype == np.int64
    assert hpx.zeros(3, dtype="i8").dtype == np.int64
    assert hpx.zeros(3, dtype=float).dtype == np.float64
    assert hpx.zeros(3, dtype=int).dtype == np.int64
    # Default stays float64.
    assert hpx.zeros(3).dtype == np.float64


def test_nd_dtype():
    a = hpx.zeros((2, 3), dtype=np.float32)
    assert a.dtype == np.float32
    assert a.shape == (2, 3)
    assert np.asarray(a).dtype == np.float32


def test_repr_includes_dtype():
    assert "dtype=float32" in repr(hpx.zeros(4, dtype=np.float32))
    assert "dtype=int64" in repr(hpx.zeros(4, dtype=np.int64))
    assert "dtype=float64" in repr(hpx.zeros(4))


# --- indexing (values + Python types) ---------------------------------------

def test_getitem_setitem_float32():
    a = hpx.zeros(4, dtype=np.float32)
    a[1] = 2.5
    assert a[1] == 2.5
    assert isinstance(a[1], float)
    np.testing.assert_array_equal(np.asarray(a), np.array([0, 2.5, 0, 0], np.float32))


def test_getitem_setitem_int64():
    a = hpx.zeros(4, dtype=np.int64)
    a[2] = 9
    assert a[2] == 9
    assert isinstance(a[2], int)          # int64 array -> Python int, not float
    np.testing.assert_array_equal(np.asarray(a), np.array([0, 0, 9, 0], np.int64))


def test_int64_setitem_casts():
    a = hpx.zeros(3, dtype=np.int64)
    a[0] = 3.9                            # cast to int64 (truncates like numpy assign)
    assert a[0] == 3
    assert isinstance(a[0], int)


def test_nd_getitem_setitem_dtype():
    a = hpx.zeros((2, 3), dtype=np.int64)
    a[1, 2] = 5
    assert a[1, 2] == 5
    assert isinstance(a[1, 2], int)
    f = hpx.zeros((2, 3), dtype=np.float32)
    f[0, 1] = 1.5
    assert f[0, 1] == 1.5
    assert isinstance(f[0, 1], float)


# --- from_numpy / to_numpy round-trip ---------------------------------------

@pytest.mark.parametrize("dt", [np.float32, np.int64])
def test_roundtrip_copy_1d(dt):
    npx = np.arange(8, dtype=dt)
    a = hpx.from_numpy(npx)               # copy
    assert a.dtype == np.dtype(dt)
    np.testing.assert_array_equal(np.asarray(a), npx)


@pytest.mark.parametrize("dt", [np.float32, np.int64])
def test_roundtrip_copy_2d(dt):
    npx = np.arange(12, dtype=dt).reshape(3, 4)
    a = hpx.from_numpy(npx)
    assert a.dtype == np.dtype(dt) and a.shape == (3, 4)
    np.testing.assert_array_equal(np.asarray(a), npx)


@pytest.mark.parametrize("dt", [np.float32, np.int64])
def test_roundtrip_borrow_shares_memory(dt):
    npx = np.arange(6, dtype=dt)
    a = hpx.from_numpy(npx, copy=False)   # zero-copy borrow
    assert a.dtype == np.dtype(dt)
    npx[0] = 100                          # numpy write -> hpxpy sees
    assert a[0] == 100
    a[1] = 50                             # hpxpy write -> numpy sees
    assert npx[1] == 50


@pytest.mark.parametrize("dt", [np.float32, np.int64])
def test_roundtrip_borrow_2d(dt):
    npx = np.arange(12, dtype=dt).reshape(3, 4)
    a = hpx.from_numpy(npx, copy=False)
    assert a.dtype == np.dtype(dt)
    npx[1, 1] = 77
    assert a[1, 1] == 77


def test_to_numpy_dtype_view_is_shared():
    a = hpx.zeros(5, dtype=np.float32)
    v = np.asarray(a)
    assert v.dtype == np.float32
    v[3] = 8.0
    assert a[3] == 8.0


# --- astype across all dtype pairs ------------------------------------------

@pytest.mark.parametrize("src", [np.float64, np.float32, np.int64])
@pytest.mark.parametrize("dst", [np.float64, np.float32, np.int64])
def test_astype_pairs(src, dst):
    npx = np.arange(6, dtype=src)
    a = hpx.from_numpy(npx)
    b = a.astype(dst)
    assert b.dtype == np.dtype(dst)
    np.testing.assert_array_equal(np.asarray(b), npx.astype(dst))
    # source unchanged
    assert a.dtype == np.dtype(src)


def test_astype_float_to_int_truncates():
    a = hpx.from_numpy(np.array([1.9, -2.7, 3.2], dtype=np.float64))
    b = a.astype(np.int64)
    np.testing.assert_array_equal(
        np.asarray(b), np.array([1.9, -2.7, 3.2]).astype(np.int64))


def test_astype_namespace_function():
    a = hpx.zeros(3, dtype=np.float64)
    b = hpx.astype(a, np.float32)
    assert b.dtype == np.float32


# --- unsupported-dtype rejection (no silent cast) ---------------------------

def test_from_numpy_rejects_int32():
    with pytest.raises(TypeError):
        hpx.from_numpy(np.arange(3, dtype=np.int32))


def test_from_numpy_rejects_float16():
    with pytest.raises(TypeError):
        hpx.from_numpy(np.arange(3, dtype=np.float16))


def test_zeros_rejects_int32():
    with pytest.raises(TypeError):
        hpx.zeros(3, dtype="int32")


def test_zeros_rejects_complex():
    with pytest.raises(TypeError):
        hpx.zeros(3, dtype=np.complex128)


def test_astype_rejects_unsupported():
    a = hpx.zeros(3)
    with pytest.raises(TypeError):
        a.astype("int32")


# --- view ops preserve dtype (no compute, no guard) -------------------------

def test_transpose_preserves_dtype():
    a = hpx.zeros((3, 4), dtype=np.float32)
    assert a.T.dtype == np.float32
    assert a.transpose().dtype == np.float32
    assert hpx.zeros((3, 4), dtype=np.int64).T.dtype == np.int64


def test_reshape_ravel_preserves_dtype():
    a = hpx.arange(12, dtype=np.float32)
    assert a.reshape((3, 4)).dtype == np.float32
    assert a.reshape((3, 4)).ravel().dtype == np.float32
    assert hpx.arange(12, dtype=np.int64).reshape((2, 6)).dtype == np.int64


def test_squeeze_expand_preserve_dtype():
    a = hpx.zeros((1, 3), dtype=np.int64)
    assert a.squeeze().dtype == np.int64
    assert hpx.zeros(3, dtype=np.float32).expand_dims(0).dtype == np.float32


def test_slice_view_preserves_dtype():
    a = hpx.arange(10, dtype=np.float32)
    assert a[2:5].dtype == np.float32
    assert a[::2].dtype == np.float32
    b = hpx.arange(12, dtype=np.int64).reshape((3, 4))
    assert b[1:3, ::2].dtype == np.int64


def test_view_values_correct_for_dtype():
    npx = np.arange(12, dtype=np.int64).reshape(3, 4)
    a = hpx.from_numpy(npx)
    np.testing.assert_array_equal(np.asarray(a.T), npx.T)
    np.testing.assert_array_equal(np.asarray(a[1:3]), npx[1:3])


# --- core element-wise compute now works on non-float64 (A2.2) --------------
# (sum/min/max/dot/add/sub/mul + scalar ops were GUARDED in A2.1; A2.2 templated
#  the kernels over the element type, so they now compute. Detailed numpy-oracle
#  coverage lives in test_dtypes_compute.py; these are the updated A2.1 guards.)

def test_float32_sum_now_works():
    assert hpx.zeros(4, dtype=np.float32).sum() == 0.0


def test_int64_sum_now_works():
    a = hpx.arange(4, dtype=np.int64)
    assert a.sum() == np.arange(4).sum()


def test_float32_add_now_works():
    a = hpx.ones(4, dtype=np.float32)
    b = hpx.ones(4, dtype=np.float32)
    np.testing.assert_array_equal(np.asarray(a + b), np.full(4, 2.0, np.float32))


def test_float32_scalar_add_now_works():
    a = hpx.ones(4, dtype=np.float32)
    np.testing.assert_array_equal(np.asarray(a + 1.0), np.full(4, 2.0, np.float32))


def test_int64_min_max_now_works():
    a = hpx.arange(4, dtype=np.int64)
    assert a.min() == 0
    assert a.max() == 3


# --- ops still GUARDED to float64 (deferred to later A2.x stages) ------------

def test_float32_copy_raises():
    with pytest.raises(RuntimeError):
        hpx.zeros(4, dtype=np.float32).copy()


def test_float32_compute_after_astype_to_f64_works():
    # The escape hatch this stage provides: cast up to float64, then compute.
    a = hpx.from_numpy(np.arange(5, dtype=np.float32)).astype(np.float64)
    assert a.sum() == pytest.approx(10.0)
