"""Phase 2: NumPy bridge — from_numpy (copy default + zero-copy borrow) and to_numpy
(zero-copy view). NumPy is used here only as a test oracle / interop partner.
"""
import numpy as np
import pytest

import hpxpy as hpx


# --- to_numpy: zero-copy writable view --------------------------------------

def test_to_numpy_is_zero_copy_view():
    a = hpx.arange(10, dtype="float64")
    v = hpx.to_numpy(a)
    assert isinstance(v, np.ndarray)
    assert v.dtype == np.float64 and v.shape == (10,)
    np.testing.assert_array_equal(v, np.arange(10.0))
    v[0] = 99.0                       # numpy writes the HPX buffer (shared)
    assert a[0] == 99.0
    a[1] = 7.0                        # hpxpy writes, numpy sees it
    assert v[1] == 7.0


def test_to_numpy_method_and_slice_view():
    a = hpx.arange(10)
    np.testing.assert_array_equal(a.to_numpy(), np.arange(10.0))
    # a slice is a view; its to_numpy is the sub-range
    np.testing.assert_array_equal(a[2:5].to_numpy(), np.array([2.0, 3.0, 4.0]))


def test_np_asarray_protocol():
    a = hpx.arange(5)
    np.testing.assert_array_equal(np.asarray(a), np.arange(5.0))


# --- from_numpy: copy (default) ---------------------------------------------

def test_from_numpy_copy_roundtrip_and_independent():
    npx = np.arange(1000, dtype=np.float64)
    a = hpx.from_numpy(npx)                       # copy=True
    np.testing.assert_array_equal(a.to_numpy(), npx)
    npx[0] = -5.0                                 # mutate the source...
    assert a[0] == 0.0                            # ...copy is independent


def test_from_numpy_copy_oracle():
    npx = np.arange(100000, dtype=np.float64)
    a = hpx.from_numpy(npx)
    assert a.sum() == pytest.approx(npx.sum(), rel=1e-12)
    assert a.dot(a) == pytest.approx(npx @ npx, rel=1e-12)


# --- from_numpy: zero-copy borrow -------------------------------------------

def test_from_numpy_borrow_shares_memory_both_ways():
    npx = np.arange(20, dtype=np.float64)
    a = hpx.from_numpy(npx, copy=False)           # borrow
    npx[0] = 100.0                                # numpy write -> hpxpy sees
    assert a[0] == 100.0
    a[1] = 200.0                                  # hpxpy write -> numpy sees
    assert npx[1] == 200.0


def test_from_numpy_borrow_sort_visible_in_numpy():
    npx = (np.arange(50, dtype=np.float64) * -1.0).copy()   # descending, writable
    a = hpx.from_numpy(npx, copy=False)
    a.sort()                                      # in place -> mutates numpy buffer
    assert a.is_sorted()
    assert np.all(np.diff(npx) >= 0)              # numpy now sorted ascending


def test_borrow_outlives_source_reference():
    # The Array keeps the numpy buffer alive even if the python name goes away.
    a = hpx.from_numpy(np.arange(1000, dtype=np.float64), copy=False)
    assert a.sum() == pytest.approx(999 * 1000 / 2)


# --- dtype / contiguity errors (no silent copy/cast) ------------------------

def test_from_numpy_float32_roundtrip():
    # float32 is now an accepted dtype (stage A2.1): round-trip preserves it.
    npx = np.arange(10, dtype=np.float32)
    a = hpx.from_numpy(npx)
    assert a.dtype == np.float32
    np.testing.assert_array_equal(np.asarray(a), npx)


def test_from_numpy_rejects_unsupported_dtype():
    # int32 / float16 remain unsupported -> TypeError (no silent cast).
    with pytest.raises(TypeError):
        hpx.from_numpy(np.arange(10, dtype=np.int32))
    with pytest.raises(TypeError):
        hpx.from_numpy(np.arange(10, dtype=np.float16))


def test_from_numpy_rejects_noncontiguous():
    with pytest.raises(TypeError):
        hpx.from_numpy(np.arange(20, dtype=np.float64)[::2])


# --- N-D bridge (stage 2: rank-dynamic to_numpy / from_numpy) ----------------

def test_to_numpy_2d_view():
    a = hpx.zeros((3, 4))
    a[1, 2] = 5.0
    v = np.asarray(a)
    assert v.shape == (3, 4) and v.dtype == np.float64
    assert v[1, 2] == 5.0          # byte strides correct -> value at right index
    assert v.sum() == 5.0
    v[0, 0] = 9.0                  # shared buffer
    assert a[0, 0] == 9.0


def test_to_numpy_3d_view_values():
    a = hpx.zeros((2, 3, 4))
    a[1, 2, 3] = 7.0
    v = a.to_numpy()
    assert v.shape == (2, 3, 4)
    assert v[1, 2, 3] == 7.0
    assert v.sum() == 7.0


def test_from_numpy_2d_copy_roundtrip():
    npx = np.arange(12, dtype=np.float64).reshape(3, 4)
    a = hpx.from_numpy(npx)                       # copy
    assert a.shape == (3, 4) and a.ndim == 2
    np.testing.assert_array_equal(np.asarray(a), npx)
    assert a[2, 3] == npx[2, 3]


def test_from_numpy_3d_roundtrip():
    npx = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    a = hpx.from_numpy(npx)
    np.testing.assert_array_equal(np.asarray(a), npx)


def test_from_numpy_2d_borrow_shares():
    npx = np.arange(12, dtype=np.float64).reshape(3, 4)
    a = hpx.from_numpy(npx, copy=False)           # borrow
    npx[1, 1] = 50.0
    assert a[1, 1] == 50.0
    a[2, 0] = 60.0
    assert npx[2, 0] == 60.0


def test_from_numpy_2d_rejects_noncontiguous():
    npx = np.arange(12, dtype=np.float64).reshape(3, 4)
    with pytest.raises(TypeError):
        hpx.from_numpy(npx.T)                      # transpose is non-C-contiguous
