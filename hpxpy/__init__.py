"""hpxpy — Python arrays backed by the HPX C++ runtime.

Phase 1: a thin wrapper over HPX — a NUMA-aware :class:`Array` (constructors
``zeros``/``full``/``arange``) and HPX reductions (``sum``/``min``/``max``), with
no NumPy in the data path. NumPy compatibility is a separate, later phase (see
docs/PLAN.md).
"""
from __future__ import annotations

import atexit

from . import _core

#: The core 1-D float64 array type, backed by a NUMA-aware HPX compute::vector.
Array = _core.Array

#: A CSR (compressed sparse row) float64 matrix (see :func:`csr_from`, :func:`laplacian_1d`).
CsrMatrix = _core.CsrMatrix

#: A row-major 2-D dense float64 matrix (SpMM operand/result; see :func:`dense_zeros`).
DenseMatrix = _core.DenseMatrix

__version__ = "0.0.1"

_initialized = False


def init(num_threads: int | None = None) -> None:
    """Start the HPX runtime. ``num_threads=None`` uses all cores. Idempotent."""
    global _initialized
    if not _initialized:
        _core.init_runtime(int(num_threads) if num_threads else 0)
        _initialized = True


def finalize() -> None:
    """Stop the HPX runtime. Idempotent; also run automatically at exit."""
    global _initialized
    if _initialized:
        _core.finalize_runtime()
        _initialized = False


atexit.register(finalize)


def sum(a: Array) -> float:  # noqa: A001 - NumPy-style namespace
    """Parallel sum of an :class:`Array` (in-place ``hpx::reduce``, no copy).

    NumPy-style alias for :meth:`Array.sum` (``hpx.sum(a)`` ≡ ``a.sum()``); a
    single kernel lives in C++.
    """
    return a.sum()


def min(a: Array) -> float:  # noqa: A001 - NumPy-style namespace
    """Parallel minimum of an :class:`Array` (empty raises ``ValueError``)."""
    return a.min()


def max(a: Array) -> float:  # noqa: A001 - NumPy-style namespace
    """Parallel maximum of an :class:`Array` (empty raises ``ValueError``)."""
    return a.max()


def dot(a: Array, b: Array) -> float:
    """Fused dot product of two :class:`Array` (single-pass ``transform_reduce``).

    NumPy-style alias for :meth:`Array.dot` (``hpx.dot(a, b)`` ≡ ``a.dot(b)``);
    one kernel in C++. Mismatched sizes raise ``ValueError``.
    """
    return a.dot(b)


def sort(a: Array) -> Array:
    """Return a new ascending sorted Array (like ``numpy.sort``; ``a`` is unchanged).

    In-place sorting is the method ``a.sort()`` (like ``numpy.ndarray.sort``).
    """
    c = a.copy()
    c.sort()
    return c


def cumsum(a: Array) -> Array:
    """Inclusive prefix sum of ``a`` (parallel ``hpx::inclusive_scan``)."""
    return a.cumsum()


def csr_from(rows: int, cols: int, row_ptr, col_idx, values) -> CsrMatrix:
    """Build a :class:`CsrMatrix` from explicit CSR arrays (row_ptr/col_idx/values)."""
    return _core.csr_from(int(rows), int(cols), row_ptr, col_idx, values)


def laplacian_1d(n: int) -> CsrMatrix:
    """1-D Laplacian CSR matrix — tridiagonal ``[-1, 2, -1]``, ``n``×``n``."""
    return _core.laplacian_1d(int(n))


def spmv(a: CsrMatrix, x: Array) -> Array:
    """Sparse matrix-vector product ``A @ x`` (parallel; ``a.spmv(x)``)."""
    return a.spmv(x)


def dense_zeros(rows: int, cols: int) -> DenseMatrix:
    """Create a ``rows``×``cols`` dense matrix of zeros (NUMA-aware)."""
    return _core.dense_zeros(int(rows), int(cols))


def dense_from(rows: int, cols: int, values) -> DenseMatrix:
    """Create a ``rows``×``cols`` dense matrix from a row-major flat sequence."""
    return _core.dense_from(int(rows), int(cols), values)


def spmm(a: CsrMatrix, b: DenseMatrix) -> DenseMatrix:
    """Sparse × dense product ``A @ B`` (parallel; ``a.spmm(b)``)."""
    return a.spmm(b)


def zeros(n: int) -> Array:
    """Create an :class:`Array` of ``n`` zeros (NUMA-aware HPX allocation)."""
    return _core.zeros(int(n))


def full(n: int, value: float) -> Array:
    """Create an :class:`Array` of ``n`` elements set to ``value`` (NUMA-aware)."""
    return _core.full(int(n), float(value))


def arange(n: int) -> Array:
    """Create an :class:`Array` ``[0, 1, ..., n-1]`` (NUMA-aware first-touch)."""
    return _core.arange(int(n))


def num_worker_threads() -> int:
    """Number of HPX worker threads in the running runtime."""
    return _core.num_worker_threads()


def hpx_version() -> str:
    """The linked HPX version string."""
    return _core.hpx_version()


__all__ = [
    "init",
    "finalize",
    "Array",
    "zeros",
    "full",
    "arange",
    "sum",
    "min",
    "max",
    "dot",
    "sort",
    "cumsum",
    "CsrMatrix",
    "csr_from",
    "laplacian_1d",
    "spmv",
    "DenseMatrix",
    "dense_zeros",
    "dense_from",
    "spmm",
    "num_worker_threads",
    "hpx_version",
    "__version__",
]
