"""hpxpy — Python arrays backed by the HPX C++ runtime.

Phase 1: a thin wrapper over HPX — a NUMA-aware :class:`Array` (constructors
``zeros``/``ones``/``full``/``arange``) and HPX reductions (``sum``/``min``/``max``),
with no NumPy in the data path. NumPy compatibility is a separate, later phase (see
docs/PLAN.md).
"""
from __future__ import annotations

import atexit

from . import _core

#: The core float64 array type, backed by a NUMA-aware HPX compute::vector.
Array = _core.Array

#: A CSR (compressed sparse row) float64 matrix (see :func:`csr_from`, :func:`laplacian_1d`).
CsrMatrix = _core.CsrMatrix

#: A row-major 2-D dense float64 matrix (SpMM operand/result; see :func:`dense_zeros`).
DenseMatrix = _core.DenseMatrix

__version__ = "0.0.1"

_initialized = False


def init(num_threads: int | None = None, hpx_args: list[str] | None = None) -> None:
    """Start the HPX runtime. ``num_threads=None`` uses all cores. Idempotent.

    ``hpx_args`` are raw HPX command-line flags appended to the runtime's argv —
    used for a distributed launch, e.g.
    ``["--hpx:localities=2", "--hpx:agas=host:7910", "--hpx:hpx=host:7910"]`` on the
    console and the same plus ``"--hpx:worker"`` (and its own ``--hpx:hpx`` port) on a
    worker. Under Slurm/``srun`` HPX auto-detects the layout, so ``hpx_args`` can be empty.
    """
    global _initialized
    if not _initialized:
        _core.init_runtime(int(num_threads) if num_threads else 0,
                           list(hpx_args) if hpx_args else [])
        _initialized = True


def finalize() -> None:
    """Stop the HPX runtime. Idempotent; also run automatically at exit."""
    global _initialized
    if _initialized:
        _core.finalize_runtime()
        _initialized = False


atexit.register(finalize)


def num_localities() -> int:
    """Number of HPX localities (processes) in the running runtime (1 if single-node)."""
    return _core.num_localities()


def locality_id() -> int:
    """This locality's id (0 on the console locality)."""
    return _core.locality_id()


#: Alias for :func:`locality_id` (this locality's id).
here = locality_id


def is_console() -> bool:
    """True on the console locality (the one that runs the user program)."""
    return _core.is_console()


def is_worker() -> bool:
    """True if this process was started as a ``--hpx:worker`` locality."""
    return _core.is_worker()


def distributed_sum(local: float) -> float:
    """All-reduce(sum) of a scalar across localities (every locality must call it).

    Returns ``local`` unchanged when there is a single locality. A cross-locality
    smoke primitive for M4; the global-view distributed :class:`Array` (segmented
    reductions over a partitioned vector) is the next milestone.
    """
    return _core.distributed_sum(float(local))


def sum(a: Array, axis=None, keepdims: bool = False):  # noqa: A001 - NumPy-style namespace
    """Parallel sum of an :class:`Array` (in-place ``hpx::reduce``, no copy).

    NumPy-style alias for :meth:`Array.sum` (``hpx.sum(a)`` ≡ ``a.sum()``); a
    single kernel lives in C++. ``axis=None`` (default) returns a scalar; an ``int``
    or tuple of ints reduces those axes and returns a new :class:`Array`.
    """
    return a.sum(axis, keepdims)


def min(a: Array, axis=None, keepdims: bool = False):  # noqa: A001 - NumPy-style namespace
    """Parallel minimum of an :class:`Array` (empty raises ``ValueError``).

    ``axis=None`` (default) returns a scalar; an ``int`` or tuple reduces those axes
    to a new :class:`Array` (reducing an empty axis raises ``ValueError``).
    """
    return a.min(axis, keepdims)


def max(a: Array, axis=None, keepdims: bool = False):  # noqa: A001 - NumPy-style namespace
    """Parallel maximum of an :class:`Array` (empty raises ``ValueError``).

    ``axis=None`` (default) returns a scalar; an ``int`` or tuple reduces those axes
    to a new :class:`Array` (reducing an empty axis raises ``ValueError``).
    """
    return a.max(axis, keepdims)


def dot(a: Array, b: Array):
    """Dot product / matrix product of two :class:`Array`.

    NumPy-style alias for :meth:`Array.dot` (``hpx.dot(a, b)`` ≡ ``a.dot(b)``); one
    kernel in C++. 1-D · 1-D returns a scalar (fused ``transform_reduce``); 2-D · 2-D
    returns a new :class:`Array` (matrix product). Other ranks raise ``ValueError``.
    """
    return a.dot(b)


def matmul(a: Array, b: Array) -> Array:
    """2-D matrix product ``A @ B`` of two :class:`Array` (``a.matmul(b)``).

    Both operands must be 2-D with matching inner dimension; otherwise ``ValueError``.
    """
    return a.matmul(b)


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


def from_numpy(a, copy: bool = True) -> Array:
    """Bring a 1-D float64 C-contiguous NumPy array into an :class:`Array`.

    ``copy=True`` (default) copies into a NUMA-aware buffer (correct first-touch, so
    HPX ops keep their performance). ``copy=False`` borrows the NumPy buffer
    zero-copy — hpxpy and NumPy then share memory both ways, but it is numa-naive,
    so prefer the default for compute. Non-float64/non-contiguous input raises.
    """
    return _core.from_numpy(a, copy)


def to_numpy(a: Array):
    """Zero-copy NumPy view of an :class:`Array` (writable; shares memory)."""
    return _core.to_numpy(a)


def zeros(shape) -> Array:
    """Create an :class:`Array` of zeros (NUMA-aware HPX allocation).

    ``shape`` may be an ``int`` (1-D, backward-compatible) or a ``tuple``/``list``
    of ints (N-D, row-major C-order).
    """
    if isinstance(shape, (tuple, list)):
        return _core.zeros(shape)
    return _core.zeros(int(shape))


def ones(shape) -> Array:
    """Create an :class:`Array` of ones (NUMA-aware HPX allocation).

    ``shape`` may be an ``int`` (1-D) or a ``tuple``/``list`` of ints (N-D,
    row-major C-order).
    """
    if isinstance(shape, (tuple, list)):
        return _core.ones(shape)
    return _core.ones(int(shape))


def full(shape, value: float) -> Array:
    """Create an :class:`Array` filled with ``value`` (NUMA-aware).

    ``shape`` may be an ``int`` (1-D, backward-compatible) or a ``tuple``/``list``
    of ints (N-D, row-major C-order).
    """
    if isinstance(shape, (tuple, list)):
        return _core.full(shape, float(value))
    return _core.full(int(shape), float(value))


def arange(n: int) -> Array:
    """Create an :class:`Array` ``[0, 1, ..., n-1]`` (NUMA-aware first-touch)."""
    return _core.arange(int(n))


def transpose(a: Array, axes=None) -> Array:
    """Permute axes of ``a`` (``None`` or empty reverses all axes). Zero-copy view."""
    return a.transpose(axes)


def reshape(a: Array, shape) -> Array:
    """Reshape ``a`` to ``shape`` (int or tuple; one -1 is inferred).

    Returns a zero-copy view when ``a`` is contiguous, otherwise a copy.
    """
    return a.reshape(shape)


def squeeze(a: Array, axis=None) -> Array:
    """Remove size-1 dimensions (all if ``axis=None``, else the named axis/axes)."""
    return a.squeeze(axis)


def expand_dims(a: Array, axis: int) -> Array:
    """Insert a size-1 dimension at position ``axis``."""
    return a.expand_dims(int(axis))


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
    "ones",
    "full",
    "arange",
    "sum",
    "min",
    "max",
    "dot",
    "matmul",
    "sort",
    "cumsum",
    "transpose",
    "reshape",
    "squeeze",
    "expand_dims",
    "CsrMatrix",
    "csr_from",
    "laplacian_1d",
    "spmv",
    "DenseMatrix",
    "dense_zeros",
    "dense_from",
    "spmm",
    "from_numpy",
    "to_numpy",
    "num_worker_threads",
    "hpx_version",
    "num_localities",
    "locality_id",
    "here",
    "is_console",
    "is_worker",
    "distributed_sum",
    "__version__",
]
