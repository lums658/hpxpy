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


# --- Element-wise unary math ufuncs (Wave 1) -------------------------------
# Preserve-dtype: negative/abs/sign. Promote-int-to-float64: sqrt/exp/log/sin/
# cos/tan/floor/ceil/trunc/round. Each is a NumPy-style alias for the method.


def negative(a: Array) -> Array:
    """Element-wise ``-a`` (preserves dtype)."""
    return a.negative()


def abs(a: Array) -> Array:  # noqa: A001 - NumPy-style namespace (shadows builtin)
    """Element-wise absolute value (preserves dtype)."""
    return a.abs()


def sign(a: Array) -> Array:
    """Element-wise sign (``-1``/``0``/``1``; preserves dtype)."""
    return a.sign()


def sqrt(a: Array) -> Array:
    """Element-wise square root (int input -> float64; float keeps its dtype)."""
    return a.sqrt()


def exp(a: Array) -> Array:
    """Element-wise ``e**x`` (int input -> float64)."""
    return a.exp()


def log(a: Array) -> Array:
    """Element-wise natural logarithm (int input -> float64)."""
    return a.log()


def sin(a: Array) -> Array:
    """Element-wise sine (int input -> float64)."""
    return a.sin()


def cos(a: Array) -> Array:
    """Element-wise cosine (int input -> float64)."""
    return a.cos()


def tan(a: Array) -> Array:
    """Element-wise tangent (int input -> float64)."""
    return a.tan()


def floor(a: Array) -> Array:
    """Element-wise floor (int input -> float64)."""
    return a.floor()


def ceil(a: Array) -> Array:
    """Element-wise ceil (int input -> float64)."""
    return a.ceil()


def trunc(a: Array) -> Array:
    """Element-wise truncate toward zero (int input -> float64)."""
    return a.trunc()


def round(a: Array) -> Array:  # noqa: A001 - NumPy-style namespace (shadows builtin)
    """Element-wise round-half-to-even / banker's rounding (int input -> float64)."""
    return a.round()


def rint(a: Array) -> Array:
    """Round to the nearest integer (half-to-even); alias for :func:`round`."""
    return a.round()


# --- Element-wise binary math ufuncs (Wave 1; preserve-dtype) --------------


def maximum(a: Array, b: Array) -> Array:
    """Element-wise ``max(a, b)`` (operands must share a dtype)."""
    return a.maximum(b)


def minimum(a: Array, b: Array) -> Array:
    """Element-wise ``min(a, b)`` (operands must share a dtype)."""
    return a.minimum(b)


def power(a: Array, b: Array) -> Array:
    """Element-wise ``a ** b`` (operands must share a dtype)."""
    return a.power(b)


def mod(a: Array, b: Array) -> Array:
    """Element-wise ``a % b`` (divisor-signed remainder, like numpy)."""
    return a.mod(b)


def floor_divide(a: Array, b: Array) -> Array:
    """Element-wise ``a // b`` (floor toward ``-inf``, like numpy)."""
    return a.floor_divide(b)


def clip(a: Array, lo: float, hi: float) -> Array:
    """Clamp each element of ``a`` to ``[lo, hi]`` (preserves dtype)."""
    return a.clip(float(lo), float(hi))


# --- Reductions added in Wave 1 --------------------------------------------


def mean(a: Array, axis=None, keepdims: bool = False):
    """Arithmetic mean of ``a`` (always float64).

    ``axis=None`` (default) returns a Python ``float``; an ``int`` or tuple reduces
    those axes and returns a new float64 :class:`Array`.
    """
    return a.mean(axis, keepdims)


def prod(a: Array, axis=None, keepdims: bool = False):
    """Product of ``a`` (preserves dtype; int wraps on overflow, like numpy).

    ``axis=None`` (default) returns a scalar; an ``int`` or tuple reduces those axes
    and returns a new :class:`Array`.
    """
    return a.prod(axis, keepdims)


def any(a: Array, axis=None) -> bool:  # noqa: A001 - NumPy-style namespace
    """True if any element of ``a`` is nonzero (Wave 1: ``axis=None`` only)."""
    return a.any(axis)


def all(a: Array, axis=None) -> bool:  # noqa: A001 - NumPy-style namespace
    """True if all elements of ``a`` are nonzero (Wave 1: ``axis=None`` only)."""
    return a.all(axis)


def count_nonzero(a: Array, axis=None) -> int:
    """Number of nonzero elements in ``a`` (Wave 1: ``axis=None`` only)."""
    return a.count_nonzero(axis)


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
    """Bring a float64/float32/int64 C-contiguous NumPy array into an :class:`Array`.

    ``copy=True`` (default) copies into a NUMA-aware buffer (correct first-touch, so
    HPX ops keep their performance). ``copy=False`` borrows the NumPy buffer
    zero-copy — hpxpy and NumPy then share memory both ways, but it is numa-naive,
    so prefer the default for compute. Unsupported-dtype or non-contiguous input
    raises ``TypeError`` (never a silent copy/cast).
    """
    return _core.from_numpy(a, copy)


def to_numpy(a: Array):
    """Zero-copy NumPy view of an :class:`Array` (writable; shares memory)."""
    return _core.to_numpy(a)


def zeros(shape, dtype=None) -> Array:
    """Create an :class:`Array` of zeros (NUMA-aware HPX allocation).

    ``shape`` may be an ``int`` (1-D, backward-compatible) or a ``tuple``/``list``
    of ints (N-D, row-major C-order). ``dtype`` defaults to ``float64`` and accepts
    ``float32``/``int64`` (numpy dtype, scalar type, or string).
    """
    if isinstance(shape, (tuple, list)):
        return _core.zeros(shape, dtype)
    return _core.zeros(int(shape), dtype)


def ones(shape, dtype=None) -> Array:
    """Create an :class:`Array` of ones (NUMA-aware HPX allocation).

    ``shape`` may be an ``int`` (1-D) or a ``tuple``/``list`` of ints (N-D,
    row-major C-order). ``dtype`` defaults to ``float64``.
    """
    if isinstance(shape, (tuple, list)):
        return _core.ones(shape, dtype)
    return _core.ones(int(shape), dtype)


def full(shape, value: float, dtype=None) -> Array:
    """Create an :class:`Array` filled with ``value`` (NUMA-aware).

    ``shape`` may be an ``int`` (1-D, backward-compatible) or a ``tuple``/``list``
    of ints (N-D, row-major C-order). ``dtype`` defaults to ``float64``.
    """
    if isinstance(shape, (tuple, list)):
        return _core.full(shape, float(value), dtype)
    return _core.full(int(shape), float(value), dtype)


def arange(n: int, dtype=None) -> Array:
    """Create an :class:`Array` ``[0, 1, ..., n-1]`` (NUMA-aware first-touch).

    ``dtype`` defaults to ``float64`` and accepts ``float32``/``int64``.
    """
    return _core.arange(int(n), dtype)


def astype(a: Array, dtype) -> Array:
    """Cast ``a`` to a new :class:`Array` of ``dtype`` (element-wise; ``a.astype``)."""
    return a.astype(dtype)


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
    "negative",
    "abs",
    "sign",
    "sqrt",
    "exp",
    "log",
    "sin",
    "cos",
    "tan",
    "floor",
    "ceil",
    "trunc",
    "round",
    "rint",
    "maximum",
    "minimum",
    "power",
    "mod",
    "floor_divide",
    "clip",
    "mean",
    "prod",
    "any",
    "all",
    "count_nonzero",
    "astype",
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
