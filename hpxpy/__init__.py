"""hpxpy — NumPy-compatible Python arrays backed by the HPX C++ runtime.

M0 substrate: a managed HPX runtime plus a zero-copy spike (``sum``). Real array
types and operations arrive in later milestones (see docs/PLAN.md).
"""
from __future__ import annotations

import atexit

import numpy as _np

from . import _core

#: The core 1-D float64 array type, backed by a NUMA-aware HPX compute::vector.
Array = _core.Array

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


def sum(a) -> float:  # noqa: A001 - NumPy-style namespace
    """Parallel sum of a 1-D float64 C-contiguous array via HPX (zero-copy).

    The input buffer is borrowed, never copied: a non-contiguous or non-float64
    array raises ``TypeError`` rather than being silently converted/copied.
    """
    return _core.array_sum(a)


def zeros(n: int) -> Array:
    """Create an :class:`Array` of ``n`` zeros (NUMA-aware HPX allocation)."""
    return _core.zeros(int(n))


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
    "sum",
    "num_worker_threads",
    "hpx_version",
    "__version__",
]
