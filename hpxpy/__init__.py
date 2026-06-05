"""hpxpy — NumPy-compatible Python arrays backed by the HPX C++ runtime.

M0 substrate: a managed HPX runtime plus a zero-copy spike (``sum``). Real array
types and operations arrive in later milestones (see docs/PLAN.md).
"""
from __future__ import annotations

import atexit

import numpy as _np

from . import _core

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
    """Parallel sum of a 1-D array via HPX (zero-copy for float64 C-contiguous)."""
    return _core.array_sum(_np.ascontiguousarray(a, dtype=_np.float64).ravel())


def num_worker_threads() -> int:
    """Number of HPX worker threads in the running runtime."""
    return _core.num_worker_threads()


def hpx_version() -> str:
    """The linked HPX version string."""
    return _core.hpx_version()


__all__ = [
    "init",
    "finalize",
    "sum",
    "num_worker_threads",
    "hpx_version",
    "__version__",
]
