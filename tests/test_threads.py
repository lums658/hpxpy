"""V&V: the HPX worker pool size is honored via ``--hpx:threads`` (hpxpy.init(N)).

A prototype lesson was that the ``hpx.os_threads`` cfg key did NOT limit the pool
while the ``--hpx:threads=N`` CLI option did; hpxpy.init builds the CLI option. HPX
inits once per process, so each thread count is checked in a FRESH subprocess.
"""
import os
import subprocess
import sys

import pytest

# Cap at the cores actually available so "request N -> get N" is a fair check.
_MAX = min(4, os.cpu_count() or 1)
_COUNTS = [n for n in (1, 2, 4) if n <= _MAX]


def _worker_count(num_threads: int) -> int:
    """Worker count reported by a fresh process that did hpxpy.init(num_threads)."""
    out = subprocess.run(
        [sys.executable, "-c",
         f"import hpxpy as h; h.init({num_threads}); print(h.num_worker_threads())"],
        capture_output=True, text=True, check=True,
    )
    return int(out.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("n", _COUNTS)
def test_threads_request_is_honored(n):
    assert _worker_count(n) == n


def test_default_uses_multiple_threads():
    # init() with no argument => all cores; on any real node that is > 1.
    if _MAX < 2:
        pytest.skip("single-core environment")
    out = subprocess.run(
        [sys.executable, "-c",
         "import hpxpy as h; h.init(); print(h.num_worker_threads())"],
        capture_output=True, text=True, check=True,
    )
    assert int(out.stdout.strip().splitlines()[-1]) >= 2
