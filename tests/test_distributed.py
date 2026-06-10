"""M4 distributed-runtime V&V: 2-locality TCP launch on a single host.

Marked ``distributed`` so it is **excluded from the default pytest run** (see
``pyproject.toml addopts``).  Run explicitly with::

    pytest -m distributed tests/test_distributed.py

Each test spawns two OS processes of the SAME script: one console (locality 0)
and one worker (locality 1).  The script is short enough to pass via ``-c``;
sys.argv[1:] therefore receives the HPX flags.

Prerequisites
-------------
* ``LD_PRELOAD=$HPXPY_TCMALLOC`` must be set in the calling shell so that both
  sub-processes inherit it (``env=os.environ`` is used for Popen).
* Two free TCP ports must be bindable on the host's hostname.
"""
import os
import socket
import subprocess
import sys
import time

import pytest

pytestmark = pytest.mark.distributed

# ---------------------------------------------------------------------------
# Smoke script (passed as -c "...").  When Python is invoked as
#   python -c "<CODE>" A B C
# sys.argv == ['-c', 'A', 'B', 'C'], so sys.argv[1:] are the HPX flags.
# ---------------------------------------------------------------------------
_SMOKE = (
    "import sys, hpxpy; "
    "hpxpy.init(2, hpx_args=sys.argv[1:]); "
    "print(f\"{hpxpy.locality_id()} {hpxpy.num_localities()} "
    "{hpxpy.distributed_sum(float(hpxpy.locality_id()+1))}\", flush=True); "
    "hpxpy.finalize()"
)


def _two_free_ports():
    """Return (p0, p1): two free TCP port numbers, or skip the test."""
    socks = []
    ports = []
    try:
        for _ in range(2):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("", 0))
            except OSError as exc:
                for o in socks:
                    o.close()
                pytest.skip(f"cannot bind a free port: {exc}")
            ports.append(s.getsockname()[1])
            socks.append(s)
    finally:
        for s in socks:
            s.close()
    return ports[0], ports[1]


def test_two_locality_distributed_sum():
    """Console + worker both report locality count 2 and all-reduce sum 3.0."""
    p0, p1 = _two_free_ports()
    host = socket.gethostname()

    common = [
        "--hpx:localities=2",
        f"--hpx:agas={host}:{p0}",
    ]

    console_cmd = [
        sys.executable, "-c", _SMOKE,
        *common,
        f"--hpx:hpx={host}:{p0}",
    ]
    worker_cmd = [
        sys.executable, "-c", _SMOKE,
        *common,
        f"--hpx:hpx={host}:{p1}",
        "--hpx:worker",
    ]

    # Start console first so its AGAS server is up before the worker connects.
    console = subprocess.Popen(
        console_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=os.environ,
    )
    time.sleep(0.5)
    worker = subprocess.Popen(
        worker_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=os.environ,
    )

    try:
        console_stdout, console_stderr = console.communicate(timeout=90)
        worker_stdout, worker_stderr = worker.communicate(timeout=90)
    except subprocess.TimeoutExpired:
        console.kill()
        worker.kill()
        console_err = ""
        worker_err = ""
        try:
            _, console_err = console.communicate(timeout=5)
        except Exception:
            pass
        try:
            _, worker_err = worker.communicate(timeout=5)
        except Exception:
            pass
        pytest.fail(
            "2-locality launch timed out after 90 s.\n"
            f"console stderr:\n{console_err}\n"
            f"worker stderr:\n{worker_err}"
        )

    assert console.returncode == 0, (
        f"console exited {console.returncode}\nstderr:\n{console_stderr}"
    )
    assert worker.returncode == 0, (
        f"worker exited {worker.returncode}\nstderr:\n{worker_stderr}"
    )

    assert "0 2 3.0" in console_stdout, (
        f"expected '0 2 3.0' in console stdout, got:\n{console_stdout}"
    )
    assert "1 2 3.0" in worker_stdout, (
        f"expected '1 2 3.0' in worker stdout, got:\n{worker_stdout}"
    )
