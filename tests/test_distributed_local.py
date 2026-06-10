"""Single-locality smoke tests for M4 distributed-runtime wrappers.

These tests run under the normal pytest session (no ``distributed`` marker) using
the autouse ``hpx_runtime`` fixture in conftest.py.  They cover every new public
wrapper so that coverage stays at 100 % without needing a multi-process launch.
"""
import hpxpy


def test_num_localities_single():
    assert hpxpy.num_localities() == 1


def test_locality_id_single():
    assert hpxpy.locality_id() == 0


def test_here_alias_single():
    assert hpxpy.here() == 0


def test_is_console_single():
    assert hpxpy.is_console() is True


def test_is_worker_single():
    assert hpxpy.is_worker() is False


def test_distributed_sum_identity():
    # Single locality: distributed_sum returns its argument unchanged.
    assert hpxpy.distributed_sum(5.0) == 5.0
