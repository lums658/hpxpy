"""Shared test fixtures.

HPX can only be started/stopped once per process, so the runtime is brought up
once per test session and torn down at the end.
"""
import pytest

import hpxpy as hpx


@pytest.fixture(scope="session", autouse=True)
def hpx_runtime():
    hpx.init()
    yield
    hpx.finalize()
