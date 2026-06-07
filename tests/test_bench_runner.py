"""Unit tests for the microbenchmark runner's pure measurement/aggregation core.

These exercise the import-light logic (no HPX, no subprocess): timing, throughput,
arg parsing, CSV, abstraction penalty, and scaling. The HPX/subprocess/CLI shell
is integration glue (marked no-cover) and validated by running an actual sweep.
"""
import csv

import pytest

from benchmarks import runner
from benchmarks.runner import (
    Measurement,
    abstraction_penalty,
    gelem_per_s,
    median_time,
    parse_sizes,
    parse_threads,
    scaling,
    write_csv,
)


# --- median_time ------------------------------------------------------------

def test_median_time_counts_calls():
    calls = {"warmup": 0, "timed": 0}

    def fn():
        calls["timed"] += 1

    med = median_time(fn, warmup=2, repeats=5)
    assert med >= 0.0
    # warmup runs are untimed but still invoked
    assert calls["timed"] == 2 + 5


def test_median_time_is_median_not_mean():
    # 5 runs with one large outlier: median ignores it, mean would not.
    sleeps = iter([0.0, 0.0, 0.0, 0.0, 0.05])

    def fn():
        import time
        time.sleep(next(sleeps))

    med = median_time(fn, warmup=0, repeats=5)
    assert med < 0.01  # the 0.05 outlier did not pull the median up


def test_median_time_rejects_bad_args():
    with pytest.raises(ValueError):
        median_time(lambda: None, repeats=0)
    with pytest.raises(ValueError):
        median_time(lambda: None, warmup=-1)


# --- gelem_per_s ------------------------------------------------------------

def test_gelem_per_s_basic():
    assert gelem_per_s(1_000_000_000, 1.0) == pytest.approx(1.0)
    assert gelem_per_s(2_000_000_000, 1.0) == pytest.approx(2.0)


def test_gelem_per_s_nonpositive_time_is_zero():
    assert gelem_per_s(1_000_000, 0.0) == 0.0
    assert gelem_per_s(1_000_000, -1.0) == 0.0


# --- Measurement ------------------------------------------------------------

def test_measurement_gelem_and_row():
    m = Measurement(op="sum", n=1_000_000_000, threads=40, impl="hpxpy",
                    median_s=0.5)
    assert m.gelem_s == pytest.approx(2.0)
    row = m.row()
    assert row["op"] == "sum" and row["n"] == 1_000_000_000
    assert row["threads"] == 40 and row["impl"] == "hpxpy"
    assert row["median_s"] == pytest.approx(0.5)
    assert row["gelem_s"] == pytest.approx(2.0)


# --- parsing ----------------------------------------------------------------

def test_parse_sizes_accepts_sci_and_underscores():
    assert parse_sizes("1e6,1e7,2_000_000") == [1_000_000, 10_000_000, 2_000_000]
    assert parse_sizes(" 100 , 200 ") == [100, 200]


def test_parse_sizes_rejects_nonpositive_and_empty():
    with pytest.raises(ValueError):
        parse_sizes("0")
    with pytest.raises(ValueError):
        parse_sizes("-5")
    with pytest.raises(ValueError):
        parse_sizes(" , ")


def test_parse_threads():
    assert parse_threads("1,2,4,8,40") == [1, 2, 4, 8, 40]
    with pytest.raises(ValueError):
        parse_threads("0")
    with pytest.raises(ValueError):
        parse_threads("")


# --- write_csv --------------------------------------------------------------

def test_write_csv_roundtrip(tmp_path):
    rows = [
        Measurement("sum", 1_000_000, 1, "hpxpy", 0.001),
        Measurement("sum", 1_000_000, 1, "cpp", 0.0008),
    ]
    out = tmp_path / "r.csv"
    write_csv(rows, out)
    with open(out, newline="") as f:
        got = list(csv.DictReader(f))
    assert [r["impl"] for r in got] == ["hpxpy", "cpp"]
    assert got[0]["op"] == "sum"
    assert float(got[0]["gelem_s"]) == pytest.approx(1.0)  # 1e6 / 1e-3 / 1e9


# --- abstraction_penalty ----------------------------------------------------

def test_abstraction_penalty_pairs_hpxpy_over_cpp():
    rows = [
        Measurement("sum", 10**7, 40, "hpxpy", 0.0015),
        Measurement("sum", 10**7, 40, "cpp", 0.0010),
        Measurement("sum", 10**7, 1, "hpxpy", 0.05),  # no cpp pair -> skipped
    ]
    pen = abstraction_penalty(rows)
    assert len(pen) == 1
    assert pen[0]["threads"] == 40
    assert pen[0]["penalty"] == pytest.approx(1.5)


def test_abstraction_penalty_empty_without_pairs():
    rows = [Measurement("sum", 10**7, 40, "hpxpy", 0.0015)]
    assert abstraction_penalty(rows) == []


# --- scaling ----------------------------------------------------------------

def test_scaling_speedup_vs_one_thread():
    rows = [
        Measurement("sum", 10**7, 1, "hpxpy", 0.040),
        Measurement("sum", 10**7, 2, "hpxpy", 0.020),
        Measurement("sum", 10**7, 4, "hpxpy", 0.010),
        Measurement("sum", 10**7, 4, "cpp", 0.009),  # other impl ignored
    ]
    sc = scaling(rows, "hpxpy")
    by_t = {r["threads"]: r["speedup"] for r in sc}
    assert by_t[1] == pytest.approx(1.0)
    assert by_t[2] == pytest.approx(2.0)
    assert by_t[4] == pytest.approx(4.0)


def test_scaling_skips_when_no_single_thread_reference():
    rows = [Measurement("sum", 10**7, 4, "hpxpy", 0.010)]
    assert scaling(rows, "hpxpy") == []


# --- registry ---------------------------------------------------------------

def test_available_ops_includes_sum():
    assert "sum" in runner.available_ops()
