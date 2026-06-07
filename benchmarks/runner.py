"""Microbenchmark runner: median-of-times throughput, thread scaling, and the
abstraction penalty (hpxpy median ÷ hand-written C++ HPX baseline median).

HPX fixes its worker-thread count once per process (``--hpx:threads``); it cannot
be changed in-process. So a thread sweep does NOT loop in one process — it spawns
ONE subprocess per thread count, re-invoking this module as
``python -m benchmarks.runner point``, each measuring its points and emitting one
JSON line per (op, n). The optional C++ baseline is a separate binary obeying the
same "point contract" (below); the sweep shells out to it and reports the penalty.

Point contract (so the C++ baseline in cpp_baseline/ matches this runner): given
``--op <name> --sizes <csv> --threads <N> --repeats <R>``, emit one JSON object
per size on stdout, one per line::

    {"op": "sum", "n": 10000000, "threads": 40, "impl": "cpp", "median_s": 0.0123}

``threads`` is the ACTUAL worker count after the runtime clamps the request.

The measurement/aggregation core (median_time, gelem_per_s, parse/plan helpers,
write_csv, abstraction_penalty, scaling) is import-light and unit-tested; the HPX,
subprocess, and CLI shell around it is integration glue.

Usage (on an EXCLUSIVE node for clean numbers):
    # one point group (one process, fixed thread count) -> JSON lines on stdout
    LD_PRELOAD=$HPXPY_TCMALLOC python -m benchmarks.runner point \\
        --op sum --sizes 1e7 --threads 8

    # full sweep -> CSV + penalty/scaling tables; one point subprocess per count
    LD_PRELOAD=$HPXPY_TCMALLOC python -m benchmarks.runner sweep \\
        --op sum --sizes 1e6,1e7,1e8 --threads 1,2,4,8,16,32,40 \\
        --out results.csv [--baseline ./cpp_baseline/build/bench]
"""
from __future__ import annotations

import csv
import statistics
import time
from dataclasses import asdict, dataclass

# Implementations recorded in results; "hpxpy" is this binding, "cpp" the C++ HPX
# baseline (the abstraction-penalty denominator).
IMPL_HPXPY = "hpxpy"
IMPL_CPP = "cpp"

CSV_FIELDS = ["op", "n", "threads", "impl", "median_s", "gelem_s"]


# --- pure measurement core --------------------------------------------------

def median_time(fn, warmup: int = 1, repeats: int = 5) -> float:
    """Median wall-clock seconds of ``fn`` over ``repeats`` timed runs.

    ``warmup`` untimed runs precede timing (caches/first-touch/JIT). Median (not
    mean) rejects outliers from scheduler jitter — the project's timing convention.
    """
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def gelem_per_s(n: int, seconds: float) -> float:
    """Throughput in giga-elements/second; 0.0 for non-positive time."""
    if seconds <= 0.0:
        return 0.0
    return n / seconds / 1e9


@dataclass(frozen=True)
class Measurement:
    """One timed point: operation ``op`` on ``n`` elements with ``threads`` HPX
    workers, for implementation ``impl``, taking ``median_s`` seconds."""

    op: str
    n: int
    threads: int
    impl: str
    median_s: float

    @property
    def gelem_s(self) -> float:
        return gelem_per_s(self.n, self.median_s)

    def row(self) -> dict:
        return {**asdict(self), "gelem_s": self.gelem_s}


# --- argument parsing helpers -----------------------------------------------

def parse_sizes(spec: str) -> list[int]:
    """``"1e6,1e7,2_000_000"`` -> ``[1000000, 10000000, 2000000]`` (positive ints)."""
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        n = int(float(tok))
        if n <= 0:
            raise ValueError(f"size must be positive: {tok!r}")
        out.append(n)
    if not out:
        raise ValueError("no sizes given")
    return out


def parse_threads(spec: str) -> list[int]:
    """``"1,2,4,8"`` -> ``[1, 2, 4, 8]`` (positive ints)."""
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        t = int(tok)
        if t <= 0:
            raise ValueError(f"threads must be positive: {tok!r}")
        out.append(t)
    if not out:
        raise ValueError("no thread counts given")
    return out


# --- aggregation ------------------------------------------------------------

def write_csv(measurements, path) -> None:
    """Write measurements as a tidy CSV (one row per point)."""
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for m in measurements:
            w.writerow(m.row())


def abstraction_penalty(measurements) -> list[dict]:
    """For each (op, n, threads) with both hpxpy and cpp present, the penalty
    ``hpxpy_median / cpp_median`` (≈1.0 = zero penalty; >1.0 = hpxpy slower)."""
    by_key: dict = {}
    for m in measurements:
        by_key.setdefault((m.op, m.n, m.threads), {})[m.impl] = m
    out = []
    for (op, n, threads), impls in sorted(by_key.items()):
        if IMPL_HPXPY in impls and IMPL_CPP in impls:
            cpp = impls[IMPL_CPP].median_s
            penalty = impls[IMPL_HPXPY].median_s / cpp if cpp > 0 else 0.0
            out.append({"op": op, "n": n, "threads": threads, "penalty": penalty})
    return out


def scaling(measurements, impl: str = IMPL_HPXPY) -> list[dict]:
    """Per (op, n) speedup vs a single thread for one impl: ``median[1]/median[t]``.
    Points without a 1-thread reference are skipped."""
    by_on: dict = {}
    for m in measurements:
        if m.impl == impl:
            by_on.setdefault((m.op, m.n), {})[m.threads] = m.median_s
    out = []
    for (op, n), by_t in sorted(by_on.items()):
        if 1 not in by_t:
            continue
        base = by_t[1]
        for t, s in sorted(by_t.items()):
            out.append({"op": op, "n": n, "threads": t,
                        "speedup": base / s if s > 0 else 0.0})
    return out


# --- op registry + HPX execution (integration glue) -------------------------

def available_ops() -> list[str]:
    """Names the runner can measure. Grows as milestones add Array operations."""
    return ["sum"]


def _build_op(op: str, n: int):  # pragma: no cover - needs numpy + HPX runtime
    """Return a zero-argument callable that runs ``op`` on ``n`` elements.

    M0/M1: only the zero-copy ``sum`` spike exists. M2 adds native Array
    reductions (sum/min/max/dot) here — registered ONCE, reused by every gate.
    """
    import numpy as np

    import hpxpy as hpx

    if op == "sum":
        a = np.arange(n, dtype=np.float64)
        return lambda: hpx.sum(a)
    raise ValueError(f"unknown op: {op!r} (have {available_ops()})")


def run_points(op, sizes, threads, repeats, warmup):  # pragma: no cover - HPX
    """Measure ``op`` over ``sizes`` in THIS process with ``threads`` HPX workers."""
    import hpxpy as hpx

    hpx.init(threads)
    actual = hpx.num_worker_threads()
    out = []
    for n in sizes:
        fn = _build_op(op, n)
        med = median_time(fn, warmup=warmup, repeats=repeats)
        out.append(Measurement(op=op, n=n, threads=actual,
                               impl=IMPL_HPXPY, median_s=med))
    return out


def _spawn_points(argv, impl):  # pragma: no cover - subprocess/JSON
    """Run a point subprocess (hpxpy re-invocation or C++ baseline) and parse its
    JSON-line output into Measurements; ``impl`` labels rows lacking one."""
    import json
    import subprocess

    res = subprocess.run(argv, capture_output=True, text=True, check=True)
    out = []
    for line in res.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        d = json.loads(line)
        out.append(Measurement(op=d["op"], n=int(d["n"]), threads=int(d["threads"]),
                               impl=d.get("impl", impl), median_s=float(d["median_s"])))
    return out


# --- CLI --------------------------------------------------------------------

def _cmd_point(args):  # pragma: no cover - HPX runtime + stdout protocol
    import json

    sizes = parse_sizes(args.sizes)
    for m in run_points(args.op, sizes, args.threads, args.repeats, args.warmup):
        print(json.dumps(m.row()))


def _cmd_sweep(args):  # pragma: no cover - subprocess orchestration + I/O
    import sys

    sizes = parse_sizes(args.sizes)
    threads = parse_threads(args.threads)
    sizes_spec = ",".join(str(n) for n in sizes)

    measurements = []
    for t in threads:
        measurements += _spawn_points(
            [sys.executable, "-m", "benchmarks.runner", "point",
             "--op", args.op, "--sizes", sizes_spec,
             "--threads", str(t), "--repeats", str(args.repeats),
             "--warmup", str(args.warmup)],
            IMPL_HPXPY)
        if args.baseline:
            measurements += _spawn_points(
                [args.baseline, "--op", args.op, "--sizes", sizes_spec,
                 "--threads", str(t), "--repeats", str(args.repeats),
                 "--warmup", str(args.warmup)],
                IMPL_CPP)

    write_csv(measurements, args.out)
    print(f"wrote {len(measurements)} rows -> {args.out}")

    print("\nthroughput (GElem/s):")
    for m in measurements:
        print(f"  {m.impl:>6} {m.op:>5} n={m.n:>12} t={m.threads:>3} "
              f"{m.gelem_s:8.2f}")

    sc = scaling(measurements, IMPL_HPXPY)
    if sc:
        print("\nhpxpy scaling (speedup vs 1 thread):")
        for r in sc:
            print(f"  {r['op']:>5} n={r['n']:>12} t={r['threads']:>3} "
                  f"{r['speedup']:6.2f}x")

    pen = abstraction_penalty(measurements)
    if pen:
        print("\nabstraction penalty (hpxpy / cpp, ~1.0 = none):")
        for r in pen:
            print(f"  {r['op']:>5} n={r['n']:>12} t={r['threads']:>3} "
                  f"{r['penalty']:6.3f}")
    elif args.baseline:
        print("\n(no overlapping hpxpy/cpp points — penalty not computed)")


def build_parser():  # pragma: no cover - argparse wiring
    import argparse

    p = argparse.ArgumentParser(description="hpxpy microbenchmark runner")
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("point", help="measure points in THIS process (fixed threads)")
    pp.add_argument("--op", required=True, choices=available_ops())
    pp.add_argument("--sizes", required=True, help="comma-separated element counts")
    pp.add_argument("--threads", type=int, default=0, help="HPX workers (0=all)")
    pp.add_argument("--repeats", type=int, default=7)
    pp.add_argument("--warmup", type=int, default=1)
    pp.set_defaults(func=_cmd_point)

    sw = sub.add_parser("sweep", help="thread/size sweep via point subprocesses")
    sw.add_argument("--op", required=True, choices=available_ops())
    sw.add_argument("--sizes", required=True, help="comma-separated element counts")
    sw.add_argument("--threads", required=True, help="comma-separated thread counts")
    sw.add_argument("--out", default="bench_results.csv")
    sw.add_argument("--baseline", default="", help="path to C++ baseline binary")
    sw.add_argument("--repeats", type=int, default=7)
    sw.add_argument("--warmup", type=int, default=1)
    sw.set_defaults(func=_cmd_sweep)
    return p


def main(argv=None):  # pragma: no cover - CLI entry
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
