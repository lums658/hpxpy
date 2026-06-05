# HPXPy

A NumPy-compatible Python array library backed by the [HPX](https://hpx.stellar-group.org/)
C++ runtime — array operations execute as real HPX parallel computations (shared-memory
now, distributed later).

> **Status: M0 (substrate).** The full toolchain is up — scikit-build-core → CMake →
> nanobind, linked against installed HPX, with a managed HPX runtime and a zero-copy
> `nb::ndarray` → HPX parallel path proven by tests. Real array types and operations
> arrive in the next milestones. See [docs/PLAN.md](docs/PLAN.md) for architecture +
> process.

## Goals

- **NumPy-replaceable** Python API (`arange`, `array`, `a*b`, `a@b`, `dot`, reductions…).
- **Backed by HPX**: one array type over `hpx::partitioned_vector`, NUMA-aware
  allocation, fused/lazy expression evaluation — no hidden serial fallbacks.
- **Measured**: every operation is benchmarked against hand-written C++ HPX and NumPy
  (throughput, scalability, abstraction penalty) and checked for NumPy idiom parity.

## Requirements

- A built/installed **HPX** (pinned), found via `find_package(HPX)`.
- **Python ≥ 3.13**, CMake ≥ 3.18, a C++20 compiler, **nanobind** (build dep).

## Building (M0, on Rostam)

```bash
source env.sh            # toolchain: gcc 15 / Boost 1.90 / Py 3.13 / HPX paths
bash scripts/check.sh    # build (editable) + lint + tests  — the local==CI gate
```

Or manually:
```bash
source env.sh
pip install -e . -C cmake.define.CMAKE_PREFIX_PATH=$HPX_ROOT -C cmake.define.HPX_DIR=$HPX_DIR
LD_PRELOAD=$HPXPY_TCMALLOC pytest          # HPX uses tcmalloc; preload at runtime
```

Quick check:
```python
import hpxpy as hpx, numpy as np
hpx.init()
print(hpx.num_worker_threads(), hpx.sum(np.arange(1e6)))   # parallel, zero-copy
hpx.finalize()
```

## Repository layout

```
src/          C++ nanobind bindings (_core)
hpxpy/        Python package
tests/        pytest: correctness + NumPy-parity
benchmarks/   microbenchmarks (throughput / scalability / abstraction penalty)
cpp_baseline/ hand-written C++ HPX reference kernels (perf ceiling)
docs/PLAN.md  architecture + development process
```

## License

MIT — see [LICENSE](LICENSE).
