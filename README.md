# HPXPy

A thin Python array library backed by the [HPX](https://hpx.stellar-group.org/) C++
runtime — array operations *are* real HPX parallel algorithms, with a measured
**abstraction penalty of ~1.0** vs hand-written C++ HPX.

📖 **Docs:** https://lums658.github.io/hpxpy/

> **Status: single-node NumPy surface — merged and zero-penalty.** Over the NUMA-aware
> `Array`:
> - **N-D arrays** in **float64 / float32 / int64** with NumPy **type promotion**
> - construction: `zeros`/`ones`/`full`/`arange`/`linspace`/`eye`/`empty` + `*_like`
> - indexing + multi-axis **slicing** (ellipsis, negative steps) and zero-copy **views**
>   (`transpose`/`.T`/`reshape`/`ravel`/`squeeze`/`expand_dims`)
> - **broadcasting** `+ - * /`; **ufuncs** (`sqrt`/`exp`/`log`/`sin`/`cos`/`abs`/`clip`/
>   `maximum`/`minimum`/`power`/…); **reductions** (`sum`/`min`/`max`/`mean`/`prod`/`any`/
>   `all` with `axis=`/`keepdims=`), `dot`, **`matmul` / `@`**; `sort`/`cumsum`
> - zero-copy **NumPy bridge** (`from_numpy`/`to_numpy`/`__array__`); sparse CSR (`spmv`/`spmm`)
>
> Abstraction penalty stays **≈1.0** — verified single-thread, scaling to 40 cores, and at
> the memory/compute roofline. *Not yet:* comparisons / `where` / bool dtype, fancy indexing,
> a distributed global-view array, GPU, and pip wheels. See [docs/PLAN.md](docs/PLAN.md).

## Goals

- **Thin HPX wrapper, ~zero abstraction penalty**: each op is a single HPX algorithm
  over the array's buffer — no copies, no between-layers buffers. `hpxpy ÷ C++-HPX ≈ 1.0`,
  measured every iteration against a hand-written baseline.
- **Backed by HPX**: one array type over a NUMA-aware `hpx::compute::vector`
  (`block_allocator`, parallel first-touch) — no hidden serial fallbacks.
- **NumPy-faithful**: NumPy semantics and dtype promotion across the surface, with a
  zero-copy `from_numpy` / `to_numpy` bridge for interop.

## Requirements

- A built/installed **HPX** (pinned), found via `find_package(HPX)`.
- **Python ≥ 3.13**, CMake ≥ 3.18, a C++20 compiler, **nanobind** (build dep).

## Building (on Rostam)

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
import hpxpy as hpx
hpx.init()                          # start the HPX runtime (all cores)
a = hpx.arange(1_000_000)           # NUMA-aware Array [0, 1, ..., n-1]
print(hpx.num_worker_threads(), a.sum(), a[2:5].sum())   # parallel reductions
hpx.finalize()
```

## Repository layout

```
src/          C++ wrapper (array.hpp, timing.hpp) + nanobind bindings (_core)
hpxpy/        Python package
tests/        pytest: correctness (analytic) + lifecycle
benchmarks/   runner + results (throughput / scaling / abstraction penalty)
cpp_baseline/ hand-written C++ HPX baseline + the in-binary penalty ladder (diag)
docs/         Sphinx site (PLAN.md = architecture + process)
```

## License

MIT — see [LICENSE](LICENSE).
