# HPXPy

A thin Python array library backed by the [HPX](https://hpx.stellar-group.org/) C++
runtime — array operations are real HPX parallel algorithms, with a measured
**abstraction penalty of ~1.0** vs hand-written C++ HPX.

📖 **Docs:** https://lums658.github.io/hpxpy/

> **Status: Phase A (single-node usability), active.**
> The single-locality foundation (N-D arrays, dtypes, ufuncs, reductions, constructors,
> views, broadcasting, NumPy bridge) is merged and tested at ~zero abstraction penalty.
> See [docs/PLAN.md](docs/PLAN.md) for the full roadmap.

## What works today

### Array type and dtypes

- N-D arrays in **float64**, **float32**, and **int64** (row-major, C-order), backed by
  `hpx::compute::vector` with NUMA-aware parallel first-touch allocation.
- NumPy-faithful **type promotion**: mixed-dtype binary ops follow NumPy's result-type
  rules (e.g. `int64 + float32 → float64`); integer division always promotes to float64;
  scalar operands promote by value (integer scalars keep the array's dtype, floating-point
  scalars promote to at least float64).
- `astype(dtype)` for explicit element-wise cast between supported dtypes.

### Constructors

```
zeros(shape, dtype)   ones(shape, dtype)   full(shape, value, dtype)
arange(start, stop, step, dtype)           linspace(start, stop, num, endpoint, dtype)
eye(N, M, k, dtype)  identity(n, dtype)   empty(shape, dtype)
zeros_like(a)        ones_like(a)         empty_like(a)         full_like(a, value)
```

`shape` may be an `int` (1-D) or a tuple/list of ints (N-D).

### Indexing and slicing

- **Scalar indexing**: `a[i]` (1-D), `a[i, j]` (N-D), negative indices, bounds-checked.
- **Multi-axis slicing**: `a[i:j, ::2, ...]` — full slice syntax (start/stop/step,
  negative indices, strided, reversed) on any combination of axes.
- **Ellipsis** (`...`): expands to fill remaining axes, e.g. `a[..., 0]`.
- All slices return zero-copy **views** that share memory with the parent.

### Views (zero-copy)

- `transpose(a, axes)` / `a.transpose(axes)` / `a.T` — permute or reverse axes.
- `reshape(a, shape)` / `a.reshape(shape)` — contiguous input returns a view; otherwise
  a copy. One `-1` dimension is inferred.
- `a.ravel()` — flatten to 1-D (view when contiguous).
- `squeeze(a, axis)` / `a.squeeze(axis)` — remove size-1 dimensions.
- `expand_dims(a, axis)` / `a.expand_dims(axis)` — insert a size-1 dimension.

### Arithmetic and broadcasting

- Element-wise `+  -  *  /` between arrays and between arrays and scalars, with
  **NumPy-style broadcasting** over N-D shapes (trailing-dimension alignment, size-1
  expansion).
- Operators `**` (power), `%` (mod), `//` (floor divide) are supported.
- Reflected operators (`2.0 - a`, `1.0 / a`, etc.) supported.

### Reductions

All reductions accept `axis` (int or tuple of ints) and `keepdims`.

| Function | Description |
|---|---|
| `sum(a, axis, keepdims)` | Parallel sum (`hpx::reduce`) |
| `min(a, axis, keepdims)` | Parallel minimum |
| `max(a, axis, keepdims)` | Parallel maximum |
| `mean(a, axis, keepdims)` | Arithmetic mean (always float64) |
| `prod(a, axis, keepdims)` | Product (preserves dtype) |
| `any(a)` | True if any element is nonzero |
| `all(a)` | True if all elements are nonzero |
| `count_nonzero(a)` | Count of nonzero elements |
| `std(a)` | Standard deviation (axis=None; two-pass) |
| `var(a)` | Variance (axis=None; two-pass) |

`any` / `all` / `count_nonzero` / `std` / `var` currently support `axis=None` only.

### Linear algebra

- `dot(a, b)` — 1-D inner product (fused `transform_reduce`); 2-D matrix product.
- `matmul(a, b)` / `a @ b` — 2-D matrix–matrix product.

### Ufuncs (element-wise)

**Unary — preserve dtype:** `negative`, `abs`, `sign`, `square`

**Unary — integer input → float64:** `sqrt`, `exp`, `log`, `sin`, `cos`, `tan`,
`floor`, `ceil`, `trunc`, `round` / `rint`

**Unary — logarithm variants:** `exp2`, `log2`, `log10`

**Unary — hyperbolic:** `sinh`, `cosh`, `tanh`

**Binary (both operands; type-promotes):** `maximum`, `minimum`, `power`, `mod`,
`floor_divide`

**Conditional:** `where(condition, x, y)` — element-wise select (nonzero = True,
matches NumPy semantics).

**Other:** `clip(a, lo, hi)` — clamp to `[lo, hi]` (preserves dtype)

### Sorting, scanning, and selection

- `sort(a)` — returns a new ascending-sorted array (like `numpy.sort`).
- `a.sort()` — in-place sort (like `numpy.ndarray.sort`).
- `cumsum(a)` / `a.cumsum()` — inclusive prefix sum (parallel `hpx::inclusive_scan`).
- `cumprod(a)` / `a.cumprod()` — inclusive prefix product.
- `argsort(a)` / `a.argsort()` — I64 array of sort-order indices.

### Statistics

| Function | Description |
|---|---|
| `std(a)` | Standard deviation (two-pass, axis=None) |
| `var(a)` | Variance (two-pass, axis=None) |

### Random (`hpx.random`)

The `hpx.random` submodule uses splitmix64 hashing with a global (seed, counter)
state — every element is index-addressable, so results are fully reproducible
across different thread counts.

```python
hpx.random.seed(42)              # set seed; reset counter to 0
hpx.random.rand(1000)            # uniform [0, 1) Array, 1-D
hpx.random.randn(100, 100)       # normal (Box-Muller), N-D shape
hpx.random.uniform(0.0, 1.0, 500)
hpx.random.randint(0, 10, 200)   # int64 Array

s = hpx.random.get_seed()        # reproducibility: save seed
state = hpx.random.get_state()   # (seed, counter) tuple
hpx.random.set_state(*state)     # restore exact point
```

The default seed comes from `std::random_device` (truly random). To reproduce a
run, save `get_seed()` before generation or use `set_state()` to checkpoint.

### Zero-copy NumPy bridge

- `from_numpy(a, copy=True)` — bring a float64/float32/int64 C-contiguous NumPy array
  into an `Array`. `copy=False` borrows the buffer zero-copy (memory shared both ways).
  Unsupported dtypes or non-contiguous input raise `TypeError`.
- `to_numpy(a)` — zero-copy writable NumPy view of an `Array` (shares memory).
- `__array__` protocol — `np.asarray(hpx_array)` works directly.

### Sparse matrices

- `CsrMatrix` — CSR float64 matrix. `csr_from`, `laplacian_1d`, `spmv`, `spmm`,
  `DenseMatrix`, `dense_zeros`, `dense_from`.

### Runtime

- `init(num_threads, hpx_args)` — start the HPX runtime (idempotent; auto-finalizes at
  exit). `num_threads=None` uses all cores.
- `finalize()`, `is_running()`, `num_threads()` / `num_worker_threads()`, `hpx_version()`.
- `array(data, dtype=None)` — construct from a Python list or NumPy array.
- `ndarray` alias for `Array`; `runtime` context manager (`with hpx.runtime(): ...`).
- Multi-locality primitives: `num_localities()`, `locality_id()`, `is_console()`,
  `is_worker()`, `distributed_sum(local)` (all-reduce over localities).

## What is NOT there yet

- **Comparison operators and boolean dtype** (`a > b`, `a == b`, `bool` arrays)
  — planned for Wave 3.
- **Distributed global-view Array** — a partitioned `Array` over multiple HPX localities
  for transparent distributed computation. The multi-locality runtime exists today
  (`num_localities`, `distributed_sum`), but the data-parallel distributed array is a
  future milestone (Phase B).
- **GPU / heterogeneous execution** — planned for Phase C, after distributed.
- **pip wheels / conda packages** — build from source for now (see below). A distribution
  story is on the roadmap.
- **`any`/`all`/`count_nonzero` with axis argument** — currently axis=None only.
- **Advanced algorithms**: `flip`, `nonzero`, `searchsorted`, `percentile`,
  `concatenate`/`stack`/`split`, advanced indexing (boolean masking, integer-array
  indexing), `argmin`/`argmax` — planned for Wave 4/5.
- **Distributed collective bindings**: Python-level `all_reduce`, `broadcast`,
  `gather`, `scatter` — the C++ multi-locality substrate exists; Python wrappers
  are a Phase B item.

## Quick start

```python
import hpxpy as hpx
import numpy as np

hpx.init()                              # start HPX runtime (all cores)

# constructors
a = hpx.arange(1_000_000)              # int64 [0, 1, ..., 999999]
b = hpx.linspace(0.0, 1.0, 1_000_000) # float64
c = hpx.zeros((1000, 1000))            # 2-D float64

# NumPy bridge
arr = np.arange(1_000_000, dtype=np.float64)
x = hpx.from_numpy(arr)                # NUMA-aware copy
back = hpx.to_numpy(x)                 # zero-copy view

# arithmetic with broadcasting
row = hpx.ones((1, 4))
col = hpx.ones((3, 1))
result = row + col                      # (3, 4) broadcast

# reductions
print(x.sum())                          # scalar, parallel hpx::reduce
m = hpx.zeros((4, 5))
print(hpx.sum(m, axis=0))              # shape (5,)

# views
t = c.T                                 # transpose, zero-copy
s = c[10:20, ::2]                       # strided slice view

# ufuncs
hpx.sqrt(hpx.arange(1_000_000, dtype="float64"))

hpx.finalize()
```

## Requirements

- A built/installed **HPX** (pinned), found via `find_package(HPX)`.
- **Python >= 3.13**, CMake >= 3.18, a C++20 compiler, **nanobind** (build dep).

## Building (on Rostam)

```bash
source env.sh            # toolchain: gcc 15 / Boost 1.90 / Py 3.13 / HPX paths
bash scripts/check.sh    # build (editable) + lint + tests — the local==CI gate
```

Or manually:
```bash
source env.sh
pip install -e . -C cmake.define.CMAKE_PREFIX_PATH=$HPX_ROOT -C cmake.define.HPX_DIR=$HPX_DIR
LD_PRELOAD=$HPXPY_TCMALLOC pytest          # HPX uses tcmalloc; preload at runtime
```

## Repository layout

```
src/               C++ wrapper (array.hpp) + per-feature nanobind TUs (bind_*.cpp)
hpxpy/             Python package (__init__.py wraps _core + pure-Python helpers)
tests/             pytest: correctness (analytic) + NumPy parity (~302 tests, 100% cov)
benchmarks/        single-node + cloud/ distributed harness (medusa-as-cloud)
cpp_baseline/      hand-written C++ HPX baseline + in-binary penalty ladder
docs/              design notes, roadmap (PLAN.md)
```

## License

MIT — see [LICENSE](LICENSE).
