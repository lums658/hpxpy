# HPXPy

A NumPy-compatible Python array library backed by the [HPX](https://hpx.stellar-group.org/)
C++ runtime — array operations execute as real HPX parallel computations (shared-memory
now, distributed later).

> **Status: early.** This is a ground-up reimplementation, built incrementally with
> correctness, performance, and API-parity gates at every step. See
> [docs/PLAN.md](docs/PLAN.md) for the architecture and the development process.

## Goals

- **NumPy-replaceable** Python API (`arange`, `array`, `a*b`, `a@b`, `dot`, reductions…).
- **Backed by HPX**: one array type over `hpx::partitioned_vector`, NUMA-aware
  allocation, fused/lazy expression evaluation — no hidden serial fallbacks.
- **Measured**: every operation is benchmarked against hand-written C++ HPX and NumPy
  (throughput, scalability, abstraction penalty) and checked for NumPy idiom parity.

## Requirements

- A built/installed **HPX** (pinned version) — found via `find_package(HPX)`.
- **Python ≥ 3.13**, CMake, a C++20 compiler, nanobind (build dep).
- Build/runtime environment is captured in `env.sh` (see docs).

## Building

Build instructions land with milestone **M0** (CMake + scikit-build-core against an
installed HPX). Until then, see [docs/PLAN.md](docs/PLAN.md).

## License

MIT — see [LICENSE](LICENSE).
