# HPXPy (rewrite) — Plan & Process

**Status:** ACTIVE. Phase 1 (thin HPX wrapper, single-locality) complete — M1–M3, slices +
strided views, and M5 sparse (SpMV/SpMM), all at ~0 abstraction penalty; the NumPy bridge +
drop-in parity suite landed; **M4a distributed runtime merged** (multi-locality via the TCP
parcelport). Now in **Phase A (single-node usability)**, starting with N-D arrays — see §8.
Repo: `github.com/lums658/hpxpy`. Created 2026-06-04; last validated 2026-06-10.

A from-scratch reimplementation of HPXPy: a NumPy-compatible Python array library
backed by the HPX C++ runtime, built incrementally with correctness + benchmark gates
at every iteration. This document is the contract for *how* we build it.

---

## 1. Vision & goals

Built in **two phases** (see §8). **Phase 1 first: a thin, ~zero-overhead wrap of HPX.
Phase 2 later: NumPy compatibility — and only after Phase 1 is validated.**

- **Thin HPX wrapper, ~zero abstraction penalty**: the data structure *is* an HPX
  container (`hpx::compute::vector` with a NUMA-aware `block_allocator` for the
  single-locality core; a distributed layer added later in M4); operations *are* HPX
  algorithms over it. Data stays in HPX — **no copies, no reassembly, no between-layers
  buffers** (allocate only for genuinely new arrays). HPXPy ÷ hand-written C++ HPX ≈ 1.0,
  measured every iteration against a C++ baseline.
- **Backed by HPX**: every operation is a real HPX parallel computation (shared-memory
  now, distributed later) — no hidden serial fallbacks.
- **Scales**: memory-bound ops approach the machine's bandwidth ceiling; compute-bound
  ops approach core count.
- **NumPy-compatible (Phase 2)**: once the wrap is proven zero-overhead, add a zero-copy
  NumPy bridge and a drop-in `np`-style API (`a*b`, `a@b`, `dot`, slicing) — as a
  separate phase, not threaded through the core.

Non-goals (Phase 1): NumPy compatibility, GPU, distributed — later, explicitly
sequenced, not bolted on.

---

## 2. Lessons carried from the prototype (what we will NOT repeat)

These are the concrete defects found while benchmarking the existing bindings; the new
core is designed to make each one impossible-by-construction:

1. **Two divergent array types** (contiguous Layer-1 + partitioned Layer-2). → ONE array
   type, `hpx::compute::vector`-backed, used everywhere.
2. **NUMA-naive allocation** (`std::vector::resize` zero-fills on one thread → all pages
   on one socket). → allocation goes through HPX's NUMA-aware `block_allocator`
   (**parallel first-touch** via the HPX partitioner); never a serial zero-fill on the
   hot path. Allocate only for genuinely new arrays — never a between-layers buffer.
3. **No expression fusion** (`a*b` materializes a temporary; `sum(a*b)` is two passes). →
   **lazy/fused expression evaluation**; `dot`/compound ops are single-pass.
4. **Eager per-op result allocation dominates** (new partitioned_vector per `a*b`). →
   result buffers come from a cheap allocation path (and fusion avoids most of them).
5. **Inconsistent thread control** (`hpx.os_threads` cfg ignored; only `--hpx:threads`
   worked). → ONE runtime/policy owner; thread count set one documented way; tested.
6. **Provenance loss** (canonical source not in git, diverged copies). → git repo + CI
   from commit #1; no work outside version control.
7. **Mixed/duplicated kernels** across workers. → kernels defined once; benchmarks and
   tests import them.

---

## 3. Core architecture (the redesign)

- **Array type:** single `Array` backed by `hpx::compute::vector<T, block_allocator<T>>`
  (T: float64 first; float32/int64 later) — a contiguous, NUMA-aware, single-locality
  container. Contiguity makes the eventual NumPy bridge a zero-copy borrow; the
  distributed layer (M4) composes per-locality arrays via collectives, never a gather.
- **Allocation:** HPX's NUMA-aware `block_allocator` performs parallel first-touch on the
  HPX thread pool at construction (run via `run_as_hpx_thread`). Used by every
  constructor; allocate only for genuinely new arrays, never a between-layers buffer.
- **Expressions:** lazy expression templates (or a small expression IR) so `a*x + y`
  builds an unevaluated node and materializes in **one** fused `hpx::transform`/
  `for_loop` pass; reductions (`dot`, `sum`) fuse the transform into the reduce.
- **Execution policy:** one policy model (`seq`/`par`/`par_unseq`), one place that owns
  the HPX runtime + thread count; default `par_unseq`.
- **NumPy interop (Phase 2):** zero-copy borrow of the contiguous compute::vector buffer;
  the distributed layer stays in HPX (explicit gather only if ever truly needed).
- **Binding layer:** `DECIDE #2 — RESOLVED: nanobind` (lighter, zero-copy ndarray,
  Py3.13 free-threading, aligns with HPyX for the eventual merge).

---

## 4. Repository & packaging

- **Standalone repo** depending on a pinned, installed HPX (`find_package(HPX)`).
  `DECIDE #1 — RESOLVED: standalone repo github.com/lums658/hpxpy` (public, MIT).
- **Layout:**
  ```
  hpxpy/                # python package
  src/                  # C++ binding sources (core, ops, reductions)
  tests/                # pytest unit + V&V (correctness vs numpy)
  benchmarks/           # micro + macro benchmarks (reuse this week's harness)
  cpp_baseline/         # hand-written C++ HPX reference kernels (perf ceiling)
  cmake/, pyproject.toml, CMakeLists.txt
  docs/                 # design notes, ADRs
  ```
- **Build:** scikit-build-core + CMake against installed HPX (Boost 1.90 / gcc 15 /
  Py3.13 on Rostam — captured by an `env.sh` like the current `rostam_env.sh`,
  incl. tcmalloc preload).
- **CI:** `DECIDE #3 — RESOLVED: self-hosted Rostam runner` (HPX must be present to
  build/test). Hosted lint (Tier-1) + Rostam build-test gate each PR; bench.yml runs the
  perf/penalty sweep on an exclusive node; docs.yml builds + deploys the site on main.

---

## 5. Process — the iteration loop

Each iteration = one GitHub issue (mirrored as a beads node). **Definition of Done —
four orthogonal axes** (each can be checked by an independent verifier agent):

1. **Correctness & coverage** — pytest vs analytic/reference values (Phase 1: NumPy only
   as a *test oracle*, never in the library); edge cases
   (empty/1-elem/odd/big); numerical V&V (tolerances) + determinism check for parallel
   ops. **Coverage gate:** every public symbol has a test; line/branch coverage ≥ 90%
   (Python via pytest-cov; binding glue via llvm-cov where feasible). CI fails below.
2. **Performance** — microbenchmark: raw throughput, **scalability** (vs threads
   {1,2,4,8,16,32,40}), and **abstraction penalty** (vs the C++ HPX baseline). Meets the
   iteration's perf threshold (§6) and **no regression** vs recorded numbers.
3. **(Phase 2 only) NumPy parity** — the drop-in parity suite passes (same snippet under
   `numpy` and `hpxpy` → same result) + reviewer rubric. **Not a Phase-1 gate**; Phase 1's
   headline gate is the abstraction penalty (axis 2).
4. **Docs & hygiene** — every public symbol has a numpydoc docstring + type hints; the
   docs site builds; doc-coverage ≥ 95%; user-guide/example updated if user-facing; ADR
   for non-trivial choices; lint/format/type-check clean; results (numbers + plots)
   committed; issue/bead closed.

Nothing merges without its gates: **Phase 1 = axes 1, 2, 4** (axis 2's abstraction-penalty
check is the headline); Phase 2 adds axis 3. Perf numbers are committed (CSV) so
regressions show.

---

## 5a. The PR cycle (one iteration, end to end)

Every unit of work flows through the same loop. "Done" = merged with all four DoD axes
green. A unit ≈ one GitHub issue ↔ one bead ↔ one PR.

0. **Ready** — issue exists with acceptance criteria (the 4 axes, incl. perf threshold +
   parity cases). Its beads deps are satisfied (in the "ready set", §11). For fan-out,
   the scheduler hands ready units to agents; an agent (or human) **claims** the bead
   (prevents double-work).
1. **Branch** — from `main`: `feat/<issue#>-<slug>` (or `fix/`, `perf/`, `docs/`).
2. **Implement together** — code + unit tests + microbenchmark + docstrings/docs in the
   *same* PR. "Done means tested *and* documented" — never a follow-up.
3. **Local pre-flight** — run the one-command gate (`make check` / `nox`): build → lint/
   format → type-check → unit tests + coverage → parity suite → docs build + doc-coverage
   → microbench smoke. Must be green before pushing (cheap CI).
4. **Open PR** — template: *what/why*, `Closes #N`, **results pasted in** (perf numbers +
   scaling/penalty plot, coverage delta, docs preview link). Draft PRs allowed for WIP.
5. **CI runs** (§5b) — fast checks on every push; the perf gate on label/nightly.
6. **Review** — (a) an **automated reviewer agent** does an adversarial pass on the 4
   axes ("try to break correctness / find a perf regression / find a non-NumPy idiom /
   find an undocumented symbol"); (b) **human approval** required to merge. Reviewer
   comments → revisions → re-run.
7. **Merge** — squash-merge to `main` (linear history). Branch deleted. Bead → `done`;
   **dependent beads recompute → new ready set** (unblocks the next fan-out wave).
8. **Post-merge CD** (§5b) — `main` pipeline publishes docs, updates the benchmark
   dashboard/baselines, builds wheels; tags trigger a release.

Branch protection on `main`: no direct pushes; PR + green required checks + ≥1 approval.

## 5b. CI/CD pipeline

The HPX dependency shapes this: building needs a prebuilt HPX, and **scaling/abstraction
numbers need real hardware + an exclusive node** — you cannot measure them in a generic
cloud runner. So CI is split into two tiers.

**Tier 1 — fast checks (container image, every push/PR; minutes):**
- Image: pinned HPX + Boost 1.90 + gcc 15 + Py 3.13 (built once, cached).
- Jobs: configure+build (nanobind ext) → ruff/clang-format lint → mypy type-check →
  `pytest` unit tests + **coverage ≥ 90%** → **NumPy parity suite** → docs build +
  **doc-coverage ≥ 95%** → import/ABI smoke (tcmalloc preload). Any red blocks merge.

**Tier 2 — perf gate (must run on Rostam; never on hosted runners):**
Hosted CI VMs (2–4 shared vCPUs) cannot measure scaling/abstraction penalty — the perf
gate has to reach a real exclusive 40-core NUMA node. Because that node **queues and
takes minutes**, perf is **nightly / on-`run-bench`-label, never per-push**. Three ways
to reach Rostam, in increasing automation (pick per cluster policy — needs admin OK):
  1. **Self-hosted Actions runner on Rostam** (`runs-on: [self-hosted, rostam]`) that
     `srun --exclusive`s the suite. Most integrated; but a daemon running PR code on a
     login node is security-sensitive → **maintainer-label-gated only**, never auto on
     untrusted PRs.
  2. **CI→SSH dispatch**: hosted job SSHes in with a deploy key, `sbatch`es, pulls
     results. No daemon, but a cluster key lives in CI secrets.
  3. **Rostam-side cron "pull"** (most cluster-friendly): cron on Rostam pulls branches/
     `main`, runs on an exclusive node, pushes results back (results branch / `gh` API).
     Outbound-only from Rostam — usually easiest to get approved. Naturally nightly.
- Whatever the transport: run `bench_suite.py` 3-way vs C++/NumPy (throughput, scaling
  {1..40}, abstraction penalty) on an exclusive `medusa` node, compare to the committed
  baseline CSV, **fail on regression > tolerance**, post plots to the PR.

**Bootstrap (during initial implementation) — agent/developer is the perf runner.**
Until a runner/cron is approved, the perf gate is triggered manually: on any
perf-relevant PR, run the harness on an exclusive node (as we already do) and paste
numbers+plots into the PR. Early on there is also no hosted HPX container, so **both
tiers bootstrap on Rostam** — fast checks via `nox` on a login node, perf on an exclusive
node — then graduate to a hosted Tier-1 container + automated Tier-2 (option 1 or 3).

**CD (on merge to `main` / tags):**
- Docs → GitHub Pages. Benchmark history → dashboard (committed CSVs + generated plots).
- Wheels: `cibuildwheel`-style, but linked against a pinned HPX (manylinux won't have
  HPX) → ship a "needs installed HPX" sdist + documented build, and/or a conda/container
  artifact. `DECIDE:` distribution model (sdist+find_package vs conda vs container).
- Versioned tag → release notes (from merged PRs) → PyPI/TestPyPI when API stabilizes.

**Concrete files:** `.github/workflows/ci.yml` (Tier 1), `bench.yml` (Tier 2, self-hosted
+ `workflow_dispatch`/label), `docs.yml` (Pages), `release.yml` (tags). `noxfile.py`
defines the local==CI gate so "works locally" == "passes CI".

## 6. Testing & benchmarking taxonomy

- **Correctness (unit):** every op verified against NumPy (`np.allclose`), multiple
  sizes/dtypes/shapes, empty/1-element/odd-size edge cases.
- **Microbenchmarks:** single op, isolated. Each reports median-of-N times (W warmup),
  throughput, and scales over threads {1,2,4,8,16,32,40} and sizes (decades). Used as
  the per-iteration gate.
- **Macrobenchmarks:** the 3-way suite (HPXPy vs C++ HPX vs NumPy) — `dot`,
  `gauss_seidel`, later SpMV/matmat — reused from this week's `bench_suite.py`.
- **Perf acceptance thresholds (example, tune per op):**
  - memory-bound op (dot, axpy): ≥ 60% of the C++ HPX baseline at 40 threads, and
    ≥ 8× speedup 1→40.
  - the C++ baseline itself defines the ceiling (NUMA-correct, fused).
- **Niceness / NumPy parity:** a **drop-in parity suite** — curated snippets run twice,
  once `import numpy as np`, once `import hpxpy as np`, asserting identical results and
  that the idiom is even expressible (operators, broadcasting, dtype rules, error types).
  Plus a reviewer-agent rubric (API signature match, docstrings, type hints, `__array__`
  protocol). Produces a parity score tracked per release.
- **Regression tracking:** committed CSV of canonical numbers; CI flags drops > tolerance.

---

## 7. Tooling — GitHub issues + beads

- **GitHub issues/milestones** = human-facing source of truth on the repo (per-iteration
  issue, milestone per phase, results attached on close). Needs `gh` installed+authed
  (not yet present) or web creation.
- **beads** = agent-facing working memory: a dependency graph of the same work so I can
  resume across sessions and see what unblocks what. To be installed + verified here
  first; kept in sync with GitHub issues (issue ID ↔ bead). If beads proves flaky, fall
  back to GitHub-only + my file memory.
- **beads doubles as the parallel scheduler** (see §11): its ready/blocked dependency
  graph is what tells the orchestrator which work units can be fanned out concurrently.

---

## 8. Phased roadmap (milestones, each gated by §5/§6)

### Phase 1 — Wrap HPX; validate zero abstraction penalty (NO NumPy in the data path)
- **M0 — Substrate.** (done) Repo + build (installed HPX) + CI + `env.sh` + harness + C++
  baseline. Exit: package builds & imports; CI green.
- **M1 — Array core.** (done) `Array` = `hpx::compute::vector<double, block_allocator>`
  wrapper + introspection (`size`/`ndim`) + HPX-native construction
  (`zeros`/`full`/`arange`), NUMA-aware first-touch. **No NumPy.**
- **M2 — Reductions.** (done) `sum`/`min`/`max`/`dot` as wrapped HPX algorithms; first
  zero-penalty validation vs C++ (≈1.0), correctness via analytic values.
- **M3 — Transforms / element-wise.** (done) element-wise `add/sub/mul/div` + operators,
  scalar broadcast, `sort`/`copy`/`cumsum`/`is_sorted` — all penalty ≈1.0.
- **Slice/view model.** (done) `a[i]`, `a[i:j]` contiguous memory-sharing views,
  `a[i]=x` (offset in Array; numpy view semantics; `step!=1` deferred → bead `x7i`).
- **M5 — Sparse (SpMV/SpMM).** (done) CSR `CsrMatrix` + `laplacian_1d`; `DenseMatrix`;
  `spmv`/`spmm` + `A@x` / `A@B`, kernel-timed penalty ≈1.0.
- **M4a — Distributed runtime.** (done, PR #33) multi-locality via the TCP parcelport (no HPX
  rebuild); `num_localities`/`locality_id`/`is_console`/`is_worker`/`distributed_sum`
  (all_reduce); worker-aware startup; 2-locality validated in CI.

**Phase-1 + NumPy-bridge status:** the single-locality op set is wrapped at measured ~0
abstraction penalty (M1–M3, slices/strided, M5); the zero-copy NumPy bridge + drop-in parity
suite landed; the distributed runtime is up. ~302 tests, 100% coverage.

### Post-Phase-1 roadmap — usability → distributed → GPU
The next arc targets a NumPy-faithful experience that scales across all resources with no
performance tax (decentralized HPX; eager/interactive):
- **Phase A — single-node usability** (current): **N-D arrays** (epic `hpxpy-3ur`, staged) →
  **dtypes** (float32/int64; resolves DECIDE #4) → eager/deferred behavior + pip-installable
  wheels. The adoption foundation.
- **Phase B — distributed data type:** global-view `Array` over `hpx::partitioned_vector` +
  segmented reductions (transparent partitioning); the M4a runtime is the substrate.
- **Phase C — GPU + heterogeneity:** device `compute::vector` (Kokkos/CUDA), CPU↔GPU↔node
  async overlap, DLPack / Array-API interop. Sequencing: **B before C**.
- **Validation/demos:** example + tutorial notebooks spanning the arc, once the wrapper is solid.

---

## 9. Decisions

Resolved:
- **Repo (was DECIDE #1) — RESOLVED: standalone `hpxpy` repo**, pinned to a tested HPX
  (`find_package(HPX)`). Not in the HPX tree.
- **Binding framework (was DECIDE #2) — RESOLVED: nanobind.** Best practice for a thin,
  low-overhead layer with first-class zero-copy (`nb::ndarray` + DLPack → NumPy/torch/
  jax/cupy); Py3.13 free-threading; aligns with HPyX for the merge. **Already proven on
  our toolchain** (HPyX, a nanobind ext, built against our HPX today). Zero-copy applies
  to contiguous/single-locality buffers; distributed `to_numpy` gathers (data-model fact).
- **Tracking — RESOLVED: GitHub issues (human source of truth) + beads (agent scheduler
  / dependency graph).** Both; not either/or.

Resolved (later):
- **DECIDE #3 — RESOLVED: self-hosted Rostam runner** (in production: `ci.yml` build-test +
  build-test-distributed, `bench.yml`, `docs.yml`).
- **DECIDE #4 — RESOLVED:** float64 first (Phase 1); **float32/int64 in Phase A after N-D**
  (bead `hpxpy-jqk`).
- **Setup — DONE:** `gh` + `beads` installed and in use.

---

## 11. Parallelization with multiple agents

Parallelism is staged by dependency, not applied uniformly:

- **Serial spine (one builder): M0 + M1 + the M2 reduction interface.** This defines the
  `Array` contract everything else builds on; parallelizing it = rework + merge hell.
- **Support workstreams — parallel from day one** (depend on the *interface*, not impl):
  test harness, microbenchmark harness, **C++ baseline kernels**, niceness/parity suite,
  CI, docs. Fan these out immediately.
- **Per-operation fan-out — after the core API freezes (post-M2).** One agent per op
  (each reduction, elementwise op, stencil, SpMV), each delivering impl + the 4-axis DoD.
  Highly parallel; near-independent.
- **Per-unit verification fan-out.** The 4 DoD axes (correctness / perf / niceness /
  hygiene) are orthogonal → an implementing agent plus independent verifier agents, or
  an adversarial "try to break it" pass before close.

**Scheduling mechanism (why the tracker matters here):** maintain a **dependency DAG** of
work units; the parallel scheduler launches the *ready set* (units whose deps are done).
beads is purpose-built for this (agent-facing ready/blocked graph) and is the intended
scheduler; GitHub issues stay the human-facing 1:1 mirror. If beads is unavailable, the
same DAG lives in a file and drives the fan-out — the mechanism matters more than the tool.

**Orchestration:** once the core is frozen, the per-op + verification fan-out is a good
fit for a multi-agent workflow (pipeline: implement → verify-correctness ∥ verify-perf ∥
verify-niceness → synthesize). We launch that explicitly per phase, not before.

## 12. HPyX integration

HPyX (task-parallel: `HPXRuntime`, `HPXExecutor`, futures, `dot1d`; nanobind) and HPXPy
(data-parallel arrays) are complementary over the *same* HPX runtime. To make merge/
interop a first-class outcome rather than an afterthought:

- **Use nanobind** (DECIDE #2 → resolved toward nanobind) so both share a binding
  framework.
- **Shared runtime module from M0**: one component owns `hpx::init/finalize` + thread
  count; both the array side and HPyX's executor attach to it (HPX allows one runtime per
  process). Design HPXPy's runtime layer to be the thing HPyX could also import.
- **Interop surface (milestone M6a):** submit an array op as an HPyX task / future;
  build an array from data produced by tasks; expose both `hpx.array` (data-parallel) and
  `hpx.executor`/`hpx.future` (task-parallel) in one namespace.
- **End state:** a single "HPX for Python" package. Coordinate with the HPyX authors (UW)
  — cross-group; both BSL-1.0.

## 13. Risks

- CI needs HPX → container/self-hosted runner (M0 must solve this, or iterations can't
  be gated automatically).
- Lazy/fused expression engine is the hardest core piece → prototype in M3 with a spike;
  fall back to eager-but-cheap-alloc if the IR slips.
- nanobind learning curve if chosen → M0 spike de-risks.
- Scope creep → milestones are hard gates; no op without tests+benchmarks.
