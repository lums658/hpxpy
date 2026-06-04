# HPXPy (rewrite) — Plan & Process

**Status:** DRAFT for review. `DECIDE:` = open choice. Working planning home:
`~/LSU/hpxpy-ng/` (the actual repo name/location is DECIDE #1 below).
Created 2026-06-04.

A from-scratch reimplementation of HPXPy: a NumPy-compatible Python array library
backed by the HPX C++ runtime, built incrementally with correctness + benchmark gates
at every iteration. This document is the contract for *how* we build it.

---

## 1. Vision & goals

- **NumPy-replaceable** at the Python level: `arange`, `array`, `a*b`, `a@b`, `dot`,
  reductions, slicing — familiar syntax.
- **Backed by HPX**: every array operation is a real HPX parallel computation
  (shared-memory now, distributed later) — no hidden serial fallbacks.
- **Scales**: memory-bound ops approach the machine's bandwidth ceiling; compute-bound
  ops approach core count. Measured against hand-written C++ HPX each iteration.
- **Honest about overhead**: the abstraction penalty vs native C++ is measured, bounded,
  and tracked over time — not discovered at the end.

Non-goals (initially): GPU, full NumPy API surface, distributed — these are later
milestones, explicitly sequenced, not bolted on.

---

## 2. Lessons carried from the prototype (what we will NOT repeat)

These are the concrete defects found while benchmarking the existing bindings; the new
core is designed to make each one impossible-by-construction:

1. **Two divergent array types** (contiguous Layer-1 + partitioned Layer-2). → ONE array
   type, `partitioned_vector`-backed, used everywhere.
2. **NUMA-naive allocation** (`std::vector::resize` zero-fills on one thread → all pages
   on one socket). → allocation is uninitialized + **parallel first-touch**; never a
   serial zero-fill on the hot path.
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

- **Array type:** single `Array` backed by `hpx::partitioned_vector<T>` (T: float64
  first; float32/int64 later). One partition per locality initially; layout abstracted.
- **Allocation:** uninitialized storage + parallel first-touch via the initializing
  HPX algorithm. A small allocation/first-touch utility used by every constructor.
- **Expressions:** lazy expression templates (or a small expression IR) so `a*x + y`
  builds an unevaluated node and materializes in **one** fused `hpx::transform`/
  `for_loop` pass; reductions (`dot`, `sum`) fuse the transform into the reduce.
- **Execution policy:** one policy model (`seq`/`par`/`par_unseq`), one place that owns
  the HPX runtime + thread count; default `par_unseq`.
- **NumPy interop:** zero-copy where contiguous & single-locality; explicit gather for
  `to_numpy` when partitioned/distributed.
- **Binding layer:** `DECIDE #2:` pybind11 (matches old code, what we know) vs nanobind
  (what HPyX uses; lighter, Py3.13 free-threading; eases eventual merge). Leaning
  **nanobind** to align with HPyX for the merge — but pybind11 is lower-risk/known.

---

## 4. Repository & packaging

- **Standalone repo** depending on a pinned, installed HPX (`find_package(HPX)`).
  `DECIDE #1:` repo name/location — proposed `hpxpy` (new GitHub repo under the group)
  or a fresh branch; planning lives in `~/LSU/hpxpy-ng/` until decided.
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
- **CI:** `DECIDE #3:` container image with pinned HPX, or self-hosted Rostam runner.
  CI runs build → unit tests → microbenchmarks (perf-regression check) on each PR.

---

## 5. Process — the iteration loop

Each iteration = one GitHub issue (mirrored as a beads node). **Definition of Done —
four orthogonal axes** (each can be checked by an independent verifier agent):

1. **Correctness & coverage** — pytest vs NumPy reference; edge cases
   (empty/1-elem/odd/big); numerical V&V (tolerances) + determinism check for parallel
   ops. **Coverage gate:** every public symbol has a test; line/branch coverage ≥ 90%
   (Python via pytest-cov; binding glue via llvm-cov where feasible). CI fails below.
2. **Performance** — microbenchmark: raw throughput, **scalability** (vs threads
   {1,2,4,8,16,32,40}), and **abstraction penalty** (vs the C++ HPX baseline). Meets the
   iteration's perf threshold (§6) and **no regression** vs recorded numbers.
3. **Niceness / NumPy parity** — the drop-in parity suite passes (same snippet under
   `numpy` and `hpxpy` → same result, idiom works) + reviewer rubric (signatures match
   `np`, broadcasting/dtype/error semantics).
4. **Docs & hygiene** — every public symbol has a numpydoc docstring + type hints; the
   docs site builds; doc-coverage ≥ 95%; user-guide/example updated if user-facing; ADR
   for non-trivial choices; lint/format/type-check clean; results (numbers + plots)
   committed; issue/bead closed.

Nothing merges without all four. Perf numbers are committed (CSV) so regressions show.

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

- **M0 — Substrate.** Repo + build (installed HPX) + CI skeleton + env.sh + test/bench
  scaffolding + the C++ baseline harness. Exit: empty package builds & imports; CI green.
- **M1 — Array core.** `Array` on `partitioned_vector`, NUMA-aware alloc, `arange/zeros/
  from_numpy/to_numpy`. Exit: round-trips vs NumPy; alloc bandwidth ≈ STREAM.
- **M2 — Reductions.** `sum`, `dot` (fused), `min/max/mean`. Exit: dot ≥60% C++, ≥8×
  scaling; correctness.
- **M3 — Fused elementwise.** lazy `a*b`, `a*x+y`, ufuncs; `a@b` (dot). Exit: `a*x+y`
  single-pass, scales; no per-op temporary.
- **M4 — Stencil/SpMV.** `gauss_seidel`, sparse matvec (NWGraph-relevant). Exit: scaling
  vs C++ baseline.
- **M5 — Distributed.** multi-locality arrays + collectives; strong/weak scaling across
  nodes. Exit: G4-style multi-node results.
- **M6 — Beyond.** GPU; HPyX (task-parallel) interop/merge; broader NumPy surface.

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

Still open:
- **DECIDE #3** — CI host: container w/ pinned HPX vs self-hosted Rostam runner.
- **DECIDE #4** — dtype scope for M1–M3 (float64 only first? add float32/int64 when?).
- **Setup** — `gh` and `beads` not installed here yet; git consolidation pending.

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
