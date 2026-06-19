# Project Instructions for AI Agents

This file provides instructions and context for AI coding agents working on this project.

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:6cd5cc61 -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

**Architecture in one line:** issues live in a local Dolt DB; sync uses `refs/dolt/data` on your git remote; `.beads/issues.jsonl` is a passive export. See https://github.com/gastownhall/beads/blob/main/docs/SYNC_CONCEPTS.md for details and anti-patterns.

## Agent Context Profiles

The managed Beads block is task-tracking guidance, not permission to override repository, user, or orchestrator instructions.

- **Conservative (default)**: Use `bd` for task tracking. Do not run git commits, git pushes, or Dolt remote sync unless explicitly asked. At handoff, report changed files, validation, and suggested next commands.
- **Minimal**: Keep tool instruction files as pointers to `bd prime`; use the same conservative git policy unless active instructions say otherwise.
- **Team-maintainer**: Only when the repository explicitly opts in, agents may close beads, run quality gates, commit, and push as part of session close. A current "do not commit" or "do not push" instruction still wins.

## Session Completion

This protocol applies when ending a Beads implementation workflow. It is subordinate to explicit user, repository, and orchestrator instructions.

1. **File issues for remaining work** - Create beads for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **Handle git/sync by active profile**:
   ```bash
   # Conservative/minimal/default: report status and proposed commands; wait for approval.
   git status

   # Team-maintainer opt-in only, unless current instructions forbid it:
   git pull --rebase
   git push
   git status
   ```
5. **Hand off** - Summarize changes, validation, issue status, and any blocked sync/commit/push step

**Critical rules:**
- Explicit user or orchestrator instructions override this Beads block.
- Do not commit or push without clear authority from the active profile or the current user request.
- If a required sync or push is blocked, stop and report the exact command and error.
<!-- END BEADS INTEGRATION -->


## Build & Test

On Rostam (the supported environment):

```bash
source env.sh            # toolchain (gcc 15, Boost 1.90, Python 3.13) + HPX paths
bash scripts/check.sh    # editable build + ruff + pytest (100% coverage) — the local==CI gate
```

- HPX is found via `find_package(HPX)` (`HPX_DIR` / `CMAKE_PREFIX_PATH` come from `env.sh`).
- The pinned HPX install is built with **AVX-512** — run on an AVX-512 node (Rostam
  **medusa**). The AMD **buran** login nodes lack AVX-512 and will **SIGILL** at HPX load.
- Tests and benchmarks need the tcmalloc preload at runtime: `LD_PRELOAD=$HPXPY_TCMALLOC`.

## Architecture Overview

A thin nanobind wrapper over HPX:

- `src/array.hpp` — one runtime-dtype N-D `Array` (`shape_` / `strides_` over a NUMA-aware
  `hpx::compute::vector` with `block_allocator` first-touch). Each operation is a single HPX
  parallel algorithm; `dispatch_dtype` instantiates kernels per element type (f64/f32/i64).
- `src/_core.cpp` — the nanobind binding (`_core`); `hpxpy/` — the Python package.
- `cpp_baseline/` — the hand-written C++-HPX baseline + the in-binary L0/L1 penalty ladder
  (`diag`); `benchmarks/` — the runner, results, and `BENCHMARK_PLAN.md`.
- `docs/PLAN.md` — the design contract (vision, phasing, the zero-penalty methodology).

## Conventions & Patterns

- **Zero abstraction penalty is the contract.** Every op mirrors a raw HPX algorithm; the
  `cpp_baseline/diag` L0/L1 ladder must stay ≈1.0. Keep the contiguous fast paths intact.
- **NumPy is the oracle.** Tests compare against NumPy; `scripts/check.sh` must be green
  (ruff + pytest + 100% coverage) before opening a PR.
- **`NOMINSIZE`** in `CMakeLists.txt` keeps `-O3` on the extension (don't let nanobind
  force `-Os`, which silently re-introduces the penalty).
- Commits have **no AI-attribution trailer**.
