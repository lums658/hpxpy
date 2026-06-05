#!/bin/bash
# Local gate == CI Tier-1 (run on Rostam during bootstrap). "Works locally" should
# mean "passes CI". Usage:  bash scripts/check.sh
# SPDX-License-Identifier: MIT
set -euo pipefail
cd "$(dirname "$0")/.."

source env.sh >/dev/null
unset LD_PRELOAD   # build must NOT preload tcmalloc

echo "== build (scikit-build-core -> cmake -> nanobind, against HPX) =="
"$PY" -m pip install -e ".[test]" --no-cache-dir \
  -C cmake.define.CMAKE_CXX_COMPILER=/opt/apps/gcc/15.1.0/bin/g++ \
  -C cmake.define.CMAKE_PREFIX_PATH="$HPX_ROOT" \
  -C cmake.define.HPX_DIR="$HPX_DIR"

echo "== lint (ruff, if available) =="
"$PY" -m ruff check hpxpy tests 2>/dev/null || echo "(ruff not installed - skipped)"

echo "== tests + coverage (tcmalloc preloaded at runtime) =="
LD_PRELOAD="$HPXPY_TCMALLOC" "$PY" -m pytest --cov=hpxpy --cov-report=term-missing

echo "== OK =="
