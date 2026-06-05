#!/bin/bash
# hpxpy build/run environment on Rostam. Source before building or testing:
#   source env.sh
#
# Captures the hard-won toolchain facts (see docs/PLAN.md §2):
#   * HPX at ~/usr/local/hpx was built with gcc 15.1.0 + Boost 1.90.0; the default
#     module is Boost 1.91.0, which leaks 1.91 headers via CPATH and breaks the
#     build -> swap to 1.90.0.
#   * The extension is a CPython 3.13 build with gcc 15 -> needs gcc 15 libstdc++
#     (GLIBCXX_3.4.31) on LD_LIBRARY_PATH.
#   * HPX uses tcmalloc; without LD_PRELOAD of tcmalloc, array ops abort with
#     "double free or corruption". Mandatory at runtime (unset it for cmake/build).
set -u
export HPX_ROOT="${HOME}/usr/local/hpx"
export HPX_DIR="${HPX_ROOT}/lib64/cmake/HPX"
export VENV_PY="${HOME}/.venv/bin/python"

source /etc/profile.d/modules.sh 2>/dev/null || true
module purge 2>/dev/null || true
module load gcc/15.1.0 cmake/3.29.2 boost/1.90.0-release python/3.13.2 2>/dev/null
module swap boost/1.91.0-release boost/1.90.0-release 2>/dev/null || true

export CMAKE_PREFIX_PATH="${HPX_ROOT}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"
export LD_LIBRARY_PATH="/opt/apps/gcc/15.1.0/lib64:${HPX_ROOT}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PY="${VENV_PY}"

# Runtime-only: required for import/execution, NOT for cmake/ninja builds.
export HPXPY_TCMALLOC="/usr/lib64/libtcmalloc_minimal.so"

echo "[hpxpy env] PY=$PY ($(${PY} --version 2>&1)) | HPX_ROOT=$HPX_ROOT | boost=${BOOST_ROOT:-?}"
echo "[hpxpy env] build:  LD_PRELOAD unset (don't preload tcmalloc into cmake/ninja)"
echo "[hpxpy env] run/test: export LD_PRELOAD=\$HPXPY_TCMALLOC"
set +u
