#!/bin/bash
# BR3 Scaling Sweep — strong + weak scaling for sum, dot, add, matmul
# Run inside a medusa srun allocation (AVX-512 node required).
# Output: human-readable lines from diag, saved to RAW_OUT.
set -euo pipefail

BENCH_DIR="/work/alumsdaine/LSU/hpxpy-bench"
DIAG="${BENCH_DIR}/cpp_baseline/build/diag"
RESULTS="${BENCH_DIR}/benchmarks/results"
TIMESTAMP=$(date +%s)
RAW_OUT="${RESULTS}/BR3_raw_${TIMESTAMP}.txt"

mkdir -p "${RESULTS}"

# Verify diag exists
if [[ ! -x "$DIAG" ]]; then
    echo "ERROR: diag binary not found at $DIAG" >&2
    exit 1
fi

cd "${BENCH_DIR}/cpp_baseline"
source "${BENCH_DIR}/env.sh" >/dev/null 2>&1
export LD_PRELOAD="${HPXPY_TCMALLOC:-}"

echo "[BR3] Starting scaling sweep on $(hostname) at $(date)" | tee "$RAW_OUT"
echo "[BR3] SLURM_JOB_ID=${SLURM_JOB_ID:-local}" | tee -a "$RAW_OUT"
echo "[BR3] diag binary: $DIAG" | tee -a "$RAW_OUT"
echo "" | tee -a "$RAW_OUT"

# Benchmark parameters
THREADS_LIST="1 2 4 8 16 20 40"
BUDGET=3.0
MIN_REPS=5
MAX_REPS=50

# -----------------------------------------------------------------------
# STRONG SCALING
# sum, n=1e8 (memory-bound, single array)
# dot, n=1e8 (memory-bound, two arrays)
# add, n=1e8 (memory-bound, two arrays, elementwise)
# matmul, n=2048 (2048x2048 compute-bound)
# -----------------------------------------------------------------------
echo "[BR3] === STRONG SCALING ===" | tee -a "$RAW_OUT"

for T in $THREADS_LIST; do
    echo "" | tee -a "$RAW_OUT"
    echo "[BR3] --- threads=$T ---" | tee -a "$RAW_OUT"

    # sum, n=100_000_000
    echo "[BR3]   sum n=1e8 threads=$T" | tee -a "$RAW_OUT"
    "$DIAG" --op sum --sizes 100000000 --threads "$T" \
        --budget "$BUDGET" --min-reps "$MIN_REPS" --max-reps "$MAX_REPS" \
        2>&1 | tee -a "$RAW_OUT"

    # dot, n=100_000_000
    echo "[BR3]   dot n=1e8 threads=$T" | tee -a "$RAW_OUT"
    "$DIAG" --op dot --sizes 100000000 --threads "$T" \
        --budget "$BUDGET" --min-reps "$MIN_REPS" --max-reps "$MAX_REPS" \
        2>&1 | tee -a "$RAW_OUT"

    # add, n=100_000_000
    echo "[BR3]   add n=1e8 threads=$T" | tee -a "$RAW_OUT"
    "$DIAG" --op add --sizes 100000000 --threads "$T" \
        --budget "$BUDGET" --min-reps "$MIN_REPS" --max-reps "$MAX_REPS" \
        2>&1 | tee -a "$RAW_OUT"

    # matmul 2048x2048
    echo "[BR3]   matmul n=2048 threads=$T" | tee -a "$RAW_OUT"
    "$DIAG" --op matmul --sizes 2048 --threads "$T" \
        --budget "$BUDGET" --min-reps "$MIN_REPS" --max-reps "$MAX_REPS" \
        2>&1 | tee -a "$RAW_OUT"

    echo "[BR3]   done threads=$T" | tee -a "$RAW_OUT"
done

# -----------------------------------------------------------------------
# WEAK SCALING: sum with n = 5_000_000 * threads
# Perfect weak scaling => constant time regardless of thread count
# -----------------------------------------------------------------------
echo "" | tee -a "$RAW_OUT"
echo "[BR3] === WEAK SCALING (sum, n = 5e6 * threads) ===" | tee -a "$RAW_OUT"

for T in $THREADS_LIST; do
    N=$(( 5000000 * T ))
    echo "" | tee -a "$RAW_OUT"
    echo "[BR3]   weak_sum n=$N threads=$T" | tee -a "$RAW_OUT"
    "$DIAG" --op sum --sizes "$N" --threads "$T" \
        --budget "$BUDGET" --min-reps "$MIN_REPS" --max-reps "$MAX_REPS" \
        2>&1 | tee -a "$RAW_OUT"
done

echo "" | tee -a "$RAW_OUT"
echo "[BR3] Sweep complete at $(date)" | tee -a "$RAW_OUT"
echo "[BR3] Raw output saved to: $RAW_OUT" | tee -a "$RAW_OUT"
echo "RAW_OUT_PATH=$RAW_OUT"
