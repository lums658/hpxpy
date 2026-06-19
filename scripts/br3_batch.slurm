#!/bin/bash
#SBATCH --job-name=br3-scaling
#SBATCH --partition=medusa
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --cpus-per-task=40
#SBATCH --time=0:35:00
#SBATCH --output=/work/alumsdaine/LSU/hpxpy-bench/benchmarks/results/BR3_slurm_%j.out
#SBATCH --error=/work/alumsdaine/LSU/hpxpy-bench/benchmarks/results/BR3_slurm_%j.err

set -euo pipefail

BENCH_DIR="/work/alumsdaine/LSU/hpxpy-bench"
DIAG="${BENCH_DIR}/cpp_baseline/build/diag"
RESULTS="${BENCH_DIR}/benchmarks/results"

mkdir -p "${RESULTS}"

cd "${BENCH_DIR}/cpp_baseline"
source "${BENCH_DIR}/env.sh" >/dev/null 2>&1
export LD_PRELOAD="${HPXPY_TCMALLOC:-}"

echo "[BR3] Starting scaling sweep on $(hostname) at $(date)"
echo "[BR3] SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "[BR3] diag binary: $DIAG"
echo ""

# Thread counts: 1 socket = 20 cores, 2 sockets = 40 cores (Xeon Gold 6148, 2×20)
THREADS_LIST="1 2 4 8 16 20 40"

# Standard adaptive budget for memory-bound ops
BUDGET=3.0
MIN_REPS=5
MAX_REPS=50

# Smaller reps for compute-bound matmul (each rep can be slow at 1 thread)
MATMUL_MIN_REPS=3
MATMUL_MAX_REPS=20

# -----------------------------------------------------------------------
# STRONG SCALING
# sum  n=1e8 (memory-bound, 1 read pass)
# dot  n=1e8 (memory-bound, 2 read passes)
# add  n=1e8 (memory-bound, 2 read + 1 write)
# matmul 1024x1024 f64 (compute-bound, naive triple-loop via for_loop)
# -----------------------------------------------------------------------
echo "[BR3] === STRONG SCALING ==="

for T in $THREADS_LIST; do
    echo ""
    echo "[BR3] --- threads=$T ---"

    echo "[BR3]   sum n=1e8 threads=$T"
    "$DIAG" --op sum --sizes 100000000 --threads "$T" \
        --budget "$BUDGET" --min-reps "$MIN_REPS" --max-reps "$MAX_REPS" 2>&1

    echo "[BR3]   dot n=1e8 threads=$T"
    "$DIAG" --op dot --sizes 100000000 --threads "$T" \
        --budget "$BUDGET" --min-reps "$MIN_REPS" --max-reps "$MAX_REPS" 2>&1

    echo "[BR3]   add n=1e8 threads=$T"
    "$DIAG" --op add --sizes 100000000 --threads "$T" \
        --budget "$BUDGET" --min-reps "$MIN_REPS" --max-reps "$MAX_REPS" 2>&1

    # matmul: 1024x1024 (2.147 GFLOP per call; budget=3s means 3+ reps at all thread counts)
    echo "[BR3]   matmul 1024x1024 threads=$T"
    "$DIAG" --op matmul --sizes 1024 --threads "$T" \
        --budget "$BUDGET" --min-reps "$MATMUL_MIN_REPS" --max-reps "$MATMUL_MAX_REPS" 2>&1

    echo "[BR3]   done threads=$T"
done

# -----------------------------------------------------------------------
# WEAK SCALING: sum with n = 5_000_000 * threads
# Perfect weak scaling => constant time regardless of thread count
# -----------------------------------------------------------------------
echo ""
echo "[BR3] === WEAK SCALING (sum, n = 5e6 * threads) ==="

for T in $THREADS_LIST; do
    N=$(( 5000000 * T ))
    echo ""
    echo "[BR3]   weak_sum n=$N threads=$T"
    "$DIAG" --op sum --sizes "$N" --threads "$T" \
        --budget "$BUDGET" --min-reps "$MIN_REPS" --max-reps "$MAX_REPS" 2>&1
done

echo ""
echo "[BR3] Sweep complete at $(date)"
echo "[BR3] DONE job=${SLURM_JOB_ID}"
echo "RAW_PATH=${RESULTS}/BR3_slurm_${SLURM_JOB_ID}.out"
