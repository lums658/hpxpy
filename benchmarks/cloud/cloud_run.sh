#!/usr/bin/env bash
# benchmarks/cloud/cloud_run.sh
# Launcher for the medusa-as-cloud distributed HPX benchmark.
#
# Usage:
#   ./cloud_run.sh --nodes N --instance {full|small|numa-naive} \
#                  --fabric {eth1g|ipoib} --op OP --n-per-node N --threads T \
#                  [--vec-k K] [--reps R] \
#                  [--netem-latency Xms] [--netem-rate Rgbit]
#
# Instance shapes (emulated cloud instance sizes):
#   full       — all 40 cores, both NUMA nodes (large HPC instance)
#   small      — 8 cores on socket0 (numactl -C 0,2,4,6,8,10,12,14 -m0)
#                analogue: 8-vCPU cloud instance with local NUMA memory
#   numa-naive — 20 cores on NUMA node 0 but memory on NUMA node 1 (numactl -N0 -m1)
#                analogue: hypervisor places memory off-socket (non-NUMA-aware VM)
#
# Fabric selection (THE cloud-network knob — same kernel, different wire):
#   ipoib  — default node hostnames -> IPoIB ibp94s0 (10.42.6.x, ~40 Gb) ≈ fast cloud
#            Uses HPX's SLURM auto-detection (no endpoint flags needed).
#   eth1g  — *-eth hostnames        -> Ethernet eno16 (10.42.5.x, 1 GbE)  ≈ cheap cloud
#            Needs an explicit -eth node list AND an explicit parcelport bind
#            address, because (a) the default hostname resolves to IPoIB and
#            (b) --hpx:ifsuffix mis-appends to the FQDN in this DNS setup.
#
# Network shaping (netem) — OPTIONAL, best-effort:
#   --netem-latency Xms  — add X ms latency      (requires root / CAP_NET_ADMIN)
#   --netem-rate Rgbit   — cap bandwidth to R    (requires root / CAP_NET_ADMIN)
#   On a shared cluster you usually lack CAP_NET_ADMIN; if tc fails we log a
#   WARNING and run UNSHAPED rather than aborting.
#
# Results appended to: benchmarks/cloud/results/<fabric>_<instance>_<op>.csv
#
# SPDX-License-Identifier: MIT

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="${SCRIPT_DIR}/build/cloud_bench"
RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"

# ---------------------------------------------------------------------------
# Defaults
NODES=2
INSTANCE="full"
FABRIC="ipoib"
OP="both"
N_PER_NODE=1000000
THREADS=0          # 0 = use the instance's default below
VEC_K=1024
REPS=25            # FIXED rep count — collectives must be lockstep across
                   # localities (see the long note in cloud_bench.cpp). Passed
                   # as --min-reps==--max-reps so every locality issues the same
                   # number of all_reduce calls and shutdown never deadlocks.
NETEM_LATENCY=""
NETEM_RATE=""

# ---------------------------------------------------------------------------
# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --nodes)          NODES="$2";          shift 2 ;;
        --instance)       INSTANCE="$2";       shift 2 ;;
        --fabric)         FABRIC="$2";         shift 2 ;;
        --op)             OP="$2";             shift 2 ;;
        --n-per-node)     N_PER_NODE="$2";     shift 2 ;;
        --threads)        THREADS="$2";        shift 2 ;;
        --vec-k)          VEC_K="$2";          shift 2 ;;
        --reps)           REPS="$2";           shift 2 ;;
        --netem-latency)  NETEM_LATENCY="$2";  shift 2 ;;
        --netem-rate)     NETEM_RATE="$2";     shift 2 ;;
        *) echo "ERROR: unknown arg: $1" >&2; exit 2 ;;
    esac
done

# ---------------------------------------------------------------------------
# Validate binary
if [[ ! -x "${BENCH}" ]]; then
    echo "ERROR: cloud_bench not found at ${BENCH}" >&2
    echo "       Build it: cmake --build ${SCRIPT_DIR}/build" >&2
    exit 2
fi

# ---------------------------------------------------------------------------
# Fabric -> interface name (for netem) ; the launch mechanism is chosen per-task.
case "${FABRIC}" in
    eth1g)  IFACE="eno16"   ;;
    ipoib)  IFACE="ibp94s0" ;;
    *)  echo "ERROR: unknown fabric '${FABRIC}' (use eth1g or ipoib)" >&2; exit 2 ;;
esac

# ---------------------------------------------------------------------------
# Instance -> numactl prefix + default thread count
case "${INSTANCE}" in
    full)
        NUMACTL=""
        [[ "${THREADS}" == "0" ]] && THREADS=40
        ;;
    small)
        NUMACTL="numactl -C 0,2,4,6,8,10,12,14 -m 0"
        [[ "${THREADS}" == "0" ]] && THREADS=8
        ;;
    numa-naive)
        NUMACTL="numactl -N 0 -m 1"
        [[ "${THREADS}" == "0" ]] && THREADS=20
        ;;
    *)  echo "ERROR: unknown instance '${INSTANCE}' (use full|small|numa-naive)" >&2; exit 2 ;;
esac

# ---------------------------------------------------------------------------
# netem options string (empty = no shaping)
NETEM_OPTS=""
[[ -n "${NETEM_LATENCY}" ]] && NETEM_OPTS="delay ${NETEM_LATENCY}"
[[ -n "${NETEM_RATE}"    ]] && NETEM_OPTS="${NETEM_OPTS} rate ${NETEM_RATE}gbit"

# tcmalloc is MANDATORY at runtime (HPX uses it; without it array ops abort).
TCMALLOC="${HPXPY_TCMALLOC:-/usr/lib64/libtcmalloc_minimal.so}"

# ---------------------------------------------------------------------------
# Result CSV setup
CSV_FILE="${RESULTS_DIR}/${FABRIC}_${INSTANCE}_${OP}.csv"
CSV_HEADER="fabric,instance,nodes,n_per_node,threads,vec_k,op,localities,median_s,gelem_s,gbytes_s,timestamp"
[[ -f "${CSV_FILE}" ]] || echo "${CSV_HEADER}" > "${CSV_FILE}"
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)

echo "======================================================================"
echo "cloud_run.sh: nodes=${NODES} instance=${INSTANCE} fabric=${FABRIC} (${IFACE})"
echo "             op=${OP} n_per_node=${N_PER_NODE} threads=${THREADS} reps=${REPS} vec_k=${VEC_K}"
[[ -n "${NETEM_OPTS}" ]] && echo "             netem: ${NETEM_OPTS} (best-effort; needs CAP_NET_ADMIN)"
echo "             results -> ${CSV_FILE}"
echo "======================================================================"

# ---------------------------------------------------------------------------
# Per-task command. Config is passed through the environment (srun --export=ALL
# is the default), so there is no placeholder substitution to get wrong.
#
# Fabric mechanism (validated on medusa):
#   ipoib : nothing — HPX auto-detects the SLURM allocation; default node
#           hostnames already resolve to the IPoIB subnet (10.42.6.x).
#   eth1g : hand HPX an explicit "<node>-eth" node list (--hpx:nodes ...
#           --hpx:endnodes --hpx:node=$rank) AND bind the TCP parcelport to this
#           node's eth IP (--hpx:ini=hpx.parcel.address=<eth_ip>). Both are
#           required; either alone hangs or fails to resolve.
export FABRIC IFACE BENCH OP N_PER_NODE VEC_K REPS THREADS NUMACTL NETEM_OPTS TCMALLOC

read -r -d '' PER_TASK <<'TASKEOF' || true
set -uo pipefail
export LD_PRELOAD="${TCMALLOC}"

# Optional netem shaping on the fabric interface (best-effort).
if [[ -n "${NETEM_OPTS}" ]]; then
    if tc qdisc replace dev "${IFACE}" root netem ${NETEM_OPTS} 2>/dev/null; then
        [[ "${SLURM_PROCID}" == "0" ]] && echo "[netem] applied '${NETEM_OPTS}' on ${IFACE}"
    else
        [[ "${SLURM_PROCID}" == "0" ]] && \
          echo "WARNING: netem failed on ${IFACE} (no CAP_NET_ADMIN?) — running UNSHAPED" >&2
    fi
fi

COMMON="--hpx:ini=hpx.parcel.tcp.enable=1 --hpx:ini=hpx.parcel.bootstrap=tcp"
REPSFLAGS="--min-reps ${REPS} --max-reps ${REPS}"
THREADFLAG=""
[[ "${THREADS}" != "0" ]] && THREADFLAG="--hpx:threads=${THREADS}"

if [[ "${FABRIC}" == "eth1g" ]]; then
    ETH_IP=$(ip -o -4 addr show "${IFACE}" | awk '{print $4}' | cut -d/ -f1 | head -1)
    HOSTS=$(scontrol show hostnames "${SLURM_NODELIST}" | sed 's/$/-eth/' | tr '\n' ' ')
    FABRIC_FLAGS="--hpx:nodes ${HOSTS} --hpx:endnodes --hpx:node=${SLURM_PROCID} --hpx:ini=hpx.parcel.address=${ETH_IP}"
else
    FABRIC_FLAGS=""   # ipoib: pure SLURM auto-detection
fi

${NUMACTL} "${BENCH}" ${FABRIC_FLAGS} ${COMMON} ${THREADFLAG} \
    --op "${OP}" --n "${N_PER_NODE}" --vec-k "${VEC_K}" ${REPSFLAGS}
RC=$?

[[ -n "${NETEM_OPTS}" ]] && tc qdisc del dev "${IFACE}" root 2>/dev/null || true
exit ${RC}
TASKEOF

# ---------------------------------------------------------------------------
# Run via srun
OUTPUT=$(timeout 300 srun \
    --partition=medusa \
    --exclusive \
    --nodes="${NODES}" \
    --ntasks="${NODES}" \
    --ntasks-per-node=1 \
    bash -c "${PER_TASK}" 2>&1) && RC=0 || RC=$?

echo "${OUTPUT}"

if [[ ${RC} -ne 0 ]]; then
    echo "ERROR: srun exited with code ${RC}" >&2
    exit ${RC}
fi

# ---------------------------------------------------------------------------
# Parse JSON result lines and append to CSV.
echo "${OUTPUT}" | grep -E '^\{"op"' | while read -r LINE; do
    read -r OP_VAL LOC MED GE GB < <(python3 -c "
import sys, json
d = json.loads(sys.argv[1])
print(d.get('op',''), d.get('localities',''), d.get('median_s',''), d.get('gelem_s',''), d.get('gbytes_s',''))
" "${LINE}" 2>/dev/null)
    echo "${FABRIC},${INSTANCE},${NODES},${N_PER_NODE},${THREADS},${VEC_K},${OP_VAL},${LOC},${MED},${GE},${GB},${TIMESTAMP}" >> "${CSV_FILE}"
done

echo ""
echo "Results appended to: ${CSV_FILE}"
