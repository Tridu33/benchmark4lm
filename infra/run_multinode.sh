#!/bin/bash
# Multi-node communication benchmark launcher
# Usage:
#   ./run_multinode.sh hosts.txt 8 nccl
#
# hosts.txt format (one per line):
#   node1 slots=8
#   node2 slots=8
#   node3 slots=8

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ $# -lt 2 ]; then
    echo "Usage: $0 <hostfile> <total_gpus> [backend] [output]"
    echo ""
    echo "  hostfile    - MPI hostfile (one host per line)"
    echo "  total_gpus  - Total number of GPUs across all nodes"
    echo "  backend     - Communication backend (default: nccl)"
    echo "  output      - Output JSON path (default: results/comm_perf_multinode.json)"
    echo ""
    echo "Example:"
    echo "  $0 hosts.txt 16 nccl results/multinode_16gpu.json"
    exit 1
fi

HOSTFILE="$1"
TOTAL_GPUS="$2"
BACKEND="${3:-nccl}"
OUTPUT="${4:-results/comm_perf_multinode.json}"

MASTER_ADDR=$(head -1 "$HOSTFILE" | awk '{print $1}')
export MASTER_ADDR
export MASTER_PORT="${MASTER_PORT:-29500}"

echo "============================================================"
echo "  Multi-node Communication Benchmark"
echo "============================================================"
echo "  Hostfile:   $HOSTFILE"
echo "  Total GPUs: $TOTAL_GPUS"
echo "  Backend:    $BACKEND"
echo "  Master:     $MASTER_ADDR:$MASTER_PORT"
echo "  Output:     $OUTPUT"
echo "============================================================"

# Try torchrun first (recommended), fallback to mpirun
if command -v torchrun &> /dev/null; then
    echo ""
    echo "Using torchrun..."
    torchrun \
        --nnodes="$(wc -l < "$HOSTFILE")" \
        --nproc-per-node="$((TOTAL_GPUS / $(wc -l < "$HOSTFILE")))" \
        --master-addr="$MASTER_ADDR" \
        --master-port="$MASTER_PORT" \
        "$SCRIPT_DIR/02_communication_perf.py" \
        --backend "$BACKEND" \
        --output "$OUTPUT"
elif command -v mpirun &> /dev/null; then
    echo ""
    echo "Using mpirun..."
    mpirun \
        -np "$TOTAL_GPUS" \
        -hostfile "$HOSTFILE" \
        -x MASTER_ADDR="$MASTER_ADDR" \
        -x MASTER_PORT="$MASTER_PORT" \
        python3 "$SCRIPT_DIR/02_communication_perf.py" \
        --backend "$BACKEND" \
        --output "$OUTPUT"
else
    echo "ERROR: Neither torchrun nor mpirun found."
    echo "Install torchrun: pip install torch"
    echo "Install mpirun: apt install openmpi-bin || yum install openmpi"
    exit 1
fi

echo ""
echo "============================================================"
echo "  Results saved to: $OUTPUT"
echo "============================================================"
