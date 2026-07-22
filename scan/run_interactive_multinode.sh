#!/bin/bash
#
# run_interactive_multinode.sh -- same config resolution + node-splitting as
# run_multinode.sh (auto-detect cart vs polar, split configs evenly across
# nodes, run up to 4 at a time per node's 4 GPUs via run_files.sh UNCHANGED)
# but meant to be run directly from inside an interactive MULTI-node
# allocation -- no sbatch, no queue wait: it uses srun against the nodes
# you already have.
#
# Usage (from inside an interactive allocation, e.g. after):
#   salloc -N 4 -C gpu -q interactive -t 04:00:00 -A m4466 --gpus-per-node=4
#
#   bash scan/run_interactive_multinode.sh input1.toml input2.toml ...
#   bash scan/run_interactive_multinode.sh scan/bayesian_opt/configs/batch_0000
#
# Same name/dir resolution and node-splitting rules as run_multinode.sh; the
# only difference is this script is invoked directly (bash ...) rather than
# submitted via sbatch, so it fans out across whatever nodes are ALREADY
# allocated to the current session instead of requesting new ones.

set -u
shopt -s nullglob

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."   # RMHD_2D/, so relative config paths resolve the same as run_files.sh

if [ "$#" -eq 0 ]; then
    echo "Usage: bash scan/run_interactive_multinode.sh <config1.toml|dir> [config2.toml|dir ...]" >&2
    exit 1
fi

if [ -z "${SLURM_JOB_NODELIST:-}" ]; then
    echo "ERROR: no SLURM_JOB_NODELIST set -- run this from inside an active salloc allocation" \
         "(e.g. salloc -N 4 -C gpu -q interactive -t 04:00:00 -A m4466 --gpus-per-node=4)." >&2
    exit 1
fi

module load python
conda activate plasma_reu
mkdir -p scan/logs

# Same resolution rule as run_multinode.sh's resolve_target(), falling back to
# $PWD if SLURM_SUBMIT_DIR isn't set (salloc sessions do set it, same as sbatch,
# but this mirrors run_interactive.sh's defensive fallback just in case).
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
resolve_target() {
    local f="$1"
    if [ -e "$SUBMIT_DIR/$f" ]; then echo "$SUBMIT_DIR/$f"; return 0; fi
    if [ -e "$f" ]; then echo "$f"; return 0; fi
    for d in scan/configs scan/configs_polar scan/configs_cart scan/configs_polar_test; do
        if [ -e "$d/$f" ]; then echo "$d/$f"; return 0; fi
    done
    return 1
}

CONFIGS=()
for f in "$@"; do
    resolved=$(resolve_target "$f") || {
        echo "ERROR: could not find '$f' (checked as given, and under scan/configs*/)." >&2
        exit 1
    }
    if [ -d "$resolved" ]; then
        mapfile -t tomls < <(printf '%s\n' "$resolved"/*.toml | sort)
        if [ ${#tomls[@]} -eq 0 ]; then
            echo "ERROR: '$resolved' is a directory but contains no .toml files." >&2
            exit 1
        fi
        CONFIGS+=("${tomls[@]}")
    else
        CONFIGS+=("$resolved")
    fi
done

TOTAL=${#CONFIGS[@]}
mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
N_NODES=${#NODES[@]}
echo "Running $TOTAL config(s) across $N_NODES already-allocated node(s), up to 4 at a time per node ($((N_NODES * 4)) total slots)."

# Same even-split logic as run_multinode.sh.
base=$((TOTAL / N_NODES))
rem=$((TOTAL % N_NODES))
idx=0
pids=()
node_names=()
TS=$(date +%Y%m%d_%H%M%S)
for n in "${!NODES[@]}"; do
    node=${NODES[$n]}
    count=$base
    if [ "$n" -lt "$rem" ]; then count=$((count + 1)); fi
    if [ "$count" -eq 0 ]; then
        echo "[node $node] nothing to run (more nodes than configs)"
        continue
    fi
    chunk=("${CONFIGS[@]:idx:count}")
    idx=$((idx + count))
    echo "[node $node] ${#chunk[@]} config(s): ${chunk[*]}"
    srun --nodes=1 --ntasks=1 --gpus-per-node=4 --exclusive -w "$node" \
        bash scan/run_files.sh "${chunk[@]}" > "scan/logs/interactive_multinode_${TS}_${node}.out" 2>&1 &
    pids+=($!)
    node_names+=("$node")
done

echo "Waiting for all nodes to finish..."
fail=0
for j in "${!pids[@]}"; do
    if ! wait "${pids[$j]}"; then
        echo "  FAILED on node ${node_names[$j]} (see scan/logs/interactive_multinode_${TS}_${node_names[$j]}.out)" >&2
        fail=1
    fi
done

if [ "$fail" -eq 0 ]; then
    echo "All nodes done, all succeeded."
else
    echo "All nodes done, at least one node had a FAILURE -- check the logs above."
fi
exit "$fail"