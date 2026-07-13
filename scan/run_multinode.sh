#!/bin/bash
#SBATCH --job-name=vortex_multinode
#SBATCH --account=m4466
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --time=01:00:00
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --output=scan/logs/multinode_%j.out
#SBATCH --error=scan/logs/multinode_%j.err
#
# run_multinode.sh -- same idea as run_files.sh (auto-detect cart vs polar,
# accept bare filenames/dirs/paths, run up to 4 at a time on one node's 4
# GPUs) but fanned out across as many nodes as you request. Each node gets an
# even slice of the config list and runs run_files.sh UNCHANGED on its slice
# via srun -- so within a node, behavior/logging is identical to running
# run_files.sh by hand.
#
#   sbatch -N 2 scan/run_multinode.sh input1.toml input2.toml ...
#   sbatch -N 3 scan/run_multinode.sh scan/configs_polar_test
#   sbatch -N 4 scan/run_multinode.sh scan/configs_polar_test input_extra.toml
#
# -N on the command line is what actually controls node count (it overrides
# the #SBATCH --nodes=2 default above). Total parallel slots = nodes * 4.
# Same name/dir resolution rules as run_files.sh: bare names are searched
# under scan/configs/, scan/configs_polar/, scan/configs_cart/,
# scan/configs_polar_test/; directories are expanded to every *.toml inside
# (non-recursive, sorted); cart vs polar is still auto-detected per file
# inside run_files.sh, not here -- this script only cares about splitting the
# flattened file list evenly across nodes.
#
# WALLTIME: the #SBATCH --time=01:00:00 default above is NOT benchmarked for
# your grid/step count (same caveat as submit_configs_dir.sh) -- pass
# --time=HH:MM:SS to sbatch directly if you know your actual per-run time.

set -u
shopt -s nullglob

# NOTE: don't use "dirname ${BASH_SOURCE[0]}" here -- sbatch copies this script
# into /var/spool/slurmd/job<id>/slurm_script before running it, so BASH_SOURCE
# points at the spool copy, not the repo. SLURM_SUBMIT_DIR (the directory you
# ran `sbatch` from) is the reliable way back to the repo root.
cd "$SLURM_SUBMIT_DIR"   # RMHD_2D/, so relative config paths resolve the same as run_files.sh

if [ "$#" -eq 0 ]; then
    echo "Usage: sbatch -N <nodes> scan/run_multinode.sh <config1.toml|dir> [config2.toml|dir ...]" >&2
    exit 1
fi

module load python
conda activate plasma_reu
mkdir -p scan/logs

# Same resolution rule as run_files.sh's resolve_target(), minus the module
# detection (that's still done per-file inside run_files.sh on each node).
resolve_target() {
    local f="$1"
    if [ -e "$SLURM_SUBMIT_DIR/$f" ]; then echo "$SLURM_SUBMIT_DIR/$f"; return 0; fi
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
echo "Running $TOTAL config(s) across $N_NODES node(s), up to 4 at a time per node ($((N_NODES * 4)) total slots)."

# Split CONFIGS into N_NODES contiguous, roughly-even chunks (any remainder
# goes to the first few nodes, one extra config each).
base=$((TOTAL / N_NODES))
rem=$((TOTAL % N_NODES))
idx=0
pids=()
node_names=()
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
        bash scan/run_files.sh "${chunk[@]}" > "scan/logs/multinode_${SLURM_JOB_ID}_${node}.out" 2>&1 &
    pids+=($!)
    node_names+=("$node")
done

echo "Waiting for all nodes to finish..."
fail=0
for j in "${!pids[@]}"; do
    if ! wait "${pids[$j]}"; then
        echo "  FAILED on node ${node_names[$j]} (see scan/logs/multinode_${SLURM_JOB_ID}_${node_names[$j]}.out)" >&2
        fail=1
    fi
done

if [ "$fail" -eq 0 ]; then
    echo "All nodes done, all succeeded."
else
    echo "All nodes done, at least one node had a FAILURE -- check the logs above."
fi