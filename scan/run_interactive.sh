#!/bin/bash
#
# run_interactive.sh -- same config resolution as run_multinode.sh (auto-detect
# cart vs polar, accept bare filenames/dirs/paths, run up to 4 at a time on
# one node's 4 GPUs via run_files.sh UNCHANGED) but meant to be run directly
# from inside an interactive allocation on a single GPU node -- no sbatch,
# no srun, no node-splitting.
#
# Usage (from inside an interactive session, e.g. after):
#   salloc -N 1 -C gpu -q interactive -t 01:00:00 -A m4466 --gpus-per-node=4
#
#   bash scan/run_interactive.sh input1.toml input2.toml ...
#   bash scan/run_interactive.sh scan/configs_polar_test
#   bash scan/run_interactive.sh scan/configs_polar_test input_extra.toml
#
# Same name/dir resolution rules as run_multinode.sh/run_files.sh: bare names
# are searched under scan/configs/, scan/configs_polar/, scan/configs_cart/,
# scan/configs_polar_test/; directories are expanded to every *.toml inside
# (non-recursive, sorted); cart vs polar is still auto-detected per file
# inside run_files.sh.

set -u
shopt -s nullglob

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."   # RMHD_2D/, so relative config paths resolve the same as run_files.sh

if [ "$#" -eq 0 ]; then
    echo "Usage: bash scan/run_interactive.sh <config1.toml|dir> [config2.toml|dir ...]" >&2
    exit 1
fi

module load python
conda activate plasma_reu
mkdir -p scan/logs

# Same resolution rule as run_multinode.sh's resolve_target(), minus the
# SLURM_SUBMIT_DIR check (no sbatch submit dir in an interactive session --
# fall back to the directory this was invoked from instead).
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
echo "Running $TOTAL config(s) on this node, up to 4 at a time (4 GPUs)."

LOG="scan/logs/interactive_$(date +%Y%m%d_%H%M%S).out"
bash scan/run_files.sh "${CONFIGS[@]}" 2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}

if [ "$status" -eq 0 ]; then
    echo "Done, all succeeded. (log: $LOG)"
else
    echo "FAILED -- see $LOG" >&2
fi
exit "$status"