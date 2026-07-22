#!/bin/bash
#
# run_files.sh -- reconstructed (original was deleted). Runs a list of
# config.toml files (bare filenames, directories, or paths) up to 4 at a
# time in parallel on one node's 4 GPUs, auto-detecting cart vs polar
# per file from its contents so you never have to say which module to use.
#
#   bash scan/run_files.sh input1.toml input2.toml ...
#   bash scan/run_files.sh scan/configs_polar_test
#   bash scan/run_files.sh scan/configs_polar_test input_extra.toml
#
# Name/dir resolution: bare names are searched under scan/configs/,
# scan/configs_polar/, scan/configs_cart/, scan/configs_polar_test/;
# directories are expanded to every *.toml inside (non-recursive, sorted).
#
# Auto-detect: polar configs use a [grid] table (nr/ntheta/r_max); cart
# configs use a [domain] table with nx/ny/Lx/Ly. Detected per file by
# grepping the toml -- see detect_kind() below.
#
# Each config's stdout/stderr goes to scan/logs/<basename>.out (no .toml),
# same convention as before. GPU assignment is round-robin over
# CUDA_VISIBLE_DEVICES=0..N_GPU-1 (override node's GPU count via
# VORTEX_GPUS_PER_NODE, default 4).

set -u
shopt -s nullglob

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."   # RMHD_2D/, so relative config paths resolve consistently

if [ "$#" -eq 0 ]; then
    echo "Usage: bash scan/run_files.sh <config1.toml|dir> [config2.toml|dir ...]" >&2
    exit 1
fi

module load python
conda activate plasma_reu
mkdir -p scan/logs

N_GPU="${VORTEX_GPUS_PER_NODE:-4}"

# Same resolution rule as run_multinode.sh/run_interactive.sh's resolve_target().
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

# Auto-detect cart vs polar from the toml contents. Primary signal is the
# section header ([grid] => polar, [domain]+nx/ny => cart); falls back to
# checking for the key names anywhere in the file for non-standard layouts.
detect_kind() {
    local f="$1"
    if grep -Eq '^[[:space:]]*\[grid\]' "$f"; then echo polar; return 0; fi
    if grep -Eq '^[[:space:]]*\[domain\]' "$f" && grep -Eq '^[[:space:]]*(nx|ny)[[:space:]]*=' "$f"; then
        echo cart; return 0
    fi
    if grep -Eq '^[[:space:]]*(nr|ntheta|r_max)[[:space:]]*=' "$f"; then echo polar; return 0; fi
    if grep -Eq '^[[:space:]]*(nx|ny|Lx|Ly)[[:space:]]*=' "$f"; then echo cart; return 0; fi
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
echo "Running $TOTAL config(s) on this node, up to $N_GPU at a time ($N_GPU GPUs)."

declare -a SLOT_PID=()
declare -a SLOT_NAME=()
fail=0

# Block until whatever was previously running in $1 (if anything) has
# finished, and record its exit status.
wait_slot() {
    local s="$1"
    if [ -n "${SLOT_PID[$s]:-}" ]; then
        if ! wait "${SLOT_PID[$s]}"; then
            local base; base="$(basename "${SLOT_NAME[$s]}" .toml)"
            echo "  FAILED: ${SLOT_NAME[$s]} (see scan/logs/${base}.out)" >&2
            fail=1
        fi
    fi
}

idx=0
for f in "${CONFIGS[@]}"; do
    slot=$(( idx % N_GPU ))
    wait_slot "$slot"

    kind=$(detect_kind "$f") || {
        echo "ERROR: could not auto-detect cart vs polar for '$f' (no [grid]/[domain] or nr/ntheta/r_max/nx/ny/Lx/Ly keys found) -- skipping." >&2
        fail=1
        idx=$((idx + 1))
        continue
    }
    mod="vortex_${kind}_jit"
    base="$(basename "$f" .toml)"
    echo "[gpu $slot] $base ($kind) -> $mod, logging to scan/logs/${base}.out"

    CUDA_VISIBLE_DEVICES="$slot" python -m "$mod" "$f" > "scan/logs/${base}.out" 2>&1 &
    SLOT_PID[$slot]=$!
    SLOT_NAME[$slot]="$f"
    idx=$((idx + 1))
done

# Drain whatever's left in each slot.
for ((slot = 0; slot < N_GPU; slot++)); do
    wait_slot "$slot"
done

if [ "$fail" -eq 0 ]; then
    echo "All $TOTAL config(s) done, all succeeded."
else
    echo "All $TOTAL config(s) done, at least one FAILED -- see above."
fi
exit "$fail"