#!/usr/bin/env python3
"""
backfill_batch.py -- recover rewards for a batch whose simulations already
finished, but whose rewards never got logged into bo_log.csv (e.g. because
compute_bo_objective crashed on the np.trapz/np.trapezoid rename before any
row got appended). Reads each trial's already-written config file to recover
the exact ring_volts used for that trial (rather than re-calling opt.ask(),
which is not guaranteed to reproduce the same points), computes the reward
via compute_bo_objective (unchanged), and appends rows to bo_log.csv -- so
run_bo.py can resume at the next batch instead of re-running already-completed
simulations.

Usage:
    python backfill_batch.py                 # backfill batch 0 of the real campaign
    python backfill_batch.py --batch 2        # backfill a different batch index
    python backfill_batch.py --test           # backfill into the --test namespace instead
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import tomllib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from run_bo import (          # noqa: E402
    CORE_ONLY, PENALTY, N_ELECTRODES,
    CONFIGS_ROOT, OUTDIR_BASE, LOG_PATH,
    TEST_CONFIGS_ROOT, TEST_OUTDIR_BASE, TEST_LOG_PATH,
    append_log_row, load_reference_derived,
)
from bo_objective import compute_bo_objective   # noqa: E402

TRIAL_ID_RE = re.compile(r"trial_(\d+)\.toml$")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", type=int, default=0, help="batch index to backfill (default 0)")
    ap.add_argument("--test", action="store_true",
                     help="backfill into the --test namespace (bo_log_test.csv/configs_test) "
                          "instead of the real campaign")
    ap.add_argument("--job-id", default="interactive",
                     help="value to log in the job_id column (default 'interactive', "
                          "matching a run_batch_interactive batch)")
    args = ap.parse_args()

    configs_root = TEST_CONFIGS_ROOT if args.test else CONFIGS_ROOT
    outdir_base = TEST_OUTDIR_BASE if args.test else OUTDIR_BASE
    log_path = TEST_LOG_PATH if args.test else LOG_PATH

    batch_dir = os.path.join(configs_root, f"batch_{args.batch:04d}")
    toml_files = sorted(glob.glob(os.path.join(batch_dir, "trial_*.toml")))
    if not toml_files:
        raise SystemExit(f"no trial_*.toml files found under {batch_dir}")

    _, expected_nframes = load_reference_derived()
    print(f"backfilling {len(toml_files)} trial(s) from {batch_dir} "
          f"(expected_nframes={expected_nframes})\n")

    for path in toml_files:
        m = TRIAL_ID_RE.search(path)
        if not m:
            print(f"  skipping {path} (couldn't parse trial id)")
            continue
        tid = int(m.group(1))

        with open(path, "rb") as fh:
            cfg = tomllib.load(fh)
        volts = [float(v) for v in cfg["bias"]["ring_volts"]]
        outdir = cfg["output"]["outdir"]
        if len(volts) != N_ELECTRODES:
            raise SystemExit(f"{path}: expected {N_ELECTRODES} ring_volts, got {len(volts)}")

        reward, info = compute_bo_objective(outdir, expected_nframes,
                                             core_only=CORE_ONLY, penalty=PENALTY)
        row = {"trial_id": tid, "batch_id": args.batch, "reward": reward,
               "status": info["status"], "outdir": outdir, "job_id": args.job_id}
        row.update({f"v{j}": volts[j] for j in range(N_ELECTRODES)})
        append_log_row(log_path, row)
        print(f"  trial {tid}: reward={reward:.6f}  status={info['status']}")

    print(f"\ndone -- appended {len(toml_files)} row(s) to {log_path}")
    print("you can now resume with the normal run_bo.py command; it will pick up "
          "at the next trial instead of re-running this batch.")


if __name__ == "__main__":
    main()