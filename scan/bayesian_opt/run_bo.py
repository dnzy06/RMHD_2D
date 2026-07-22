#!/usr/bin/env python3
"""
run_bo.py -- resumable Bayesian-optimization driver for the bias-profile
search. Maximizes the masked area under the tau_E(t) curve (bo_objective.py)
by tuning 5 ring_volts values at 5 fixed ring_radii (generate_bo_config.py),
using skopt's ask/tell Optimizer (GP surrogate, EI acquisition) so real,
expensive SLURM evaluations can be batched B at a time instead of one at a
time.

Run this on a LOGIN NODE (not inside a compute job) via the free, long-running
"workflow" QOS, since it just submits/polls other jobs and does no compute
itself -- see the NERSC docs on the workflow QOS for how to register it
(https://docs.nersc.gov/jobs/workflow/workflow-queue/). A screen/tmux session
also works fine for a shorter test run.

Every trial is appended to bo_log.csv as soon as its reward is known, and on
startup this script re-seeds the optimizer from any existing bo_log.csv --
so it's safe to Ctrl-C and rerun; it picks up where it left off instead of
restarting the search or re-running completed trials.

--test uses a COMPLETELY SEPARATE log/config/outdir namespace (bo_log_test.csv,
configs_test/, .../bayesian_opt/test/trial_*) so shaking out the pipeline on
debug QOS never mixes into, or consumes budget from, the real campaign's
resumable state.

Usage:
    python run_bo.py                        # continue/start the real campaign (regular QOS)
    python run_bo.py --test                 # one small batch on debug QOS, to shake out the pipeline
    python run_bo.py --interactive          # run from INSIDE an active salloc allocation
                                             # instead of sbatch-ing a new job per batch --
                                             # see run_interactive_multinode.sh for the
                                             # salloc command to use first
    python run_bo.py --test --interactive   # shake out the interactive path specifically
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import re
import subprocess
import sys
import time
import tomllib

import numpy as np
from skopt import Optimizer
from skopt.space import Real

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))   # .../2D_sim
sys.path.insert(0, SCRIPT_DIR)
from generate_bo_config import build_trial_config           # noqa: E402
from bo_objective import compute_bo_objective                # noqa: E402

# ── Campaign parameters ──────────────────────────────────────────────────────
REFERENCE_TOML = os.path.join(SCRIPT_DIR, "reference.toml")
N_ELECTRODES = 5
V_MIN, V_MAX = -500.0, 500.0
N_CALLS = 300
BATCH_SIZE = 16                 # trials per batch; needs ceil(BATCH_SIZE/4) nodes
N_INITIAL = 50                   # LHS-sampled points before the GP starts steering
GPUS_PER_NODE = 4                # matches run_files.sh's fixed slot count
CORE_ONLY = True

MAIN_QOS = "regular"             # the real 300-run campaign should NOT run on debug
TEST_QOS = "debug"                # --test uses this instead (max 8 nodes, 0.5 hr)
BATCH_TIME_LIMIT = "00:20:00"    # !! placeholder -- time one real run of this exact
                                  #    config first and adjust (see earlier discussion)
POLL_INTERVAL_S = 30

TEST_N_CALLS = 8                 # small, fixed shakedown batch size for --test

OUTDIR_BASE = "/pscratch/sd/d/dnzy06/plasma_sim_runs/bayesian_opt"
CONFIGS_ROOT = os.path.join(SCRIPT_DIR, "configs")
LOG_PATH = os.path.join(SCRIPT_DIR, "bo_log.csv")

TEST_OUTDIR_BASE = os.path.join(OUTDIR_BASE, "test")
TEST_CONFIGS_ROOT = os.path.join(SCRIPT_DIR, "configs_test")
TEST_LOG_PATH = os.path.join(SCRIPT_DIR, "bo_log_test.csv")

LOG_FIELDS = ["trial_id", "batch_id"] + [f"v{i}" for i in range(N_ELECTRODES)] + \
             ["reward", "status", "outdir", "job_id"]

SBATCH_ID_RE = re.compile(r"Submitted batch job (\d+)")


# ── Reference-config-derived constants ───────────────────────────────────────
def load_reference_derived():
    with open(REFERENCE_TOML, "rb") as fh:
        cfg = tomllib.load(fh)
    r_max = float(cfg["grid"]["r_max"])
    nframes = int(cfg["time"]["nframes"])
    # span the full grid, not just out to limiter_radius -- otherwise phi_w's
    # CubicSpline has no data past the last ring and extrapolates unconstrained
    # polynomial values across the rest of the domain, which is what was
    # blowing trials up almost immediately regardless of ring_volts magnitude.
    ring_radii = np.linspace(0.1, r_max, N_ELECTRODES).tolist()
    # fields.h5 gets one write_frame() call before the main loop, plus one per
    # loop iteration (see vortex_polar_jit/__main__.py / vortex_cart_jit/__main__.py)
    # -- so the true output frame count is nframes + 1, NOT derived from nts/nfdump
    # (nts is steps-per-frame, and nfdump only controls restart-checkpoint cadence,
    # both unrelated to how many frames actually land in fields.h5).
    expected_nframes = nframes + 1
    return ring_radii, expected_nframes


# ── SLURM submit + poll ──────────────────────────────────────────────────────
def submit_and_wait(configs_dir, n_nodes, qos, time_limit):
    cmd = ["sbatch", f"--qos={qos}", f"-N{n_nodes}", f"--time={time_limit}",
           os.path.join(REPO_ROOT, "scan", "run_multinode.sh"), configs_dir]
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    m = SBATCH_ID_RE.search(result.stdout)
    if not m:
        raise RuntimeError(f"couldn't parse job id from sbatch output: {result.stdout!r}")
    job_id = m.group(1)
    print(f"  submitted job {job_id} ({n_nodes} node(s), qos={qos}, time={time_limit})")

    while True:
        time.sleep(POLL_INTERVAL_S)
        q = subprocess.run(["squeue", "-j", job_id, "-h"], capture_output=True, text=True)
        if not q.stdout.strip():
            break
    print(f"  job {job_id} finished")
    return job_id


def run_batch_interactive(configs_dir):
    """Run one batch directly against nodes already held by an active salloc
    allocation -- no sbatch, no queue wait. Blocks until the batch finishes
    (run_interactive_multinode.sh itself waits on every node's srun).

    Raises on any failure (missing script, non-zero exit) instead of just
    warning -- a silently-swallowed failure here means every trial in the
    batch gets logged as a fake no_rundir/penalty result, burning through
    the real trial budget on nothing."""
    script = os.path.join(REPO_ROOT, "scan", "run_interactive_multinode.sh")
    if not os.path.isfile(script):
        raise RuntimeError(
            f"{script} does not exist -- create it on this machine before "
            f"running --interactive (it does not get copied automatically)."
        )
    result = subprocess.run(["bash", script, configs_dir], cwd=REPO_ROOT)
    if result.returncode != 0:
        raise RuntimeError(
            f"run_interactive_multinode.sh exited {result.returncode} -- "
            f"check scan/logs/interactive_multinode_*.out before retrying."
        )
    return "interactive"   # no real SLURM job id in this mode; logged as a placeholder


# ── Resumable logging ─────────────────────────────────────────────────────────
def load_completed_trials(log_path):
    if not os.path.exists(log_path):
        return [], []
    points, neg_rewards = [], []
    with open(log_path, "r", newline="") as fh:
        for row in csv.DictReader(fh):
            v = [float(row[f"v{i}"]) for i in range(N_ELECTRODES)]
            points.append(v)
            neg_rewards.append(-float(row["reward"]))
    return points, neg_rewards


def append_log_row(log_path, row):
    write_header = not os.path.exists(log_path)
    with open(log_path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=LOG_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow(row)


def report_best(log_path):
    best_row, best_reward = None, -np.inf
    with open(log_path, "r", newline="") as fh:
        for row in csv.DictReader(fh):
            r = float(row["reward"])
            if r > best_reward:
                best_reward, best_row = r, row
    print("\n" + "=" * 60)
    print("BEST TRIAL SO FAR")
    print(f"  trial_id : {best_row['trial_id']}")
    print(f"  reward   : {best_reward:.6f}")
    print(f"  voltages : {[round(float(best_row[f'v{i}']), 4) for i in range(N_ELECTRODES)]}")
    print(f"  outdir   : {best_row['outdir']}")
    print("=" * 60)


# ── Main loop ─────────────────────────────────────────────────────────────────
def run_campaign(n_calls, batch_size, qos, outdir_base, configs_root, log_path,
                  interactive=False):
    ring_radii, expected_nframes = load_reference_derived()
    print(f"ring_radii = {[round(r, 4) for r in ring_radii]}")
    print(f"expected_nframes = {expected_nframes} (from reference.toml [time])")

    space = [Real(V_MIN, V_MAX, name=f"v{i}") for i in range(N_ELECTRODES)]
    opt = Optimizer(space, base_estimator="GP", acq_func="EI",
                     n_initial_points=N_INITIAL, initial_point_generator="lhs",
                     random_state=42)

    prev_points, prev_neg_rewards = load_completed_trials(log_path)
    if prev_points:
        opt.tell(prev_points, prev_neg_rewards)
        print(f"resumed: re-seeded optimizer with {len(prev_points)} completed trial(s)")

    trial_id = len(prev_points)
    batch_id = trial_id // batch_size
    os.makedirs(configs_root, exist_ok=True)
    ref_text = open(REFERENCE_TOML, "r").read()

    while trial_id < n_calls:
        n = min(batch_size, n_calls - trial_id)
        points = opt.ask(n_points=n)

        batch_dir = os.path.join(configs_root, f"batch_{batch_id:04d}")
        os.makedirs(batch_dir, exist_ok=True)

        outdirs = []
        for i, v in enumerate(points):
            tid = trial_id + i
            outdir = os.path.join(outdir_base, f"trial_{tid:05d}")
            text = build_trial_config(ref_text, ring_radii, v, outdir, tid,
                                       ref_name="reference.toml")
            cfg_path = os.path.join(batch_dir, f"trial_{tid:05d}.toml")
            with open(cfg_path, "w") as fh:
                fh.write(text)
            outdirs.append(outdir)

        if interactive:
            print(f"\nbatch {batch_id}: {n} trial(s) (running against current salloc allocation)")
            job_id = run_batch_interactive(batch_dir)
        else:
            n_nodes = math.ceil(n / GPUS_PER_NODE)
            print(f"\nbatch {batch_id}: {n} trial(s), {n_nodes} node(s)")
            job_id = submit_and_wait(batch_dir, n_nodes, qos, BATCH_TIME_LIMIT)

        rewards = []
        for i, v in enumerate(points):
            tid = trial_id + i
            reward, info = compute_bo_objective(outdirs[i], expected_nframes,
                                                 core_only=CORE_ONLY)
            rewards.append(reward)
            row = {"trial_id": tid, "batch_id": batch_id, "reward": reward,
                   "status": info["status"], "outdir": outdirs[i], "job_id": job_id}
            row.update({f"v{j}": v[j] for j in range(N_ELECTRODES)})
            append_log_row(log_path, row)
            print(f"  trial {tid}: reward={reward:.6f}  status={info['status']}")

        opt.tell(points, [-r for r in rewards])
        trial_id += n
        batch_id += 1

    report_best(log_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true",
                     help="run a small shakedown batch on debug QOS, in a separate "
                          "log/config/outdir namespace -- does not touch the real campaign")
    ap.add_argument("--interactive", action="store_true",
                     help="run batches directly against an already-active salloc "
                          "allocation (via run_interactive_multinode.sh) instead of "
                          "submitting a new sbatch job per batch. Run this from INSIDE "
                          "an interactive allocation, e.g. after: "
                          "salloc -N 4 -C gpu -q interactive -t 04:00:00 -A m4466 "
                          "--gpus-per-node=4")
    args = ap.parse_args()

    if args.test:
        run_campaign(TEST_N_CALLS, min(TEST_N_CALLS, BATCH_SIZE), TEST_QOS,
                     TEST_OUTDIR_BASE, TEST_CONFIGS_ROOT, TEST_LOG_PATH,
                     interactive=args.interactive)
    else:
        run_campaign(N_CALLS, BATCH_SIZE, MAIN_QOS,
                     OUTDIR_BASE, CONFIGS_ROOT, LOG_PATH,
                     interactive=args.interactive)


if __name__ == "__main__":
    main()