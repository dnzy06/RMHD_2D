#!/usr/bin/env python3
"""
plot_bo_results.py -- generate the standard set of plots for a bo_log.csv
written by run_bo.py:

    plots/bayesian_opt/convergence.png       (best reward so far vs trial #)
    plots/bayesian_opt/all_evaluations.png   (every trial's reward + best-so-far)
    plots/bayesian_opt/optimal_bias_profile.png   (best trial's ring_volts, spline fit)
    plots/bayesian_opt/optimal_tau_E.png          (best trial's tau_E(t) curve)
    plots/bayesian_opt/optimal_evolution_montage.png  (best trial's field snapshots)
    plots/bayesian_opt/optimal_spectrum_growth.png    (best trial's mode spectrum +
                                                        manual-window growth-rate fit,
                                                        same T0_MANUAL/T1_MANUAL window
                                                        the gamma objective itself uses)

Reuses plot_bias_profile (make_bias_scan_plots.py), plot_tau_E_comparison /
plot_evolution_montage (make_run_plots.py), and RunData / compute_tau_E /
find_rundir (make_scan_plots.py) unchanged -- this script only adds the
convergence/all-evaluations plots and wires the best trial's data into the
existing plotting functions.

Usage:
    python plot_bo_results.py                      # uses run_bo.py's default bo_log.csv
    python plot_bo_results.py --log bo_log_test.csv --out plots/bo_test
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, ".."))

from run_bo import load_reference_derived, N_ELECTRODES, LOG_PATH as DEFAULT_LOG_PATH  # noqa: E402
from make_scan_plots import (RunData, compute_tau_E, find_rundir,                      # noqa: E402
                              compute_mode_spectrum, plot_spectrum_growth, FIELD)
from make_run_plots import plot_evolution_montage, plot_tau_E_comparison               # noqa: E402
from make_bias_scan_plots import plot_bias_profile                                     # noqa: E402


def load_log(log_path):
    rows = []
    with open(log_path, "r", newline="") as fh:
        for row in csv.DictReader(fh):
            row["reward"] = float(row["reward"])
            row["trial_id"] = int(row["trial_id"])
            rows.append(row)
    rows.sort(key=lambda r: r["trial_id"])
    return rows


def plot_convergence(rows, out_path):
    rewards = np.array([r["reward"] for r in rows])
    best_so_far = np.maximum.accumulate(rewards)
    fig = plt.figure(figsize=(8, 5))
    plt.plot(best_so_far, "o-", markersize=3, color="red")
    plt.xlabel("Evaluation #", fontsize=12)
    plt.ylabel("Best reward so far (tau_E area)", fontsize=12)
    plt.title("Bayesian optimization convergence", fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_all_evaluations(rows, out_path):
    rewards = np.array([r["reward"] for r in rows])
    best_so_far = np.maximum.accumulate(rewards)
    fig = plt.figure(figsize=(8, 5))
    plt.scatter(range(len(rewards)), rewards, s=10, alpha=0.6, color="steelblue",
                label="reward per eval")
    plt.plot(best_so_far, color="red", linewidth=2, label="Best so far")
    plt.xlabel("Evaluation #", fontsize=12)
    plt.ylabel("reward (tau_E area)", fontsize=12)
    plt.title("Reward across all evaluations", fontsize=13)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", default=DEFAULT_LOG_PATH, help="path to bo_log.csv")
    ap.add_argument("--out", default=os.path.join(SCRIPT_DIR, "plots"),
                    help="directory to write plots into")
    ap.add_argument("--core-only", action="store_true", default=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rows = load_log(args.log)
    if not rows:
        raise SystemExit(f"no rows in {args.log}")

    ok_rows = [r for r in rows if r["status"] == "ok"]
    if not ok_rows:
        print("WARNING: no trial has status=='ok' -- best trial will be picked from "
              "penalized/failed trials, which is probably not meaningful.")
    pool = ok_rows or rows
    best = max(pool, key=lambda r: r["reward"])

    print(f"{len(rows)} trial(s) loaded ({len(ok_rows)} with status=='ok')")
    print(f"best trial: id={best['trial_id']}  reward={best['reward']:.6f}  "
          f"outdir={best['outdir']}")

    plot_convergence(rows, os.path.join(args.out, "convergence.png"))
    plot_all_evaluations(rows, os.path.join(args.out, "all_evaluations.png"))

    ring_radii, _ = load_reference_derived()
    best_volts = [float(best[f"v{i}"]) for i in range(N_ELECTRODES)]
    plot_bias_profile(ring_radii, best_volts,
                       f"optimal bias profile (reward={best['reward']:.4f})",
                       os.path.join(args.out, "optimal_bias_profile.png"))

    rundir = find_rundir(best["outdir"])
    if rundir is None:
        print(f"WARNING: could not find a completed run under {best['outdir']} -- "
              f"skipping tau_E / evolution-montage plots for the best trial.")
    else:
        run = RunData(rundir)
        try:
            t, W, tau = compute_tau_E(run, core_only=args.core_only)
            plot_tau_E_comparison([(f"trial {best['trial_id']}", t, tau)],
                                   os.path.join(args.out, "optimal_tau_E.png"))
            plot_evolution_montage(run, run.coord,
                                    os.path.join(args.out, "optimal_evolution_montage.png"))

            # same call (same defaults -> same T0_MANUAL/T1_MANUAL fit window) as
            # bo_objective.py's gamma branch, so this is the actual quantity that
            # was optimized for the best trial, not a stand-in.
            spec = compute_mode_spectrum(run)
            plot_spectrum_growth(run, spec, FIELD,
                                  os.path.join(args.out, "optimal_spectrum_growth.png"))
        finally:
            run.close()

    print(f"\nplots written to {args.out}/")


if __name__ == "__main__":
    main()