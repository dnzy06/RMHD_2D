#!/usr/bin/env python3
"""make_bias_scan_plots.py -- generate the standard diagnostic plots for every
run in a bias-profile scan (the .toml files written by generate_bias_configs.py),
laid out as:

    <plots_dir>/<profile_name>/bias_profile.png       (spline fit + 2nd derivative
                                                         of that config's own
                                                         ring_radii/ring_volts)
    <plots_dir>/<profile_name>/spectrum_growth.png
    <plots_dir>/<profile_name>/growth_vs_m.png
    <plots_dir>/<profile_name>/evolution_montage.png
    <plots_dir>/<profile_name>/instantaneous_gamma.png
    <plots_dir>/<profile_name>/radial_spectrum_3d.html
    <plots_dir>/tau_E_comparison.png       (all profiles overlaid)
    <plots_dir>/gamma_m1_comparison.png    (all profiles overlaid)

The mode-spectrum/growth/montage plots reuse make_scan_plots.py's RunData,
compute_mode_spectrum, plot_spectrum_growth, plot_growth_vs_m,
plot_evolution_montage, and find_rundir UNCHANGED; the instantaneous-gamma
plot, the radius-resolved 3D spectrum video, and the two cross-profile
comparison plots reuse make_run_plots.py's plot_instantaneous_gamma,
compute_radial_mode_spectrum/plot_radial_spectrum_video, compute_tau_E, der_t,
plot_tau_E_comparison, and plot_gamma_m1_comparison UNCHANGED -- this script
only adds the bias-profile plot and organizes/labels everything by profile
name instead of by U/H or outdir basename.

Usage:
    python scan/make_bias_scan_plots.py <configs_dir> <plots_dir>

    python scan/make_bias_scan_plots.py scan/bias_profile_scan_configs plots/bias_scan

<configs_dir> is the directory of generated .toml files (from
generate_bias_configs.py) -- each file's own [output].outdir is used to find
its completed run (same find_rundir() logic as make_scan_plots.py), and its
own [bias].ring_radii/ring_volts are read directly from the file to draw the
bias-profile plot (not re-evaluated from bias_profiles.py, so this reflects
exactly what was actually simulated even if the config was hand-edited).

Profile name for the output subfolder is read from the "# Profile: <name> --"
header line that generate_bias_configs.py writes at the top of each file;
falls back to the file's basename (without .toml) if that header isn't found.

Requires: numpy, h5py, matplotlib, scipy, tomllib (stdlib, Python 3.11+).
Optional: plotly (for radial_spectrum_3d.html -- skipped with a warning if absent).
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import tomllib

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless -- no display on the compute/login node
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_scan_plots import (  # noqa: E402
    FIELD,
    RunData,
    compute_mode_spectrum,
    find_rundir,
    plot_evolution_montage,
    plot_growth_vs_m,
    plot_spectrum_growth,
)
from make_run_plots import (  # noqa: E402
    compute_radial_mode_spectrum,
    compute_tau_E,
    der_t,
    plot_gamma_m1_comparison,
    plot_instantaneous_gamma,
    plot_radial_spectrum_video,
    plot_tau_E_comparison,
)

PROFILE_HEADER_RE = re.compile(r'^#\s*Profile:\s*(\S+)', re.MULTILINE)


def get_profile_name(toml_path, text):
    m = PROFILE_HEADER_RE.search(text)
    if m:
        return m.group(1)
    return os.path.splitext(os.path.basename(toml_path))[0]


def plot_bias_profile(radii, volts, title, out_path):
    cs = CubicSpline(radii, volts)

    r = np.linspace(min(radii), max(radii), 400)
    v = cs(r)
    v2 = cs(r, 2)  # second derivative

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(title, fontsize=12)

    ax1.plot(r, v, label="spline")
    ax1.scatter(radii, volts, color="red", zorder=5, label="data")
    ax1.set_xlabel("Radius")
    ax1.set_ylabel("Voltage")
    ax1.legend()
    ax1.set_title("Cubic Spline Fit")

    ax2.plot(r, v2, color="darkorange")
    ax2.axhline(0, color="gray", linewidth=0.8)
    ax2.set_xlabel("Radius")
    ax2.set_ylabel("2nd derivative")
    ax2.set_title("Second Derivative")

    plt.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("configs_dir", help="directory of generated bias-scan .toml files")
    ap.add_argument("plots_dir", help="directory to write <profile_name>/*.png into")
    args = ap.parse_args()

    toml_paths = sorted(glob.glob(os.path.join(args.configs_dir, "*.toml")))
    if not toml_paths:
        raise SystemExit(f"No .toml files found in {args.configs_dir}")

    os.makedirs(args.plots_dir, exist_ok=True)
    tau_results = []
    gamma_m1_results = []

    n_ok = 0
    for tp in toml_paths:
        text = open(tp, "r").read()
        with open(tp, "rb") as fh:
            cfg = tomllib.load(fh)

        name = os.path.basename(tp)
        profile_name = get_profile_name(tp, text)

        try:
            ring_radii = cfg["bias"]["ring_radii"]
            ring_volts = cfg["bias"]["ring_volts"]
        except KeyError:
            print(f"[skip] {name}: no [bias].ring_radii/ring_volts found")
            continue

        try:
            outdir = cfg["output"]["outdir"]
        except KeyError:
            print(f"[skip] {name}: no [output].outdir found")
            continue

        rundir = find_rundir(outdir)
        if rundir is None:
            print(f"[skip] {profile_name}: no completed run found under {outdir}")
            continue

        print(f"[{profile_name}] loading {rundir}")
        out_dir = os.path.join(args.plots_dir, profile_name)
        os.makedirs(out_dir, exist_ok=True)

        plot_bias_profile(ring_radii, ring_volts, profile_name,
                           os.path.join(out_dir, "bias_profile.png"))

        run = RunData(rundir)
        try:
            spec = compute_mode_spectrum(run)
            plot_spectrum_growth(run, spec, FIELD, os.path.join(out_dir, "spectrum_growth.png"))
            plot_growth_vs_m(run, spec, FIELD, os.path.join(out_dir, "growth_vs_m.png"))
            plot_evolution_montage(run, run.coord, os.path.join(out_dir, "evolution_montage.png"))
            plot_instantaneous_gamma(run, spec, FIELD, os.path.join(out_dir, "instantaneous_gamma.png"))

            radial_spec = compute_radial_mode_spectrum(run)
            plot_radial_spectrum_video(run, radial_spec, FIELD,
                                        os.path.join(out_dir, "radial_spectrum_3d.html"))

            mask_m1 = spec["A"][:, 1] > 0
            gamma_m1_t = der_t(np.log(spec["A"][mask_m1, 1]), run.t[mask_m1])
            gamma_m1_results.append((profile_name, run.t[mask_m1], gamma_m1_t))

            t, W, tau = compute_tau_E(run, core_only=True)
            tau_results.append((profile_name, t, tau))

            n_ok += 1
        except Exception as e:
            print(f"  [error] {profile_name}: {e}")
        finally:
            run.close()

        print(f"[{profile_name}] wrote plots to {out_dir}")

    if tau_results:
        plot_tau_E_comparison(tau_results, os.path.join(args.plots_dir, "tau_E_comparison.png"))
    else:
        print("No completed runs found for any config -- no tau_E comparison plot written.")

    if gamma_m1_results:
        plot_gamma_m1_comparison(gamma_m1_results, os.path.join(args.plots_dir, "gamma_m1_comparison.png"))
    else:
        print("No completed runs found for any config -- no gamma_m1 comparison plot written.")

    print(f"\nDone. {n_ok}/{len(toml_paths)} run(s) plotted under {args.plots_dir}")


if __name__ == "__main__":
    main()