"""
bo_objective.py -- the Bayesian-optimization reward for one trial: area under
the tau_E(t) curve, masking out non-finite entries (wherever dW/dt <= 0, per
compute_tau_E's own convention) and integrating only the valid stretches with
np.trapz. Reuses find_rundir / RunData / compute_tau_E from make_scan_plots.py
unchanged.

A trial that never produced a rundir, never reached the expected frame count,
ever wrote a non-finite pressure field, or has no valid tau_E stretch at all
gets a fixed `penalty` value instead of whatever partial area it happened to
accumulate -- otherwise a run that crashes early right after one big
transient tau_E spike could look deceptively good to the optimizer, since a
short interval containing a spike can integrate to more than a long, healthy,
but modest tau_E trace.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from make_scan_plots import RunData, compute_tau_E, find_rundir, compute_mode_spectrum  # noqa: E402

# np.trapz was removed in newer numpy (renamed to np.trapezoid in 2.0); fall
# back for older numpy that doesn't have trapezoid yet.
_trapz = getattr(np, "trapezoid", None) or np.trapz

METRIC = 'gamma'

def compute_bo_objective(outdir, expected_nframes, core_only=True):
    """Returns (reward, info) for one trial's output directory.

    reward:  float -- masked area under tau_E(t), or `penalty` if the run
             failed/was incomplete in any of the ways described above.
    info:    dict -- {"status": <str>, ...diagnostic fields...} for logging.

    `penalty` should be set BELOW the area you'd expect from your worst
    still-physically-valid run -- inspect a handful of real completed runs'
    areas first (e.g. via the standalone `python bo_objective.py <outdir>
    <expected_nframes>` usage below) before picking a number, since tau_E's
    plausible magnitude depends on your normalization/units and isn't
    something to guess blind.
    """
    if METRIC == 'gamma':
        penalty = -5.0

    else:
        penalty = 0.0

    rundir = find_rundir(outdir)
    if rundir is None:
        return penalty, {"status": "no_rundir", "outdir": outdir}

    try:
        run = RunData(rundir)
    except Exception as e:
        return penalty, {"status": f"load_failed: {e}", "outdir": outdir}

    try:
        if run.nf < expected_nframes:
            return penalty, {"status": "incomplete",
                              "n_frames": run.nf, "expected": expected_nframes,
                              "rundir": rundir}

        if METRIC == 'gamma':

            spec = compute_mode_spectrum(run)   # uses FIELD/RHO/LIMITER_RHO/
                                             # N_SAMPLE_POINTS/MMAX/AVERAGE
                                             # defaults from make_scan_plots.py

            if not np.isfinite(spec["A_tot"]).all():
                return penalty, {"status": "non-finite spectrum (blew up)", "rundir": rundir}

            g_tot = spec["g_tot"]
            if not np.isfinite(g_tot):
                return penalty, {"status": "growth fit failed", "rundir": rundir}

            reward = -float(g_tot)
            return reward, {"status": "ok", "g_tot": float(g_tot),
                            "r2": float(spec["r2"]), "n_frames": run.nf, "rundir": rundir}

        else:

            t, W, _ = compute_tau_E(run, core_only=core_only)

            if not np.isfinite(W).all():
                return penalty, {"status": "non-finite W (blew up)", "rundir": rundir}

            area = float(_trapz(W, t))
            return area, {"status": "ok", "n_valid": 1,
                        "n_frames": run.nf, "rundir": rundir}

    finally:
        run.close()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("outdir", help="the [output].outdir a trial's config pointed at")
    ap.add_argument("expected_nframes", type=int)
    ap.add_argument("--penalty", type=float, default=0.0)
    ap.add_argument("--no-core-only", action="store_true")
    args = ap.parse_args()

    reward, info = compute_bo_objective(args.outdir, args.expected_nframes,
                                         core_only=not args.no_core_only)
    print(f"reward = {reward:.6f}")
    for k, v in info.items():
        print(f"  {k}: {v}")