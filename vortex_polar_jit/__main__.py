"""Entry point:  python -m vortex_polar_jit [input.toml]

Runs the POLAR (r, theta) Beklemishev vortex-confinement model (annular scaffold).
"""
from __future__ import annotations

import os
import sys
import time
import tomllib

# --- precision: read [precision] mode and export VORTEX_PREC BEFORE importing the
#     backend (jax_enable_x64 must be set at jax import time). ---
_INP = sys.argv[1] if len(sys.argv) > 1 else \
    os.path.join(os.path.dirname(__file__), "input_polar.toml")
try:
    with open(_INP, "rb") as _fh:
        _mode = tomllib.load(_fh).get("precision", {}).get("mode", "double")
except Exception:
    _mode = "double"
os.environ.setdefault("VORTEX_PREC", str(_mode))

import numpy as np

from .backend import xp, asarray, BACKEND, HAS_JIT, DTYPE
from .config import load_config
from .state import State
from .stepper import make_run_frame, _kpars
from .kernels import getphi
from .output_hdf5 import Output, load_restart


def init_state(cfg):
    """Fresh start: plasma at rest (vort=phi=0) with the initial pressure profile."""
    pres0 = asarray(cfg.pres0)
    z = xp.zeros_like(pres0)
    return State(vort=z, vorti=z, pres=pres0, presi=pres0, phi=z)


def main():
    cfg = load_config(_INP)

    print(f"[2D_Vortex_Polar_JIT] backend={BACKEND}  jit={HAS_JIT}  prec={DTYPE}  "
          f"grid={cfg.nr}x{cfg.ntheta} (r x theta)  r in ({cfg.r_in:g},{cfg.r_max:g}]  "
          f"dt={cfg.dt:.2e}")
    cfg.print_normalization()
    print(f"[2D_Vortex_Polar_JIT] U={cfg.U} H={cfg.H} kappa={cfg.kappa}  nu4={cfg.nu4} "
          f"nu5={cfg.nu5} nu4p={cfg.nu4p} nu5p={cfg.nu5p}  limiter r={cfg.limiter_radius} "
          f"x{cfg.limiter_factor}  bias rings={cfg.n_rings}")

    jobid = os.environ.get("SLURM_JOBID") or os.environ.get("SLURM_JOB_ID")
    rundir = os.path.join(cfg.outdir, jobid) if jobid else cfg.outdir

    step0, t0 = 0, 0.0
    state = init_state(cfg)
    if cfg.nrst == 1:
        rpath = os.path.join(cfg.restartdir, cfg.restart_file)
        arrs, step0, t0 = load_restart(rpath)
        vorti = asarray(arrs["vorti"])
        phi = getphi(vorti, _kpars(cfg))
        state = State(vort=asarray(arrs["vort"]), vorti=vorti,
                      pres=asarray(arrs["pres"]), presi=asarray(arrs["presi"]), phi=phi)
        print(f"[2D_Vortex_Polar_JIT] restarted from {rpath}: step={step0} t={t0:.4f}")

    out = Output(cfg, rundir)
    run_frame = make_run_frame(cfg)
    frame_dt = cfg.nts * cfg.dt

    step, t = step0, t0
    out.write_frame(state, t)
    wall0 = time.perf_counter()

    for frame in range(cfg.nframes):
        tf = time.perf_counter()
        state = run_frame(state)
        try:
            state.phi.block_until_ready()
        except Exception:
            pass
        step += cfg.nts; t += frame_dt
        out.write_frame(state, t)

        P = np.asarray(state.presi); W = np.asarray(state.vorti)
        pmin, pmax = float(P.min()), float(P.max()); wmax = float(np.abs(W).max())
        dtf = time.perf_counter() - tf
        print(f"frame {frame + 1}/{cfg.nframes}  t={t:.4f} "
              f"({t * cfg.t_bar * 1e3:.3e} ms)  P[{pmin:.3e},{pmax:.3e}] "
              f"|vort|max={wmax:.3e}  {dtf:.2f}s ({1e3 * dtf / cfg.nts:.3f} ms/step)")
        if not (np.isfinite(pmax) and np.isfinite(wmax)):
            print("[2D_Vortex_Polar_JIT] NaN/Inf detected — stopping."); break
        if (frame + 1) % cfg.nfdump == 0:
            out.write_restart(state, step, t)

    out.write_restart(state, step, t)
    print(f"[2D_Vortex_Polar_JIT] done. wall={time.perf_counter() - wall0:.1f}s "
          f"output -> {rundir}")


if __name__ == "__main__":
    main()
