"""output_hdf5.py — HDF5 output for the POLAR Beklemishev vortex model.

  setup.h5    : polar grid (r, theta) + Cartesian mesh (x, y = r cos/sin theta,
                for plotting) + static profiles + normalization bases.  The
                attribute coord_type="polar" lets analysis tools branch.
  fields.h5   : resizable datasets phi, vort, pres (nframe, nr, ntheta) + time t.
  restart.h5  : vort, vorti, pres, presi + step/time attrs.
"""
from __future__ import annotations

import os
import numpy as np
import h5py

from .backend import to_host, DTYPE


class Output:
    def __init__(self, cfg, rundir):
        os.makedirs(rundir, exist_ok=True)
        self.rundir = rundir
        self.path_fields = os.path.join(rundir, "fields.h5")
        self.path_restart = os.path.join(rundir, cfg.restart_file)
        self.nframe = 0

        R, TH = np.meshgrid(cfg.r, cfg.theta, indexing="ij")   # (nr, ntheta)
        with h5py.File(os.path.join(rundir, "setup.h5"), "w") as f:
            f.attrs["coord_type"] = "polar"
            f["nr"] = cfg.nr; f["ntheta"] = cfg.ntheta
            f["r"] = np.asarray(cfg.r); f["theta"] = np.asarray(cfg.theta)
            f["x"] = R * np.cos(TH); f["y"] = R * np.sin(TH)   # for Cartesian plots
            f["phi_w"] = to_host(cfg.phi_w)
            f["nu5p_field"] = to_host(cfg.nu5p_field)
            f["pres0"] = to_host(cfg.pres0)
            for k in ("B0", "n0", "T0", "L0", "rho_s", "cs", "Omega_i",
                      "t_bar", "phi_bar", "P_bar", "v_bar",
                      "U", "H", "kappa", "nu4", "nu5", "nu4p", "nu5p",
                      "limiter_radius", "limiter_factor", "r_max", "r_in", "dt"):
                f.attrs[k] = float(getattr(cfg, k))

        with h5py.File(self.path_fields, "w") as f:
            for name in ("phi", "vort", "pres"):
                f.create_dataset(name, shape=(0, cfg.nr, cfg.ntheta),
                                 maxshape=(None, cfg.nr, cfg.ntheta),
                                 chunks=(1, cfg.nr, cfg.ntheta), dtype=DTYPE)
            f.create_dataset("t", shape=(0,), maxshape=(None,), dtype="float64")

    def write_frame(self, state, t):
        with h5py.File(self.path_fields, "a") as f:
            for name, arr in (("phi", state.phi), ("vort", state.vorti),
                              ("pres", state.presi)):
                d = f[name]; d.resize(self.nframe + 1, axis=0)
                d[self.nframe] = to_host(arr)
            d = f["t"]; d.resize(self.nframe + 1, axis=0); d[self.nframe] = t
        self.nframe += 1

    def write_restart(self, state, step, t):
        tmp = self.path_restart + ".tmp"
        with h5py.File(tmp, "w") as f:
            for name in ("vort", "vorti", "pres", "presi"):
                f[name] = to_host(getattr(state, name))
            f.attrs["step"] = int(step); f.attrs["t"] = float(t)
        os.replace(tmp, self.path_restart)


def load_restart(path):
    with h5py.File(path, "r") as f:
        arrs = {n: np.asarray(f[n]) for n in ("vort", "vorti", "pres", "presi")}
        step = int(f.attrs["step"]); t = float(f.attrs["t"])
    return arrs, step, t
