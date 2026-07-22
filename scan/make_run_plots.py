#!/usr/bin/env python3
"""make_run_plots.py -- given a directory of .toml configs (any mix of cart and
polar, no U/H structure required), find each config's completed run and make
two diagnostic plots per run, plus one comparison plot across all of them:

    plots/<configs_dir_name>_plots/<config_basename>/evolution_montage.png
    plots/<configs_dir_name>_plots/<config_basename>/spectrum_growth.png
    plots/<configs_dir_name>_plots/tau_E_comparison.png   (all runs overlaid)

All plotting math is ported unchanged from vortex_analysis.ipynb / make_scan_plots.py
(sample_ring, mode_amplitudes, fit_growth, compute_tau_E, draw_field). RunData is
coordinate-aware PER RUN (reads each run's own setup.h5 "coord_type" attribute),
so cart and polar configs can be freely mixed in the same input directory.

Run from anywhere -- paths are resolved relative to this script's own location:

    python scan/make_run_plots.py scan/configs_polar_test
    python scan/make_run_plots.py /path/to/some/configs --plots-dir /path/to/out

Requires: numpy, h5py, matplotlib, tomllib (stdlib, Python 3.11+).
"""
from __future__ import annotations

import os
import glob
import argparse
import tomllib

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless -- no display on the compute/login node
import matplotlib.pyplot as plt
import h5py

# ---------------------------------------------------------------------------
# Paths -- resolved relative to this script's own location, so it works
# regardless of which directory you invoke it from.
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Mode-spectrum / growth-rate settings (mirrors vortex_analysis.ipynb defaults)
# ---------------------------------------------------------------------------
FIELD = "phi"
LIMITER_RHO = 1.5
N_SAMPLE_POINTS = 10
RHO = 0.3
MMAX = 8
NTHETA = 256
LO, HI = 0.0, 0.9
N_MONTAGE = 6
AVERAGE = True
N_RHO_POINTS = 60 


# ---------------------------------------------------------------------------
# Helpers ported from vortex_analysis.ipynb (unchanged math)
# ---------------------------------------------------------------------------
def sample_ring(field2d, x, y, rho, ntheta):
    theta = np.arange(ntheta) * (2.0 * np.pi / ntheta)

    rho_in = rho
    rho_arr = np.atleast_1d(rho).astype(float)

    rho_col = rho_arr[:, None]
    xs = rho_col * np.cos(theta)
    ys = rho_col * np.sin(theta)

    nx = x.size; ny = y.size
    dx = x[1] - x[0]; dy = y[1] - y[0]

    fi = (xs - x[0]) / dx; fj = (ys - y[0]) / dy
    i0 = np.floor(fi).astype(int); j0 = np.floor(fj).astype(int)
    ti = fi - i0; tj = fj - j0

    i0m = i0 % nx; i1m = (i0 + 1) % nx
    j0m = j0 % ny; j1m = (j0 + 1) % ny

    f = field2d
    result = (f[j0m, i0m] * (1 - ti) * (1 - tj) + f[j0m, i1m] * ti * (1 - tj)
              + f[j1m, i0m] * (1 - ti) * tj + f[j1m, i1m] * ti * tj)

    if np.ndim(rho_in) == 0:
        return theta, result[0]      # shape (ntheta,) for scalar rho
    return theta, result             # shape (N, ntheta) for array rho


def mode_amplitudes(ring, mmax):
    N = ring.size
    amp = np.abs(np.fft.rfft(ring) / N)
    amp[1:] *= 2.0
    return amp[: mmax + 1]

def der_t(a, t):
    """2nd-order central d/dt along axis 0, one-sided at the boundaries
    (backend-agnostic: built with slicing + concatenate, no in-place update)."""
    dt = t[1] - t[0]
    tdt = 1 / (2 * dt)
    inner = (a[2:] - a[:-2]) * tdt                   # (nf-2,)
    top = ((a[1] - a[0]) * (2.0 * tdt))[None]        # forward difference
    bot = ((a[-1] - a[-2]) * (2.0 * tdt))[None]      # backward difference
    return np.concatenate([top, inner, bot], axis=0)

def plot_instantaneous_gamma(run, spec, field, out_path):
    """Instantaneous growth rate d/dt log(|a_m|) vs time, for m=1..MMAX
    (m=0 excluded -- axisymmetric mode isn't part of the growth spectrum).
    Uses semilogy -- negative gamma (decay/saturation phases) will be dropped
    from the plot rather than shown, since log-scale can't represent them."""
    t = run.t; A = spec["A"]
    mmax = A.shape[1] - 1
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for m in range(1, mmax + 1):
        mask = A[:, m] > 0        # skip zero-amplitude frames (log(0) undefined)
        gamma_t = der_t(np.log(A[mask, m]), t[mask])
        ax.semilogy(t[mask], gamma_t, lw=1, alpha=0.7, label=f"m={m}")
    ax.set_xlabel("t  (t_bar)")
    ax.set_ylabel(r"instantaneous $\gamma_m$  [1/$\bar t$]")
    ax.set_title(f"instantaneous growth rate ({field}, {_rho_label(spec)})")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)


def fit_growth(t, a, lo=0.0, hi=0.9, wmin=5):
    t = np.asarray(t, float); a = np.asarray(a, float)
    none = (np.nan, np.nan, np.nan, np.nan, np.zeros_like(a, bool))
    good = np.isfinite(a) & (a > 0)
    if good.sum() < wmin:
        return none
    amax = a[good].max(); imax = int(np.argmax(a))
    cand = good & (a >= lo * amax) & (a <= hi * amax)
    cand[imax + 1:] = False
    ii = np.where(cand)[0]
    if ii.size < wmin:
        ii = np.where(good[:imax + 1])[0]
    if ii.size < wmin:
        return none
    ll = np.full_like(a, -np.inf); ll[good] = np.log(a[good])
    best = (-np.inf, np.nan, np.nan, np.nan, np.nan, None)
    for p in range(ii.size - wmin + 1):
        for q in range(p + wmin - 1, ii.size):
            seg = ii[p:q + 1]; ts = t[seg]; ls = ll[seg]
            g, b = np.polyfit(ts, ls, 1)
            if g <= 0:
                continue
            pred = g * ts + b
            r2 = 1.0 - np.sum((ls - pred) ** 2) / max(np.sum((ls - ls.mean()) ** 2), 1e-30)
            score = r2 * np.sqrt(seg.size)
            if score > best[0]:
                best = (score, g, ts[0], ts[-1], r2, seg)
    if best[5] is None:
        return none
    _, g, t0, t1, r2, seg = best
    mask = np.zeros_like(a, bool); mask[seg] = True
    return g, t0, t1, r2, mask


# ---------------------------------------------------------------------------
# Per-run data loading -- coordinate-aware PER RUN, via each run's own
# setup.h5 "coord_type" attribute. Works for cart or polar runs interchangeably.
# ---------------------------------------------------------------------------
class RunData:
    def __init__(self, rundir):
        self.rundir = rundir
        self._r2 = None
        self.X2 = None; self.Y2 = None
        with h5py.File(os.path.join(rundir, "setup.h5"), "r") as s:
            attrs = dict(s.attrs)
            self.coord = attrs.get("coord_type", "cart")
            self.t_bar = float(attrs.get("t_bar", 1.0))
            self.limiter_radius = attrs.get("limiter_radius")
            if self.coord == "polar":
                self.r = np.asarray(s["r"], float); self.theta = np.asarray(s["theta"], float)
                self.dA_full = (self.r * (self.r[1] - self.r[0]) * (self.theta[1] - self.theta[0]))[:, None]
                self.rmax_dom = float(self.r.max())
                # Cartesian mesh for plotting (written by output_hdf5.py: x=r*cos(theta),
                # y=r*sin(theta), shape (nr,ntheta)) -- needed by draw_field's pcolormesh.
                self.X2 = np.asarray(s["x"], float); self.Y2 = np.asarray(s["y"], float)
            else:
                self.x = np.asarray(s["x"], float); self.y = np.asarray(s["y"], float)
                self.dA_full = (self.x[1] - self.x[0]) * (self.y[1] - self.y[0])
                self.rmax_dom = float(np.hypot(max(abs(self.x[0]), abs(self.x[-1])),
                                               max(abs(self.y[0]), abs(self.y[-1]))))
                if "r2" in s:
                    self._r2 = np.asarray(s["r2"])

        self.fh = h5py.File(os.path.join(rundir, "fields.h5"), "r")
        self.t = np.asarray(self.fh["t"], float)
        self.nf = self.fh["phi"].shape[0]

    def close(self):
        self.fh.close()

    def ring_at(self, field2d, rho):
        if self.coord == "polar":
            r = self.r; nr = r.size
            scalar_input = np.ndim(rho) == 0
            rho_arr = np.atleast_1d(rho).astype(float)
            j = np.clip(np.searchsorted(r, rho_arr), 1, nr - 1)
            w = (rho_arr - r[j - 1]) / (r[j] - r[j - 1])
            result = (1 - w)[:, None] * field2d[j - 1] + w[:, None] * field2d[j]
            return result[0] if scalar_input else result   # (ntheta,) or (N, ntheta)
        _, ring = sample_ring(field2d, self.x, self.y, rho, NTHETA)
        return ring

    def draw_field(self, ax, F, cmap, vmin, vmax):
        """imshow (cart) or closed-disk pcolormesh (polar); returns the mappable."""
        if self.coord == "polar":
            Xc = np.concatenate([self.X2, self.X2[:, :1]], axis=1)   # close the theta seam
            Yc = np.concatenate([self.Y2, self.Y2[:, :1]], axis=1)
            Fc = np.concatenate([F, F[:, :1]], axis=1)
            im = ax.pcolormesh(Xc, Yc, Fc, cmap=cmap, vmin=vmin, vmax=vmax, shading="gouraud")
        else:
            im = ax.imshow(F, origin="lower",
                           extent=[self.x.min(), self.x.max(), self.y.min(), self.y.max()],
                           cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_aspect("equal")
        return im

    def dA(self, core_only=False):
        if not core_only or self.limiter_radius is None:
            return self.dA_full
        if self.coord == "polar":
            return self.dA_full * (self.r < self.limiter_radius)[:, None]
        if self._r2 is None:
            return self.dA_full
        return self.dA_full * (self._r2 < self.limiter_radius ** 2)


def compute_mode_spectrum(run, field=FIELD, rho=RHO, limiter_rho=LIMITER_RHO,
                           n_sampled_points=N_SAMPLE_POINTS, mmax=MMAX, average=AVERAGE):
    if average:
        sample_rhos = np.linspace(0.1, limiter_rho - 0.1, n_sampled_points)
        rho_eff = None
    else:
        rho_eff = rho if rho <= run.rmax_dom else 0.98 * run.rmax_dom

    A = np.zeros((run.nf, mmax + 1))
    for k in range(run.nf):
        if average:
            rings = run.ring_at(np.asarray(run.fh[field][k]), sample_rhos)
            ring = np.average(rings, axis=0)
        else:
            ring = run.ring_at(np.asarray(run.fh[field][k]), rho_eff)
        A[k] = mode_amplitudes(ring, mmax)
    A_tot = np.sqrt(np.sum(A[:, 1:] ** 2, axis=1))
    g_tot, t0, t1, r2, band = fit_growth(run.t, A_tot, LO, HI)
    return dict(A=A, A_tot=A_tot, g_tot=g_tot, t0=t0, t1=t1, r2=r2, band=band,
                rho=rho_eff, averaged=average)


def compute_tau_E(run, core_only=True):
    dA = run.dA(core_only=core_only)
    W = np.array([np.sum(np.asarray(run.fh["pres"][k]) * dA) for k in range(run.nf)])
    with np.errstate(divide="ignore", invalid="ignore"):
        L = -np.gradient(W, run.t)
        tau = np.where(L > 0, W / L, np.nan)
    return run.t, W, tau


def _rho_label(spec):
    return "rho-averaged" if spec.get("averaged") else f"rho={spec['rho']:.2f} L0"


# ---------------------------------------------------------------------------
# Plotting (each function saves one PNG and closes its figure)
# ---------------------------------------------------------------------------
def plot_spectrum_growth(run, spec, field, out_path):
    t = run.t; A = spec["A"]; A_tot = spec["A_tot"]
    g_tot, t0, t1, band = spec["g_tot"], spec["t0"], spec["t1"], spec["band"]
    mmax = A.shape[1] - 1
    fig, (axS, axG) = plt.subplots(1, 2, figsize=(13, 5.2))
    ms = np.arange(mmax + 1)
    axS.bar(ms[1:], A[-1, 1:], color="steelblue")
    axS.set_xlabel("poloidal mode number m"); axS.set_ylabel(f"|a_m|  ({field})")
    axS.set_title(f"poloidal spectrum ({_rho_label(spec)}) (t={t[-1]:.3f})")
    for m in range(1, mmax + 1):
        axG.semilogy(t, A[:, m], lw=1, alpha=0.7, label=f"m={m}")
    axG.semilogy(t, A_tot, "k-", lw=2, label="non-axisym RMS")
    if np.isfinite(g_tot):
        tf = t[band]
        axG.semilogy(tf, A_tot[band][0] * np.exp(g_tot * (tf - tf[0])), "r--", lw=2,
                     label=f"fit gamma={g_tot:.3f}/tbar")
        axG.axvspan(t0, t1, color="red", alpha=0.08)
    axG.set_xlabel("t  (t_bar)"); axG.set_ylabel(f"|a_m|  ({field})")
    axG.set_title("mode growth + linear-phase fit"); axG.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)


def plot_evolution_montage(run, coord_label, out_path, n=N_MONTAGE):
    nf = run.nf; t = run.t
    idxs = np.linspace(0, nf - 1, min(n, nf)).astype(int)
    fig, axes = plt.subplots(2, len(idxs), figsize=(2.6 * len(idxs), 4.2),
                         constrained_layout=True)
    if len(idxs) == 1:
        axes = axes.reshape(2, 1)
    for c, k in enumerate(idxs):
        P = np.asarray(run.fh["pres"][k]); V = np.asarray(run.fh["phi"][k])
        vm = float(np.abs(V).max()) or 1.0
        pmn, pmx = float(P.min()), float(P.max())
        if pmx <= pmn:
            pmx = pmn + 1e-30
        imP = run.draw_field(axes[0, c], P, "inferno", pmn, pmx)
        imV = run.draw_field(axes[1, c], V, "RdBu_r", -vm, vm)
        axes[0, c].set_title(f"t={t[k]:.3f}", fontsize=9)
        fig.colorbar(imP, ax=axes[0, c], shrink=0.85)   # per-panel -- scale is
        fig.colorbar(imV, ax=axes[1, c], shrink=0.85)   # per-frame adaptive, so a
                                                          # shared colorbar would mislead
        for rr in (0, 1):
            axes[rr, c].tick_params(axis="both", labelsize=5)
            axes[rr, c].locator_params(axis="both", nbins=3)  # keep tiny ticks uncluttered
    axes[0, 0].set_ylabel("pressure P"); axes[1, 0].set_ylabel("potential phi")
    fig.suptitle(f"evolution montage ({coord_label}, per-frame adaptive colour)")
    fig.savefig(out_path, dpi=130); plt.close(fig)


def plot_tau_E_comparison(tau_results, out_path):
    """tau_results: list of (label, t, tau_E) -- one entry per config/run."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = plt.cm.viridis(np.linspace(0, 1, max(len(tau_results), 1)))
    for (label, t, tau), c in zip(tau_results, colors):
        ax.plot(t, tau, lw=1.8, label=label, color=c)
    ax.set_xlabel("t (t_bar)"); ax.set_ylabel("tau_E  (t_bar)")
    ax.set_title("energy confinement time -- run comparison")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)

def plot_gamma_m1_comparison(gamma_m1_results, out_path):
    """gamma_m1_results: list of (run_name, t, gamma_m1_t) -- one entry per run,
    m=1 instantaneous growth rate overlaid across all runs. Labeled the same way
    as tau_E_comparison: by each run's [output] outdir basename, not its .toml
    config filename."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = plt.cm.viridis(np.linspace(0, 1, max(len(gamma_m1_results), 1)))
    for (run_name, t, gamma_t), c in zip(gamma_m1_results, colors):
        ax.semilogy(t, gamma_t, lw=1.5, label=run_name, color=c)
    ax.set_xlabel("t  (t_bar)")
    ax.set_ylabel(r"instantaneous $\gamma_{m=1}$  [1/$\bar t$]")
    ax.set_title("m=1 instantaneous growth rate -- run comparison")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)


def find_rundir(outdir):
    """outdir/<jobid>/setup.h5 -- pick the most-recently-modified match."""
    candidates = glob.glob(os.path.join(outdir, "*", "setup.h5"))
    if not candidates:
        return None
    candidates.sort(key=os.path.getmtime)
    return os.path.dirname(candidates[-1])

def compute_radial_mode_spectrum(run, field=FIELD, limiter_rho=LIMITER_RHO,
                                  n_rho_points=N_RHO_POINTS, mmax=MMAX):
    """Radius-resolved poloidal spectrum for every frame: A_r_all[frame, i_rho, m].
    Same [0.1, limiter_rho - 0.1] bounds convention as the averaged spectrum, just
    keeping every rho as its own row instead of averaging them together."""
    rho_grid = np.linspace(0.1, limiter_rho - 0.1, n_rho_points)
    A_r_all = np.zeros((run.nf, n_rho_points, mmax + 1))
    for k in range(run.nf):
        field2d = np.asarray(run.fh[field][k])
        rings = run.ring_at(field2d, rho_grid)          # (n_rho_points, ntheta)
        for i in range(n_rho_points):
            A_r_all[k, i] = mode_amplitudes(rings[i], mmax)
    return dict(rho_grid=rho_grid, A_r_all=A_r_all, mmax=mmax)


def plot_radial_spectrum_video(run, radial_spec, field, out_path):
    """Interactive rho-vs-m surface with a frame slider + play/pause, saved as
    standalone HTML (open in any browser -- no server needed)."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        print(f"  [skip] {os.path.basename(out_path)}: plotly not installed "
              f"(pip install plotly)")
        return

    t = run.t
    rho_grid = radial_spec["rho_grid"]
    A_r_all = radial_spec["A_r_all"]           # (nf, n_rho, mmax+1)
    m_axis = np.arange(radial_spec["mmax"] + 1)
    zmax = A_r_all.max() if A_r_all.size else 1.0
    nf = A_r_all.shape[0]

    k0 = 0
    fig = go.Figure(
        data=[go.Surface(
            x=rho_grid, y=m_axis, z=A_r_all[k0].T,      # z[i,j] ~ y[i], x[j] -> transpose (rho, m) -> (m, rho)
            colorscale="Viridis", cmin=0, cmax=zmax,
            colorbar=dict(title=f"|a_m|  ({field})"),
            hovertemplate="rho=%{x:.3f}<br>m=%{y}<br>|a_m|=%{z:.3e}<extra></extra>",
        )],
        frames=[
            go.Frame(
                data=[go.Surface(x=rho_grid, y=m_axis, z=A_r_all[k].T,
                                  colorscale="Viridis", cmin=0, cmax=zmax)],
                name=str(k),
            )
            for k in range(nf)
        ],
    )
    fig.update_layout(
        title=f"radius-resolved spectrum ({field}, {run.coord})",
        scene=dict(xaxis_title="rho", yaxis_title="m",
                   zaxis_title=f"|a_m|  ({field})", zaxis=dict(range=[0, zmax])),
        width=850, height=700,
        updatemenus=[dict(
            type="buttons", showactive=False, y=0, x=0.05, xanchor="left", yanchor="top",
            buttons=[
                dict(label="Play", method="animate",
                     args=[None, dict(frame=dict(duration=80, redraw=True), fromcurrent=True)]),
                dict(label="Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]),
            ],
        )],
        sliders=[dict(
            active=k0, currentvalue=dict(prefix="frame: "),
            steps=[
                dict(method="animate",
                     args=[[str(k)], dict(mode="immediate", frame=dict(duration=0, redraw=True))],
                     label=f"{k} (t={t[k]:.2f})")
                for k in range(nf)
            ],
        )],
    )
    fig.write_html(out_path, include_plotlyjs="cdn")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("configs_dir",
                     help="directory of .toml configs to plot (any mix of cart/polar). "
                          "Each config's run is located from its own [output] outdir.")
    ap.add_argument("--plots-dir", default=None,
                     help="override the output directory -- default is "
                          "RMHD_2D/plots/<configs_dir name>_plots")
    args = ap.parse_args()

    configs_dir = args.configs_dir
    dir_name = os.path.basename(os.path.normpath(configs_dir))
    plots_dir = args.plots_dir or os.path.join(SCRIPT_DIR, "..", "plots", f"{dir_name}_plots")

    toml_paths = sorted(glob.glob(os.path.join(configs_dir, "*.toml")))
    if not toml_paths:
        raise SystemExit(f"No .toml files found in {configs_dir}")

    os.makedirs(plots_dir, exist_ok=True)
    tau_results = []
    gamma_m1_results = []
    used_names = set()

    for tp in toml_paths:
        cfg_name = os.path.splitext(os.path.basename(tp))[0]
        with open(tp, "rb") as f:
            cfg = tomllib.load(f)
        outdir = cfg["output"]["outdir"]
        rundir = find_rundir(outdir)
        if rundir is None:
            print(f"[skip] {cfg_name}: no completed run found under {outdir}")
            continue

        # Label everything by the run's own saved name -- the basename of its
        # [output] outdir (e.g. outdir=".../polar_200f_Um5_H5_smoothbiased" with
        # rundir=".../polar_200f_Um5_H5_smoothbiased/<jobid>") -- NOT the .toml
        # filename, since those can be arbitrary/short while outdir is usually
        # the descriptive name you actually gave the run.
        run_name = os.path.basename(os.path.normpath(outdir))
        base_run_name = run_name
        dedup = 1
        while run_name in used_names:          # guard against two configs
            dedup += 1                          # saving to same-named outdirs
            run_name = f"{base_run_name}_{dedup}"
        used_names.add(run_name)

        print(f"[{cfg_name}] loading {rundir}  ->  '{run_name}'")
        run_dir_out = os.path.join(plots_dir, run_name)
        os.makedirs(run_dir_out, exist_ok=True)

        run = RunData(rundir)
        run = RunData(rundir)
        try:
            spec = compute_mode_spectrum(run)
            plot_spectrum_growth(run, spec, FIELD, os.path.join(run_dir_out, "spectrum_growth.png"))
            plot_evolution_montage(run, run.coord, os.path.join(run_dir_out, "evolution_montage.png"))
            plot_instantaneous_gamma(run, spec, FIELD, os.path.join(run_dir_out, "instantaneous_gamma.png"))

            mask_m1 = spec["A"][:, 1] > 0
            gamma_m1_t = der_t(np.log(spec["A"][mask_m1, 1]), run.t[mask_m1])
            gamma_m1_results.append((run_name, run.t[mask_m1], gamma_m1_t))

            radial_spec = compute_radial_mode_spectrum(run)
            plot_radial_spectrum_video(run, radial_spec, FIELD,
                                        os.path.join(run_dir_out, "radial_spectrum_3d.html"))

            t, W, tau = compute_tau_E(run, core_only=True)
            tau_results.append((run_name, t, tau))
        except Exception as e:
            print(f"  [error] {run_name} (config {cfg_name}): {e}")
        finally:
            run.close()

    if tau_results:
        plot_tau_E_comparison(tau_results, os.path.join(plots_dir, "tau_E_comparison.png"))
    else:
        print("No completed runs found for any config -- no comparison plot written.")

    if gamma_m1_results:
        plot_gamma_m1_comparison(gamma_m1_results, os.path.join(plots_dir, "gamma_m1_comparison.png"))
    else:
        print("No completed runs found for any config -- no gamma_m1 comparison plot written.")

    print(f"\nDone. All plots under {plots_dir}")


if __name__ == "__main__":
    main()