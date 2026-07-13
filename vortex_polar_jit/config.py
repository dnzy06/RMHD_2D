"""config.py — TOML config, polar grid, per-m Poisson operators, and static radial
profiles for the POLAR Beklemishev vortex model (Eqs. 7-8), annular scaffold.

Grid is (nr, ntheta): r cell-centred on (r_in, r_max], theta periodic.  The
per-poloidal-mode radial operator  A_m phi = phi'' + (1/r)phi' - (m^2/r^2)phi  is
assembled here (2nd-order FD, Neumann inner / Dirichlet-0 outer) together with its
inverse `Ainv`; kernels.py applies A (delsq) and Ainv (getphi).

Shares the normalization + unit conventions of the Cartesian code (see
../vortex_cart_jit and ../Vortex/vortex_normalization.pdf).  Naming: `phi` =
potential, `vort` = Laplacian(phi) (evolved), `pres` = P.
"""
from __future__ import annotations

import tomllib
import numpy as np

from .backend import asarray, BACKEND
from scipy.interpolate import CubicSpline


class Config:
    def __init__(self, d: dict):
        grd = d.get("grid", d.get("domain", {})); phy = d.get("physics", {})
        bias = d.get("bias", {}); ini = d.get("initial", {})
        tim = d.get("time", {}); out = d.get("output", {}); rst = d.get("restart", {})

        # ---- polar grid / resolution ----
        self.nr = int(grd.get("nr", 256)); self.ntheta = int(grd.get("ntheta", 256))
        self.r_max = float(grd.get("r_max", 1.0))
        self.r_in = float(grd.get("r_in", 0.0))          # 0 -> cell-centred full disk

        # ---- Beklemishev physics scalars ----
        self.U = float(phy.get("U", 0.0))
        self.H = float(phy.get("H", 10.0))
        self.kappa = float(phy.get("kappa", 1.0))
        self.nu4 = float(phy.get("nu4", 0.002)); self.nu5 = float(phy.get("nu5", 0.02))
        self.nu4p = float(phy.get("nu4p", 0.002)); self.nu5p = float(phy.get("nu5p", 0.02))
        self.limiter_radius = float(phy.get("limiter_radius", 0.9))
        self.limiter_factor = float(phy.get("limiter_factor", 20.0))
        # limiter sponge: beyond r_lim, rapidly damp pressure & vorticity toward 0 and
        # clamp the potential toward the wall value phi_w (models a fast-loss / conducting
        # limiter edge).  Analogous to the Cartesian limiter_phi_clamp.
        self.limiter_phi_clamp = bool(phy.get("limiter_phi_clamp", True))
        self.limiter_clamp_width = float(phy.get("limiter_clamp_width", 0.05))

        # ---- input-unit conventions (physical -> normalized) ----
        self.length_units = str(grd.get("length_units", "L0")).lower()
        self.bias_units = str(bias.get("units", "volts")).lower()
        self.interpolation = str(bias.get("interpolation", "smooth")).lower()
        self.pres_units = str(ini.get("pres_units", "normalized")).lower()
        self._len_phys = self.length_units in ("m", "meter", "meters", "physical")
        self._volt_phys = self.bias_units in ("volts", "v", "physical")
        self._pres_phys = self.pres_units in ("pa", "pascal", "physical")

        # ---- time / output / restart ----
        self.dt = float(tim.get("dt", 4.0e-5)); self.nts = int(tim.get("nts", 2000))
        self.nframes = int(tim.get("nframes", 50)); self.nfdump = int(tim.get("nfdump", 10))
        self.outdir = str(out.get("outdir", "./out"))
        self.nrst = int(rst.get("nrst", 0))
        self.restartdir = str(rst.get("restartdir", self.outdir))
        self.restart_file = str(rst.get("restart_file", "restart.h5"))

        self._bias = bias; self._ini = ini
        self._normalization(d.get("normalization", {}))
        self._apply_units()
        self._derive()
        self._build_profiles()

    # -----------------------------------------------------------------
    def _normalization(self, nrm):
        """Physical reference bases (identical convention to the Cartesian code)."""
        e = 1.602176634e-19; mp = 1.67262192e-27
        self.B0 = float(nrm.get("B0", 1.0)); self.n0 = float(nrm.get("n0", 1.0e19))
        self.T0 = float(nrm.get("T0", 100.0)); self.Ai = float(nrm.get("Ai", 1.0))
        self.Zi = float(nrm.get("Zi", 1.0)); L0 = float(nrm.get("L0", 0.0))
        mi = self.Ai * mp
        self.Omega_i = self.Zi * e * self.B0 / mi
        self.cs = np.sqrt(self.Zi * self.T0 * e / mi)
        self.rho_s = self.cs / self.Omega_i
        self.L0 = L0 if L0 > 0.0 else self.rho_s
        self.phi_bar = self.T0; self.P_bar = self.n0 * self.T0 * e
        self.v_bar = self.phi_bar / (self.B0 * self.L0); self.t_bar = self.L0 / self.v_bar

    def _apply_units(self):
        self._Lfac = self.L0 if self._len_phys else 1.0
        self._Vfac = self.phi_bar if self._volt_phys else 1.0
        self._Pfac = self.P_bar if self._pres_phys else 1.0
        if self._len_phys:
            self.r_max /= self._Lfac; self.r_in /= self._Lfac
            self.limiter_radius /= self._Lfac

    def print_normalization(self):
        print("*** Normalization bases (physical <-> normalized) ***")
        print(f"    B0={self.B0:.4g} T  n0={self.n0:.3e} /m^3  T0={self.T0:.4g} eV "
              f"(Ai={self.Ai:g}, Zi={self.Zi:g})")
        print(f"    rho_s={self.rho_s:.4e} m  L0={self.L0:.4e} m  t_bar={self.t_bar:.4e} s")
        print(f"    phi_bar={self.phi_bar:.4g} V  P_bar={self.P_bar:.4e} Pa")
        print(f"    input units: lengths={'m/L0' if self._len_phys else 'L0'}, "
              f"ring_volts={'V/phi_bar' if self._volt_phys else 'normalized'}, "
              f"pres={'Pa/P_bar' if self._pres_phys else 'normalized'}")

    # -----------------------------------------------------------------
    def _derive(self):
        """Polar grid + per-m radial Poisson operator A_m and its inverse Ainv."""
        p = self
        nr, nth = p.nr, p.ntheta
        p.dr = (p.r_max - p.r_in) / nr
        p.r = p.r_in + (np.arange(nr) + 0.5) * p.dr        # cell-centred (nr,)
        p.theta = np.arange(nth) * (2.0 * np.pi / nth)
        p.M = nth // 2 + 1
        marr = np.arange(p.M)
        p.marr = asarray(marr.astype(float))               # for spectral d/dtheta
        p.rinv = asarray((1.0 / p.r)[:, None])             # (nr,1)
        p.rdr = 1.0 / (2.0 * p.dr)                         # central d/dr prefactor

        # radial stencil:  A_m phi = phi'' + (1/r)phi' - (m^2/r^2)phi
        r = p.r; dr2 = p.dr * p.dr
        lower = 1.0 / dr2 - 1.0 / (2.0 * r * p.dr)          # sub-diagonal coeff (nr,)
        upper = 1.0 / dr2 + 1.0 / (2.0 * r * p.dr)          # super-diagonal coeff (nr,)
        diag_r = np.full(nr, -2.0 / dr2)
        diagm = diag_r[None, :] - (marr[:, None] ** 2) / (r[None, :] ** 2)   # (M,nr)
        # inner BC.  Full disk (r_in=0, cell-centred axis): pole regularity
        # phi_{-1,m} = (-1)^m phi_{0,m}  (even m symmetric across axis, odd m -> 0).
        # Annulus (r_in>0): Neumann wall  phi_{-1} = phi_0.
        inner_sign = ((-1.0) ** marr) if p.r_in <= 0.0 else np.ones(p.M)
        diagm[:, 0] = diagm[:, 0] + inner_sign * lower[0]

        # Store the three diagonals of A_m instead of the dense matrix/inverse:
        #   delsq  -> banded mat-vec (kernels.delsq), O(M*nr) memory & work
        #   getphi -> jax.lax.linalg.tridiagonal_solve, O(M*nr)  (was dense O(M*nr^2)).
        # Layouts: ds_* in (nr,M) for the banded mat-vec; tri_* in (M,nr) for the solve
        # (dl[.,0]=0 and du[.,-1]=0 are the out-of-band entries the BCs already drop).
        p.ds_lower = asarray(lower[:, None])                # (nr,1) sub-diagonal
        p.ds_upper = asarray(upper[:, None])                # (nr,1) super-diagonal
        p.ds_diag = asarray(diagm.T)                        # (nr,M) diagonal per mode
        dl = np.zeros((p.M, nr)); dl[:, 1:] = lower[1:][None, :]
        du = np.zeros((p.M, nr)); du[:, :-1] = upper[:-1][None, :]
        p.tri_dl = asarray(dl); p.tri_du = asarray(du); p.tri_d = asarray(diagm)
        # Dense inverse only as a fallback for the non-JAX (numpy/cupy) test backend,
        # which has no batched tridiagonal solver.  JAX (production/GPU) skips it.
        if BACKEND == "jax":
            p.Ainv = None
        else:
            A = np.zeros((p.M, nr, nr)); idx = np.arange(nr)
            A[:, idx, idx] = diagm
            A[:, idx[1:], idx[:-1]] = lower[1:][None, :]
            A[:, idx[:-1], idx[1:]] = upper[:-1][None, :]
            p.Ainv = asarray(np.linalg.inv(A))

        # cos/sin(theta) for the FLR Cartesian-gradient chain rule (1,ntheta)
        p.cos_t = asarray(np.cos(p.theta)[None, :])
        p.sin_t = asarray(np.sin(p.theta)[None, :])
        # near-axis azimuthal filter: at radius r keep only m <= r/dr (an azimuthal
        # cell r*dtheta must resolve the mode) -> removes the tiny-CFL high-m modes
        # near the axis.  (nr, M) mask, ~all-ones except in the first few rings.
        mmax_r = np.floor(p.r / p.dr)
        p.axis_filt = asarray((marr[None, :] <= mmax_r[:, None]).astype(float))

    # -----------------------------------------------------------------
    def _build_profiles(self):
        """Static radial profiles broadcast over theta (axisymmetric), plus the
        seeded initial pressure (r-Gaussian + (r,theta) perturbation)."""
        p = self
        r = p.r; nth = p.ntheta

        # --- 11-ring wall-bias potential phi_w(r) ---
        volts = np.asarray(self._bias.get("ring_volts", [0.0] * 11), float) / self._Vfac
        radii = np.asarray(self._bias.get(
            "ring_radii", list(np.linspace(0.1, p.r_max, len(volts)))), float) / self._Lfac
        idx = np.clip(np.searchsorted(radii, r, side="left"), 0, volts.size - 1)

        if (self.interpolation == "smooth"):
            p.phi_w = CubicSpline(radii, volts)(r)[:, None]
        else:
            p.phi_w = asarray(volts[idx][:, None])

        p.n_rings = int(volts.size)

        # --- limiter-enhanced axial loss (r-dependent) ---
        mult = np.where(r > p.limiter_radius, p.limiter_factor, 1.0)
        p.nu5_field = asarray((p.nu5 * mult)[:, None])
        p.nu5p_field = asarray((p.nu5p * mult)[:, None])

        # --- limiter sponge mask: exactly 1 for r <= r_lim (plasma preserved), then a
        #     Gaussian roll-off beyond (applied every step, so it compounds to ~0 fast).
        #     One-sided so it does NOT erode the plasma inside the limiter. ---
        wd = max(p.limiter_clamp_width, 1.0e-6)
        _cw = np.where(r <= p.limiter_radius, 1.0, np.exp(-((r - p.limiter_radius) / wd) ** 2))
        p.clamp_w = asarray(_cw[:, None])

        # --- pressure source Q_p ---
        Qp = self._ini.get("Qp", 0.0)
        if isinstance(Qp, str) and Qp == "gaussian":
            w = float(self._ini.get("Qp_width", self._ini.get("pres_width", 1.0))) / self._Lfac
            a = float(self._ini.get("Qp_amp", 0.0))
            p.Qp_field = asarray((a * np.exp(-r * r / (w * w)))[:, None])
        else:
            p.Qp_field = asarray(float(Qp))

        # --- initial pressure profile (r-Gaussian/parabolic) + seed perturbation ---
        ini = self._ini
        amp = float(ini.get("pres_amp", 1.0)) / self._Pfac
        wid = float(ini.get("pres_width", 1.0)) / self._Lfac
        if str(ini.get("pres_type", "gaussian")) == "parabolic":
            base_r = amp * np.clip(1.0 - r * r / (wid * wid), 0.0, None)
        else:
            base_r = amp * np.exp(-r * r / (wid * wid))
        base = np.repeat(base_r[:, None], nth, axis=1)      # (nr,ntheta)
        perturb = float(ini.get("perturb_amp", 1.0e-3))
        rng = np.random.default_rng(int(ini.get("seed", 12345)))
        base = base * (1.0 + perturb * rng.standard_normal(base.shape))
        p.pres0 = asarray(base)


def load_config(path: str) -> Config:
    with open(path, "rb") as fh:
        return Config(tomllib.load(fh))
