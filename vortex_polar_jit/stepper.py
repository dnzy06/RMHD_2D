"""stepper.py — POLAR Beklemishev RHS + jit/scan trap-leapfrog.

Solves (Eqs. 7-8) on the polar (r,theta) grid, vort = Laplacian(phi):

  (8) pressure :  d_t P    = -{phi,P}    + nu4p Lap(P)    - nu5p(r) P + Q_p
  (7) vorticity:  d_t vort = -{phi,vort} + H(phi - phi_w) + kappa {P,r^2}
                             + nu4 Lap(vort) - nu5(r) vort            [ + U FLR: TODO ]

In polar the curvature drive SIMPLIFIES exactly:
      {P, r^2} = (1/r)(d_r P * d_theta(r^2) - d_theta P * d_r(r^2)) = -2 d_theta P,
so it is computed directly as -2*kappa*der_theta(P) (no bracket, no r^2 field).

phi is recovered from vort each substep by the per-m polar Poisson solve
(kernels.getphi).  The trap-leapfrog structure mirrors the Cartesian code.
"""
from __future__ import annotations

from types import SimpleNamespace

from .backend import jit, scan_lower, HAS_JIT, xp
from .kernels import akw, delsq, der_r, der_theta, getphi
from .state import State


def _kpars(cfg):
    """Namespace of the polar operator arrays the kernels expect."""
    return SimpleNamespace(
        ds_lower=cfg.ds_lower, ds_upper=cfg.ds_upper, ds_diag=cfg.ds_diag,
        tri_dl=cfg.tri_dl, tri_d=cfg.tri_d, tri_du=cfg.tri_du, Ainv=cfg.Ainv,
        marr=cfg.marr, rinv=cfg.rinv,
        rdr=float(cfg.rdr), ntheta=int(cfg.ntheta), nr=int(cfg.nr),
        cos_t=cfg.cos_t, sin_t=cfg.sin_t, axis_filt=cfg.axis_filt,
        clamp_w=cfg.clamp_w,
    )


def _filter_theta(a, kp):
    """Zero out sub-CFL high-m poloidal modes near the axis."""
    return xp.fft.irfft(kp.axis_filt * xp.fft.rfft(a, axis=1), n=kp.ntheta, axis=1)

def make_step(cfg):
    kp = _kpars(cfg)
    dt = float(cfg.dt)
    H = float(cfg.H); kappa = float(cfg.kappa)
    nu4 = float(cfg.nu4); nu4p = float(cfg.nu4p)
    U = float(cfg.U)
    apply_U = abs(U) > 1.0e-12                        # FLR (Fig. 6) terms on/off
    phi_w = cfg.phi_w
    nu5_field = cfg.nu5_field; nu5p_field = cfg.nu5p_field; Qp = cfg.Qp_field
    cos_t = kp.cos_t; sin_t = kp.sin_t; rinv = kp.rinv
    # static FLR source -U*Lap(S_p), S_p ~ Q_p(r); broadcast Q_p to (nr,ntheta) for delsq.
    U_source = ((-U) * delsq(Qp + xp.zeros((kp.nr, kp.ntheta)), kp)) if apply_U else None

    # limiter sponge: rapidly damp pressure/vorticity toward 0 and clamp phi toward the
    # wall value phi_w beyond r_lim (matches the Cartesian limiter behaviour).
    apply_clamp = bool(getattr(cfg, "limiter_phi_clamp", False))
    clamp_w = kp.clamp_w

    def edge_damp(a):
        return a * clamp_w if apply_clamp else a       # -> 0 beyond the limiter

    def solve_phi(vort):
        phi = _filter_theta(getphi(vort, kp), kp)      # Poisson + near-axis filter
        if apply_clamp:
            phi = phi * clamp_w + phi_w * (1.0 - clamp_w)   # phi -> phi_w beyond r_lim
        return phi

    def rhs(phi, vort, pres):
        # Eq. 8 — pressure
        presdot = (-akw(phi, pres, kp)
                   + nu4p * delsq(pres, kp)
                   - nu5p_field * pres
                   + Qp)
        # Eq. 7 — vorticity.  Curvature drive kappa{P,r^2} = -2 kappa d_theta P.
        vortdot = (-akw(phi, vort, kp)
                   + H * (phi - phi_w)
                   - 2.0 * kappa * der_theta(pres, kp)
                   + nu4 * delsq(vort, kp)
                   - nu5_field * vort)
        if apply_U:
            # FLR polarization U * div{grad phi, P}.  The bracket akw is coordinate-
            # invariant; form the CARTESIAN gradient components of phi via the chain
            # rule, bracket each with P, then take the Cartesian divergence (also via
            # the chain rule).  No 3rd derivatives; reduces to the Cartesian code's
            # d_x{phi_x,P}+d_y{phi_y,P} form.  Plus the static source -U*Lap(S_p).
            phir = der_r(phi, kp); phit = der_theta(phi, kp)
            phix = cos_t * phir - sin_t * rinv * phit
            phiy = sin_t * phir + cos_t * rinv * phit
            gx = akw(phix, pres, kp); gy = akw(phiy, pres, kp)
            divg = (cos_t * der_r(gx, kp) - sin_t * rinv * der_theta(gx, kp)
                    + sin_t * der_r(gy, kp) + cos_t * rinv * der_theta(gy, kp))
            vortdot = vortdot + U * divg + U_source
        return presdot, vortdot

    def step(state):
        vort, vorti, pres, presi, phi = state
        pdot, wdot = rhs(phi, vorti, presi)

        # near-axis azimuthal filter on every field update: bounds the m^2/r^2
        # stiffness of the perp-diffusion + 1/r advection at small r.
        pres_new = presi
        presi_p = edge_damp(_filter_theta(0.5 * (pres + presi) + pdot * dt, kp))
        vort_new = vorti
        vorti_p = edge_damp(_filter_theta(0.5 * (vort + vorti) + wdot * dt, kp))
        phi_p = solve_phi(vorti_p)

        pdot_p, wdot_p = rhs(phi_p, vorti_p, presi_p)
        presi_c = edge_damp(_filter_theta(pres_new + pdot_p * dt, kp))
        vorti_c = edge_damp(_filter_theta(vort_new + wdot_p * dt, kp))
        phi_c = solve_phi(vorti_c)

        return State(vort=vort_new, vorti=vorti_c,
                     pres=pres_new, presi=presi_c, phi=phi_c)

    if HAS_JIT:
        step = jit(step)
    return step


def make_run_frame(cfg):
    step = make_step(cfg)
    nts = int(cfg.nts)

    def run_frame(state):
        def body(s, _):
            return step(s), None
        return scan_lower(body, state, nts)

    if HAS_JIT:
        run_frame = jit(run_frame)
    return run_frame
