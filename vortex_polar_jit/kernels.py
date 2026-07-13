"""kernels.py — POLAR (r, theta) operators for the Beklemishev vortex model.

Field layout is (nr, ntheta): axis 0 = radius r (cell-centred, non-periodic),
axis 1 = poloidal angle theta (periodic, handled spectrally).

Design (see README):
  * theta derivatives are SPECTRAL   (FFT in theta -> multiply by i*m).
  * r derivatives are 4th/2nd-order FINITE DIFFERENCE (here: 2nd-order central).
  * the Laplacian and the Poisson solve are done PER poloidal mode m:
        FFT in theta  ->  for each m a radial operator  A_m  acts on phi_m(r):
            A_m phi_m = phi_m'' + (1/r) phi_m' - (m^2/r^2) phi_m
        `delsq` applies A_m ;  `getphi` applies A_m^{-1}  (both precomputed in
        config.py), so  delsq(getphi(w)) == w  by construction (branch-cut-free,
        no square-mesh m=4 imprint).

This is the ANNULAR / cell-centred scaffold: r in (r_in, r_max], BCs = Neumann at
the inner edge, Dirichlet(phi=0) at the outer edge.  The r=0 pole regularity +
near-axis azimuthal filtering are the main TODO (see README).

Only `akw` (the Poisson bracket) is a plain central-difference form here; replacing
it with an energy/enstrophy-conserving Arakawa-in-polar Jacobian is a TODO.
"""
from __future__ import annotations

from .backend import xp, BACKEND

if BACKEND == "jax":
    import jax as _jax


# ---------------------------------------------------------------------------
def der_theta(a, pars):
    """Spectral d/dtheta along axis 1 (periodic)."""
    ak = xp.fft.rfft(a, axis=1)
    ak = ak * (1j * pars.marr)[None, :]
    return xp.fft.irfft(ak, n=pars.ntheta, axis=1)


def der_r(a, pars):
    """2nd-order central d/dr along axis 0, one-sided at the radial boundaries
    (backend-agnostic: built with slicing + concatenate, no in-place update)."""
    rdr = pars.rdr                                   # 1/(2*dr)
    inner = (a[2:] - a[:-2]) * rdr                   # (nr-2, ntheta)
    top = ((a[1] - a[0]) * (2.0 * rdr))[None]        # forward difference
    bot = ((a[-1] - a[-2]) * (2.0 * rdr))[None]      # backward difference
    return xp.concatenate([top, inner, bot], axis=0)


def delsq(a, pars):
    """Polar Laplacian, applied per poloidal mode as a banded (tridiagonal)
    mat-vec in theta-Fourier space: irfft( A_m @ a_m ), O(M*nr)."""
    ak = xp.fft.rfft(a, axis=1)                       # (nr, M)
    z = xp.zeros((1, ak.shape[1]), dtype=ak.dtype)
    ak_dn = xp.concatenate([z, ak[:-1]], axis=0)      # a_{j-1} (0 below the axis)
    ak_up = xp.concatenate([ak[1:], z], axis=0)       # a_{j+1} (0 = Dirichlet at r_max)
    lap = pars.ds_diag * ak + pars.ds_lower * ak_dn + pars.ds_upper * ak_up
    return xp.fft.irfft(lap, n=pars.ntheta, axis=1)


def getphi(vort, pars):
    """Solve the polar Poisson problem  Laplacian(phi) = vort  for phi.
    FFT in theta -> per-m tridiagonal solve.  JAX uses lax tridiagonal_solve
    (O(M*nr)); the numpy/cupy test backend falls back to the dense inverse."""
    wk = xp.fft.rfft(vort, axis=1)                    # (nr, M)
    if pars.Ainv is not None:                         # dense fallback (test backend)
        phik = xp.einsum("mij,jm->im", pars.Ainv, wk)
    else:                                             # JAX batched tridiagonal solve
        b = wk.T[..., None]                           # (M, nr, 1) complex
        ts = _jax.lax.linalg.tridiagonal_solve
        xr = ts(pars.tri_dl, pars.tri_d, pars.tri_du, b.real)
        xi = ts(pars.tri_dl, pars.tri_d, pars.tri_du, b.imag)
        phik = (xr[..., 0] + 1j * xi[..., 0]).T       # (nr, M)
    return xp.fft.irfft(phik, n=pars.ntheta, axis=1)


def akw(a, b, pars):
    """Poisson bracket {a,b} = (1/r)(d_r a d_theta b - d_theta a d_r b).

    TODO: replace with an energy/enstrophy-conserving Arakawa-in-polar Jacobian
    (this plain central-difference form is fine for early tests but is not
    conservative for long-time turbulence runs)."""
    ar = der_r(a, pars); at = der_theta(a, pars)
    br = der_r(b, pars); bt = der_theta(b, pars)
    return (ar * bt - at * br) * pars.rinv            # rinv = 1/r, shape (nr,1)
