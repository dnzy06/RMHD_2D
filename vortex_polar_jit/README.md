# `vortex_polar_jit` — polar (r, θ) Beklemishev vortex solver (scaffold)

Polar sibling of the validated Cartesian code [`vortex_cart_jit`](../vortex_cart_jit).
Same physics (Beklemishev Eqs. 7–8), same normalization, same trap-leapfrog +
jit/scan time integration and HDF5 output — only the **coordinates and operators**
differ. Motivation: an axisymmetric mirror is naturally polar, so this removes the
square-Cartesian-mesh **m=4 artifact** and gives clean poloidal-mode spectra.

## Discretization
- Grid `(nr, nθ)`: `r` cell-centred on `(r_in, r_max]` (non-periodic), `θ` periodic.
- **θ: spectral** (FFT → `i·m`), **r: finite difference** (2nd-order central).
- Laplacian & Poisson are done **per poloidal mode m**: FFT in θ decouples each `m`
  into a radial operator `A_m φ = φ'' + (1/r)φ' − (m²/r²)φ`. `config.py` precomputes
  `A` (forward, used by `delsq`) and `Ainv` (used by `getphi`), so
  `delsq(getphi(w)) == w` by construction — branch-cut-free, no square-mesh imprint.
- **Curvature simplifies exactly:** `κ{P,r²} = −2κ ∂_θP` (no bracket, no r² field).

## Run
```bash
cd vortex
JAX_PLATFORMS=cpu  python -m vortex_polar_jit vortex_polar_jit/input_polar.toml   # CPU test
JAX_PLATFORMS=cuda python -m vortex_polar_jit vortex_polar_jit/input_polar.toml   # A100
```
Output `setup.h5` carries `coord_type="polar"`, the `(r,θ)` axes, and reconstructed
`(x,y)` for plotting.

## Status
**Implemented:** grid + profiles, spectral θ / FD r operators, per-m Poisson
(`getphi`) and Laplacian (`delsq`) with exact round-trip, pressure & vorticity RHS
with the **FLR term** `U·∇·{∇φ,P}` (via the Cartesian-gradient chain rule + the
invariant bracket), trap-leapfrog, HDF5 I/O, unit conventions (shared with the
Cartesian code). BCs: Dirichlet(φ=0) outer.

**Limiter edge (like the Cartesian code):** beyond `limiter_radius` the axial loss
`nu5`/`nu5p` is `x limiter_factor` (=20), and with `limiter_phi_clamp=true` a limiter
**sponge** additionally damps pressure & vorticity toward 0 and clamps `phi` toward the
wall value `phi_w` each step (one-sided Gaussian mask, exactly 1 inside `r_lim` so the
plasma core is untouched).  Result: density and potential are quickly damped for
`r > limiter_radius`, the plasma preserved inside.

**r=0 pole — handled:** cell-centred axis (first point at `dr/2`) + pole regularity
BC (`φ₋₁,ₘ = (−1)ᵐ φ₀,ₘ`, so odd-m→0, even-m symmetric across the axis) + a
**near-axis azimuthal filter** (keep only `m ≤ r/dr` at each radius, applied to every
field update) to bound the `m²/r²` diffusion + `1/r` advection stiffness. `r_in=0`
(full disk) now runs stably; `r_in>0` gives an annulus (Neumann inner wall).

### TODO (remaining)
1. **Energy/enstrophy-conserving bracket.** `akw` is a plain central-difference
   Jacobian; a finite-volume (flux-form) `{φ,·}` on the (r,θ) cells would conserve
   P/enstrophy — the recommended next step.
2. **Near-axis dt.** The pole region is the CFL bottleneck (see the dt note in the
   validation); a stronger axis filter or a locally-implicit azimuthal step would
   relax it.
3. `jax.lax.linalg.tridiagonal_solve` for O(nr) Poisson (scaffold precomputes dense
   `A_m⁻¹`, O(nr²) apply — fine for moderate `nr`, heavy for `nr ≳ 512`).
4. Higher-order radial FD.

## Analysis
`../vortex_analysis.py` reads `x,y` from `setup.h5`, so `modes`/`view` work on polar
runs too (the m-spectrum is even more natural — θ is already a grid axis). Making it
branch on `coord_type` for a native polar plot is a small follow-up.
