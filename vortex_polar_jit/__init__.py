"""vortex_polar_jit — POLAR (r, theta) GPU (JAX) solver for the Beklemishev
vortex-confinement model (Eqs. 7-8).  Annular scaffold: theta spectral, r finite
difference, per-m radial Poisson solve (no square-mesh m=4 imprint).

Sibling of the validated Cartesian code `vortex_cart_jit`.  Run:
    python -m vortex_polar_jit input_polar.toml

Status: scaffold — see README.md for what is implemented vs TODO (pole treatment,
conserving polar Arakawa bracket, FLR term).
"""
