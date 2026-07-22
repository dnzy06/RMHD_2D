"""
bias_profiles.py -- 34 polynomial bias-potential profile shapes (17 flip
pairs, each original immediately followed by its vertically-flipped
counterpart), each exactly normalized so min=0 / max=1 over a normalized
radius u in [0, 1] (u=0 -> innermost ring, u=1 -> outermost ring/limiter).

Used by both bias_profile_exploration.py (plotting, in a notebook) and
generate_bias_configs.py (writes one TOML per profile).

PROFILES is a flat list of (name, description, f) tuples. f(u) takes a
scalar or numpy array in [0, 1] and returns the profile value(s).
"""
import numpy as np


def smootherstep(x):
    """Quintic smoothstep: monotonic 0->1 on [0,1], zero 1st & 2nd deriv at both ends."""
    return 6 * x**5 - 15 * x**4 + 10 * x**3


def bezier(u, c1, c2):
    """Cubic Bezier from 1 (at u=0) through control points c1, c2, down to 0 (at u=1)."""
    return (1 - u)**3 * 1 + 3 * (1 - u)**2 * u * c1 + 3 * (1 - u) * u**2 * c2 + u**3 * 0


PROFILES = [
    ("linear_decreasing", "straight-line drop (baseline reference)",
     lambda u: 1 - u),
    ("linear_increasing", "straight-line rise (reverse of baseline)",
     lambda u: u),

    ("convex_decreasing_quad", "steep drop near center, flattens near edge",
     lambda u: (1 - u)**2),
    ("concave_increasing_quad", "steep rise near center, flattens toward the edge (flip of convex_decreasing_quad)",
     lambda u: 1 - (1 - u)**2),

    ("concave_decreasing_quad", "flat near center, steep drop near edge",
     lambda u: 1 - u**2),
    ("convex_increasing_quad", "flat near center, steep rise near edge (flip of concave_decreasing_quad)",
     lambda u: u**2),

    ("convex_decreasing_cubic", "even steeper initial drop than quadratic",
     lambda u: (1 - u)**3),
    ("concave_increasing_cubic", "very steep rise right at center, flattens quickly (flip of convex_decreasing_cubic)",
     lambda u: 1 - (1 - u)**3),

    ("concave_decreasing_cubic", "even flatter near center, drop concentrated at edge",
     lambda u: 1 - u**3),
    ("convex_increasing_cubic", "flat near center, sharp rise concentrated at the edge (flip of concave_decreasing_cubic)",
     lambda u: u**3),

    ("convex_decreasing_quartic", "very sharp drop right at center, long flat tail",
     lambda u: (1 - u)**4),
    ("concave_increasing_quartic", "very sharp rise right at center, long flat tail (flip of convex_decreasing_quartic)",
     lambda u: 1 - (1 - u)**4),

    ("concave_decreasing_quartic", "very flat near center, sharp drop only near edge",
     lambda u: 1 - u**4),
    ("convex_increasing_quartic", "very flat near center, sharp rise only near edge (flip of concave_decreasing_quartic)",
     lambda u: u**4),

    ("smoothstep_decreasing", "smooth S-curve, symmetric inflection at mid-radius",
     lambda u: 1 - (3 * u**2 - 2 * u**3)),
    ("smoothstep_increasing", "smooth S-curve rise, symmetric inflection at mid-radius (flip of smoothstep_decreasing)",
     lambda u: 3 * u**2 - 2 * u**3),

    ("smootherstep_decreasing", "flatter S-curve, zero curvature at both endpoints too",
     lambda u: 1 - smootherstep(u)),
    ("smootherstep_increasing", "flatter S-curve rise, zero curvature at both endpoints (flip of smootherstep_decreasing)",
     lambda u: smootherstep(u)),

    ("early_drop_asym", "asymmetric S-curve, transition happens near the inner radius",
     lambda u: bezier(u, 0.15, 0.02)),
    ("early_rise_asym", "asymmetric S-curve, rises quickly near the inner radius then levels off (flip of early_drop_asym)",
     lambda u: 1 - bezier(u, 0.15, 0.02)),

    ("late_drop_asym", "asymmetric S-curve, stays high until near the edge",
     lambda u: bezier(u, 0.90, 0.55)),
    ("late_rise_asym", "asymmetric S-curve, stays low until near the edge then rises (flip of late_drop_asym)",
     lambda u: 1 - bezier(u, 0.90, 0.55)),

    ("symmetric_bump_quad", "single bump peaked at mid-radius, zero at both ends",
     lambda u: 4 * u * (1 - u)),
    ("hollow_center", "dip at mid-radius, maxed out at both inner and outer edges (flip of symmetric_bump_quad)",
     lambda u: 1 - 4 * u * (1 - u)),

    ("symmetric_bump_narrow", "narrower / more peaked bump at mid-radius",
     lambda u: 16 * u**2 * (1 - u)**2),
    ("hollow_center_narrow", "narrower/deeper dip at mid-radius, maxed out at both edges (flip of symmetric_bump_narrow)",
     lambda u: 1 - 16 * u**2 * (1 - u)**2),

    ("offcenter_bump_inner", "bump peaked toward the inner radius",
     lambda u: (27 / 4) * u * (1 - u)**2),
    ("notch_inner", "dip toward the inner radius, high at both edges (flip of offcenter_bump_inner)",
     lambda u: 1 - (27 / 4) * u * (1 - u)**2),

    ("offcenter_bump_outer", "bump peaked toward the outer radius",
     lambda u: (27 / 4) * u**2 * (1 - u)),
    ("notch_outer", "dip toward the outer radius, high at both edges (flip of offcenter_bump_outer)",
     lambda u: 1 - (27 / 4) * u**2 * (1 - u)),

    ("double_bump_symmetric", "two bumps with a dip at mid-radius and at both ends",
     lambda u: 4 * ((2 * u - 1)**2 - (2 * u - 1)**4)),
    ("chebyshev_wiggle", "oscillating wiggle, alternating high/low across radius (flip of double_bump_symmetric)",
     lambda u: 4 * (2 * u - 1)**4 - 4 * (2 * u - 1)**2 + 1),

    ("very_steep_transition", "near step-function drop, very sharp transition mid-radius",
     lambda u: 1 - smootherstep(smootherstep(u))),
    ("very_steep_transition_rise", "near step-function rise, very sharp transition mid-radius (flip of very_steep_transition)",
     lambda u: smootherstep(smootherstep(u))),
]
assert len(PROFILES) == 34