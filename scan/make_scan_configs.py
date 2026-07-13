"""make_scan_configs.py -- generate one TOML per (U, H) pair from a base
template (cart OR polar, selected with --coord), each with a unique [output]
outdir (to avoid different SLURM array tasks -- which share the same
$SLURM_JOB_ID -- clobbering each other's output, since rundir = outdir/<jobid>).

Usage:
    python scan/make_scan_configs.py --coord cart      # -> configs/cart_*.toml
    python scan/make_scan_configs.py --coord polar     # -> configs_polar/polar_*.toml

    # override the output config dir / scratch base for a one-off test sweep,
    # without touching COORD_SETTINGS below:
    python scan/make_scan_configs.py --coord polar \\
        --config-out-dir scan/configs_polar_test \\
        --scratch-base /pscratch/sd/d/dnzy06/plasma_sim_runs/scan_polar_test_no_theta_filter

Run from anywhere -- paths (BASE_TOML, output dirs) are resolved relative to
this script's own location, not the caller's cwd.

Edit U_VALUES / H_VALUES below to change the scan grid (shared by both coords).
"""
import os
import re
import argparse
import itertools

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- per-coordinate-system settings ----
COORD_SETTINGS = {
    "cart": dict(
        base_toml=os.path.join(SCRIPT_DIR, "..", "vortex_cart_jit", "input_cart.toml"),
        config_out_dir=os.path.join(SCRIPT_DIR, "configs_cart"),
        scratch_base="/pscratch/sd/d/dnzy06/plasma_sim_runs/scan_cart",
        prefix="cart",
    ),
    "polar": dict(
        base_toml=os.path.join(SCRIPT_DIR, "..", "vortex_polar_jit", "input_polar.toml"),
        config_out_dir=os.path.join(SCRIPT_DIR, "configs_polar_test"),
        scratch_base="/pscratch/sd/d/dnzy06/plasma_sim_runs/scan_polar_test_no_theta_filter",
        prefix="polar",
    ),
}

# ---- EDIT THESE: scan grid (shared by both coords) ----
U_VALUES = [0.0, -1.0, -2.0, -5.0, -10.0, -20.0]
H_VALUES = [5.0]
# ---------------------


def set_scalar(text, key, value, section_hint=None):
    """Replace a `key = <number>` line (first match) with a new value,
    preserving everything else on the line after the number is gone (i.e. we
    just splice in the new number and drop any trailing inline comment, to
    keep this robust to varying comment text)."""
    pattern = re.compile(rf"^({re.escape(key)}\s*=\s*)([^\s#]+)(.*)$", re.MULTILINE)
    m = pattern.search(text)
    if not m:
        raise ValueError(f"could not find a '{key} = ...' line to replace")
    return pattern.sub(lambda m: f"{m.group(1)}{value}{m.group(3)}", text, count=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coord", choices=["cart", "polar"], default="cart",
                     help="which template/output layout to generate (default: cart)")
    ap.add_argument("--config-out-dir", default=None,
                     help="override where generated .toml files go -- default is "
                          "the config_out_dir set in COORD_SETTINGS for --coord")
    ap.add_argument("--scratch-base", default=None,
                     help="override the /pscratch base each combo's outdir is "
                          "written under -- default is the scratch_base set in "
                          "COORD_SETTINGS for --coord")
    args = ap.parse_args()

    s = COORD_SETTINGS[args.coord]
    base_toml, config_out_dir, scratch_base, prefix = (
        s["base_toml"], s["config_out_dir"], s["scratch_base"], s["prefix"])

    if args.config_out_dir is not None:
        config_out_dir = args.config_out_dir
    if args.scratch_base is not None:
        scratch_base = args.scratch_base

    os.makedirs(config_out_dir, exist_ok=True)
    with open(base_toml) as f:
        template = f.read()

    n = 0
    for U, H in itertools.product(U_VALUES, H_VALUES):
        tag = f"U{U:g}_H{H:g}".replace("-", "m").replace(".", "p")
        text = template
        text = set_scalar(text, "U", f"{U}")
        text = set_scalar(text, "H", f"{H}")
        text = set_scalar(text, "outdir", f'"{scratch_base}/{tag}"')
        out_path = os.path.join(config_out_dir, f"{prefix}_{tag}.toml")
        with open(out_path, "w") as f:
            f.write(text)
        n += 1
        print(f"wrote {out_path}  (U={U}, H={H}, outdir={scratch_base}/{tag})")

    print(f"\n{n} configs written to {config_out_dir}/")
    print(f"Submit with:  sbatch --array=0-{n - 1} submit_scan.sbatch")


if __name__ == "__main__":
    main()